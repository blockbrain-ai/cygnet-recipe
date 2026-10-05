#!/usr/bin/env python3
"""Offline protocol/audit tests; the optional reference check needs JevBench."""

import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import summarize as report


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.tasks = [SimpleNamespace(id="public-a"), SimpleNamespace(id="public-b")]
        self.scores = {
            "public-a": dict(valid=True, strict_valid=True, renormalized=False,
                             correct=True, predicted="yes", probs={"no": 0.1, "yes": 0.9}),
            "public-b": dict(valid=True, strict_valid=True, renormalized=False,
                             correct=False, predicted="no", probs={"no": 0.6, "yes": 0.4}),
        }
        self.records = [
            dict(task_id=task.id, status="ok", ok=True, latency_s=0.1,
                 probs_source="native", probs_as_returned=self.scores[task.id]["probs"],
                 **self.scores[task.id]) for task in self.tasks
        ]

    def scorer(self, probs, task):
        self.assertEqual(probs, self.scores[task.id]["probs"])
        return self.scores[task.id]

    def audit(self, records=None):
        return report.audit_records(self.tasks, self.records if records is None else records, self.scorer)

    def test_reorders_records_without_changing_task_alignment(self):
        audited = self.audit(list(reversed(self.records)))
        self.assertEqual([r["task_id"] for r in audited], [t.id for t in self.tasks])

    def test_duplicate_cannot_replace_a_missing_result(self):
        with self.assertRaisesRegex(ValueError, "Duplicate result"):
            self.audit([self.records[0], self.records[0]])

    def test_missing_and_unknown_results_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "Missing results"):
            self.audit(self.records[:1])
        changed = copy.deepcopy(self.records)
        changed[1]["task_id"] = "private-or-unknown"
        with self.assertRaisesRegex(ValueError, "Unknown"):
            self.audit(changed)

    def test_cannot_trust_stored_correctness_over_official_scorer(self):
        self.records[1]["correct"] = True
        with self.assertRaisesRegex(ValueError, "Stored correct"):
            self.audit()

    def test_cannot_change_stored_probabilities_after_scoring(self):
        self.records[0]["probs"] = {"yes": 1.0, "no": 0.0}
        with self.assertRaisesRegex(ValueError, "Stored probs"):
            self.audit()

    def test_failed_request_remains_in_denominator_without_invented_probs(self):
        self.records[1].update(status="failed", ok=False, valid=False, strict_valid=False,
                               correct=False, predicted=None, probs=None)
        audited = self.audit()
        self.assertEqual(len(audited), 2)
        self.assertFalse(audited[1]["correct"])
        self.assertIsNone(audited[1]["probs"])

    def test_invalid_distribution_uses_pinned_scorer_result(self):
        self.scores["public-b"] = dict(valid=False, strict_valid=False, renormalized=False,
                                       correct=False, predicted=None)
        self.records[1].update(self.scores["public-b"], probs=None,
                               probs_as_returned={"yes": 4.0})
        def invalid_scorer(probs, task):
            return self.scores[task.id]
        audited = report.audit_records(self.tasks, self.records, invalid_scorer)
        self.assertFalse(audited[1]["valid"])

    def test_latency_must_cover_every_attempt_and_be_finite(self):
        for value in (None, -1, float("nan"), float("inf"), True):
            with self.subTest(value=value):
                changed = copy.deepcopy(self.records)
                changed[0]["latency_s"] = value
                with self.assertRaisesRegex(ValueError, "latency"):
                    self.audit(changed)

    def test_verbalized_probabilities_are_not_silently_accepted(self):
        self.records[0]["probs_source"] = "verbalized"
        with self.assertRaisesRegex(ValueError, "native distribution"):
            self.audit()

    def test_private_fields_are_not_forwarded_to_metrics(self):
        self.records[0].update(error="PRIVATE ERROR", runtime={"prompt": "PRIVATE STATE"},
                               model="/private/model/path", cost_basis="PRIVATE ACCOUNT")
        encoded = json.dumps(self.audit())
        self.assertNotIn("PRIVATE", encoded)
        self.assertNotIn("/private", encoded)

    def test_comparison_distinguishes_regressions_from_disagreement(self):
        reference = self.audit()
        candidate = copy.deepcopy(reference)
        candidate[0].update(predicted="no", correct=False)
        candidate[1].update(predicted="yes", correct=True)
        result = report.compare(candidate, reference)
        self.assertEqual(result["agreement"], 0)
        self.assertEqual(result["regression_ids"], ["public-a"])
        self.assertEqual(result["improvement_ids"], ["public-b"])

    def test_two_invalid_answers_are_not_agreement(self):
        records = self.audit()
        records[0].update(valid=False, predicted=None, correct=False)
        self.assertEqual(report.compare(records, records)["agreement"], 0.5)

    def test_wrong_manifest_protocol_is_rejected(self):
        manifest = dict(adapter="typesafe", dataset_hash=report.DATASET_HASH,
                        n_planned=231, n_attempted=231, request_options={}, delay_s=0)
        report.check_manifest(manifest, 231)
        for key, value in (("dataset_hash", "different"), ("n_attempted", 230),
                           ("adapter", "openai_compat"), ("request_options", {"changed": True})):
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, key):
                    report.check_manifest({**manifest, key: value}, 231)

    def test_nonfinite_json_literals_are_rejected(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "results.jsonl"
            path.write_text('{"latency_s": NaN}\n')
            with self.assertRaisesRegex(ValueError, "Non-finite"):
                report.read_records(path)

    def test_candidate_delay_is_checked_separately_from_zero_delay_references(self):
        manifest = dict(adapter="typesafe", dataset_hash=report.DATASET_HASH,
                        n_planned=231, n_attempted=231, request_options={}, delay_s=0.5)
        report.check_manifest(manifest, 231, expected_delay_s=0.5)
        with self.assertRaisesRegex(ValueError, "delay_s"):
            report.check_manifest(manifest, 231, expected_delay_s=0)
        for delay in (float("nan"), float("inf"), -1, 6):
            with self.subTest(delay=delay):
                with self.assertRaisesRegex(ValueError, "request delay"):
                    report.check_manifest(manifest, 231, expected_delay_s=delay)


class PinnedReferenceTests(unittest.TestCase):
    def test_real_gpu_reference_matches_documented_counts_and_near_tie(self):
        root = Path(__file__).resolve().parents[1]
        checkout = root / ".runtime" / "jevbench"
        if not checkout.exists():
            self.skipTest("Optional integration check requires the pinned JevBench checkout")
        result = report.build_report(checkout, root / "runs/l40s-pinned/results.jsonl",
                                     root / "runs/l40s-pinned/manifest.json", expected_delay_s=0)
        self.assertEqual(result["protocol"]["candidate_delay_s"], 0)
        self.assertEqual(result["protocol"]["reference_delay_s"], 0)
        self.assertEqual(result["candidate"]["all"]["n_correct"], 203)
        self.assertEqual([v["n_correct"] for v in result["candidate"]["per_tier"].values()], [48, 70, 85])
        self.assertEqual(result["candidate"]["per_tier"]["standard"]["latency"]["n"], 72)
        a6000 = result["references"]["a6000-pinned"]
        self.assertEqual(a6000["all"]["n_correct"], 204)
        self.assertEqual(a6000["candidate_comparison"]["matching_predictions"], 230)
        self.assertEqual(a6000["candidate_comparison"]["regression_ids"], ["hard-sol-a-multi_hop-07"])
        self.assertEqual(result["references"]["l40s-pinned"]["candidate_comparison"]["agreement"], 1)
        self.assertNotIn(str(root), json.dumps(result))


if __name__ == "__main__":
    unittest.main()
