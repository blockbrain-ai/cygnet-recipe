#!/usr/bin/env python3
"""Audit an unchanged JevBench CLI run against Cygnet's two pinned GPU runs.

This is an offline report, not a runner or a JevBench leaderboard scorer.
Only aggregates, public task IDs, and hashes are exported; request/response
text, local paths, and arbitrary runner metadata stay in the local run folder.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
from pathlib import Path
import subprocess
import sys


JEVBENCH_COMMIT = "2fa63fa3226cb369795525ed011800f57dcbd894"
DATASET_HASH = "dc3995d8ae1e2fc8e81ce38431add509eb8bb39b85aadfd0c7c32079382dde51"
# Fixed before the Metal run: filenames also determine the tier and run order.
DATASETS = {
    "easy": ("easy.jsonl", 48, "231df3c2c8e88a1a8c137ebe85de96ba70fabd330849098ac7b3c52c70b7172b"),
    "standard": ("original.jsonl", 72, "5c2414edb3006b8bfcb70fda433f0f9ca015759433849f8d3104328a1f7c4180"),
    "hard": ("hard.jsonl", 111, "89e9e6becb33ed88c1de7d42dcc87531b2fb64cfaef4e1986faf7c37b3f80ebb"),
}
REFERENCES = {
    "a6000-pinned": "07cf4dd6d3dd984296d2804a2a536045ad50d1afee7e37b067d71fa756c282cc",
    "l40s-pinned": "22e4280a47e938624ee4232fce74282e2949837cd4fd18873c3758a311444121",
}
DEFINITIONS = {
    "scope": "231 public decisions only; not the complete or sealed JevBench leaderboard.",
    "accuracy": "Unchanged pinned score_task: argmax label, including Score; lexical label tie-break; invalid/failed answers count wrong.",
    "brier_mean": "Mean sum over labels of (probability - one-hot gold)^2; binary uses both classes; valid distributions only.",
    "ece": "Top-label expected calibration error with 10 equal-width bins; valid distributions only; lower is better.",
    "ordinal_mae": "Mean absolute error of probability-weighted Score level versus gold, in level units; lower is better.",
    "latency": "Unadjusted serial HTTP wall time from the official runner, all attempts including failures; the pause between requests is excluded. p50/p95 use linear interpolation at (n-1)*q. Standard tier has 72 items, matching the upstream README latency scope. Hardware differs from GPU references.",
    "agreement": "Exact argmax label match with both answers valid, divided by all 231 tasks; not statistical equivalence.",
    "cost": "No cost or composite score inferred from local runtime or zero ledger charges.",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reject_constant(value):
    raise ValueError(f"Non-finite JSON number: {value}")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)


def read_records(path: Path) -> list[dict]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line, parse_constant=reject_constant)
            if not isinstance(record, dict):
                raise ValueError("Each result row must be an object")
            records.append(record)
    return records


def check_manifest(manifest: dict, n: int, expected_delay_s: float = 0) -> None:
    if not math.isfinite(expected_delay_s) or not 0 <= expected_delay_s <= 5:
        raise ValueError("Expected request delay must be finite and between 0 and 5 seconds")
    expected = {
        "adapter": "typesafe", "dataset_hash": DATASET_HASH,
        "n_planned": n, "n_attempted": n, "request_options": {}, "delay_s": expected_delay_s,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"CLI manifest does not match the frozen protocol: {key}")


def audit_records(tasks, records: list[dict], score_task) -> list[dict]:
    """Require complete IDs and verify the CLI's stored scores with its scorer."""
    expected = {task.id for task in tasks}
    if len(expected) != len(tasks):
        raise ValueError("Duplicate task IDs")
    by_id = {}
    for record in records:
        task_id = record.get("task_id")
        if not isinstance(task_id, str) or task_id not in expected:
            raise ValueError("Unknown or missing result task_id")
        if task_id in by_id:
            raise ValueError(f"Duplicate result: {task_id}")
        by_id[task_id] = record
    if set(by_id) != expected:
        raise ValueError(f"Missing results: {sorted(expected - set(by_id))}")
    audited = []
    for task in tasks:
        record = by_id[task.id]
        if type(record.get("ok")) is not bool:
            raise ValueError(f"Invalid operational status: {task.id}")
        if record.get("status") != ("ok" if record["ok"] else "failed"):
            raise ValueError(f"Inconsistent operational status: {task.id}")
        latency = record.get("latency_s")
        if (isinstance(latency, bool) or not isinstance(latency, (int, float))
                or not math.isfinite(latency) or latency < 0):
            raise ValueError(f"Missing or invalid latency: {task.id}")
        if record["ok"]:
            if record.get("probs_source") != "native":
                raise ValueError(f"Expected a native distribution: {task.id}")
            scored = score_task(record.get("probs_as_returned") or {}, task)
        else:
            scored = dict(valid=False, strict_valid=False, renormalized=False,
                          correct=False, predicted=None)
        for field in ("valid", "strict_valid", "renormalized", "correct", "predicted", "ordinal_ev", "probs"):
            if record.get(field) != scored.get(field):
                raise ValueError(f"Stored {field} disagrees with the pinned scorer: {task.id}")
        # Only fields consumed by official metrics survive. Error text, usage,
        # model names, raw request metadata and arbitrary extra keys do not.
        audited.append({
            "task_id": task.id, "ok": record["ok"], "latency_s": latency,
            "cost_usd": None, **{key: scored.get(key) for key in (
                "valid", "strict_valid", "renormalized", "correct", "predicted", "ordinal_ev", "probs")},
        })
    return audited


def compact_metrics(summary: dict) -> dict:
    fields = ("n_planned", "n_attempted", "n_scorable", "n_valid", "n_correct",
              "accuracy", "coverage", "schema_validity", "schema_validity_strict",
              "n_renormalized", "operational_success", "calibration_n", "brier_mean",
              "ordinal_mae", "latency", "latency_failures")
    result = {key: summary[key] for key in fields}
    result["ece"] = summary["ece"]["ece"] if summary["ece"] else None
    return result


def metrics_report(tiers: dict, records: list[dict], summarize) -> dict:
    tasks = [task for group in tiers.values() for task in group]
    summary = summarize(tasks, records)
    result = {
        "all": compact_metrics(summary),
        "per_tier": {},
        "macro_family_accuracy": summary["macro_accuracy"],
        "paraphrase_consistency": summary["paraphrase_consistency"],
        "error_ids": sorted(record["task_id"] for record in records if not record["correct"]),
    }
    for tier, group in tiers.items():
        ids = {task.id for task in group}
        subset = [record for record in records if record["task_id"] in ids]
        result["per_tier"][tier] = compact_metrics(summarize(group, subset))
    return result


def compare(records: list[dict], reference: list[dict]) -> dict:
    by_id = {record["task_id"]: record for record in reference}
    if set(by_id) != {record["task_id"] for record in records} or len(by_id) != len(records):
        raise ValueError("Comparison requires the same unique task IDs")
    disagreement, regressions, improvements = [], [], []
    for record in records:
        other = by_id[record["task_id"]]
        if not (record["valid"] and other["valid"] and record["predicted"] == other["predicted"]):
            disagreement.append(record["task_id"])
        if other["correct"] and not record["correct"]:
            regressions.append(record["task_id"])
        if record["correct"] and not other["correct"]:
            improvements.append(record["task_id"])
    return {
        "n": len(records), "matching_predictions": len(records) - len(disagreement),
        "agreement": (len(records) - len(disagreement)) / len(records) if records else None,
        "disagreement_ids": sorted(disagreement),
        "regression_ids": sorted(regressions), "improvement_ids": sorted(improvements),
    }


def load_protocol(root: Path):
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if head != JEVBENCH_COMMIT:
        raise ValueError("JevBench checkout must be pinned to " + JEVBENCH_COMMIT)
    if subprocess.run(["git", "-C", str(root), "diff", "--quiet", "HEAD", "--"], check=False).returncode:
        raise ValueError("JevBench tracked files must be unchanged")
    sys.path.insert(0, str(root))
    package = importlib.import_module("jevbench")
    if Path(package.__file__).resolve().parent != root / "jevbench":
        raise ValueError("An unrelated JevBench package is already imported")
    tasks_module = importlib.import_module("jevbench.tasks")
    score_task = importlib.import_module("jevbench.scoring").score_task
    summarize = importlib.import_module("jevbench.summarize").summarize
    tiers = {}
    for tier, (filename, count, digest) in DATASETS.items():
        path = root / "datasets" / "public" / filename
        if sha256(path) != digest:
            raise ValueError("Frozen dataset file hash mismatch: " + filename)
        tasks = tasks_module.load_jsonl(str(path))
        if len(tasks) != count or any(task.split != "public" for task in tasks):
            raise ValueError("Unexpected public task inventory: " + filename)
        tiers[tier] = tasks
    tasks = [task for group in tiers.values() for task in group]
    if tasks_module.dataset_hash(tasks) != DATASET_HASH:
        raise ValueError("Canonical dataset hash mismatch")
    return tiers, score_task, summarize


def build_report(jevbench: Path, results: Path, manifest: Path, expected_delay_s: float = 0.5) -> dict:
    tiers, score_task, summarize = load_protocol(jevbench.resolve())
    tasks = [task for group in tiers.values() for task in group]
    candidate_manifest = read_json(manifest)
    check_manifest(candidate_manifest, len(tasks), expected_delay_s)
    records = audit_records(tasks, read_records(results), score_task)
    output = {
        "protocol": {
            "jevbench_commit": JEVBENCH_COMMIT, "adapter": "typesafe",
            "dataset_hash": DATASET_HASH, "n": len(tasks),
            "candidate_delay_s": candidate_manifest["delay_s"],
            "reference_delay_s": 0,
            "datasets": {name: {"file": filename, "n": count, "sha256": digest}
                         for name, (filename, count, digest) in DATASETS.items()},
        },
        "definitions": DEFINITIONS,
        "candidate": {"results_sha256": sha256(results), "manifest_sha256": sha256(manifest),
                      **metrics_report(tiers, records, summarize)},
        "references": {},
    }
    for name, digest in REFERENCES.items():
        folder = Path(__file__).resolve().parents[1] / "runs" / name
        path = folder / "results.jsonl"
        if sha256(path) != digest:
            raise ValueError("Pinned reference hash mismatch: " + name)
        check_manifest(read_json(folder / "manifest.json"), len(tasks))
        reference = audit_records(tasks, read_records(path), score_task)
        output["references"][name] = {
            "results_sha256": digest, **metrics_report(tiers, reference, summarize),
            "candidate_comparison": compare(records, reference),
        }
    return output


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jevbench", type=Path, required=True, help="Unchanged checkout at pinned commit")
    parser.add_argument("--results", type=Path, required=True, help="Official CLI results.jsonl")
    parser.add_argument("--manifest", type=Path, required=True, help="Official CLI manifest.json")
    parser.add_argument("--expected-delay-s", type=float, default=0.5,
                        help="Validate the CLI request pause (default: 0.5 seconds; excluded from latency)")
    parser.add_argument("--output", type=Path, help="Compact JSON report (stdout if omitted)")
    args = parser.parse_args(argv)
    try:
        report = build_report(args.jevbench, args.results, args.manifest, args.expected_delay_s)
    except (ValueError, KeyError, OSError, subprocess.SubprocessError) as error:
        parser.exit(2, f"Audit failed: {error}\n")
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
