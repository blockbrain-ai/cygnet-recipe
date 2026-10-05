"""Boundary tests: real HTTP + a real subprocess, never a model or GPU."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, contextmanager
import http.client
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest import mock


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ROOT = Path(__file__).resolve().parents[1]
server = load_module("cygnet_metal_test_server", ROOT / "metal/server.py")
shim = load_module("cygnet_metal_test_shim", ROOT / "shim/cygnet_shim.py")

# Distinct native IDs may decode to the same text. A wins after aggregation
# (.25 + .35 > .4), although the single most probable token is B.
FAKE_WORKER = r'''
import json, math, os, sys, time
mode, log_path = sys.argv[1:3]
if mode == "startup_hang":
    time.sleep(10)
if mode == "startup_exit":
    sys.exit(2)
if mode == "bad_ready":
    print(json.dumps({"ready": True, "context_size": 2048}), flush=True)
else:
    print(json.dumps({"ready": True, "context_size": 4096}), flush=True)
if mode == "no_read":
    time.sleep(10)
for line in sys.stdin:
    req = json.loads(line)
    with open(log_path, "a") as out:
        out.write(json.dumps(req) + "\n")
    if mode == "hang":
        time.sleep(10)
    if mode == "exit":
        sys.exit(2)
    if mode == "bad_json":
        print("not JSON", flush=True)
        continue
    if mode == "oversized":
        print("x" * (2 * 1024 * 1024 + 1), flush=True)
        continue
    if req["prompt"].startswith("__context__"):
        print(json.dumps({"id": req["id"], "error": "prompt too long and exceeds maximum context length 4096", "error_type": "input"}), flush=True)
        continue
    if req["prompt"].startswith("__inference__"):
        print(json.dumps({"id": req["id"], "error": "x" * 5000, "error_type": "inference"}), flush=True)
        continue
    reply = {
        "id": req["id"], "ok": True, "prompt_tokens": 123, "completion_tokens": 0,
        "top_logprobs": [
            {"token": "B", "token_id": 103, "logprob": math.log(.4)},
            {"token": "A", "token_id": 102, "logprob": math.log(.35)},
            {"token": "A", "token_id": 101, "logprob": math.log(.25)},
        ], "kv_cache": {"reused_tokens": 0}, "timings_ms": {"prefill": 2.3},
    }
    if mode == "wrong_id":
        reply["id"] = "unrelated"
    elif mode == "repeated_id":
        reply["top_logprobs"][1]["token_id"] = 103
    elif mode == "bad_logprob":
        reply["top_logprobs"][0]["logprob"] = .5
    elif mode == "bad_usage":
        reply["prompt_tokens"] = 999999
    elif mode == "generated":
        reply["completion_tokens"] = 1
    print(json.dumps(reply), flush=True)
'''


def request_body():
    return {
        "model": "cygnet", "messages": [
            {"role": "system", "content": shim.SYSTEM},
            {"role": "user", "content": "Some state.\nOptions:\nA. Yes\nB. No"},
        ], "max_tokens": 1, "temperature": 1.0, "logprobs": True,
        "top_logprobs": 20, "chat_template_kwargs": {"enable_thinking": False},
        "structured_outputs": {"choice": ["A", "B"]},
    }


@contextmanager
def running(mode="normal", request_timeout=3):
    with tempfile.TemporaryDirectory() as folder:
        log = Path(folder) / "requests.jsonl"
        with server.NativeWorker(
            [sys.executable, "-u", "-c", FAKE_WORKER, mode, str(log)],
            context_size=4096, lock_path=Path(folder) / "worker.lock",
            startup_timeout=2, request_timeout=request_timeout,
        ) as worker:
            with server.MetalServer(("127.0.0.1", 0), worker) as httpd:
                thread = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": .01}, daemon=True)
                thread.start()
                try:
                    yield httpd, worker, log
                finally:
                    httpd.shutdown()
                    thread.join(timeout=2)


def http_request(httpd, body=None, *, method="POST", path="/v1/chat/completions", raw=None, headers=None):
    data = json.dumps(request_body() if body is None else body).encode() if raw is None else raw
    if method == "GET":
        data = None
    actual_headers = {"Content-Type": "application/json"}
    actual_headers.update(headers or {})
    with closing(http.client.HTTPConnection("127.0.0.1", httpd.server_port, timeout=5)) as conn:
        conn.request(method, path, body=data, headers=actual_headers)
        response = conn.getresponse()
        return response.status, json.loads(response.read())


class TransportTests(unittest.TestCase):
    def test_http_preserves_native_aliases_and_reports_no_generation(self):
        with running() as (httpd, worker, log):
            status, models = http_request(httpd, method="GET", path="/v1/models")
            self.assertEqual(status, 200)
            self.assertEqual(models["data"][0]["max_model_len"], 4096)
            self.assertEqual(models["data"][0]["id"], "cygnet")
            body = request_body()
            body["messages"][1]["content"] = "  Unicode: français 中文\nKeep this whitespace.  "
            status, result = http_request(httpd, body)
            self.assertEqual(status, 200)
            lp = result["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
            self.assertEqual([r["token"] for r in lp], ["B", "A", "A"])
            self.assertEqual([r["token_id"] for r in lp], [103, 102, 101])
            self.assertEqual(result["usage"], {"prompt_tokens": 123, "completion_tokens": 0, "total_tokens": 123})
            self.assertEqual(result["metal"]["readout"], "prefill_logits")
            sent = json.loads(log.read_text())
            self.assertEqual(sent["system"], shim.SYSTEM)
            self.assertEqual(sent["prompt"], body["messages"][1]["content"])
            self.assertEqual(sent["letters"], ["A", "B"])
            self.assertEqual(sent["top_logprobs"], 20)
            self.assertTrue(worker.is_alive())

    def test_unchanged_shim_aggregates_aliases_then_applies_temperature(self):
        with running() as (httpd, _, log), mock.patch.object(shim, "VLLM", f"http://127.0.0.1:{httpd.server_port}/v1/chat/completions"), mock.patch.object(shim, "_server_context", None):
            self.assertEqual(shim.server_context(), 4096)
            for temperature in (1.0, 3.4):
                with self.subTest(temperature=temperature), mock.patch.object(shim, "TEMPERATURE", temperature):
                    result, usage, error = shim.answer_for(
                        {"evidence": "Keep structured state intact."},
                        {"type": "choice", "instructions": "Select the best description.", "criteria": {"yes": "Yes", "no": "No"}},
                    )
                    self.assertIsNone(error)
                    expected = .6 ** (1 / temperature) / (.6 ** (1 / temperature) + .4 ** (1 / temperature))
                    self.assertAlmostEqual(result["probabilities"]["yes"], expected)
                    self.assertAlmostEqual(result["probabilities"]["no"], 1 - expected)
                    self.assertEqual(result["choice"], "yes")
                    self.assertEqual(usage["completion_tokens"], 0)
            requests = [json.loads(line) for line in log.read_text().splitlines()]
            expected_prompt = shim.build_prompt({"evidence": "Keep structured state intact."}, "Select the best description.", [("A", "yes", "Yes"), ("B", "no", "No")])
            self.assertEqual(requests[0]["prompt"], expected_prompt)

    def test_unchanged_shim_noul_score_and_context_error(self):
        with running() as (httpd, worker, _), mock.patch.object(shim, "VLLM", f"http://127.0.0.1:{httpd.server_port}/v1/chat/completions"), mock.patch.object(shim, "TEMPERATURE", 1.0):
            result, _, error = shim.answer_for("state", {"type": "noul", "criteria": {"false": "No", "true": "Yes"}})
            self.assertIsNone(error)
            self.assertAlmostEqual(result["noul"], .4)
            result, _, error = shim.answer_for("state", {"type": "score", "criteria": ["Low", "High"]})
            self.assertIsNone(error)
            self.assertAlmostEqual(result["probabilities"]["0"], .6)
            with self.assertRaises(shim.Unprocessable) as refused:
                shim.call_vllm("__context__ too long", ["A", "B"])
            # The upstream shim owns the HTTPError, but does not close it itself.
            refused.exception.__cause__.close()
            self.assertTrue(worker.is_alive())
            self.assertEqual(http_request(httpd)[0], 200)

    def test_concurrent_calls_are_serialized_without_crossed_replies(self):
        with running() as (httpd, _, log):
            with ThreadPoolExecutor(max_workers=5) as pool:
                responses = list(pool.map(lambda _: http_request(httpd), range(10)))
            self.assertEqual([r[0] for r in responses], [200] * 10)
            self.assertEqual(len({r[1]["id"] for r in responses}), 10)
            self.assertEqual(len(log.read_text().splitlines()), 10)

    def test_invalid_openai_profiles_never_reach_worker(self):
        alterations = [
            {"stream": False}, {"model": "other"}, {"max_tokens": 2}, {"max_tokens": True},
            {"temperature": 0}, {"temperature": True}, {"logprobs": False},
            {"top_logprobs": 5}, {"top_logprobs": 20.0},
            {"chat_template_kwargs": {"enable_thinking": True}},
            {"chat_template_kwargs": {"enable_thinking": 0}},
            {"chat_template_kwargs": {}}, {"chat_template_kwargs": []},
            {"structured_outputs": {"choice": ["B", "A"]}},
            {"structured_outputs": {"choice": ["A", "A"]}},
            {"structured_outputs": {"choice": []}},
            {"structured_outputs": {"choice": ["A"], "regex": ".*"}},
            {"messages": [{"role": "user", "content": "Hi"}]},
            {"messages": [{"role": "system", "content": []}, {"role": "user", "content": "Hi"}]},
            {"messages": [{"role": "system", "content": "X"}, {"role": "assistant", "content": "Hi"}]},
            {"messages": [{"role": "system", "content": "X"}, {"role": "user", "content": "\0"}]},
            {"messages": [{"role": "system", "content": "X"}, {"role": "user", "content": "\ud800"}]},
        ]
        with running() as (httpd, _, log):
            for change in alterations:
                with self.subTest(change=change):
                    body = request_body()
                    body.update(change)
                    self.assertEqual(http_request(httpd, body)[0], 400)
            self.assertFalse(log.exists())

    def test_json_body_and_size_checks(self):
        with running() as (httpd, _, log):
            for raw in (b"[]", b"null", b"{}", b'{"x":1,"x":2}', b'{"x":NaN}', b"\xff", b"{"):
                with self.subTest(raw=raw):
                    self.assertEqual(http_request(httpd, raw=raw)[0], 400)
            self.assertEqual(http_request(httpd, raw=b"", headers={"Content-Length": str(server.MAX_BODY + 1)})[0], 413)
            self.assertEqual(http_request(httpd, headers={"Content-Type": "text/plain"})[0], 415)
            self.assertEqual(http_request(httpd, headers={"Transfer-Encoding": "chunked"})[0], 400)
            self.assertEqual(http_request(httpd, headers={"Content-Encoding": "gzip"})[0], 400)
            self.assertFalse(log.exists())

    def test_local_origin_host_and_route_checks(self):
        with running() as (httpd, _, log):
            for headers in ({"Host": "attacker.example"}, {"Origin": "https://attacker.example"}, {"Origin": "null"}, {"Sec-Fetch-Site": "cross-site"}):
                with self.subTest(headers=headers):
                    self.assertEqual(http_request(httpd, headers=headers)[0], 403)
            self.assertEqual(http_request(httpd, path="/v1/completions")[0], 404)
            self.assertEqual(http_request(httpd, method="GET", path="/")[0], 404)
            self.assertFalse(log.exists())
            self.assertEqual(http_request(httpd, headers={"Origin": f"http://127.0.0.1:{httpd.server_port}"})[0], 200)

    def test_inference_failure_is_bounded_and_stops_worker(self):
        with running() as (httpd, worker, _):
            body = request_body()
            body["messages"][1]["content"] = "__inference__"
            status, result = http_request(httpd, body)
            self.assertEqual(status, 503)
            self.assertLessEqual(len(result["error"]["message"]), server.MAX_ERROR)
            self.assertFalse(worker.is_alive())
            self.assertEqual(http_request(httpd, method="GET", path="/v1/models")[0], 503)

    def test_timeout_dead_and_corrupt_workers_fail_closed(self):
        for mode in ("hang", "exit", "bad_json", "oversized", "wrong_id", "repeated_id", "bad_logprob", "bad_usage", "generated"):
            with self.subTest(mode=mode), running(mode, request_timeout=.2) as (httpd, worker, _):
                self.assertEqual(http_request(httpd)[0], 503)
                self.assertFalse(worker.is_alive())
                self.assertIsNotNone(worker._process.poll())

    def test_stuck_reader_cannot_block_on_full_stdin_pipe(self):
        with running("no_read", request_timeout=.2) as (httpd, worker, _):
            body = request_body()
            body["messages"][1]["content"] = "x" * 500000
            self.assertEqual(http_request(httpd, body)[0], 503)
            self.assertFalse(worker.is_alive())


class LifecycleTests(unittest.TestCase):
    def test_single_workspace_lock_and_owned_child_cleanup(self):
        with tempfile.TemporaryDirectory() as folder:
            command = [sys.executable, "-u", "-c", FAKE_WORKER, "normal", str(Path(folder) / "log")]
            args = {"context_size": 4096, "lock_path": Path(folder) / "lock", "startup_timeout": 2}
            worker = server.NativeWorker(command, **args)
            try:
                with self.assertRaisesRegex(server.WorkerError, "another Metal worker"):
                    server.NativeWorker(command, **args)
                self.assertTrue(worker.is_alive())
            finally:
                worker.close()
            self.assertIsNotNone(worker._process.poll())
            worker.close()
            with server.NativeWorker(command, **args) as replacement:
                self.assertTrue(replacement.is_alive())
            self.assertIsNotNone(replacement._process.poll())

    def test_child_retains_model_lock_after_parent_descriptor_closes(self):
        with tempfile.TemporaryDirectory() as folder:
            command = [sys.executable, "-u", "-c", FAKE_WORKER, "normal", str(Path(folder) / "log")]
            args = {"context_size": 4096, "lock_path": Path(folder) / "lock", "startup_timeout": 2}
            worker = server.NativeWorker(command, **args)
            try:
                # Simulate the parent losing its lock descriptor while the native
                # child remains resident (for example SIGKILL during Metal decode).
                worker._lock_file.close()
                worker._lock_file = None
                with self.assertRaisesRegex(server.WorkerError, "another Metal worker"):
                    server.NativeWorker(command, **args)
                self.assertTrue(worker.is_alive())
            finally:
                worker.close()
            with server.NativeWorker(command, **args) as replacement:
                self.assertTrue(replacement.is_alive())

    def test_startup_failure_releases_lock_and_reaps_child(self):
        for mode in ("startup_hang", "startup_exit", "bad_ready"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as folder:
                command = [sys.executable, "-u", "-c", FAKE_WORKER, mode, str(Path(folder) / "log")]
                args = {"context_size": 4096, "lock_path": Path(folder) / "lock", "startup_timeout": .15}
                with self.assertRaises(server.WorkerError):
                    server.NativeWorker(command, **args)
                command[4] = "normal"
                with server.NativeWorker(command, **args) as replacement:
                    self.assertTrue(replacement.is_alive())

    def test_refuses_nonlocal_bind(self):
        with running() as (_, worker, _):
            with self.assertRaisesRegex(ValueError, "127.0.0.1"):
                server.MetalServer(("0.0.0.0", 0), worker)


if __name__ == "__main__":
    unittest.main()
