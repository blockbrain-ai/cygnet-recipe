#!/usr/bin/env python3
"""Checks the shim's answers and status codes against a stand-in for vLLM. No GPU, standard library only.

    python3 shim/test_shim.py

WHY THE STATUS CODES ARE TESTED. JevBench's runner stops a run after three consecutive failed items unless a
failure is an HTTP 422, which it scores as one wrong answer and moves on; it stops at once on a 401, 403 or
429. So an input the shim cannot answer (over the server's context or size limit, more than 26 options, an
unknown question type) must be a 422, while a broken, unreachable or misconfigured server must not be, so
that it still stops the run.
"""
import importlib.util
import json
import os
import re
import sys
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# vLLM's reply to a prompt over --max-model-len, as returned by the image's vLLM on 2026-09-24.
OVERFLOW = ("This model's maximum context length is 8192 tokens. However, you requested 1 output tokens and your "
            "prompt contains at least 8192 input tokens, for a total of at least 8193 tokens. Please reduce the "
            "length of the input prompt or the number of requested output tokens. (parameter=input_tokens, value=8192)")


class FakeVLLM(BaseHTTPRequestHandler):
    """Answers like vLLM under a structured-output mask: at most 20 top_logprobs, the allowed letters first.

    A marker in the prompt picks the behaviour: MOCK_OVERFLOW, MOCK_BAD400, MOCK_<status> for 401, 403, 413,
    429 or 500, or MOCK_PICK=<letter>. GET /v1/models lists served_id with max_model_len (omitted when None).
    """
    last_prompt = ""
    served_id = "mock"
    max_model_len = 262144

    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        card = {"id": FakeVLLM.served_id}
        if FakeVLLM.max_model_len is not None:
            card["max_model_len"] = FakeVLLM.max_model_len
        self._send(200, {"object": "list", "data": [card]})

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        text = FakeVLLM.last_prompt = req["messages"][-1]["content"]
        allowed = req["structured_outputs"]["choice"]
        if "MOCK_OVERFLOW" in text:
            return self._send(400, {"error": {"message": OVERFLOW, "type": "BadRequestError", "code": 400}})
        if "MOCK_BAD400" in text:
            return self._send(400, {"error": {"message": "unknown parameter", "code": 400}})
        status = re.search(r"MOCK_(401|403|413|429|500)", text)
        if status:
            return self._send(int(status.group(1)), {"error": {"message": f"mock {status.group(1)}"}})
        m = re.search(r"MOCK_PICK=([A-Z])", text)
        pick = m.group(1) if m else allowed[0]
        ranked = [pick] + [a for a in allowed if a != pick]
        top = [{"token": a, "logprob": -0.1 if i == 0 else -3.0 - 0.1 * i} for i, a in enumerate(ranked)][:20]
        if "MOCK_DUP" in text:        # Gemma-4: a second token decoding to each letter, far below the first
            top += [{"token": a, "logprob": -38.0 - 0.1 * i} for i, a in enumerate(ranked)]
        top += [{"token": t, "logprob": -9999.0} for t in ("The", "**", " A", "To")][: max(0, 20 - len(top))]
        top = top[:20]
        self._send(200, {"choices": [{"logprobs": {"content": [{"token": pick, "logprob": -0.1, "top_logprobs": top}]}}],
                         "usage": {"prompt_tokens": len(text) // 4, "completion_tokens": 1,
                                   "total_tokens": len(text) // 4 + 1}})


def serve(handler):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv.server_address[1]


vllm_port = serve(FakeVLLM)
os.environ["SHIM_VLLM"] = f"http://127.0.0.1:{vllm_port}/v1/chat/completions"
os.environ["SHIM_MODEL"] = "mock"
spec = importlib.util.spec_from_file_location("shim", os.path.join(os.path.dirname(__file__), "cygnet_shim.py"))
shim = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shim)
shim_port = serve(shim.Handler)


def post(raw):
    req = urllib.request.Request(f"http://127.0.0.1:{shim_port}/v1/systemone", data=raw,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def ask(state, qtype, criteria, instructions="Decide."):
    return post(json.dumps({"state": state, "questions": {"decision": {
        "type": qtype, "instructions": instructions, "criteria": criteria}}}).encode())


def options(n):
    return {f"label_{i}": f"option {i}" for i in range(n)}


def picked(want):
    return lambda b: b["answers"]["decision"]["choice"] == want


failures = 0


def check(name, status, body, want_status, ok=None):
    global failures
    good = status == want_status and (ok is None or ok(body))
    failures += not good
    print(f"{'PASS' if good else 'FAIL'}  {name}: HTTP {status}" + ("" if good else f"  {json.dumps(body)[:200]}"))


check("3 options, answer B", *ask("s MOCK_PICK=B", "choice", options(3)), 200, picked("label_1"))
check("11 options, answer K", *ask("s MOCK_PICK=K", "choice", options(11)), 200, picked("label_10"))
check("26 options, answer Z", *ask("s MOCK_PICK=Z", "choice", options(26)), 200, picked("label_25"))
check("noul with 'false' first reads P(true) from the true letter",
      *ask("s MOCK_PICK=B", "noul", {"false": "no", "true": "yes"}), 200,
      lambda b: b["answers"]["decision"]["noul"] > 0.9)
state = {"order": {"id": "A-17", "paid_with": "gift card"}}
check("a JSON-object state is serialised with indent=1 (the rendering Cygnet was measured with)", *ask(state, "choice", options(2)), 200,
      lambda b: json.dumps(state, ensure_ascii=False, indent=1) in FakeVLLM.last_prompt)

check("two tokens for the same letter keep the letter's mass (duplicates are summed, not overwritten)",
      *ask("s MOCK_DUP MOCK_PICK=B", "choice", options(3)), 200,
      lambda b: b["answers"]["decision"]["probabilities"]["label_1"] > 0.9)


def tempered(T):
    import math
    base = {"label_0": -0.1, "label_1": -3.1, "label_2": -3.2}           # the stand-in's logprobs for MOCK_PICK=A
    z = {k: math.exp(v) for k, v in base.items()}; s = sum(z.values())
    p = {k: (v / s) ** (1 / T) for k, v in z.items()}; s = sum(p.values())
    return {k: v / s for k, v in p.items()}


shim.TEMPERATURE = 2.0
check("SHIM_TEMPERATURE=2 flattens the distribution as p^(1/T) renormalised",
      *ask("s MOCK_PICK=A", "choice", options(3)), 200,
      lambda b: all(abs(b["answers"]["decision"]["probabilities"][k] - v) < 1e-6 for k, v in tempered(2.0).items()))
shim.TEMPERATURE = 1.0
check("SHIM_TEMPERATURE=1 leaves the distribution unchanged",
      *ask("s MOCK_PICK=A", "choice", options(3)), 200,
      lambda b: all(abs(b["answers"]["decision"]["probabilities"][k] - v) < 1e-6 for k, v in tempered(1.0).items()))

check("27 options is a 422", *ask("s", "choice", options(27)), 422)
check("prompt over the context limit is a 422", *ask("MOCK_OVERFLOW", "choice", options(3)), 422)
check("request over the server's size limit (413) is a 422", *ask("MOCK_413", "choice", options(3)), 422)
check("unknown question type is a 422", *ask("s", "ranking", options(3)), 422)
check("noul without a 'true' key is a 422", *ask("s", "noul", {"yes": "y", "no": "n"}), 422)

check("vLLM 401 passes through", *ask("MOCK_401", "choice", options(3)), 401)
check("vLLM 403 passes through", *ask("MOCK_403", "choice", options(3)), 403)
check("vLLM 429 passes through", *ask("MOCK_429", "choice", options(3)), 429)
check("vLLM 400 about something else is a 502", *ask("MOCK_BAD400", "choice", options(3)), 502)
check("vLLM 500 is a 502", *ask("MOCK_500", "choice", options(3)), 502)
check("a body that is not JSON is a 400", *post(b"{not json"), 400)
check("a JSON body that is not an object is a 400", *post(b"[1, 2]"), 400)
check("a body without questions.decision is a 400", *post(b'{"state": "s", "questions": {}}'), 400)


def server(served_id="mock", max_model_len=262144):
    """Reconfigure the stand-in and make the shim read it again."""
    FakeVLLM.served_id, FakeVLLM.max_model_len, shim._server_context = served_id, max_model_len, None


server(max_model_len=2048)
check("a server below SHIM_MIN_CONTEXT is a 503", *ask("s", "choice", options(3)), 503)
server(served_id="another-model")
check("a server that does not list SHIM_MODEL is a 503", *ask("s", "choice", options(3)), 503)
server(max_model_len=None)
check("a server that reports no max_model_len is a 503", *ask("s", "choice", options(3)), 503)
server()
check("a server at or above SHIM_MIN_CONTEXT answers", *ask("s MOCK_PICK=A", "choice", options(3)), 200)

shim.VLLM, shim._server_context = "http://127.0.0.1:9/v1/chat/completions", None   # nothing listens on port 9
check("vLLM unreachable is a 502", *ask("s", "choice", options(3)), 502)

print(f"\n{'all passed' if not failures else f'{failures} failed'}")
sys.exit(1 if failures else 0)
