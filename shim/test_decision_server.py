#!/usr/bin/env python3
"""Checks the decision server against a stand-in for vLLM, and against the benchmark shim itself. Standard library
only, no GPU.

    python3 shim/test_decision_server.py

Three kinds of check:
  1. Identity: on inputs the benchmark path takes (one question, at most CYGNET_GROUP_SIZE options, text
     descriptions, noul given false first), the server's probabilities equal cygnet_shim's exactly, at T 1 and T 2.
  2. The API: every named question answered, confidence, score, legend, optional noul criteria, null and JSON
     descriptions, limits, usage names, the models listing, the optional key and the status codes.
  3. The grouped readout past CYGNET_GROUP_SIZE options, with a stand-in that scores each option by its own text
     (MOCK_TARGET=<word>): the option carrying the word must win, the final pass must show each group winner's own
     text, and the pass count must be ceil(K / group size) + 1.
"""
import importlib.util
import json
import math
import os
import re
import sys
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

OVERFLOW = ("This model's maximum context length is 8192 tokens. However, you requested 1 output tokens and your "
            "prompt contains at least 8192 input tokens, for a total of at least 8193 tokens.")


def option_lines(text):
    """{letter: option text} from a prompt the shim built."""
    body = text.split("\nOptions:\n", 1)[1].split("\n\nAnswer with the letter", 1)[0]
    return dict(re.match(r"^([A-Z])\. (.*)$", line).groups() for line in body.split("\n"))


class FakeVLLM(BaseHTTPRequestHandler):
    """vLLM under a structured-output mask: at most 20 top_logprobs, log-softmax over the allowed letters.

    MOCK_PICK=<letter> puts the mass on that letter (the shim tests' stand-in); MOCK_TARGET=<word> scores each option by
    whether its own text contains the word, with a small per-position offset so there are no ties; MOCK_OVERFLOW,
    MOCK_429, MOCK_500 fail as vLLM does. Every prompt is recorded.
    """
    prompts = []
    lock = threading.Lock()
    served_id = "mock"
    max_model_len = 262144
    in_flight = peak = 0

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
        self._send(200, {"object": "list", "data": [{"id": FakeVLLM.served_id, "max_model_len": FakeVLLM.max_model_len}]})

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        text = req["messages"][-1]["content"]
        with FakeVLLM.lock:
            FakeVLLM.prompts.append(text)
            FakeVLLM.in_flight += 1
            FakeVLLM.peak = max(FakeVLLM.peak, FakeVLLM.in_flight)
        try:
            if "MOCK_SLOW" in text:
                import time
                time.sleep(0.2)
            self._answer(req, text)
        finally:
            with FakeVLLM.lock:
                FakeVLLM.in_flight -= 1

    def _answer(self, req, text):
        allowed = req["structured_outputs"]["choice"]
        if "MOCK_OVERFLOW" in text:
            return self._send(400, {"error": {"message": OVERFLOW, "code": 400}})
        status = re.search(r"MOCK_(429|500)", text)
        if status:
            return self._send(int(status.group(1)), {"error": {"message": "mock"}})
        target = re.search(r"MOCK_TARGET=(\w+)", text)
        pick = re.search(r"MOCK_PICK=([A-Z])", text)
        if target:
            lines = option_lines(text)
            logits = {a: (0.0 if target.group(1) in lines[a] else -4.0) - 0.01 * i for i, a in enumerate(allowed)}
        else:
            p = pick.group(1) if pick else allowed[0]
            logits = {a: -0.1 if a == p else -3.0 - 0.1 * i for i, a in enumerate(allowed)}
        z = math.log(sum(math.exp(v) for v in logits.values()))
        top = sorted(({"token": a, "logprob": v - z} for a, v in logits.items()), key=lambda d: -d["logprob"])[:20]
        self._send(200, {"choices": [{"logprobs": {"content": [{"token": top[0]["token"], "logprob": top[0]["logprob"],
                                                                 "top_logprobs": top}]}}],
                         "usage": {"prompt_tokens": len(text) // 4, "completion_tokens": 1}})


def serve(handler):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv.server_address[1]


HERE = os.path.dirname(os.path.abspath(__file__))
vllm_port = serve(FakeVLLM)
os.environ.update({"SHIM_VLLM": f"http://127.0.0.1:{vllm_port}/v1/chat/completions", "SHIM_MODEL": "mock",
                   "CYGNET_MODEL_NAME": "cygnet-test"})


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


frozen = load("frozen_shim", os.path.join(HERE, "cygnet_shim.py"))
server = load("decision_server", os.path.join(HERE, "decision_server.py"))
frozen_port, server_port = serve(frozen.Handler), serve(server.Handler)


def call(port, path, body=None, headers=None, raw=None):
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data, method="POST" if data is not None else "GET",
                                 headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def ask(questions, state="s", headers=None):
    return call(server_port, "/v1/systemone", {"state": state, "model": "any-model", "questions": questions}, headers)


failures = 0


def _crash(kind, value, tb):
    """An exception in a check is a failure, reported like one."""
    import traceback
    print(f"FAIL  unexpected {kind.__name__}: {value}")
    traceback.print_tb(tb, limit=2)
    print("\n1 or more failed")
    sys.exit(1)


sys.excepthook = _crash


def check(name, good, detail=""):
    global failures
    failures += not good
    print(f"{'PASS' if good else 'FAIL'}  {name}" + ("" if good else f"  {str(detail)[:300]}"))


def options(n, word=None, at=None):
    return {f"label_{i}": (f"option {i} {word}" if i == at else f"option {i}") for i in range(n)}


# ---- 1. identity with the benchmark shim
cases = []
for n in (2, 3, 7, 11, 20):
    for pick in sorted({"A", "B", "ABCDEFGHIJKLMNOPQRST"[n - 1]}):
        cases.append(("choice", options(n), f"MOCK_PICK={pick}"))
cases += [("score", ["low", "mid", "high"], "MOCK_PICK=B"), ("score", ["a", "b", "c", "d", "e"], "MOCK_PICK=E"),
          ("noul", {"false": "No, it does not.", "true": "Yes, it does."}, "MOCK_PICK=B"),
          ("noul", {"false": "No, it does not.", "true": "Yes, it does."}, "MOCK_PICK=A")]
for T in (1.0, 2.0):
    frozen.TEMPERATURE = server.shim.TEMPERATURE = T
    same = 0
    for qtype, criteria, marker in cases:
        q = {"type": qtype, "instructions": f"Decide. {marker}", "criteria": criteria}
        for state in ("plain state", {"order": {"id": "A-17", "paid_with": "gift card"}}, "", {}, []):
            s1, b1 = call(frozen_port, "/v1/systemone", {"state": state, "questions": {"decision": q}})
            p1 = FakeVLLM.prompts[-1]
            s2, b2 = call(server_port, "/v1/systemone", {"state": state, "questions": {"decision": q}})
            p2 = FakeVLLM.prompts[-1]
            a1, a2 = b1["answers"]["decision"], b2["answers"]["decision"]
            ok = s1 == s2 == 200 and p1 == p2 and (
                a1["noul"] == a2["noul"] if qtype == "noul" else a1["probabilities"] == a2["probabilities"])
            same += ok
            if not ok:
                print("   differs:", qtype, marker, a1, a2)
    check(f"identity with cygnet_shim at T {T}: {same} of {5 * len(cases)} questions send the same prompt and give exactly "
          f"the same probabilities (text, JSON, empty text, empty object and empty list states)", same == 5 * len(cases))
same = 0
for instructions in (None, ""):
    q = {"type": "choice", "instructions": instructions, "criteria": options(3)}
    s1, b1 = call(frozen_port, "/v1/systemone", {"state": "s", "questions": {"decision": q}})
    p1 = FakeVLLM.prompts[-1]
    s2, b2 = call(server_port, "/v1/systemone", {"state": "s", "questions": {"decision": q}})
    same += s1 == s2 == 200 and p1 == FakeVLLM.prompts[-1] and (
        b1["answers"]["decision"]["probabilities"] == b2["answers"]["decision"]["probabilities"])
check("null and empty instructions send the same prompt as cygnet_shim", same == 2)
frozen.TEMPERATURE = server.shim.TEMPERATURE = 1.0

# ---- 2. the API
FakeVLLM.prompts.clear()
st, b = ask({"id_route_x7": {"type": "choice", "instructions": "Which team? MOCK_PICK=B",
                       "criteria": {"billing": "Payments", "technical": "Bugs", "sales": "Pricing"}},
             "id_urgent_x7": {"type": "noul", "instructions": "Is it urgent? MOCK_PICK=B"},
             "id_anger_x7": {"type": "score", "instructions": "How angry? MOCK_PICK=C", "criteria": ["Calm", "Annoyed", "Angry"]}})
check("three named questions in one request are all answered, under their names",
      st == 200 and set(b["answers"]) == {"id_route_x7", "id_urgent_x7", "id_anger_x7"}, b)
check("names never reach the model", not any(n in p for p in FakeVLLM.prompts for n in ("id_route_x7", "id_urgent_x7", "id_anger_x7")))
ch, no, sc = b["answers"]["id_route_x7"], b["answers"]["id_urgent_x7"], b["answers"]["id_anger_x7"]
k, pmax = 3, max(ch["probabilities"].values())
check("choice: choice, probabilities summing to 1, confidence = (K p_max - 1) / (K - 1)",
      ch["type"] == "choice" and ch["choice"] == "technical" and abs(sum(ch["probabilities"].values()) - 1) < 1e-9
      and abs(ch["confidence"] - (k * pmax - 1) / (k - 1)) < 1e-12, ch)
check("noul without criteria is answered; yes is P(true)", no == {"type": "noul", "noul": no["noul"]} and no["noul"] > 0.9, no)
check("noul without criteria is shown as 'A. No' then 'B. Yes'",
      any(option_lines(p) == {"A": "No", "B": "Yes"} for p in FakeVLLM.prompts), FakeVLLM.prompts)
check("score: probability-weighted score, legend and level-keyed probabilities, confidence",
      sc["type"] == "score" and sc["legend"] == {"0": "Calm", "1": "Annoyed", "2": "Angry"}
      and set(sc["probabilities"]) == {"0", "1", "2"}
      and abs(sc["score"] - sum(int(i) * p for i, p in sc["probabilities"].items())) < 1e-12
      and 0 <= sc["confidence"] <= 1 and sc["score"] > 1.8, sc)
check("usage is input_tokens / output_tokens, summed over the passes; model is the server's name",
      b["usage"]["output_tokens"] == 3 and b["usage"]["input_tokens"] > 0 and b["model"] == "cygnet-test", b)

FakeVLLM.prompts.clear()
ask({"q": {"type": "noul", "instructions": "Given true first. MOCK_PICK=A",
           "criteria": {"true": "It is urgent.", "false": "It is not urgent."}}})
check("noul criteria given true first are shown false first", option_lines(FakeVLLM.prompts[-1]) ==
      {"A": "It is not urgent.", "B": "It is urgent."}, FakeVLLM.prompts[-1:])
st, b = ask({"q": {"type": "noul", "instructions": "One side. MOCK_PICK=B", "criteria": {"true": "Urgent."}}})
check("noul with only one side given fills the other", st == 200 and option_lines(FakeVLLM.prompts[-1]) ==
      {"A": "No", "B": "Urgent."}, FakeVLLM.prompts[-1:])
check("noul criteria with another key is a 422", ask({"q": {"type": "noul", "instructions": "x",
                                                             "criteria": {"true": "y", "maybe": "m"}}})[0] == 422)
st, b = ask({"q": {"type": "choice", "instructions": "Pick. MOCK_PICK=A",
                   "criteria": {"calm": None, "orders": {"what": "Order status", "examples": ["Where is it?"]},
                                "empty": "", "plain": "A plain description"}}})
check("choice descriptions: null and empty show the name, JSON follows the name, text is shown as it is",
      st == 200 and option_lines(FakeVLLM.prompts[-1]) == {
          "A": "calm", "B": 'orders: {"what": "Order status", "examples": ["Where is it?"]}', "C": "empty",
          "D": "A plain description"}, FakeVLLM.prompts[-1:])
st, b = ask({"q": {"type": "score", "instructions": "Rate. MOCK_PICK=A", "criteria": [None, {"level": "high"}]}})
check("score levels: null shows 'Level n', JSON is shown as JSON; legend keeps the given values",
      st == 200 and option_lines(FakeVLLM.prompts[-1]) == {"A": "Level 0", "B": '{"level": "high"}'}
      and b["answers"]["q"]["legend"] == {"0": "", "1": {"level": "high"}}, b)
st, b = ask({"q": {"type": "choice", "instructions": "Only one.", "criteria": {"only": "the only option"}}})
check("a Choice with one option answers without a model call, confidence 1",
      st == 200 and b["answers"]["q"]["choice"] == "only" and b["answers"]["q"]["confidence"] == 1.0
      and b["usage"] == {"input_tokens": 0, "output_tokens": 0}, b)
st, b = ask({"q": {"type": "choice", "instructions": "Array state. MOCK_PICK=A", "criteria": options(3)}},
            state=[{"from": "customer", "text": "hi"}])
check("an array state is accepted", st == 200, b)
st, b = ask({"q": {"type": "choice", "instructions": {"question": "Structured. MOCK_PICK=A", "data": [1, 2]},
                   "criteria": options(3)}})
check("structured instructions are accepted", st == 200, b)

check("255 options are answered", ask({"q": {"type": "choice", "instructions": "x MOCK_PICK=A",
                                              "criteria": options(255)}})[0] == 200)
check("256 options is a 422", ask({"q": {"type": "choice", "instructions": "x", "criteria": options(256)}})[0] == 422)
check("a Score with 11 levels is a 422", ask({"q": {"type": "score", "instructions": "x",
                                                    "criteria": [str(i) for i in range(11)]}})[0] == 422)
check("an unknown question type is a 422", ask({"q": {"type": "ranking", "instructions": "x", "criteria": options(3)}})[0] == 422)
check("a request without questions is a 422", call(server_port, "/v1/systemone", {"state": "s"})[0] == 422)
check("a body that is not JSON is a 400", call(server_port, "/v1/systemone", raw=b"{not json")[0] == 400)
check("a JSON body that is not an object is a 422", call(server_port, "/v1/systemone", raw=b"[1, 2]")[0] == 422)
check("over the model's context is a 422", ask({"q": {"type": "choice", "instructions": "MOCK_OVERFLOW",
                                                       "criteria": options(3)}})[0] == 422)
check("a request with one over-context question and one fine question is a 422",
      ask({"a": {"type": "choice", "instructions": "MOCK_OVERFLOW", "criteria": options(3)},
           "b": {"type": "choice", "instructions": "MOCK_PICK=A", "criteria": options(3)}})[0] == 422)
check("a request with one over-context question and one vLLM failure is a 422 (the error a retry cannot fix)",
      ask({"a": {"type": "choice", "instructions": "MOCK_500", "criteria": options(3)},
           "b": {"type": "choice", "instructions": "MOCK_OVERFLOW", "criteria": options(3)}})[0] == 422)
check("vLLM's 429 passes through ahead of a 422 in the same request",
      ask({"a": {"type": "choice", "instructions": "MOCK_OVERFLOW", "criteria": options(3)},
           "b": {"type": "choice", "instructions": "MOCK_429", "criteria": options(3)}})[0] == 429)
check("vLLM 429 passes through", ask({"q": {"type": "choice", "instructions": "MOCK_429", "criteria": options(3)}})[0] == 429)
check("vLLM 500 is a 502", ask({"q": {"type": "choice", "instructions": "MOCK_500", "criteria": options(3)}})[0] == 502)
st, b = call(server_port, "/v1/models")
check("GET /v1/models lists name, description and release_date",
      st == 200 and set(b["models"][0]) == {"name", "description", "release_date"} and b["models"][0]["name"] == "cygnet-test", b)
check("an unknown path is a 404", call(server_port, "/v1/other")[0] == 404)

server.API_KEY = "k-123"
check("with CYGNET_API_KEY: no key is a 401", ask({"q": {"type": "noul", "instructions": "x"}})[0] == 401)
check("with CYGNET_API_KEY: a wrong key is a 401",
      ask({"q": {"type": "noul", "instructions": "x"}}, headers={"Authorization": "Bearer nope"})[0] == 401)
check("with CYGNET_API_KEY: the right key is answered",
      ask({"q": {"type": "noul", "instructions": "x MOCK_PICK=A"}}, headers={"Authorization": "Bearer k-123"})[0] == 200)
check("with CYGNET_API_KEY: the models listing needs the key too", call(server_port, "/v1/models")[0] == 401)
server.API_KEY = ""
check("without CYGNET_API_KEY any Authorization header is accepted",
      ask({"q": {"type": "noul", "instructions": "x MOCK_PICK=A"}}, headers={"Authorization": "Bearer anything"})[0] == 200)

FakeVLLM.served_id, server.shim._server_context = "another-model", None
check("a vLLM that does not serve SHIM_MODEL is a 503", ask({"q": {"type": "noul", "instructions": "x"}})[0] == 503)
FakeVLLM.served_id, FakeVLLM.max_model_len, server.shim._server_context = "mock", 2048, None
check("a vLLM below SHIM_MIN_CONTEXT is a 503", ask({"q": {"type": "noul", "instructions": "x"}})[0] == 503)
FakeVLLM.max_model_len, server.shim._server_context = 262144, None

picks = "ABCDEFGHIJKL"
st, b = ask({f"q{i}": {"type": "choice", "instructions": f"Pick. MOCK_PICK={p}", "criteria": options(12)}
             for i, p in enumerate(picks)})
check("12 questions run concurrently and each gets its own answer",
      st == 200 and all(b["answers"][f"q{i}"]["choice"] == f"label_{i}" for i in range(12)), b)

import socket
import subprocess


def raw_post(head):
    """Send a request head exactly as given (no body), return the status code."""
    with socket.create_connection(("127.0.0.1", server_port), timeout=10) as c:
        c.sendall(head.encode())
        return int(c.recv(4096).split(b" ", 2)[1])


check("a POST without Content-Length is a 411", raw_post("POST /v1/systemone HTTP/1.1\r\nHost: x\r\n\r\n") == 411)
check("a POST with a bad Content-Length is a 400",
      raw_post("POST /v1/systemone HTTP/1.1\r\nHost: x\r\nContent-Length: -5\r\n\r\n") == 400)


def start(env):
    """Start the server with extra environment and expect it to stop at once; (return code, stderr)."""
    try:
        r = subprocess.run([sys.executable, os.path.join(HERE, "decision_server.py")], env={**os.environ, **env},
                           capture_output=True, text=True, timeout=10)
    except subprocess.TimeoutExpired:
        return None, "it started and kept running"
    return r.returncode, r.stderr


rc, err = start({"CYGNET_GROUP_SIZE": "12", "CYGNET_PORT": "1"})
check("CYGNET_GROUP_SIZE 12 is refused: 255 options would need 22 group winners in one final pass",
      rc != 0 and "CYGNET_GROUP_SIZE must be between 13 and 20" in err, err[-200:])
rc, err = start({"CYGNET_HOST": "192.0.2.1", "CYGNET_API_KEY": "", "CYGNET_PORT": "1"})   # a documentation address
check("the server refuses to listen beyond localhost without CYGNET_API_KEY", rc != 0 and "refusing to listen" in err, err[-200:])

server._in_flight = threading.BoundedSemaphore(2)
FakeVLLM.peak = 0
st, b = ask({f"q{i}": {"type": "choice", "instructions": f"Pick. MOCK_SLOW MOCK_PICK={p}", "criteria": options(12)}
             for i, p in enumerate("ABCDEFGH")} | {"big": {"type": "choice", "instructions": "MOCK_SLOW MOCK_TARGET=zebra",
                                                            "criteria": options(45, "zebra", 30)}})
check(f"CYGNET_MAX_PARALLEL caps vLLM requests in flight across questions and group passes (peak {FakeVLLM.peak}, cap 2)",
      st == 200 and FakeVLLM.peak <= 2 and b["answers"]["big"]["choice"] == "label_30", (st, FakeVLLM.peak))
server._in_flight = threading.BoundedSemaphore(server.MAX_PARALLEL)

# ---- 3. the grouped readout
G = server.GROUP_SIZE
for K in (21, 40, 64, 255):
    for at in sorted({0, K // 2, K - 1}):
        FakeVLLM.prompts.clear()
        st, b = ask({"q": {"type": "choice", "instructions": "Find the zebra. MOCK_TARGET=zebra",
                           "criteria": options(K, "zebra", at)}})
        a = b["answers"]["q"] if st == 200 else {}
        m = -(-K // G)
        passes = len(FakeVLLM.prompts)
        final = option_lines(FakeVLLM.prompts[-1]) if passes else {}
        check(f"{K} options, the zebra at {at}: chosen with p {a.get('probabilities', {}).get(f'label_{at}', 0):.3f}; "
              f"{passes} passes (want {m + 1}); final pass shows each winner's own text",
              st == 200 and a["choice"] == f"label_{at}" and passes == m + 1 and len(set(final.values())) == m
              and sum(1 for t in final.values() if "zebra" in t) == 1, (st, b if st != 200 else final))

FakeVLLM.prompts.clear()
ask({"q": {"type": "choice", "instructions": "x MOCK_TARGET=zebra", "criteria": options(21, "zebra", 3)}})
sizes = sorted(len(option_lines(p)) for p in FakeVLLM.prompts)
check(f"21 options are read as groups of near-equal size, then the winners: pass sizes {sizes}", sizes == [2, 10, 11])

# composition against a reference computed independently from the stand-in's scoring rule
def softmax(v):
    m = max(v); e = [math.exp(x - m) for x in v]; t = sum(e)
    return [x / t for x in e]


def reference(K, at):
    """What the grouped readout must return for the stand-in: near-equal groups, a softmax per group over the
    stand-in's logits (0 or -4 by content, minus 0.01 per position in the pass), the winners' softmax the same way,
    P(option) = P(winner of its group) x P(option | group)."""
    m = -(-K // G); bounds = [round(i * K / m) for i in range(m + 1)]
    base = [0.0 if i == at else -4.0 for i in range(K)]
    within, winners = [], []
    for g in range(m):
        idx = list(range(bounds[g], bounds[g + 1]))
        p = softmax([base[i] - 0.01 * j for j, i in enumerate(idx)])
        within.append((idx, p)); winners.append(idx[max(range(len(idx)), key=lambda j: p[j])])
    q = softmax([base[w] - 0.01 * g for g, w in enumerate(winners)])
    out = {f"label_{i}": q[g] * pi for g, (idx, p) in enumerate(within) for i, pi in zip(idx, p)}
    s = sum(out.values())
    return {k: v / s for k, v in out.items()}


for K, at in ((21, 20), (64, 40), (255, 3)):
    st, b = ask({"q": {"type": "choice", "instructions": "x MOCK_TARGET=zebra", "criteria": options(K, "zebra", at)}})
    got, want = b["answers"]["q"]["probabilities"], reference(K, at)
    err = max(abs(got[k] - want[k]) for k in want)
    check(f"{K} options: the composed probabilities equal an independent reference computation (largest difference {err:.1e})",
          err < 1e-12 and set(got) == set(want), err)
K, at = 64, 40
st, b = ask({"q": {"type": "choice", "instructions": "x MOCK_TARGET=zebra", "criteria": options(K, "zebra", at)}})
got = b["answers"]["q"]["probabilities"]
server.shim.TEMPERATURE = 2.0
st, b2 = ask({"q": {"type": "choice", "instructions": "x MOCK_TARGET=zebra", "criteria": options(K, "zebra", at)}})
server.shim.TEMPERATURE = 1.0
z = {k: v ** 0.5 for k, v in got.items()}
s = sum(z.values())
check("with T 2 the temperature is applied once, to the composed distribution",
      all(abs(b2["answers"]["q"]["probabilities"][k] - v / s) < 1e-12 for k, v in z.items()))

print(f"\n{'all passed' if not failures else f'{failures} failed'}")
sys.exit(1 if failures else 0)
