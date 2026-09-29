#!/usr/bin/env python3
"""Cygnet decision server: typed decisions for applications, on the benchmark's one-token readout.

`cygnet_shim.py` is the file the benchmark measured, and it stays as it is. This server imports it and uses its
prompt builder, its vLLM call, its letter parser, its context check and its calibration temperature unchanged. What it
adds is what an application needs from the same API, `POST /v1/systemone` and `GET /v1/models`:

  - Every question in a request is answered, under the name the caller gave it. Names never reach the model; each
    question is read in its own pass over the same state, and questions run concurrently.
  - Choice and Score answers carry `confidence`: (K * p_max - 1) / (K - 1) over the K options or levels, clipped to
    [0, 1] (1 for a single option). Score answers carry `score` (the probability-weighted level), `legend` and
    level-keyed `probabilities`. Usage is reported as `input_tokens` and `output_tokens`.
  - Noul `criteria` is optional, and so is either side of it. The two options are always shown "false" first, the
    order Cygnet was measured in; the answer is P(true) either way.
  - Descriptions may be text, JSON or null. Text is shown as it is (the benchmark path); JSON is shown after the option
    name; a null or empty Choice description shows the option name, a null Score level shows "Level <n>".
  - A Choice takes up to 255 options. Up to CYGNET_GROUP_SIZE options (default 20, the number of letters vLLM returns
    logprobs for) are read in one pass. Past that the options are read in groups, one pass per group, then one pass
    over the group winners, each shown with its own description; P(option) = P(its group's winner) x P(option |
    its group), and the calibration temperature is applied once to that result.
  - With CYGNET_API_KEY set, requests need `Authorization: Bearer <key>` (401 otherwise); without it any
    Authorization header is accepted.

Identity with the benchmark path: one question with at most CYGNET_GROUP_SIZE options, text descriptions and noul
criteria given false first is read exactly as `cygnet_shim.answer_for` reads it (`test_decision_server.py` compares the
two on every question type).

Status codes: 422 for a request this server cannot answer (a malformed question, over 255 options, over the model's
context); 401 as above; vLLM's 401, 403 and 429 pass through; 502 when vLLM fails; 503 when the served model's context
is below SHIM_MIN_CONTEXT; 400 for a body that is not JSON; 413 for a body over CYGNET_MAX_BODY bytes.

    SHIM_VLLM=http://127.0.0.1:8890/v1/chat/completions SHIM_MODEL=cygnet SHIM_TEMPERATURE=3.4 \\
        python3 shim/decision_server.py

The ideas this server takes from others are credited in CREDITS.md; the code is ours.
"""
from __future__ import annotations

import hmac
import importlib.util
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_spec = importlib.util.spec_from_file_location("cygnet_shim", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                          "cygnet_shim.py"))
shim = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(shim)

HOST = os.environ.get("CYGNET_HOST", "127.0.0.1")
PORT = int(os.environ.get("CYGNET_PORT", "8010"))
API_KEY = os.environ.get("CYGNET_API_KEY", "")
MAX_PARALLEL = int(os.environ.get("CYGNET_MAX_PARALLEL", "8"))
GROUP_SIZE = int(os.environ.get("CYGNET_GROUP_SIZE", str(min(20, shim.TOP_LOGPROBS))))
MAX_BODY = int(os.environ.get("CYGNET_MAX_BODY", str(16 * 1024 * 1024)))
MODEL_NAME = os.environ.get("CYGNET_MODEL_NAME", shim.MODEL)
MODEL_DESCRIPTION = os.environ.get("CYGNET_MODEL_DESCRIPTION",
                                   "google/gemma-4-12B-it with a one-token option readout (Cygnet)")
MODEL_RELEASE_DATE = os.environ.get("CYGNET_MODEL_RELEASE_DATE", "2026-09-24")
MAX_CHOICE_OPTIONS = 255
MAX_SCORE_LEVELS = 10
if not 2 <= GROUP_SIZE <= min(len(shim.LETTERS), shim.TOP_LOGPROBS):
    raise SystemExit(f"CYGNET_GROUP_SIZE must be between 2 and {min(len(shim.LETTERS), shim.TOP_LOGPROBS)}")

_passes = ThreadPoolExecutor(max_workers=MAX_PARALLEL)    # group passes within a question


def _text(value):
    """A description as the model sees it: text as it is, anything else as compact JSON."""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _blank(value):
    return value is None or (isinstance(value, str) and not value.strip())


def parse_question(name, q):
    """(type, instructions, [(label, shown text)], legend or None), or Unprocessable naming what is wrong."""
    where = f"questions.{name}"
    if not isinstance(q, dict):
        raise shim.Unprocessable(f"{where} must be an object")
    qtype = q.get("type")
    instructions = q.get("instructions")
    instructions = "" if instructions is None else instructions
    criteria = q.get("criteria")
    if qtype == "choice":
        if not isinstance(criteria, dict) or not criteria:
            raise shim.Unprocessable(f"{where}.criteria must be a non-empty object of options")
        if len(criteria) > MAX_CHOICE_OPTIONS:
            raise shim.Unprocessable(f"{where} has {len(criteria)} options; a Choice takes at most {MAX_CHOICE_OPTIONS}")
        items = []
        for label, desc in criteria.items():
            if not isinstance(desc, (str, dict, list)) and desc is not None:
                raise shim.Unprocessable(f"{where}.criteria.{label} must be text, JSON or null")
            items.append((label, label if _blank(desc) else desc if isinstance(desc, str) else f"{label}: {_text(desc)}"))
        return qtype, instructions, items, None
    if qtype == "score":
        if not isinstance(criteria, list) or not criteria:
            raise shim.Unprocessable(f"{where}.criteria must be a non-empty list of levels")
        if len(criteria) > MAX_SCORE_LEVELS:
            raise shim.Unprocessable(f"{where} has {len(criteria)} levels; a Score takes at most {MAX_SCORE_LEVELS}")
        items, legend = [], {}
        for i, desc in enumerate(criteria):
            if not isinstance(desc, (str, dict, list)) and desc is not None:
                raise shim.Unprocessable(f"{where}.criteria[{i}] must be text, JSON or null")
            items.append((str(i), f"Level {i}" if _blank(desc) else _text(desc)))
            legend[str(i)] = "" if desc is None else desc
        return qtype, instructions, items, legend
    if qtype == "noul":
        if criteria is None:
            criteria = {}
        if not isinstance(criteria, dict):
            raise shim.Unprocessable(f"{where}.criteria must be an object with 'true' and 'false' descriptions")
        sides = {}
        for key, desc in criteria.items():
            side = str(key).lower()
            if side not in ("true", "false") or side in sides:
                raise shim.Unprocessable(f"{where}.criteria takes only 'true' and 'false', once each")
            if not isinstance(desc, (str, dict, list)) and desc is not None:
                raise shim.Unprocessable(f"{where}.criteria.{key} must be text, JSON or null")
            sides[side] = desc
        items = [("false", "No" if _blank(sides.get("false")) else _text(sides["false"])),
                 ("true", "Yes" if _blank(sides.get("true")) else _text(sides["true"]))]
        return qtype, instructions, items, None
    raise shim.Unprocessable(f"{where}.type must be 'choice', 'score' or 'noul', not {qtype!r}")


def read_pass(state, instructions, items):
    """One benchmark-path pass over at most GROUP_SIZE options: raw probabilities (before the temperature) and usage.
    A single option needs no pass."""
    if len(items) == 1:
        return [1.0], {}
    opts = [(shim.LETTERS[i], label, text) for i, (label, text) in enumerate(items)]
    resp = shim.call_vllm(shim.build_prompt(state, instructions, opts), [letter for letter, _l, _t in opts])
    usage = resp.get("usage") or {}
    try:
        # a list of (token, logprob) pairs: Gemma-4 has two tokens that decode to each letter (cygnet_shim.answer_for)
        top = [(d["token"], d["logprob"]) for d in resp["choices"][0]["logprobs"]["content"][0]["top_logprobs"]]
    except (KeyError, IndexError, TypeError):
        raise shim.UpstreamError("vLLM returned no top_logprobs at the answer slot")
    probs = shim.letter_probs(top, len(opts))
    if probs is None:
        raise shim.UpstreamError("could not recover an option-letter distribution from the answer slot")
    return [probs.get(i, 0.0) for i in range(len(opts))], usage


def temper(probs):
    """cygnet_shim's calibration: p^(1/T) renormalised, with the same floor; T = 1 leaves it unchanged."""
    if shim.TEMPERATURE == 1.0:
        return list(probs)
    z = [max(p, 1e-12) ** (1.0 / shim.TEMPERATURE) for p in probs]
    s = sum(z)
    return [v / s for v in z]


def distribution(state, instructions, items):
    """Calibrated probabilities over `items` (list of (label, text)), and summed usage."""
    tokens = {"input": 0, "output": 0}

    def count(usage):
        tokens["input"] += usage.get("prompt_tokens") or 0
        tokens["output"] += usage.get("completion_tokens") or 0

    if len(items) <= GROUP_SIZE:
        raw, usage = read_pass(state, instructions, items)
        count(usage)
        return temper(raw), tokens
    # near-equal groups, in the given order, so no winner reaches the final pass through a small group
    m = -(-len(items) // GROUP_SIZE)
    bounds = [round(i * len(items) / m) for i in range(m + 1)]
    groups = [items[bounds[i]:bounds[i + 1]] for i in range(m)]
    reads = list(_passes.map(lambda g: read_pass(state, instructions, g), groups))
    within = []
    for raw, usage in reads:
        count(usage)
        within.append(raw)
    # each group's winner is shown with its own description, so the final pass compares their content
    winners = [group[max(range(len(group)), key=lambda i: p[i])] for group, p in zip(groups, within)]
    between, usage = read_pass(state, instructions, winners)
    count(usage)
    composed = [between[g] * p for g, p_group in enumerate(within) for p in p_group]
    s = sum(composed)
    if s <= 0:
        raise shim.UpstreamError("the grouped readout gave every option zero probability")
    return temper([p / s for p in composed]), tokens


def confidence(probs):
    k = len(probs)
    return 1.0 if k == 1 else max(0.0, min(1.0, (k * max(probs) - 1.0) / (k - 1.0)))


def answer(state, qtype, instructions, items, legend):
    probs, tokens = distribution(state, instructions, items)
    if qtype == "noul":
        return {"type": "noul", "noul": probs[1]}, tokens          # items are (false, true)
    s = sum(probs)                     # cygnet_shim renormalises Choice and Score once more; so do we, for identity
    probs = [p / s for p in probs]
    if qtype == "score":
        return {"type": "score",
                "score": sum(i * p for i, p in enumerate(probs)),
                "legend": legend,
                "probabilities": {str(i): p for i, p in enumerate(probs)},
                "confidence": confidence(probs)}, tokens
    dist = {label: p for (label, _t), p in zip(items, probs)}
    return {"type": "choice", "choice": max(dist, key=dist.get), "probabilities": dist,
            "confidence": confidence(probs)}, tokens


def evaluate(body):
    """(status, response object) for one parsed request body."""
    questions = body.get("questions")
    if not isinstance(questions, dict) or not questions:
        raise shim.Unprocessable("questions must be a non-empty object of named questions")
    parsed = {name: parse_question(name, q) for name, q in questions.items()}
    context = shim.server_context()
    if context < shim.MIN_CONTEXT:
        raise shim.UpstreamError(f"the server's max_model_len is {context}, below SHIM_MIN_CONTEXT={shim.MIN_CONTEXT};"
                                 f" restart vLLM with a larger --max-model-len", 503)
    state = body.get("state")
    state = "" if state is None else state

    def one(name):
        qtype, instructions, items, legend = parsed[name]
        try:
            return name, answer(state, qtype, instructions, items, legend), None
        except (shim.Unprocessable, shim.UpstreamError) as e:
            return name, None, e

    with ThreadPoolExecutor(max_workers=max(1, min(MAX_PARALLEL, len(parsed)))) as pool:
        results = list(pool.map(one, parsed))
    errors = [e for _n, _r, e in results if e is not None]
    unprocessable = [e for e in errors if isinstance(e, shim.Unprocessable)]
    if unprocessable:                  # deterministic: a retry cannot fix it, so it is reported first
        raise unprocessable[0]
    if errors:
        raise errors[0]
    answers, input_tokens, output_tokens = {}, 0, 0
    for name, (ans, tokens), _e in results:
        answers[name] = ans
        input_tokens += tokens["input"]
        output_tokens += tokens["output"]
    return {"model": MODEL_NAME, "answers": answers,
            "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens}}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        sys.stderr.write("decision-server %s\n" % (fmt % args))

    def _send(self, code, obj):
        payload = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _authorised(self):
        if not API_KEY:
            return True
        given = self.headers.get("Authorization") or ""
        return given.startswith("Bearer ") and hmac.compare_digest(given[7:].strip().encode(), API_KEY.encode())

    def do_GET(self):
        if not self.path.startswith("/v1/models"):
            return self._send(404, {"error": "not found"})
        if not self._authorised():
            return self._send(401, {"error": "missing or invalid API key"})
        self._send(200, {"models": [{"name": MODEL_NAME, "description": MODEL_DESCRIPTION,
                                     "release_date": MODEL_RELEASE_DATE}]})

    def do_POST(self):
        if not self.path.startswith("/v1/systemone"):
            return self._send(404, {"error": "not found"})
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_BODY:
            self.close_connection = True           # the unread body must not be taken for the next request
            return self._send(413, {"error": f"request body over {MAX_BODY} bytes"})
        raw = self.rfile.read(n)
        if not self._authorised():
            return self._send(401, {"error": "missing or invalid API key"})
        try:
            body = json.loads(raw or b"{}")
        except ValueError as e:
            return self._send(400, {"error": f"request body is not JSON: {e}"})
        if not isinstance(body, dict):
            return self._send(422, {"error": "request body must be a JSON object"})
        try:
            self._send(200, evaluate(body))
        except shim.Unprocessable as e:
            sys.stderr.write(f"decision-server 422 {e}\n")
            self._send(422, {"error": str(e)})
        except shim.UpstreamError as e:
            self._send(e.status, {"error": str(e)})
        except Exception as e:
            self._send(500, {"error": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    if HOST not in ("127.0.0.1", "localhost", "::1") and not API_KEY:
        sys.stderr.write("decision-server: listening beyond localhost without CYGNET_API_KEY\n")
    srv = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"Cygnet decision server on {HOST}:{PORT} -> {shim.VLLM} (model {shim.MODEL}, T {shim.TEMPERATURE},"
          f" groups of {GROUP_SIZE})", flush=True)
    srv.serve_forever()
