#!/usr/bin/env python3
"""Cygnet as a System One server.

WHAT THIS IS
    The recipe's shim/cygnet_shim.py implements the slice of the System One protocol that
    JevBench's `typesafe` adapter needs: ONE question, named "decision", returning `type`, `choice`
    and `probabilities`. Everything else about the recipe -- `options_from`, `build_prompt`,
    `call_vllm`, `letter_probs`, the calibration temperature -- is reused here UNCHANGED, by
    importing the recipe module. This file only makes the server protocol-complete and backend
    agnostic.

    It is written to be correct regardless of what is behind it: vLLM on CUDA (the documented
    deployment), vLLM on ROCm, llama.cpp on a GPU, or llama.cpp on a CPU. Nothing here is tuned to
    one machine. The knobs that DO depend on your hardware are environment variables, with their
    tradeoffs documented next to them.

PROTOCOL GAPS CLOSED (each one observed against a real client, not inferred)
    1. Every question in `questions` is answered, keyed by name. The recipe answered only
       "decision" and returned HTTP 400 otherwise.
       -> jev-ultrafast sends operation + click_target + fill_target together.
    2. `confidence` is returned on choice and score answers, as the chosen option's probability.
       -> required by typesafe-sdk's ChoiceAnswer; optional for @ai-sdk/typesafe-ai.
    3. `noul` / boolean criteria may be omitted. The real API documents Noul.criteria as optional;
       the recipe required a "true" key and answered HTTP 422.
       -> jevgrep sends {"type": "boolean", "instructions": ...} with no criteria.
    4. `score` answers carry `score`, `legend` (the rubric) and integer-keyed probabilities.
    5. `usage` uses the documented `input_tokens` / `output_tokens` names, alongside the
       `prompt_tokens` / `completion_tokens` the recipe's own accounting used.
    6. `GET /v1/models` serves BOTH shapes: `models` (TypeSafe: name/description/release_date) and
       `data` (OpenAI: id/max_model_len, which the recipe's context guard reads). type-safe-sdk
       calls this endpoint; llama.cpp servers also emit both.
    7. `Authorization: Bearer ...` is accepted and ignored.
    8. More than 26 options are supported (see LABEL WIDTH below).

LABEL WIDTH -- the readout generalises past 26 options
    The recipe presents options as letters A..Z, which caps a question at 26 options. That cap is
    fatal for browser agents: jev-ultrafast builds one option per observable DOM element, so any
    page with more than 26 controls fails. The fix does not change the readout, it composes it:

        options 1..K, split into groups of at most 26
        pass per group   -> a distribution within that group
        one extra pass   -> a distribution over the group winners
        P(option) = P(its group wins) * P(option | group)

    So K <= 676 options costs ceil(K/26) + 1 single-token passes. Each pass is still the recipe's
    one-token letter readout, so the validated machinery is untouched. Set SHIM_LABEL_GROUPS=0 to
    restore the hard 26-option refusal.

CONCURRENCY -- read this, it is hardware dependent
    Questions within one request are independent, so they are dispatched concurrently. Whether that
    helps depends entirely on the server behind this shim:

      * vLLM (CUDA or ROCm): continuous batching turns concurrency into real throughput. Keep
        SHIM_MAX_PARALLEL high (16-64). This is the documented deployment and the default is sized
        for it.
      * llama.cpp on CPU: compute-bound. Measured on an 8-core Ryzen 9, 8-way concurrency gave
        1.5x at best, was slower for small batches, and DISABLED the prompt cache that makes a
        shared state cheap (a 2k-token state costs ~22 s to prefill once, then ~0.7 s per further
        question when serial). Set SHIM_MAX_PARALLEL=1 there, and give the server one slot:
        `llama-server -np 1`.

    The default below is chosen for the documented vLLM deployment. On CPU it is the wrong choice;
    that is a hardware setting, not a design decision.

Usage:
    SHIM_VLLM=http://127.0.0.1:8890/v1/chat/completions SHIM_MODEL=cygnet SHIM_PORT=8009 \
        SHIM_TEMPERATURE=3.4 python3 systemone_shim.py
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from http.server import ThreadingHTTPServer

_HERE = os.path.dirname(os.path.abspath(__file__))
# Works both when this file sits next to cygnet_shim.py (i.e. inside the recipe's shim/) and in an
# analysis checkout that keeps the recipe under recipe/shim/.
_RECIPE_CANDIDATES = (
    os.path.join(_HERE, "cygnet_shim.py"),
    os.path.join(_HERE, "recipe", "shim", "cygnet_shim.py"),
)
RECIPE = os.environ.get("CYGNET_SHIM") or next(
    (p for p in _RECIPE_CANDIDATES if os.path.exists(p)), _RECIPE_CANDIDATES[0])

# Import the recipe's shim so its readout is reused verbatim, never re-implemented.
_spec = importlib.util.spec_from_file_location("cygnet_shim", RECIPE)
shim = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(shim)

MODEL = os.environ.get("SHIM_MODEL", "cygnet")
PORT = int(os.environ.get("SHIM_PORT", "8009"))
LETTERS = shim.LETTERS

# Sized for vLLM's continuous batching. On a CPU backend set this to 1 (see CONCURRENCY above).
MAX_PARALLEL = int(os.environ.get("SHIM_MAX_PARALLEL", "16"))
# 0 disables the >26-option composition and restores the recipe's refusal.
LABEL_GROUPS = os.environ.get("SHIM_LABEL_GROUPS", "1") not in ("0", "false", "False")
MAX_OPTIONS = len(LETTERS) * len(LETTERS) if LABEL_GROUPS else len(LETTERS)

MODEL_DESCRIPTION = os.environ.get(
    "SHIM_MODEL_DESCRIPTION",
    "Gemma-4-12B-it behind the Cygnet one-token option readout, served over a System One API.")
MODEL_RELEASE_DATE = os.environ.get("SHIM_MODEL_RELEASE_DATE", "2025-01-01")

_context_lock = threading.Lock()


def _pass(state, instructions, group):
    """One recipe readout pass over <=26 options. Returns {index: probability}, renormalised.

    `group` is a list of (label, description) in presentation order.
    """
    letters = LETTERS[:len(group)]
    opts = [(letters[i], label, desc) for i, (label, desc) in enumerate(group)]
    resp = shim.call_vllm(shim.build_prompt(state, instructions, opts), letters)

    usage = resp.get("usage") or {}
    try:
        lp = resp["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        top = [(d["token"], d["logprob"]) for d in lp]
    except (KeyError, IndexError, TypeError):
        raise shim.UpstreamError("the model server returned no top_logprobs at the answer slot")

    probs = shim.letter_probs(top, len(group))
    if probs is None:
        raise shim.UpstreamError("could not recover an option distribution from the answer slot")

    if shim.TEMPERATURE != 1.0:
        z = {i: max(p, 1e-12) ** (1.0 / shim.TEMPERATURE) for i, p in probs.items()}
        zs = sum(z.values())
        probs = {i: v / zs for i, v in z.items()}
    return probs, usage


def _distribution(state, instructions, options):
    """Normalised distribution over `options` (list of (label, description)).

    One pass when there are at most 26; otherwise a group pass per 26 plus one pass over the group
    winners, multiplied back together. Always uses the recipe's one-token letter readout per pass.
    """
    n = len(options)
    if n <= len(LETTERS):
        probs, usage = _pass(state, instructions, options)
        return {options[i][0]: probs.get(i, 0.0) for i in range(n)}, usage

    groups = [options[i:i + len(LETTERS)] for i in range(0, n, len(LETTERS))]
    if len(groups) > len(LETTERS):
        raise shim.Unprocessable(
            f"{n} options exceeds the {MAX_OPTIONS}-option limit of the composed readout")

    within, tokens = [], {"prompt_tokens": 0, "completion_tokens": 0}
    for g in groups:
        p, u = _pass(state, instructions, g)
        within.append({g[i][0]: p.get(i, 0.0) for i in range(len(g))})
        tokens["prompt_tokens"] += u.get("prompt_tokens") or 0
        tokens["completion_tokens"] += u.get("completion_tokens") or 0

    winners = [(max(w, key=w.get), "the option this subset ranked highest") for w in within]
    gp, u = _pass(state, instructions, winners)
    tokens["prompt_tokens"] += u.get("prompt_tokens") or 0
    tokens["completion_tokens"] += u.get("completion_tokens") or 0

    out = {}
    for gi, w in enumerate(within):
        weight = gp.get(gi, 0.0)
        for label, p in w.items():
            out[label] = weight * p
    return out, tokens


def _spec_options(spec):
    """(qtype, instructions, [(label, description)]) from a System One question object."""
    qtype = spec.get("type")
    if qtype not in ("choice", "score", "noul"):
        raise shim.Unprocessable(f"unsupported question type {qtype!r}")
    instructions = spec.get("instructions") or ""
    criteria = spec.get("criteria")

    if qtype == "score":
        if criteria is None:
            raise shim.Unprocessable("score without a criteria list")
        if not isinstance(criteria, (list, tuple)):
            raise shim.Unprocessable("score criteria must be an ordered list of levels")
        return qtype, instructions, [(str(i), desc) for i, desc in enumerate(criteria)]

    if qtype == "noul":
        # The real API documents Noul.criteria as optional; only Noul allows this.
        if not criteria:
            criteria = {"true": "yes", "false": "no"}
    if not isinstance(criteria, dict):
        raise shim.Unprocessable(f"{qtype} criteria must be a mapping of label to description")
    if qtype == "noul" and not any(str(k).lower() == "true" for k in criteria):
        raise shim.Unprocessable("noul criteria had no 'true' key")
    if not criteria:
        raise shim.Unprocessable("no options")
    return qtype, instructions, list(criteria.items())


def _answer(state, spec):
    """One question -> (answer object, usage)."""
    qtype, instructions, options = _spec_options(spec)
    if len(options) > MAX_OPTIONS:
        raise shim.Unprocessable(
            f"{len(options)} options exceeds the {MAX_OPTIONS}-option limit")
    dist, usage = _distribution(state, instructions, options)

    if qtype == "noul":
        total = sum(dist.values())
        p_true = dist.get("true", 0.0) / total if total > 0 else 0.0
        return {"type": "noul", "noul": p_true}, usage

    total = sum(dist.values())
    if total <= 0:
        raise shim.UpstreamError("empty distribution")
    dist = {k: v / total for k, v in dist.items()}
    best = max(dist, key=dist.get)
    answer = {"type": qtype, "choice": best, "probabilities": dist, "confidence": dist[best]}
    if qtype == "score":
        # ScoreAnswer carries its rubric so a stored score stays self-describing.
        answer["score"] = int(best) if str(best).isdigit() else best
        answer["legend"] = {int(i): desc for i, (_l, desc) in enumerate(options)}
        answer["probabilities"] = {int(i): dist.get(str(i), 0.0) for i in range(len(options))}
    return answer, usage


def _models_payload(max_model_len):
    """Serve both shapes: `models` for TypeSafe clients, `data` for the recipe's own guard."""
    return {
        "models": [{"name": MODEL,
                    "description": MODEL_DESCRIPTION,
                    "release_date": MODEL_RELEASE_DATE}],
        "object": "list",
        "data": [{"id": MODEL, "object": "model", "max_model_len": max_model_len}],
    }


class Handler(shim.Handler):
    def do_GET(self):
        if self.path.startswith("/v1/models"):
            try:
                ctx = shim.server_context()
            except shim.UpstreamError as e:
                self._send(e.status, {"error": str(e)})
                return
            self._send(200, _models_payload(ctx))
        elif self.path.startswith("/health"):
            self._send(200, {"status": "ok"})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if not self.path.startswith("/v1/systemone"):
            self._send(404, {"error": "not found"})
            return
        try:
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
            questions = body.get("questions") if isinstance(body, dict) else None
            if not isinstance(questions, dict) or not questions:
                raise ValueError("expected a JSON object with a non-empty questions mapping")
        except Exception as e:
            self._send(400, {"error": f"bad request body: {e}"})
            return

        try:
            context = shim.server_context()
            if context < shim.MIN_CONTEXT:
                raise shim.UpstreamError(
                    f"the server's max_model_len is {context}, below SHIM_MIN_CONTEXT="
                    f"{shim.MIN_CONTEXT}", 503)
            state = body.get("state") or ""
            for name, spec in questions.items():
                if not isinstance(spec, dict):
                    raise shim.Unprocessable(f"question {name!r} is not an object")

            answers, prompt_tokens, completion_tokens = {}, 0, 0
            with ThreadPoolExecutor(max_workers=min(len(questions), MAX_PARALLEL)) as pool:
                futures = [(name, pool.submit(_answer, state, spec))
                           for name, spec in questions.items()]
                for name, future in futures:
                    answer, usage = future.result()
                    answers[name] = answer
                    prompt_tokens += usage.get("prompt_tokens") or 0
                    completion_tokens += usage.get("completion_tokens") or 0
        except shim.Unprocessable as e:
            sys.stderr.write(f"systemone 422 {e}\n")
            self._send(422, {"error": str(e)})
            return
        except shim.UpstreamError as e:
            self._send(e.status, {"error": str(e)})
            return
        except Exception as e:
            self._send(500, {"error": f"{type(e).__name__}: {e}"})
            return

        self._send(200, {
            "model": MODEL,
            "answers": answers,
            "usage": {
                # Documented TypeSafe names, plus the recipe's own names for its accounting.
                "input_tokens": prompt_tokens,
                "output_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
        })


if __name__ == "__main__":
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"Cygnet System One server on 127.0.0.1:{PORT} -> {shim.VLLM} "
          f"(model {MODEL}, max {MAX_OPTIONS} options, parallel {MAX_PARALLEL})", flush=True)
    srv.serve_forever()
