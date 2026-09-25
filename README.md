# Cygnet — typed decisions from frozen Gemma-4-12B, one token per decision

Cygnet answers JevBench's typed decision requests (`choice`, `noul`, `score`) with **frozen
`google/gemma-4-12B-it`**, no fine-tuning, served by **unmodified vLLM 0.30.0**. A small shim presents the options
as letters, reads the model's own probability for each letter at a single answer position, and applies one
calibration temperature. Cost is input tokens only, with one output token per decision.

## Measured (JevBench's public set)

All figures come from **JevBench's own CLI at commit `2fa63fa`**, `--adapter typesafe`, run on the public items
(`datasets/public/easy.jsonl`, `original.jsonl`, `hard.jsonl`; 231 items) with the commands below.

| | RTX A6000 48 GB | L40S 48 GB |
|---|---|---|
| correct | 203 / 231 (87.9 %) | 203 / 231 (87.9 %) |
| by tier: easy / standard / hard | 48/48 · 70/72 · 85/111 | 48/48 · 70/72 · 85/111 |
| answered and valid | 231 / 231 | 231 / 231 |
| mean input tokens per decision | 704 | 704 |
| latency p50 / p95, standard tier, serial | 0.066 s / 0.072 s | 0.050 s / 0.052 s |

The two cards gave the same answer on all 231 items; probabilities differ slightly between cards (by at most 0.165
on any option). The L40S run pinned the weights to revision `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`; the A6000
run fetched the repository's main branch without recording a revision. Both runs' per-item results, CLI manifests
and hashes are in `runs/` and `PROVENANCE.md`.

**Identity check for an evaluator:** a correct setup reproduces 203/231 with the tier split above, or 204/231 with
hard 86/111. One public item, `hard-sol-a-multi_hop-07`, sits on a near-tie between two options, and GPU batch order
can flip it: a third run with the pinned revision on an RTX A6000 (`runs/a6000-pinned/`) scored 204/231 and gave the
same answer as both runs above on the other 230 items.

## Run it

Hardware measured: one RTX A6000 (48 GB) and one L40S (48 GB). Other cards have not been measured. Docker image:
`vllm/vllm-openai:v0.30.0` (digest `sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90`), or
`pip install vllm==0.30.0`.

**1. Serve the model** (the documented context limit is **16384 tokens**; longer inputs get HTTP 422):

```bash
vllm serve google/gemma-4-12B-it --revision 707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7 \
  --served-model-name cygnet --host 127.0.0.1 --port 8890 \
  --max-model-len 16384 --gpu-memory-utilization 0.90
```

**2. Start the shim on the same machine** (standard library only):

```bash
SHIM_VLLM=http://127.0.0.1:8890/v1/chat/completions SHIM_MODEL=cygnet SHIM_PORT=8009 SHIM_TEMPERATURE=3.4 \
python3 shim/cygnet_shim.py
```

`SHIM_MODEL` must match `--served-model-name`. Before answering, the shim reads the server's context limit; if the
server does not list `SHIM_MODEL`, reports no limit, or has less than 4096, it answers 503, so a misconfigured
server stops a run instead of scoring it.

**3. Warm up** until this returns HTTP 200:

```bash
curl -s http://127.0.0.1:8009/v1/systemone -H 'Content-Type: application/json' -d '{"state": "warm-up",
  "questions": {"decision": {"type": "noul", "instructions": "Is this a warm-up?",
  "criteria": {"true": "yes", "false": "no"}}}}'
```

**4. Run JevBench's harness**:

```bash
python3 -m jevbench.cli run \
  --tasks <JEVBENCH>/datasets/public/easy.jsonl,<JEVBENCH>/datasets/public/original.jsonl,<JEVBENCH>/datasets/public/hard.jsonl \
  --adapter typesafe --endpoint http://127.0.0.1:8009 --key-env '' \
  --model cygnet --cost-basis self_hosted_gpu --reserve-usd 0 \
  --results <OUT>/results.jsonl --raw-dir <OUT>/raw --ledger <OUT>/ledger.jsonl --manifest <OUT>/manifest.json \
  --run-label cygnet --delay-s 0
```

**Status codes.** An input the shim cannot take (over the context limit, more than 26 options, an unknown question
type) gets HTTP 422, which the runner scores as one wrong answer. vLLM's 401, 403 and 429 pass through; any other
vLLM failure is a 502, which counts toward the runner's three-consecutive-failures stop.

**Tests** (no GPU): `python3 shim/test_shim.py`.

## How the readout works

For each decision the shim sends one chat request with the state, the instructions and the options lettered A, B,
C…, asking for one letter. vLLM masks the answer position to the option letters (`structured_outputs.choice`) and
returns the top 20 log-probabilities. The shim sums the probability of every token that is an option letter,
renormalises over the options, and maps the letters back to the benchmark's labels.

- **Duplicate letter tokens.** Gemma-4's vocabulary has more than one token that decodes to the same letter. The
  shim keeps every (token, log-probability) pair and sums per letter; keeping them in a dictionary keyed by text
  would drop mass and flatten the distribution. `shim/test_shim.py` covers this case.
- **Calibration temperature.** Probabilities are raised to `1/T` and renormalised. `T = 3.4` was fitted by
  negative log-likelihood on 241 items we generated ourselves; **JevBench's public items were never used for the
  fit**, only to measure. Fitting the same way on 121 of those items (T = 3.1) and testing on the other 120 lowered
  expected calibration error from 0.140 to 0.101 on the held-out half. `SHIM_TEMPERATURE=1.0` turns it off. `calibration/fit.py`
  reproduces the fit from the packaged items and reads.
- **Structured state** (a JSON object instead of text) is rendered with `json.dumps(..., indent=1)`, which is how
  every figure above was measured. JevBench's reference `openai_compat` adapter renders compactly.

## Disclosures

- The options are presented as letters in the benchmark's own label order, with every label's criteria text
  verbatim. The letter presentation is ours.
- Price basis for an evaluator: this is Google's `gemma-4-12B-it` at bf16, one output token per decision, 704 mean
  input tokens on the public set.
- No network calls other than the local vLLM server; no rules keyed to benchmark items' wording, ids or answers.

## Licence and credits

`shim/` and the documentation: MIT (`LICENSE`). Weights: Google DeepMind's Gemma-4-12B-it, Apache-2.0, subject to
Google's Gemma Prohibited Use Policy — see `NOTICE.md`. The one-token readout approach is
[NInfer](https://github.com/igorls/ninfer)'s — see `CREDITS.md`.
