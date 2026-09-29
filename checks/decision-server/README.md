# Decision server on the real model

`shim/decision_server.py` (commit `dc6d473`) with frozen `google/gemma-4-12B-it` at revision `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`,
served as the recipe README says (`vllm/vllm-openai:v0.30.0` by digest, `--max-model-len 16384`,
`--gpu-memory-utilization 0.90`) on one NVIDIA H100 NVL (driver 580.159.04; vLLM 0.30.0, torch 2.13.0+cu130,
transformers 5.17.0), 2026-09-29. The pass lines were set before the run. Every figure below is in `analyse-output.txt`,
which `python3 analyse.py` reproduces from `out/`.

## A. Same results as the benchmark shim (JevBench public set, 231 items)

JevBench's CLI at `2fa63fa`, `--adapter typesafe`, the recipe README's command, serially on one server: A1 through
`cygnet_shim.py` (T 3.4), A2 through `decision_server.py` (T 3.4), A3 through the shim again.

| | correct | predictions differing from A3 | largest probability difference from A3 |
|---|---|---|---|
| A1 shim (first read of each prompt) | 203 / 231 | 0 | 0.0176 |
| A2 decision server | 203 / 231 | 0 | 0 |
| A3 shim | 203 / 231 | — | — |

A2 and A3 both follow A1, whose prompts are then in vLLM's prefix cache, so A2 against A3 is the like-for-like
comparison. **Pass:** A2 predicts what A3 predicts on all 231 items, with a probability difference no larger than
A1's.

## B. Grouped reads past 20 options (public intent datasets)

Each item is one Choice: the user's message as state, "Which of these intents does the user's message express?", and
the intent names as options with null descriptions (the server shows each name). Items from `build_items.py`
(seed 20260929); read serially at T 1, with T 3.4 applied afterwards exactly as the server applies it (once, to the
composed distribution).

| set | options | passes per question | accuracy | ECE at T 1 | ECE at T 3.4 | latency p50 |
|---|---|---|---|---|---|---|
| BANKING77, 400 test messages, gold + 19 other intents, one pass | 20 | 1 | 86.50 % | 0.125 | 0.070 | 0.059 s |
| the same 400, grouped (`CYGNET_GROUP_SIZE=13`: 2 × 10, then 2) | 20 | 3 | 87.00 % | 0.118 | 0.056 | 0.142 s |
| BANKING77, 400 test messages, all intents | 77 | 5 | 73.50 % | 0.249 | 0.067 | 0.193 s |
| CLINC150, 400 in-scope test messages, all intents | 150 | 9 | 91.25 % | 0.074 | 0.171 | 0.267 s |

On the 400 paired 20-option items the grouped reads chose the same option as one pass on 373; 10 were right only in
one pass and 12 only grouped (exact McNemar p 0.832). **Pass:** grouped accuracy no more than 3 points below one pass
(it is 0.50 points above). ECE is expected calibration error over 10 equal-width bins of the top option's probability.

The calibration temperature (3.4) was fitted on single-pass reads of other items. On grouped reads it lowered ECE on 77
options and raised it on 150, where its top-option probabilities (mean 0.741) sit below the accuracy (91.25 %). It is
not established that the temperature carries over to grouped reads.

## C. Latency and throughput

20-option requests (one pass) through the server with its defaults, each run on BANKING77 test messages not used
elsewhere, so no prompt was cached:

| clients | `CYGNET_MAX_PARALLEL` | requests | requests / s | latency p50 | p95 | p99 |
|---|---|---|---|---|---|---|
| 1 | 8 | 200 | 16.3 | 0.062 s | 0.069 s | 0.084 s |
| 8 | 8 | 400 | 38.5 | 0.209 s | 0.255 s | 0.968 s |
| 32 | 8 | 400 | 33.5 | 0.623 s | 3.975 s | 4.300 s |
| 32 | 32 | 400 | 37.0 | 0.562 s | 3.339 s | 3.808 s |

77-option requests (5 passes), 8 clients: 8.3 requests/s, p50 0.933 s. No request failed in any run.

## Reproduce

Fetch `test.csv` from [PolyAI-LDN/task-specific-datasets](https://github.com/PolyAI-LDN/task-specific-datasets) and
`data_full.json` from [clinc/oos-eval](https://github.com/clinc/oos-eval) at the commits in `data/SOURCES.md` into
`data/` as `banking77-test.csv` and `clinc150-data_full.json`; `python3 build_items.py` then writes the items with the
sha256 values in `items-build.json`. With the model served and the servers started as above (T 1 for B, T 3.4 for A
and C), `python3 gpu_check.py read <items> <server url> <out.jsonl>` reads a set and
`python3 gpu_check.py load <items> <server url> <out.json> <clients>` runs a load.

Data: BANKING77 (Casanueva et al., 2020) under CC BY 4.0 and CLINC150 (Larson et al., 2019) under CC BY 3.0; the reads
in `out/` contain their intent labels. They were used only to measure.
