# Cygnet on Apple Silicon

This optional backend runs Cygnet's unchanged benchmark shim with Apple's Metal
GPU and Google's official Gemma 4 12B IT QAT Q4_0 checkpoint. The resident native
worker is adapted from [GemmaJev](https://github.com/dashidhy/GemmaJev).

It keeps Cygnet's system prompt, option descriptions and order, duplicate-letter
probability aggregation, top-20 token readout, and calibration temperature. The
replacement is the local inference server; `shim/cygnet_shim.py` stays unchanged.
Model weights are not fine-tuned for this backend.

## Setup

Requirements: macOS on Apple Silicon, 16 GB unified memory, Python 3.11 or newer,
CMake 3.20 or newer, and Xcode Command Line Tools. Run these commands from this
repository's root:

```bash
python3 metal/download_model.py
bash metal/build.sh
```

The model download is approximately 6.50 GiB. Its revision, byte size and SHA-256
are pinned in `download_model.py`; interrupted downloads can resume. The build
downloads and verifies a pinned llama.cpp archive and uses two build threads by
default. Weights go into `.models/`, while runtime source, objects and binaries
go into `.runtime/`. Both directories are Git-ignored.

If you already have the exact checkpoint, pass its path to the server instead
of downloading another copy. `CYGNET_BUILD_JOBS` controls compilation parallelism.
`CYGNET_LLAMA_SOURCE` optionally points to the pinned source archive or Git
checkout; the build never modifies that supplied source.

## Run

Start the Metal backend in one terminal:

```bash
python3 metal/server.py \
  --model .models/12b/gemma-4-12b-it-qat-q4_0.gguf \
  --ctx-size 4096
```

It loads the model before reporting readiness. Then start Cygnet's existing
benchmark shim in another terminal:

```bash
SHIM_VLLM=http://127.0.0.1:8890/v1/chat/completions \
SHIM_TEMPERATURE=3.4 \
python3 shim/cygnet_shim.py
```

Requests go to `http://127.0.0.1:8009/v1/systemone`, as in the GPU instructions.
The same backend also accepts requests from `shim/decision_server.py`; its
multi-question requests queue for the single resident Metal worker. It does not
provide GPU-style concurrent execution. Stop both servers with Ctrl-C when done.

## Readout and scope

- The backend uses the official Gemma 4 12B system/user chat template with
  thinking disabled. All token IDs decoding exactly to a permitted uppercase
  letter participate in the constrained distribution, including duplicate
  letter tokens. The top 20 token records are returned without deduplicating
  their text; Cygnet's existing shim performs aggregation and calibration.
- It directly reads prefill logits. No output token is generated or appended to
  KV state, so actual usage reports `completion_tokens: 0`. The compatibility
  message is the highest-probability token's text, not an additional decoding
  pass. Google model softcapping and media-token suppression remain intact.
- Identical token prefixes can reuse KV state. Requests are serialized, and a
  workspace lock prevents loading a second worker through this adapter. This
  server only accepts the small `/v1/models` and `/v1/chat/completions` profile
  used by Cygnet, and listens on loopback only.
- Context defaults to 4,096 tokens. Inputs are never truncated. Larger context
  sizes, up to 16,384, are configurable but need more memory. This backend is
  text-only; GemmaJev's separate image/audio interface is not included here.

## Measured public-subset result

One complete run on an Apple M2 Pro with 16 GB unified memory, using QAT Q4_0,
the unchanged benchmark shim and its existing T=3.4 calibration:

| Metric | M2 Pro, QAT Q4_0 | Published L40S, BF16 | Published A6000, BF16 |
| --- | ---: | ---: | ---: |
| Correct / attempted | 198 / 231 | 203 / 231 | 204 / 231 |
| Accuracy | 85.71% | 87.88% | 88.31% |
| Easy / standard / hard correct | 48 / 69 / 81 | 48 / 70 / 85 | 48 / 70 / 86 |
| Valid responses | 231 / 231 | 231 / 231 | 231 / 231 |
| Brier mean (lower is better) | 0.18791 | 0.17476 | 0.17057 |
| Top-label ECE, 10 bins (lower is better) | 0.06369 | 0.02163 | 0.02907 |
| Score expected-level MAE (lower is better) | 0.16818 | 0.20779 | 0.20252 |

The Metal run agrees with the GPU runs on 222/231 and 223/231 decisions,
respectively. It retains most of the reference decision accuracy, with a
2.16–2.60 percentage-point deficit and higher calibration error. This does not
establish statistical equivalence. No prompt, temperature or model setting was
tuned against these benchmark answers.

The standard tier's raw HTTP p50/p95 was 0.474/0.618 seconds. Across all tasks it
was 0.619/13.972 seconds; the longest request took 20.830 seconds. Loading is
outside these timings. This was a normal desktop session, with OS swap in use,
not an isolated latency or memory benchmark. The hardware, precision, runtime,
and between-request pacing differ from the published GPU runs.

All 231 prompts fit the configured 4,096-token context without truncation; the
largest contained 3,909 tokens. Native counts match Google's pinned HF tokenizer
and template for every item. The published GPU records report one additional
prompt token per item; their prompt token IDs are unavailable, so we do not claim
token-for-token parity with their vLLM run. The pinned GGUF vocabulary and BPE
merges match the official HF tokenizer.

The [run archive](../runs/metal-m2-pro-qat-q4_0/) contains original per-item
probabilities and timing records, the CLI manifest, and model/runtime provenance.
It contains no model files, task passages, or private/sealed data. With the pinned
JevBench checkout below, summarize the saved records using its existing CLI,
without loading a model:

```bash
PYTHONPATH=.runtime/jevbench python3 -m jevbench.cli summarize \
  --tasks .runtime/jevbench/datasets/public/easy.jsonl,.runtime/jevbench/datasets/public/original.jsonl,.runtime/jevbench/datasets/public/hard.jsonl \
  --results runs/metal-m2-pro-qat-q4_0/results.jsonl
```

## Reproduce the public JevBench run

Use the same frozen CLI and 231 public tasks as Cygnet's pinned GPU runs:

```bash
git clone https://github.com/fstandhartinger/jevbench.git .runtime/jevbench
git -C .runtime/jevbench checkout 2fa63fa3226cb369795525ed011800f57dcbd894
mkdir -p .runtime/jevbench-metal

PYTHONPATH=.runtime/jevbench python3 -m jevbench.cli run \
  --tasks .runtime/jevbench/datasets/public/easy.jsonl,.runtime/jevbench/datasets/public/original.jsonl,.runtime/jevbench/datasets/public/hard.jsonl \
  --adapter typesafe --endpoint http://127.0.0.1:8009 --key-env '' \
  --model cygnet --cost-basis self_hosted_metal --reserve-usd 0 \
  --results .runtime/jevbench-metal/results.jsonl \
  --raw-dir .runtime/jevbench-metal/raw \
  --ledger .runtime/jevbench-metal/ledger.jsonl \
  --manifest .runtime/jevbench-metal/manifest.json \
  --run-label cygnet-metal --delay-s 0.5

PYTHONPATH=.runtime/jevbench python3 -m jevbench.cli summarize \
  --tasks .runtime/jevbench/datasets/public/easy.jsonl,.runtime/jevbench/datasets/public/original.jsonl,.runtime/jevbench/datasets/public/hard.jsonl \
  --results .runtime/jevbench-metal/results.jsonl
```

The half-second pause between requests reduces sustained load; it is outside
the runner's per-request latency measurement. The CLI manifest records the dataset
hash and run settings. To summarize a GPU reference, use its existing
`runs/l40s-pinned/results.jsonl` or `runs/a6000-pinned/results.jsonl` with the same
summary command. This is a public-subset
portability check, not a full or sealed JevBench score. Local latency must not
be interpreted as a hardware-normalized comparison with the GPU runs.

Tests without model loading:

```bash
python3 -m unittest discover -s metal -p 'test_*.py'
.runtime/build/bin/cygnet-metal-worker --self-test
python3 shim/test_shim.py
python3 shim/test_decision_server.py
```

The native worker retains GemmaJev's MIT notice; see
[LICENSE.gemmajev](LICENSE.gemmajev). The model and llama.cpp retain their
upstream licenses.
