# Credits

- **Weights: Google DeepMind's `google/gemma-4-12B-it`** (Apache-2.0, with Google's Gemma Prohibited Use Policy).
  They are fetched from Hugging Face at run time and are not redistributed here.
- **The one-token readout approach: [NInfer](https://github.com/igorls/ninfer)** (Apache-2.0), whose
  [JevBench entry](https://github.com/fstandhartinger/jevbench/issues/12) reads a model's own probabilities over the
  option letters from a single forward pass instead of asking it to write the answer. Cygnet does the same over stock
  vLLM. The idea is theirs; the code in `shim/` is ours.
- **Serving: [vLLM](https://github.com/vllm-project/vllm)** 0.30.0, unmodified.
- **Optional Apple Metal backend: [GemmaJev](https://github.com/dashidhy/GemmaJev)** (MIT),
  by Hongyuan Du. The resident llama.cpp worker and prefix-cache approach in `metal/`
  are adapted from GemmaJev, with its license notice retained. This backend uses
  Google's official QAT Q4_0 checkpoint and leaves Cygnet's shim and calibration unchanged.
- **Benchmark: [JevBench](https://github.com/fstandhartinger/jevbench)**, whose CLI and scoring produced every public-set
  figure in the README.

- **The decision server was prompted by Fabián Gonzalo Artur de la Villarmois
  ([@notf0und](https://github.com/notf0und))**, whose [issue #1](https://github.com/blockbrain-ai/cygnet-recipe/issues/1)
  and [pull request #2](https://github.com/blockbrain-ai/cygnet-recipe/pull/2) set out what applications need from it:
  answering every named question, `confidence`, optional noul criteria, the score and usage fields, and composing
  one-token passes over groups of options. The llama.cpp thinking note in the README comes from the same issue.
  `shim/decision_server.py` is our own implementation of those ideas; it contains no code from the pull request. Thank
  you, Fabián.

What is ours: the shim (`shim/cygnet_shim.py`, MIT), the decision server (`shim/decision_server.py`, MIT), their
tests, the calibration temperature and its fitting data (our own generated items), and this packaging.
