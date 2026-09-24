# Credits

- **Weights: Google DeepMind's `google/gemma-4-12B-it`** (Apache-2.0, with Google's Gemma Prohibited Use Policy).
  They are fetched from Hugging Face at run time and are not redistributed here.
- **The one-token readout approach: [NInfer](https://github.com/igorls/ninfer)** (Apache-2.0), whose
  [JevBench entry](https://github.com/fstandhartinger/jevbench/issues/12) reads a model's own probabilities over the
  option letters from a single forward pass instead of asking it to write the answer. Cygnet does the same over stock
  vLLM. The idea is theirs; the code in `shim/` is ours.
- **Serving: [vLLM](https://github.com/vllm-project/vllm)** 0.30.0, unmodified.
- **Benchmark: [JevBench](https://github.com/fstandhartinger/jevbench)**, whose CLI and scoring produced every public-set
  figure in the README.

What is ours: the shim (`shim/cygnet_shim.py`, MIT), its tests, the calibration temperature and its fitting data
(our own generated items), and this packaging.
