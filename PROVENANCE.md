# Provenance

The README's measured table comes from `runs/l40s-pinned/` and `runs/a6000/`, and its identity check also cites
`runs/a6000-pinned/`; the calibration figures come from
`calibration/fit.py`. Both used JevBench's CLI at commit `2fa63fa`
(`--adapter typesafe`), public dataset hash `dc3995d8ae1e2fc8e81ce38431add509eb8bb39b85aadfd0c7c32079382dde51` (as recorded by the CLI).

| | `runs/l40s-pinned/` | `runs/a6000/` |
|---|---|---|
| GPU | NVIDIA L40S 48 GB (`env.txt`) | NVIDIA RTX A6000 48 GB (`env.txt`) |
| serving | vLLM 0.30.0, image `vllm/vllm-openai` (tag `latest` = `v0.30.0`, digest `sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90`) | same image |
| weights | `google/gemma-4-12B-it` `--revision 707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7` | `google/gemma-4-12B-it`, main branch, revision not recorded |
| shim | `shim/cygnet_shim.py`, sha256 `34816429ed14e0aaecb82b50f0a40e7b33055170f0698dbdc6cb5dedbb2b1ef3` | an earlier copy, sha256 `485e65b870027bb6a42bfe679d3b10405b9aaf81fbdf3908a472d8bd06eecb86`, whose runtime behaviour differs from `cygnet_shim.py` only in the default `SHIM_MODEL` and one alias removed from `GET /v1/models` (neither is used by the runner); the files also differ in docstrings and comments; same `SHIM_TEMPERATURE=3.4` |
| `--max-model-len` | 16384 | 16384 |
| results | `results.jsonl` sha256 `22e4280a47e938624ee4232fce74282e2949837cd4fd18873c3758a311444121` | `results.jsonl` sha256 `ec9de6c35d861531cca74473c7aafee1bf72cf320837ca3f8cec75d08a39e539` |
| CLI manifest | `manifest.json` (finished 2026-09-24T08:33:41.902430+00:00) | `manifest.json` (finished 2026-09-24T07:49:37.588362+00:00) |

`runs/a6000-pinned/`: NVIDIA RTX A6000 48 GB; vLLM 0.30.0, image tag `vllm/vllm-openai:v0.30.0` (digest not recorded);
`google/gemma-4-12B-it` `--revision 707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`; `shim/cygnet_shim.py` sha256
`34816429ed14e0aaecb82b50f0a40e7b33055170f0698dbdc6cb5dedbb2b1ef3`, `SHIM_TEMPERATURE=3.4`; `--max-model-len 16384`;
`results.jsonl` sha256 `07cf4dd6d3dd984296d2804a2a536045ad50d1afee7e37b067d71fa756c282cc`; CLI manifest finished
2026-09-24T23:15:31.027867+00:00. It scored 204/231 (easy 48/48, standard 70/72, hard 86/111) and gives the same
answer as both runs above on 230 items. On `hard-sol-a-multi_hop-07`, `admit` and `deny_prerequisite` tie at 0.4298
and JevBench's argmax returns `admit` (correct); the other two runs return `deny_prerequisite`.

Calibration: `calibration/fit.py` reproduces `T = 3.4` from `calibration/items.jsonl` (sha256
`6d114d0643ebab2ce52c1b91ace905bcc6f18acb115b607917fd327f1fc6bf3e`) and `calibration/letters-reads.jsonl` (sha256
`0a5adca43cee698d29939770f43b1d70af7e387e9cf917d2d681bc94769f734e`). None of the 241 items shares a state with a JevBench public item.
