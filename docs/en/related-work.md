# Related work

## Upstream issues

| Source | What it says | Relation to this work |
|---|---|---|
| [vllm#66387](https://github.com/vllm-project/vllm/issues/66387) | DFlash acceptance collapses at TP>1 (3.12 → 1.10) | We run TP2 and see TP1-class acceptance (0.69/0.45/0.26…), so we do not hit it |
| [vllm#55960](https://github.com/vllm-project/vllm/pull/55960) / [#58250](https://github.com/vllm-project/vllm/pull/58250) | fused DFlash2 grouped conv merged / revert still open | our image contains #55960; effect ~4% |
| [z-lab/dflash#160](https://github.com/z-lab/dflash/issues/160) | quantized target + drafter interaction (w4a16 6.25%, w4a32 75%) | we are FP4 target + bf16 drafter: same class of risk |
| [z-lab/dflash#172](https://github.com/z-lab/dflash/issues/172) | hybrid recurrent targets: rejected drafts stay in recurrent state | GLM-5.3 has KDA layers; likely one cause of cross-engine divergence |
| [z-lab/dflash#170](https://github.com/z-lab/dflash/issues/170) | same drafter, llama.cpp AL ~3 vs vLLM ~5.5 | runtime implementation can move AL by 2× |
| [z-lab/dflash#180](https://github.com/z-lab/dflash/issues/180) | DFlash2 conv bug with short verification blocks | same shape as our block-tail acceptance decay |
| [sglang#40286](https://github.com/sgl-project/sglang/issues/40286) | no usable DSA attention backend for GLM-5.3 on SM121 | why DGX Spark has to use the vLLM B12X path |
| [sglang#37237](https://github.com/sgl-project/sglang/pull/37237) | SGLang HYBRID speculation (NGRAM + neural routing) | this fork cannot mix methods |

## Other public deployments

- [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) — the recipe runner and
  container build this repo plugs into.
- [NVIDIA dgx-spark-playbooks](https://github.com/NVIDIA/dgx-spark-playbooks) — official
  two-Spark interconnect and vLLM cluster guide.
- Community 4-Spark / 1M-context GLM-5.3 NVFP4 reports (for example
  `tonyd2wild/GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark` and
  `sfxnz/GLM-5.3-Flash-NVFP4-vLLM-2x-DGX-Spark`). Their MTP-4 baseline is 21.8 tok/s single
  stream, so their multipliers are **not** directly comparable with ours.

## Why our numbers differ from community "2.2×" claims

Most of those are four Sparks against a lower MTP baseline. On two Sparks and MTP K=3 our ring
baseline is 28.7 tok/s, and aggregate throughput is bounded by bandwidth (see
[performance.md](performance.md) §6).
