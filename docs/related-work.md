# 相关工作与上游问题 / Related work

> **English abstract**: The blockers we hit are upstream, not local misconfiguration. Links to the
> vLLM / dflash / SGLang issues we used, plus other public DGX Spark deployments of the same model.
> All links are external and were checked on 2026-11-14.

## 1. 上游 issue（我们踩到或对照过的）

| 来源 | 内容 | 与本方案的关系 |
|---|---|---|
| [vllm#66387](https://github.com/vllm-project/vllm/issues/66387) | DFlash 在 TP>1 上接受率崩（AL 3.12 → 1.10） | 我们 TP2 实测逐位接受率 0.69/0.45/0.26…，是 TP1 级，没有踩到这个 bug |
| [vllm#55960](https://github.com/vllm-project/vllm/pull/55960) / [#58250](https://github.com/vllm-project/vllm/pull/58250) | 融合 DFlash2 grouped conv 合入 / revert 仍 open | 镜像含 #55960，影响量级约 4% |
| [z-lab/dflash#160](https://github.com/z-lab/dflash/issues/160) | 量化 target 与 drafter 协同崩（w4a16 6.25%，w4a32 恢复 75%） | 我们是 FP4 target + bf16 drafter，同类风险 |
| [z-lab/dflash#172](https://github.com/z-lab/dflash/issues/172) | hybrid 递归目标上 dflash 不无损：Cache.crop 把被拒草稿留在 recurrent state | GLM-5.3 有 KDA 线性注意力层，这很可能是跨引擎不一致的机制之一 |
| [z-lab/dflash#170](https://github.com/z-lab/dflash/issues/170) | 同一 drafter，llama.cpp AL ~3 vs vLLM ~5.5 | 说明运行时实现差异可达 2× |
| [z-lab/dflash#180](https://github.com/z-lab/dflash/issues/180) | 短验证块下 DFlash2 卷积 bug | 与「块尾接受率衰减」同型 |
| [sglang#40286](https://github.com/sgl-project/sglang/issues/40286) | SM121 上 GLM-5.3 没有可用 DSA attention 后端 | 这就是为什么 DGX Spark 上只能走 vLLM B12X |
| [sglang#37237](https://github.com/sgl-project/sglang/pull/37237) | SGLang HYBRID 投机（NGRAM + neural 路由） | 本 fork 的 SpeculativeMethod 是单值，不支持混用 |

## 2. 同类公开部署

- [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker)：DGX Spark 的 vLLM 容器与
  recipe runner，本仓库的配方就是给它用的。
- [NVIDIA dgx-spark-playbooks](https://github.com/NVIDIA/dgx-spark-playbooks)：官方两机互联与
  vLLM cluster 教程。
- 社区里还有多机（4 Spark）、1M 上下文的 GLM-5.3 NVFP4 部署与 DFlash2 报告
  （例如 `tonyd2wild/GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark`、`sfxnz/GLM-5.3-Flash-NVFP4-vLLM-2x-DGX-Spark`）。
  它们的基线（MTP-4 单流 21.8 tok/s）与本方案不同，**倍数不可直接横比**。

## 3. 我们的结论与它们的差异

- 社区「code 2.2×」多数建立在 4 台 Spark + 更低 MTP 基线上；我们两机、MTP K=3 的 ring 基线是
  28.7 tok/s，起点不同。
- 两机的聚合上限由带宽决定（见 performance.md §6），不是配置没调对。
