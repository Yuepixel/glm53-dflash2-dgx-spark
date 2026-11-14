# GLM-5.3-Flash-FP4 + vLLM B12X + DFlash2 · 2× DGX Spark

在 **2 台 DGX Spark（GB10 / sm_121a）** 上用 **vLLM B12X** 栈跑 **GLM-5.3-Flash-FP4**，并挂上
**DFlash2 投机解码器（K=7）** 的完整可用配方、实测数据与踩坑记录。

- **vLLM 源码零改动**：一份 recipe + 三个运行期开关，全部是公开参数。
- **不吹 2×**：高可预测内容 1.5–1.7×，通用 code 约 1.05×（9 条真实提示词），散文在高并发下反而更慢；
  两机 2× 在物理上不可达（带宽账见 [docs/performance.md](docs/performance.md)）。
- 所有数字都是 2026-11-14 在真机上测的；压测脚本在 [scripts/](scripts)，code 负载的表格你可以原样复现。

## 这套东西解决什么问题

在 DGX Spark 上想给 GLM-5.3 开 DFlash2，会连续撞三堵墙，而且三堵墙的报错都不指向真因：

1. drafter 的 KV 继承了 target 的 `fp8_ds_mla` → attention 后端选择直接失败；
2. GLM 的 C4 indexer 要求 MLA 页面 stride 是 8448 B 的整数倍 → 多一个 drafter KV 组就破约；
3. b12x 的 RoCE one-shot all-gather 在候选 logits 的 128 B 张量上越界 → device-side assert + 对端超时。

三个开关同时到位就能起服。本仓库给的就是这三条 + 一份可直接用的 recipe，以及量化到「什么时候值得开」。

## 硬件 / 软件基线

| 项 | 值 |
|---|---|
| 硬件 | 2× NVIDIA DGX Spark（GB10，`sm_121a`），每台 121 GiB 统一内存，直连 ConnectX-7 200 GbE（RoCE） |
| 引擎镜像 | `vllm-node-b12x`（B12X profile，`--exp-b12x` 构建）；实测版本为 2026-11-02，digest `sha256:19ab778dd57f…` |
| vLLM | `0.1.dev21830+g8244d1c44.d20261102`（B12X fork） |
| 目标模型 | [`nvidia/GLM-5.3-Flash-FP4-Local`](https://huggingface.co/nvidia/GLM-5.3-Flash-FP4-Local)（约 166 GB，ModelOpt NVFP4） |
| Drafter | [`incoai/GLM-5.3-Flash-DFlash2`](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2)（约 2.3 GB，block size 8 → K 必须取 7） |
| 并行 | 两机 TP2，decode-context-parallel = 1 |
| 上下文 | 524,288 tokens；`max_num_seqs 8`、`max_num_batched_tokens 4096`、每卡 8 GiB KV |
| KV dtype | target：`fp8`；drafter（独立 KV 组）：`bfloat16` |

## 快速开始

前提：两台 GB10 已直连，head 到 worker 免密 SSH。

```bash
# 1. 拉 runner + 集群发现（只需一次）
git clone https://github.com/eugr/spark-vllm-docker.git
cd spark-vllm-docker
./run-recipe.sh --discover           # 结果写入 .env 的 CLUSTER_NODES

# 2. 构建 B12X 镜像
./run-recipe.sh recipes/glm-5.3-flash-dflash2.yaml --setup --build-only

# 3. 下载权重（约 166 GB）
./run-recipe.sh recipes/glm-5.3-flash-dflash2.yaml --setup --download-only

# 4. 下载 drafter（独立 HF 仓库，约 2.3 GB）
huggingface-cli download incoai/GLM-5.3-Flash-DFlash2 \
  --local-dir "$HOME/models/GLM-5.3-Flash-DFlash2"

# 5. 放入配方并起服
cp /path/to/this/repo/recipes/glm-5.3-flash-dflash2.yaml recipes/
REPO_DIR="$PWD" /path/to/this/repo/scripts/serve.sh

# 6. 验证
curl -s localhost:8888/v1/models
/path/to/this/repo/scripts/health.sh
```

首次起服很慢：kernel 自动调优 + FlashInfer 的一次性 JIT 编译（**每台约 14 分钟**）。
缓存热了以后重启约 6–8 分钟。正常/异常日志的判别见 [docs/reproduction.md](docs/reproduction.md)。

## 三个必须同时到位的开关

| # | 开关 | 缺了会怎样 |
|---|---|---|
| 1 | `--speculative-config` 里写 `"kv_cache_dtype":"bfloat16"`（字面量必须是 `bfloat16`，写 `bf16` 会被 pydantic Literal 拒绝） | `No valid attention backend found … kv_cache_dtype=fp8_ds_mla, use_mla=False …`：drafter 的 attention 不是 MLA，却继承了 target 的 `fp8_ds_mla` |
| 2 | `-e VLLM_GLM53_SPLIT_TARGET_BLOCK_SIZE=5120 -e VLLM_GLM53_SPLIT_MAMBA_BLOCK_SIZE=5120` | `GLM MLA parent-page stride must be an exact number of C4 pages`：C4 indexer 要求页面 stride 是 8448 B 的整数倍，加一个 drafter KV 组就破了统一分页 |
| 3 | `-e VLLM_ROCE_ALLGATHER_MAX_SIZE=64` | 候选 logits 的 all-gather 触发 `vectorized_gather index out of bounds` → `device-side assert`，对端 `RoCE collective timed out`。限到 64 B 让这一个 all-gather 走 NCCL，all-reduce 仍走 RoCE |

另外两个「本来就在镜像里、别关」的：`VLLM_B12X_MOE_FP4_FORCE_A16=1`（关掉实测掉到 ~14 tok/s）、
`VLLM_USE_V2_MODEL_RUNNER=1`（DFlash2 依赖 V2 runner）。

## 实测 A/B（可复现口径）

code 负载，聚合 tok/s，max_tokens 400，temperature 0，3 次取中位；`mtp K=3` 是同镜像内置的 MTP 头。

| 提示词（见 `scripts/bench-code-decode.py`） | 引擎 | c1 | c2 | c4 | c8 | 倍数 |
|---|---|---|---|---|---|---|
| count-to-60 | mtp K=3 | 44.8 | 64.6 | 95.6 | 139.8 | — |
| | DFlash2 K=7 | **75.6** | **108.0** | **155.4** | **210.7** | **1.51–1.69×** |
| py_api | mtp K=3 | 45.0 | 59.7 | 86.1 | 128.5 | — |
| | DFlash2 K=7 | 48.1 | 57.7 | 99.3 | 125.1 | 0.97–1.15× |
| code_ts | mtp K=3 | 44.4 | 57.7 | 80.7 | 105.5 | — |
| | DFlash2 K=7 | 46.7 | 62.1 | 84.4 | 112.3 | 1.03–1.08× |
| ring buffer | mtp K=3 | 28.7 | 43.1 | 57.1 | 78.6 | — |
| | DFlash2 K=7 | 26.3 | 35.4 | 51.0 | 68.9 | 0.82–0.92× |

**接受长度视角**（决定一切的量）：

| 提示词 | AL（mtp K=3） | AL（DFlash2 K=7） | 每验证行产出（K=3 → K=7） |
|---|---|---|---|
| count-to-60 | 未测 | 7.1–7.5 | — → 0.89–0.93 |
| py_api | 3.3–3.5 | 4.7–5.0 | 0.88 → 0.58 |
| code_ts | 3.2–3.3 | 3.7–3.8 | 0.83 → 0.47 |
| ring buffer | 3.2 | 2.3–2.6 | 0.80 → 0.29 |

**首字延迟**：结构化负载 TTFT 63 ms → 25 ms；prefill 吞吐不变。

完整表（Structured / JSON / Code / Prose 的 c1–c8）与负面结论（MTP K=7、关投机、SGLang 在 SM121 跑不起来）
见 [docs/performance.md](docs/performance.md)。

## 什么时候开 DFlash2 才划算

decode 吞吐 ≈ 接受长度 AL ÷ 步时，而步时随「验证行数 K+1」增长。所以 DFlash2（K=7）要赢 MTP（K=3）：

```
AL_dflash / (7+1) > AL_mtp / (3+1)      ⇔      AL_dflash > 2 × AL_mtp
```

本机 MTP 基线的 AL 是 2.2–3.5，所以 DFlash2 要把 AL 推到 **4.4–7.0** 才有净收益。
落在「高度可预测」的内容上（表格、计数、严格格式化、样板代码）；写新算法/长散文不划算。

| 内容类型 | 建议 |
|---|---|
| 表格 / 计数 / 严格格式 | **DFlash2**（1.5–1.9×） |
| JSON、工具调用参数、样板代码 | **DFlash2**（1.1–1.3×） |
| 新算法、长散文 | **MTP K=3**（DFlash2 ≈ 0.8–1.0×） |
| agent 混合流量 | 要首字延迟就 DFlash2；自己的提示词自己测 |

## 坑（都是线上真撞过的）

| 现象 | 根因 | 修法 |
|---|---|---|
| `No valid attention backend found … kv_cache_dtype=fp8_ds_mla` | drafter KV 继承了 target 的 MLA dtype | `"kv_cache_dtype":"bfloat16"` |
| `GLM MLA parent-page stride must be an exact number of C4 pages` | C4 indexer 的 8448 B 页面 stride 被多出来的 KV 组打破 | split-page 两个环境变量（`5120`/`5120`） |
| 起服时 `device-side assert` + 对端 `RoCE collective timed out` | b12x one-shot RoCE all-gather 在 128 B 张量上越界 | `VLLM_ROCE_ALLGATHER_MAX_SIZE=64` |
| 一台 GPU 96%，另一台看起来闲置约 14 分钟 | FlashInfer `top_k` 首次使用在 JIT 编译 `cub_topk.cu` | 等；把 `/root/.cache/flashinfer` 挂出来保住缓存 |
| `bf16` 被 pydantic `Literal` 拒绝 | `kv_cache_dtype` 只认 enum 名，不认 torch dtype 名 | 写 `bfloat16` |
| OOM / 首次起服极慢 | 166 GB 权重 + 每卡 8 GiB KV + kernel 缓存 | 别动 KV 预算，见文档 |

日志片段与源码锚点：docs/pitfalls.md。

## 目录

```
recipes/   可直接放进 eugr/spark-vllm-docker 的配方（DFlash2 K=7 与 MTP K=3 基线）
scripts/   serve / stop / health 包装、压测脚本、接受长度探针
docs/      复现步骤、坑、性能、测量方法、相关工作
bench/     测量表格（CSV）与轮次记录
```

## 复现边界（先说清楚）

- code 那几张表可以用仓库里的脚本原样复现；「Structured / JSON / Prose」面板数据来自非公开提示词集，
  **无法**凭本仓库复现，只作为背景信息列出。
- temperature 0 在这套栈上**不保证逐字一致**：FP4 MoE 路由 + bf16 批形状会让基线自己跟自己对不上。
  要比就比语义和单元测试，别比逐字 diff。
- 你的接受长度主要由你的提示词决定，其次才是这份配置。

## 致谢

- [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker)：recipe runner、镜像构建、集群发现。
- NVIDIA：FP4-Local 权重与 DGX Spark 平台；`incoai`：DFlash2 drafter。
- [vLLM](https://github.com/vllm-project/vllm)（Apache-2.0）。

## 许可

本仓库内容采用 MIT（见 [LICENSE](LICENSE)）。模型权重、容器镜像与上游工具各自遵循其原许可；
本仓库不重分发任何第三方二进制或权重。
