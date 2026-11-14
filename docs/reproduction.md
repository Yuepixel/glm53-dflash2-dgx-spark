# 复现步骤 / Reproduction

> **English abstract**: Two DGX Sparks, passwordless SSH, the eugr runner, ~166 GB of weights on
> each node, and a ~2.3 GB drafter that must exist at the same path on both nodes. Build the B12X
> image, download both models, drop the recipe in, run `serve.sh`. First start compiles kernels and
> runs a one-off FlashInfer JIT (~14 min per node); warm restarts take 6–8 min.

## 0. 你需要准备什么

| 项 | 要求 |
|---|---|
| 节点 | 2× DGX Spark（GB10，`sm_121a`），每台 121 GiB 统一内存 |
| 互联 | 两台直连的 ConnectX-7（RoCE / 200 GbE），本方案按 TP2 跨机跑 |
| SSH | head → worker 免密（`BatchMode=yes` 能直接跑通） |
| 磁盘 | 每台 ≥ 250 GB 空闲（镜像约 28 GB + 权重 166 GB + kernel 缓存） |
| 软件 | 能跑 `docker` 的 DGX OS；Python 3.10+（runner 用） |
| 网络 | 能访问 Hugging Face（或预先准备好本地缓存） |

## 1. 集群发现与镜像

```bash
git clone https://github.com/eugr/spark-vllm-docker.git
cd spark-vllm-docker
./run-recipe.sh --discover        # 探测节点、写 .env 的 CLUSTER_NODES
./run-recipe.sh recipes/glm-5.3-flash-dflash2.yaml --setup --build-only
```

镜像走 `--exp-b12x` profile，最终 tag 为 `vllm-node-b12x`。本仓库验证的版本是 2026-11-02 构建
（digest `sha256:19ab778dd57f…`）；上游 `latest` 会漂移，要字节级复现就固定这个 digest。

## 2. 权重与 drafter

```bash
# 主权重（runner 会下载并分发到所有节点）
./run-recipe.sh recipes/glm-5.3-flash-dflash2.yaml --setup --download-only

# drafter：独立 HF 仓库，必须两台机器同一路径
huggingface-cli download incoai/GLM-5.3-Flash-DFlash2 \
  --local-dir "$HOME/models/GLM-5.3-Flash-DFlash2"

# 同步到 worker（路径必须一致，容器里挂的是这个路径）
rsync -a "$HOME/models/GLM-5.3-Flash-DFlash2" worker:"$HOME/models/"
```

> 为什么强调「两台同路径」：`-v $HOME/models/GLM-5.3-Flash-DFlash2:/models/dflash2-draft:ro`
> 会被原样传给两个 rank 的容器。worker 上没有这个目录时，容器会静默挂出一个空目录，
> 报错是 `/models/dflash2-draft 不是有效的模型目录`，看起来像下载坏了。

## 3. 配方落盘并起服

```bash
cp /path/to/this/repo/recipes/glm-5.3-flash-dflash2.yaml recipes/
REPO_DIR="$PWD" /path/to/this/repo/scripts/serve.sh
```

`serve.sh` 做三件事：切到 runner 目录、带 `--daemon` 起服、把 drafter 挂载和三个必带 `-e` 加上。
不想用脚本就手敲 README 里那条完整命令。

## 4. 首次起服的时间线（别以为卡死了）

| 阶段 | 首次 | 缓存热 |
|---|---|---|
| 镜像构建 / 拉取 | 30–60 min | — |
| 权重下载 + 分发到 worker | 取决于带宽 | — |
| kernel 自动调优 + CUDA graph capture | 10–20 min | 复用 |
| FlashInfer `top_k` JIT（每台一次） | ~14 min/台 | 命中缓存 |
| 权重加载（82.87 GiB 分片） | 150–260 s | 同 |
| **总计** | **约 30–45 min** | **6–8 min** |

判断「在编译」还是「死锁」：

- 编译期：一台 GPU 96% 自旋，另一台 0%，日志在推进（`Compiling` / `cub_topk`）；不要杀。
- 死锁：两边都 0% 且日志 `No available shared memory broadcast block found in 60 seconds` 超过
  两个 60 s 窗口；这通常意味着对端 rank 已经崩了，去看对端 `docker logs`。

## 5. 起服成功的标志

日志里应依次出现：

```
Resolved architecture: DFlash2DraftModel
Draft model DFlash2Qwen3ForCausalLM
Overriding draft model max model len from 1048576 to 524288
Model loading took 82.87 GiB memory
Using split GLM-5.3 cache pages: target block size 5120
Keeping split GLM-5.3 cache groups with physical page sizes [2605056, 2872320]
Using RoCEnante ... all-reduce max=2097152 bytes
Application startup complete.
```

然后：

```bash
curl -s -o /dev/null -w '%{http_code}\n' localhost:8888/health     # 200
curl -s localhost:8888/v1/models | python3 -m json.tool
WORKER_SSH=worker /path/to/this/repo/scripts/health.sh
python3 scripts/probe-decode.py http://127.0.0.1:8888 py_api 400    # ~40 tok/s
```

## 6. 在 DFlash2 与 MTP K=3 之间切换

两个配方只差 `--speculative-config` 与 `num_speculative_tokens`：

```bash
# 切回 MTP K=3（通用 code / 散文更稳）
WORKER_SSH=worker scripts/stop.sh
RECIPE="$HOME/src/spark-vllm-docker/recipes/glm-5.3-flash-mtp-k3.yaml" scripts/serve.sh

# 切回 DFlash2（serve.sh 的默认配方）
WORKER_SSH=worker scripts/stop.sh && scripts/serve.sh
```

更省事的做法：两个配方都以 `glm-5.3-flash.yaml` 的名字轮换（一个存成
`glm-5.3-flash.yaml.mtp-k3`），切的时候 `cp` 回来、重启即可；这也和 `RECIPE=...` 等价。

## 7. 停服与重启

```bash
WORKER_SSH=worker scripts/stop.sh
# 重启 = stop + serve；不会自动拉起，没有看门狗（本仓库刻意不带自愈逻辑）
```

## 8. 故障速查

| 现象 | 大概率原因 | 动作 |
|---|---|---|
| `No valid attention backend` | drafter KV dtype 没写 `bfloat16` | 看 pitfalls §1 |
| `GLM MLA parent-page stride` | 缺 split-page 两个 `-e` | pitfalls §2 |
| `device-side assert` / `RoCE collective timed out` | 缺 `VLLM_ROCE_ALLGATHER_MAX_SIZE=64` | pitfalls §3 |
| 起服 40 min 还没好，但日志在动 | 首次 JIT / 编译 | 等；把 `/root/.cache/flashinfer` 挂出来 |
| `/health` 200 但吞吐为 0 | worker rank 死了 | `scripts/health.sh` 查双机容器 |
| OOM | KV 预算 / 别的进程占卡 | 确认 8 GiB/card KV 没被改大 |
| 只有 14 tok/s | `VLLM_B12X_MOE_FP4_FORCE_A16=0` 了 | 改回 1 |

## 9. 日志与指标

- 容器日志：`docker logs -f vllm_node`
- 接受长度：日志里的 `SpecDecoding metrics`，或 `/metrics` 计数器的差分（方法见 benchmark-method）
- 启动耗时统计：`docker inspect -f '{{.State.StartedAt}}' vllm_node`
