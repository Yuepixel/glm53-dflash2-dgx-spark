# 坑 / Pitfalls

> **English abstract**: Three independent blockers. (1) The drafter inherits the target's
> `fp8_ds_mla` KV dtype — set `kv_cache_dtype=bfloat16` in `--speculative-config`. (2) GLM's C4
> indexer requires an MLA page stride that is a multiple of 8448 B; a second KV group breaks it —
> enable split cache pages (5120/5120). (3) The B12X one-shot RoCE all-gather overflows on the
> candidate-logits all-gather — cap it with `VLLM_ROCE_ALLGATHER_MAX_SIZE=64`. Each blocker reports
> something that does not name its cause.

## 1. drafter 的 KV 继承了 target 的 `fp8_ds_mla`

**现象**

```
ValueError: No valid attention backend found for ...
  kv_cache_dtype=fp8_ds_mla, use_mla=False, has_sliding_window=True, use_non_causal=True
```

**根因**：DFlash2 drafter 的 attention 不是 MLA，但它作为新 KV 组继承了 target 的 MLA 专用
`kv_cache_dtype`，后端选择器找不到匹配实现。

**解法**：在 `--speculative-config` 里给 drafter 单独写 KV dtype：

```json
{"method":"dflash","model":"/models/dflash2-draft","num_speculative_tokens":7,"kv_cache_dtype":"bfloat16"}
```

**注意字面量**：`kv_cache_dtype` 是 pydantic `Literal`，只认 vLLM 的枚举名。写 `bf16` 会在
30 秒内 422：

```
Input should be 'auto', 'float16', 'bfloat16', 'fp8', 'fp8_e4m3', 'fp8_e5m2', 'fp8_inc',
'fp8_ds_mla', 'nvfp4_ds_mla', ... [type=literal_error, input_value='bf16']
```

**通过标志**

```
Resolved architecture: DFlash2DraftModel
Draft model DFlash2Qwen3ForCausalLM ... using text-only draft inputs instead
```

## 2. C4 indexer 的页面 stride 不变量被第二个 KV 组打破

**现象**

```
RuntimeError: Worker 0 failed with error 'GLM MLA parent-page stride must be an exact number of C4 pages'
```

**根因**：GLM-5.3 的 C4 pooled indexer 假设「整块 KV cache 的 stride」是 8448 B（64×132）的整数倍，
见 `pooled_indexer` 里 `parent_stride_bytes % _INDEX_PAGE_BYTES == 0`。K=3 单 KV 组时页面几何
刚好满足（5120 tok × 528 B = 8448 × 320）；加入 drafter KV 组后页面几何被重算，stride 不再整除。

这是 vLLM 侧的布局假设问题，不是你配错了。

**解法**：开 split page，让 target / recurrent 各自独立物理页，外层 block stride 重新按 8448 对齐：

```bash
-e VLLM_GLM53_SPLIT_TARGET_BLOCK_SIZE=5120
-e VLLM_GLM53_SPLIT_MAMBA_BLOCK_SIZE=5120
```

**通过标志**

```
Using split GLM-5.3 cache pages: target block size 5120
Keeping split GLM-5.3 cache groups with physical page sizes [2605056, 2872320]
```

## 3. RoCE one-shot all-gather 在候选 logits 上越界

**现象**：profiling 阶段（`_dummy_run → speculator.propose → compute_candidates → get_top_k_tokens`）：

```
ATen: vectorized_gather_kernel ... Assertion 'ind >= 0 && ind < ind_dim_size' failed
torch.AcceleratorError: CUDA error: device-side assert triggered
(rank1) RoCE collective on rank 0 timed out waiting for rank 1 ... sequence 53
```

**根因**：drafter 的候选 logits 需要一次 TP all-gather（k=16、每行 128 B）。这个形状正好走 b12x 的
RoCE one-shot all-gather「direct」路径，而该路径在这个尺寸上越界。MTP 路径不调用
`get_top_k_tokens`，所以这条链以前从没被走到过。

**解法**：把触发 one-shot 的阈值压到 64 B，让这一个 all-gather 回退 NCCL；all-reduce 仍走 RoCE：

```bash
-e VLLM_ROCE_ALLGATHER_MAX_SIZE=64
```

**通过标志**

```
Using RoCEnante ... all-reduce max=2097152 bytes
```

**对照实验**：`-e VLLM_ENABLE_ROCE_ALLREDUCE=0` 也能绕过（全部 TP 集合通信走 NCCL），但整体更慢，
不推荐作为常驻配置。

## 4. FlashInfer `top_k` 首次使用 JIT 编译（不是死锁）

**现象**：起服停在 profiling，一台 GPU 96%、对端 0%，约 14 分钟没有新日志；
如果等不及去杀进程，对端会报
`No available shared memory broadcast block found in 60 seconds`。

**根因**：`logits_processor.get_top_k_tokens() → flashinfer.top_k` 首次触发 nvcc 编译
`cub_topk.cu`（`arch=compute_121a`）。

**处理**：等。编译产物在容器内 `/root/.cache/flashinfer`，把它 bind 到宿主就能复用：

```bash
-v "$HOME/.cache/flashinfer:/root/.cache/flashinfer"
```

我们的经验：两台的缓存放同一个宿主目录，重装/换镜像后不用再编译。

## 5. 不要关的两个开关

| 开关 | 关掉的后果 |
|---|---|
| `VLLM_B12X_MOE_FP4_FORCE_A16=1` | 实测掉到 ~14 tok/s（A8 激活路径在这套 MoE 上是灾难） |
| `VLLM_USE_V2_MODEL_RUNNER=1` | DFlash2 依赖 V2 runner；关掉直接不支持 |

## 6. 不能混用 MTP 与 DFlash2

`SpeculativeMethod` 是单值 Literal，非 ngram 方法会强制把 `prompt_lookup` 置零；
`use_dflash()` 只认 `method=="dflash"`。所以「MTP + DFlash2 按步路由」在这套 fork 里不存在，
只能按负载切配置（两条命令 + 一次重启）。

## 7. 起服时长与「看起来卡住」

正常首次 30–45 分钟，热启动 6–8 分钟。超过 45 分钟先做三件事：

1. `docker logs --tail 200 vllm_node` 看最后一行是不是编译/加载；
2. 看两台 `nvidia-smi` 是否都活着；
3. 看对端容器是否还在（`scripts/health.sh`）。
