# 测量方法 / Benchmark methodology

> **English abstract**: One streaming client, fixed prompts, 400 output tokens, 3 runs at each
> concurrency level, median reported. Aggregated tok/s = completion tokens / wall time across all
> streams; TTFT = time to the first content/reasoning delta. Acceptance length comes from the
> standard spec-decode counters (accepted / draft + 1). Temperature 0 is not bitwise reproducible on
> this stack, so we compare semantics.

## 1. 两种口径

| 口径 | 参数 | 用于 |
|---|---|---|
| code harness | 400 tok、thinking off、temp 0、c1/2/4/8、每档 3 次取中位 | 主表、可复现 |
| 内部面板 | 400 tok、thinking on、temp 0.7、c1–c6、c8、固定提示词集 | 结构化/JSON/散文表（背景） |

两个口径不混比。面板数字是「我们自己的 agent 流量形态」，不是通用基准。

## 2. 吞吐怎么算

- 单请求 token 数取响应 `usage.completion_tokens`（没有就数 SSE chunk）
- 聚合 tok/s = 所有流 token 之和 ÷ 墙钟时间（从第一个请求发出到最后一个流结束）
- 单流 tok/s = 聚合 ÷ 并发
- TTFT = 第一个 content 或 reasoning_content delta 的时间；取 p50

## 3. 为什么取 3 次中位

单次运行噪声 ±10%（GPU 时钟、其他进程、KV 命中）。3 次取中位能把 ±10% 压到 ±3% 以内。
对比两个引擎时，同一提示词、同一客户端、同一时段连续跑。

## 4. 接受长度 AL 从哪来

vLLM 的 spec-decode 计数器（Prometheus `/metrics`）：

```
vllm:spec_decode_num_accepted_tokens_total
vllm:spec_decode_num_draft_tokens_total
vllm:spec_decode_num_emitted_tokens_total
```

一段稳态区间的差分：

```
AL ≈ accepted/draft + 1        （每条验证步至少产出 1 个 token）
```

与日志里的 `SpecDecoding metrics` 对得上（±0.1）。计数器名以你的 vLLM 版本为准。

## 5. temp 0 不是逐字可复现的（重要）

同引擎、同配置、同提示词、temp 0，重复三次也不是逐字一致：

| 任务 | 基线内部相似度（rep1 vs 2/3） | DFlash2 内部相似度 | 跨引擎相似度 |
|---|---|---|---|
| py_api | 1.000 / 0.975 | 1.000 / 0.981 | 0.975 |
| code_ts | 0.734 / 0.726 | 0.930 / 0.930 | 0.588 |

机制：(a) FP4 MoE 路由 + bf16 激活在不同批形状下有数值差异，近似并列处会翻转；
(b) 被拒草稿在 recurrent state 里的残留（混合递归模型的老问题）。

所以：**别用「逐字 diff 为空」做验收**，基线自己都过不了。可用验收：同引擎同批形状稳定、
语义/单元测试通过、长上下文不崩、输出语法完整。

## 6. 复现注意事项

- 起服后先跑 2–3 个 warmup 请求，丢掉第一次结果。
- 客户端和引擎在同一台机器，避免网络抖动进入 TTFT。
- 固定 `max_tokens`，不要让不同引擎生成长度不同（聚合吞吐按实际 token 算，但长度会影响调度）。
- 8 并发是本方案 `max_num_seqs` 的上限；更想要高聚合吞吐就调大 seqs + batched_tokens（会改内存账）。
- 想测「你的提示词」：直接改 `scripts/bench-code-decode.py` 里的 `TASKS`。
