# Performance

## 1. Test stack

2× DGX Spark (GB10, `sm_121a`), direct 200 GbE RoCE, TP2. Engine `vllm-node-b12x`
(2026-11-02 build, vLLM `0.1.dev21830`). Target `nvidia/GLM-5.3-Flash-FP4-Local`. Drafter
`incoai/GLM-5.3-Flash-DFlash2` at K=7; baseline is the built-in MTP head at K=3.

## 2. Code A/B (reproducible, wall-clock)

Aggregated tok/s, 400 output tokens, temperature 0, median of 3:

| Prompt | Engine | c1 | c2 | c4 | c8 | Ratio |
|---|---|---|---|---|---|---|
| count-to-60 | mtp K=3 | 44.8 | 64.6 | 95.6 | 139.8 | — |
| | DFlash2 K=7 | 75.6 | 108.0 | 155.4 | 210.7 | **1.51–1.69×** |
| py_api | mtp K=3 | 45.0 | 59.7 | 86.1 | 128.5 | — |
| | DFlash2 K=7 | 48.1 | 57.7 | 99.3 | 125.1 | 0.97–1.15× |
| code_ts | mtp K=3 | 44.4 | 57.7 | 80.7 | 105.5 | — |
| | DFlash2 K=7 | 46.7 | 62.1 | 84.4 | 112.3 | 1.03–1.08× |
| ring buffer | mtp K=3 | 28.7 | 43.1 | 57.1 | 78.6 | — |
| | DFlash2 K=7 | 26.3 | 35.4 | 51.0 | 68.9 | 0.82–0.92× |

## 3. Nine-prompt code corpus (c1, temperature 0)

| Prompt | mtp tok/s | mtp AL | DFlash2 tok/s | DFlash2 AL | Ratio |
|---|---|---|---|---|---|
| ring | 30.2 | 2.19 | 25.7 | 2.61 | 0.85× |
| py_api | 43.9 | 3.27 | 48.1 | 4.99 | 1.10× |
| code_ts | 42.8 | 3.19 | 36.5 | 3.80 | 0.85× |
| sql | 39.1 | 2.87 | 47.9 | 5.02 | **1.23×** |
| pytest | 39.6 | 2.98 | 44.4 | 4.75 | 1.12× |
| regex | 36.9 | 3.18 | 48.8 | 5.64 | **1.32×** |
| docker | 42.8 | 3.09 | 37.8 | 3.76 | 0.88× |
| react | 45.7 | 3.41 | 46.9 | 4.96 | 1.03× |
| refactor | 34.8 | 2.62 | 37.5 | 3.84 | 1.08× |
| **mean / median** | **39.5** | 2.98 | **41.5** | 4.37 | **1.05× / 1.12×** |

DFlash2 accepts more tokens per step (4.4 vs 3.0) but verifies four extra rows per step.

## 4. Structured / JSON / code / prose (panel, decode window)

400 tokens, thinking on, temperature 0.7, concurrency 1–6 and 8. Prompt sets and harness:
`bench/panel/`. These numbers measure the decode window only, so they are not directly
comparable with the wall-clock tables above.

| Load | Engine | c1 | c2 | c3 | c4 | c5 | c6 | c8 | Mean ratio |
|---|---|---|---|---|---|---|---|---|---|
| Structured | DFlash2 | 78.9 | 124.6 | 159.4 | 194.4 | 202.0 | **227.5** | 206.2 | **1.85×** |
| | mtp K=3 | 47.9 | 57.7 | 70.3 | 104.8 | 99.1 | 135.1 | 160.2 | |
| JSON | DFlash2 | 61.7 | 70.7 | 105.7 | 110.9 | 126.4 | 131.7 | **132.7** | 1.27× |
| | mtp K=3 | 49.0 | 67.1 | 65.6 | 88.7 | 87.9 | 104.9 | 129.3 | |
| Code | DFlash2 | 68.6 | 77.7 | 87.6 | 92.6 | 84.4 | 101.3 | **106.8** | 1.26× |
| | mtp K=3 | 49.2 | 60.3 | 57.7 | 76.5 | 73.5 | 85.8 | 99.9 | |
| Prose | DFlash2 | 35.4 | 48.5 | 54.9 | 61.5 | 68.2 | 77.8 | 71.1 | 0.99× |
| | mtp K=3 | 35.8 | 48.4 | 47.5 | 65.0 | 63.6 | 79.3 | **91.9** | |

TTFT on structured: **63 ms → 25 ms**. Prefill is unchanged: DFlash2 2413/2358/2352/2327 tok/s
vs K=3 2385/2340/2324/2302 at 8k/16k/32k/64k.

## 5. Step-cost model

| Task | Engine | AL | tok/s | Step time | Tokens per verified row AL/(K+1) |
|---|---|---|---|---|---|
| ring | mtp K=3 | 3.19 | 28.7 | ~111 ms | **0.80** |
| ring | mtp K=7 | 2.48 | 23.8 | ~150 ms | 0.31 |
| ring | DFlash2 K=7 | 2.28 | 26.3 | ~98 ms | 0.29 |
| py_api | mtp K=3 | 3.53 | 45.0 | ~78 ms | **0.88** |
| py_api | mtp K=7 | 6.00 | 39.2 | ~153 ms | 0.75 |
| py_api | DFlash2 K=7 | 4.67 | 48.1 | ~90 ms | 0.58 |
| count | DFlash2 K=7 | 7.05–7.52 | 75.6 | ~90 ms | 0.88–0.94 |

1. Step time ≈ rows × per-row cost (18–26 ms each). A wider drafter does not change the cost
   per row; it only changes how many tokens a step yields.
2. The payoff criterion is `AL/(K+1)`, not AL.

```
DFlash2 beats mtp K=3  ⇔  AL_dflash/(7+1) > AL_mtp/(3+1)  ⇔  AL_dflash > 2 × AL_mtp
```

Our MTP baseline accepts 2.2–3.5, so DFlash2 needs AL > 4.4–7.0. That is a property of the
content, not a tuning knob.

## 6. Bandwidth roofline: why 2× is unreachable on two Sparks

- Peak: 2 × 273 GB/s = **546 GB/s**
- 42 of 45 layers are MoE, 288 experts/layer, top-8; one expert ≈ 25.2 M params at NVFP4
  (≈ 0.56 B/param) ≈ **14 MB**
- Lower bound with no expert reuse: 42 × 8 × 14 MB ≈ **4.7 GB/token**

| Config | c8 tok/s | Implied expert bytes/token |
|---|---|---|
| mtp K=3 + py_api | 128.5 | 4.26 GB (≈ 91% of the floor) |
| DFlash2 K=7 + py_api | 125.1 | 4.37 GB |
| mtp K=3 + ring | 78.6 | ~6.9 GB |
| DFlash2 K=7 + count | 210.7 | 2.6 GB (high AL reuses experts) |

2× (157 tok/s at c8) needs < 3.5 GB/token, below the model's own 4.7 GB floor. Without more
GPUs, lower bits or fewer experts, 2× is impossible. Typical workloads cap at 116–130 tok/s;
only very high acceptance length reaches 210.

## 7. Negative results

| Attempt | Result |
|---|---|
| MTP K=7 | slower at every level: 21.6/29.5/42.2/62.5 (c1/2/4/8), 0.68–0.80× |
| No speculative decoding | ring 25.7/38.0/56.3/76.9, 12–74% below K=3 |
| A8 activation (`VLLM_B12X_MOE_FP4_FORCE_A16=0`) | 14 tok/s |
| SGLang on GLM-5.3 | no usable DSA attention backend on SM121 |
| MTP + DFlash2 hybrid routing | single-value SpeculativeMethod; SGLang hybrid not available here |

## 8. Benchmark method

- **Wall-clock tables** (`scripts/bench-code-decode.py`, `scripts/bench-corpus.py`):
  aggregated tok/s = completion tokens / wall time; TTFT = first content/reasoning delta;
  median of 3 runs at each concurrency level.
- **Panel tables** (`bench/panel/run-panel.py`): 32-token warmup per stream, then
  `(completion_tokens-1)/(last_delta-first_delta)` per stream and the concurrent window for
  the aggregate. Prefill/TTFT excluded by design.
- **Acceptance length** from the standard counters
  `vllm:spec_decode_num_accepted_tokens_total` and `vllm:spec_decode_num_drafts_total`:
  `AL = 1 + accepted/drafts`.
- Temperature 0 is **not** bitwise reproducible here: FP4 MoE routing plus bf16 batch shapes
  make the baseline disagree with itself across runs. Compare semantics, not token diffs.
