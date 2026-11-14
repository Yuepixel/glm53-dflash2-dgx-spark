# GLM-5.3-Flash-FP4 + vLLM B12X + DFlash2 on 2× DGX Spark

A working, measured deployment of **DFlash2 speculative decoding** on top of the **vLLM B12X** stack,
serving **GLM-5.3-Flash-FP4** on a **dual DGX Spark (GB10)** cluster.

- **No vLLM source changes.** One recipe file + three runtime flags, all public.
- **Honest numbers, not a 2× story.** DFlash2 wins on predictable content (1.5–1.7×), ties on generic
  code (+5–12% on a 9-prompt corpus), and loses on prose at high concurrency. On two Sparks, 2× is
  **physically unreachable** — the arithmetic is in [docs/performance.md](docs/performance.md).
- Everything was measured on real hardware on 2026-11-14. The benchmark scripts are in [scripts/](scripts)
  so you can reproduce the code-load tables yourself.

[中文说明 / Chinese README](README.zh-CN.md) · [English docs](docs/en/README.md)

## What this repo gives you

1. A drop-in recipe for [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) that starts
   GLM-5.3-Flash-FP4 (TP2) with a DFlash2 drafter at K=7.
2. The three blockers that stop a naive DFlash2 attempt — each with its exact log signature, root cause
   and fix.
3. A/B measurements, acceptance-length analysis, and the roofline math that explains the ceiling.
4. Benchmark scripts (OpenAI-compatible, streaming) and the prompt set used for the reproducible tables.

Weights and container images are **not** redistributed here. This repo is configuration, scripts and docs.

## Tested stack

| Item | Value |
|---|---|
| Hardware | 2× NVIDIA DGX Spark (GB10, `sm_121a`), 121 GiB unified memory each, direct ConnectX-7 200 GbE (RoCE) |
| Engine image | `vllm-node-b12x` (B12X profile, built with `--exp-b12x`); verified with the 2026-11-02 build, digest `sha256:19ab778dd57f…` |
| vLLM | `0.1.dev21830+g8244d1c44.d20261102` (B12X fork image) |
| Target model | [`nvidia/GLM-5.3-Flash-FP4-Local`](https://huggingface.co/nvidia/GLM-5.3-Flash-FP4-Local) (~166 GB, ModelOpt NVFP4) |
| Drafter | [`incoai/GLM-5.3-Flash-DFlash2`](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) (~2.3 GB, block size 8 → K must be 7) |
| Parallelism | TP2 across both nodes, decode-context-parallel 1 |
| Context | 524,288 tokens, `max_num_seqs 8`, `max_num_batched_tokens 4096`, 8 GiB KV per card |
| KV dtype | target: `fp8`; drafter (separate KV group): `bfloat16` |

## Quick start

Assumes two DGX Sparks with passwordless SSH between them, and the head node has the model cache.

```bash
# 1. Runner + cluster discovery (once)
git clone https://github.com/eugr/spark-vllm-docker.git
cd spark-vllm-docker
./run-recipe.sh --discover            # writes CLUSTER_NODES to .env

# 2. Container image (B12X profile)
./run-recipe.sh recipes/glm-5.3-flash-dflash2.yaml --setup --build-only

# 3. Weights (~166 GB) through the runner
./run-recipe.sh recipes/glm-5.3-flash-dflash2.yaml --setup --download-only

# 4. Drafter (separate HF repo, ~2.3 GB)
huggingface-cli download incoai/GLM-5.3-Flash-DFlash2 \
  --local-dir "$HOME/models/GLM-5.3-Flash-DFlash2"

# 5. Drop the recipe into the runner and start
cp /path/to/this/repo/recipes/glm-5.3-flash-dflash2.yaml recipes/
REPO_DIR="$PWD" /path/to/this/repo/scripts/serve.sh

# 6. Verify
curl -s localhost:8888/v1/models
/path/to/this/repo/scripts/health.sh
```

First start is slow: kernel autotuning + a one-off FlashInfer JIT compile (~14 min **per node**).
Warm restarts take ~6–8 min. See [docs/reproduction.md](docs/reproduction.md) for what "normal" looks like.

## The three switches (miss one and it will not start)

| # | Switch | Without it |
|---|---|---|
| 1 | `"kv_cache_dtype":"bfloat16"` inside `--speculative-config` (exact literal; `bf16` is rejected) | `No valid attention backend found … kv_cache_dtype=fp8_ds_mla, use_mla=False …` — the drafter inherits the target's `fp8_ds_mla` even though its attention is not MLA |
| 2 | `-e VLLM_GLM53_SPLIT_TARGET_BLOCK_SIZE=5120 -e VLLM_GLM53_SPLIT_MAMBA_BLOCK_SIZE=5120` | `GLM MLA parent-page stride must be an exact number of C4 pages` — the C4 indexer requires a page stride that is a multiple of 8448 B, and a second (drafter) KV group breaks the unified paging |
| 3 | `-e VLLM_ROCE_ALLGATHER_MAX_SIZE=64` | `vectorized_gather index out of bounds` → `device-side assert` on the candidate-logits all-gather, peer dies with `RoCE collective timed out`. Capping at 64 B routes that one all-gather to NCCL; all-reduce stays on RoCE |

Also keep `VLLM_B12X_MOE_FP4_FORCE_A16=1` (disabling it drops to ~14 tok/s) and
`VLLM_USE_V2_MODEL_RUNNER=1` (DFlash2 requires the V2 runner).

## Measured A/B (reproducible harness)

Code decode throughput, aggregated tok/s, 400 output tokens, temperature 0, 3 runs → median.
`mtp K=3` is the same engine with the built-in MTP head instead of the DFlash2 drafter.

| Prompt (in `scripts/bench-code-decode.py`) | Engine | c1 | c2 | c4 | c8 | Ratio |
|---|---|---|---|---|---|---|
| count-to-60 | mtp K=3 | 44.8 | 64.6 | 95.6 | 139.8 | — |
| | DFlash2 K=7 | **75.6** | **108.0** | **155.4** | **210.7** | **1.51–1.69×** |
| py_api | mtp K=3 | 45.0 | 59.7 | 86.1 | 128.5 | — |
| | DFlash2 K=7 | 48.1 | 57.7 | 99.3 | 125.1 | 0.97–1.15× |
| code_ts | mtp K=3 | 44.4 | 57.7 | 80.7 | 105.5 | — |
| | DFlash2 K=7 | 46.7 | 62.1 | 84.4 | 112.3 | 1.03–1.08× |
| ring buffer | mtp K=3 | 28.7 | 43.1 | 57.1 | 78.6 | — |
| | DFlash2 K=7 | 26.3 | 35.4 | 51.0 | 68.9 | 0.82–0.92× |

**Acceptance-length view** (the only thing that matters):

| Prompt | AL (mtp K=3) | AL (DFlash2 K=7) | Tokens per verified row, K=3 → K=7 |
|---|---|---|---|
| count-to-60 | not measured | 7.1–7.5 | — → 0.89–0.93 |
| py_api | 3.3–3.5 | 4.7–5.0 | 0.88 → 0.58 |
| code_ts | 3.2–3.3 | 3.7–3.8 | 0.83 → 0.47 |
| ring buffer | 3.2 | 2.3–2.6 | 0.80 → 0.29 |

**Latency**: TTFT on structured prompts 63 ms → 25 ms. Prefill throughput is unchanged.

Full tables (structured / JSON / code / prose, c1–c8) and the negative results (MTP K=7, no-spec,
SGLang on SM121) are in [docs/performance.md](docs/performance.md).

## When DFlash2 actually wins

Decode throughput ≈ acceptance length / step time, and step time grows with the number of verified
rows (K+1). So DFlash2 (K=7) only beats MTP (K=3) when:

```
AL_dflash / (7+1) > AL_mtp / (3+1)      ⇔      AL_dflash > 2 × AL_mtp
```

Our MTP baseline accepts 2.2–3.5 tokens/step, so DFlash2 needs **AL > 4.4–7.0** to pay for itself.
That happens on highly predictable content (tables, counting, formatting, boilerplate), and not on
novel prose or genuinely new algorithms.

Rule of thumb:

| Content | Recommendation |
|---|---|
| Tabular / counting / strict formatting | **DFlash2** (1.5–1.9×) |
| JSON, tool-call arguments, boilerplate code | **DFlash2** (1.1–1.3×) |
| New algorithms, long-form prose | **MTP K=3** (DFlash2 ≈ 0.8–1.0×) |
| Mixed agent traffic | DFlash2 on latency; measure your own prompts |

## Pitfalls (all hit in production)

| Symptom | Cause | Fix |
|---|---|---|
| `No valid attention backend found … kv_cache_dtype=fp8_ds_mla` | Drafter KV inherits the target's MLA dtype | ` "kv_cache_dtype":"bfloat16" ` in `--speculative-config` |
| `GLM MLA parent-page stride must be an exact number of C4 pages` | C4 indexer page stride (8448 B) broken by the extra KV group | split-page env vars (`5120` / `5120`) |
| `device-side assert` + `RoCE collective timed out` at startup | B12X one-shot RoCE all-gather overflows on 128 B tensors | `VLLM_ROCE_ALLGATHER_MAX_SIZE=64` |
| One node at 96% GPU, the other idle for ~14 min | FlashInfer `top_k` is JIT-compiling `cub_topk.cu` on first use | wait; bind-mount `/root/.cache/flashinfer` to keep the cache |
| `bf16` rejected with a pydantic `Literal` error | `kv_cache_dtype` takes enum names, not torch dtype names | use `bfloat16` |
| OOM / very long first start | 166 GB weights + 8 GiB KV/card + kernel cache | keep the KV budget, see docs |

Details, log excerpts and source-code anchors: [docs/pitfalls.md](docs/pitfalls.md).

## Layout

```
recipes/    drop-in recipes for eugr/spark-vllm-docker (DFlash2 K=7 and MTP K=3 baseline)
scripts/    serve / stop / health wrappers, benchmark and acceptance-length probes
bench/      measurement tables (CSV) + bench/panel/ (reproducible four-load panel)
docs/       reproduction, pitfalls, performance, benchmark methodology, related work
docs/en/    English versions of the above
tools/      repo checks (relative-link checker, used by CI)
LICENSES/   third-party license texts (Apache-2.0 for bench/panel/prompts.json)
```

## Reproducibility notes

- The code tables are reproducible with the included scripts. The structured/JSON/code/prose panel
  numbers are reproducible with `bench/panel/run-panel.py` (same prompts, same warmup).
- Panel numbers measure the **decode window only** (no prefill/TTFT), so they are higher than the
  wall-clock numbers for the same engine. Do not mix the two tables.
- Temperature 0 is **not** bitwise reproducible on this stack: FP4 MoE routing + bf16 batch shapes make
  even the baseline disagree with itself across runs. Compare semantics, not token-by-token diffs.
- Your acceptance length depends on your prompts more than on anything else in this repo.

## Credits

- [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) — recipe runner, container build,
  cluster discovery. This repo only adds a recipe and notes.
- NVIDIA for the FP4-Local checkpoint and the DGX Spark platform; `incoai` for the DFlash2 drafter.
- [vLLM](https://github.com/vllm-project/vllm) — Apache-2.0.

## License

MIT for the files in this repository (see [LICENSE](LICENSE)). Model weights, container images and
upstream tools keep their own licenses; nothing third-party is redistributed here.
