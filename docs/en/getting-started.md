# Getting started

## 0. What you need

| Item | Requirement |
|---|---|
| Nodes | 2× DGX Spark (GB10, `sm_121a`), 121 GiB unified memory each |
| Interconnect | direct ConnectX-7 between the two (RoCE / 200 GbE); the recipe runs TP2 across nodes |
| SSH | head → worker passwordless (`BatchMode=yes` works) |
| Disk | ≥ 250 GB free per node (image ~28 GB + weights 166 GB + kernel caches) |
| Software | DGX OS with working `docker`; Python 3.10+ for the runner |
| Network | Hugging Face reachable, or a pre-populated local cache |

## 1. Cluster discovery and image

```bash
git clone https://github.com/eugr/spark-vllm-docker.git
cd spark-vllm-docker
./run-recipe.sh --discover        # finds nodes, writes CLUSTER_NODES to .env
./run-recipe.sh recipes/glm-5.3-flash-dflash2.yaml --setup --build-only
```

The image is built with the `--exp-b12x` profile and tagged `vllm-node-b12x`. We validated
the 2026-11-02 build (digest `sha256:19ab778dd57f…`). Upstream `latest` moves; pin the
digest if you need byte-identical behaviour.

## 2. Weights and drafter

```bash
# main weights (~166 GB); the runner downloads and copies them to all nodes
./run-recipe.sh recipes/glm-5.3-flash-dflash2.yaml --setup --download-only

# drafter: separate HF repo, must exist at the SAME path on both nodes
huggingface-cli download incoai/GLM-5.3-Flash-DFlash2 \
  --local-dir "$HOME/models/GLM-5.3-Flash-DFlash2"

rsync -a "$HOME/models/GLM-5.3-Flash-DFlash2" worker:"$HOME/models/"
```

Why the same path: `-v $HOME/models/GLM-5.3-Flash-DFlash2:/models/dflash2-draft:ro` is passed
to both ranks. If the directory is missing on the worker, Docker mounts an empty directory and
the engine reports that `/models/dflash2-draft` is not a valid model directory — which looks
like a broken download.

## 3. Install the recipe and start

```bash
cp /path/to/this/repo/recipes/glm-5.3-flash-dflash2.yaml recipes/
REPO_DIR="$PWD" /path/to/this/repo/scripts/serve.sh
```

`serve.sh` cds into the runner checkout, launches with `--daemon`, adds the drafter mount and
the three mandatory `-e` flags.

## 4. First-start timeline

| Stage | Cold | Warm |
|---|---|---|
| Image build / pull | 30–60 min | — |
| Weight download + copy to worker | depends on bandwidth | — |
| Kernel autotuning + CUDA graph capture | 10–20 min | reused |
| FlashInfer `top_k` JIT (once per node) | ~14 min/node | cached |
| Weight load (82.87 GiB shard) | 150–260 s | same |
| **Total** | **~30–45 min** | **6–8 min** |

Telling "compiling" from "deadlocked":

- Compiling: one GPU at 96% busy, the other idle, log still advancing (`Compiling` /
  `cub_topk`). Do not kill it.
- Deadlock: both GPUs idle and `No available shared memory broadcast block found in 60 seconds`
  repeats for more than two windows. That usually means the peer rank already died — check its
  `docker logs`.

## 5. Success criteria

The log should contain, in order:

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

Then:

```bash
curl -s -o /dev/null -w '%{http_code}\n' localhost:8888/health     # 200
curl -s localhost:8888/v1/models | python3 -m json.tool
WORKER_SSH=worker /path/to/this/repo/scripts/health.sh
python3 scripts/probe-decode.py http://127.0.0.1:8888 py_api 400    # ~40 tok/s
```

## 6. Switching between DFlash2 and MTP K=3

The two recipes differ only in `--speculative-config` and `num_speculative_tokens`:

```bash
# back to MTP K=3 (more stable on general code / prose)
WORKER_SSH=worker scripts/stop.sh
RECIPE="$HOME/src/spark-vllm-docker/recipes/glm-5.3-flash-mtp-k3.yaml" scripts/serve.sh

# back to DFlash2 (serve.sh default)
WORKER_SSH=worker scripts/stop.sh && scripts/serve.sh
```

## 7. Stop / restart

```bash
WORKER_SSH=worker scripts/stop.sh
# restart = stop + serve; there is no watchdog on purpose
```

## 8. Troubleshooting

| Symptom | Likely cause | Action |
|---|---|---|
| `No valid attention backend` | drafter KV dtype is not `bfloat16` | pitfall 1 |
| `GLM MLA parent-page stride` | missing split-page `-e` flags | pitfall 2 |
| `device-side assert` / `RoCE collective timed out` | missing `VLLM_ROCE_ALLGATHER_MAX_SIZE=64` | pitfall 3 |
| > 40 min and still starting, log moving | first JIT / compile | wait; keep `/root/.cache/flashinfer` |
| `/health` 200 but throughput ~0 | worker rank died | `scripts/health.sh` |
| OOM | KV budget / another process on the GPU | keep 8 GiB KV per card |
| Only ~14 tok/s | `VLLM_B12X_MOE_FP4_FORCE_A16=0` | set it back to 1 |

## 9. Pitfalls

### 9.1 The drafter inherits the target's `fp8_ds_mla`

Symptom:

```
ValueError: No valid attention backend found for ...
  kv_cache_dtype=fp8_ds_mla, use_mla=False, has_sliding_window=True, use_non_causal=True
```

The DFlash2 drafter's attention is not MLA, but as a new KV group it inherits the target's
MLA-only `kv_cache_dtype`. Fix: set the drafter's KV dtype explicitly.

```json
{"method":"dflash","model":"/models/dflash2-draft","num_speculative_tokens":7,"kv_cache_dtype":"bfloat16"}
```

The literal matters: `kv_cache_dtype` is a pydantic `Literal` over enum names. `bf16` fails
within 30 s with a 422 `literal_error`; use `bfloat16`.

### 9.2 C4 page-stride invariant broken by a second KV group

```
RuntimeError: Worker 0 failed with error 'GLM MLA parent-page stride must be an exact number of C4 pages'
```

GLM-5.3's C4 pooled indexer assumes the whole KV cache stride is a multiple of 8448 B
(64×132). A single KV group satisfies it (5120 tok × 528 B = 8448 × 320); adding a drafter group
changes the page geometry. This is a vLLM layout assumption, not a misconfiguration.

Fix: split cache pages.

```bash
-e VLLM_GLM53_SPLIT_TARGET_BLOCK_SIZE=5120
-e VLLM_GLM53_SPLIT_MAMBA_BLOCK_SIZE=5120
```

### 9.3 RoCE one-shot all-gather overflows on candidate logits

At profiling time:

```
ATen: vectorized_gather_kernel ... Assertion 'ind >= 0 && ind < ind_dim_size' failed
torch.AcceleratorError: CUDA error: device-side assert triggered
(rank1) RoCE collective on rank 0 timed out waiting for rank 1 ... sequence 53
```

The drafter's candidate logits need a TP all-gather (k=16, 128 B rows). That shape takes the
b12x RoCE one-shot "direct" path, which overflows. MTP never calls `get_top_k_tokens`, so this
path was never exercised before.

Fix: cap the one-shot threshold at 64 B so that gather falls back to NCCL; all-reduce stays on
RoCE.

```bash
-e VLLM_ROCE_ALLGATHER_MAX_SIZE=64
```

### 9.4 FlashInfer `top_k` JIT on first use (not a deadlock)

The first `get_top_k_tokens()` compiles `cub_topk.cu` for `compute_121a`: ~14 min per node.
Keep the cache by bind-mounting `/root/.cache/flashinfer`.

### 9.5 Do not disable these two switches

| Switch | Effect if disabled |
|---|---|
| `VLLM_B12X_MOE_FP4_FORCE_A16=1` | drops to ~14 tok/s |
| `VLLM_USE_V2_MODEL_RUNNER=1` | DFlash2 is unsupported |

### 9.6 MTP and DFlash2 cannot be mixed

`SpeculativeMethod` is a single-value Literal and `use_dflash()` only matches
`method=="dflash"`. Per-step routing between MTP and DFlash2 does not exist in this fork; switch
recipes and restart instead.
