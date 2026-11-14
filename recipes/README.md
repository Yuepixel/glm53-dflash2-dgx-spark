# Recipes

Drop these YAML files into a clone of
[eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) under `recipes/`.

| File | Purpose |
|---|---|
| `glm-5.3-flash-dflash2.yaml` | Production config: GLM-5.3-Flash-FP4 + DFlash2 drafter, K=7 |
| `glm-5.3-flash-mtp-k3.yaml` | A/B baseline: same engine and weights, built-in MTP head, K=3 |

Both are `cluster_only: true` and assume two nodes. Differences between the two files:
only `--speculative-config` and `num_speculative_tokens`.

## Why the container is `vllm-node-b12x`

`run-recipe.sh ... --exp-b12x` pulls/tags the B12X image as `vllm-node-b12x`. The recipe
intentionally uses the untagged name so it works with whatever B12X build you have locally.
We validated the 2026-11-02 build (digest `sha256:19ab778dd57f019c44caf5df91a671cf688923d1ca9fa8ce92be0c9acb46d215`).
If you need byte-identical behaviour, build/pull that exact image and re-tag it as `vllm-node-b12x`.

## The drafter is mounted, not declared as `model:`

The runner's `model:` field drives the main-weight download. The DFlash2 drafter lives in a
separate HF repo (`incoai/GLM-5.3-Flash-DFlash2`) and is exposed to the container as a bind mount:

```bash
-v "$HOME/models/GLM-5.3-Flash-DFlash2:/models/dflash2-draft:ro"
```

`scripts/serve.sh` does this for you and also adds the three mandatory `-e` flags.

## Runtime flags that are NOT in the YAML

Three flags must be passed on the `run-recipe.sh` command line (they are `docker -e` passthroughs):

```bash
-e VLLM_GLM53_SPLIT_TARGET_BLOCK_SIZE=5120
-e VLLM_GLM53_SPLIT_MAMBA_BLOCK_SIZE=5120
-e VLLM_ROCE_ALLGATHER_MAX_SIZE=64
```

They are kept on the command line because they are scoped to the drafter KV group and are the
first thing to remove when bisecting a startup failure. See `docs/pitfalls.md`.
