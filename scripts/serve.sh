#!/usr/bin/env bash
# serve.sh - start GLM-5.3-Flash-FP4 + DFlash2 (K=7) on a 2-node DGX Spark cluster.
#
# Required:
#   eugr/spark-vllm-docker cloned, cluster discovered (./run-recipe.sh --discover),
#   the two recipes copied into its recipes/ directory, weights + drafter downloaded.
#
# Environment overrides:
#   REPO_DIR    runner checkout             (default: $HOME/src/spark-vllm-docker)
#   RECIPE      recipe path                 (default: $REPO_DIR/recipes/glm-5.3-flash-dflash2.yaml)
#   DRAFTER_DIR local DFlash2 drafter        (default: $HOME/models/GLM-5.3-Flash-DFlash2)
#   HF_HOME     HF cache with the weights    (default: $HOME/hf-cache)
#   PORT        host port                    (default: 8888)
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/src/spark-vllm-docker}"
RECIPE="${RECIPE:-$REPO_DIR/recipes/glm-5.3-flash-dflash2.yaml}"
DRAFTER_DIR="${DRAFTER_DIR:-$HOME/models/GLM-5.3-Flash-DFlash2}"
HF_HOME="${HF_HOME:-$HOME/hf-cache}"
PORT="${PORT:-8888}"

[[ -d "$REPO_DIR" ]] || { echo "error: runner checkout not found: $REPO_DIR" >&2; exit 1; }
[[ -f "$RECIPE" ]] || { echo "error: recipe not found: $RECIPE (copy recipes/*.yaml into $REPO_DIR/recipes/)" >&2; exit 1; }
[[ -f "$DRAFTER_DIR/config.json" ]] || { echo "error: drafter not found: $DRAFTER_DIR" >&2; exit 1; }

cd "$REPO_DIR"

# --enable-prompt-tokens-details: exposes per-request token details (used by the bench scripts).
# The three -e flags below are mandatory; see docs/pitfalls.md.
exec ./run-recipe.sh "$RECIPE" --port "$PORT" --daemon \
  --enable-prompt-tokens-details \
  -e HF_HUB_OFFLINE=1 \
  -e PYTHONUNBUFFERED=1 \
  -e VLLM_LOGGING_LEVEL=INFO \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -v "$DRAFTER_DIR:/models/dflash2-draft:ro" \
  -e VLLM_GLM53_SPLIT_TARGET_BLOCK_SIZE=5120 \
  -e VLLM_GLM53_SPLIT_MAMBA_BLOCK_SIZE=5120 \
  -e VLLM_ROCE_ALLGATHER_MAX_SIZE=64
