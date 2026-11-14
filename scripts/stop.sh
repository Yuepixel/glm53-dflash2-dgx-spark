#!/usr/bin/env bash
# stop.sh - stop the vLLM container on the head node and, optionally, on the worker.
#
# Set WORKER_SSH to an ssh alias/host of the second node (e.g. WORKER_SSH=spark-worker).
# Health of a TP2 engine is only meaningful when both ranks are up; stopping the head
# alone leaves the worker spinning, so stop both.
set -euo pipefail

CONTAINER="${CONTAINER:-vllm_node}"
WORKER_SSH="${WORKER_SSH:-}"

echo "stopping $CONTAINER on head"
docker stop "$CONTAINER" >/dev/null 2>&1 || echo "warn: head container was not running"

if [[ -n "$WORKER_SSH" ]]; then
  echo "stopping $CONTAINER on worker ($WORKER_SSH)"
  ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$WORKER_SSH" \
    "docker stop $CONTAINER >/dev/null 2>&1 || echo 'warn: worker container was not running'" || true
else
  echo "note: WORKER_SSH unset - worker container not checked"
fi
