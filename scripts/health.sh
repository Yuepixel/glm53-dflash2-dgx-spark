#!/usr/bin/env bash
# health.sh - engine health = head /health 200 AND the container exists on every node.
#
# A TP2 engine can keep answering /health on the head after the worker rank died,
# so both containers must be checked. Set WORKER_SSH to check the second node.
set -euo pipefail

PORT="${PORT:-8888}"
CONTAINER="${CONTAINER:-vllm_node}"
WORKER_SSH="${WORKER_SSH:-}"
URL="http://127.0.0.1:$PORT/health"

ok=0
code=000
for _ in 1 2 3; do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 "$URL" || true)
  if [[ "$code" == "200" ]]; then ok=1; break; fi
  sleep 2
done

if [[ "$ok" != "1" ]]; then
  echo "FAIL: $URL returned $code (3 samples)"
  exit 1
fi
echo "ok: health 200"

if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
  echo "FAIL: /health is 200 but the head container is missing"
  exit 1
fi
echo "ok: head container up"

if [[ -n "$WORKER_SSH" ]]; then
  if ! ssh -n -o BatchMode=yes -o ConnectTimeout=8 "$WORKER_SSH" \
       "docker ps --format '{{.Names}}'" 2>/dev/null | grep -qx "$CONTAINER"; then
    echo "FAIL: /health is 200 but the WORKER container is missing (rank died)"
    exit 1
  fi
  echo "ok: worker container up"
fi

echo "health: OK"
