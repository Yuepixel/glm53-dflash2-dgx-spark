#!/usr/bin/env python3
"""probe-decode.py - one streaming request, then print decode throughput and TTFT.

Smoke test after a restart: if this returns ~40 tok/s on py_api the engine is healthy.

Usage:
  python3 probe-decode.py http://127.0.0.1:8888 [task] [max_tokens]
  tasks: count | py_api | code_ts | ring | edit   (default: count)
"""
import json
import sys
import time
import urllib.request

URL = sys.argv[1].rstrip("/")
TASK = sys.argv[2] if len(sys.argv) > 2 else "count"
MAXT = int(sys.argv[3]) if len(sys.argv) > 3 else 300
MODEL = "nvidia/GLM-5.3-Flash-FP4-Local"

TASKS = {
    "count": "Count from 1 to 60, one number per line.",
    "code_ts": "Write a TypeScript utility that debounces async calls with cancellation "
               "and retries with exponential backoff. Output only code.",
    "edit": "Refactor this function to be iterative instead of recursive and fix the off-by-one:\n\n"
            "def walk(node):\n    if not node: return []\n    return walk(node.left) + [node.v] + walk(node.right)",
    "ring": "Write a Python implementation of a lock-free ring buffer for single-producer "
            "single-consumer with memory barriers documented. Output only code.",
    "py_api": "Write a FastAPI endpoint for user registration with pydantic validation, "
              "password hashing and duplicate-email handling. Output only code.",
}
PROMPT = TASKS[TASK]

body = {"model": MODEL, "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": MAXT, "temperature": 0, "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False}}
req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json"})
t0 = time.time()
ttft = None
n = 0
usage = 0
with urllib.request.urlopen(req, timeout=600) as r:
    for raw in r:
        line = raw.decode("utf-8", "ignore").strip()
        if not line.startswith("data:"):
            continue
        d = line[5:].strip()
        if d == "[DONE]":
            break
        try:
            ev = json.loads(d)
        except Exception:
            continue
        if ev.get("usage"):
            usage = ev["usage"].get("completion_tokens") or usage
        ch = (ev.get("choices") or [{}])[0].get("delta") or {}
        t = (ch.get("content") or "") + (ch.get("reasoning_content") or "")
        if t:
            if ttft is None:
                ttft = time.time() - t0
            n += 1
wall = time.time() - t0
toks = usage or n
print(json.dumps({"task": TASK, "tokens": toks, "wall_s": round(wall, 2),
                  "ttft_ms": round(1000 * (ttft or 0)),
                  "decode_tok_s": round(toks / max(wall - (ttft or 0), 1e-6), 1)}))
