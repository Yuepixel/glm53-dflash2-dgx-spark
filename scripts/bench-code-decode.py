#!/usr/bin/env python3
"""bench-code-decode.py - concurrent decode benchmark for OpenAI-compatible endpoints.

Streaming SSE, fixed prompts, aggregated throughput per concurrency level.

Usage:
  python3 bench-code-decode.py http://127.0.0.1:8888 [--model M] [--max-tokens 400]
      [--conc 1,2,4,8] [--repeat 3] [--think 0] [--task code] [--dump-prefix PATH]

Output: one JSON line per run plus a MEDIAN line per concurrency level and a
SUMMARY line at the end. aggregated tok/s = completion tokens / wall time.
"""
import json
import os
import statistics
import sys
import threading
import time
import urllib.request

URL = sys.argv[1].rstrip("/")
args = {"model": None, "max-tokens": 400, "conc": "1,2,4,8", "repeat": 3,
        "think": "0", "task": "code", "dump-prefix": ""}
i = 2
while i < len(sys.argv):
    k = sys.argv[i].lstrip("-")
    if k in args:
        args[k] = sys.argv[i + 1]
        i += 2
    else:
        i += 1

MAXT = int(args["max-tokens"])
CONC = [int(x) for x in str(args["conc"]).split(",")]
REP = int(args["repeat"])
THINK = args.get("think") == "1"

TASKS = {
    "code": "Write a Python implementation of a lock-free ring buffer for single-producer "
            "single-consumer with memory barriers documented. Output only code.",
    "code_ts": "Write a TypeScript utility that debounces async calls with cancellation "
               "and retries with exponential backoff. Output only code.",
    "py_api": "Write a FastAPI endpoint for user registration with pydantic validation, "
              "password hashing and duplicate-email handling. Output only code.",
    "count": "Count from 1 to 60, one number per line.",
    "edit": "Refactor this function to be iterative instead of recursive and fix the off-by-one:\n\n"
            "def walk(node):\n    if not node: return []\n    return walk(node.left) + [node.v] + walk(node.right)",
}
PROMPT = TASKS[args["task"]]


def one(idx, out):
    body = {"model": args["model"],
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": MAXT, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": THINK}}
    if args["model"] is None:
        body.pop("model")
    req = urllib.request.Request(URL + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    ttft = None
    n = 0
    usage_tokens = 0
    text = []
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                ev = json.loads(data)
            except Exception:
                continue
            if ev.get("usage"):
                usage_tokens = ev["usage"].get("completion_tokens") or usage_tokens
            ch = (ev.get("choices") or [{}])[0].get("delta") or {}
            txt = (ch.get("content") or "") + (ch.get("reasoning_content") or "")
            if txt:
                if ttft is None:
                    ttft = time.time() - t0
                n += 1
                text.append(txt)
    out[idx] = {"tokens": usage_tokens or n, "streamed": n,
                "wall": time.time() - t0, "ttft": ttft or 0, "text": "".join(text)}


results = []
for c in CONC:
    rows = []
    for rep in range(REP):
        out = [None] * c
        ths = [threading.Thread(target=one, args=(k, out)) for k in range(c)]
        t0 = time.time()
        for t in ths:
            t.start()
        for t in ths:
            t.join()
        wall = time.time() - t0
        toks = sum(x["tokens"] for x in out if x)
        ttfts = sorted(x["ttft"] for x in out if x)
        rows.append({"conc": c, "rep": rep + 1,
                     "agg_tok_s": round(toks / wall, 1),
                     "stream_tok_s": round(toks / wall / c, 1),
                     "tok_per_stream": round(toks / c, 0),
                     "ttft_p50_ms": round(1000 * ttfts[len(ttfts) // 2], 0),
                     "wall_s": round(wall, 1)})
        if args["dump-prefix"]:
            d = os.path.dirname(args["dump-prefix"])
            if d:
                os.makedirs(d, exist_ok=True)
            with open("%s.c%d.r%d.txt" % (args["dump-prefix"], c, rep + 1), "w", encoding="utf-8") as fh:
                fh.write("\n<<<STREAM>>>\n".join(x["text"] for x in out if x))
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    aggs = sorted(r["agg_tok_s"] for r in rows)
    srow = {"conc": c, "agg_tok_s_median": round(statistics.median(aggs), 1),
            "agg_tok_s_best": aggs[-1],
            "stream_tok_s_median": round(statistics.median(aggs) / c, 1),
            "ttft_p50_ms_median": round(statistics.median(r["ttft_p50_ms"] for r in rows), 0),
            "reps": aggs}
    results.append(srow)
    print("MEDIAN " + json.dumps(srow, ensure_ascii=False), flush=True)
print("SUMMARY " + json.dumps(results, ensure_ascii=False), flush=True)
