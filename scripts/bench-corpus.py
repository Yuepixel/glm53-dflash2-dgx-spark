#!/usr/bin/env python3
"""bench-corpus.py - per-prompt code corpus benchmark with real acceptance length.

For each prompt: read the spec-decode counters, send one streaming request
(temperature 0), read the counters again, and report
  AL = 1 + accepted_drafts / drafts
plus wall-clock tok/s and TTFT. This is the prompt list behind bench/corpus.csv.

Usage:
  python3 bench-corpus.py http://127.0.0.1:8888 [--tag dflash2-k7] [--max-tokens 400]
      [--model nvidia/GLM-5.3-Flash-FP4-Local]
"""
import json
import sys
import time
import urllib.request

BASE = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://127.0.0.1:8888"
opts = {"tag": "run", "max-tokens": 400, "model": "nvidia/GLM-5.3-Flash-FP4-Local"}
i = 2
while i < len(sys.argv):
    k = sys.argv[i].lstrip("-")
    if k in opts:
        opts[k] = sys.argv[i + 1]
        i += 2
    else:
        i += 1
TAG = opts["tag"]
MAXT = int(opts["max-tokens"])
MODEL = opts["model"]

PROMPTS = [
    ("ring", "Write a Python implementation of a lock-free ring buffer for single-producer "
             "single-consumer with memory barriers documented. Output only code."),
    ("py_api", "Write a FastAPI endpoint for user registration with pydantic validation, "
               "password hashing and duplicate-email handling. Output only code."),
    ("code_ts", "Write a TypeScript utility that debounces async calls with cancellation "
                "and retries with exponential backoff. Output only code."),
    ("sql", "Write a PostgreSQL query that returns, for each customer, their total order value "
            "in the last 90 days and the name of their most expensive product. Output only SQL."),
    ("pytest", "Write pytest unit tests for a function parse_span(s) -> (start, end, op) that "
               "parses strings like '12..34:add'. Cover malformed input. Output only code."),
    ("regex", "Write a Python regex and a short function that extracts timestamp, level and "
              "message from log lines like '2026-11-14T01:02:03Z ERROR foo: bar baz'. Output only code."),
    ("docker", "Write a production-ready multi-stage Dockerfile for a Python 3.12 FastAPI app "
               "that installs deps with uv, runs as non-root and exposes 8000. Output only code."),
    ("react", "Write a React hook useFetch(url) with AbortController cleanup, loading and error "
              "state, and TypeScript types. Output only code."),
    ("refactor", "Refactor this function to be iterative instead of recursive and fix the off-by-one:\n\n"
                 "def walk(node):\n    if not node: return []\n    return walk(node.left) + [node.v] + walk(node.right)"),
]

DRAFT_COUNTERS = ("vllm:spec_decode_num_drafts_total",
                  "vllm:spec_decode_num_draft_tokens_total")
ACCEPT_COUNTER = "vllm:spec_decode_num_accepted_tokens_total"


def counters():
    out = {"drafts": 0.0, "acc": 0.0}
    try:
        with urllib.request.urlopen(BASE + "/metrics", timeout=10) as r:
            for line in r:
                s = line.decode("utf-8", "ignore")
                for name in DRAFT_COUNTERS:
                    if s.startswith(name):
                        out["drafts"] = float(s.rsplit(" ", 1)[1])
                if s.startswith(ACCEPT_COUNTER):
                    out["acc"] = float(s.rsplit(" ", 1)[1])
    except Exception as e:
        out["err"] = str(e)
    return out


def run(prompt):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": MAXT, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    ttft = None
    n = 0
    usage = 0
    with urllib.request.urlopen(req, timeout=900) as r:
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
    return {"tokens": toks, "wall_s": round(wall, 2),
            "ttft_ms": round(1000 * (ttft or 0)),
            "tok_s": round(toks / wall, 1),
            "decode_tok_s": round(toks / max(wall - (ttft or 0), 1e-6), 1)}


rows = []
for name, prompt in PROMPTS:
    c0 = counters()
    res = run(prompt)
    c1 = counters()
    drafts = c1["drafts"] - c0["drafts"]
    acc = c1["acc"] - c0["acc"]
    al = round(1.0 + acc / drafts, 2) if drafts > 0 else None
    row = {"tag": TAG, "task": name, **res, "al": al,
           "drafts": int(drafts), "accepted": int(acc)}
    rows.append(row)
    print(json.dumps(row, ensure_ascii=False), flush=True)

al_vals = [r["al"] for r in rows if r["al"]]
ts_vals = [r["tok_s"] for r in rows]
print("SUMMARY " + json.dumps({
    "tag": TAG,
    "mean_al": round(sum(al_vals) / len(al_vals), 2) if al_vals else None,
    "mean_tok_s": round(sum(ts_vals) / len(ts_vals), 1),
    "median_tok_s": sorted(ts_vals)[len(ts_vals) // 2]}, ensure_ascii=False))
