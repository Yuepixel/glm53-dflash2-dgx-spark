#!/usr/bin/env python3
"""bench-decode.py - single-prompt decode benchmark (prose), same harness as bench-code-decode.py.

Usage:
  python3 bench-decode.py http://127.0.0.1:8888 [--model M] [--max-tokens 400]
      [--conc 1,2,4,6] [--repeat 1]
"""
import json
import sys
import threading
import time
import urllib.request

URL = sys.argv[1].rstrip("/")
args = {"model": None, "max-tokens": 400, "conc": "1,2,4,6", "repeat": 1, "think": "0"}
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

PROMPT = ("Write a clear, plain-English explanation of how a small fishing town's harbour "
          "economy works across a year: the seasons, what each group of people does, how money "
          "moves between boats, markets and shops, and what happens in a bad year. "
          "Use flowing prose, no lists, about eight paragraphs.")


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
    usage = 0
    with urllib.request.urlopen(req, timeout=600) as r:
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
                usage = ev["usage"].get("completion_tokens") or usage
            ch = (ev.get("choices") or [{}])[0].get("delta") or {}
            txt = (ch.get("content") or "") + (ch.get("reasoning_content") or "")
            if txt:
                if ttft is None:
                    ttft = time.time() - t0
                n += 1
    out[idx] = {"tokens": usage or n, "streamed": n, "wall": time.time() - t0, "ttft": ttft or 0}


for c in CONC:
    best = None
    for _ in range(REP):
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
        row = {"conc": c, "agg_tok_s": round(toks / wall, 1),
               "stream_tok_s": round(toks / wall / c, 1),
               "tok_per_stream": round(toks / c, 0),
               "ttft_p50_ms": round(1000 * ttfts[len(ttfts) // 2], 0),
               "streamed_per_stream": round(sum(x["streamed"] for x in out if x) / c, 0),
               "wall_s": round(wall, 1)}
        if best is None or row["agg_tok_s"] > best["agg_tok_s"]:
            best = row
    print(json.dumps(best, ensure_ascii=False), flush=True)
