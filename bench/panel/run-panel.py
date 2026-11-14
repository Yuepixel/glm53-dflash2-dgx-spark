#!/usr/bin/env python3
"""run-panel.py - reproduce the four-load decode panel behind bench/panel-ab.csv.

Method (matches the panel harness the numbers came from):
  * streaming requests; temperature 0.7, top_p 1, max_tokens 400, thinking on
  * every measured wave is preceded by a 32-token warmup stream per stream
  * throughput is measured on the DECODE WINDOW only:
        per stream : (completion_tokens - 1) / (last delta - first delta)
        aggregate  : sum(decode tokens) / (max(last) - min(first)) over the wave
    Prefill, TTFT and teardown are excluded. These numbers are therefore NOT
    comparable with scripts/bench-code-decode.py, which uses wall time.
  * for concurrency > 1, non-code prompts get a "Request ID: decode-N" prefix
    so streams do not share a prefix-cache block
  * code waves use a different task per stream (bench/panel/prompts.json)

Usage:
  python3 bench/panel/run-panel.py http://127.0.0.1:8888 --type structured
  python3 bench/panel/run-panel.py http://127.0.0.1:8888 --type code --conc 1,2,4,6,8 --repeat 3
  types: structured | json | code | prose
"""
import json
import os
import statistics
import sys
import threading
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "prompts.json"), encoding="utf-8") as fh:
    CAT = json.load(fh)

BASE = sys.argv[1].rstrip("/")
opts = {"type": "structured", "conc": "1,2,3,4,5,6,8", "repeat": 1,
        "max-tokens": 400, "temp": 0.7, "top-p": 1.0,
        "warmup-tokens": 32, "think": 1, "model": ""}
i = 2
while i < len(sys.argv):
    k = sys.argv[i].lstrip("-")
    if k in opts:
        opts[k] = sys.argv[i + 1]
        i += 2
    else:
        i += 1

KIND = opts["type"]
if KIND not in ("structured", "json", "code", "prose"):
    sys.exit("error: --type must be structured|json|code|prose")
CONC = [int(x) for x in str(opts["conc"]).split(",")]
REPEAT = int(opts["repeat"])
MAXT = int(opts["max-tokens"])
WARM = int(opts["warmup-tokens"])
TEMP = float(opts["temp"])
TOPP = float(opts["top-p"])
THINK = str(opts["think"]) == "1"
MODEL = opts["model"] or None

THINK_KW = ({"enable_thinking": True, "thinking": True, "thinking_mode": "on"}
            if THINK else {"enable_thinking": False})


def stream(prompt, max_tokens):
    body = {"messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": TEMP, "top_p": TOPP,
            "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": THINK_KW}
    if MODEL:
        body["model"] = MODEL
    req = urllib.request.Request(BASE + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    first = last = None
    n = 0
    usage = 0
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
                usage = ev["usage"].get("completion_tokens") or usage
            ch = (ev.get("choices") or [{}])[0].get("delta") or {}
            txt = (ch.get("content") or "") + (ch.get("reasoning_content") or "")
            if txt:
                now = time.time()
                if first is None:
                    first = now
                last = now
                n += 1
    return {"tokens": usage or n, "first": first, "last": last,
            "ttft": (first - t0) if first else None}


def prompts_for(kind, c):
    p = CAT["prompts"]
    if kind == "code":
        tasks = p["code_tasks"][:c]
        if len(tasks) < c:
            sys.exit("error: not enough code tasks for concurrency %d" % c)
        return ["%s\n%s\n%s" % (t["name"], t["spec"], p["code_tail"]) for t in tasks]
    base = p[kind]
    if c == 1:
        return [base]
    return ["Request ID: decode-%d\n%s" % (k, base) for k in range(c)]


def wave(prompts):
    c = len(prompts)
    out = [None] * c
    barrier = threading.Barrier(c)

    def work(k):
        barrier.wait()
        out[k] = stream(prompts[k], MAXT)

    # warm up every stream with its own prompt before measuring (32 tokens)
    for k in range(c):
        stream(prompts[k], WARM)

    ths = [threading.Thread(target=work, args=(k,)) for k in range(c)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    firsts = [x["first"] for x in out if x and x["first"]]
    lasts = [x["last"] for x in out if x and x["last"]]
    if not firsts:
        return None
    window = max(lasts) - min(firsts)
    dec = [max(x["tokens"] - 1, 0) for x in out]
    per = [d / (x["last"] - x["first"]) for d, x in zip(dec, out) if x["last"] > x["first"]]
    ttfts = sorted(1000 * x["ttft"] for x in out if x["ttft"])
    return {"conc": c,
            "agg_tok_s": round(sum(dec) / window, 1),
            "stream_tok_s_median": round(statistics.median(per), 1) if per else None,
            "ttft_p50_ms": round(ttfts[len(ttfts) // 2]) if ttfts else None,
            "decode_tokens": sum(dec),
            "window_s": round(window, 2)}


for c in CONC:
    prompts = prompts_for(KIND, c)
    rows = [wave(prompts) for _ in range(REPEAT)]
    rows = [r for r in rows if r]
    if not rows:
        continue
    aggs = sorted(r["agg_tok_s"] for r in rows)
    summary = {"type": KIND, "conc": c,
               "agg_tok_s_median": round(statistics.median(aggs), 1),
               "agg_tok_s_best": aggs[-1],
               "ttft_p50_ms_median": round(statistics.median(r["ttft_p50_ms"] for r in rows)) if rows[0]["ttft_p50_ms"] is not None else None,
               "reps": aggs}
    print("MEDIAN " + json.dumps(summary, ensure_ascii=False), flush=True)
