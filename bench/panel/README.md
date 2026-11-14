# Panel benchmark (structured / json / code / prose)

This directory reproduces the numbers in [../panel-ab.csv](../panel-ab.csv) — the
four-load table quoted in the main README.

## Method

Matches the harness the numbers came from:

| Item | Value |
|---|---|
| Transport | streaming SSE, OpenAI-compatible endpoint |
| Sampling | temperature 0.7, top_p 1, max_tokens 400, thinking on |
| Warmup | one 32-token stream per stream before every measured wave |
| Measurement | decode window only: per stream (completion_tokens-1)/(last_delta-first_delta); aggregate = sum(decode tokens) / (max(last)-min(first)) |
| Concurrency | 1, 2, 3, 4, 5, 6, 8 |
| Prompt sharing | concurrency > 1 gets a "Request ID: decode-N" prefix; code uses a distinct task per stream |

Two consequences worth knowing:

1. These numbers are higher than wall-clock numbers (which include prefill, TTFT and
   teardown) for the same engine. Do not mix them with the code tables produced by
   `scripts/bench-code-decode.py`.
2. Prompt-cache warmth is part of the method: the warmup stream primes the prefix, then
   the measured stream only pays decode.

## Usage

```bash
python3 bench/panel/run-panel.py http://127.0.0.1:8888 --type structured
python3 bench/panel/run-panel.py http://127.0.0.1:8888 --type json --conc 1,2,4,6,8 --repeat 3
python3 bench/panel/run-panel.py http://127.0.0.1:8888 --type code --conc 1,2,4,6,8
python3 bench/panel/run-panel.py http://127.0.0.1:8888 --type prose --conc 1,2,4,6,8
```

The published panel numbers were single-run (noise about ±10%). Use `--repeat 3` and the
median if you want tighter numbers.

## Prompts and licensing

`prompts.json` contains the four prompt sets and the per-stream code task list, copied
verbatim from sparkDash (https://github.com/miaai-lab/sparkdash, Apache-2.0,
`src/shared/llmPrompts.js`) so results stay comparable with the published table. The
file is Apache-2.0; a copy of the license is in
[../../LICENSES/Apache-2.0.txt](../../LICENSES/Apache-2.0.txt) and attribution is in
[../../NOTICE](../../NOTICE).

## Caveats

- Single-run panel data has ±10% noise. The aggregate ratios are stable; individual cells
  are not.
- `structured` (count 1→200) and `prose` (hash-map explanation) are "lab protocol"
  prompts, not representative chat traffic.
- Concurrency above 8 queues behind `max_num_seqs=8` unless you raise it in the recipe.
