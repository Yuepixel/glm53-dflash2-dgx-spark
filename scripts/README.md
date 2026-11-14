# scripts

| Script | What it does |
|---|---|
| `serve.sh` | Wraps `run-recipe.sh` with the drafter mount and the three mandatory `-e` flags |
| `health.sh` | Verifies `/health` + container presence on both nodes |
| `stop.sh` | Stops the container on head (and worker, if `WORKER_SSH` is set) |
| `bench-code-decode.py` | Concurrency sweep over fixed code prompts (the reproducible tables) |
| `bench-decode.py` | Same harness, single prose prompt |
| `probe-decode.py` | One request; prints TTFT and decode tok/s (post-restart smoke test) |

Typical run:

```bash
# smoke
python3 scripts/probe-decode.py http://127.0.0.1:8888 py_api 400

# A/B the same prompt on the two engines (switch recipes between runs)
python3 scripts/bench-code-decode.py http://127.0.0.1:8888 --task count --conc 1,2,4,8 --repeat 3
```

All scripts take a base URL and use only the standard library, so they run on the head node
without installing anything. The model name is passed through `--model` where needed; pass
`--model nvidia/GLM-5.3-Flash-FP4-Local` if your gateway requires it.
