# 2026-11-14 测量记录

三个 CSV 是同一批测量的机器可读版本，表头见各文件。说明与结论见
[../docs/performance.md](../docs/performance.md)，方法见
[../docs/benchmark-method.md](../docs/benchmark-method.md)。

## 文件

| 文件 | 内容 |
|---|---|
| `code-ab.csv` | 4 条 code 提示词 × 2 引擎 × c1/2/4/8（可复现） |
| `panel-ab.csv` | Structured / JSON / Code / Prose × 2 引擎 × c1–c6,c8（decode 窗口口径；提示词与 harness 在 `bench/panel/`，可复现） |
| `corpus.csv` | 9 条真实 code 提示词 c1 对照 + 接受长度 |

## 复现命令

```bash
# 同一提示词、同一并发档，两个引擎各跑一次（切换配方后重启）
python3 scripts/bench-code-decode.py http://127.0.0.1:8888 \
  --task count --conc 1,2,4,8 --repeat 3 --max-tokens 400
python3 scripts/bench-code-decode.py http://127.0.0.1:8888 \
  --task py_api --conc 1,2,4,8 --repeat 3 --max-tokens 400
```

CSV 里的 `agg_tok_s`：code-ab 是 3 次中位；panel-ab 是单次运行（±10%）。`al` 是同一区间的接受长度
（见方法文档 §4）。

## 面板口径（panel-ab.csv）

decode 窗口口径（不含 prefill/TTFT），提示词、32-token 预热和算法都在 `panel/`：

```bash
python3 bench/panel/run-panel.py http://127.0.0.1:8888 --type structured --conc 1,2,3,4,5,6,8
python3 bench/panel/run-panel.py http://127.0.0.1:8888 --type json --repeat 3
```

## corpus（corpus.csv）

9 条真实 code 提示词，墙钟 tok/s + 从 `/metrics` 差分得到的接受长度：

```bash
python3 scripts/bench-corpus.py http://127.0.0.1:8888 --tag dflash2-k7
```
