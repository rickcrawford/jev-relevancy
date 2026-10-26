# Jev 1.14.0 against 1.13.0

> **Premise:** the newer Jev judges and reranks better than 1.13.0, the version turbopuffer hosts and the main runs use.

| Status | Dataset | Cost | Keys |
| --- | --- | --- | --- |
| Done | ESCI, 500 sampled queries | one step 8 run, one step 6 run | TypeSafe, OpenRouter (Gemma) |

There's no script. It reruns the main scripts on a copy of the warehouse, so the main runs stay as they were:

```bash
cp warehouse/relevancy.duckdb /tmp/jev114.duckdb
export RELEVANCY_DB=/tmp/jev114.duckdb
uv run scripts/08_judge_runs.py --judge jev-batched:jev-1.13.0 --judge jev-batched:jev-1.14.0 --judge google/gemma-3-27b-it
uv run scripts/06_rerank.py --model jev:jev-1.14.0
uv run scripts/05_score.py --base <run_id of your jev:jev-1.13.0 rerank>
```

The two CSVs in `results/` were exported from that copy: 1.14.0's grades (`jev114_judge.csv`) and its reranked order (`jev114_rerank.csv`). The export query wasn't saved. `evidence.py` reads them for §15.

## Result

**A better judge, the same reranker.** As a judge on the same 3,692 labeled pairs, 1.14.0's exact agreement is 0.031 higher (+0.019 to +0.045) and its kappa is 0.471 against 0.436. Step 8 showed the same Kendall tau for both, but that value wasn't saved. As a reranker it's level: -0.001 nDCG@10 (-0.007 to +0.004). The two versions give the same grade on 87.8% of pairs.
