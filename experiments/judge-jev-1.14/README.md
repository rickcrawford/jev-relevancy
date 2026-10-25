# Jev 1.14.0 against 1.13.0

> **Premise:** The newer Jev is a better judge and reranker than 1.13.0, the version turbopuffer hosts and the main runs use.

| Status | Dataset | Cost |
| --- | --- | --- |
| Done | ESCI, the 500 sampled queries | One step 8 and one step 6 run |

## Method

Run the main scripts against a copy of the warehouse, so the main runs stay as they were.

## Run

```bash
cp warehouse/relevancy.duckdb /tmp/jev114.duckdb
export RELEVANCY_DB=/tmp/jev114.duckdb
uv run scripts/08_judge_runs.py --judge jev-batched:jev-1.13.0 --judge jev-batched:jev-1.14.0 --judge google/gemma-3-27b-it
uv run scripts/06_rerank.py --model jev:jev-1.14.0
uv run scripts/05_score.py --base jev-1.13.0-b53bce
```

## Result

**A better judge, the same reranker.** As a judge, 1.14.0 has +0.031 exact agreement on the same 3,692 pairs (interval +0.019 to +0.045), kappa 0.471 against 0.436, and the same tau. As a reranker it's level: -0.001 nDCG@10 (interval -0.007 to +0.004). Files: `results/jev114_judge.csv` and `results/jev114_rerank.csv`, which `evidence.py` reads for section 15.
