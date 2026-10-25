# Where turbopuffer's time goes

> **Premise:** Before tuning retrieval, know what each query costs: how much server time goes to embedding the query or to execution, whether the cache is warm, whether results are stable across runs, and how much recall ANN gives up compared with exhaustive search.

| Status | Dataset | Cost |
| --- | --- | --- |
| Done | ESCI, the 500 sampled queries, `esci-us-full` | About 10 minutes, under a cent |

## Method

Rerun the first-stage retrievers twice each, one query at a time, and record every field turbopuffer returns. Then ask turbopuffer's recall endpoint how ANN compares with exhaustive search.

## Run

```bash
uv run experiments/turbopuffer-perf/turbopuffer_perf.py
uv run experiments/turbopuffer-perf/turbopuffer_perf.py --report
```

## Result

- Of ANN's 76 ms median server time, 66 ms is embedding the query text.
- BM25 and hybrid run about 103 ms at p50 and 320 ms at p99.
- Every query reported a hot cache.
- Repeating the 500 queries gave the same top 10 for 100% of BM25 queries and 99.4% of ANN and hybrid queries.
- ANN recall against exhaustive search is 0.994 at 50 and 0.995 at 10, from 100 vectors sampled from the namespace.
- 9,785 documents (2%) were still unindexed and scanned exhaustively on every query.

Files: `results/turbopuffer_perf.{csv,txt}`, `results/turbopuffer_recall.json`.
