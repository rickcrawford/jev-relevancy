# Where turbopuffer's time goes

> **Premise:** before tuning retrieval, know what a query costs: how much server time is embedding and how much is execution, whether the cache is warm, whether results repeat, and how much recall ANN gives up.

| Status | Dataset | Cost | Keys |
| --- | --- | --- | --- |
| Done | ESCI, 500 sampled queries, `esci-us-full` | about 10 minutes, under a cent | turbopuffer |

It reruns the first-stage retrievers twice each, one query at a time, and keeps every field turbopuffer returns. Then it asks turbopuffer's recall endpoint how ANN compares with exhaustive search.

```bash
uv run experiments/turbopuffer-perf/turbopuffer_perf.py
uv run experiments/turbopuffer-perf/turbopuffer_perf.py --report > experiments/turbopuffer-perf/results/turbopuffer_perf.txt
```

## Result

- Of ANN's 76 ms median server time, 66 ms is embedding the query.
- BM25 and hybrid run about 103 ms at p50 and 320 ms at p99.
- Every query reported a hot cache.
- Repeating the 500 queries gave the same top 10 for every BM25 query and for 99.4% of ANN and hybrid queries.
- ANN recall against exhaustive search is 0.994 at 50 and 0.995 at 10, on 100 vectors sampled from the namespace.
- 9,785 documents (2%) were still unindexed and scanned exhaustively on every query.

Files: `results/turbopuffer_perf.csv`, `.txt`, and `results/turbopuffer_recall.json`.
