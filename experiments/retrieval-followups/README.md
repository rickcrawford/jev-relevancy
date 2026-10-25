# Four retrieval follow-ups

> **Premise:** Four small changes the main runs leave open each move quality or latency: BM25 field weights, hybrid search and reranking in one request, a brand filter, and Elastic's judged-only setup.

| Status | Dataset | Cost |
| --- | --- | --- |
| Done | ESCI, the 500 sampled queries, `esci-us-full` | Well under $1 |

## Method

- `bm25`: four BM25 field variants.
- `onecall`: hybrid search plus Voyage in one `multi_query` request, with 50 and 100 candidates per subquery.
- `filters`: a brand filter on queries that name a brand. The brand comes from the human-Exact products, so this is an upper bound.
- `judged`: reranking only each query's judged products, as in KDD Cup Task 1.

## Run

```bash
uv run experiments/retrieval-followups/retrieval_followups.py bm25   # also onecall, filters, judged
uv run experiments/retrieval-followups/retrieval_followups.py report
```

## Result

- **Title-only BM25** beats title plus body: 17 ms p50 and 43 ms p99, against about 100 ms and 300 ms, and +0.020 nDCG@10.
- **One request** with `rerank_by` on `multi_query`: wall time 362 to 308 ms, and +0.006 nDCG@10. With 100 candidates per subquery, +0.016.
- **Brand filter** on 81 queries: +0.047 nDCG@10 (an upper bound), server time 79 to 77 ms.
- **Judged products only:** Jev 0.868, Voyage 0.848, product-id order 0.679.

Full output: `results/retrieval_followups.txt`. Follow-ups: [`recall-fusion`](../recall-fusion/) and [`relevance-ontology`](../relevance-ontology/).
