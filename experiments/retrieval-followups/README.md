# Four retrieval follow-ups

> **Premise:** four small changes the main runs left open each move quality or latency: BM25 field choice, hybrid and reranking in one request, a brand filter, and Elastic's judged-only setup.

| Status | Dataset | Cost | Keys |
| --- | --- | --- | --- |
| Done | ESCI, 500 sampled queries, `esci-us-full` | well under $1 | turbopuffer; TypeSafe for `judged` |

| Command | What it runs |
| --- | --- |
| `bm25` | four BM25 field variants |
| `onecall` | hybrid plus Voyage in one `multi_query` request, at 50 and 100 candidates per subquery |
| `filters` | a brand filter on queries that name a brand, with the brand taken from the human-Exact products (an upper bound) |
| `judged` | reranking only each query's labeled products, as in KDD Cup Task 1 |

```bash
uv run experiments/retrieval-followups/retrieval_followups.py bm25     # then onecall, filters, judged
uv run experiments/retrieval-followups/retrieval_followups.py report > experiments/retrieval-followups/results/retrieval_followups.txt
```

`report` compares `onecall` with the latest stored Voyage rerank on `esci-us-full`.

## Result

- **Title-only BM25** beats the main BM25 (title counted twice, plus body) by +0.020 nDCG@10 (+0.008 to +0.033), at 17 ms p50 and 43 ms p99 server time against about 100 and 300.
- **One request** with `rerank_by` on `multi_query`: wall time from 362 to 308 ms and +0.006 nDCG@10. With 100 candidates per subquery, +0.016 (+0.010 to +0.022).
- **Brand filter** on 81 queries: +0.047 nDCG@10 (+0.001 to +0.096), an upper bound. Server time barely moved, 79 to 77 ms.
- **Labeled products only:** Jev 0.868, Voyage 0.848, product-id order 0.679.

The +0.016 at 100 candidates already shows that more depth helps Voyage, which [`recall-fusion`](../recall-fusion/) builds on.
