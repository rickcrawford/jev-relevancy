# Follow-up experiments

Each script here tests one thing the main walkthrough raised. They read the
warehouse read-only and write their results to `experiments/results/`, so
they never change a run or a judgment the docs report.

| Script | What it tests | Result |
| --- | --- | --- |
| `negation_audit.py` | Whether Jev or the human labels are wrong on "without" and "not" queries | Mostly the labels. Of 43 disagreements on real negations, the product had the excluded thing in 32, Jev was wrong on 4, and 7 were unclear. |
| `batch_order.py` | Whether the order of products in a batched Jev request changes its grades | No. Regrading 3,177 pairs matched the stored grade 97.7% of the time in the same order and 97.3% reversed. |
| `truncation.py` | Whether showing the judge more of the product description changes agreement with humans | No. 800 characters: exact 0.600, kappa 0.410. 4,000 characters: 0.606 and 0.406, on 1,091 pairs. |
| Jev 1.14.0 (main scripts) | The newer Jev as a reranker and as a judge | Same reranker (-0.001 nDCG@10, interval -0.007 to +0.004). Better judge: +0.031 exact agreement on the same 3,692 pairs (interval +0.019 to +0.045), kappa 0.471 against 0.436, same tau. |
| `turbopuffer_perf.py` | Where turbopuffer's server time goes, cache temperature, result stability, and ANN recall | Of ANN's 76 ms median server time, 66 ms is embedding the query text. BM25 and hybrid run about 103 ms at p50 and 320 ms at p99. Every query reported a hot cache, and repeating the 500 queries gave the same top 10 for 100% of BM25 queries and 99.4% of ANN and hybrid queries. ANN recall against exhaustive search is 0.994 at 50 and 0.995 at 10 (100 vectors sampled from the namespace). 9,785 documents (2%) were still unindexed and scanned exhaustively on every query. |
| `retrieval_followups.py` | BM25 field variants, hybrid and reranking in one request, a brand filter, and Elastic's judged-only setup | Title-only BM25: 17 ms p50, 43 ms p99, nDCG@10 +0.020. One request with `rerank_by` on `multi_query`: 362 to 308 ms wall, +0.006 nDCG@10 (+0.016 with 100 per subquery). Brand filter on 81 queries: 79 to 77 ms, +0.047 (an upper bound). Judged products only: Jev 0.868, Voyage 0.848, product-id order 0.679. Output in `results/retrieval_followups.txt`. |
| `ontology_search.py` | Whether tagging products and queries against the Google Product Taxonomy (vector search shortlist, Jev `Choice` picks) and boosting or filtering on the query's category improves hybrid search, before and after Voyage | Proposed, not run. Test plan and decision rules in [`ontology_search.md`](ontology_search.md). |
| `recall_fusion.py` | Where first-stage search loses Exact and Substitute products (ranked too low, or never found), and how much fusion settings and deeper candidates get back, before and after Voyage. Fuses stored top-1,000 lists offline. | Proposed, not run. Test plan and decision rules in [`recall_fusion.md`](recall_fusion.md). |
| `07_judge_bakeoff.py` (main scripts) | Judge latency | The script prints p50 latency but doesn't store it. The final run's lines are saved in `results/judge_bakeoff_latency.txt`: Jev 110 to 118 ms, the LLMs 435 to 656 ms. |
| `evidence.py` | The numbers the docs cite | Recomputes all of them; output in `results/evidence.txt`. |

The Jev 1.14.0 follow-up ran the main scripts against a copy of the
warehouse, so the main runs stay as they were:

```bash
cp warehouse/relevancy.duckdb /tmp/jev114.duckdb
export RELEVANCY_DB=/tmp/jev114.duckdb
uv run scripts/08_judge_runs.py --judge jev-batched:jev-1.13.0 --judge jev-batched:jev-1.14.0 --judge google/gemma-3-27b-it
uv run scripts/06_rerank.py --model jev:jev-1.14.0
uv run scripts/05_score.py --base jev-1.13.0-b53bce
```

Its grades and reranked results are saved in `results/jev114_judge.csv` and
`results/jev114_rerank.csv`, which `evidence.py` reads for section 15.

The negation audit was rated by a Claude model that saw only the query and
the product text, never either grade. One rater is thin; if you rerun it,
add a second and report where they disagree.
