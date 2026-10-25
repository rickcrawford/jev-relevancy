# Test plan: recall diagnosis and fusion tuning

**Status: proposed, not run.** `recall_fusion.py` implements it. Nothing
below is a result.

## The questions

1. Of the products humans rated Exact or Substitute, how many does first-stage
   search miss, and is each one missed because it ranks too low (a depth and
   fusion problem) or because nothing finds it at all (a vocabulary problem)?
2. How much of that can fusion settings get back, before any reranker: which
   lists to fuse, how deep, RRF k, and how much weight the vector list gets?
3. Does the extra recall turn into nDCG@10 once Voyage reranks the candidates?

The answers decide what to try next. A large "not found" share points to
query rewrites or document expansion. A large 51 to 1,000 share points to
deeper candidate sets and better fusion, which are much cheaper.

## Why fuse offline

`fetch` pulls each list's top 1,000 once (1,500 requests). Every fusion
setting is then computed from those lists locally, so 180 settings cost no
more requests than one. `check` makes sure this matches the server: the
local version of the stored hybrid-rrf (RRF, k=60, top 50 of vector search
and BM25) should match the stored run's top 50 almost exactly. Turbopuffer's
RRF may break ties differently or count ranks from 0, so a top-50 overlap
of about 0.95 or more is good enough, and below that the sweep's numbers
don't describe the server.

turbopuffer's RRF takes a weight per subquery and a rank constant
(`rerank_by=("RRF", {"weights": [...], "rank_constant": k})`), so any
setting the sweep picks can run on the server as is. Run the winner there
once to confirm it matches.

## Pre-registered settings

- The grid, fixed now: lists ann+bm25, ann+title, ann+bm25+title; depth 50,
  100, 200, 1,000 per list; RRF k 20, 60, 120; vector weight 0.3 to 0.7. That
  is 180 settings.
- The 500 sampled queries split in two by a hash of the query id. Settings
  are ranked on the tuning half; only the held-out half's numbers count.
- Baseline: ann+bm25, depth 50, k=60, equal weights, which is hybrid-rrf.

## Comparisons and decision rules

| | Comparison | Counts as a win if |
| --- | --- | --- |
| **Primary** | Best setting by tuning-half nDCG@10 minus baseline, held-out nDCG@10 | Lower bound above zero, and condensed nDCG@10 not clearly negative |
| Candidate recall | Best setting by tuning-half Recall@100 minus baseline, held-out Recall@100 | Lower bound above zero |
| **Does it survive reranking** | Voyage over the best recall setting's top 100 minus Voyage over the baseline's top 50, held-out nDCG@10 | Lower bound above zero. `baseline top 100` separates the effect of depth alone from the effect of better fusion. |

On the held-out half, about 250 queries, intervals will be roughly 1.4×
wider than on all 500, so gains under about +0.015 nDCG@10 won't separate
from zero. A smaller gain that holds on both halves is still worth noting,
but call it unconfirmed.

## What diagnose reports

- Each Exact and Substitute product's best rank under any of the three
  lists, in buckets: top 10, 11 to 50, 51 to 100, 101 to 1,000, not found.
- Per bucket, the share of query words that appear in the product title.
  If products nothing finds have much lower overlap, that's vocabulary
  mismatch, and query rewrites or document expansion are next. If overlap
  is the same, the misses are something else, such as labels for products
  the query doesn't really describe.
- Products only one list finds in its top 50: what each list adds.
- Recall at 50, 100, 200, and 1,000 for each list and for all three
  together. The gap between "any list at 1,000" and hybrid at 50 is the most
  any fusion setting could recover.

## Threats to validity

- **Unlabeled products.** Deeper fusion brings in products ESCI never
  labeled, which count as irrelevant. Condensed nDCG@10 shows how much that
  matters. For a fuller check, grade the unlabeled top-10 products with Jev,
  as the ontology plan does.
- **Labels drawn from one system.** ESCI's judged products came from
  Amazon's own search, so a product no list finds may still be relevant but
  out of reach for any lexical or vector method. That's a property of the
  labels, not a recall bug.
- **Picking from 180 settings.** The split guards against choosing a lucky
  setting, but only one held-out half exists. Before adopting a setting,
  rerun it on a fresh draw of ESCI test queries.
- **Latency.** Depth 1,000 costs server time. `fetch` records each list's
  server time at depth 1,000, so the cost of the chosen depth is on record.

## Running it

```bash
uv run experiments/recall_fusion.py fetch      # 1,500 queries, a few minutes
uv run experiments/recall_fusion.py check      # local RRF against the stored hybrid run
uv run experiments/recall_fusion.py diagnose
uv run experiments/recall_fusion.py sweep
uv run experiments/recall_fusion.py rerank     # 1,500 Voyage reranks
```

The ranked lists, about 1.5M rows, go to `data/recall/lists.parquet`
(git-ignored). Everything else goes to `experiments/results/recall_*.csv`.
