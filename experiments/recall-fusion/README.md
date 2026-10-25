# Recall diagnosis and fusion tuning

> **Premise:** many of the products humans rate relevant are retrieved, but below the top 50, so the reranker never sees them. Depth and fusion settings get them back cheaply, and the gain survives reranking.

| Status | Dataset | Cost | Keys |
| --- | --- | --- | --- |
| Ready, not yet run | ESCI, `esci-us-full` | 1,500 searches for `fetch`, 750 Voyage reranks; the sweep is offline | turbopuffer |

`fetch` pulls each list's top 1,000 once: vector search, the main BM25, and title-only BM25. Everything after it is computed locally, so 180 fusion settings cost no more requests than one.

```bash
uv run relevancy run recall-fusion --yes          # every step, or:
uv run experiments/recall-fusion/recall_fusion.py fetch      # resumes by list if it stops
uv run experiments/recall-fusion/recall_fusion.py check      # local RRF against the stored hybrid run
uv run experiments/recall-fusion/recall_fusion.py diagnose
uv run experiments/recall-fusion/recall_fusion.py sweep
uv run experiments/recall-fusion/recall_fusion.py rerank
```

## Method

- **`check`** fuses the stored lists the way turbopuffer does and compares them with the stored hybrid run. Below about 0.95 top-50 overlap, the sweep isn't describing the server, so the step stops the run; find out why before going on. turbopuffer's RRF takes per-subquery `weights` and a `rank_constant`, so any setting the sweep picks can run on the server as is.
- **`diagnose`** buckets every Exact and Substitute product by its best rank under any list (top 10, 11 to 50, 51 to 100, 101 to 1,000, not found), with the share of query words in its title. Recall is averaged per query, as the stored 0.471 is.
- **`sweep`** runs a fixed grid: which lists, depth 50 to 1,000, RRF k of 20, 60, or 120, and the vector list's weight. It picks a winner on half the queries by nDCG@10, condensed nDCG, and Recall@100, and reports each on the other half.
- **`rerank`** runs Voyage on the held-out half only, over three candidate sets: the baseline's top 50, the baseline at depth 100, and the best recall setting's top 100. The middle one separates the effect of depth from the effect of better fusion. [`retrieval-followups`](../retrieval-followups/) already found depth 100 worth +0.016 with Voyage, so what this adds is whether better fusion beats depth alone.

## What counts

- **Primary:** Voyage over the best setting's top 100 against Voyage over the baseline's top 100, held-out nDCG@10. It counts if the lower bound is above zero, and condensed nDCG agrees in sign.
- About 250 held-out queries can only detect a difference of about 0.017 to 0.027.
- If most misses land in "not found" with low title overlap, the fix is vocabulary, and [`recall-rewrites`](../recall-rewrites/) is next.

## Could mislead

- Deeper settings surface products ESCI never labeled, and standard nDCG scores them as irrelevant. That's why the sweep also picks on condensed nDCG.
- ESCI's labels come from what Amazon's search returned, so a product nothing finds may be out of reach for any lexical or vector method.
- `fetch` records latency at depth 1,000 only. Rerun the winning setting on the server to get its own latency.

## Needs

The base, nothing else. Its ranked lists (about 1.5M rows) go to `data/recall/`.
