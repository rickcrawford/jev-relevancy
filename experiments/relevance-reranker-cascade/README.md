# Reranker ensemble and cascade

> **Premise:** Voyage and Jev make different mistakes. Combining them gets more quality, and sending a query to the second reranker only when the first is unsure gets the same quality for less.

| Status | Dataset | Cost | Keys |
| --- | --- | --- | --- |
| Ready, not yet run | ESCI | ensembles: nothing new; cascades: 500 Voyage reranks and about 1,500 Jev requests | TypeSafe, turbopuffer |

## Variants

The ensembles reuse the stored reranks of the hybrid top 50, so they compare with stored Voyage at 50. The cascades need new requests at 100, so they compare with Voyage at 100, which [`retrieval-followups`](../retrieval-followups/) found worth +0.016 over 50.

| Variant | How | Baseline | Calls a query |
| --- | --- | --- | --- |
| Mean rank | average of Voyage's and Jev's ranks (and all four stored rerankers, if present) | Voyage, top 50 | every reranker combined |
| Score mix | per-query normalized scores, weight picked on the tuning half from 0.1 to 0.9 | Voyage, top 50 | Voyage and Jev |
| Cascade | Voyage over the top 100, then Jev over Voyage's top 20 | Voyage, top 100 | 1 Voyage, 1 Jev |
| Escalate when unsure | Jev over the top 100; Voyage only when Jev's top two expected grades are within a margin picked on the tuning half | Voyage, top 100 | 2 Jev, plus Voyage on some queries |

## What counts

Two kinds of win, each fixed in advance, with intervals Bonferroni-corrected for the number of variants in each group:

- **Better:** held-out nDCG@10 above Voyage's, lower bound above zero.
- **Same quality, cheaper:** an equivalence test (TOST) within ±0.01 nDCG@10, and a lower serving cost a query. Cost needs prices: set `VOYAGE_PRICE` (dollars a rerank call) and `JEV_PRICE` (dollars a Jev request). Without them the report gives call counts and won't call anything cheaper. The cascade always pays for Voyage and a Jev call, so it can only be better, never cheaper.

Every reranker is called one query at a time from the same client, and latency is the wall time each call adds.

## Run

```bash
uv run relevancy run relevance-reranker-cascade          # dry run
uv run relevancy run relevance-reranker-cascade --yes
```

Run `ensembles` first; it costs nothing, and if it already answers the question, stop there. The paid steps save each query as it finishes, so a rerun resumes.

## Needs

The base, nothing else.
