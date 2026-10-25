# Reranker ensemble and cascade

> **Premise:** Voyage and Jev make different mistakes. Combining them gets more quality, and sending a query to the second reranker only when the first is unsure gets the same quality for less.

| Status | Dataset | Cost | Keys |
| --- | --- | --- | --- |
| Written, not run | ESCI | ensembles: no new requests; cascades: about 500 per reranker | TypeSafe, turbopuffer |

## Variants

The ensembles reuse the stored reranker scores for the hybrid top 50, so they compare with stored Voyage at top 50. The cascades need new requests at top 100, so they compare with Voyage at top 100, which [`retrieval-followups`](../retrieval-followups/) found worth +0.016 over 50.

| Variant | How | Baseline | Serving cost per query |
| --- | --- | --- | --- |
| Mean rank | average of Voyage's and Jev's ranks | Voyage, top 50 | Voyage plus Jev |
| Score mix | per-query normalized scores, mixed with a weight picked on the tuning half | Voyage, top 50 | Voyage plus Jev |
| Cascade | Voyage over the top 100, then Jev over Voyage's top 20 | Voyage, top 100 | Voyage plus a small Jev call |
| Escalate when unsure | Jev over the top 100; Voyage only when Jev's top two expected grades are within a margin picked on the tuning half | Voyage, top 100 | Jev, plus Voyage on some share of queries |

## What counts

Two kinds of win, each fixed in advance:

- **Better:** held-out nDCG@10 above Voyage's, lower bound above zero.
- **Same quality, cheaper:** an equivalence test (TOST) with a margin of ±0.01 nDCG@10, plus a lower serving cost per query. A plain non-significant difference isn't enough, since 250 queries can't rule out much.

Measure latency the same way for every variant: wall time from one client, each reranker call timed in the same loop. Jev is called directly and Voyage through turbopuffer, so compare the wall time each adds, not server time.

## Run

```bash
uv run relevancy show relevance-reranker-cascade                # the steps, and which are paid
uv run relevancy run relevance-reranker-cascade            # dry run
uv run relevancy run relevance-reranker-cascade --yes
```

The ensembles run first and cost nothing; if they already answer the question, stop there.

## Needs

The base, nothing else.
