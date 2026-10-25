# Plan 5: reranker ensemble and cascade

**Why customers care:** teams that already rerank want the same quality for
less money and latency, or more quality for the same. That's easy to put in
dollars.

## Dataset: ESCI, mostly from stored runs

The Voyage, Cohere, zerank-2, and Jev reranker runs are already in the
warehouse, with each product's score for the same hybrid top 50. The
ensemble costs no requests at all. Only the cascade with 100 candidates
needs new calls.

## Variants

| Variant | How | New requests |
| --- | --- | --- |
| Ensemble: mean rank | Average each product's rank under Voyage and Jev | none |
| Ensemble: score mix | Normalize each reranker's scores per query, then mix with weight w, fixed on the tuning half | none |
| Ensemble: all four | Mean rank across Voyage, Jev, Cohere, and zerank-2 | none |
| Cascade A | Voyage over the top 100, then Jev over Voyage's top 20 | 500 Voyage and 500 Jev requests |
| Cascade B | Jev over the top 100, then Voyage over Jev's top 20 | the same |
| Cheap first | Jev over the top 50, then Voyage only on queries where Jev isn't confident (its top two products' expected grades are close) | about half the Voyage requests |

"Cheap first" uses something only Jev offers, probabilities, to decide when
the second reranker is worth paying for. Report the share of queries sent
on, alongside nDCG.

## Metrics and decision

- nDCG@10 against Voyage alone, on the held-out half for anything with a
  tuned weight.
- p50 and p99 wall time, and cost per 1,000 queries.
- Plot quality against cost for every variant. A variant counts if it beats
  Voyage on quality at similar cost, or matches it within the interval at
  clearly lower cost.

## Note

The ensemble is run from the `results` table in DuckDB. It's mostly a SQL
recipe and could go in `docs/warehouse.md` as an example.
