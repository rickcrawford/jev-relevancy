# Plan 8: a second catalogue, WANDS

**Why customers care:** the first question from a prospect who isn't Amazon
is whether the ESCI result carries over to their catalogue. WANDS answers it
with a different retailer, category mix, and labeling process.

## Dataset: WANDS (Wayfair)

480 queries, 43k furniture and home products, 233k labels in three levels:
Exact, Partial, Irrelevant. Every query has many labeled products, so far
fewer results go unlabeled than on ESCI, and nDCG is less biased against
new retrievers. That makes WANDS a useful check on ESCI's numbers as well.

## Setup

1. **Rubric.** `rubric.py` requires exactly four levels. Either allow 2 to
   5 levels, so Jev's `Score` uses Wayfair's three, or keep four by mapping
   Exact to 3, Partial to 2, and Irrelevant to 0, with the 1 level unused.
   Allowing three levels is cleaner and makes the rubric match the labels.
   It needs small changes in `rubric.py`, `metrics.py` (gains), and the
   agreement code.
2. Convert WANDS's CSVs to `02_prepare_custom.py`'s inputs (docs parquet,
   queries csv, labels csv) with a small loader script.
3. Run the full pipeline: steps 03, 04, 06 (Voyage, Cohere, zerank-2, Jev),
   07, 08, then 05.

## What to report

- The ESCI results table again for WANDS: retrievers, rerankers, and
  nDCG@10 with intervals.
- Judge agreement with Wayfair's labels: exact, κ, and τ over the runs.
- Whether the reranker order and Jev's place in it match ESCI.

## Decision

This is a replication, not a test with a pass mark. Either outcome is
useful: matching ESCI strengthens the claim, and a different order is worth
knowing before a customer finds it.

## Cost

A few million embedding tokens to load (43k products), a few minutes under the default limit. 480 × 4 reranks. Jev
judging the full label set would be about 4.7k requests; the 200-pair
bake-off and step 8 cost far less.

## Source

[WANDS](https://github.com/wayfair/WANDS)
