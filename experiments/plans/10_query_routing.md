# Plan 10: query routing by query type

**Why customers care:** one fusion setting for all traffic is a compromise.
If brand-led queries want more BM25 and vague queries want more vector
search, routing gets a gain without new models. It's modest, and worth it
only after plans 2 and 4.

## Dataset: ESCI

Queries already carry a `slice` (length and brand), and every stored run is
in the warehouse. So the best retriever for each query type can be read off
existing results before anything new runs.

## Steps

1. **Is there anything to gain?** Using stored runs only, compute the best
   retriever for each slice (bm25, ann, hybrid-rrf), picked on the tuning
   half. Score that routing on the held-out half against always using
   hybrid. If even perfect slice routing gains under +0.01, stop.
2. **A Jev classifier.** One `Choice` question per query with types written
   in advance:
   - **brand or model:** names a brand or model number;
   - **spec-heavy:** sizes, quantities, materials;
   - **descriptive:** what it's for or looks like;
   - **vague or exploratory;**
   - **negation.**
3. **Route.** For each type, pick the fusion setting from plan 2's sweep
   grid, or the best retriever, on the tuning half. Apply it on the held-out
   half.

## Metrics and decision

- Routed minus hybrid, nDCG@10, held-out half.
- Also routed minus plan 2's single best setting. Routing has to beat the
  best single setting, not just the default.
- How often the type classifier agrees with the slice labels, and with a
  person on 100 queries.

## Cost

500 Jev questions (about 10 requests). Everything else is offline, using
plan 2's stored top-1,000 lists.
