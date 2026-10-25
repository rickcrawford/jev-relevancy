# Query rewrites and document expansion

> **Premise:** Products that no retriever finds use different words from the shopper. Rewriting the query, or adding the queries a product would answer to its text, closes that vocabulary gap.

| Status | Dataset | Cost | Keys |
| --- | --- | --- | --- |
| Planned. Run only if `recall-fusion`'s diagnosis shows vocabulary misses | ESCI | LLM calls per query (rewrites) or per product (expansion) | turbopuffer, TypeSafe, OpenRouter |

**Why customers care:** when shoppers and listings use different words,
neither BM25 nor vector search finds the product. These fixes work, but
they cost money, document expansion especially.

**Run only if [`recall-fusion`](../recall-fusion/) says so:** if `recall_fusion.py diagnose` shows a
large share of Exact and Substitute products that no retriever finds in its
top 1,000, with low word overlap between the query and the title.

## Dataset: ESCI

Same queries and labels, so recall gains compare directly with [`recall-fusion`](../recall-fusion/)'s
numbers. The diagnosis output says which queries have vocabulary misses,
which gives a natural slice to report on.

## Part A: query rewrites (cheap)

1. An LLM writes 3 alternative phrasings per query, through OpenRouter
   with a small model and a prompt fixed in advance. Jev classifies but
   doesn't write text.
2. Run each rewrite as extra BM25 and vector subqueries in the same
   `multi_query` request, fused with RRF: 2 subqueries plus 6.
3. Variant with a Jev check: drop a rewrite when a `Noul` says "this
   rewrite asks for something different from the original query". This
   guards against drift, such as "apple charger" rewritten to fruit.

## Part B: document expansion (expensive)

1. Generate 5 queries each product would answer, with a small LLM, and
   store them in a new full-text field `expansion`.
2. Add BM25 over `expansion` as a third subquery.
3. **Run on the smoke namespace first** (21k products, one small-LLM call
   each). The full set is 482k calls; only do it
   if the smoke run shows a clear gain on the vocabulary-miss queries.

## Metrics and decision

- Recall@50 and Recall@100, and nDCG@10 before and after Voyage. Report on
  all queries and on [`recall-fusion`](../recall-fusion/)'s vocabulary-miss queries.
- Counts if Recall@100 gains with a lower bound above zero, and
  nDCG@10 after Voyage doesn't fall.
- Latency: rewrites add an LLM call (300 ms or more) before search. Report
  it, and test whether caching rewrites for repeated queries is realistic.

## Needs

- **Keys:** OpenRouter, for the rewrites and the expansion.
- ⚠️ Expansion adds a full-text field. Check whether a patch can add a full-text-indexed attribute to an existing namespace. If not, write the smoke namespace again with the field.
