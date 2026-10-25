# Query rewrites and document expansion

> **Premise:** products that no retriever finds use different words from the shopper. Rewriting the query, or adding the queries a product would answer to its text, closes that gap.

| Status | Dataset | Cost | Keys |
| --- | --- | --- | --- |
| Written, not run | ESCI | an LLM call per query (rewrites) or per product (expansion) | turbopuffer, TypeSafe, OpenRouter |

Run this only if [`recall-fusion`](../recall-fusion/)'s diagnosis shows many Exact and Substitute products that nothing finds in 1,000, with low title overlap.

## Part A: rewrites

1. A small LLM writes three rephrasings per query, with the prompt fixed in advance.
2. Each rephrasing adds a BM25 and a vector subquery to the same `multi_query`. RRF weights give the original query half the total weight, so six rewrite lists can't outvote two original ones.
3. A variant drops a rewrite when a Jev `Noul` says it asks for something different from the original.

## Part B: expansion

Generate five queries per product, index them as a new full-text field, and add BM25 over it as a subquery. The smoke namespace is fine for checking the plumbing, but its gains are inflated (it holds every labeled product for the sampled queries), so only the full namespace decides. The LLM may also have seen ESCI's queries in training, which would flatter expansion. Report that caveat.

## What counts

Recall@50, since that's what the reranker sees, and nDCG@10 after Voyage. It counts if Recall@50 rises with a lower bound above zero and nDCG@10 after Voyage doesn't fall. Also report the latency the LLM call adds before search.

## Run

```bash
uv run relevancy show recall-rewrites                # the steps, and which are paid
uv run relevancy run recall-rewrites --scope smoke            # dry run
uv run relevancy run recall-rewrites --scope smoke --yes
```

Run it only if `recall-fusion`'s diagnosis found vocabulary misses. `--scope smoke` checks the expansion plumbing cheaply; only `--scope full` decides.

## Needs

Check whether a patch can add a full-text-indexed attribute. If not, the namespace has to be written again with the new field.
