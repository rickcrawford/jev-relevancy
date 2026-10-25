# Query rewrites and document expansion

> **Premise:** products that no retriever finds use different words from the shopper. Rewriting the query, or adding the queries a product would answer to its text, closes that gap.

| Status | Dataset | Cost | Keys |
| --- | --- | --- | --- |
| Ready, not yet run | ESCI | 500 LLM calls for rewrites; one per product for expansion (482k on full) | turbopuffer, TypeSafe, OpenRouter |

Run this only if [`recall-fusion`](../recall-fusion/)'s diagnosis shows many Exact and Substitute products that nothing finds in 1,000, with under half the query's words in their titles.

## Part A: rewrites

1. An LLM (`REWRITE_MODEL`, default `openai/gpt-4o-mini`) writes three rephrasings per query, with the prompt fixed in advance. Each call is timed.
2. Each rephrasing adds a BM25 and a vector subquery to the same `multi_query`. RRF weights give the original query half the total, so six rewrite lists can't outvote its two.
3. A `checked` variant drops a rewrite when a Jev `Noul` says it asks for something different from the original.

## Part B: expansion

Five queries per product, indexed as a full-text `expansion` field, with BM25 over it as a third subquery. The smoke namespace is fine for checking the plumbing, but its gains are inflated (it holds every labeled product for the sampled queries), so only the full namespace decides. The LLM may also have seen ESCI's queries in training, which would flatter expansion; the report says so.

## What counts

Recall@50 rises with a lower bound above zero, and nDCG@10 after Voyage doesn't fall: its lower bound stays above -0.01 (fixed in advance). Reported on all queries and on the vocabulary-miss queries, with the rewrite call's p50 and p99 latency.

## Run

```bash
uv run relevancy run recall-rewrites                          # dry run
uv run relevancy run recall-rewrites --yes                    # full namespace
uv run relevancy run recall-rewrites --yes --scope smoke --from expand   # expansion plumbing only
```

Expansion runs eight LLM calls at a time and saves by chunk, so a rerun resumes and retries failures.

## Needs

`patch-expansion` adds a full-text attribute to the namespace with a patch. If turbopuffer refuses, the step says so, and the namespace has to be written again with the field.
