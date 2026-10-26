# Constraint and negation checks

> **Premise:** the products users notice are the ones that look right but miss a spec: wrong size, wrong brand, or containing the thing the query excludes. Checking each constraint in the query with its own yes/no question catches them.

| Status | Dataset | Cost | Keys |
| --- | --- | --- | --- |
| Ready, not yet run (ExcluIR manual) | ESCI; NevIR test split; ExcluIR ⚠️ | 500 LLM calls; about 1,000 Jev requests on ESCI, about 2 a pair on NevIR | TypeSafe, OpenRouter |

In ESCI terms, this is the gap between Exact and Substitute.

## Method

1. **Find the constraints.** An LLM (`CONSTRAINT_MODEL`, default `openai/gpt-4o-mini`) lists each query's constraints with their values ("brand: Tascam", "size: 12 oz", "excludes: wood"). Jev classifies but doesn't write text, so it can't extract values.
2. **Check each product.** One `Noul` per candidate per constraint: "does this product match {constraint}?", or for an exclusion, "does it contain it?". A query with 50 candidates and two constraints is two Jev requests.
3. **Combine.** The primary variant, fixed in advance: the reranker's score (normalized onto 0.1 to 1 per query) times every P(met). A penalty variant, with its weight picked on the tuning half, is reported as exploratory.

It runs over the stored hybrid top 50, with Voyage and with Jev as the base reranker.

## What counts

- **ESCI:** Voyage times the checks against Voyage alone, held-out nDCG@10, lower bound above zero. Queries with an exclusion are left out of this and reported on their own line: a one-rater audit ([`judge-negation-audit`](../judge-negation-audit/)) found the human labels wrong on most disagreements there, so they can't score exclusions reliably. The by-type breakdown is descriptive.
- **NevIR:** pairwise accuracy on the test split only, with a bootstrap interval, against the published test numbers (GPT-4o 70.1%, RankGPT with o3-mini 77.3%).
- **ExcluIR:** Recall@1 and MRR against the paper's baselines. Not automated; see Needs.

## Run

```bash
uv run relevancy run relevance-constraints          # dry run
uv run relevancy run relevance-constraints --yes
```

`check` and `nevir` save as they go, so a rerun resumes.

## Needs

- `uv sync --extra studies` for `nevir` (Hugging Face `datasets`).
- ExcluIR is a Google Drive folder with no license file and no documented format, so it's left out. Check its terms before publishing anything from it.
