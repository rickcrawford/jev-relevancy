# Constraint and negation checks

> **Premise:** the products users notice are the ones that look right but miss a spec: wrong size, wrong brand, or containing the thing the query excludes. Checking each constraint in the query with its own yes/no question catches them.

| Status | Dataset | Cost | Keys |
| --- | --- | --- | --- |
| Planned | ESCI; NevIR test split; ExcluIR ⚠️ | about 1,000 Jev requests on ESCI | TypeSafe, OpenRouter |

In ESCI terms, this is the gap between Exact and Substitute.

## Method

1. **Find the constraints.** A small LLM call lists each query's constraints with their values ("brand: Tascam", "size: 12 oz", "excludes: wood"). Jev classifies but doesn't write text, so it can't extract values.
2. **Check each product.** One `Noul` per candidate per constraint: "does this product match {constraint}?". Jev takes 50 questions a request, so a query with 50 candidates and two constraints is two requests.
3. **Combine.** The reranker's score is multiplied by the probability that each constraint is met. Pick the softer variant's penalty weight on the tuning half.

Run it over the stored hybrid top 50 with Voyage and with Jev as the base reranker, and compare with each reranker alone.

## What counts

- **ESCI:** `voyage+checks` against `voyage`, held-out nDCG@10, lower bound above zero. Break it down by constraint type. Negation queries are reported separately, because [`judge-negation-audit`](../judge-negation-audit/) found the human labels wrong on 74% of disagreements there, so human labels can't score them.
- **NevIR:** pairwise accuracy on the test split only, with a bootstrap interval, against the published test numbers (GPT-4o 70.1%, RankGPT with o3-mini 77.3%).
- **ExcluIR:** Recall@1 and MRR against the paper's baselines.

## Needs

- NevIR from Hugging Face (`orionweller/NevIR`). Pairwise accuracy is a few lines, so its own environment isn't needed.
- ExcluIR from Google Drive. It has no licence file, so check its terms before publishing, and check the corpus size after download.
