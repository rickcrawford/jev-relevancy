# Constraint and negation checks

> **Premise:** Products that look right but miss one spec (wrong size, wrong brand, contains the excluded thing) are what users notice most. Checking each constraint in the query with a separate yes/no question catches them, and closes the gap between Exact and Substitute.

| Status | Dataset | Cost | Keys |
| --- | --- | --- | --- |
| Planned | ESCI, then NevIR and ExcluIR | About 500 Jev requests on ESCI; about 200 for NevIR | TypeSafe, OpenRouter |

**Why customers care:** these are the failures users notice and complain
about: the wrong size or brand at the top, or "without X" returning X. In
ESCI terms, this is the gap between Exact and Substitute.

## Datasets

| Dataset | Why | Size |
| --- | --- | --- |
| **ESCI** (primary) | Shows whether it matters on real shopping queries, next to the headline runs. The negation audit already found Jev handling "without" queries better than the labels. | 500 sampled queries. The parsing step counts how many have a constraint (brand, size, color, quantity, negation). |
| **NevIR** | A public leaderboard for negation. Each instance is two queries and two documents that differ by a negation, and a system scores only if it ranks both pairs correctly. Published: RankGPT with o3-mini 77.3%, GPT-4o 70.1%, most rerankers near chance. | 2,556 instances |
| **ExcluIR** | A public benchmark for "X but not Y" queries, scored by Recall@N and MRR. | 3,452 queries |

## Method

1. **Find each query's constraints.** One Jev `Choice` question per query
   about its type (brand, size, color, quantity, exclusion, none). The
   constraint values (the brand, "12 oz") come from a small LLM call or a
   few rules; Jev classifies but doesn't write text.
2. **Check each product.** For each candidate and each constraint, ask one
   `Noul`: "Does this product match {constraint}?" For an exclusion: "Does
   this product contain {X}?" All questions for one query fit in one Jev
   request.
3. **Combine.** Final score = reranker score × product of P(constraint met).
   Also test a softer version: subtract λ × P(failed) for each constraint.
   Fix λ on the tuning half.

## Variants on ESCI

Run each over the same hybrid top 50 as `06_rerank.py`:

| Variant | |
| --- | --- |
| `voyage` | stored run, the baseline |
| `jev` | stored run |
| `voyage+checks` | Voyage's score combined with the constraint probabilities |
| `jev+checks` | the same with Jev's score |
| `checks-only` | hybrid order, with failing products moved down |

## NevIR and ExcluIR

- Jev as a pointwise `Score` reranker, with the same rubric wording as
  ESCI's generic rubric.
- The same plus the `Noul` exclusion check.
- Report pairwise accuracy on NevIR and Recall@1 and MRR on ExcluIR,
  against the published numbers.

## Decision

- **ESCI:** `voyage+checks` minus `voyage`, nDCG@10, held-out half, with a
  lower bound above zero. Also break results down by constraint type, since
  it may only help some types.
- **NevIR:** at 70% or above, Jev matches GPT-4o at a fraction of the cost.
  Above 77.3%, it leads.

## Cost

- ESCI: about 500 Jev requests.
- NevIR: about 10k pairs, roughly 200 requests.
- ExcluIR: depends on corpus size, which isn't stated in the paper; check
  the repo before running.

## Sources

[NevIR](https://arxiv.org/abs/2305.07614) ·
[repo](https://github.com/orionw/NevIR) ·
[reproduction with current numbers](https://arxiv.org/abs/2502.13506) ·
[ExcluIR](https://arxiv.org/abs/2404.17288) ·
[repo](https://github.com/zwh-sdu/ExcluIR)

## Needs

- **Data:**
  - NevIR is on Hugging Face (`orionweller/NevIR`). Pairwise accuracy is a few lines, so its own conda environment isn't needed.
  - ExcluIR is a Google Drive folder with no licence file. ⚠️ Check its corpus size after downloading, and its terms before publishing.
- **Keys:** OpenRouter, to pull constraint values (the brand, "12 oz") out of queries.
