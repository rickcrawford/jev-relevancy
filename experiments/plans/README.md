# Experiment plans, ranked by value to customers

One plan per idea, in the order of how much it would matter to a team
running search. Each one says what dataset it uses and why, what to run, and
what result would count. None of these has been run. Pick them up as needed.

**Default dataset: ESCI.** The warehouse, namespace, 500-query sample, and
stored runs are already built, so any result sits next to the headline
numbers. A plan only uses another dataset when ESCI can't answer the
question, or when a public leaderboard is the point.

The ranking is a judgment, not customer data. The top three make one pitch:
measure your search without labels, find where you lose results, and fix it
with attributes you compute once.

| # | Plan | Datasets | Status |
| --- | --- | --- | --- |
| 1 | [Jev as an eval judge: public proof](01_judge_trec.md) | ESCI (done), TREC DL 2019/2020, LLMJudge (TREC DL 2023) | Plan |
| 2 | [Recall diagnosis and fusion tuning](../recall_fusion.md) | ESCI | Script ready (`recall_fusion.py`) |
| 3 | [Semantic attributes at index time](03_semantic_attributes.md) (ontology search is the first case: [plan](../ontology_search.md)) | ESCI | Ontology script ready (`ontology_search.py`); other attributes are a plan |
| 4 | [Constraint and negation checks](04_constraints_negation.md) | ESCI, NevIR, ExcluIR | Plan |
| 5 | [Reranker ensemble and cascade](05_reranker_cascade.md) | ESCI (mostly stored runs) | Plan |
| 6 | [Semantic SQL operators and filter cascades](06_semantic_operators.md) | ESCI products, SemBench, ScaleDoc's collections | Plan |
| 7 | [Multimodal product search](07_multimodal_sqid.md) | SQID (ESCI with images) | Plan |
| 8 | [Second catalogue: WANDS](08_wands.md) | WANDS (Wayfair) | Plan |
| 9 | [Learning to rank](09_learning_to_rank.md) | ESCI train split | Plan |
| 10 | [Query routing by query type](10_query_routing.md) | ESCI | Plan |
| 11 | [Query rewrites and document expansion](11_rewrites_expansion.md) | ESCI | Plan, run only if plan 2 shows vocabulary misses |
| 12 | [Instruction-following and semi-structured retrieval](12_instructions_stark.md) | FollowIR, InfoSearch, STaRK-Amazon | Plan |
| 13 | [Reasoning retrieval: BRIGHT](13_bright.md) | BRIGHT, a few domains | Plan, lowest priority |

[`requirements.md`](requirements.md) lists what each plan needs before it
can run: keys, data access, licences, code changes, and blockers. Read it
first. TREC DL and LLMJudge are licensed for non-commercial research only,
and ScaleDoc's workloads don't appear to be public.

## Rules every plan follows

- **Fix the settings before looking at results.** If a plan picks among
  options (thresholds, weights, prompts), pick on half the queries, chosen
  by a hash of the query id, and report the other half.
- **Compare with paired bootstrap intervals** (`relevancy.metrics.paired_delta`)
  on the same queries. A gain counts when the interval's lower bound is
  above zero.
- **Account for unlabeled results.** A new retriever that finds products
  ESCI never labeled is scored down for it. Report condensed nDCG@10, and
  Jev-filled nDCG@10 where it matters. Keep human labels as the primary
  metric, since Jev is also what's being tested.
- **Leave the main runs alone.** Experiments read the warehouse read-only,
  or use a copy (`RELEVANCY_DB=/tmp/copy.duckdb`), and write to
  `experiments/results/`.
- **Report cost and latency next to quality.**
