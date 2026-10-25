# Where else Jev could be tested: public benchmarks

The ESCI result shows Jev two ways: as a reranker it ties Voyage, and as a
judge it agrees with humans more than any LLM tested. The benchmarks below
would test the same claims on public leaderboards, or test something the
ESCI setup can't. They're ranked by how well they fit Jev's typed questions
(`Score`, `Choice`, `Noul`, image input) and how credible a result would be.

Compiled in October 2026 from papers and repos found by web search. Every
benchmark below links to its paper or repo. Baseline numbers come from the
papers and summaries of them; check each against the source before citing it.

## Jev as a relevance judge

**1. LLMJudge (SIGIR 2024 LLM4Eval) and UMBRELA on TREC DL 2019–2023 and TREC RAG.**
LLM relevance labels on a 0–3 scale, scored by Cohen's κ against NIST
assessors and by Kendall's τ of the system rankings they produce. UMBRELA,
the open reproduction of Bing's GPT-4o assessor, reports τ of about 0.87 to
0.94 on DL19 to DL23, and 0.89 on 77 TREC RAG 2024 runs. Its per-label
accuracy is weak: about 30% on "highly relevant". LLMJudge has 42
submitted label sets to compare against. Jev plugs in as a 4-level `Score`,
which is what 07 and 08 already do. Report κ on its top grade, τ from its
expected grade, and calibration error, which judge papers rarely report and
Jev's probabilities make possible. A few thousand to tens of thousands of
pairs, so cents to dollars.
[LLMJudge](https://arxiv.org/abs/2408.08896) ·
[repo](https://github.com/llm4eval/LLMJudge) ·
[UMBRELA](https://arxiv.org/abs/2406.06519) ·
[repo](https://github.com/castorini/umbrela) ·
[TREC RAG 2024 judging](https://arxiv.org/abs/2411.08275) ·
[the case against LLM judges](https://arxiv.org/abs/2412.17156)

*Why first:* it's the same experiment as 07 and 08 on a corpus the IR
community already trusts, and the cheapest item here.

## Jev as a reranker, where cross-encoders fail

**2. Negation and exclusion: NevIR and ExcluIR.** In NevIR, each instance is
two queries and two documents that differ by a negation, and a system is
right only if it ranks both pairs correctly. Published pairwise accuracy:
RankGPT with o3-mini 77.3%, Rank1-32B and GPT-4o 70.1%, most rerankers near
chance. ExcluIR has 3,452 "X but not Y" queries. Test Jev two ways: as a
pointwise `Score` reranker, and with a `Noul` "does this document contain
the excluded thing?" folded into the score. The negation audit here already
found Jev handling ESCI's "without" queries better than the labels did.
About 10k pairs.
[NevIR](https://arxiv.org/abs/2305.07614) ·
[repo](https://github.com/orionw/NevIR) ·
[ExcluIR](https://arxiv.org/abs/2404.17288) ·
[repo](https://github.com/zwh-sdu/ExcluIR)

**3. Instruction-following retrieval: FollowIR, InfoSearch, BIRCO.**
- FollowIR measures whether a reranker changes its ranking when the
  instruction changes (p-MRR); most models score zero or below. Put the
  instruction in Jev's `state`.
- InfoSearch asks for documents with given properties, such as audience,
  format, language, or source. That's one `Noul` per property.
- BIRCO has 50 to 100 candidates per query, so it's cheap.

[FollowIR](https://arxiv.org/abs/2403.15246) ·
[repo](https://github.com/orionw/FollowIR) ·
[InfoSearch](https://arxiv.org/abs/2410.23841) ·
[BIRCO](https://arxiv.org/abs/2402.14151)

**4. More product search: WANDS and SQID.**
- **WANDS** (Wayfair): 480 queries, 43k products, 233k labels in three
  levels (Exact, Partial, Irrelevant). It loads through `02_prepare_custom.py`
  with a 3-level rubric, so the whole ESCI study can be repeated on it.
  Rubrics currently need exactly 4 levels, so this needs a small change to
  `rubric.py`.
- **SQID** adds product images to ESCI. That tests Jev with image and text
  together, against text alone, on the same queries as the headline result.

[WANDS](https://github.com/wayfair/WANDS) ·
[SQID](https://arxiv.org/abs/2405.15190)

**5. Semi-structured queries: STaRK.** The queries combine text with
relational constraints ("a sunscreen by a brand that also makes X").
Verifying one constraint at a time with `Noul` is the same idea as
constraint checking on ESCI.
[STaRK](https://github.com/snap-stanford/stark)

**Lower fit: BRIGHT.** Reasoning-heavy retrieval, where the best systems
reach about 41 to 46 nDCG@10. Jev is unlikely to lead it, but it would give
a cost-against-quality point, or work as a cheap filter ahead of an
expensive reasoning reranker.
[BRIGHT](https://arxiv.org/abs/2407.12883)

## Jev as a database operator

Semantic query engines add language-model operators to SQL: filter by a
natural-language condition, classify, rank, join. Jev's question types map
onto them directly: `Noul` is a filter, `Choice` a classifier, `Score` a
ranker. Engines that do this today include LOTUS, Palimpzest, BigQuery's
`AI.IF` and `AI.CLASSIFY`, Snowflake's `AI_FILTER`, and FlockMTL for DuckDB.

**6. SemBench (VLDB 2026).** 55 queries in 5 scenarios, one of them Amazon
products, over text, images, and audio. Most of them are filters, plus
joins, classifications, and rankings. Each system is scored on quality,
dollar cost, and latency, with BigQuery, LOTUS, Palimpzest, and ThalamusDB
as published baselines. Adding a system means writing one runner class.
Jev's cost and latency are where it would stand out.
[paper](https://arxiv.org/abs/2511.01716) ·
[repo](https://github.com/skddj/sembench) ·
[PVLDB](https://www.vldb.org/pvldb/vol19/p1754-trummer.pdf)

**7. Semantic-filter cascades.** To run a yes/no condition over a large
table, a cheap model handles the clear cases and an expensive LLM handles
the rest. The aim is a guaranteed precision or recall at the fewest
expensive calls. Current methods train a proxy model per query:
- LOTUS's approximation cascades;
- ScaleDoc;
- EPFL's June 2026 method, which trains its proxy on the LLM's own
  confidence;
- BigQuery's and Snowflake's optimized modes.

A calibrated `Noul` probability is exactly the signal a cascade needs, and
Jev needs no per-query training. Compare expensive calls saved and how often
the guarantee is missed, on the papers' own datasets.
[LOTUS](https://arxiv.org/abs/2407.11418) ·
[cascades docs](https://lotus-ai.readthedocs.io/en/latest/approximation_cascades.html) ·
[ScaleDoc](https://arxiv.org/abs/2509.12610) ·
[EPFL](https://arxiv.org/abs/2606.08090) ·
[BigQuery](https://docs.cloud.google.com/bigquery/docs/optimize-ai-functions) ·
[Snowflake](https://docs.snowflake.com/en/release-notes/2025/other/2025-09-23-ai-filter-optimization)

**8. TAG-Bench.** 80 questions over tables that need both SQL and judgment.
Plain text-to-SQL answers 20% or fewer correctly, and hand-written LOTUS
pipelines do much better. Jev would supply their filter and ranking steps.
[TAG-Bench](https://github.com/TAG-Research/TAG-Bench)

Capital One's "Large Databases Need Small, Open-Weight Language Models"
(VLDB 2026 workshop) argues the same case from the other side: commercial
LLM APIs make these operators too expensive at database scale.
[paper](https://arxiv.org/abs/2606.31808)

## Turbopuffer specifically

Filtered vector search benchmarks, such as the NeurIPS 2023 big-ANN filter
track, only use filters on exact labels, never semantic ones. Jev's
place there is offline: compute semantic labels once (`Choice`, `Noul`),
store them as filterable attributes, and filter on them at query time.
`ontology_search.py` does that with categories. The same pattern works for
"suitable for kids", "contains the excluded ingredient", or "is a
replacement part".

## What to run first

1. **LLMJudge and UMBRELA qrels:** κ, τ, and calibration against GPT-4o
   UMBRELA and the 42 LLMJudge label sets. Cheapest, and closest to what
   this repo already shows.
2. **NevIR and ExcluIR:** pairwise accuracy against the published 46% to 77%,
   with and without the `Noul` exclusion check.
3. **WANDS:** the full ESCI study on a second product catalogue, as judge
   and as reranker.
4. **SemBench:** a Jev runner for the filter, classify, and rank queries,
   aimed at a database venue.
5. **Filter cascades:** Jev's `Noul` confidence as the cheap first stage,
   against trained proxies.
