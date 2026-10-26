# Public benchmarks for Jev

On ESCI, Jev ties Voyage as a reranker and agrees with the human labels more than any LLM I tested. These public benchmarks would test the same claims somewhere else, or test things ESCI can't. Each one that's worth running has a study in [`experiments/`](../experiments/README.md), which has the method and what would count.

I put this together in October 2026 from papers and repos. The baseline numbers come from those papers; check them against the source before you cite one.

## As a judge

**TREC DL and LLMJudge** ([`judge-trec`](../experiments/judge-trec/)). The standard test for LLM relevance judges is whether their labels order TREC systems the way NIST's do. UMBRELA, an open version of Bing's GPT-4o judge, reports Kendall's tau of 0.87 to 0.94 on DL19 to DL23, but gets "perfectly relevant" right only about 30% of the time. LLMJudge collected 42 label sets from eight teams on DL23. The data is MS MARCO, which is licensed for non-commercial research only. [UMBRELA](https://arxiv.org/abs/2406.06519), [LLMJudge](https://arxiv.org/abs/2408.08896), [the 42 label sets](https://arxiv.org/abs/2502.13908), [a critique of LLM judges](https://arxiv.org/abs/2412.17156).

**WANDS** ([`judge-wands`](../experiments/judge-wands/)). Wayfair's 480 queries, 43k products, and 233k labels make a second catalog with an MIT license. [Repo](https://github.com/wayfair/WANDS).

## As a reranker

**NevIR and ExcluIR** ([`relevance-constraints`](../experiments/relevance-constraints/)). NevIR pairs two queries and two documents that differ by a negation, and a system only scores if it ranks both right. On the test split, GPT-4o gets 70.1% and RankGPT with o3-mini 77.3%, and most rerankers sit near chance. ExcluIR has 3,452 "X but not Y" queries. [NevIR](https://arxiv.org/abs/2305.07614), [the reproduction with current numbers](https://arxiv.org/abs/2502.13506), [ExcluIR](https://arxiv.org/abs/2404.17288).

**STaRK and FollowIR** ([`benchmark-instructions-stark`](../experiments/benchmark-instructions-stark/)). STaRK-Amazon's product queries mix text with relational conditions. FollowIR checks whether a reranker changes its ranking the right way when the instruction changes, and most rerankers don't. InfoSearch and BIRCO test similar things. [STaRK](https://github.com/snap-stanford/stark), [FollowIR](https://arxiv.org/abs/2403.15246), [InfoSearch](https://arxiv.org/abs/2410.23841), [BIRCO](https://arxiv.org/abs/2402.14151).

**BRIGHT** ([`benchmark-bright`](../experiments/benchmark-bright/)). Queries that need reasoning to find the right document. The best systems reach 41 to 46 nDCG@10 averaged over 12 domains. Jev won't lead here; the question is whether it makes a cheap first pass in front of a reasoning model. [Paper](https://arxiv.org/abs/2407.12883).

## As a database operator

Semantic query engines add LLM operators to SQL: filter on a natural-language condition, classify, rank. Jev's `Noul`, `Choice`, and `Score` map onto them. LOTUS, Palimpzest, BigQuery's `AI.IF`, Snowflake's `AI_FILTER`, and FlockMTL for DuckDB all do this today ([`database-semantic-operators`](../experiments/database-semantic-operators/)).

**SemBench** (VLDB 2026) has 55 queries in five scenarios over text, images, and audio, scored on quality, dollars, and latency against BigQuery, LOTUS, Palimpzest, and ThalamusDB. Its e-commerce scenario is a 44k-item Kaggle fashion catalog. [Paper](https://arxiv.org/abs/2511.01716), [repo](https://github.com/skddj/sembench).

**Filter cascades** let a cheap model decide the clear cases and send the rest to an LLM. ScaleDoc, LOTUS, and a June 2026 EPFL paper all train the cheap model per condition. Capital One's VLDB 2026 workshop paper makes the cost case: commercial APIs ran one set of its experiments past $10,000. [ScaleDoc](https://arxiv.org/abs/2509.12610), [LOTUS](https://arxiv.org/abs/2407.11418), [EPFL](https://arxiv.org/abs/2606.08090), [Capital One](https://arxiv.org/abs/2606.31808).

## On turbopuffer

Filtered vector search benchmarks, like the NeurIPS 2023 big-ANN filter track, filter on exact labels. Jev fits before the filter: label documents once with a `Choice` or `Noul`, store the label as an attribute, and filter on it at query time. [`relevance-ontology`](../experiments/relevance-ontology/) does that with categories and [`relevance-attributes`](../experiments/relevance-attributes/) with yes/no concepts.
