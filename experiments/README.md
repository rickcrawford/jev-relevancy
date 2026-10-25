# Pick a test

Each folder here tests one premise about search, and says before it runs what result would count. Start from the question you have, build the base once, then run the study.

## Build the base first

Every study reads the full ESCI namespace and the main runs, so the quick start's 21k-product smoke namespace isn't enough. The one exception is `relevance-ontology`'s pilot, which can run on smoke.

```bash
./scripts/01_download_esci.sh
uv run scripts/02_prepare_esci.py                     # 500 sampled queries
uv run scripts/03_load_turbopuffer.py --scope full    # about 80M embedding tokens, 40+ minutes
uv run scripts/04_search.py --scope full
uv run scripts/06_rerank.py --model voyage/rerank-3
uv run scripts/06_rerank.py --model jev:jev-1.13.0
uv run scripts/08_judge_runs.py --judge jev-batched:jev-1.13.0
```

Studies open the warehouse read-only. Set `RELEVANCY_DB` to point them, and the main scripts, at a copy.

## Start from your question

**Can an automated grader stand in for human labels?** Start with [`judge-wands`](judge-wands/), which asks whether Jev's ESCI agreement holds on a second retailer. [`judge-trec`](judge-trec/) is the same claim on TREC, but its data is non-commercial. Four robustness checks are finished: [negation](judge-negation-audit/), [batch order](judge-batch-order/), [description length](judge-truncation/), and [Jev 1.14](judge-jev-1.14/).

**Am I losing relevant products before the reranker sees them?** [`recall-fusion`](recall-fusion/). Its diagnosis tells you whether the misses rank too low (fusion and depth fix that) or never show up at all, in which case try [`recall-rewrites`](recall-rewrites/).

**Will this change improve relevance?** Labels computed at index time ([`relevance-ontology`](relevance-ontology/), [`relevance-attributes`](relevance-attributes/)), per-constraint checks ([`relevance-constraints`](relevance-constraints/)), reranker cost ([`relevance-reranker-cascade`](relevance-reranker-cascade/)), a trained ranker ([`relevance-ltr`](relevance-ltr/)), and routing by query type ([`relevance-routing`](relevance-routing/)).

**What does turbopuffer do with my queries?** [`turbopuffer-perf`](turbopuffer-perf/) and [`retrieval-followups`](retrieval-followups/), both finished.

**How does Jev do on public benchmarks?** [`judge-trec`](judge-trec/), [`benchmark-instructions-stark`](benchmark-instructions-stark/), [`benchmark-bright`](benchmark-bright/). [`docs/benchmarks.md`](../docs/benchmarks.md) surveys the rest.

**Can Jev run LLM operators in SQL?** [`database-semantic-operators`](database-semantic-operators/).

## Every study

Ranked by how much the answer would matter to a team running search. That's my judgment, not customer data. Image search is out of scope.

| Rank | Study | Premise | Status | Dataset |
| --- | --- | --- | --- | --- |
| 1 | [`judge-wands`](judge-wands/) | Jev's agreement with human labels holds on another retailer's catalogue | Planned | WANDS |
| 2 | [`recall-fusion`](recall-fusion/) | Relevant products are retrieved but rank below 50; depth and fusion recover them | Written, not run | ESCI |
| 3 | [`relevance-ontology`](relevance-ontology/) | Matching the query's category to product categories improves hybrid search | Written, not run | ESCI, Google taxonomy ⚠️ |
| 4 | [`relevance-attributes`](relevance-attributes/) | A concept stored once per product ("for kids") fixes the queries that ask for it | Planned | ESCI |
| 5 | [`relevance-constraints`](relevance-constraints/) | Checking each query constraint separately closes the Exact/Substitute gap | Planned | ESCI, NevIR, ExcluIR ⚠️ |
| 6 | [`relevance-reranker-cascade`](relevance-reranker-cascade/) | Combining rerankers, or escalating only when unsure, matches Voyage for less | Planned | ESCI |
| 7 | [`judge-trec`](judge-trec/) | Jev orders systems like NIST's assessors, as well as GPT-4o judges | Planned | TREC DL, LLMJudge ⚠️ |
| 8 | [`database-semantic-operators`](database-semantic-operators/) | Jev makes LLM filters in SQL affordable at table scale | Planned | SemBench ⚠️, ESCI |
| 9 | [`relevance-ltr`](relevance-ltr/) | A trained ranker beats any single reranker, even trained on Jev's labels | Planned | ESCI train split |
| 10 | [`relevance-routing`](relevance-routing/) | Different query types want different retrieval settings | Planned | ESCI |
| 11 | [`recall-rewrites`](recall-rewrites/) | Rewrites and document expansion close vocabulary gaps | Planned, if `recall-fusion` says so | ESCI |
| 12 | [`benchmark-instructions-stark`](benchmark-instructions-stark/) | Per-condition checks let a cheap model follow multi-condition requests | Planned | STaRK ⚠️, FollowIR ⚠️ |
| 13 | [`benchmark-bright`](benchmark-bright/) | As a first pass, Jev keeps most of a reasoning reranker's gain for less | Planned | BRIGHT |
| | [`judge-negation-audit`](judge-negation-audit/) | On negation queries the human labels may be wrong | Done: mostly the labels | ESCI |
| | [`judge-batch-order`](judge-batch-order/) | Product order in a batched Jev request changes grades | Done: no | ESCI |
| | [`judge-truncation`](judge-truncation/) | The judge needs more of the description | Done: no | ESCI |
| | [`judge-jev-1.14`](judge-jev-1.14/) | Jev 1.14.0 judges and reranks better | Done: better judge, same reranker | ESCI |
| | [`turbopuffer-perf`](turbopuffer-perf/) | Where server time goes, and what ANN gives up | Done | ESCI |
| | [`retrieval-followups`](retrieval-followups/) | Four small retrieval changes move quality or latency | Done | ESCI |

*Written, not run* means the script exists and has only been checked on synthetic data. ⚠️ means the licence needs a look before you publish (see below).

If I had to pick two to run next, they'd be `recall-fusion` and `relevance-reranker-cascade`. Once the base exists, the first costs 1,500 queries and the second's ensembles cost nothing, and between them they decide whether `recall-rewrites` and `relevance-routing` are worth doing.

## What a study looks like

```
experiments/<group>-<name>/
  README.md     premise, method, how to run it, what counts, what it needs
  *.py          the script, if it has one
  results/      small CSVs and text it wrote, committed
```

Big intermediate files go to `data/<name>/`, which git ignores. [`TEMPLATE.md`](TEMPLATE.md) starts a new one.

## Rules every study follows

- **Settings are fixed before the run.** If a study picks among options, it picks on half the queries (split by a hash of the query id) and reports the other half.
- **Comparisons are paired** over the same queries, with bootstrap intervals from `relevancy.metrics.paired_delta`. A gain counts when the lower bound is above zero.
- **Power is stated.** On 500 queries the smallest difference we can detect with 80% power is 0.012 to 0.019 nDCG@10 (§5 of [`records/evidence.txt`](records/evidence.txt)). On a 250-query held-out half it's about 0.017 to 0.027.
- **Unlabeled results are accounted for.** ESCI labels about 20 products a query, and standard nDCG scores every other product as irrelevant. Report condensed nDCG next to it, and Jev-filled nDCG where it matters. When Jev is what's being tested, the human labels stay primary.
- **Main runs stay untouched.** A study that changes a namespace (like `relevance-ontology`'s patch) says so.
- **Cost and latency go next to quality**, measured the same way for every variant.

## Keys and licences

You need `TURBOPUFFER_API_KEY` for anything that searches, `TYPESAFE_API_KEY` for Jev, and `OPENROUTER_API_KEY` for studies that need an LLM to write text or label ground truth. Each README lists the rest under **Needs**.

| Data | Licence (checked October 2026) |
| --- | --- |
| ESCI | Apache-2.0 |
| WANDS | MIT |
| BRIGHT | CC BY 4.0 for the benchmark; the StackExchange text under it is CC BY-SA |
| Google Product Taxonomy | Google's terms, no open licence stated ⚠️ |
| TREC DL 2019/2020, LLMJudge | MS MARCO terms: "non-commercial research purposes only" ⚠️ |
| FollowIR | Text from licensed news collections (Robust04, NYT, Washington Post) ⚠️ |
| ExcluIR, SemBench | No licence file ⚠️ |
| STaRK | Code MIT; the Amazon data's terms aren't stated ⚠️ |
| NevIR | Code MIT, built on CondaQA (Apache-2.0) |

[`NOTICE`](../NOTICE) has the attributions.

## Recomputing the documented numbers

```bash
uv run experiments/evidence.py | tee experiments/records/evidence.txt
```

It recomputes every number in `docs/results.md` from the warehouse and the finished studies. [`records/`](records/) also keeps two logs the warehouse doesn't: the first full load, and the judge bake-off's latency lines.
