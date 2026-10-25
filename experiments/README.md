# Pick a test

The walkthrough in `scripts/` builds the base: a dataset in the warehouse,
documents in turbopuffer, first-stage runs, rerankers, and judges. Every
study here starts from that base, tests one premise, and says up front what
result would count. Start from the question you have.

## Start from your question

**Can I trust an automated grader for my search?**
Go to the `judge-` studies.
- Start with [`judge-wands`](judge-wands/) to see whether Jev's ESCI result
  holds on a second catalogue.
- [`judge-trec`](judge-trec/) for the public proof.
- The finished robustness checks: [`judge-negation-audit`](judge-negation-audit/),
  [`judge-batch-order`](judge-batch-order/), [`judge-truncation`](judge-truncation/),
  and [`judge-jev-1.14`](judge-jev-1.14/).

**Am I losing relevant results before the reranker sees them?**
Start with [`recall-fusion`](recall-fusion/). Its diagnosis says whether the
misses are ranked too low, which fusion fixes cheaply, or never found, in
which case go on to [`recall-rewrites`](recall-rewrites/).

**Will this change improve relevance?**
The `relevance-` studies:
- [`relevance-ontology`](relevance-ontology/) and
  [`relevance-attributes`](relevance-attributes/): labels computed at index
  time;
- [`relevance-constraints`](relevance-constraints/): specs and negation;
- [`relevance-reranker-cascade`](relevance-reranker-cascade/): cost against
  quality;
- [`relevance-multimodal`](relevance-multimodal/): images;
- [`relevance-ltr`](relevance-ltr/) and [`relevance-routing`](relevance-routing/).

**What does turbopuffer do with my queries?**
[`turbopuffer-perf`](turbopuffer-perf/) and [`retrieval-followups`](retrieval-followups/), both finished.

**How does Jev compare on public benchmarks?**
[`judge-trec`](judge-trec/), [`benchmark-instructions-stark`](benchmark-instructions-stark/),
and [`benchmark-bright`](benchmark-bright/). [`docs/benchmarks.md`](../docs/benchmarks.md)
surveys the field.

**Can Jev power LLM operators in SQL?**
[`database-semantic-operators`](database-semantic-operators/).

## Every study

Ranked by value to a team running search. The ranking is a judgment, not
customer data. The top three make one pitch: measure your search without
labels, find where you lose results, and fix it with attributes you compute
once.

| Rank | Study | Premise | Status | Dataset |
| --- | --- | --- | --- | --- |
| 1 | [`judge-trec`](judge-trec/) | Jev orders search systems like human assessors, as well as GPT-4o-class judges, for far less | Planned | TREC DL, LLMJudge ⚠️ licence |
| 2 | [`recall-fusion`](recall-fusion/) | Relevant products are retrieved but ranked below 50; fusion and depth recover them | **Ready** | ESCI |
| 3 | [`relevance-ontology`](relevance-ontology/) | Matching the query's category to products' categories improves hybrid search | **Ready** | ESCI + Google taxonomy |
| 3 | [`relevance-attributes`](relevance-attributes/) | Concepts computed once per product ("for kids") fix queries that ask for them | Planned | ESCI |
| 4 | [`relevance-constraints`](relevance-constraints/) | Checking each query constraint separately closes the Exact/Substitute gap | Planned | ESCI, NevIR, ExcluIR |
| 5 | [`relevance-reranker-cascade`](relevance-reranker-cascade/) | Combining rerankers, or escalating only when unsure, gives the same quality for less | Planned | ESCI (stored runs) |
| 6 | [`database-semantic-operators`](database-semantic-operators/) | Jev makes LLM SQL operators and filter cascades affordable at table scale | Planned | SemBench, ESCI |
| 7 | [`relevance-multimodal`](relevance-multimodal/) | Images carry signal the text misses, especially in thin listings | Planned | SQID |
| 8 | [`judge-wands`](judge-wands/) | The ESCI result carries over to another retailer | Planned | WANDS |
| 9 | [`relevance-ltr`](relevance-ltr/) | A model trained on every signal beats any single reranker, even when trained on Jev's labels | Planned | ESCI train split |
| 10 | [`relevance-routing`](relevance-routing/) | Different query types want different retrieval settings | Planned | ESCI |
| 11 | [`recall-rewrites`](recall-rewrites/) | Rewrites and document expansion close vocabulary gaps | Planned, if `recall-fusion` says so | ESCI |
| 12 | [`benchmark-instructions-stark`](benchmark-instructions-stark/) | Per-condition checks let a cheap model follow multi-condition requests | Planned | STaRK, FollowIR |
| 13 | [`benchmark-bright`](benchmark-bright/) | As a cheap first pass, Jev keeps most of a reasoning reranker's gain | Planned | BRIGHT |
| — | [`judge-negation-audit`](judge-negation-audit/) | On negation queries, the human labels may be the ones that are wrong | Done: mostly the labels | ESCI |
| — | [`judge-batch-order`](judge-batch-order/) | Batching products in one Jev request could change grades | Done: no | ESCI |
| — | [`judge-truncation`](judge-truncation/) | The judge needs more of the description | Done: no | ESCI |
| — | [`judge-jev-1.14`](judge-jev-1.14/) | Jev 1.14.0 is a better judge and reranker | Done: better judge, same reranker | ESCI |
| — | [`turbopuffer-perf`](turbopuffer-perf/) | Know where server time goes and what ANN gives up | Done | ESCI |
| — | [`retrieval-followups`](retrieval-followups/) | Four small retrieval changes each move quality or latency | Done | ESCI |

**Status:** *Planned* means the README is a test plan. *Ready* means the
script is written and checked offline, but hasn't run against
turbopuffer or Jev. *Done* means the result is in the study's README and
`results/`. If you have to choose, run `recall-fusion` and
`relevance-reranker-cascade` first: they need nothing new, cost little, and
decide whether `recall-rewrites` and `relevance-routing` are worth doing.

## What a study looks like

```
experiments/<group>-<name>/
  README.md     premise, status, method, how to run, how to decide, what it needs
  *.py          the script, if it has one
  results/      what it wrote: small CSVs and text, committed
```

Large intermediate files (ranked lists, tags for every product) go to
`data/<name>/`, which is git-ignored. Copy [`TEMPLATE.md`](TEMPLATE.md) to
start a new one.

## Rules every study follows

- **Fix the settings before looking at results.** A study that picks among
  options (thresholds, weights, prompts) picks on half the queries, split by
  a hash of the query id, and reports the other half.
- **Compare with paired bootstrap intervals** (`relevancy.metrics.paired_delta`)
  on the same queries. A gain counts when the interval's lower bound is
  above zero.
- **Account for unlabeled results.** A retriever that finds products the
  labels never covered is scored down for it. Report condensed nDCG@10, and
  Jev-filled nDCG@10 where it matters. Keep human labels as the primary
  metric whenever Jev is what's being tested.
- **Leave the main runs alone.** Read the warehouse read-only, or work on a
  copy (`RELEVANCY_DB=/tmp/copy.duckdb`). A study that changes a namespace,
  like `relevance-ontology`'s patch, says so.
- **Report cost and latency next to quality.**

## Before you start

**Keys:**
- `TURBOPUFFER_API_KEY`: anything that searches.
- `TYPESAFE_API_KEY` (Jev): almost everything.
- `OPENROUTER_API_KEY`: studies that need an LLM to write text or label
  ground truth (`judge-trec`, `relevance-constraints`,
  `database-semantic-operators`, `recall-rewrites`, `benchmark-bright`).

Each study's README lists the rest under **Needs**.

**Licences** (checked October 2026):

| Dataset | Licence |
| --- | --- |
| ESCI | Apache-2.0 ✅ |
| WANDS | MIT ✅ |
| BRIGHT | CC BY 4.0 ✅ |
| SQID | MIT for the data. The images are Amazon's, so evaluate on them but don't redistribute them. ⚠️ |
| TREC DL 2019/2020, LLMJudge | MS MARCO terms: "intended for non-commercial research purposes only". Get a legal review before customer-facing use. ⚠️ |
| FollowIR | Text from licensed news collections (Robust04, the NYT corpus, the Washington Post) ⚠️ |
| ExcluIR, SemBench | No licence file in the repo ⚠️ |
| STaRK | Code MIT. The Amazon data's terms aren't stated. ⚠️ |
| NevIR | Code MIT; built on CondaQA (Apache-2.0) |
| ScaleDoc's workloads | Not public ❌ |

## Recomputing the documented numbers

`evidence.py` recomputes every number in `docs/results.md` from the
warehouse and the finished studies' results:

```bash
uv run experiments/evidence.py | tee experiments/records/evidence.txt
```

[`records/`](records/) keeps logs from the main runs that the warehouse
doesn't store: the first full load, and the judge bake-off's latency lines.
