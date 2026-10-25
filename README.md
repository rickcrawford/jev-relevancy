# jev-relevancy

A search evaluation framework with [Jev](https://docs.typesafe.ai/introduction) as the judge. [turbopuffer](https://turbopuffer.com) does the retrieval and the hosted reranking. [DuckDB](https://duckdb.org) keeps every run, result, grade, and latency in one file and computes the metrics in SQL. The scripts talk to turbopuffer through [tpuf-arrow](https://github.com/rickcrawford/tpuf-arrow), which returns hits as Arrow tables DuckDB can query.

When someone changes your search (a new embedding model, a reranker, different BM25 weights), the question is whether results got better, and the usual answer is a handful of spot checks. Human labels would settle it, but most teams don't have them for their own data, and labeling a few thousand results takes weeks. Jev is TypeSafe's grading model: you ask it a typed question, like "how well does this result match the query, on this 4-level scale?", and in about 100 ms it returns a probability for each level, for a small fraction of a cent. A cheap judge is only useful if you know how far to trust it, so this repo has three parts:

1. **Calibrate on ESCI.** Amazon's [Shopping Queries dataset](https://github.com/amazon-science/esci-data) has shopping queries graded by people. You run several retrievers and rerankers, score them with the human labels, then have Jev grade the same results and see whether it ranks the systems the same way.
2. **Run it on your own data.** Load your documents and queries, write a rubric that says what "relevant" means for your users, and let Jev grade. Same scripts and tables, no labels needed.
3. **Pick a test.** Each study in [`experiments/`](experiments/README.md) tests one premise, such as "relevant products are retrieved but rank too low", with the result that would count written down first.

```
your queries ─▶ turbopuffer ─▶ top 50 per query ─▶ reranker (optional) ─▶ results
                                                                              │
          rubric (rubrics/*.toml) ─▶ Jev or an LLM grades the top 10 ─────────┤
                                                                              ▼
                              DuckDB: runs, results, judgments ─▶ nDCG@10, recall, CIs
```

## Results

500 ESCI queries over all 482,105 products. Every reranker reorders the same hybrid top 50. Two of the nine runs (Gemma as a reranker, and title-only BM25) are left out here.

| Retriever | Reranker | nDCG@10 [95% CI] | vs hybrid | Wall p50 ms |
| --- | --- | --- | --- | --- |
| hybrid-rrf | voyage/rerank-3 | 0.463 [0.436, 0.489] | +0.096 | 362 |
| hybrid-rrf | jev:jev-1.13.0 | 0.452 [0.425, 0.478] | +0.085 | 499 |
| hybrid-rrf | zeroentropy/zerank-2 | 0.441 [0.414, 0.467] | +0.074 | 412 |
| hybrid-rrf | cohere/rerank-v3.5 | 0.436 [0.409, 0.462] | +0.070 | 433 |
| hybrid-rrf | none | 0.367 [0.340, 0.392] | base | 160 |
| ann | none | 0.350 [0.322, 0.376] | -0.017 | 130 |
| bm25 | none | 0.311 [0.286, 0.335] | -0.056 | 156 |

- As a reranker, Jev and Voyage's rerank-3 tie. Voyage leads by 0.011 on standard nDCG@10 (the interval just reaches zero), and Jev leads by 0.003 when unlabeled products are dropped.
- As a judge, Jev had the highest exact agreement with the human graders of the seven judges tested, 8 points ahead of the best LLM, at a quarter to a sixth of the LLMs' latency.
- Asked to rank eight setups, Jev put 26 of the 28 pairs in the same order as the human labels on standard nDCG@10. One of the two it flipped was a real difference: the humans put Voyage ahead of zerank-2 by 0.022, and Jev had them the other way around.

[docs/results.md](docs/results.md) has the full tables, the judge comparisons, what turbopuffer did during the runs, costs, and how much to trust all of it. `uv run experiments/evidence.py` recomputes every number from the warehouse.

## Requirements

- A turbopuffer API key
- A [TypeSafe](https://typesafe.ai) API key, for Jev
- [uv](https://docs.astral.sh/uv/), which installs Python 3.12 and the dependencies
- About 2 GB of disk for ESCI and the warehouse
- Optional: an [OpenRouter](https://openrouter.ai) key, to compare Jev with LLM judges

## Setup

```bash
git clone https://github.com/rickcrawford/jev-relevancy.git && cd jev-relevancy
cp .env.example .env        # then fill in your keys
uv sync
```

## Try it on the sample

[`sample/`](sample/README.md) has 176 ESCI products, 8 queries, and their human grades. This loads them into a namespace called `shop-sample-full`, runs the four retrievers, and scores them. It needs only the turbopuffer key, and it doesn't touch the ESCI data or namespaces.

```bash
uv run scripts/02_prepare_custom.py --dataset shop-sample --docs sample/docs.parquet \
    --queries sample/queries.csv --judgments sample/labels.csv --rubric esci
uv run scripts/03_load_turbopuffer.py --dataset shop-sample --scope full
uv run scripts/04_search.py --dataset shop-sample --scope full
uv run scripts/06_rerank.py --dataset shop-sample --model voyage/rerank-3     # if your org has it
uv run scripts/05_score.py --dataset shop-sample
```

## Quick start

A 50-query run on a 21,000-product namespace. The download is 1.1 GB, and the load embeds about 3.5M tokens.

```bash
uv run scripts/00_check_setup.py
./scripts/01_download_esci.sh
uv run scripts/02_prepare_esci.py --sample 50
uv run scripts/03_load_turbopuffer.py --scope smoke
uv run scripts/04_search.py --scope smoke
uv run scripts/05_score.py
```

## Steps

Run them in order. Each takes `--help`, and the [walkthrough](docs/walkthrough.md) covers them in detail.

| Script | |
| --- | --- |
| [`00_check_setup.py`](scripts/00_check_setup.py) | Check the turbopuffer key, the warehouse, and which hosted rerankers your org can call |
| [`01_download_esci.sh`](scripts/01_download_esci.sh) | Download ESCI (1.1 GB) |
| [`02_prepare_esci.py`](scripts/02_prepare_esci.py) | Load ESCI into the warehouse and draw a stratified query sample |
| [`02_prepare_custom.py`](scripts/02_prepare_custom.py) | The same for your own documents, queries, and optional labels |
| [`03_load_turbopuffer.py`](scripts/03_load_turbopuffer.py) | Write the documents to a namespace, paced under the embedding token limit |
| [`04_search.py`](scripts/04_search.py) | Run BM25, title-only BM25, vector, and hybrid search for every sampled query |
| [`05_score.py`](scripts/05_score.py) | nDCG@10, condensed nDCG, Recall@50, and MRR@10, with paired bootstrap intervals |
| [`06_rerank.py`](scripts/06_rerank.py) | Rerank the hybrid top 50 with a hosted model, Jev, or an LLM |
| [`07_judge_bakeoff.py`](scripts/07_judge_bakeoff.py) | Grade 200 human-labeled pairs with each judge and compare |
| [`08_judge_runs.py`](scripts/08_judge_runs.py) | Have each judge grade every run's top 10, and check whether it ranks the runs like the humans |

On your own data, start at `02_prepare_custom.py` and skip the steps that need human labels:

```bash
uv run scripts/02_prepare_custom.py --dataset mydocs --docs docs.parquet --queries queries.csv --rubric generic
uv run scripts/03_load_turbopuffer.py --dataset mydocs --scope full
uv run scripts/04_search.py --dataset mydocs --scope full
uv run scripts/08_judge_runs.py --dataset mydocs --judge jev-batched:jev-1.13.0
uv run scripts/05_score.py --dataset mydocs --judge jev-batched:jev-1.13.0
```

## Pick a test

Once the ESCI base is built (the full load plus steps 4, 6, and 8; [the list](experiments/README.md#build-the-base-first)), each study in [`experiments/`](experiments/README.md) tests one premise and says beforehand what result would count.

| If you want to know | Start with |
| --- | --- |
| Whether an automated grader can stand in for human labels | [`judge-wands`](experiments/judge-wands/), plus four finished robustness checks |
| Where you lose relevant products before the reranker | [`recall-fusion`](experiments/recall-fusion/) |
| Whether a change improves relevance | [`relevance-ontology`](experiments/relevance-ontology/), [`relevance-constraints`](experiments/relevance-constraints/), [`relevance-reranker-cascade`](experiments/relevance-reranker-cascade/), and others in the catalog |
| What turbopuffer does with your queries | [`turbopuffer-perf`](experiments/turbopuffer-perf/), [`retrieval-followups`](experiments/retrieval-followups/) (both finished) |
| How Jev does on public benchmarks | [`judge-trec`](experiments/judge-trec/) (non-commercial data), [`benchmark-instructions-stark`](experiments/benchmark-instructions-stark/), [`benchmark-bright`](experiments/benchmark-bright/) |
| Whether Jev can run LLM filters in SQL | [`database-semantic-operators`](experiments/database-semantic-operators/) |

The [catalog](experiments/README.md) ranks all of them, with status, dataset, and licence. `uv run experiments/run.py list` shows the same from the terminal, and `run <study>` runs one; paid steps only run with `--yes`. To have an agent run them, point it at [`AGENTS.md`](AGENTS.md), or use the `run-study` skill in Claude Code. [`llms.txt`](llms.txt) indexes the repo for models.

## How it uses tpuf-arrow

The scripts double as an example of tpuf-arrow. Most of the calls are in [`src/relevancy/tpuf.py`](src/relevancy/tpuf.py).

| Step | Call |
| --- | --- |
| Load products from DuckDB, 256 per request | `tp.upsert(ns, reader, schema=..., distance_metric="cosine_distance")` in [`03_load_turbopuffer.py`](scripts/03_load_turbopuffer.py) |
| Find what a partial load missed | `tp.scan(ns, columns=[])`, then an anti-join in DuckDB |
| BM25 and vector search | `tp.query(ns, rank_by, top_k=50, run_id=..., query_id=...)` |
| Hybrid search, fused on the server | `tp.multi_query(ns, [ann, bm25], rerank_by=("RRF", {}), limit=50)` |
| Rerank a fixed candidate set | `tp.query(ns, ("id", "asc"), top_k=len(ids), filters=("id", "In", ids), rerank_by={"model": ...})` |
| Latency and cost per query | `tp.save_log(con)` into `tpuf_requests`, joined to `results` on `run_id` and `query_id` in [`04_search.py`](scripts/04_search.py) |
| Search from SQL | `ta.register(con, tp)` and `tpuf_bm25(...)`, in a [warehouse recipe](docs/warehouse.md#recipes) |

## Documentation

- [Walkthrough](docs/walkthrough.md): every step, your own data, rubrics, and the two ways to call Jev
- [Results](docs/results.md): the 500-query run, judge agreement, turbopuffer performance, and costs
- [Querying the warehouse](docs/warehouse.md): the tables, things to know, and SQL recipes
- [Pick a test](experiments/README.md): every study, ranked, with its premise, status, and what it needs
- [Public benchmarks](docs/benchmarks.md): where else Jev could be tested, as a judge, a reranker, and a database operator
- [Connector design](docs/duckdb-turbopuffer-connector.md): the design that became tpuf-arrow

## Layout

```
scripts/        the walkthrough, one numbered step per file
experiments/    one folder per study (README, study.toml, script, results), run.py to run them, and evidence.py, which recomputes every number
src/relevancy/  shared code: warehouse, turbopuffer queries, judges, metrics
rubrics/        what "relevant" means, one TOML file per rubric
sample/         176 ESCI products, 8 queries, and their grades, for a quick try
warehouse/      schema.sql (checked in) and relevancy.duckdb (built by the scripts)
docs/           the walkthrough, results, warehouse guide, benchmark survey, and connector design
data/           downloaded datasets (git-ignored)
.claude/skills/ run-study, the skill an agent follows to run a study (see AGENTS.md and llms.txt)
```

Every run records its config and git commit, so a number in a results table traces back to the code that produced it. (The headline runs recorded a dirty tree; [docs/results.md](docs/results.md) says which commit.)

## Data

ESCI, and the slice of it in `sample/`, is from Amazon's [Shopping Queries dataset](https://github.com/amazon-science/esci-data) ([Reddy et al., 2022](https://arxiv.org/abs/2206.06588)), Apache-2.0.

## License

The code is MIT, see [LICENSE](LICENSE). The ESCI data keeps its Apache-2.0 license ([text](LICENSES/Apache-2.0.txt)). [NOTICE](NOTICE) lists every file here that contains ESCI material, and the terms of each dataset the studies download.
