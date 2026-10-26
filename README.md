# jev-relevancy

A search evaluation framework with [Jev](https://docs.typesafe.ai/introduction) as the judge. [turbopuffer](https://turbopuffer.com) does the retrieval and the hosted reranking. [DuckDB](https://duckdb.org) keeps every run, result, grade, and latency in one file and computes the metrics in SQL. The scripts talk to turbopuffer through [tpuf-arrow](https://github.com/rickcrawford/tpuf-arrow), which returns hits as Arrow tables DuckDB can query.

When someone changes your search (a new embedding model, a reranker, different BM25 weights), the question is whether results got better, and the usual answer is a handful of spot checks. Human labels would settle it, but most teams don't have them for their own data, and labeling a few thousand results takes weeks. Jev is TypeSafe's grading model: you ask it a typed question, like "how well does this result match the query, on this 4-level scale?", and in about 100 ms it returns a probability for each level, for a small fraction of a cent. A cheap judge is only useful if you know how far to trust it, so this repo has three parts:

1. **Calibrate on ESCI.** Amazon's [Shopping Queries dataset](https://github.com/amazon-science/esci-data) has shopping queries graded by people. You run several retrievers and rerankers, score them with the human labels, then have Jev grade the same results and see whether it ranks the systems the same way.
2. **Run it on your own data.** Load your documents and queries, write a rubric that says what "relevant" means for your users, and let Jev grade. Same scripts and tables, no labels needed.
3. **Run a study.** Each study in [`experiments/`](experiments/README.md) tests one premise, such as "relevant products are retrieved but rank too low", with the result that would count written down before it runs.

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

[docs/results.md](docs/results.md) has the full tables, the judge comparisons, what turbopuffer did during the runs, costs, and how much to trust all of it. `uv run relevancy evidence` recomputes every number from the warehouse.

## What's been run, and what hasn't

There are two kinds of "test" here, and it's worth keeping them apart:

- **Offline tests** (`uv run pytest`) run every step of every study against fake turbopuffer, Jev, and LLM clients. They show the scripts run end to end. They say nothing about whether a method works.
- **Real runs** call the real services and produce the numbers a study reports. Only these count as results.

### Offline tests: passing

As of October 26, 2026: 18 tests, all passing, and `ruff check .` is clean. One of them (`test_ltr.py`) skips unless you install the optional dependencies:

```bash
uv sync --extra studies     # datasets, ir-datasets, lightgbm
uv run pytest -q            # 18 passed
uv run ruff check .
```

`test_manifests.py` checks that every step in every `study.toml` resolves to a real command. The other files cover the 13 studies that haven't run yet.

### Real runs: done, with results in the repo

| What | When | Where the numbers are |
| --- | --- | --- |
| ESCI pipeline, 500 queries, full 482k namespace: 4 retrievers and 5 rerankers | Oct 23 | [docs/results.md](docs/results.md), §1 to §4 of [evidence.txt](experiments/records/evidence.txt) |
| Smoke runs, 50 queries on the 21k namespace | Oct 23 | the warehouse only |
| Judge bake-off (step 7): 7 judges on 200 human-labeled pairs | Oct 23 | [docs/results.md](docs/results.md), §6 |
| Judges ranking runs (step 8): Jev 1.13.0 and Gemma 3 27B over the pooled top 10 | Oct 23 | [docs/results.md](docs/results.md) |
| [`judge-negation-audit`](experiments/judge-negation-audit/): on negation queries the human labels are mostly the ones that are wrong | Oct 23 | the study's README |
| [`judge-batch-order`](experiments/judge-batch-order/): product order in a batched request doesn't change Jev's grades | Oct 23 | the study's README |
| [`judge-truncation`](experiments/judge-truncation/): 4,000 characters of description instead of 800 doesn't help | Oct 23 | the study's README |
| [`judge-jev-1.14`](experiments/judge-jev-1.14/): a better judge, the same reranker | Oct 23 | the study's README |
| [`turbopuffer-perf`](experiments/turbopuffer-perf/): where server time goes, and ANN recall | Oct 23 | the study's README |
| [`retrieval-followups`](experiments/retrieval-followups/): title-only BM25, one-request rerank, brand filter, judged-only | Oct 23 | the study's README |

`uv run relevancy status` shows "never run here" for these six studies. They ran before the study runner existed, so they have no `results/runs/*.json`. Their results are real and are recorded in each README.

### Punchlist: still to run

In order. Nothing below has run against real services.

- [ ] **Reload the `esci-us-full` namespace.** With the key in `.env`, turbopuffer lists no namespaces in `aws-us-east-1` (`uv run relevancy pipeline check-setup`). `relevancy base` and `relevancy check` still say the base is built, because they only look at the warehouse. Every ESCI study needs the namespace back first: `uv run relevancy pipeline load --scope full --yes` (about 80M embedding tokens, roughly $1.60 to embed and under $4 to write, 40+ minutes).
- [ ] **Run the sample end to end** ([below](#try-it-on-the-sample)). The headline runs came from commit `35727d9`, before the scripts moved to tpuf-arrow, and the warehouse has no `tpuf_requests` table, so the current load, search, and rerank code has no recorded run against real turbopuffer. The sample is the cheapest way to check it.
- [ ] **Studies, by rank.** All 13 are scripted and pass the offline tests. "By hand" is a part the CLI can't do for you.

| Rank | Study | Needs the ESCI base | By hand, or blocked |
| --- | --- | --- | --- |
| 1 | [`judge-wands`](experiments/judge-wands/) | no, brings its own data | |
| 2 | [`recall-fusion`](experiments/recall-fusion/) | yes | |
| 3 | [`relevance-ontology`](experiments/relevance-ontology/) | yes (pilot runs on smoke) | check Google taxonomy terms ⚠️; `patch` writes to `esci-us-full` |
| 4 | [`relevance-attributes`](experiments/relevance-attributes/) | yes | about two hours of a person rating labels; `patch` writes to `esci-us-full` |
| 5 | [`relevance-constraints`](experiments/relevance-constraints/) | yes | ExcluIR left out (no license file) ⚠️ |
| 6 | [`relevance-reranker-cascade`](experiments/relevance-reranker-cascade/) | yes | |
| 7 | [`judge-trec`](experiments/judge-trec/) | no | TREC run files for tau, LLMJudge; non-commercial data ⚠️ |
| 8 | [`database-semantic-operators`](experiments/database-semantic-operators/) | yes | SemBench setup (no license file) ⚠️ |
| 9 | [`relevance-ltr`](experiments/relevance-ltr/) | yes | |
| 10 | [`relevance-routing`](experiments/relevance-routing/) | yes | blocked on `recall-fusion`'s `fetch` and `sweep` |
| 11 | [`recall-rewrites`](experiments/recall-rewrites/) | yes | run only if `recall-fusion` finds vocabulary misses; `patch-expansion` writes to `esci-us-full` |
| 12 | [`benchmark-instructions-stark`](experiments/benchmark-instructions-stark/) | no | `stark-qa` recommends Python 3.11 or older; data terms ⚠️ |
| 13 | [`benchmark-bright`](experiments/benchmark-bright/) | no | |

⚠️ means check the data's license before publishing a result ([details](experiments/README.md#keys-and-licenses)). If I had to pick two to run next, they'd be `recall-fusion` and `relevance-reranker-cascade`: between them they decide whether `recall-rewrites` and `relevance-routing` are worth running.

- [ ] **Open questions with no study yet:**
  - A second rater for the negation audit. The first was one Claude model.
  - Jev inside turbopuffer (`typesafe/jev-1.13.0` in `rerank_by`). It's a preview that wasn't enabled for my org.
  - From [docs/results.md](docs/results.md): throughput and tail latency under concurrent load from inside the region, a cold namespace, write throughput with vectors computed ahead of time, and how long indexing takes to catch up after a big load.

When a study finishes, [AGENTS.md](AGENTS.md) says how to record it: the README's Result section, `status = "done"` in `study.toml`, and the row in [experiments/README.md](experiments/README.md). Update the tables above at the same time.

## Requirements

- A turbopuffer API key
- A [TypeSafe](https://typesafe.ai) API key, for Jev
- [uv](https://docs.astral.sh/uv/), which installs Python 3.12 and the dependencies
- About 2 GB of disk for ESCI and the warehouse
- Optional: an [OpenRouter](https://openrouter.ai) key, to compare Jev with LLM judges and for studies where an LLM writes text

## Setup

```bash
git clone https://github.com/rickcrawford/jev-relevancy.git && cd jev-relevancy
cp .env.example .env        # then fill in your keys
uv sync                     # add --extra studies for the benchmarks, judge-trec, NevIR, and relevance-ltr
```

## The CLI

Everything runs through one command, `uv run relevancy`. Anything that calls a paid service (turbopuffer queries or writes, Jev, an LLM) refuses to run without `--yes`.

| Command | What it does | Paid |
| --- | --- | --- |
| `relevancy pipeline <step> [args]` | Run a numbered script in `scripts/`. Extra args pass through to the script | `load`, `search`, `rerank`, `bakeoff`, `judge-runs` |
| `relevancy list` | Every study, ranked, with its status | no |
| `relevancy show <study>` | A study's premise, steps, and pass mark | no |
| `relevancy check [<study>]` | What a study is missing: keys, base, files (it reads the warehouse, not turbopuffer) | no |
| `relevancy base` | What the ESCI base still needs, with the command for each piece | with `--yes`, it builds them |
| `relevancy run <study>` | Print a study's steps, paid ones marked | no |
| `relevancy run <study> --yes` | Run them. `--step <name>` (repeatable) runs only those, `--from <name>` starts partway, `--scope smoke\|full` for studies that take it | yes |
| `relevancy status` | The last runner run of every study | no |
| `relevancy latest-run` | The latest plain hybrid run on the full namespace, for `--base-run` | no |
| `relevancy evidence` | Recompute every number in `docs/results.md` | no |

`list`, `show`, `check`, `base`, and `status` take `--json` for an agent to parse. Every `run` writes `results/runs/<time>.json` in the study folder, with exit codes, durations, and the git commit. `RELEVANCY_DB=<path>` points the CLI, scripts, and studies at a copy of the warehouse.

The pipeline steps map to the scripts like this. `relevancy pipeline <step> --help` shows the CLI's own help, so for a script's flags run it directly (`uv run scripts/04_search.py --help`). The [walkthrough](docs/walkthrough.md) covers each one in detail.

| Step | Script | What it does |
| --- | --- | --- |
| `check-setup` | [`00_check_setup.py`](scripts/00_check_setup.py) | Check the turbopuffer key, its namespaces, the warehouse, and which hosted rerankers your org can call |
| `download` | [`01_download_esci.sh`](scripts/01_download_esci.sh) | Download ESCI (1.1 GB) |
| `prepare` | [`02_prepare_esci.py`](scripts/02_prepare_esci.py) | Load ESCI into the warehouse and draw a stratified query sample (`--sample 500`) |
| `prepare-custom` | [`02_prepare_custom.py`](scripts/02_prepare_custom.py) | The same for your own documents, queries, and optional labels |
| `load` | [`03_load_turbopuffer.py`](scripts/03_load_turbopuffer.py) | Write the documents to a namespace, paced under the embedding token limit (`--scope smoke\|full`, `--resume`) |
| `search` | [`04_search.py`](scripts/04_search.py) | BM25, title-only BM25, vector, and hybrid search for every sampled query |
| `score` | [`05_score.py`](scripts/05_score.py) | nDCG@10, condensed nDCG, Recall@50, and MRR@10, with paired bootstrap intervals (`--judge`, `--by-slice`) |
| `rerank` | [`06_rerank.py`](scripts/06_rerank.py) | Rerank the hybrid top 50 with a hosted model, Jev, or an LLM (`--model`) |
| `bakeoff` | [`07_judge_bakeoff.py`](scripts/07_judge_bakeoff.py) | Grade 200 human-labeled pairs with each judge and compare |
| `judge-runs` | [`08_judge_runs.py`](scripts/08_judge_runs.py) | Have each judge grade every run's top 10, and check whether it ranks the runs like the humans |

To have an agent run a study, point it at [`AGENTS.md`](AGENTS.md), or use the `run-study` skill in Claude Code. [`llms.txt`](llms.txt) indexes the repo for models.

## Try it on the sample

[`sample/`](sample/README.md) has 176 ESCI products, 8 queries, and their human grades. This loads them into a namespace called `shop-sample-full`, runs the four retrievers, and scores them. It needs only the turbopuffer key, and it doesn't touch the ESCI data or namespaces.

```bash
uv run relevancy pipeline prepare-custom --dataset shop-sample --docs sample/docs.parquet \
    --queries sample/queries.csv --judgments sample/labels.csv --rubric esci
uv run relevancy pipeline load --yes --dataset shop-sample --scope full
uv run relevancy pipeline search --yes --dataset shop-sample --scope full
uv run relevancy pipeline rerank --yes --dataset shop-sample --model voyage/rerank-3   # if your org has it
uv run relevancy pipeline score --dataset shop-sample
```

## Quick start on ESCI

A 50-query run on a 21,000-product namespace. The download is 1.1 GB, and the load embeds about 3.5M tokens.

```bash
uv run relevancy pipeline check-setup
uv run relevancy pipeline download
uv run relevancy pipeline prepare --sample 50
uv run relevancy pipeline load --yes --scope smoke
uv run relevancy pipeline search --yes --scope smoke
uv run relevancy pipeline score
```

The full base that most studies need is the same with `--sample 500` and `--scope full`, plus a Voyage and a Jev rerank and step 8. `uv run relevancy base` prints each command, and `uv run relevancy base --yes` runs them (an hour or more).

## Your own data

Start at `prepare-custom` and skip the steps that need human labels:

```bash
uv run relevancy pipeline prepare-custom --dataset mydocs --docs docs.parquet --queries queries.csv --rubric generic
uv run relevancy pipeline load --yes --dataset mydocs --scope full
uv run relevancy pipeline search --yes --dataset mydocs --scope full
uv run relevancy pipeline judge-runs --yes --dataset mydocs --judge jev-batched:jev-1.13.0
uv run relevancy pipeline score --dataset mydocs --judge jev-batched:jev-1.13.0
```

The [walkthrough](docs/walkthrough.md#part-2-your-own-data) covers the file formats, writing a rubric, and calibrating Jev against a few hundred labels of your own.

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
- [Studies](experiments/README.md): every study, ranked, with its premise, status, and what it needs
- [Public benchmarks](docs/benchmarks.md): where else Jev could be tested, as a judge, a reranker, and a database operator
- [Connector design](docs/duckdb-turbopuffer-connector.md): the design that became tpuf-arrow

## Layout

```
scripts/          the pipeline, one numbered step per file
src/relevancy/    shared code: warehouse, turbopuffer queries, judges, metrics, and cli.py (the `relevancy` command)
experiments/      one folder per study (README, study.toml, script, results), plus evidence.py
tests/            offline tests: every study's steps against fakes
rubrics/          what "relevant" means, one TOML file per rubric
sample/           176 ESCI products, 8 queries, and their grades, for a quick try
warehouse/        schema.sql (checked in) and relevancy.duckdb (built by the scripts)
docs/             the walkthrough, results, warehouse guide, benchmark survey, and connector design
data/             downloaded datasets (git-ignored)
.claude/skills/   run-study, the skill an agent follows to run a study
```

Every run records its config and git commit, so a number in a results table traces back to the code that produced it. (The headline runs recorded a dirty tree; [docs/results.md](docs/results.md) says which commit.)

## Data

ESCI, and the slice of it in `sample/`, is from Amazon's [Shopping Queries dataset](https://github.com/amazon-science/esci-data) ([Reddy et al., 2022](https://arxiv.org/abs/2206.06588)), Apache-2.0.

## License

The code is MIT, see [LICENSE](LICENSE). The ESCI data keeps its Apache-2.0 license ([text](LICENSES/Apache-2.0.txt)). [NOTICE](NOTICE) lists every file here that contains ESCI material, and the terms of each dataset the studies download.
