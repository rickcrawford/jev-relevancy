# Walkthrough

Part 1 runs the whole framework on ESCI, where human labels say how far to trust the judge. Part 2 runs the same scripts on your own documents and queries.

## What you need

- A turbopuffer account and API key ([turbopuffer.com](https://turbopuffer.com)).
- A TypeSafe API key for Jev ([typesafe.ai](https://typesafe.ai)).
- [uv](https://docs.astral.sh/uv/) for Python. It installs Python 3.12 and every dependency for you.
- About 2 GB of free disk for ESCI and the warehouse.
- Optional: an [OpenRouter](https://openrouter.ai) key, if you want to compare Jev with LLM judges. $5 of credit covers everything here several times over.

## Set up

```bash
git clone https://github.com/rickcrawford/jev-relevancy.git
cd jev-relevancy
cp .env.example .env        # then fill in your keys
uv sync
```

`.env` takes these:

| Variable | Needed for | Notes |
| --- | --- | --- |
| `TURBOPUFFER_API_KEY` | everything | |
| `TYPESAFE_API_KEY` | Jev, in steps 6 to 8 | |
| `OPENROUTER_API_KEY` | LLM judges and rerankers | optional |
| `TURBOPUFFER_REGION` | | defaults to `aws-us-east-1`. Pick the region closest to you. Skip `gcp-us-central1`; turbopuffer's [regions page](https://turbopuffer.com/docs/regions) says it's compute-constrained and asks new workloads to go elsewhere. |
| `EMBED_MODEL` | | defaults to `voyage/voyage-4-lite`. Any model on turbopuffer's [embedding page](https://turbopuffer.com/docs/embedding). Change it and you need to reload the namespace. |

## Part 1: calibrate on ESCI

Each step is one script in `scripts/`, numbered in the order you run them, and each explains itself with `--help`.

### 0. Check your setup

```bash
uv run scripts/00_check_setup.py
```

This confirms your key works and the warehouse opens. Run it again after step 3 and it also tells you which hosted rerankers your org can call. On my account it printed:

```
rerankers
  todo  typesafe/jev-1.13.0: reranking model `typesafe/jev-1.13.0` is not enabled for your organization, please reach out to us
  ok    voyage/rerank-3
  ok    voyage/rerank-3-lite
  ok    cohere/rerank-v3.5
  ok    zeroentropy/zerank-2
```

The Jev line is expected. turbopuffer lists `typesafe/jev-1.13.0` on its [reranking page](https://turbopuffer.com/docs/reranking) as a preview, and it wasn't enabled for my org, so this walkthrough calls Jev through TypeSafe's API instead (see [Jev](#jev)).

### 1. Download ESCI

```bash
./scripts/01_download_esci.sh
```

Three files, about 1.1 GB, into `data/raw/esci/`. The products file is the big one and took about five minutes on my connection.

### 2. Load it into the warehouse and pick queries

```bash
uv run scripts/02_prepare_esci.py --sample 50
```

This reads the reduced English split (the one the KDD Cup 2022 ranking task used) into the warehouse:

- **docs:** 482,105 products, the retrieval corpus
- **queries:** the 8,956 English test queries, each tagged with a slice by length (short, medium, long) and by whether it names a brand
- **judgments:** 181,701 human labels, stored as grades: Exact 3, Substitute 2, Complement 1, Irrelevant 0

It then draws a sample stratified by slice, so short, long, branded, and unbranded queries show up in the same proportions as the full set. Start with 50 queries. Once everything works, rerun with `--sample 500`, which narrows the intervals to about ±0.03, and use `--scope full` in steps 3 and 4, since the smoke namespace only has products for the first sample. Either way the script takes about 15 seconds.

It also registers the dataset with the `esci` rubric, Amazon's own grading guidelines in [`rubrics/esci.toml`](../rubrics/esci.toml).

### 3. Load products into turbopuffer

```bash
uv run scripts/03_load_turbopuffer.py --scope smoke
```

The `smoke` scope loads every product judged for your sampled queries, plus 20,000 random products as distractors. Without them the corpus holds only judged products, and every retriever scores higher than it would on real data. For a 50-query sample that's about 21,000 products, 3.5M embedding tokens, and 3 minutes.

Each product becomes a turbopuffer document with:

- `title` and `body`, full-text indexed for BM25
- `text`, made from the title, brand, color, and the first 800 characters of the description. turbopuffer embeds this field itself, so you don't need a separate embedding API, and rerankers and judges read it too.
- `brand` and `color`, filterable

The writing goes through [tpuf-arrow](https://github.com/rickcrawford/tpuf-arrow)'s `upsert`, which reads the products from DuckDB as Arrow, sends 256 per request (the most turbopuffer accepts when it does the embedding, per its [limits](https://turbopuffer.com/docs/limits)), runs 8 requests at once, and lets the SDK retry 429s for about 20 seconds. New turbopuffer orgs get 2M embedding tokens per minute per model ([embedding docs](https://turbopuffer.com/docs/embedding)), and eight writers hit that in a few seconds, so the script paces the products it hands over to 1.5M tokens a minute. The token limit is what sets the speed. If your org has a higher limit, raise `--tokens-per-minute`. If requests still hit the limit after the SDK's retries, the script waits a minute and loads whatever is missing, up to three more times. Each write request lands in the warehouse's `tpuf_requests` table with the tokens turbopuffer embedded for it, so `SELECT sum(embedding_tokens) FROM tpuf_requests WHERE run_id = 'load-esci-us-smoke'` gives the load's token count, resumes included (use `load-esci-us-full` for the full scope).

`--scope full` loads all 482k products. That's every product ESCI judged for at least one query, so even the full corpus has no products from outside the labeled set; a product just isn't labeled for most queries. Real catalogs have more junk to wade through, so expect absolute scores on your own data to run lower. The full load is about 80M embedding tokens (§17 of [`evidence.txt`](../experiments/records/evidence.txt)). That's an estimate: turbopuffer reported 42.6M tokens for the first 286,720 products (171M characters, so 4.0 characters a token), and the whole corpus is 323M characters, so the later products are longer. My first full run wrote those 286,720 products in 42 minutes, about 1M tokens a minute against the pacer's 1.5M budget, then stopped on a 429 the script didn't catch yet. I didn't keep the log of the run that finished, so I don't have its final token count or time. With a higher embedding limit, or vectors you computed yourself, the load would go faster. A load that long could outlast the old script's rate-limit retries on a batch or two, which is why it now waits and retries on its own. Mine stopped at 480k products, and `--resume` (which lists what the namespace already has and writes only the rest) finished the last 1,024 in a few seconds.

### 4. Search

```bash
uv run scripts/04_search.py --scope smoke
```

This runs four retrievers over every sampled query and keeps the top 50 from each:

- **bm25:** BM25 on title and body, with title counted twice
- **bm25-title:** BM25 on the title alone, a deliberately weak baseline so you can see whether a judge notices it's worse
- **ann:** vector search over the embedded `text`
- **hybrid-rrf:** BM25 and vector in one `multi_query` call, merged on the server with reciprocal rank fusion

Every result lands in the warehouse with its rank, score, wall-clock latency, turbopuffer's server time, and billable bytes. BM25 and vector search use tpuf-arrow's `query`, and hybrid uses its `multi_query`. Each request lands in `tpuf_requests` with everything else turbopuffer reported, like cache temperature and the newest write the query saw, and the result rows get their timing and cost from it by a join on `run_id` and `query_id`.

### 5. Score with the human labels

```bash
uv run scripts/05_score.py
uv run scripts/05_score.py --by-slice     # nDCG@10 per query slice
uv run scripts/05_score.py --gain linear  # grades as gains, instead of 2^grade - 1
```

The metrics are SQL in [`src/relevancy/metrics.py`](../src/relevancy/metrics.py):

- **nDCG@10** counts unjudged products as irrelevant. ESCI labels only about 20 products per query, so retrieval pulls in plenty of unlabeled ones, and many of them are fine. Read this number as a floor.
- **nDCG@10 condensed** drops the unjudged results before scoring. Read it next to the standard number, and don't compare it with standard nDCG from somewhere else.
- **Recall@50** is the share of Exact and Substitute products that made the top 50. No reranker can beat it, because a reranker only reorders those 50.
- **MRR@10** measures how high the first Exact or Substitute result lands.

The interval next to nDCG@10 is a 95% bootstrap over queries. The "vs base" column is sharper: it compares each run with plain hybrid search query by query, and bootstraps the difference. If that interval excludes zero, the difference probably isn't noise. Two runs' own intervals can overlap while the paired difference is still clear, because some queries are hard for every system.

### 6. Rerank

```bash
uv run scripts/06_rerank.py --model voyage/rerank-3
uv run scripts/06_rerank.py --model jev:jev-1.13.0
uv run scripts/06_rerank.py --model llm:google/gemma-3-27b-it
uv run scripts/05_score.py
```

This takes the hybrid run's top 50 for each query and has a model reorder exactly those 50, so any change in nDCG comes from the reranker alone. Three kinds work, picked by the `--model` prefix:

- **A turbopuffer-hosted model** (`voyage/rerank-3`, `cohere/rerank-v3.5`, `zeroentropy/zerank-2`, ...). The script queries turbopuffer filtered to those 50 ids, with `rerank_by` set.
- **`jev:`** grades all 50 candidates against the dataset's rubric in one TypeSafe request and sorts by expected grade (the sum of each grade times its probability).
- **`llm:`** does the same with any OpenRouter model. It asks for a single digit and reads the probability of each grade from the first token's logprobs, which is the closest an LLM gets to how Jev answers.

Latency for a reranked run covers both stages, retrieval plus reranking.

### 7. Judge bake-off: grade by grade against the humans

```bash
uv run scripts/07_judge_bakeoff.py
```

This takes 200 human-labeled pairs from queries outside your sample, 50 per grade, and has each judge grade them. It reports exact agreement, agreement within one grade, weighted kappa, the correlation of the judge's expected grade with the human grade, and latency. A few cents for every judge on the list.

The sample is balanced across grades on purpose. Real results are mostly Exact and Substitute, which would hide how a judge handles the rarer grades, so these numbers measure how well a judge separates the four grades. Agreement on live results runs higher (see step 8).

### 8. Judge the runs: ranking whole systems

```bash
uv run scripts/08_judge_runs.py
```

The default judges are Jev and Gemma 3 27B. Gemma goes through OpenRouter, so without an `OPENROUTER_API_KEY` pass `--judge jev-batched:jev-1.13.0`.

For an eval framework, single-grade agreement matters less than whether the judge picks the right winner between config A and config B. Step 8 pools the top 10 from every run, has each judge grade the pool, scores every run under each judge, and compares the judge's ranking of runs with the humans' (Kendall's tau, where 1 means the same order).

Two rules keep it honest. A run reranked by Jev is never scored by Jev, because the judge would be grading results it sorted itself. And when a judge scores a run, its ideal ranking comes from the same pool it graded, so no run loses points for products nobody showed the judge.

Grades are saved as they come back, every 25 queries, and pairs already graded are skipped on a rerun.

One thing to know about a judge's nDCG: its ideal ranking is built from the pool, so a run's score depends on which other runs were pooled with it. Voyage scores 0.897 under Jev when pooled with three other runs, 0.861 with eight, and 0.854 with nine (evidence §9). Compare judge scores only between runs scored together in the same step 8 call.

## Part 2: your own data

Once you've seen how far Jev agrees with humans on ESCI, the same scripts run on your data. You need two files: documents and queries.

```bash
uv run scripts/02_prepare_custom.py --dataset mydocs \
    --docs path/to/docs.parquet --queries path/to/queries.csv --rubric generic
uv run scripts/03_load_turbopuffer.py --dataset mydocs --scope full
uv run scripts/04_search.py --dataset mydocs --scope full
uv run scripts/06_rerank.py --dataset mydocs --model voyage/rerank-3     # optional
uv run scripts/08_judge_runs.py --dataset mydocs --judge jev-batched:jev-1.13.0
uv run scripts/05_score.py --dataset mydocs --judge jev-batched:jev-1.13.0
```

Files can be CSV, Parquet, JSON, or JSONL:

- **docs:** `id`, plus `title` and/or `body`. `brand` and `color` are optional and become filterable attributes.
- **queries:** `query_id` and `text`. Pull a few hundred real queries from your search logs; a stratified sample across query types beats the top 500 by volume, which are mostly easy. An optional `slice` column lets step 5 break results down by type.
- **judgments** (optional): `query_id`, `doc_id`, `grade` from 0 to 3. If you have even a hundred human labels, load them. Step 8 will then check Jev against them on your data, which beats any number from ESCI.

With no human labels, step 8 grades the pool and reports each run's nDCG@10 under Jev, and step 5 with `--judge` gives you the paired difference against your current setup. That's the number to put in front of whoever approves the change.

### Write a rubric

The rubric is where you tell the judge what "relevant" means. [`rubrics/generic.toml`](../rubrics/generic.toml) is a reasonable start:

```toml
state = "Search query: {query}"
question = "How well does this result answer the search query?"
item = "Result"

levels = [
    "Irrelevant: has nothing useful for this query",
    "Related: on the topic, but doesn't answer the query",
    "Relevant: answers part of the query, or answers it with gaps",
    "Perfect: fully answers the query; this is what the searcher wanted",
]
```

Copy it to `rubrics/<yours>.toml` and make the levels specific. For a support knowledge base, "Perfect" might be "resolves the customer's problem without another search". For a legal corpus, it might name the jurisdiction. Jev reads `state` as context and the levels as the answer choices, and the LLM judges get the same levels as a numbered prompt, so every judge grades against the same definition. Keep four levels, lowest first: the grade is the position, and recall and MRR count grade 2 and up as relevant.

After editing a rubric, regrade with `uv run scripts/08_judge_runs.py --dataset mydocs --judge jev-batched:jev-1.13.0 --regrade`.

## Jev

Jev doesn't return text. You send a `state` (here, the query) and typed questions (here, one 4-level `Score` per result), and it returns a probability for each level. The expected grade works as a ranking score, and the most likely grade works as a label. Elastic [tested it as an ESCI reranker](https://www.elastic.co/search-labs/blog/ecommerce-search-reranking-llm-alternative-jev), and part 1 runs a similar test on turbopuffer. The numbers aren't directly comparable: here every reranker sees turbopuffer's top 50 from the whole corpus, and unlabeled products count as irrelevant.

There are two ways to call it:

1. **TypeSafe's API** (`TYPESAFE_API_KEY`), which is what this repo does. [`src/relevancy/jev.py`](../src/relevancy/jev.py) puts up to 50 results for one query into a single request as separate questions. That takes about as long as one result, and grades come back the same as one at a time within Jev's own run-to-run noise ([step 7](results.md#judges-against-humans-step-7)). Input is $0.042 per million tokens.
2. **Inside turbopuffer**, as `typesafe/jev-1.13.0` in `rerank_by`. turbopuffer's reranking page lists it as a preview with US inference, and it wasn't enabled for my org; ask turbopuffer about access. Then `uv run scripts/06_rerank.py --model typesafe/jev-1.13.0` reranks with no code changes, though that path only reranks; it can't hand back grades for judging.

The judge names in this repo:

| Name | What it does |
| --- | --- |
| `jev:jev-1.13.0` | one request per (query, result) pair |
| `jev-batched:jev-1.13.0` | all of a query's results in one request; the default for reranking and step 8 |
| `openai/gpt-4o-mini`, or any OpenRouter model | an LLM read through logprobs |

TypeSafe's API defaults to a newer Jev (1.14.0 when I ran this). I pinned 1.13.0 to match what turbopuffer hosts and what Elastic tested. As a judge, 1.14.0 agrees with the humans more often ([step 8](results.md#judges-ranking-systems-step-8)), so pass `--judge jev-batched:jev-1.14.0` to step 8 and `--model jev:jev-1.14.0` to step 6 unless you need to match turbopuffer's hosted version.
