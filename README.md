# jev-relevancy

A small, reproducible search-quality lab. turbopuffer does the retrieval, a hosted model reranks the results, and DuckDB keeps every run and computes the metrics. The test set is Amazon's [Shopping Queries dataset (ESCI)](https://github.com/amazon-science/esci-data): real shopping queries with human relevance labels, Apache-2.0 licensed.

You can get through the whole loop in about 15 minutes and well under a dollar of turbopuffer usage. When you finish, you'll have a table comparing BM25, vector search, hybrid search, and two rerankers on the same 50 queries, scored three ways and with confidence intervals.

This README covers testing rerankers on top of turbopuffer's hybrid search.

## What you need

- A turbopuffer account and API key ([turbopuffer.com](https://turbopuffer.com)).
- [uv](https://docs.astral.sh/uv/) for Python. It installs Python 3.12 and every dependency for you.
- About 2 GB of free disk for the dataset and the warehouse.
- Optional: a [TypeSafe](https://typesafe.ai) API key if you want to use Jev directly (see [Jev](#jev) below).

## Set up

```bash
git clone https://github.com/rickcrawford/jev-relevancy.git
cd jev-relevancy
cp .env.example .env        # then paste your key after TURBOPUFFER_API_KEY=
uv sync
```

Two optional settings go in `.env` too:

| Variable | Default | Why you'd change it |
| --- | --- | --- |
| `TURBOPUFFER_REGION` | `aws-us-east-1` | Pick the region closest to where you run the scripts. Stay in a US region if you want to rerank with Jev, which turbopuffer only serves there. Skip `gcp-us-central1`; turbopuffer's docs say it's compute-constrained and ask new workloads to go elsewhere. |
| `EMBED_MODEL` | `voyage/voyage-4-lite` | Any model on turbopuffer's [embedding page](https://turbopuffer.com/docs/embedding). If you change it, reload the namespace. |

## The walkthrough

Each step is one script in `scripts/`, numbered in the order you run them. Each script explains itself with `--help`.

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

### 1. Download ESCI

```bash
./scripts/01_download_esci.sh
```

Three files, about 1.1 GB, into `data/raw/esci/`. The products file is the big one (1 GB) and took about five minutes on my connection.

### 2. Load it into the warehouse and pick queries

```bash
uv run scripts/02_prepare_esci.py --sample 50
```

This reads the reduced English split (the one the KDD Cup 2022 ranking task used) into `warehouse/relevancy.duckdb`:

- **docs:** 482,105 products, the retrieval corpus
- **queries:** the 8,956 English test queries, each tagged with a slice by length (short, medium, long) and by whether it names a brand
- **judgments:** 181,701 human labels, stored as grades: Exact 3, Substitute 2, Complement 1, Irrelevant 0

It then draws a sample stratified by slice, so the sample mixes short, long, branded, and unbranded queries in the same proportions as the full set. Start with 50 queries. Once everything works, rerun with `--sample 500` for numbers you can trust. It takes about 6 seconds either way.

### 3. Load products into turbopuffer

```bash
uv run scripts/03_load_turbopuffer.py --scope smoke
```

The `smoke` scope loads every product judged for your sampled queries, plus 20,000 random products as distractors. Without distractors, any search would look good. For a 50-query sample that's about 21,000 products, 3.5M embedding tokens, and 3 minutes.

Each product becomes a turbopuffer document with:

- `title` and `body`, full-text indexed for BM25
- `text`, made from the title, brand, color, and the first 800 characters of the description. turbopuffer embeds this field itself, so you don't need a separate embedding API, and the rerankers read it too.
- `brand` and `color`, filterable

One gotcha: new turbopuffer orgs get 2M embedding tokens per minute per model. Eight parallel writers hit that in a few seconds, so the script paces itself to 1.5M tokens a minute and backs off if it still gets a 429. If your org has a higher limit, raise `--tokens-per-minute`.

`--scope full` loads all 482k products: about 80M embedding tokens and about an hour at the default pace.

### 4. Search

```bash
uv run scripts/04_search.py --scope smoke
```

This runs three retrievers over every sampled query and keeps the top 50 from each:

- **bm25:** BM25 on title and body, with title counted twice
- **ann:** vector search over the embedded `text`
- **hybrid-rrf:** both at once in a single `multi_query` call, merged on the server with reciprocal rank fusion

Every result row lands in the warehouse with its rank, score, wall-clock latency, turbopuffer's server time, and billable bytes.

### 5. Score

```bash
uv run scripts/05_score.py
uv run scripts/05_score.py --by-slice     # nDCG@10 per query slice
uv run scripts/05_score.py --gain linear  # grades as gains, instead of 2^grade - 1
```

The metrics are SQL in [`src/relevancy/metrics.py`](src/relevancy/metrics.py):

- **nDCG@10** counts unjudged products as irrelevant. ESCI labels only about 20 products per query, so retrieval pulls in plenty of unlabeled ones, and many of them are fine. This number is a floor.
- **nDCG@10 condensed** drops the unjudged results before scoring. Read it next to the standard number, and don't compare it with standard nDCG from somewhere else.
- **Recall@50** is the share of Exact and Substitute products that made the top 50. No reranker can beat this, because it only reorders those 50.
- **MRR@10** measures how high the first Exact or Substitute result lands.

The interval next to nDCG@10 is a 95% bootstrap over queries. If two systems' intervals overlap a lot, you haven't shown that either one is better.

### 6. Rerank

```bash
uv run scripts/06_rerank.py --model voyage/rerank-3
uv run scripts/06_rerank.py --model zeroentropy/zerank-2
uv run scripts/05_score.py
```

The rerank step takes the hybrid run's top 50 for each query and has a hosted model reorder exactly those 50. It does this with a turbopuffer query filtered to those 50 ids, with `rerank_by` set. Fixing the candidates means any change in nDCG comes from the reranker alone. Latency for a reranked run covers both stages, retrieval plus reranking.

## What my smoke run looked like

50 queries, 21k-product namespace, `aws-us-east-1`, scripts run from a laptop, so wall-clock times include the trip over the public internet:

| Retriever | Reranker | nDCG@10 [95% CI] | Condensed | Recall@50 | MRR@10 | p50 wall ms | p50 server ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| hybrid-rrf | voyage/rerank-3 | 0.733 [0.662, 0.799] | 0.833 | 0.778 | 0.927 | 379 | 255 |
| hybrid-rrf | zeroentropy/zerank-2 | 0.704 [0.637, 0.769] | 0.818 | 0.778 | 0.903 | 429 | 307 |
| hybrid-rrf | none | 0.635 [0.554, 0.711] | 0.783 | 0.778 | 0.835 | 177 | 123 |
| ann | none | 0.621 [0.535, 0.705] | 0.764 | 0.739 | 0.776 | 134 | 81 |
| bm25 | none | 0.593 [0.526, 0.656] | 0.754 | 0.678 | 0.852 | 168 | 113 |

My read: with only 50 queries, the three first-stage retrievers overlap too much to rank against each other. The voyage reranker's gain over plain hybrid (about +0.10 nDCG@10) is the one result here that looks like more than noise, and I'd still want the 500-query run before saying so. Server times are higher than turbopuffer usually reports. I loaded the namespace minutes before searching it, so part of that is probably indexing that hadn't finished. ANN's server time also includes roughly 57 ms to embed each query.

## What it costs

These are list prices from turbopuffer's [pricing page](https://turbopuffer.com/pricing), applied to what the smoke run used. They ignore the batch-write discount, so treat them as an upper bound.

| Item | Smoke run | Full corpus (scaled from smoke) |
| --- | --- | --- |
| Embedding (voyage-4-lite, $0.02 / 1M tokens) | 3.5M tokens, ~$0.07 | ~80M tokens, ~$1.60 |
| Writes ($2 / GB) | 117 MB, ~$0.23 | ~2.7 GB, ~$5.40 |
| Reranking (voyage/rerank-3, $0.05 / 1M tokens) | under $0.05 per 50 queries | same per query |
| Queries | fractions of a cent | fractions of a cent |

On top of usage, your plan has a monthly minimum ($16 on Launch).

## Jev

[Jev](https://docs.typesafe.ai/introduction) is TypeSafe's "System One" model. You don't get text back from it; you ask it typed questions, such as a 4-level relevance `Score`, and get back probabilities. Elastic [tested it as an ESCI reranker](https://www.elastic.co/search-labs/blog/ecommerce-search-reranking-llm-alternative-jev), and the main point of this repo is to repeat that test on turbopuffer.

turbopuffer hosts Jev as `typesafe/jev-1.13.0` (US regions only, price not yet listed), but it isn't on by default. Step 0 shows the error if your org doesn't have it. There are two ways around that:

1. Ask turbopuffer to enable it for your org. Then `uv run scripts/06_rerank.py --model typesafe/jev-1.13.0` works with no code changes.
2. Call TypeSafe's API directly with a `TYPESAFE_API_KEY`. This path isn't written yet. Grading results, rather than reranking them, will need it.

## Repo layout

```
scripts/        the walkthrough, one numbered step per file
src/relevancy/  shared code: warehouse connection, turbopuffer queries, metrics
warehouse/      schema.sql (checked in) and relevancy.duckdb (built by the scripts)
docs/           design notes
data/           downloaded datasets (git-ignored)
```

The warehouse schema is set up so that a new dataset, retriever, reranker, or judge adds rows rather than code. Every run records its config and git commit, so you can trace any number in a results table back to what produced it.
