# jev-relevancy

Set up a search evaluation framework with [Jev](https://docs.typesafe.ai/introduction) as the judge.

When someone changes your search (a new embedding model, a reranker, different BM25 weights), the honest question is whether results got better, and the usual answer is a handful of spot checks. Human relevance labels would settle it, but most teams don't have them for their own data, and labeling a few thousand results takes weeks. Jev is TypeSafe's "System One" model: you ask it a typed question, like "how well does this result match the query, on this 4-level scale?", and it returns a probability for each level in about 100 ms, for a small fraction of a cent.

A cheap judge is only useful if you know how far to trust it, so this walkthrough has two parts:

1. **Calibrate on a dataset with human labels.** Amazon's [Shopping Queries dataset (ESCI)](https://github.com/amazon-science/esci-data) has real shopping queries graded by people (Apache-2.0). You run several retrievers and rerankers, score them with the human labels, then have Jev grade the same results and check whether it ranks the systems the way the humans did.
2. **Point it at your own data.** Load your documents and queries, write a rubric that says what "relevant" means for your users, and let Jev grade. Same scripts, same tables, no labels needed.

turbopuffer does the retrieval (BM25, vector, and hybrid, plus hosted rerankers), and DuckDB keeps every run, result, grade, and latency in one file and computes the metrics in SQL.

On 500 ESCI queries over all 482k products, Jev reranked turbopuffer's hybrid results about as well as the best hosted reranker I tried (Voyage's rerank-3): 0.011 behind on standard nDCG@10, with a 95% interval that just reaches zero, and 0.003 ahead on condensed nDCG. As a judge, it had the highest exact agreement with the human graders of the seven I tested, 8 points ahead of the best LLM, at a quarter to a sixth of the LLMs' latency. Asked to rank eight search setups, it put 26 of the 28 pairs in the same order as the human labels (on standard nDCG@10). One of the two it flipped was a real difference. The numbers are [below](#what-the-500-query-run-showed), and [`experiments/evidence.py`](experiments/evidence.py) recomputes every one of them from the warehouse ([how much to trust them](#how-much-to-trust-these-numbers)).

## How the pieces fit

```
your queries ─▶ turbopuffer ─▶ top 50 per query ─▶ reranker (optional) ─▶ results
                                                                              │
          rubric (rubrics/*.toml) ─▶ Jev or an LLM grades the top 10 ─────────┤
                                                                              ▼
                              DuckDB: runs, results, judgments ─▶ nDCG@10, recall, CIs
```

Everything lands in `warehouse/relevancy.duckdb`:

| Table | One row per |
| --- | --- |
| `datasets` | dataset, and the rubric its judges use |
| `queries`, `docs` | query and document |
| `runs` | retriever and reranker configuration you tried, with its config and git commit |
| `results` | ranked result (run, query, rank, doc, latency) |
| `judgments` | grade, from a human or a judge, keyed by `source` |
| `metrics` | run, judge, metric, and query slice |

A judge is a name, and its grades are rows. `human`, `jev-batched:jev-1.13.0`, and `openai/gpt-4o-mini` are all just values of `judgments.source`, so the same nDCG query scores a run with any of them.

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
| `JEV_MODEL` | | defaults to `jev-1.13.0`, the version turbopuffer hosts and Elastic tested. TypeSafe's API serves newer ones too. |

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

It also registers the dataset with the `esci` rubric, Amazon's own grading guidelines in [`rubrics/esci.toml`](rubrics/esci.toml).

### 3. Load products into turbopuffer

```bash
uv run scripts/03_load_turbopuffer.py --scope smoke
```

The `smoke` scope loads every product judged for your sampled queries, plus 20,000 random products as distractors. Without them the corpus holds only judged products, and every retriever scores higher than it would on real data. For a 50-query sample that's about 21,000 products, 3.5M embedding tokens, and 3 minutes.

Each product becomes a turbopuffer document with:

- `title` and `body`, full-text indexed for BM25
- `text`, made from the title, brand, color, and the first 800 characters of the description. turbopuffer embeds this field itself, so you don't need a separate embedding API, and rerankers and judges read it too.
- `brand` and `color`, filterable

New turbopuffer orgs get 2M embedding tokens per minute per model ([embedding docs](https://turbopuffer.com/docs/embedding)). Eight parallel writers hit that in a few seconds, so the script paces itself to 1.5M tokens a minute and backs off if it still gets a 429. Each request already carries 256 products, the most turbopuffer accepts when it does the embedding ([limits](https://turbopuffer.com/docs/limits)), so the token limit is what sets the speed. If your org has a higher limit, raise `--tokens-per-minute`.

`--scope full` loads all 482k products. That's every product ESCI judged for at least one query, so even the full corpus has no products from outside the labeled set; a product just isn't labeled for most queries. Real catalogs have more junk to wade through, so expect absolute scores on your own data to run lower. The full load is about 80M embedding tokens (§17 of [`evidence.txt`](experiments/results/evidence.txt)). That's an estimate: turbopuffer reported 42.6M tokens for the first 286,720 products (171M characters, so 4.0 characters a token), and the whole corpus is 323M characters, so the later products are longer. My first full run wrote those 286,720 products in 42 minutes, about 1M tokens a minute against the pacer's 1.5M budget, then stopped on a 429 the script didn't catch yet. I didn't keep the log of the run that finished, so I don't have its final token count or time. With a higher embedding limit, or vectors you computed yourself, the load would go faster. A load that long can outlast the rate-limit retries on a batch or two. Mine stopped at 480k products, and `--resume` (which lists what the namespace already has and writes only the rest) finished the last 1,024 in a few seconds.

### 4. Search

```bash
uv run scripts/04_search.py --scope smoke
```

This runs four retrievers over every sampled query and keeps the top 50 from each:

- **bm25:** BM25 on title and body, with title counted twice
- **bm25-title:** BM25 on the title alone, a deliberately weak baseline so you can see whether a judge notices it's worse
- **ann:** vector search over the embedded `text`
- **hybrid-rrf:** BM25 and vector in one `multi_query` call, merged on the server with reciprocal rank fusion

Every result lands in the warehouse with its rank, score, wall-clock latency, turbopuffer's server time, and billable bytes.

### 5. Score with the human labels

```bash
uv run scripts/05_score.py
uv run scripts/05_score.py --by-slice     # nDCG@10 per query slice
uv run scripts/05_score.py --gain linear  # grades as gains, instead of 2^grade - 1
```

The metrics are SQL in [`src/relevancy/metrics.py`](src/relevancy/metrics.py):

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

For an eval framework, single-grade agreement matters less than whether the judge picks the right winner between config A and config B. Step 8 pools the top 10 from every run, has each judge grade the pool, scores every run under each judge, and compares the judge's ranking of runs with the humans' (Kendall's tau, where 1 means the same order).

Two rules keep it honest. A run reranked by Jev is never scored by Jev, because the judge would be grading results it sorted itself. And when a judge scores a run, its ideal ranking comes from the same pool it graded, so no run loses points for products nobody showed the judge.

Grades are saved as they come back, every 25 queries, and pairs already graded are skipped on a rerun.

One thing to know about a judge's nDCG: its ideal ranking is built from the pool, so a run's score depends on which other runs were pooled with it. Voyage scores 0.897 under Jev when pooled with three other runs, 0.861 with eight, and 0.854 with nine (evidence §9). Compare judge scores only between runs scored together in the same step 8 call.

## What the 500-query run showed

All 482,105 products in the namespace, 500 stratified queries, `aws-us-east-1`, scripts run from a laptop, so wall-clock times include the trip over the public internet. Every reranker reorders the same hybrid top 50. The section numbers (§) point to the matching part of [`experiments/results/evidence.txt`](experiments/results/evidence.txt).

| Retriever | Reranker | nDCG@10 [95% CI] | vs hybrid [95% CI] | Condensed | vs hybrid, condensed | MRR@10 | Wall p50 ms | turbopuffer server p50 / p99 ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hybrid-rrf | voyage/rerank-3 | 0.463 [0.436, 0.489] | +0.096 [+0.080, +0.111] | 0.635 | +0.025 [+0.019, +0.032] | 0.695 | 362 | 248 / 535 |
| hybrid-rrf | jev:jev-1.13.0 | 0.452 [0.425, 0.478] | +0.085 [+0.068, +0.102] | 0.638 | +0.029 [+0.021, +0.036] | 0.685 | 499 | 103 / 322 + Jev |
| hybrid-rrf | zeroentropy/zerank-2 | 0.441 [0.414, 0.467] | +0.074 [+0.057, +0.090] | 0.630 | +0.020 [+0.013, +0.027] | 0.664 | 412 | 291 / 533 |
| hybrid-rrf | cohere/rerank-v3.5 | 0.436 [0.409, 0.462] | +0.070 [+0.054, +0.085] | 0.625 | +0.015 [+0.009, +0.022] | 0.657 | 433 | 314 / 728 |
| hybrid-rrf | llm:google/gemma-3-27b-it | 0.435 [0.408, 0.460] | +0.068 [+0.050, +0.085] | 0.637 | +0.027 [+0.019, +0.034] | 0.666 | 3,520 | 103 / 322 + Gemma |
| hybrid-rrf | none | 0.367 [0.340, 0.392] | base | 0.610 | base | 0.592 | 160 | 103 / 322 |
| ann | none | 0.350 [0.322, 0.376] | -0.017 [-0.032, -0.002] | 0.583 | -0.026 [-0.042, -0.012] | 0.572 | 130 | 75 / 96 |
| bm25-title | none | 0.332 [0.305, 0.356] | -0.035 [-0.052, -0.019] | 0.525 | -0.085 [-0.102, -0.069] | 0.548 | 69 | 15 / 36 |
| bm25 | none | 0.311 [0.286, 0.335] | -0.056 [-0.069, -0.043] | 0.523 | -0.087 [-0.102, -0.072] | 0.528 | 156 | 100 / 300 |

Recall@50 is 0.471 for every hybrid row, since rerankers only reorder (§1). Wall time is what my laptop saw, about 45 to 55 ms of it the round trip to `aws-us-east-1`; server time is what turbopuffer reports, summed over both requests for the hosted rerankers. Jev and Gemma run outside turbopuffer, so their server column is only the retrieval.

What I take from it:

- **Every reranker beats plain hybrid on both measures.** The paired intervals sit above zero, from +0.068 to +0.096 on standard nDCG@10 and +0.015 to +0.029 on condensed (§1). The condensed gains are 2.5 to 4.7 times smaller. Reranked runs also have fewer unlabeled products in their top 10 (55% to 60%, against 63% for hybrid, §3), so part of a reranker's gain on standard nDCG likely comes from pulling labeled products up past unlabeled ones, which condensed nDCG ignores. The order shifts too: on condensed, Jev (+0.029) and Gemma (+0.027) edge past Voyage (+0.025), but those three are within 0.004 of each other and no pair of them is significant even before correcting for multiple comparisons (p from 0.22 to 0.57, §4).
- **Jev and Voyage are close, and which one "wins" depends on how you count unlabeled products.** Head to head, Voyage leads Jev by 0.011 on standard nDCG@10 (95% CI -0.022 to 0.000, p = 0.08). On condensed nDCG, Jev leads by 0.003 (-0.002 to +0.009). Per query, Jev did better on 198, Voyage on 202, and 100 tied (§2). The standard-nDCG gap tracks the labels: 58% of Jev's top 10 has no human label, against 55% of Voyage's (§3), and standard nDCG counts every unlabeled product as irrelevant. I'd call it a tie I can't break with 500 queries. A judge doesn't have this problem, because it grades whatever your system returns.
- **Jev beats Cohere and Gemma as a reranker** on standard nDCG (by 0.016 and 0.018, both intervals above zero), and is level with zerank-2 (+0.011, interval -0.000 to +0.022) (§2).
- **Gemma, read through logprobs, reranks about as well as Cohere** but takes 3.5 seconds a query, which rules it out for serving. Per query, Jev added a median 323 ms over the network from my laptop. Voyage, called through turbopuffer's `rerank_by`, added 191 ms of wall time, of which 132 ms was turbopuffer's server time and the rest a second round trip (§1). Zerank-2 and Cohere added 181 and 197 ms of server time.
- **Vector search beat BM25, and fusing them beat both** (§1). That's with this BM25 setup (title counted twice, plus a long, noisy body field) and voyage-4-lite, a small, cheap embedding model; other choices could change it.
- **BM25 over titles alone beat BM25 over title and body on standard nDCG, but that's a label artifact.** On condensed nDCG the two are 0.002 apart (p = 0.77, §4), and title-only BM25 has slightly fewer unlabeled products in its top 10 (65.5% against 66.9%, §3). Treat them as equal.
- The 50-query smoke run had the right order but wide intervals. It's good for checking the pipeline, but too small to choose between rerankers.

`uv run scripts/05_score.py --by-slice` breaks this down by query type (§13). Most queries (278 of 500) are medium-length and unbranded. The hardest slices are short unbranded (83 queries, hybrid 0.299, best Voyage at 0.377) and long unbranded (58 queries, hybrid 0.293, best Gemma at 0.359). Medium and long queries that name a brand score highest (hybrid 0.507 and 0.502). The branded short and long slices have 15 and 11 queries, too few to read a winner from.

### Judges against humans (step 7)

200 held-out pairs, 50 per grade. Brackets are 95% bootstrap intervals over pairs (§6). Latency and cost are from step 7's printout on my run; the warehouse doesn't store them, so the latency lines are saved in [`experiments/results/judge_bakeoff_latency.txt`](experiments/results/judge_bakeoff_latency.txt).

| Judge | Exact [95% CI] | Within 1 | Weighted kappa [95% CI] | r(expected, human) | p50 ms | Cost |
| --- | --- | --- | --- | --- | --- | --- |
| jev-1.14.0 | 0.55 [0.48, 0.62] | 0.78 | 0.41 [0.28, 0.54] | 0.48 | 118 | under $0.01 |
| jev-1.13.0 | 0.52 [0.45, 0.58] | 0.75 | 0.38 [0.25, 0.50] | 0.46 | 110 | under $0.01 |
| jev-1.13.0, batched | 0.52 [0.46, 0.59] | 0.75 | 0.37 [0.25, 0.49] | 0.45 | 111 | under $0.01 |
| gemma-3-27b-it | 0.44 [0.38, 0.50] | 0.74 | 0.33 [0.20, 0.46] | 0.34 | 501 | $0.004 |
| llama-4-maverick | 0.41 [0.35, 0.48] | 0.78 | 0.37 [0.24, 0.49] | 0.39 | 435 | $0.015 |
| gpt-4o-mini | 0.39 [0.32, 0.45] | 0.74 | 0.34 [0.22, 0.47] | 0.40 | 656 | $0.009 |
| llama-3.1-8b-instruct | 0.36 [0.29, 0.43] | 0.66 | 0.28 [0.14, 0.41] | 0.34 | 500 | $0.002 |

None of these judges matches the humans on single grades very often. Some of that is the task: whether a product is Exact or a Substitute often turns on a size or brand detail that people read differently.

With 200 pairs the intervals are wide, so here's what holds up when you compare judges on the same pairs (§6):

- **Jev 1.13.0 beats the LLMs on exact agreement.** It's 0.080 ahead of Gemma (interval +0.005 to +0.160) and 0.105 ahead of llama-4-maverick (+0.035 to +0.170). On kappa the intervals overlap, so I can't say Jev beats maverick there.
- **It's four to six times faster** than the LLMs (110 ms against 435 to 656 ms).
- **Jev 1.14.0 against 1.13.0 is a wash on these 200 pairs** (+0.035 exact, interval -0.010 to +0.080). On the 3,692 pooled pairs in step 8 the gap is clear; see below.
- **Batching doesn't change the answers beyond run-to-run noise.** Every bake-off query has one pair, so `jev:` and `jev-batched:` sent identical requests here; their grades still differed on 4.5% of pairs, which is Jev's test-retest noise. A separate experiment regraded 3,177 pooled pairs in real 50-product batches, once in the same order and once reversed: 97.7% and 97.3% of grades matched the stored ones (§12). Order inside a batch makes no difference I can measure.

### Judges ranking systems (step 8)

Jev and Gemma each graded the pooled top 10 of all nine runs, about 15,000 (query, product) pairs per judge. Only about a quarter of those pairs have a human label, which is the usual situation: whatever your system returns, most of it was never graded.

nDCG@10 for each run under each judge:

| Run | Human | Human, condensed | Jev 1.13.0 | Gemma 3 27B |
| --- | --- | --- | --- | --- |
| hybrid + voyage/rerank-3 | 0.463 | 0.635 | 0.854 | 0.843 |
| hybrid + jev:jev-1.13.0 | 0.452 | 0.638 | *0.983* | 0.868 |
| hybrid + zeroentropy/zerank-2 | 0.441 | 0.630 | 0.858 | 0.835 |
| hybrid + cohere/rerank-v3.5 | 0.436 | 0.625 | 0.797 | 0.799 |
| hybrid + gemma-3-27b-it | 0.435 | 0.637 | 0.845 | *0.987* |
| hybrid-rrf | 0.367 | 0.610 | 0.666 | 0.696 |
| ann | 0.350 | 0.583 | 0.656 | 0.673 |
| bm25-title | 0.332 | 0.525 | 0.565 | 0.603 |
| bm25 | 0.311 | 0.523 | 0.553 | 0.604 |

The italic cells show why a judge never scores its own reranker. Jev rates the Jev-reranked run 0.983 and Gemma rates the Gemma-reranked run 0.987: of course they do, each one sorted the results by its own opinion. Those runs are left out of that judge's agreement numbers.

Tau intervals resample the 500 queries; agreement intervals resample queries too, keeping each query's pairs together (§7). All three judges are compared on the same 3,692 labeled pool pairs.

| Judge | Runs ranked | Kendall's tau vs human [95% CI] | vs human, condensed | Exact agreement [95% CI] | Weighted kappa [95% CI] |
| --- | --- | --- | --- | --- | --- |
| Jev 1.13.0 | 8 | 0.857 [0.71, 1.00] | 0.786 [0.64, 0.93] | 0.599 [0.570, 0.630] | 0.436 [0.387, 0.483] |
| Jev 1.14.0 | 8 | 0.857 | 0.786 | 0.631 | 0.471 |
| Gemma 3 27B | 8 | 0.857 [0.79, 0.93] | 0.929 [0.86, 1.00] | 0.575 [0.547, 0.600] | 0.402 [0.354, 0.445] |

A tau of 0.857 across 8 runs means 26 of the 28 pairs of runs come out in the same order as with the human labels. For Jev, the two pairs that flipped are Cohere against Gemma, which the humans have 0.001 apart, and Voyage against zerank-2. That second one is a real miss: the humans put Voyage ahead by 0.022 (paired CI 0.012 to 0.031, §2), and Jev has zerank-2 ahead by 0.004. The decisions with big gaps all match: reranked beats not reranked, hybrid beats vector alone, vector beats BM25. Only 9 of the 28 pairs are within 0.03 of each other on human nDCG (§7), and those are where every flip happens. So I'd trust a Jev verdict between setups that differ by a lot, and want human labels before acting on a difference of 0.02.

**Jev 1.14.0 is the better judge.** Graded against the same pool, it agrees with the humans on 0.031 more of the labeled pairs than 1.13.0 (interval +0.019 to +0.045), and its kappa is 0.471 against 0.436 (§15). Its tau is the same (from a step 8 run on a copy of the warehouse, logged in [`experiments/README.md`](experiments/README.md)), and as a reranker it's level with 1.13.0 (-0.001, interval -0.007 to +0.004). The two versions give the same grade on 87.8% of pairs. If you're starting fresh, use 1.14.0 as the judge; I kept 1.13.0 for the main runs to match what turbopuffer hosts.

Single-grade agreement is higher here than in step 7 (0.60 against 0.52 for Jev), and it isn't because the pool is mostly Exact products. Reweighting the pool to an even mix of grades still gives 0.572 (§8). The difference is which products of each grade show up. On human-Exact products, Jev agrees 66% of the time in the pool but 36% in the bake-off. The Exacts a good search system returns are the obvious ones, while the bake-off draws Exacts at random, including the borderline cases.

Gemma ties Jev on standard nDCG and has a higher tau on condensed nDCG (0.929 against 0.786), but the intervals overlap, so I can't call Gemma better there. It's also four to five times slower. For an eval loop you run on every change, I'd use Jev and check it against a few hundred human labels on your own data now and then (the protocol is [below](#how-much-to-trust-these-numbers)).

## How much to trust these numbers

`uv run experiments/evidence.py` recomputes every figure above from the warehouse and writes [`experiments/results/evidence.txt`](experiments/results/evidence.txt). It's read-only and takes a few seconds. Here's what it says about the limits:

- **500 queries can separate rerankers about 0.012 to 0.019 apart** on standard nDCG@10 (80% power, §5). Voyage against Jev (0.011) is below that, which is why I call it a tie.
- **Many comparisons, a few false alarms.** I compared 36 pairs of runs. 30 are significant at p < 0.05, and 26 survive a Bonferroni correction (§4). The four that drop out are two close reranker pairs (Jev against Gemma and against Cohere), BM25 title-only against title and body, and hybrid against vector. I've avoided naming a winner on the reranker pairs.
- **The gain function doesn't matter here.** Exponential, linear, and binary gains put the nine runs in the same order (§10).
- **Jev isn't perfectly repeatable.** The same request sent twice gave a different grade 4.5% of the time (§6). Over hundreds of queries that averages out, but don't read much into one query's grades.
- **The human labels aren't ground truth either.** ESCI is built from "difficult Amazon search queries and results" ([Reddy et al., 2022](https://arxiv.org/abs/2206.06588)), so the labeled products are ones Amazon's search returned. On negation queries, 74% of the disagreements I audited were the human label getting it wrong (§11).
- **Jev and the LLMs may have seen ESCI in training.** It's a public dataset from 2022. I can't rule out that some of the agreement is memory, which is one more reason to calibrate on your own labels.
- **The judges read a shortened product.** They see the title, brand, color, and first 800 characters of the description, and the humans saw the full product page. Raising the limit to 4,000 characters didn't help: exact agreement went from 0.600 to 0.606 and kappa from 0.410 to 0.406 on 1,091 pairs (§12).

### Calibrating on your own data

The ESCI numbers tell you Jev can rank systems. They don't tell you it understands your domain. Before you trust it on your data:

1. Label at least 300 (query, result) pairs yourself, from the pooled top 10 of the runs you care about, so the labels cover what your systems return, not a random sample of the catalog.
2. Load them with `02_prepare_custom.py --judgments` and run step 8. Look at exact agreement and kappa with their intervals, and at the disagreements query by query. They'll tell you which rubric lines to rewrite.
3. Keep 50 of those pairs aside as a retest set. Regrade them whenever you change the rubric or the Jev version, and if agreement drops, find out why before you trust new results.

## Follow-up experiments

[`experiments/`](experiments/) has the follow-ups: the negation audit, batch order, description length, Jev 1.14.0, and two that look at turbopuffer itself. [`experiments/README.md`](experiments/README.md) lists each one with its command and result.

### What turbopuffer did during the runs

[`experiments/turbopuffer_perf.py`](experiments/turbopuffer_perf.py) reran the three first-stage retrievers twice and saved every field turbopuffer returns ([`turbopuffer_perf.txt`](experiments/results/turbopuffer_perf.txt)). [`experiments/retrieval_followups.py`](experiments/retrieval_followups.py) tried BM25 field variants, hybrid and reranking in one request, a brand filter, and Elastic's judged-only setup ([`retrieval_followups.txt`](experiments/results/retrieval_followups.txt)). Every query ran one at a time from my laptop against `aws-us-east-1`, so none of this says anything about throughput under concurrency.

**Most of the wall time was network.** The laptop's round trip was a median of about 54 ms (§1), and turbopuffer's server time was 76 ms for ANN, 102 ms for BM25, and 103 ms for hybrid at p50. Of ANN's 76 ms, 66 ms was embedding the query text. Hybrid's server time is about BM25's alone, not the sum of the two.

**The BM25 tail comes from the body field and long queries.** Body-only BM25 went from a 29 ms median on one-word queries to 260 ms on queries of six words or more, and title-only BM25 stayed between 13 and 29 ms. Title-only BM25 also scored higher on nDCG@10 (0.332 against 0.311, +0.020 [+0.008, +0.033]), though most of that is the label artifact described in [the results](#what-the-500-query-run-showed).

**The cache was warm and the results were stable.** Every query in both passes reported `cache_temperature: hot`, so these are warm numbers; I didn't test a cold namespace. The second pass returned the same top 10 for 100% of BM25 queries and 99.4% of ANN and hybrid queries.

**ANN gave up almost no recall.** turbopuffer's recall endpoint measured 0.994 at 50 and 0.995 at 10 against exhaustive search, over 100 query vectors it samples from the namespace. It doesn't take text queries, so this measures the index rather than these exact 500 queries.

**2% of the namespace was still unindexed.** Every response reported 9,785 documents (of 482,105) searched exhaustively because they weren't indexed yet. I didn't measure how long indexing takes to catch up.

**Hybrid plus a hosted reranker works as one request, and it's better.** turbopuffer's [reranking docs](https://turbopuffer.com/docs/reranking) allow `rerank_by` on a `multi_query`, which reranks the union of the subqueries' results. With Voyage, median wall time fell from 362 ms (two requests) to 308 ms, server time stayed level (248 against 245 ms), and nDCG@10 rose 0.006 [+0.003, +0.010], likely because the reranker sees the union instead of the fused top 50. Asking each subquery for 100 results raised it 0.016 [+0.010, +0.022] at 323 ms.

**Filters cost almost nothing in time.** On the 81 sampled queries that name a brand, filtering hybrid search to that brand moved median server time from 79 to 77 ms and raised nDCG@10 from 0.477 to 0.524 (+0.047 [+0.001, +0.096]). I took the brand from the products humans labeled Exact, so that gain is an upper bound on what a real brand detector would get.

**On judged products only, Jev beats Voyage.** [Elastic](https://www.elastic.co/search-labs/blog/ecommerce-search-reranking-llm-alternative-jev) reranked only each query's judged products. Doing the same on these 500 queries, Jev scored 0.868 nDCG@10 and Voyage 0.848 (+0.020 [+0.010, +0.029]); with the KDD Cup gains (Exact 1, Substitute 0.1, Complement 0.01) it was 0.817 against 0.801 (+0.016 [+0.003, +0.029]). Elastic reported 0.9565 for Jev and 0.8595 for a random order. Ordering by product id scored 0.679 here, so the two setups don't compute nDCG the same way and the absolute numbers don't compare. The direction does. On turbopuffer's real top 50 they tie.

**Still untested:** throughput and tail latency from inside the region under concurrent load, a cold namespace, write throughput with vectors computed ahead of time instead of embedded by turbopuffer, and how long indexing takes to catch up after a big load.

## Part 2: your own data

Once you've seen how far Jev agrees with humans on ESCI, the same scripts run on your data. You need two files: documents and queries.

```bash
uv run scripts/02_prepare_custom.py --dataset mydocs \
    --docs path/to/docs.parquet --queries path/to/queries.csv --rubric generic
uv run scripts/03_load_turbopuffer.py --dataset mydocs --scope full
uv run scripts/04_search.py --dataset mydocs --scope full
uv run scripts/06_rerank.py --dataset mydocs --model voyage/rerank-3     # optional
uv run scripts/08_judge_runs.py --dataset mydocs
uv run scripts/05_score.py --dataset mydocs --judge jev-batched:jev-1.13.0
```

Files can be CSV, Parquet, JSON, or JSONL:

- **docs:** `id`, plus `title` and/or `body`. `brand` and `color` are optional and become filterable attributes.
- **queries:** `query_id` and `text`. Pull a few hundred real queries from your search logs; a stratified sample across query types beats the top 500 by volume, which are mostly easy. An optional `slice` column lets step 5 break results down by type.
- **judgments** (optional): `query_id`, `doc_id`, `grade` from 0 to 3. If you have even a hundred human labels, load them. Step 8 will then check Jev against them on your data, which beats any number from ESCI.

With no human labels, step 8 grades the pool and reports each run's nDCG@10 under Jev, and step 5 with `--judge` gives you the paired difference against your current setup. That's the number to put in front of whoever approves the change.

### Write a rubric

The rubric is where you tell the judge what "relevant" means. [`rubrics/generic.toml`](rubrics/generic.toml) is a reasonable start:

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

After editing a rubric, regrade with `uv run scripts/08_judge_runs.py --dataset mydocs --regrade`.

## Querying the warehouse

Every number in this README came out of `warehouse/relevancy.duckdb`, and it answers plenty of questions the scripts don't ask. DuckDB needs nothing beyond `uv sync`: it's a Python library, so there's no server to run and no extension to install.

### Open it

From Python, with nothing else installed:

```bash
uv run python -c "import duckdb; duckdb.connect('warehouse/relevancy.duckdb', read_only=True).sql('SELECT retriever, reranker, count(*) FROM runs GROUP BY ALL').show()"
```

For poking around, the DuckDB command-line shell is nicer. Install it (`brew install duckdb` on a Mac, or a download from [duckdb.org](https://duckdb.org/docs/installation/)) and open the file read-only:

```bash
duckdb -readonly warehouse/relevancy.duckdb
```

In Python, `.show()` prints a table, `.fetchall()` returns tuples, and `.arrow()` returns a pyarrow table. `.df()` needs pandas, which this repo doesn't install (`uv add pandas` if you want it).

### Things to know before you query

- **One process writes at a time.** DuckDB locks the file while a process has it open for writing, and a second process gets an error instead of waiting, even a read-only one. The scripts work around it: they read what they need, close the connection, do the slow network work, and reopen to write. If a script is mid-write when you open the file, wait a second and retry, or copy the file and query the copy.
- **`RELEVANCY_DB` points every script at another file.** `RELEVANCY_DB=/tmp/scratch.duckdb uv run scripts/02_prepare_custom.py ...` builds a separate warehouse. That's how I tested the custom-data path without touching the real one, and it's the way to run two experiments at once.
- **The schema is applied on every connect.** `warehouse/schema.sql` uses `CREATE TABLE IF NOT EXISTS` and `ADD COLUMN IF NOT EXISTS`, so to add a column, add it there and the next script run picks it up.
- **Per-query fields repeat on every result row.** `latency_ms`, `server_ms`, and `billable_bytes` belong to the query, but they sit on each of its 50 rows. Summing them over `results` overcounts 50 times; take `DISTINCT run_id, query_id` first, as the latency query below does. Reranked runs leave `billable_bytes` empty.
- **`metrics` holds averages, not per-query scores.** Step 5 recomputes per-query nDCG with `relevancy.metrics.per_query()` and stores only the run-level numbers. For query-by-query work, call that function (the head-to-head recipe below shows how). It works on a read-only connection.
- **Runs are tied to a query sample.** Each run's `config` stores a fingerprint of the sample it searched (`config->>'sample'`). Rerunning step 2 with a new sample size changes the fingerprint, and scoring skips the old runs. The sample is drawn with DuckDB's `hash()`, which can change between major DuckDB versions, so an upgrade can change it too.

### The tables

| Table | Key | Columns worth knowing |
| --- | --- | --- |
| `datasets` | `dataset` | `rubric`, the file in `rubrics/` its judges use |
| `queries` | `dataset, query_id` | `text`, `slice` (like `short\|brand`), `sampled` |
| `docs` | `dataset, doc_id` | `title`, `body`, `brand`, `color` |
| `judgments` | `dataset, query_id, doc_id, source` | `grade` (0 to 3), `prob` and `expected` from judges that return probabilities |
| `runs` | `run_id` | `retriever`, `reranker`, `config` (JSON: namespace, region, sample, base run), `git_sha`, `started_at` |
| `results` | `run_id, query_id, rank` | `doc_id`, `score`, `latency_ms`, `server_ms`, `billable_bytes` |
| `metrics` | `run_id, judge, metric, slice` | `value`, `ci_low`, `ci_high` |

`judgments.source` is `human`, a judge name like `jev-batched:jev-1.13.0`, or a bake-off grade like `jev:jev-1.13.0@bakeoff`. After the 500-query run the file is 784 MB: 482,105 docs, 8,956 queries, about 214,000 judgments, and about 254,000 result rows.

### Recipes

Each of these runs as is against a warehouse built by the walkthrough. Swap in your own run ids from the first one.

**Runs on the current sample.** The fingerprint is the same expression `warehouse.sample_id()` uses.

```sql
SELECT run_id, retriever, reranker, config->>'namespace' AS namespace, git_sha, started_at
FROM runs
WHERE dataset = 'esci-us'
  AND config->>'sample' = (SELECT count(*) || '-' || (hash(string_agg(query_id, ',' ORDER BY query_id)) % 1000000)
                           FROM queries WHERE dataset = 'esci-us' AND sampled)
ORDER BY started_at;
```

**Every run under every judge, side by side.**

```sql
PIVOT (
  SELECT r.retriever || ' / ' || r.reranker AS run, m.judge, round(m.value, 3) AS ndcg10
  FROM metrics m JOIN runs r USING (run_id)
  WHERE m.metric = 'ndcg10' AND m.slice = 'all' AND r.config->>'namespace' = 'esci-us-full'
) ON judge USING first(ndcg10) ORDER BY run;
```

**One query's top 10, with what the humans and Jev said.** This is the query I run most. Reading ten real results next to their grades tells you more than any average.

```sql
SELECT r.rank, left(d.title, 60) AS title, h.grade AS human, j.grade AS jev, round(j.expected, 2) AS jev_expected
FROM results r
JOIN docs d ON d.dataset = 'esci-us' AND d.doc_id = r.doc_id
LEFT JOIN judgments h ON h.dataset = 'esci-us' AND h.query_id = r.query_id
                     AND h.doc_id = r.doc_id AND h.source = 'human'
LEFT JOIN judgments j ON j.dataset = 'esci-us' AND j.query_id = r.query_id
                     AND j.doc_id = r.doc_id AND j.source = 'jev-batched:jev-1.13.0'
WHERE r.run_id = 'hybrid-rrf-full-4cea30'        -- your run id
  AND r.query_id = (SELECT query_id FROM queries WHERE dataset = 'esci-us' AND sampled ORDER BY query_id LIMIT 1)
  AND r.rank <= 10
ORDER BY r.rank;
```

On my first sampled query, "t shirt dresses for women sexy", the humans had graded 2 of hybrid's top 10. Jev graded all 10 and called 6 of them Exact (§16).

**Where Jev and the humans disagree hardest.**

```sql
SELECT q.text AS query, left(d.title, 50) AS title, h.grade AS human, j.grade AS jev, round(j.prob, 2) AS jev_prob
FROM judgments h
JOIN judgments j USING (dataset, query_id, doc_id)
JOIN queries q USING (dataset, query_id)
JOIN docs d USING (dataset, doc_id)
WHERE h.dataset = 'esci-us' AND h.source = 'human' AND j.source = 'jev-batched:jev-1.13.0'
  AND abs(h.grade - j.grade) = 3
ORDER BY j.prob DESC
LIMIT 20;
```

This one found something I hadn't looked for. The top of the list was "boys swim trunks without mesh lining", "face masks without nose wire", "hand sanitizer without alcohol", and "1 12 off-road vehicle not rc": products the humans called Exact and Jev called Irrelevant with near-certainty. Splitting the human-Exact products on a negation word in the query makes it a number. The regex also catches the flower "forget me not", so the query drops it:

```sql
SELECT regexp_matches(lower(q.text), '\b(without|not|no|non)\b')
           AND NOT contains(lower(q.text), 'forget me not') AS negation,
       count(DISTINCT q.query_id) AS queries,
       count(*) AS human_exact,
       round(avg((j.grade = 0)::INT), 3) AS jev_called_irrelevant
FROM judgments h
JOIN judgments j USING (dataset, query_id, doc_id)
JOIN queries q USING (dataset, query_id)
WHERE h.dataset = 'esci-us' AND h.source = 'human' AND h.grade = 3
  AND j.source = 'jev-batched:jev-1.13.0' AND q.sampled
GROUP BY ALL;
```

On 138 human-Exact products from 27 negation queries, Jev said Irrelevant 31.2% of the time, against 6.6% of 2,121 on the other 409 queries (§11).

So who's wrong? [`experiments/negation_audit.py`](experiments/negation_audit.py) exports every one of those disagreements with both grades hidden, for a third rater to answer one question: does the product have the thing the query excludes? I had a Claude model do the rating, given only the query and the product text. Of the 43 pairs on real negations, the product had the excluded thing in 32 (74%), so the human label was wrong and Jev was right. Jev was wrong on 4, and on 7 the product text didn't say (§11). My guess is that ESCI's graders matched on the product type and skipped the "without". That's worth knowing before you trust any human labels on negation queries, and it's the kind of slice I'd watch on my own data.

**How Jev's grades line up with the humans' (a confusion matrix).**

```sql
PIVOT (
  SELECT h.grade AS human, j.grade AS jev
  FROM judgments h JOIN judgments j USING (dataset, query_id, doc_id)
  WHERE h.dataset = 'esci-us' AND h.source = 'human' AND j.source = 'jev-batched:jev-1.13.0'
) ON jev USING count(*) ORDER BY human;
```

Rows are the human grade and columns are Jev's. This counts every pair both have graded, including the 50-query smoke run's pool. Jev's most common miss is grading a human Substitute (2) as Irrelevant (0): 216 of 1,075.

**How much of each run's top 10 has no human label.** This is the number behind the Jev and Voyage tie.

```sql
SELECT ru.retriever || ' / ' || ru.reranker AS run,
       round(avg((h.doc_id IS NULL)::INT), 3) AS unlabeled_share_top10
FROM results r
JOIN runs ru USING (run_id)
LEFT JOIN judgments h ON h.dataset = ru.dataset AND h.source = 'human'
                     AND h.query_id = r.query_id AND h.doc_id = r.doc_id
WHERE ru.config->>'namespace' = 'esci-us-full' AND r.rank <= 10
GROUP BY ALL ORDER BY 2;
```

**Latency and query cost per run.** Note the `DISTINCT`, which collapses the 50 repeated rows per query.

```sql
WITH per_query AS (
  SELECT DISTINCT run_id, query_id, latency_ms, server_ms, billable_bytes FROM results
)
SELECT ru.retriever || ' / ' || ru.reranker AS run,
       count(*) AS queries,
       round(quantile_cont(latency_ms, 0.5)) AS p50_ms,
       round(quantile_cont(latency_ms, 0.99)) AS p99_ms,
       round(sum(billable_bytes) / 1e12, 2) AS tb_queried,
       round(sum(billable_bytes) / 1e15, 4) AS usd_at_1_per_pb
FROM per_query JOIN runs ru USING (run_id)
WHERE ru.config->>'namespace' = 'esci-us-full'
GROUP BY ALL ORDER BY p50_ms;
```

BM25, title-only BM25, and vector search each queried 0.85 TB across 500 queries, and hybrid queried 1.71 TB because it runs two subqueries (§14). At list price that's under $0.002 per run.

**One run by query slice.** Intervals are stored for `all` only.

```sql
SELECT slice, round(value, 3) AS ndcg10
FROM metrics
WHERE run_id = 'jev-1.13.0-b53bce' AND judge = 'human' AND metric = 'ndcg10'
ORDER BY value;
```

**Head to head, query by query (Python).** Per-query scores come from `per_query()`, and DuckDB can join a pyarrow table by its variable name:

```python
import duckdb
import pyarrow as pa
from relevancy.metrics import per_query

con = duckdb.connect("warehouse/relevancy.duckdb", read_only=True)
a, b = "jev-1.13.0-b53bce", "rerank-3-977611"   # your run ids
pq = per_query(con, "esci-us", [a, b], judge="human")
diffs = pa.Table.from_pylist([
    {"query_id": q, "a": m["ndcg10"], "b": pq[(b, q)]["ndcg10"]}
    for (run, q), m in pq.items() if run == a and m["ndcg10"] is not None
])
con.sql("""
    SELECT q.text, round(a, 3) AS a, round(b, 3) AS b, round(a - b, 3) AS diff
    FROM diffs JOIN queries q ON q.dataset = 'esci-us' AND q.query_id = diffs.query_id
    ORDER BY abs(a - b) DESC LIMIT 10
""").show()
```

Save it as a file and run it with `uv run python yourfile.py`. For Jev against Voyage, Jev did better on 198 queries, Voyage on 202, and 100 tied. The average gap of 0.011 is a few big swings in each direction, like "rose gold series 3 apple watch", where Jev scored 0.728 and Voyage 0.

**Export.** One table, or the whole warehouse as Parquet:

```sql
COPY (SELECT * FROM judgments WHERE source <> 'human') TO 'judge_grades.parquet' (FORMAT parquet);
EXPORT DATABASE 'warehouse_export' (FORMAT parquet);
```

`EXPORT DATABASE` writes one Parquet file per table plus `schema.sql` and `load.sql`, and `IMPORT DATABASE 'warehouse_export'` rebuilds it. That's the easy way to share results or archive a run.

### Back up and reset

To back up, copy the file while nothing is writing to it, or use `EXPORT DATABASE`.

To start over, delete `warehouse/relevancy.duckdb` and rerun step 2. Everything else rebuilds from the scripts, except the judge grades: those cost money and time, so export `judgments` first if you might want them back.

## What it costs

List prices, from turbopuffer's [pricing page](https://turbopuffer.com/pricing) and each provider's, for everything in part 1 at 500 queries:

| Item | What I used | Cost |
| --- | --- | --- |
| Embedding the corpus (voyage-4-lite, $0.02 / 1M tokens) | about 80M tokens (323M characters at 4.0 a token) | about $1.60 |
| Writes ($2 / GB, less batch discounts and 50% off non-filterable attributes) | about 1.9 GB logical, my estimate from the load | under $4 |
| Queries ($1 / PB queried, at least 1.28 GB per query, plus bytes returned) | 1.71 GB per query, 3.42 GB for hybrid | under a cent for all of it |
| Storage ($0.33 / GB) | the 482k-product namespace | well under $1 a month |
| Hosted reranking ($0.05 / 1M tokens for voyage/rerank-3) | 500 queries x 50 candidates | roughly $0.30 per reranker, my estimate |
| Jev ($0.042 / 1M input tokens) | reranking plus judging | well under $1 |
| OpenRouter (three bake-offs, Gemma reranking and judging, smoke runs) | everything in this README | $0.91 in total |

The query line surprised me. Each query bills the logical bytes it searched, which is the namespace's size, with a 1.28 GB floor: 1.71 GB per subquery on the full namespace, so 3.42 GB for hybrid (§14). The hosted rerank calls are queries too and bill the same way; the full runs didn't record those bytes. Even so, at $1 per petabyte it's a fraction of a cent for thousands of queries. Writing and embedding the corpus once is most of the bill. On top of usage, turbopuffer plans have a monthly minimum ($16 on Launch).

## Jev

Jev doesn't return text. You send a `state` (here, the query) and typed questions (here, one 4-level `Score` per result), and it returns a probability for each level. The expected grade works as a ranking score, and the most likely grade works as a label. Elastic [tested it as an ESCI reranker](https://www.elastic.co/search-labs/blog/ecommerce-search-reranking-llm-alternative-jev), and part 1 runs a similar test on turbopuffer. The numbers aren't directly comparable: here every reranker sees turbopuffer's top 50 from the whole corpus, and unlabeled products count as irrelevant.

There are two ways to call it:

1. **TypeSafe's API** (`TYPESAFE_API_KEY`), which is what this repo does. [`src/relevancy/jev.py`](src/relevancy/jev.py) puts up to 50 results for one query into a single request as separate questions. That takes about as long as one result, and grades come back the same as one at a time within Jev's own run-to-run noise ([step 7](#judges-against-humans-step-7)). Input is $0.042 per million tokens.
2. **Inside turbopuffer**, as `typesafe/jev-1.13.0` in `rerank_by`. turbopuffer's reranking page lists it as a preview with US inference, and it wasn't enabled for my org; ask turbopuffer about access. Then `uv run scripts/06_rerank.py --model typesafe/jev-1.13.0` reranks with no code changes, though that path only reranks; it can't hand back grades for judging.

The judge names in this repo:

| Name | What it does |
| --- | --- |
| `jev:jev-1.13.0` | one request per (query, result) pair |
| `jev-batched:jev-1.13.0` | all of a query's results in one request; the default for reranking and step 8 |
| `openai/gpt-4o-mini`, or any OpenRouter model | an LLM read through logprobs |

TypeSafe's API defaults to a newer Jev (1.14.0 when I ran this). I pinned 1.13.0 to match what turbopuffer hosts and what Elastic tested. As a judge, 1.14.0 agrees with the humans more often ([step 8](#judges-ranking-systems-step-8)), so set `JEV_MODEL=jev-1.14.0` unless you need to match turbopuffer's hosted version.

## Why DuckDB + turbopuffer

turbopuffer answers one question well: which 50 documents match this query, with the data on object storage. Here that meant 482k products and a median of 15 to 103 ms of server time per query, depending on the retriever (§1). It doesn't do joins across tables. Everything an eval asks after retrieval is a join or an aggregate. Which of these 50 did a human grade? Which did Jev grade, and how? What did the run cost? Did branded queries get worse after the reload? DuckDB runs in your process with no server, reads Arrow and Parquet, and turns nDCG into one SQL query.

The split also keeps each side simple. turbopuffer never stores labels or run history, and DuckDB never stores vectors. `results` holds ids and ranks; titles and grades come from joins.

Logging turbopuffer's response fields paid for itself right away. On the 21k-product smoke namespace, every first-stage query billed exactly 1.28 GB (turbopuffer's per-query minimum), and hybrid-rrf billed 2.56 GB because it runs two subqueries (§14). I wouldn't have known without a `GROUP BY retriever`.

The scripts here hand-roll code any turbopuffer team would need for offline evals: pacing embedding writes under the token limit, backing off on 429s, reranking a fixed candidate set through an id filter, and logging billing per request. [`docs/duckdb-turbopuffer-connector.md`](docs/duckdb-turbopuffer-connector.md) sketches how to turn that into a reusable connector: a Python package that returns Arrow tables first, then a shared eval schema, and a native DuckDB extension only if those get used.

## Repo layout

```
scripts/        the walkthrough, one numbered step per file
experiments/    follow-up experiments and evidence.py, which recomputes every number here
src/relevancy/  shared code: warehouse, turbopuffer queries, judges, metrics
rubrics/        what "relevant" means, one TOML file per rubric
warehouse/      schema.sql (checked in) and relevancy.duckdb (built by the scripts)
docs/           the project plan and the connector design
data/           downloaded datasets (git-ignored)
```

Every run records its config and git commit (marked `-dirty` if you had uncommitted changes), so any number in a results table traces back to what produced it. All nine 500-query runs here recorded `5212f86-dirty`: I ran them mid-edit, so the commit is close to, but not exactly, the code that produced them. The `config` column has the parameters that matter.
