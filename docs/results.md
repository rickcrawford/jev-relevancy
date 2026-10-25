# Results

What the walkthrough found on ESCI, and how far to trust it. The section numbers (§) point to the matching part of [`experiments/records/evidence.txt`](../experiments/records/evidence.txt), which [`experiments/evidence.py`](../experiments/evidence.py) recomputes from the warehouse.

## What the 500-query run showed

All 482,105 products in the namespace, 500 stratified queries, `aws-us-east-1`, scripts run from a laptop, so wall-clock times include the trip over the public internet. Every reranker reorders the same hybrid top 50. All nine runs recorded commit [`35727d9-dirty`](https://github.com/rickcrawford/jev-relevancy/commit/35727d977b6b2afb5a003325b4af2343a5032062): I ran them mid-edit, so the commit is close to, but not exactly, the code that produced them. Each run's `config` column has the parameters that matter.

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

200 held-out pairs, 50 per grade. Brackets are 95% bootstrap intervals over pairs (§6). Latency and cost are from step 7's printout on my run; the warehouse doesn't store them, so the latency lines are saved in [`experiments/records/judge_bakeoff_latency.txt`](../experiments/records/judge_bakeoff_latency.txt).

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

**Jev 1.14.0 is the better judge.** Graded against the same pool, it agrees with the humans on 0.031 more of the labeled pairs than 1.13.0 (interval +0.019 to +0.045), and its kappa is 0.471 against 0.436 (§15). Its tau is the same (from a step 8 run on a copy of the warehouse, logged in [`experiments/judge-jev-1.14/README.md`](../experiments/judge-jev-1.14/README.md)), and as a reranker it's level with 1.13.0 (-0.001, interval -0.007 to +0.004). The two versions give the same grade on 87.8% of pairs. If you're starting fresh, use 1.14.0 as the judge; I kept 1.13.0 for the main runs to match what turbopuffer hosts.

Single-grade agreement is higher here than in step 7 (0.60 against 0.52 for Jev), and it isn't because the pool is mostly Exact products. Reweighting the pool to an even mix of grades still gives 0.572 (§8). The difference is which products of each grade show up. On human-Exact products, Jev agrees 66% of the time in the pool but 36% in the bake-off. The Exacts a good search system returns are the obvious ones, while the bake-off draws Exacts at random, including the borderline cases.

Gemma ties Jev on standard nDCG and has a higher tau on condensed nDCG (0.929 against 0.786), but the intervals overlap, so I can't call Gemma better there. It's also four to five times slower. For an eval loop you run on every change, I'd use Jev and check it against a few hundred human labels on your own data now and then (the protocol is [below](#how-much-to-trust-these-numbers)).

## How much to trust these numbers

`uv run experiments/evidence.py` recomputes every figure above from the warehouse and writes [`experiments/records/evidence.txt`](../experiments/records/evidence.txt). It's read-only and takes a few seconds. Here's what it says about the limits:

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

[`experiments/`](../experiments/) has the follow-ups: the negation audit, batch order, description length, Jev 1.14.0, and two that look at turbopuffer itself. [`experiments/README.md`](../experiments/README.md) lists them, and each study's README has its command and result.

### What turbopuffer did during the runs

[`experiments/turbopuffer-perf/turbopuffer_perf.py`](../experiments/turbopuffer-perf/turbopuffer_perf.py) reran the three first-stage retrievers twice and saved every field turbopuffer returns ([`turbopuffer_perf.txt`](../experiments/turbopuffer-perf/results/turbopuffer_perf.txt)). [`experiments/retrieval-followups/retrieval_followups.py`](../experiments/retrieval-followups/retrieval_followups.py) tried BM25 field variants, hybrid and reranking in one request, a brand filter, and Elastic's judged-only setup ([`retrieval_followups.txt`](../experiments/retrieval-followups/results/retrieval_followups.txt)). Every query ran one at a time from my laptop against `aws-us-east-1`, so none of this says anything about throughput under concurrency.

**Most of the wall time was network.** The laptop's round trip was a median of about 54 ms (§1), and turbopuffer's server time was 76 ms for ANN, 102 ms for BM25, and 103 ms for hybrid at p50. Of ANN's 76 ms, 66 ms was embedding the query text. Hybrid's server time is about BM25's alone, not the sum of the two.

**The BM25 tail comes from the body field and long queries.** Body-only BM25 went from a 29 ms median on one-word queries to 260 ms on queries of six words or more, and title-only BM25 stayed between 13 and 29 ms. Title-only BM25 also scored higher on nDCG@10 (0.332 against 0.311, +0.020 [+0.008, +0.033]), though most of that is the label artifact described in [the results](#what-the-500-query-run-showed).

**The cache was warm and the results were stable.** Every query in both passes reported `cache_temperature: hot`, so these are warm numbers; I didn't test a cold namespace. The second pass returned the same top 10 for 100% of BM25 queries and 99.4% of ANN and hybrid queries.

**ANN gave up almost no recall.** turbopuffer's recall endpoint measured 0.994 at 50 and 0.995 at 10 against exhaustive search, over 100 query vectors it samples from the namespace. It doesn't take text queries, so this measures the index rather than these exact 500 queries.

**2% of the namespace was still unindexed.** Every response reported 9,785 documents (of 482,105) searched exhaustively because they weren't indexed yet. I didn't measure how long indexing takes to catch up.

**Hybrid plus a hosted reranker works as one request, and it's better.** turbopuffer's [reranking docs](https://turbopuffer.com/docs/reranking) allow `rerank_by` on a `multi_query`, which reranks the union of the subqueries' results. With Voyage, median wall time fell from 362 ms (two requests) to 308 ms, server time stayed level (248 against 245 ms), and nDCG@10 rose 0.006 [+0.003, +0.010], likely because the reranker sees the union instead of the fused top 50. Asking each subquery for 100 results raised it 0.016 [+0.010, +0.022] at 323 ms.

**Filters cost almost nothing in time.** On the 81 sampled queries that name a brand, filtering hybrid search to that brand moved median server time from 79 to 77 ms and raised nDCG@10 from 0.477 to 0.524 (+0.047 [+0.001, +0.096]). I took the brand from the products humans labeled Exact, so that gain is an upper bound on what a real brand detector would get.

**On judged products only, Jev beats Voyage.** [Elastic](https://www.elastic.co/search-labs/blog/ecommerce-search-reranking-llm-alternative-jev) reranked only each query's judged products. Doing the same on these 500 queries, Jev scored 0.868 nDCG@10 and Voyage 0.848 (+0.020 [+0.010, +0.029]); with the KDD Cup gains (Exact 1, Substitute 0.1, Complement 0.01) it was 0.817 against 0.801 (+0.016 [+0.003, +0.029]). Elastic reported 0.9565 for Jev and 0.8595 for a random order. Ordering by product id scored 0.679 here, so the two setups don't compute nDCG the same way and the absolute numbers don't compare. The direction does. On turbopuffer's real top 50 they tie.

**Still untested:** throughput and tail latency from inside the region under concurrent load, a cold namespace, write throughput with vectors computed ahead of time instead of embedded by turbopuffer, and how long indexing takes to catch up after a big load.

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
| OpenRouter (three bake-offs, Gemma reranking and judging, smoke runs) | everything in these docs | $0.91 in total |

The query line surprised me. Each query bills the logical bytes it searched, which is the namespace's size, with a 1.28 GB floor: 1.71 GB per subquery on the full namespace, so 3.42 GB for hybrid (§14). The hosted rerank calls are queries too and bill the same way. The full runs didn't record those bytes, though step 6 now saves them in `tpuf_requests`. Even so, at $1 per petabyte it's a fraction of a cent for thousands of queries. Writing and embedding the corpus once is most of the bill. On top of usage, turbopuffer plans have a monthly minimum ($16 on Launch).
