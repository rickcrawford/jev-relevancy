# Reasoning retrieval: BRIGHT

> **Premise:** Jev won't lead a reasoning benchmark. But as a cheap first pass, it keeps most of a reasoning reranker's gain at a fraction of its cost.

| Status | Dataset | Cost | Keys |
| --- | --- | --- | --- |
| Planned. Lowest priority | BRIGHT (3 domains) | Small for Jev; the reasoning LLM is the cost | TypeSafe, OpenRouter |

**Why customers care:** little, today. BRIGHT's queries need reasoning to
find the right document, such as a coding question whose answer is in a
document about a related technique. Jev isn't built for that and is
unlikely to lead.

The only reason to run it is a cost-against-quality data point, or to test
Jev as a cheap filter ahead of an expensive reasoning reranker.

## Dataset: BRIGHT, a few domains

1,384 queries over 12 domains, scored by nDCG@10. The best systems reach
about 41 to 46 with ensembles. Rank1-32B reranking BM25's top 100 scores
29.4.

Run 3 domains only, chosen in advance: biology, economics (StackExchange),
and LeetCode. Together they cover science, social science, and code.

## Variants

| Variant | |
| --- | --- |
| BM25 top 100 | BRIGHT's baseline |
| Jev reranking the top 100 | Directly comparable with Rank1's setup |
| Jev keeps the top 20, then a reasoning LLM reranks them | Does Jev cut the expensive model's cost without losing its quality? |

## Decision

No pass mark. Report nDCG@10 and cost per query next to Rank1 and the
leaderboard. The cascade result is the interesting one: whether a cheap
first pass keeps most of a reasoning reranker's gain at a fraction of its
cost.

## Source

[BRIGHT](https://arxiv.org/abs/2407.12883)

## Needs

- **Data:** Hugging Face `xlangai/BRIGHT`, CC BY 4.0. Loads with the `datasets` library.
- **Keys:** OpenRouter, for the reasoning reranker.
