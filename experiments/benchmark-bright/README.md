# Reasoning retrieval: BRIGHT

> **Premise:** Jev won't lead a reasoning benchmark, but as a first pass it keeps most of a reasoning reranker's gain at a fraction of the cost.

| Status | Dataset | Cost | Keys |
| --- | --- | --- | --- |
| Written, not run | BRIGHT: biology, economics, LeetCode | small for Jev; the reasoning LLM is the cost | TypeSafe, OpenRouter |

Three domains, chosen in advance, covering science, social science, and code.

## Variants

| Variant | |
| --- | --- |
| BM25 top 100 | BRIGHT's baseline |
| Reasoning LLM over the top 100 | the ceiling |
| Jev over the top 100 | |
| Jev keeps the top 20, then the reasoning LLM reranks them | the cascade |

## What counts

The share of the reasoning LLM's gain the cascade keeps, per domain, with its cost per query: (cascade - BM25) / (LLM - BM25). Compare per domain only. The published averages (Rank1-32B 29.4, leaderboard 41 to 46) cover all 12 domains and don't compare with three.

## Run

```bash
uv run relevancy show benchmark-bright                # the steps, and which are paid
uv run relevancy run benchmark-bright            # dry run
uv run relevancy run benchmark-bright --yes
```

Needs the `studies` extra. `llm` is the costly step; pass `--limit` to the script by hand to try a few queries first.

## Needs

`xlangai/BRIGHT` from Hugging Face. The benchmark is CC BY 4.0; its StackExchange text is CC BY-SA.
