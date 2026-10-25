# Instruction-following and multi-condition retrieval

> **Premise:** agents send long requests with several conditions. Checking each condition with Jev lets a cheap model follow them, where most rerankers ignore the instruction.

| Status | Dataset | Cost | Keys |
| --- | --- | --- | --- |
| Written, not run | STaRK-Amazon ⚠️, then FollowIR ⚠️ | small | TypeSafe |

## STaRK-Amazon (first)

Product queries with text and relational conditions. Take the standard retriever's top 100, rerank with Jev `Score`, then apply [`relevance-constraints`](../relevance-constraints/)'s per-condition checks. It counts if Hit@1 beats the best published reranker at a similar cost, with a bootstrap interval over queries.

## FollowIR (second)

p-MRR measures whether a reranker's ranking changes the right way when the instruction changes. FollowIR gives each query an original and an altered instruction; rerank with each in Jev's `state` and compute p-MRR with MTEB's code. It counts if the interval sits above zero, which would put Jev with the instruction-tuned models (FollowIR-7B, Promptriever) rather than with most rerankers.

## Run

```bash
uv run relevancy show benchmark-instructions-stark                # the steps, and which are paid
uv run relevancy run benchmark-instructions-stark            # dry run
uv run relevancy run benchmark-instructions-stark --yes
```

`stark-retrieve` pulls in `stark-qa` and `bm25s` with `uv run --with`; `stark-qa` is heavy (it depends on PyTorch). FollowIR's steps need the `studies` extra. Run one benchmark with `--step`, e.g. `--from followir-load`.

## Needs

- `pip install stark-qa` in its own environment (it recommends Python 3.8 to 3.11, and this repo pins 3.12). The Amazon knowledge base downloads on first use and can take an hour to process. Its data terms aren't stated.
- FollowIR runs through `mteb`, with the test sets on Hugging Face. Its text comes from licensed news collections.
