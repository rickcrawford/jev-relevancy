# Instruction-following and semi-structured retrieval

> **Premise:** Agents send long requests with several conditions rather than keywords. Checking each condition with Jev lets a cheap model follow them, where most rerankers ignore the instruction.

| Status | Dataset | Cost | Keys |
| --- | --- | --- | --- |
| Planned | STaRK-Amazon, then FollowIR (InfoSearch optional) | Small | TypeSafe |

**Why customers care:** agents and RAG systems send long, specific requests
("recent papers, not surveys, written for practitioners") rather than
keywords. It's a growing use, but a niche one today.

## Datasets

| Dataset | What it tests | Fit for Jev |
| --- | --- | --- |
| **FollowIR** | Does a reranker change its ranking when the instruction changes? Built on Robust04, Core17, and News21 with TREC narratives as instructions. Metric is p-MRR (−100 to 100). Most models score zero or below; instruction-tuned ones are positive. | Put the instruction in Jev's `state` and rerank with `Score`. FollowIR plugs into MTEB's reranking interface. |
| **InfoSearch** | Instructions about properties of the document: audience, keyword, format, language, length, source | One `Noul` per property, combined with a relevance `Score` |
| **STaRK-Amazon** | Product queries with text and relational conditions ("a lightweight tent from a brand that also makes sleeping bags"). Metrics are Hit@1 and MRR. | `Noul` per condition, as in [`relevance-constraints`](../relevance-constraints/). It's the product-search member of the set, so it relates most to ESCI. |

Start with **STaRK-Amazon**: it's products, it reuses [`relevance-constraints`](../relevance-constraints/)'s
constraint-checking code, and it has a public leaderboard. Run FollowIR
second, for an instruction-following number on a standard benchmark.

## Method

- **FollowIR:** rerank the official candidate lists with Jev, both with and
  without the instruction in `state`. p-MRR measures the difference between
  the two, so both runs are needed anyway.
- **STaRK-Amazon:** take the top 100 from its standard retriever (or load
  the corpus into turbopuffer as hybrid search), rerank with Jev `Score`,
  then apply [`relevance-constraints`](../relevance-constraints/)'s per-condition `Noul` checks.

## Decision

- FollowIR: a positive p-MRR puts Jev with the instruction-tuned models; at
  or below zero, it's with the rest.
- STaRK: report Hit@1 and MRR against the leaderboard. There's no pass
  mark; the point is where it falls for its cost.

## Sources

[FollowIR](https://arxiv.org/abs/2403.15246) ·
[repo](https://github.com/orionw/FollowIR) ·
[InfoSearch](https://arxiv.org/abs/2410.23841) ·
[STaRK](https://github.com/snap-stanford/stark)

## Needs

- **STaRK:** `pip install stark-qa` (MIT). It downloads the Amazon knowledge base from Hugging Face, and processing can take up to an hour. ⚠️ It recommends Python 3.8 to 3.11, and this repo pins 3.12, so use a separate environment. The Amazon data's terms aren't stated.
- **FollowIR:** runs through `mteb`, with the test sets on Hugging Face, so no NIST disks are needed. ⚠️ Its text comes from licensed news collections (Robust04, the NYT corpus, the Washington Post).
- **InfoSearch:** not checked yet.
