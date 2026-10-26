# Semantic SQL operators and filter cascades

> **Premise:** LLM-powered SQL filters cost too much at table scale. Jev's yes/no questions run them quickly and cheaply, and its probability can decide the clear cases in a filter cascade without training a model per condition.

| Status | Dataset | Cost | Keys |
| --- | --- | --- | --- |
| Ready, not yet run (Part B); Part A by hand | SemBench ⚠️ (text queries); ESCI products | small for Jev; the expensive LLM that labels ground truth is the cost | turbopuffer, TypeSafe, OpenRouter |

The buyer here is a data team, not a search team, so it ranks below the search studies.

## Part A: SemBench

Write a Jev runner for SemBench's filter, classify, and rank operators on its text queries: the movie reviews scenario, and the e-commerce scenario's text queries. Image and audio queries are out of scope; leave them out and say so. Score quality, dollar cost, and latency against the published LOTUS, Palimpzest, ThalamusDB, and BigQuery numbers. It counts if quality is within the published systems' range at clearly lower cost.

## Part B: filter cascades on ESCI

Jev's probability decides the clear cases of a condition, and an LLM decides the rest. The goal is a recall target (0.9, then 0.95) at the fewest LLM calls.

1. **Six conditions, fixed now:** replacement part, for children, contains electronics, a set or bundle, is clothing, needs batteries.
2. **Ground truth:** a strong LLM labels 4,000 products per condition, oversampling Jev-yes products so rare conditions have at least 300 positives. A person checks 100 of them. The LLM that labels is a different model from the one the cascade escalates to.
3. **Calibration first:** measure how well Jev's probabilities are calibrated on these labels. Nothing in this repo has measured that yet.
4. Pick cutoffs on 2,000 labels and test on the other 2,000. Repeat with 20 fresh random splits. Every count is weighted back to the full table, since the sample oversamples Jev-yes products.
5. The cascade only aims at recall: Jev's clear yeses pass, its clear noes drop, and the rest go to the LLM. Each first stage gets the smallest safety margin that misses the target in at most 10% of splits, so the two are compared at the same miss rate.

The labeling LLM defaults to `openai/gpt-4o` (`ORACLE_MODEL`), the escalation LLM to `openai/gpt-4o-mini` (`ESCALATE_MODEL`). The `check-export` step writes 100 labels for a person to check, and the report prints their agreement with the labeling LLM.

**Counts** if the cascade sends fewer products to the LLM than an embedding-similarity first stage at the same recall-target miss rate. ScaleDoc's workloads would allow a published comparison, but they don't appear to be public; ask the authors.

## Run

```bash
uv run relevancy run database-semantic-operators            # dry run
uv run relevancy run database-semantic-operators --yes
```

This runs Part B, the cascades. Part A, SemBench, runs inside SemBench's own repo and environment, so it isn't automated here.

## Needs

SemBench's own environment and setup script (it downloads its data). SemBench has no license file, and each scenario keeps its source's terms.
