# Semantic SQL operators and filter cascades

> **Premise:** LLM-powered SQL filters cost too much at table scale. Jev's yes/no questions run them quickly and cheaply, and its probability can decide the clear cases in a filter cascade without training a model per condition.

| Status | Dataset | Cost | Keys |
| --- | --- | --- | --- |
| Planned | SemBench ⚠️ (text queries); ESCI products | small for Jev; the expensive LLM that labels ground truth is the cost | TypeSafe, OpenRouter |

The buyer here is a data team, not a search team, so it ranks below the search studies.

## Part A: SemBench

Write a Jev runner for SemBench's filter, classify, and rank operators on its text queries: the movie reviews scenario, and the e-commerce scenario's text queries. Image and audio queries are out of scope; leave them out and say so. Score quality, dollar cost, and latency against the published LOTUS, Palimpzest, ThalamusDB, and BigQuery numbers. It counts if quality is within the published systems' range at clearly lower cost.

## Part B: filter cascades on ESCI

Jev's probability decides the clear cases of a condition, and an LLM decides the rest. The goal is a recall target (0.9, then 0.95) at the fewest LLM calls.

1. **Six conditions, fixed now:** replacement part, for children, contains electronics, a set or bundle, is clothing, needs batteries.
2. **Ground truth:** a strong LLM labels 4,000 products per condition, oversampling Jev-yes products so rare conditions have at least 300 positives. A person checks 100 of them. The LLM that labels is a different model from the one the cascade escalates to.
3. **Calibration first:** measure how well Jev's probabilities are calibrated on these labels. Nothing in this repo has measured that yet.
4. Pick cutoffs on 2,000 labels and test on the other 2,000. Repeat with 20 fresh random splits.

**Counts** if the cascade sends fewer products to the LLM than an embedding-similarity first stage at the same recall-target miss rate. ScaleDoc's workloads would allow a published comparison, but they don't appear to be public; ask the authors.

## Needs

SemBench's own environment and setup script (it downloads its data). SemBench has no licence file, and each scenario keeps its source's terms.
