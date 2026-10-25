# Learning to rank

> **Premise:** a model trained on every signal beats any single reranker. If it still wins when trained on Jev's grades instead of human labels, Jev can stand in for training data too.

| Status | Dataset | Cost | Keys |
| --- | --- | --- | --- |
| Written, not run | ESCI train split (training); the 500 test queries (scoring) | retrieval plus Voyage and Jev for 5,000 train queries and the 500 test queries at depth 100 | turbopuffer, TypeSafe |

## Method

1. Load 5,000 train queries, stratified by slice. The raw ESCI file has them; `02_prepare_esci.py` only loads test queries.
2. **Features** for each (query, product) in the hybrid top 100: vector distance; BM25 on title, body, and title only; RRF rank; Voyage's score; Jev's four grade probabilities; brand and color match; title word overlap. Most of these aren't stored, so collect them with new requests: per-field BM25 and vector scores for every candidate, Voyage and Jev at depth 100, and Jev's full probabilities (the warehouse keeps only the top grade and the expected grade). A feature a list didn't return gets a missing-value flag, not a zero.
3. Train LightGBM LambdaRank on train only, with hyperparameters from cross-validation on train.
4. Ablations: without Jev, without Voyage, without both. Then the same model trained on Jev's grades instead of the human ones.

## What counts

The model against Voyage over the top 100 on the 500 test queries, nDCG@10 and condensed nDCG, lower bound above zero.

## Could mislead

ESCI labels about 20 products a query, so training treats most candidates as irrelevant and can learn what Amazon's search returned. Condensed nDCG on the test side shows how much that matters.

## Run

```bash
uv run relevancy show relevance-ltr                # the steps, and which are paid
uv run relevancy run relevance-ltr            # dry run
uv run relevancy run relevance-ltr --yes
```

Needs the `studies` extra (`uv sync --extra studies`) for LightGBM, and the raw ESCI download for the train split.

## Needs

- `uv sync --extra studies` for LightGBM.
- The raw ESCI files from `uv run relevancy pipeline download`.
