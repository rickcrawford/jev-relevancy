# Learning to rank

> **Premise:** a model trained on every signal beats any single reranker. If it still wins when trained on Jev's grades instead of human labels, Jev can stand in for training data too.

| Status | Dataset | Cost | Keys |
| --- | --- | --- | --- |
| Ready, not yet run | ESCI train split (training); the 500 test queries (scoring) | per query: 6 turbopuffer requests (candidates, four score lookups, Voyage) and 2 Jev requests; about 33k and 11k for 5,500 queries | turbopuffer, TypeSafe |

## Method

1. Load 5,000 train queries, stratified by slice. The raw ESCI file has them; `02_prepare_esci.py` only loads test queries.
2. **Features** for each (query, product) in the hybrid top 100: vector distance; three BM25 scores (title twice plus body, title, body); RRF rank; Voyage's score; Jev's four grade probabilities; brand and color match; title word overlap. None of this is stored at depth 100, so it's all fetched. A product a BM25 field doesn't match gets a missing-value flag, not a zero.
3. Train LightGBM LambdaRank on train only, with hyperparameters from cross-validation on train.
4. Ablations: without Jev, without Voyage, without both. Then a model trained on Jev's grades instead of the human ones, without Jev's features, since the labels are their argmax and it would just relearn Jev.

## What counts

The model against Voyage over the top 100 on the 500 test queries, nDCG@10 and condensed nDCG, lower bound above zero.

## Could mislead

ESCI labels about 20 products a query, so training treats most candidates as irrelevant and can learn what Amazon's search returned. Condensed nDCG on the test side shows how much that matters.

## Run

```bash
uv run relevancy run relevance-ltr          # dry run
uv run relevancy run relevance-ltr --yes
```

`features` saves every 100 queries, so a rerun resumes.

## Needs

- `uv sync --extra studies` for LightGBM.
- The raw ESCI files, from `uv run relevancy pipeline download`.
