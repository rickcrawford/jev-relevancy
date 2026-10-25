# Learning to rank

> **Premise:** A model trained on every signal (BM25, vector distance, both rerankers, attribute matches) beats any single reranker. If it's trained on Jev's grades instead of human labels and still wins, Jev can stand in for training data too.

| Status | Dataset | Cost | Keys |
| --- | --- | --- | --- |
| Planned | ESCI train split (training), the same 500 test queries (scoring) | Retrieval, Voyage, and Jev for 5,000 train queries | turbopuffer, TypeSafe |

**Why customers care:** a trained model combining every signal is often the
largest single gain. But it needs labels or click data and ML upkeep, so
it's for the customers who have those.

## Dataset: ESCI's train split

`02_prepare_esci.py` loads only test queries. The raw file also has the
reduced English train split, which has human labels for many thousands of
queries, none of them in the 500-query test sample. Training on train and
scoring on the same 500 test queries keeps the result comparable with
every stored run, with no leakage.

## Steps

1. Load a train sample: 5,000 train queries, stratified by slice the same
   way the test sample is, as dataset `esci-us-train`. They search the same
   namespace, because the products are shared.
2. **Features** for each (query, product) in the hybrid top 100:
   - BM25 score (title, body, title-only);
   - vector distance;
   - RRF rank;
   - Voyage score;
   - Jev's expected grade and its four probabilities;
   - brand and color match, and query-title word overlap;
   - the category match from `ontology_search.py`, if it has run.

   All of these are already in tpuf-arrow's tables or the warehouse, so the
   features are one DuckDB query.
3. **Model:** LightGBM LambdaRank, optimizing nDCG@10. Hyperparameters are
   chosen by cross-validation on train only. This adds a `lightgbm`
   dependency.
4. **Ablations:** without Jev features, without Voyage features, and
   without both, to show what each reranker adds inside a combined model.

## Metrics and decision

- nDCG@10 on the 500 test queries: the model against Voyage alone, and
  against the best ensemble from [`relevance-reranker-cascade`](../relevance-reranker-cascade/).
- Counts if the lower bound is above zero against Voyage.
- Feature importance, to show where the signal is.

## Cost

Retrieving and reranking 5,000 train queries: about 5k Voyage and 5k Jev
requests. Training takes minutes.

## Caveat

ESCI's labels came from Amazon's pool, so a model trained on them partly
learns that pool. Customers would train on their own clicks or Jev labels.
A variant trained on Jev's labels instead of human ones shows whether Jev
can stand in for training data too. That's a strong customer story if it
holds.

## Needs

- **Data:** the train split is in the raw ESCI parquet already downloaded, but `02_prepare_esci.py` only loads test queries. It needs a train loader or a `--split` flag.
- **Dependencies:** `lightgbm`.
