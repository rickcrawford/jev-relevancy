# Query routing by query type

> **Premise:** brand-led, spec-heavy, and vague queries want different retrieval settings, so routing each type to its own setting beats the best single setting.

| Status | Dataset | Cost | Keys |
| --- | --- | --- | --- |
| Written, not run | ESCI | about 10 Jev requests and 1,000 Voyage reranks; the rest is offline | TypeSafe, turbopuffer |

Worth doing only after [`recall-fusion`](../recall-fusion/) and [`relevance-constraints`](../relevance-constraints/).

## Method

1. **Is there anything to gain?** `uv run scripts/05_score.py --by-slice` already shows each retriever's nDCG@10 per slice. If the best retriever is the same for every slice, stop.
2. **Types.** One Jev `Choice` per query, with three types fixed in advance: names a brand or model, specifies sizes or quantities, or neither. Fewer types keep each one big enough to tune.
3. **Route.** Each type chooses among five candidate settings from `recall-fusion`'s sweep, picked on the tuning half. With 180 settings and 30 queries per type, the winner would be noise.
4. Rerank each routed top 50 with Voyage, since first-stage gains can vanish after reranking.

## What counts

Routed against `recall-fusion`'s best single setting (not the default), held-out nDCG@10 after Voyage, lower bound above zero.

## Run

```bash
uv run relevancy show relevance-routing                # the steps, and which are paid
uv run relevancy run relevance-routing            # dry run
uv run relevancy run relevance-routing --yes
```

Run [`recall-fusion`](../recall-fusion/) through `sweep` first; this reads its lists and sweep. If `slices` says one retriever wins every slice, stop there.

## Needs

`recall-fusion`'s stored lists in `data/recall/`.
