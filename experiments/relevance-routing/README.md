# Query routing by query type

> **Premise:** brand-led, spec-heavy, and vague queries want different retrieval settings, so routing each type to its own setting beats the best single setting.

| Status | Dataset | Cost | Keys |
| --- | --- | --- | --- |
| Ready, not yet run | ESCI | about 10 Jev requests and 1,000 Voyage reranks; the rest is offline | TypeSafe, turbopuffer |

It reads [`recall-fusion`](../recall-fusion/)'s lists and sweep, so run that through `sweep` first.

## Method

1. **Is there anything to gain?** `slices` gives each stored retriever's nDCG@10 per query slice. If one retriever wins every slice, it stops the run there.
2. **Types.** One Jev `Choice` per query, with three types fixed in advance: names a brand or model, specifies sizes or quantities, or neither. Fewer types keep each one big enough to tune.
3. **Route.** Each type picks among the sweep's five best settings (by tuning-half nDCG@10) on its tuning-half queries. With 180 settings and about 30 queries a type, picking from all of them would be noise.
4. Voyage reranks each routed top 50, and the baseline's.

## What counts

The baseline is the sweep's best single setting by tuning-half nDCG@10, its top 50 reranked by Voyage. Routed against it, held-out nDCG@10 after Voyage, lower bound above zero.

## Run

```bash
uv run relevancy run relevance-routing          # dry run
uv run relevancy run relevance-routing --yes
```

## Needs

`data/recall/lists-*.parquet` and `experiments/recall-fusion/results/recall_sweep.csv`, from `recall-fusion`'s `fetch` and `sweep`.
