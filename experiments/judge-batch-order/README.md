# Does batch order change Jev's grades?

> **Premise:** Steps 6 and 8 send all of a query's products to Jev in one request, one question each. If one answer leaked into the next, the order of the products would change the grades.

| Status | Dataset | Cost |
| --- | --- | --- |
| Done | ESCI, pooled pairs for 100 sampled queries | Jev requests for about 3,200 pairs, twice |

## Method

Regrade every pair twice: once in the original order, to measure plain run-to-run noise, and once with each query's products reversed. Compare both with the grades step 8 stored.

## Run

```bash
uv run experiments/judge-batch-order/batch_order.py
```

## Result

**No.** Regrading 3,177 pairs matched the stored grade 97.7% of the time in the same order, and 97.3% reversed, so reversing adds almost nothing beyond normal noise. File: `results/batch_order.csv`.
