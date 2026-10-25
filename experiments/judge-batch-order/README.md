# Does batch order change Jev's grades?

> **Premise:** steps 6 and 8 send all of a query's products to Jev in one request. If one answer leaked into the next, product order would change the grades.

| Status | Dataset | Cost | Keys |
| --- | --- | --- | --- |
| Done | ESCI, pooled pairs for 100 sampled queries | about 6,400 Jev questions | TypeSafe |

It regrades every pair twice, once in the original order (plain run-to-run noise) and once with each query's products reversed, and compares both with the grades step 8 stored.

```bash
uv run experiments/judge-batch-order/batch_order.py
```

## Result

**No.** On 3,177 pairs the grade matched the stored one 97.7% of the time in the same order and 97.3% reversed. Reversing adds almost nothing over normal noise. File: `results/batch_order.csv`.
