# Does the judge need more of the description?

> **Premise:** Every judge and reranker sees the title, brand, color, and the first 800 characters of the description, while human ESCI raters saw the whole product page. Showing the judge more text might change how well it agrees with them.

| Status | Dataset | Cost |
| --- | --- | --- |
| Done | ESCI, 1,091 human-labeled pairs from 150 queries | Jev requests for about 2,200 pairs |

## Method

Grade the same human-labeled pairs twice with Jev, once with 800 characters of description and once with 4,000, and compare agreement with the humans.

## Run

```bash
uv run experiments/judge-truncation/truncation.py
```

## Result

**No.** At 800 characters: exact agreement 0.600, kappa 0.410. At 4,000: 0.606 and 0.406. Keeping descriptions short costs nothing in agreement and keeps requests small. File: `results/truncation.csv`.
