# Does the judge need more of the description?

> **Premise:** judges and rerankers see the title, brand, color, and the first 800 characters of the description, while ESCI's raters saw the whole product page. More text might change how well Jev agrees with them.

| Status | Dataset | Cost | Keys |
| --- | --- | --- | --- |
| Done | ESCI, 1,091 labeled pairs from 136 queries | about 2,200 Jev questions | TypeSafe |

It grades the same pairs twice, at 800 and at 4,000 characters of description, and compares agreement with the humans. (It drew 150 queries; 14 had no pairs in step 8's pool.)

```bash
uv run experiments/judge-truncation/truncation.py
```

## Result

**No.** At 800 characters, exact agreement 0.600 and kappa 0.410. At 4,000, 0.606 and 0.406. Short descriptions cost nothing in agreement and keep requests small. File: `results/truncation.csv`.
