# When Jev disagrees with humans on negation

> **Premise:** when Jev and the human labels disagree on "without", "not", or "no" queries, the humans may be the ones who are wrong. They matched the product type and skipped the negation.

| Status | Dataset | Cost | Keys |
| --- | --- | --- | --- |
| Done | ESCI, 500 sampled queries | one blind rater | none to report; TypeSafe to rerun step 8 |

Jev calls a human-Exact product Irrelevant far more often on negation queries than on other queries. `export` writes every such disagreement, shuffled, with both grades hidden. A rater answers one question per pair: does the product have the thing the query excludes? `report` joins the answers back.

```bash
uv run experiments/judge-negation-audit/negation_audit.py export
# fill in results/negation_answers.csv: pair, has_excluded (yes/no/unclear/not_negation)
uv run experiments/judge-negation-audit/negation_audit.py report
```

## Result

**Mostly the labels.** Of 43 disagreements on real negations, the product had the excluded thing in 32 (74%), so Jev was right. Jev was wrong on 4, and 7 were unclear. Another 5 pairs were "forget me not" (a flower), flagged and left out.

The rater was a Claude model that saw only the query and the product text. One rater is thin; if you rerun this, add a second and report where they disagree. It also means human labels can't score exclusion queries on their own, which matters for [`relevance-constraints`](../relevance-constraints/).
