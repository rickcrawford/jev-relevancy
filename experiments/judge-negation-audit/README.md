# When Jev disagrees with humans on negation

> **Premise:** When Jev and the human labels disagree on queries with "without", "not", or "no", the humans may be the ones who are wrong: they matched the product type and skipped the negation.

| Status | Dataset | Cost |
| --- | --- | --- |
| Done | ESCI, the 500 sampled queries | One blind rater, a Claude model given only the query and the product text |

## Method

Jev calls a human-Exact product Irrelevant far more often on negation queries than on others. `negation_audit.py export` writes every such disagreement, shuffled and with both grades hidden. A rater answers one question per pair: does the product have the thing the query excludes? `report` joins the answers back.

## Run

```bash
uv run experiments/judge-negation-audit/negation_audit.py export
# a rater fills in results/negation_answers.csv: pair, has_excluded (yes/no/unclear/not_negation)
uv run experiments/judge-negation-audit/negation_audit.py report
```

## Result

**Mostly the labels were wrong.** Of 43 disagreements on real negations, the product had the excluded thing in 32 (74%), so Jev was right. Jev was wrong on 4, and 7 were unclear. The regex also catches "forget me not", a flower name; those pairs are flagged and left out of the rates.

One rater is thin. If you rerun it, add a second and report where they disagree. Files: `results/negation_pairs.csv`, `results/negation_answers.csv`. Follow-up: [`relevance-constraints`](../relevance-constraints/).
