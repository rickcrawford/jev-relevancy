# A second catalogue: WANDS

> **Premise:** Jev's agreement with human labels isn't an ESCI quirk. On Wayfair's catalogue, with its own categories and labeling process, it orders systems the way the humans do about as well as it did on Amazon's.

| Status | Dataset | Cost | Keys |
| --- | --- | --- | --- |
| Written, not run | WANDS (MIT): 480 queries, 42,994 products, 233,448 labels | a few million embedding tokens; a few thousand Jev requests | turbopuffer, TypeSafe |

This is the first question a prospect who isn't Amazon asks, and WANDS answers it on data anyone can use commercially. It also labels many products per query, so far fewer results go unlabeled than on ESCI.

## Method

1. Load WANDS through `02_prepare_custom.py` with `--sample 380`. Step 7 only draws from queries outside the sample, so this holds 100 queries back for the judge bake-off.
2. Run steps 3, 4, 6 (Voyage, Cohere, zerank-2, Jev), 7, 8, and 5, as on ESCI.

## What counts

Fixed before the run, against step 8 on ESCI (tau 0.857, interval 0.714 to 1.000; pooled kappa 0.436):

- **Holds** if Jev's tau over the runs is 0.71 or more and its pooled kappa is 0.35 or more.
- **Fails** if tau is under 0.60 or kappa under 0.30.
- Anything between is a partial result, reported as such.

Tau comes first, because it doesn't depend on how many grade levels there are. Kappa across a 3-level and a 4-level scale is only roughly comparable.

## Run

```bash
uv run relevancy show judge-wands                # the steps, and which are paid
uv run relevancy run judge-wands            # dry run
uv run relevancy run judge-wands --yes
```

This runs the whole pipeline on WANDS as dataset `wands`, and needs no ESCI base.

## Needs

- The WANDS files download in the first step (MIT). `rubrics/wands.toml` maps Exact, Partial, and Irrelevant to grades 3, 2, and 0.
- `product.csv` also has a `category_hierarchy`, which can check [`relevance-ontology`](../relevance-ontology/) against real categories.
