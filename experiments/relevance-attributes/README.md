# Semantic attributes at index time

> **Premise:** shoppers ask for concepts the catalog doesn't store. Labeling each product once with Jev and storing the label as a turbopuffer attribute fixes those queries at normal filter speed, with no model at query time.

| Status | Dataset | Cost | Keys |
| --- | --- | --- | --- |
| Ready, not yet run; one manual rating step | ESCI | about 10k Jev requests per attribute over 482k products | turbopuffer, TypeSafe |

Categories are the first case of this, in [`relevance-ontology`](../relevance-ontology/). Query-dependent checks like "without X" or a size can't be computed once per product, so they belong to [`relevance-constraints`](../relevance-constraints/).

## Method

1. **One primary attribute**, chosen now: *replacement part* ("is this a replacement part, refill, or accessory for another product, rather than the product itself?"). It's common in ESCI and easy to get wrong. *For kids* is secondary.
2. Find the queries that ask for it with a pattern list fixed in advance ("replacement", "refill", "spare", "compatible with"). If the sample has fewer than 100, draw more from ESCI's test split for this attribute only.
3. Ask one `Noul` per product and store the label and its probability. Do this for every product in the namespace, so the filter isn't limited to a pool.
4. **Check the labels** on a stratified sample: 100 products Jev said yes to and 100 it said no to, rated by a person. That gives precision and a recall estimate even when the attribute is rare. [`database-semantic-operators`](../database-semantic-operators/) labels the same kind of conditions with an LLM; share the labels.
5. Run `hybrid`, `attr-soft` (hybrid plus filtered subqueries, fused by RRF), and `attr-filter` on the matching queries.

## What counts

`attr-soft` against `hybrid` on the primary attribute's queries, nDCG@10. It counts if the lower bound is above zero. With 100 queries, expect to detect only a gain of about 0.04 or more. The secondary attribute is reported with a Bonferroni-corrected interval and doesn't count on its own. If label precision is under 0.9, stop before the retrieval runs; the `precision` step stops the run there. Precision comes with a Wilson interval.

## Run

```bash
uv run relevancy run relevance-attributes            # dry run
uv run relevancy run relevance-attributes --yes
```

After `check-export`, a person fills in the `rater` column of `results/check-replacement_part.csv` without opening the key file; then resume with `--from precision`. For the secondary attribute, run the script's steps directly with `--attribute for_kids`. The `patch` step writes the label to `esci-us-full`, the namespace the main runs use.

## Needs

The same patch check as `relevance-ontology`, and about two hours of a person's time to rate the labels.
