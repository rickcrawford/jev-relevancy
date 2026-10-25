# Semantic attributes at index time

> **Premise:** Shoppers ask for concepts the catalogue doesn't store, such as "for kids", "replacement part", or "without X". Computing them once per product with Jev and storing them as turbopuffer attributes fixes those queries at normal filter speed, with no model at query time.

| Status | Dataset | Cost | Keys |
| --- | --- | --- | --- |
| Planned. The category case is ready in [`relevance-ontology`](../relevance-ontology/) | ESCI | About 10k Jev requests per attribute over the full namespace | turbopuffer, TypeSafe |

**Why customers care:** teams want to filter on concepts their data doesn't
store, such as "for kids", "replacement part", "vegan", or a product
category. With Jev, the label is computed once per document at indexing,
stored as a turbopuffer attribute, and then filtered at normal filter speed.
No model runs at query time.

The first case is product categories: [`relevance-ontology`](../relevance-ontology/). This study generalizes it to yes/no attributes.

## Dataset: ESCI

ESCI has enough queries that ask for an attribute to test the idea without a
new dataset, and the namespace and labels are already there. Find the
queries in SQL with a fixed list of patterns, chosen before looking at any
results:

| Attribute | Query pattern (examples) | `Noul` question per product |
| --- | --- | --- |
| For kids | "for kids", "toddler", "boys", "girls", "baby" | Is this product made for children? |
| Replacement part | "replacement", "refill", "spare", "compatible with" | Is this a replacement part or accessory for another product, rather than the product itself? |
| Without X | "without", "free", "no " | Does this product contain X? (X taken from the query) |
| Size or quantity | a number with a unit or "pack" | Does this product match {size}? |

Each attribute needs at least 40 sampled queries. If the 500-query sample
has fewer, draw more from ESCI's test split for that attribute only.

## Steps

1. **Label.** Ask one `Noul` question per product for each attribute, for
   every product in the namespace: 482k products, 50 per request, so about
   9.7k requests per attribute. To start cheaper, label only products that
   appear in some retriever's top 200 for the matching queries. Report that
   it was done this way, since products outside the pool can't be
   filtered in.
2. **Check the labels.** Have a person rate 100 random products per
   attribute and report precision and recall. If precision is under about
   0.9, filtering will do more harm than good.
3. **Store.** Patch each attribute as a boolean, with its probability as a
   float, the same way `ontology_search.py` patches `category_path`.
4. **Retrieve.** For each matching query, run three variants:
   - `hybrid`;
   - `attr-soft`: hybrid plus the same subqueries filtered to the attribute,
     fused with RRF;
   - `attr-filter`: hybrid with the filter only.

## Metrics and decision

nDCG@10 and Recall@50 with human labels, compared with paired intervals,
per attribute. An attribute is worth shipping if `attr-soft` gains with a
lower bound above zero. A gain on one attribute is enough to make the case
for the pattern.

## Cost

About 10k Jev requests per attribute over the full namespace, plus a patch
for each. Searches are trivial. The human check is about an hour per
attribute.

## Open question

Asking for the attribute's probability lets the filter threshold change at
query time, for example `p_kids > 0.7`. If the binary version works, test
whether thresholds help.

## Needs

- Same as `relevance-ontology`: confirm turbopuffer's `patch_columns` and a filter on the patched attribute on the smoke namespace first.
- About an hour of a person's time per attribute, to check 100 labels.
