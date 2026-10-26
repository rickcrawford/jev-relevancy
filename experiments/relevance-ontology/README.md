# Ontology-matched search

> **Premise:** if you know which product category a query wants, filtering or boosting products in that category improves hybrid search. Tagging both against a public taxonomy with Jev gets you there without a catalog that has categories.

| Status | Dataset | Cost | Keys |
| --- | --- | --- | --- |
| Ready, not yet run | ESCI, Google Product Taxonomy ⚠️ | about 9,700 Jev requests and 482k small vector searches to tag every product, then 5 × 500 searches | turbopuffer, TypeSafe |

The brand-filter follow-up gained +0.047 with a perfect brand detector on 81 queries. Categories apply to nearly every query, but they're fuzzier than brands.

```bash
uv run relevancy run relevance-ontology --yes          # every step, or:
uv run experiments/relevance-ontology/ontology_search.py taxonomy
uv run experiments/relevance-ontology/ontology_search.py tag-docs      # resumable
uv run experiments/relevance-ontology/ontology_search.py patch
uv run experiments/relevance-ontology/ontology_search.py tag-queries
uv run experiments/relevance-ontology/ontology_search.py run
uv run experiments/relevance-ontology/ontology_search.py judge
uv run experiments/relevance-ontology/ontology_search.py report > experiments/relevance-ontology/results/ontology_search.txt
```

Every command takes `--scope smoke` for a pilot.

## Method

- **Tagging.** For each product, vector search shortlists its 8 nearest taxonomy paths and a Jev `Choice` picks one (or "none of these"). A product gets the deepest category holding 50% of Jev's probability, counting subcategories. A query needs 80%, and if no category reaches it, the query runs as plain hybrid.
- **The patch.** It writes `category_path` (the category and all its parents) to the namespace. This is the one step that changes a namespace the main runs use. It leaves the BM25 and vector fields alone.
- **`run`.** It waits until the namespace is fully indexed after the patch, then runs the variants in turn query by query, so cache warmth hits them alike:
  - `hybrid`, rerun as the baseline;
  - `onto-soft`: hybrid plus the same two subqueries filtered to the category, fused by RRF, so nothing is dropped;
  - `onto-filter`: hybrid filtered to the category;
  - `oracle-soft` and `oracle-filter`, with the category taken from half of the query's Exact products and scored with that half's labels removed (a residual collection), so the oracle isn't scored on the products that gave it the answer;
  - Voyage over the top 50 of `hybrid` and of `onto-soft`.

## What counts

Fixed before the run:

- **Primary:** `onto-soft` against `hybrid`, nDCG@10 on human labels. It counts if the lower bound is above zero, and neither condensed nor Jev-filled nDCG is clearly negative. On 500 queries that needs a gain of about 0.012 to 0.019, more if many queries abstain.
- **Worth building** only if `onto-soft+voyage` also beats `hybrid+voyage`.
- **Stop** if `oracle-soft` gains less than 0.01. Then even a perfect query tagger wouldn't help with these product tags.

## Could mislead

- Jev tags the products and also fills in the unlabeled grades, so the filled metric may favor this. Human labels stay primary.
- A product tagged into the wrong category is invisible to every filtered subquery, oracle included.
- Read the pilot's tags by hand on queries outside the 500, or any fix you make after the pilot is tuned on the test set.

## Needs

- The taxonomy file (Google's terms). If the download is blocked, put it at `data/ontology/taxonomy-with-ids.en-US.txt`.
- Confirm on the smoke namespace that turbopuffer accepts `patch_columns` with a `[]string` attribute and a `Contains` filter on it. The SDK's types allow both.
