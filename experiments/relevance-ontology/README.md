# Ontology-matched search

> **Premise:** Knowing which product category a query is after, and filtering or boosting products tagged with that category, improves hybrid search. Tagging both against a public taxonomy with Jev makes this possible without a catalogue that has categories.

| Status | Dataset | Cost | Keys |
| --- | --- | --- | --- |
| Ready: script written, not run | ESCI, Google Product Taxonomy | About 9,700 Jev requests and 482k small vector searches to tag every product | turbopuffer, TypeSafe |

Nothing below is a result yet.

## The question

Does matching a query to a product category improve hybrid search, if
products and queries are both tagged against the same taxonomy? And does
any gain survive a reranker, which already knows that "cast iron pan" isn't
asking for a pan scraper?

The brand-filter follow-up in [`retrieval-followups`](../retrieval-followups/) suggests it might:
with a perfect brand detector, filtering branded queries to their brand
gained +0.047 nDCG@10 [+0.001, +0.096] on 81 queries. Categories apply to
nearly every query rather than one in six, but they're fuzzier than brands.

## How it works

1. **Ontology:** the Google Product Taxonomy, about 5,600 categories up to 7
   levels deep, loaded into its own turbopuffer namespace so its paths are
   embedded.
2. **Tag products:** vector search shortlists each product's 8 nearest
   category paths, and a Jev `Choice` question picks one of them, or "none of
   these". A product gets the deepest category that holds 50% of Jev's
   probability, counting its subcategories. It's written to the namespace as
   `category_path`: that category and all its parents. A patch, so nothing is
   embedded again.
3. **Tag queries:** the same, except a query needs 80% to get a category. A
   query below that gets no category and runs as plain hybrid.
4. **Retrieve:** `onto-soft` adds a second copy of hybrid's two subqueries,
   filtered with `("category_path", "Contains", node)`, and fuses all four
   with RRF. In-category products get more votes, and no product is dropped.
   `onto-filter` runs only the filtered pair.

## Pre-registered settings

Fixed before any results: 8 candidates in the shortlist, 50% probability
for a product's category and 80% for a query's, Jev 1.13.0, RRF with
turbopuffer's defaults, the same 500 sampled queries and namespace as the
main runs. If any of them is tuned after looking at the 500, report the
tuned number as tuned, and confirm it on a fresh draw of ESCI test queries
before claiming it.

## Comparisons

All paired, on the same queries, with a 95% bootstrap interval, using the
human ESCI labels.

| | Comparison | Why |
| --- | --- | --- |
| **Primary** | `onto-soft` minus `hybrid`, nDCG@10 | Does the realistic version help first-stage search? |
| Secondary | `onto-soft+voyage` minus `hybrid+voyage`, nDCG@10 | Does it still help once a reranker runs? This is the number that matters for a production stack. |
| Ceiling | `oracle-soft` and `oracle-filter` minus `hybrid` | The query's category taken from its human-Exact products: a perfect query tagger, with the real product tags |
| Hard filter | `onto-filter` minus `hybrid` | What a filter costs when the tagger is wrong |
| Recall | Recall@50 for each of the above | A reranker can only promote what's in the top 50 |

`hybrid` is rerun in the same session so latencies are comparable, and it
should match the stored `hybrid-rrf` run's nDCG@10 (0.367) to within
turbopuffer's run-to-run variation. If it doesn't, find out why before
reading anything else.

## Decision rules

- **It works** if the primary interval's lower bound is above zero, and
  neither the condensed nDCG@10 nor the Jev-filled nDCG@10 (below) has an
  interval entirely below zero.
- **Worth building** only if the secondary comparison, after Voyage, also
  has a lower bound above zero.
- **Stop early** if `oracle-soft` gains less than +0.01, or its interval
  includes zero. Then even a perfect query tagger can't help on ESCI with
  these product tags, and improving the tagger won't change that.
- **The tagger is the problem** if the oracle gains and `onto-soft` doesn't.
  The report's "on the oracle's path" rate shows how often the predicted
  category is the oracle's or one of its parents, and the per-confidence
  rows show whether the losses come from the less certain queries.

## How big an effect it can detect

Paired intervals on these 500 queries have been ±0.004 to ±0.013 wide
(one-request reranking, the BM25 field variants), so about +0.01 nDCG@10 is
the smallest gain this can separate from zero. That assumes most queries
get a category. If the tagger abstains on many, the comparison only moves
on the queries that do, and the report's per-confidence rows give their
own n.

## Threats to validity

- **Unlabeled products.** ESCI labels about 20 products per query, and any
  unlabeled product counts as gain 0. A retriever that finds new products is
  scored down even when they're good. Two checks: condensed nDCG@10 (drop
  unlabeled results) and Jev-filled nDCG@10 (`judge` has Jev grade the
  unlabeled top-10 products; human labels win where both exist). If the
  human-only and filled numbers disagree in sign, say so and don't call it.
- **Jev tags and judges.** The filled metric uses the same model that did the
  tagging, so it may favor the ontology. That's why it's only a check and the
  primary metric uses human labels only.
- **The oracle needs product tags.** It reads the categories Jev gave the
  Exact products, so it's a ceiling on query tagging, not on product tagging.
  A product tagged into the wrong category is invisible to every filtered
  subquery, oracle included.
- **The smoke namespace is easy.** Every judged product for the sampled
  queries is in it, diluted by only 20,000 random ones, so its nDCG runs high.
  Use it to check the plumbing and look at tags by hand, not to decide.
- **Taxonomy fit.** Google's taxonomy is built for retail ads. Books, media,
  and replacement parts are thin in it, so expect more "none of these" and
  more abstained queries there. The per-confidence rows will show it.
- **Latency.** `onto-soft` runs four subqueries instead of two. Report the
  server and wall p50 and p99 next to any gain.

## Running it

Pilot on the smoke namespace: check that tagging works, and read 50 product
tags and 50 query tags by hand.

```bash
uv run experiments/relevance-ontology/ontology_search.py taxonomy
uv run experiments/relevance-ontology/ontology_search.py tag-docs --scope smoke
uv run experiments/relevance-ontology/ontology_search.py patch --scope smoke
uv run experiments/relevance-ontology/ontology_search.py tag-queries --scope smoke
uv run experiments/relevance-ontology/ontology_search.py run --scope smoke
uv run experiments/relevance-ontology/ontology_search.py report --scope smoke
```

Then the real test on all 482k products:

```bash
uv run experiments/relevance-ontology/ontology_search.py tag-docs        # resumable; rerun if it stops
uv run experiments/relevance-ontology/ontology_search.py patch
uv run experiments/relevance-ontology/ontology_search.py tag-queries
uv run experiments/relevance-ontology/ontology_search.py run
uv run experiments/relevance-ontology/ontology_search.py judge
uv run experiments/relevance-ontology/ontology_search.py report > experiments/relevance-ontology/results/ontology_search.txt
```

## What it costs

- **Jev:** one `Choice` question per product (482k, 50 per request, so about
  9,700 requests), one per query (500), and grades for the unlabeled top-10
  products across the seven variants (at most 35,000, probably far fewer,
  since the variants overlap).
- **turbopuffer:** 5,600 taxonomy rows written; 482k shortlist queries
  against that small namespace, which embed about 10M tokens of product
  titles; a patch of the tagged products (no embedding); and 7 × 500
  searches plus 1,000 Voyage reranks.
- **Time:** the shortlist is the slow part, about 482k requests at 8 at a
  time.

Product tags go to `data/ontology/` (git-ignored, about 50 MB for the
full set). Query tags, runs, and grades go to this folder's `results/`.

## Needs

- **Data:** the Google Product Taxonomy file. If the download is blocked, put it at `data/ontology/taxonomy-with-ids.en-US.txt`.
- ⚠️ It assumes turbopuffer accepts `patch_columns` with a `[]string` attribute and a `Contains` filter on it. The SDK's types allow both; confirm on the smoke namespace first.
- ⚠️ It assumes Jev's `Choice` answers return probabilities keyed by choice name. Check the first batch.
