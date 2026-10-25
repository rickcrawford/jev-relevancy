# Multimodal product search

> **Premise:** Product images carry relevance signal the text misses, most of all in thin listings. Jev reading the image along with the text reranks and judges better than text alone.

| Status | Dataset | Cost | Keys |
| --- | --- | --- | --- |
| Planned | SQID (ESCI test split plus images) | About 500 Jev requests per variant, plus fetching the images once | TypeSafe |

**Why customers care:** many catalogues have thin or messy text and good
images. If Jev reading the product image closes that gap, it helps every
catalogue like that.

## Dataset: SQID

SQID adds images to ESCI: image URLs for 156,545 of its products, plus CLIP
embeddings. It's the same queries, products, and labels as the headline
result, so the only thing that changes is the image. The authors report
that adding images to text helps a cross-encoder by about 2.4% nDCG.

Use the 500 sampled queries. Keep only candidates whose image is available;
report how many are dropped. Fetch and cache the images once in
`data/sqid/`.

## Variants (all over the stored hybrid top 50)

| Variant | Jev sees |
| --- | --- |
| `jev-text` | Title, brand, color, and description, as stored (the baseline) |
| `jev-image` | The image and the title only |
| `jev-both` | The image and the full text |
| `voyage` | Stored run, for reference |

The Jev SDK accepts images (`typesafe_sdk.Image`) as question content, so a
product becomes `[Image(url), text]`.

## Also test as a judge

Grade the 200-pair bake-off sample with text only and with text plus image,
and compare agreement with the human labels. ESCI's raters saw the product
page, images included, so the image may explain some disagreements.

## Metrics and decision

- `jev-both` minus `jev-text`, nDCG@10, paired.
- Break it down by how long the description is. The guess is the gain is
  concentrated in products with little text, so check the shortest third
  separately.
- For the judge, exact agreement and κ against the humans.

Counts if `jev-both` gains with a lower bound above zero overall or on the
short-text third. Report latency too, since images make requests bigger.

## Sources

[SQID paper](https://arxiv.org/abs/2405.15190) · repo: Crossing-Minds/shopping-queries-image-dataset on GitHub

## Needs

- **Data:** `product_image_urls.csv` from SQID. It covers the ESCI test split, which is all the 500 sampled queries need. 156,545 of 164,900 products had an image when it was scraped in 2024.
- ⚠️ The links point at Amazon's image CDN and may have broken since. Fetch them once into `data/sqid/` and report how many are missing. Evaluate on them, but don't redistribute them.
- ✅ Jev's SDK accepts `typesafe_sdk.Image` inside question content.
