# Plan 7: multimodal product search

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
