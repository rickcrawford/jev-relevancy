# Semantic SQL operators and filter cascades

> **Premise:** LLM-powered SQL operators (filter, classify, rank) are too expensive at table scale. Jev's typed questions run them fast and cheaply, and its calibrated yes/no probability makes the cheap first stage of a filter cascade without training a model per condition.

| Status | Dataset | Cost | Keys |
| --- | --- | --- | --- |
| Planned | SemBench; ESCI products for cascades | Small; the expensive LLM labels the ground truth | TypeSafe, OpenRouter |

**Why customers care:** data teams want to run conditions like "is this
review a complaint?" or "is this product for kids?" over whole tables.
Commercial LLM APIs make that expensive at database scale; Capital One's
VLDB 2026 workshop paper cites single experiments costing over $10,000. A
fast, cheap, calibrated yes/no model changes that. The buyer is a data team,
not a search team, so this ranks below the search plans.

Jev's question types map onto the operators: `Noul` is a filter, `Choice` a
classifier, `Score` a ranker.

## Part A: SemBench, the public comparison

SemBench (VLDB 2026) has 55 queries in 5 scenarios:
- wildlife images and audio;
- medical;
- movie reviews;
- question answering;
- e-commerce, built on a Kaggle fashion catalogue of about 44k items with
  text, attributes, and images.

Published systems are BigQuery, LOTUS, Palimpzest, and ThalamusDB. Each is
scored on quality (F1 or similar), dollar cost, and latency.

- Write a Jev runner (SemBench's `GenericRunner` interface) covering the
  filter, classify, and rank operators. For queries with joins and maps,
  either skip them and say so, or pass them to an LLM.
- Start with the movie reviews and e-commerce scenarios, which are text and
  images in the form Jev handles best.
- **Counts:** similar quality to LOTUS or Palimpzest at clearly lower cost
  and latency. Submit to the leaderboard at sembench.org.

## Part B: filter cascades

A cascade lets a cheap model decide the clear cases of a yes/no condition
and sends the rest to an expensive LLM, aiming for a guaranteed precision or
recall at the fewest expensive calls. Current methods train a cheap model
for each condition:
- ScaleDoc (SIGMOD 2026);
- the EPFL method (June 2026);
- LOTUS's cascades.

Jev's `Noul` probability needs no training.

**Datasets:**
1. **ESCI products** (482k), for consistency with everything else here and
   because it's a turbopuffer corpus. Use about 10 conditions written
   before running ("is a replacement part", "is for children", "contains
   electronics", ...). The expensive LLM labels a random 2,000 products per
   condition as ground truth.
2. **ScaleDoc's collections** (BigPatent, PubMed, GovReport: 10k documents
   and 20 conditions each, GPT-4o labels) for a direct comparison with
   published cascades. Check the authors have released them before
   planning on it.

**Method:** use Jev's probability as the cheap stage. Pick the confidence
cutoffs on a labeled sample to hit a target recall (0.9 and 0.95), and send
the documents in between to the LLM. Compare with:
- the LLM on everything;
- an embedding-similarity cheap stage;
- the published cascade numbers.

**Metrics:**
- the share of documents sent to the LLM;
- how often the recall target is missed, over 20 repeated samples;
- total cost and latency.

**Counts:** fewer LLM calls than an embedding cheap stage at the same miss
rate, with no per-condition training.

## Sources

[SemBench](https://arxiv.org/abs/2511.01716) ·
[repo](https://github.com/skddj/sembench) ·
[ScaleDoc](https://arxiv.org/abs/2509.12610) ·
[EPFL](https://arxiv.org/abs/2606.08090) ·
[LOTUS](https://arxiv.org/abs/2407.11418) ·
[Capital One](https://arxiv.org/abs/2606.31808)

## Needs

- **SemBench:**
  - Its setup script downloads the data from Google Drive. A Jev runner goes under `runner/`, in the repo's own environment.
  - Leaderboard: contact the authors. ⚠️ There's no licence file at the repo root, and each scenario has its own source terms.
- **Filter cascades:** ❌ ScaleDoc's workloads don't appear to be public. Run them on ESCI products, or ask the authors.
- **Keys:** OpenRouter, for the expensive LLM that labels ground truth. Rerunning SemBench's own baselines would also need Gemini or OpenAI keys; comparing with published numbers doesn't.
