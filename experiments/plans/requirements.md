# What each plan needs before it can run

Checked in October 2026 against each dataset's repo, README, and licence
file, and against this repo's code. ✅ means checked, ⚠️ means it needs a
decision or a closer look, and ❌ means a blocker as things stand.

## Licences first

Some of these datasets are licensed for non-commercial research only. That
matters if results go into customer-facing material.

| Dataset | Licence | Status |
| --- | --- | --- |
| ESCI | Apache-2.0 | ✅ |
| WANDS | MIT (repo LICENSE) | ✅ |
| BRIGHT | CC BY 4.0 (repo LICENSE) | ✅ |
| SQID | MIT on the Hugging Face card. The images are Amazon's, linked by URL. | ⚠️ fine for evaluation; don't redistribute the images |
| STaRK | Code MIT. The Amazon data's terms aren't stated. | ⚠️ check before commercial use |
| NevIR | Code MIT. Built on CondaQA (Apache-2.0), whose passages are from Wikipedia. The dataset card's licence wasn't checked. | ⚠️ probably fine; read the card |
| **TREC DL 2019/2020, LLMJudge (DL 2023)** | **MS MARCO terms: "intended for non-commercial research purposes only".** Microsoft suggests your own legal review for any use in products or services. | ⚠️ **legal review before using in marketing** |
| FollowIR | The test sets on Hugging Face are drawn from Robust04 (TREC disks, NIST-licensed), Core17 (New York Times corpus, LDC-licensed), and News21 (Washington Post, TREC-licensed) | ⚠️ redistributed licensed text; check terms |
| ExcluIR | No licence file. Data is a Google Drive download. | ⚠️ unclear; ask the authors before publishing |
| SemBench | No licence file at the repo root. Each scenario comes from a different source, such as a Kaggle fashion dataset. | ⚠️ check each scenario's terms |
| ScaleDoc's collections | No public release found | ❌ see plan 6 |

**This affects the ranking:** plan 1's public proof rests on MS MARCO,
which is for non-commercial research only. If legal review says no, WANDS
(MIT) is the clean second proof point for Jev as a judge. It has 233k human
labels, and plan 8 already loads it.

## Keys

| Key | Used by |
| --- | --- |
| `TURBOPUFFER_API_KEY` | 2, 3, 4, 5 (cascade), 7, 8, 9, 10, 11 |
| `TYPESAFE_API_KEY` (Jev) | every plan |
| `OPENROUTER_API_KEY` | 1 (an LLM reference judge), 4 (extracting constraint values), 6 (the expensive LLM that labels ground truth for cascades), 11 (rewrites, expansion), 13 (the reasoning reranker) |
| A Gemini or OpenAI key | 6, only to rerun SemBench's own baselines. Comparing with published numbers needs none. |

## Per plan

### 1. Jev as an eval judge (TREC DL, LLMJudge)

- **Data:**
  - DL19 and DL20 come through `ir_datasets` (MS MARCO v1). Its document
    store downloads the full passage collection, about 3 GB, even though only
    the judged passages are needed. ✅
  - LLMJudge's repo ships the passages as JSONL, plus ID files that map them
    to TREC DL 2023, so the 138M-passage v2 corpus isn't needed. ✅
- ⚠️ **LLMJudge's human test labels were hidden from participants.** Score
  against the official TREC DL 2023 qrels instead, joined through the ID
  mapping files. Confirm the qrels cover the test pairs.
- ⚠️ **System rankings need TREC's submitted runs.** NIST's results pages
  show per-run evaluation summaries; the raw run files may need a NIST
  past-results login. Fallback: rank systems this repo can run itself on
  DL19 and DL20 (BM25, vector, hybrid, four rerankers). That's fewer
  systems, and a weaker τ than UMBRELA's.
- **Code:**
  - `07_judge_bakeoff.py` only grades labeled pairs from queries *not* in
    the sample, 50 per grade. Plan 1 needs every pair graded, so it needs a
    small "grade all labeled pairs" script (or a flag).
  - A `rubrics/trec-dl.toml`.
  - Calibration error (ECE) isn't computed anywhere yet.
- **Dependencies:** `ir-datasets` (on PyPI ✅).

### 2. Recall diagnosis and fusion: `recall_fusion.py`

- Nothing missing. Runs against the existing warehouse and `esci-us-full`.
- ✅ **turbopuffer's RRF accepts per-subquery `weights` and `rank_constant`**
  (`RrfParams` in the turbopuffer SDK), so any weighted setting the sweep
  picks can run on the server as is.

### 3. Semantic attributes at index time: `ontology_search.py` and the plan

- **Data:** the Google Product Taxonomy file. It's blocked from the
  container these scripts were written in; download it from a normal
  network, or put it at `data/ontology/taxonomy-with-ids.en-US.txt`.
- ⚠️ The scripts assume turbopuffer accepts `patch_columns` with a
  `[]string` attribute and a `Contains` filter on it. The SDK's types allow
  both; confirm on the smoke namespace first.
- **Bonus:** WANDS ships its own `category_hierarchy`. Once plan 8 loads
  it, it can check the ontology approach against real categories.

### 4. Constraint and negation checks

- **Data:**
  - NevIR is on Hugging Face (`orionweller/NevIR`). Its own evaluation
    code expects a conda environment with ColBERT and others, which isn't
    needed: pairwise accuracy is a few lines. ✅
  - ExcluIR is a Google Drive folder. ⚠️ Its corpus size isn't documented;
    check it after downloading.
- **Code:** a new script. An LLM call (OpenRouter) extracts constraint
  values from queries.

### 5. Reranker ensemble and cascade

- ✅ Reranked runs store each product's score (`results.score`, written by
  `06_rerank.py`), so the ensembles are SQL over existing rows.
- The cascade needs a small script. No new dependencies.

### 6. Semantic SQL operators and filter cascades

- **SemBench:**
  - The setup script downloads the data from Google Drive and builds the
    databases. ✅
  - A Jev runner goes under `runner/`, following the existing ones.
  - The repo's own environment is separate from this one.
  - Leaderboard: "reach out to discuss how to contribute". ⚠️
- **Filter cascades:** ❌ **ScaleDoc's BigPatent, PubMed, and GovReport
  workloads don't appear to be public.** Run the cascade on ESCI products
  with conditions written in advance (no dependency), and ask ScaleDoc's
  authors for their workloads if a direct comparison matters.
- **Keys:** OpenRouter, for the expensive LLM that labels ground truth.

### 7. Multimodal product search (SQID)

- **Data:**
  - `product_image_urls.csv` from the SQID repo or its Hugging Face mirror.
    ✅
  - It covers the ESCI test split only, which is all the 500 sampled
    queries need. ✅
  - Images were scraped in 2024 for 156,545 of 164,900 products (95%); 442
    are a placeholder image.
- ⚠️ The links point at Amazon's image CDN and may have broken since. Fetch
  every image once into `data/sqid/` and report how many are missing.
- ✅ Jev's SDK accepts `typesafe_sdk.Image` inside question content.
- **Code:** an image fetcher, and a reranker variant that sends
  `[Image, text]`.

### 8. WANDS

- **Data:** `product.csv`, `query.csv`, and `label.csv` from the GitHub repo
  (MIT) ✅. 42,994 products, 480 queries, 233,448 labels.
- **Code:**
  - `rubric.py` requires exactly 4 levels (`LEVELS = 4`), and Wayfair uses
    3.
  - `agreement.weighted_kappa` already takes `k`; `metrics.py` works from
    grade values, so it's fine as is.
  - A loader that writes `02_prepare_custom.py`'s inputs.

### 9. Learning to rank

- **Data:** the ESCI train split is in the raw parquet already downloaded,
  but `02_prepare_esci.py` only loads test queries. It needs a train loader
  or a `--split` flag. ✅
- **Dependencies:** `lightgbm`.
- **Cost:** retrieval plus Voyage and Jev for 5,000 train queries.

### 10. Query routing

- Nothing beyond plan 2's output (the stored top-1,000 lists) and about 10
  Jev requests.

### 11. Query rewrites and document expansion

- **Keys:** OpenRouter.
- ⚠️ Document expansion adds a full-text field. Check whether a patch can
  add a full-text-indexed attribute to an existing namespace. If not, write
  the smoke namespace again with the new field; it's small.

### 12. Instruction-following and STaRK

- **STaRK:** `pip install stark-qa` (MIT); it downloads the Amazon
  knowledge base from Hugging Face, and processing can take up to an hour.
  ⚠️ It recommends Python 3.8 to 3.11, and this repo pins 3.12, so use a
  separate environment.
- **FollowIR:** evaluation runs through `mteb` with the test sets on
  Hugging Face. No NIST disks needed. ✅ (Licence ⚠️ above.)
- **InfoSearch:** not checked yet.

### 13. BRIGHT

- **Data:** Hugging Face `xlangai/BRIGHT`, CC BY 4.0 ✅. Loads with the
  `datasets` library.
- **Keys:** OpenRouter, for the reasoning-reranker cascade.

## Order this suggests

Plans 2 and 5 need nothing new, so run them first. Plan 3's ontology script
needs only the taxonomy file. Plan 1 waits on the licence question, and
plan 8 (WANDS, MIT) is the unrestricted alternative for the same claim.
Plan 6's cascade half runs on ESCI rather than ScaleDoc's workloads.

## Sources

- [MS MARCO terms](https://microsoft.github.io/msmarco/Notice)
- [LLMJudge repo](https://github.com/llm4eval/LLMJudge)
- [UMBRELA repo](https://github.com/castorini/umbrela)
- [TREC DL 2019 passage results](https://trec.nist.gov/pubs/trec28/appendices/deepPassages.html)
- [NevIR repo](https://github.com/orionw/NevIR)
- [CondaQA](https://huggingface.co/datasets/lasha-nlp/CONDAQA)
- [ExcluIR repo](https://github.com/zwh-sdu/excluir)
- [FollowIR repo](https://github.com/orionw/FollowIR)
- [STaRK on PyPI](https://pypi.org/project/stark-qa) · [STaRK repo](https://github.com/snap-stanford/stark)
- [BRIGHT repo](https://github.com/xlang-ai/BRIGHT)
- [SQID repo](https://github.com/Crossing-Minds/shopping-queries-image-dataset) · [SQID on Hugging Face](https://huggingface.co/datasets/crossingminds/shopping-queries-image-dataset)
- [WANDS repo](https://github.com/wayfair/WANDS)
- [SemBench repo](https://github.com/skddj/sembench)
- [ScaleDoc paper](https://arxiv.org/abs/2509.12610)
