# Jev as an eval judge: TREC DL and LLMJudge

> **Premise:** Jev's agreement with human relevance labels on ESCI isn't an artifact of shopping data. On the TREC collections the field already trusts, it orders search systems the way NIST's assessors do, about as well as GPT-4o-class judges, for far less money.

| Status | Dataset | Cost | Keys |
| --- | --- | --- | --- |
| Planned | TREC DL 2019/2020 passage, LLMJudge (TREC DL 2023) | About 600 Jev requests | TypeSafe; OpenRouter for an LLM reference judge |

**Why customers care:** the pitch of this repo is "measure your search
without labels". A buyer trusts that once Jev's labels match human labels on
a collection the field already trusts, beyond ESCI.

## Datasets

| Dataset | Why | Size |
| --- | --- | --- |
| ESCI (done) | Already shows the best exact agreement of seven judges, and 26 of 28 system pairs ordered like the humans | 500 queries |
| **TREC DL 2019 and 2020 passage** | The standard collections for LLM-judge papers. UMBRELA, the open version of Bing's GPT-4o judge, reports Kendall's τ of about 0.87 to 0.94 on DL19 to DL23. The corpus is MS MARCO v1 (8.8M passages), which `ir_datasets` serves. | 43 + 54 queries, about 9k + 11k judged pairs |
| **LLMJudge (TREC DL 2023)** | 42 label sets from 8 teams to rank against on the same pairs | 25 test queries, about 7.2k pairs. The corpus is MS MARCO v2, so only the judged passages need fetching. |

Start with DL19 and DL20: the corpus is smaller and UMBRELA's numbers are
directly comparable. Add LLMJudge if they look good.

## Setup

1. A rubric `rubrics/trec-dl.toml` with TREC's four levels: 0 Irrelevant,
   1 Related, 2 Highly relevant, 3 Perfectly relevant, using the wording of
   the NIST assessor guidelines.
2. Load each collection's queries, judged passages, and qrels through
   `02_prepare_custom.py` as datasets `trec-dl19`, `trec-dl20`,
   `trec-dl23`. Only the judged passages are needed, not the full corpus.
3. Grade every judged pair with `07_judge_bakeoff.py`, or a small variant
   of it that takes all pairs instead of a 200-pair sample, using `jev-batched:jev-1.13.0`,
   1.14.0, and one LLM judge as a reference.
4. **System ordering:** download the official TREC run files for each year,
   score every run with the human qrels and with Jev's qrels, and compute
   Kendall's τ between the two system rankings. This reuses step 8's
   agreement code.

## Metrics

- Cohen's κ, weighted and unweighted, between Jev's top grade and NIST's.
- Kendall's τ of the system rankings, from nDCG@10 under each set of labels.
  For each run, also report nDCG computed from Jev's expected grade.
- **Calibration:** expected calibration error of Jev's probabilities against
  NIST's grades, with a reliability plot. Judge papers rarely report it, and
  only a judge that returns probabilities can.
- Per-level confusion. UMBRELA's weak point is "Perfectly relevant" (about
  30% right).

## Counts as a win

τ within the UMBRELA range (0.87 or higher) at a small fraction of GPT-4o's
cost, or κ at or above the best LLMJudge submission. Either is publishable.
Beating both would be a headline.

## Cost and effort

About 30k pairs at 50 per request, roughly 600 Jev requests. The effort is
data loading and fetching the run files, about a day.

## Sources

[UMBRELA](https://arxiv.org/abs/2406.06519) ·
[repo](https://github.com/castorini/umbrela) ·
[LLMJudge](https://arxiv.org/abs/2408.08896) ·
[42 label sets](https://arxiv.org/abs/2502.13908) ·
[benchmark site](https://llm4eval.github.io/LLMJudge-benchmark/) ·
[ir_datasets](https://ir-datasets.com/msmarco-passage.html)

## Needs

- ⚠️ **Licence:** MS MARCO data is "intended for non-commercial research purposes only". Get a legal review before using results in customer-facing material. [`judge-wands`](../judge-wands/) makes the same claim on MIT-licensed data.
- **Data:** DL19 and DL20 come through `ir_datasets`, whose document store downloads the full MS MARCO v1 passages (about 3 GB). LLMJudge's repo ships its passages as JSONL, with ID files mapping them to TREC DL 2023.
- ⚠️ LLMJudge's human test labels were hidden from participants. Score against the official DL23 qrels, joined through the ID mapping files.
- ⚠️ Ranking systems needs TREC's submitted run files, which may need a NIST past-results login. Fallback: this repo's own retrievers and rerankers on DL19 and DL20. That's fewer systems, and a weaker τ.
- **Code:**
  - A "grade every labeled pair" script. `07_judge_bakeoff.py` samples 50 per grade from queries outside the sample.
  - `rubrics/trec-dl.toml`.
  - Calibration error (ECE).
- **Dependencies:** `ir-datasets`.
