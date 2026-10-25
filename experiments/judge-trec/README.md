# Jev as a judge on TREC

> **Premise:** on TREC DL, the collection the IR field trusts, Jev orders search systems the way NIST's assessors do, about as well as UMBRELA's GPT-4o judge and for far less.

| Status | Dataset | Cost | Keys |
| --- | --- | --- | --- |
| Planned | TREC DL 2020 passage (primary); DL19 and LLMJudge (DL23) secondary | about 600 Jev requests per Jev version | TypeSafe; OpenRouter for an LLM reference |

**Research only.** MS MARCO's terms say "intended for non-commercial research purposes only", so get a legal review before any result goes in customer-facing material. [`judge-wands`](../judge-wands/) makes the same claim on MIT data, which is why it ranks first and this one ranks seventh.

## Method

1. A `rubrics/trec-dl.toml` with TREC's four levels: Irrelevant, Related, Highly relevant, Perfectly relevant.
2. Load DL20's queries, judged passages, and qrels as a dataset, and grade every judged pair with Jev 1.14.0.
3. Score each system on DL20 with NIST's qrels and with Jev's, and compare the two rankings.

## What counts

One primary comparison: Kendall's tau between the system rankings on DL20, with a bootstrap interval over topics. It counts if the interval's lower bound is at or above 0.80, which puts it in UMBRELA's reported range (0.87 to 0.94 on DL19 to DL23). DL19, LLMJudge, Jev 1.13.0, kappa, and calibration error get reported, but they don't decide anything.

## Could mislead

- MS MARCO is more likely than ESCI to be in Jev's training data. Say so next to any result.
- Ranking systems needs TREC's submitted runs, and the raw run files may need a NIST login. The fallback is this repo's own retrievers and rerankers on DL20: fewer systems, and a weaker tau.

## Needs

- `ir_datasets`, whose document store downloads the full MS MARCO v1 passages (about 3 GB).
- A script that grades every labeled pair. `07_judge_bakeoff.py` only samples 50 per grade from queries outside the sample.
- For LLMJudge: its human test labels were hidden from participants, so join its ID mapping files to the official DL23 qrels.
