# Jev as a judge on TREC

> **Premise:** on TREC DL, the collection the IR field trusts, Jev orders search systems the way NIST's assessors do, about as well as UMBRELA's GPT-4o judge and for far less.

| Status | Dataset | Cost | Keys |
| --- | --- | --- | --- |
| Ready, not yet run; tau needs run files by hand | TREC DL 2020 passage (primary); DL19 and LLMJudge (DL23) secondary | about 250 Jev requests per Jev version | TypeSafe |

**Research only.** MS MARCO's terms say "intended for non-commercial research purposes only", so get a legal review before any result goes in customer-facing material. [`judge-wands`](../judge-wands/) makes the same claim on MIT data, which is why it ranks first and this one ranks seventh.

## Method

1. A `rubrics/trec-dl.toml` with TREC's four levels: Irrelevant, Related, Highly relevant, Perfectly relevant.
2. Load DL20's queries, judged passages, and qrels as a dataset, and grade every judged pair with Jev 1.14.0.
3. Score each system on DL20 with NIST's qrels and with Jev's, and compare the two rankings.

## What counts

One primary comparison: Kendall's tau between the system rankings on DL20, with a bootstrap interval over topics, using linear gain and ordering each run by score, as trec_eval does. It counts if the interval's lower bound is at or above 0.80, which puts it in UMBRELA's reported range (0.87 to 0.94 on DL19 to DL23). DL19, LLMJudge, Jev 1.13.0, kappa, and calibration error get reported, but they don't decide anything.

## Could mislead

- MS MARCO is more likely than ESCI to be in Jev's training data. Say so next to any result.
- Ranking systems needs TREC's submitted runs, and the raw run files may need a NIST login. The fallback is this repo's own retrievers and rerankers on DL20: fewer systems, and a weaker tau.

## Run

```bash
uv run relevancy run judge-trec            # dry run
uv run relevancy run judge-trec --yes
```

Needs the `studies` extra (`uv sync --extra studies`) for `ir_datasets`. For tau, put TREC's submitted run files in a folder and run `uv run experiments/judge-trec/trec.py report --runs-dir <folder>` by hand.

## Needs

- `uv sync --extra studies` for `ir_datasets`, whose document store downloads the full MS MARCO v1 passages (about 3 GB).
- `trec.py` covers DL19 and DL20. LLMJudge (DL23) isn't automated: its human test labels were hidden from participants, so it needs its ID mapping files joined to the official DL23 qrels first.
- TREC's submitted run files for tau. The raw files may need a NIST login.
