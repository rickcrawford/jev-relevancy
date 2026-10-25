---
name: run-study
description: Run, record, or build a search-quality study in this repo's experiments/ folder (jev-relevancy). Use when asked to run a test or study, check whether one is ready, build the ESCI base, record a result, or implement a planned study. Covers the experiments/run.py runner, the study.toml manifests, cost confirmation for paid steps, judging a result against the pass mark fixed in the study's README, and writing the result up.
---

# Running a study

Every folder in `experiments/` is one study: a premise, a pass mark written before anything ran, and (for studies that are ready) a script. `experiments/run.py` reads each folder's `study.toml` and runs its steps. Your job is to run them honestly and report what they showed, including when the answer is no.

## 1. Pick and check

```bash
uv run experiments/run.py list --json            # every study, ranked, with status
uv run experiments/run.py show <study> --json    # premise, keys, steps, pass mark
uv run experiments/run.py check <study> --json   # what's missing
```

Read the study's `README.md` before running anything. The pass mark is under **What counts**, and `study.toml` has a one-line copy in `decide`.

Statuses:
- **done**: there's a result already. Rerun only if asked, and expect it to overwrite files in `results/`.
- **ready**: the script exists but has never run against turbopuffer or Jev.
- **planned**: there's no script yet. See section 4.

## 2. Ask before spending money

Steps marked `paid` call turbopuffer, Jev, or an LLM. `run` without `--yes` prints the plan and runs nothing. Before you pass `--yes`, tell the person which paid steps will run and what the README says they cost, and wait for a yes. Skip that only if they already told you to go ahead for this study.

If `check` says the ESCI base is missing, `uv run experiments/run.py base` lists what to build. The full load alone is about 80M embedding tokens and 40+ minutes, so it needs its own confirmation.

## 3. Run and read

```bash
uv run experiments/run.py run <study>                    # dry run
uv run experiments/run.py run <study> --yes              # every step
uv run experiments/run.py run <study> --yes --from sweep # resume from a step
uv run experiments/run.py run relevance-ontology --yes --scope smoke   # pilot
```

Each step's output goes to the study's `results/` (the file named in `output`, or `results/logs/<step>.log`). Each run writes `results/runs/<time>.json` with exit codes, durations, and the git commit. If a step fails, the later ones don't run. Read the error, fix the cause, and resume with `--from`.

Then compare the output with the pass mark, exactly as written:

- **Don't move the pass mark.** If you think it's wrong after seeing the numbers, say so, but report the result against the mark as written. Any re-analysis is labeled exploratory.
- **Report the interval, not just the mean.** A gain whose lower bound sits at or below zero didn't pass, however good the mean looks.
- **Check the guards the README names**: condensed nDCG next to standard, the stop rule (for example `recall-fusion`'s check overlap under 0.95, or `relevance-ontology`'s oracle under +0.01), and n.
- **A negative result is a result.** Write it up the same way.

## 4. Record it

Only after the person has seen the result, or asked you to record it:

1. Fill in the README's **Result** section: the numbers with intervals, the files in `results/`, the run record, and the commit. Keep the README's voice: first person, plain, specific.
2. In `study.toml`, set `status = "done"` and replace `decide` with a one-line verdict.
3. Update the study's row in `experiments/README.md` (Status column).
4. Commit `results/` (CSVs, text, `runs/*.json`), but not `results/logs/` or anything in `data/`.

## 5. Build a planned study

Follow the README's method, and `experiments/TEMPLATE.md` if you add sections. Match the existing scripts (`recall-fusion/recall_fusion.py` is a good model):

- `ROOT = Path(__file__).resolve().parents[2]`, `OUT = Path(__file__).resolve().parent / "results"`, big files in `data/<name>/`.
- Open the warehouse with `relevancy.warehouse.read_only()`, never for writing.
- Turbopuffer calls go through `relevancy.tpuf` (tpuf-arrow), Jev through `relevancy.jev` or `relevancy.judges.grade_pairs`.
- One subcommand per step, so steps can be resumed.
- Pick settings on the tuning half (a hash of the query id) and report the held-out half.

Before you call it ready, run every free step on synthetic data (a small fake warehouse and fake ranked lists, the way the existing scripts were checked), lint with `uv run ruff check .`, then add its `[[steps]]` to `study.toml` and set `status = "ready"`. Don't change the README's pass mark while building. If the method can't be built as written, stop and say why.

## Things that are easy to get wrong

- Studies need the **full** namespace (`esci-us-full`). The quick start's smoke namespace only works for `relevance-ontology --scope smoke`.
- `relevance-ontology`'s `patch` step writes a new attribute to the namespace the main runs use. Mention it before running.
- Jev is also the judge in the filled metrics. When Jev is the thing being tested, human labels are primary.
- TREC DL, LLMJudge, FollowIR, ExcluIR, SemBench, and STaRK data have restricted or unstated licences (see `experiments/README.md`). Say so next to any result from them.
- Never put API keys in a file other than `.env`, and never commit `.env`.
