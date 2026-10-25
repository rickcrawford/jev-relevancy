# AGENTS.md

Instructions for coding agents working in this repo. If you're asked to run a test, check a study, record a result, or build a planned study, follow [`.claude/skills/run-study/SKILL.md`](.claude/skills/run-study/SKILL.md). Claude Code loads it as the `run-study` skill. Other agents should read it first.

## The short version

```bash
uv sync
uv run relevancy list --json            # studies, ranked, with status
uv run relevancy check <study> --json   # keys, base, files: what's missing
uv run relevancy run <study>            # dry run: the steps, paid ones marked
uv run relevancy run <study> --yes      # run them
```

- **Ask before `--yes` on paid steps.** They call turbopuffer, Jev, or an LLM. Building the ESCI base (`uv run relevancy base --yes`) is about 80M embedding tokens.
- **Judge results against the pass mark in the study's README as written.** If you think the mark is wrong, say so separately.
- **Record a result only once the person has seen it:** fill in the README's Result section, set `status = "done"` in `study.toml`, and update the catalog row in `experiments/README.md`.
- **Commit `results/`,** but not `results/logs/`, `data/`, or `.env`.

## Checks before you commit code

```bash
uv run ruff check .
uv run pytest -q          # every study's steps against fake turbopuffer, Jev, and LLM clients
```

The studies can't run here without the warehouse and API keys; the tests in `tests/` run every step offline instead. A new study gets a test there too, built on the `world` fixture in `tests/conftest.py`.
