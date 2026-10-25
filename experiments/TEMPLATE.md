# Study title

> **Premise:** what you believe, stated so a result could prove it wrong.

| Status | Dataset | Cost | Keys |
| --- | --- | --- | --- |
| Planned | ESCI, unless it can't answer the question | requests, tokens, time | turbopuffer, TypeSafe |

Who would act on the answer, and what they'd change.

## Method

The variants, each against a baseline on the same queries. If you pick a setting, pick it on half the queries and report the other half. Use ESCI unless it can't answer the question; if you use something else, say why and give its licence.

## Run

```bash
uv run experiments/<group>-<name>/<script>.py <command> > experiments/<group>-<name>/results/<name>.txt
```

## What counts

The one comparison that answers the premise, and the result that would count (usually a paired 95% interval with its lower bound above zero). Say how big an effect the sample can detect, and what result would make you stop.

## Could mislead

Unlabeled results, Jev being both the method and the judge, labels leaking into the method, picking the best of many settings.

## study.toml

The runner reads this file next to the README:

```toml
title = "Study title"
premise = "One sentence a result could prove wrong."
status = "planned"            # planned, ready, or done
rank = 5
keys = ["TURBOPUFFER_API_KEY", "TYPESAFE_API_KEY"]
base = true                   # needs the full ESCI base
decide = "The pass mark from What counts, in one line."

[[steps]]
name = "fetch"
run = "uv run experiments/<group>-<name>/<script>.py fetch"
paid = true                   # calls turbopuffer, Jev, or an LLM
output = "results/fetch.txt"  # optional; otherwise results/logs/<name>.log
```

## Needs

Keys, downloads, code changes, and anything blocked.

## Result

Once it has run: the numbers, the files in `results/`, and the commit.
