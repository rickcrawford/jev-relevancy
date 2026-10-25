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

## Needs

Keys, downloads, code changes, and anything blocked.

## Result

Once it has run: the numbers, the files in `results/`, and the commit.
