# Study title

> **Premise:** the belief you're testing, in one or two sentences, stated
> so that a result can prove it wrong.

| Status | Dataset | Cost | Keys |
| --- | --- | --- | --- |
| Planned | ESCI, unless it can't answer the question | requests, tokens, time | turbopuffer, TypeSafe, ... |

**Why it matters:** who would change what they do based on the answer.

## Dataset

Use ESCI unless it can't answer the question, so results sit next to the
headline runs. If you use something else, say why and give its licence.

## Method

The variants, each compared with a baseline on the same queries. Settings
are fixed before running, or picked on half the queries and reported on the
other half.

## Run

```bash
uv run experiments/<group>-<name>/<script>.py <command>
```

## Decide

The comparison that answers the premise, and what counts as a win: usually
a paired 95% interval with a lower bound above zero. Also say what result
would make you stop, and what you'd try next for each outcome.

## Threats to validity

What could make the result misleading: unlabeled results, the judge also
being the method, leakage from labels, picking the best of many settings.

## Needs

Keys, data downloads, code changes, and anything blocked.

## Result

Filled in once it has run: the numbers, the files in `results/`, and the
commit they came from.
