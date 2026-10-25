"""On negation queries, is Jev wrong, or are the human labels?

Jev grades a human-Exact product Irrelevant far more often on queries with
"without", "not", "no", or "non" than on other queries. Reading the cases
suggests the humans missed the negation. This script exports every such
disagreement, shuffled and with both grades hidden, for a blind third rater
to answer one question per pair: does the product have what the query
excludes? Then it joins the answers back and reports who was right.

The regex also catches "forget me not", a flower name. Those pairs stay in
the export on purpose so the rater can flag them (not_negation); the
report and experiments/evidence.py leave them out of the rates.

  uv run experiments/judge-negation-audit/negation_audit.py export     # writes results/negation_pairs.csv
  # a rater fills in results/negation_answers.csv: pair, has_excluded (yes/no/unclear/not_negation)
  uv run experiments/judge-negation-audit/negation_audit.py report
"""

import csv
import sys
from pathlib import Path

from relevancy.warehouse import read_only

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "results"
JUDGE = "jev-batched:jev-1.13.0"
NEGATION = r"\b(without|not|no|non)\b"


def disagreements(con):
    return con.execute(f"""
        SELECT row_number() OVER (ORDER BY hash(h.query_id || h.doc_id)) AS pair,
               q.text, d.title, left(d.body, 800) AS body, h.grade AS human, j.grade AS jev
        FROM judgments h
        JOIN judgments j USING (dataset, query_id, doc_id)
        JOIN queries q USING (dataset, query_id)
        JOIN docs d USING (dataset, doc_id)
        WHERE h.dataset = 'esci-us' AND h.source = 'human' AND j.source = '{JUDGE}'
          AND regexp_matches(lower(q.text), '{NEGATION}')
          AND h.grade = 3 AND j.grade = 0
        ORDER BY pair
    """).fetchall()


def main() -> None:
    con = read_only()
    rows = disagreements(con)
    OUT.mkdir(parents=True, exist_ok=True)
    if sys.argv[1:] == ["export"]:
        with open(OUT / "negation_pairs.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["pair", "query", "title", "description"])
            w.writerows([(r[0], r[1], r[2], r[3]) for r in rows])  # no grades: the rater works blind
        print(f"{len(rows)} pairs written to {OUT / 'negation_pairs.csv'}")
        return

    answers = {}
    with open(OUT / "negation_answers.csv") as f:
        for a in csv.DictReader(f):
            answers[int(a["pair"])] = a["has_excluded"].strip().lower()
    counts = {}
    for r in rows:
        counts[answers.get(r[0], "missing")] = counts.get(answers.get(r[0], "missing"), 0) + 1
    total = len(rows) - counts.get("not_negation", 0)
    print(f"{len(rows)} pairs where the humans said Exact and Jev said Irrelevant, on negation queries; "
          f"{total} are real negations (percentages are of those)\n")
    labels = {
        "yes": "product has what the query excludes (Jev right, human label wrong)",
        "no": "product doesn't have it (Jev wrong)",
        "unclear": "the product text doesn't say",
        "not_negation": "the query isn't really a negation (\"forget me not\")",
        "missing": "no answer",
    }
    for key, label in labels.items():
        if counts.get(key):
            share = "" if key == "not_negation" else f"{counts[key] / total:.1%}"
            print(f"  {counts[key]:>3}  {share:>5}  {label}")


if __name__ == "__main__":
    if any(a in ("-h", "--help") for a in sys.argv[1:]):
        raise SystemExit(print(__doc__))
    main()
