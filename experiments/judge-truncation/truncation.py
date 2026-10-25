"""Does showing the judge more of the product description change agreement?

Every judge and reranker here sees the title, brand, color, and the first
800 characters of the description. Human ESCI raters saw the product page.
This grades the same human-labeled pairs twice with Jev, once at 800
characters and once at 4,000, and compares agreement with the humans.

  uv run experiments/judge-truncation/truncation.py
"""

import csv
from pathlib import Path

import duckdb

from relevancy.agreement import weighted_kappa
from relevancy.judges import grade_pairs
from relevancy.rubric import load

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "results"
JUDGE = "jev-batched:jev-1.13.0"
LENGTHS = [800, 4000]


def main() -> None:
    con = duckdb.connect(str(ROOT / "warehouse" / "relevancy.duckdb"), read_only=True)
    # Human-labeled pairs from step 8's pool: what good systems actually return.
    rows = con.execute(f"""
        WITH picked AS (
            SELECT query_id FROM queries WHERE dataset = 'esci-us' AND sampled
            ORDER BY hash(query_id || 'truncation') LIMIT 150
        )
        SELECT h.query_id, h.doc_id, q.text, d.title, d.brand, d.color, d.body, h.grade
        FROM judgments h
        JOIN judgments j ON j.dataset = h.dataset AND j.query_id = h.query_id
                        AND j.doc_id = h.doc_id AND j.source = '{JUDGE}'
        JOIN picked p ON p.query_id = h.query_id
        JOIN queries q ON q.dataset = h.dataset AND q.query_id = h.query_id
        JOIN docs d ON d.dataset = h.dataset AND d.doc_id = h.doc_id
        WHERE h.dataset = 'esci-us' AND h.source = 'human'
        ORDER BY h.query_id, h.doc_id
    """).fetchall()
    con.close()
    rubric = load("esci")
    longer = sum(len(r[6] or "") > 800 for r in rows)
    print(f"{len(rows):,} human-labeled pairs from 150 queries; "
          f"{longer / len(rows):.0%} have descriptions over 800 characters\n")

    human = [r[7] for r in rows]
    results = {}
    for n in LENGTHS:
        def text(r, n=n):
            return ". ".join(x for x in (r[3], r[4], r[5], (r[6] or "")[:n]) if x)
        grades = grade_pairs(JUDGE, [(r[0], r[1], r[2], text(r)) for r in rows], rubric, workers=8)
        ok = [(h, g.grade) for h, g in zip(human, grades) if g.grade is not None]
        exact = sum(h == g for h, g in ok) / len(ok)
        kappa = weighted_kappa([h for h, _ in ok], [g for _, g in ok])
        results[n] = grades
        print(f"  first {n:>5,} characters   exact {exact:.3f}   kappa {kappa:.3f}   ({len(ok)} graded)")

    changed = sum(a.grade != b.grade for a, b in zip(*results.values()) if a.grade is not None and b.grade is not None)
    print(f"\n{changed} of {len(rows)} grades changed between the two lengths")
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "truncation.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query_id", "doc_id", "human", *[f"jev_{n}" for n in LENGTHS]])
        for r, *gs in zip(rows, *results.values()):
            w.writerow([r[0], r[1], r[7], *[g.grade for g in gs]])


if __name__ == "__main__":
    main()
