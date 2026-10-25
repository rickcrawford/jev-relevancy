"""Does the order of products in a batched Jev request change its grades?

Step 6 and step 8 send all of a query's products to Jev in one request, as
separate questions. If Jev's answer to one question leaked into the next,
the order of products would change the grades. This regrades the pooled
pairs for 100 sampled queries twice: once in the original order (to
measure plain run-to-run noise), and once with each query's products
reversed. Then it compares both with the grades step 8 stored.

  uv run experiments/judge-batch-order/batch_order.py
"""

import csv
from pathlib import Path

from relevancy.judges import grade_pairs
from relevancy.rubric import load
from relevancy.warehouse import read_only

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "results"
JUDGE = "jev-batched:jev-1.13.0"


def compare(name, stored, pairs, grades):
    same = [stored[(p[0], p[1])][0] == g.grade for p, g in zip(pairs, grades) if g.grade is not None]
    diff = [abs(stored[(p[0], p[1])][1] - g.expected) for p, g in zip(pairs, grades) if g.expected is not None]
    print(f"  {name:<28} same grade {sum(same) / len(same):6.1%}   "
          f"mean |change in expected grade| {sum(diff) / len(diff):.3f}   ({len(same)} pairs)")
    return sum(same) / len(same), sum(diff) / len(diff)


def main() -> None:
    con = read_only()
    rows = con.execute(f"""
        WITH picked AS (
            SELECT query_id FROM queries WHERE dataset = 'esci-us' AND sampled
            ORDER BY hash(query_id || 'batch-order') LIMIT 100
        )
        SELECT j.query_id, j.doc_id, q.text,
               concat_ws('. ', d.title, d.brand, d.color, left(d.body, 800)), j.grade, j.expected
        FROM judgments j
        JOIN picked USING (query_id)
        JOIN queries q USING (dataset, query_id)
        JOIN docs d USING (dataset, doc_id)
        WHERE j.dataset = 'esci-us' AND j.source = '{JUDGE}'
        ORDER BY j.query_id, j.doc_id
    """).fetchall()
    con.close()
    rubric = load("esci")
    stored = {(r[0], r[1]): (r[4], r[5]) for r in rows}
    pairs = [r[:4] for r in rows]
    print(f"{len(pairs):,} pairs over 100 queries, judged by {JUDGE}\n")

    same_order = grade_pairs(JUDGE, pairs, rubric, workers=8)
    # Reversed within each query: grade_pairs batches by query in the order pairs arrive.
    reversed_pairs = sorted(pairs, key=lambda p: (p[0], [-ord(c) for c in p[1]]))
    by_key = dict(zip([(p[0], p[1]) for p in reversed_pairs],
                      grade_pairs(JUDGE, reversed_pairs, rubric, workers=8)))
    reversed_order = [by_key[(p[0], p[1])] for p in pairs]

    a = compare("same order, graded again", stored, pairs, same_order)
    b = compare("products reversed", stored, pairs, reversed_order)

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "batch_order.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query_id", "doc_id", "stored_grade", "stored_expected",
                    "same_order_grade", "same_order_expected", "reversed_grade", "reversed_expected"])
        for p, g1, g2 in zip(pairs, same_order, reversed_order):
            s = stored[(p[0], p[1])]
            w.writerow([p[0], p[1], s[0], s[1], g1.grade, g1.expected, g2.grade, g2.expected])
    verdict = "order matters" if a[0] - b[0] > 0.02 else "order makes no difference beyond run-to-run noise"
    print(f"\n{b[0]:.1%} unchanged after reversing, {a[0]:.1%} after regrading as is: {verdict}.")
    print(f"Rows: {OUT / 'batch_order.csv'}")


if __name__ == "__main__":
    main()
