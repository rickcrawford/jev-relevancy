"""Step 7: how well do Jev and LLM judges agree with the human labels?

Picks human-labeled pairs from test queries that are NOT in the evaluation
sample (50 per grade by default), has each judge grade them, and reports
agreement with the humans, plus speed and cost.

Judges, picked by prefix:
  jev:jev-1.13.0          Jev via TypeSafe's API, one request per pair
  jev-batched:jev-1.13.0  Jev with every pair for a query in one request,
                          the way step 6 reranks; checks that batching
                          doesn't change the answers
  openai/gpt-4o-mini      any OpenRouter model, read through logprobs

The sample is class-balanced on purpose: real traffic is mostly Exact and
Substitute, which would hide how a judge handles the rarer grades. So read
these numbers as "how well does it separate the four grades", not as the
agreement you'd see on live results. Kappa in particular depends on the mix.

Grades are saved to `judgments` as source '<judge>@bakeoff', kept apart from
the pooled grades step 8 stores under the plain judge name.

Latency is per request. For jev-batched that's one request for all of a
query's pairs, so it isn't comparable with the one-pair-per-request judges.

  uv run scripts/07_judge_bakeoff.py
  uv run scripts/07_judge_bakeoff.py --judge jev:jev-1.13.0 --per-grade 100

A few cents per judge at the default size.
"""

import argparse

import pyarrow as pa

from relevancy.agreement import pearson, weighted_kappa
from relevancy.judges import grade_pairs
from relevancy.warehouse import connect

JUDGES = [
    "jev:jev-1.13.0",
    "jev:jev-1.14.0",
    "jev-batched:jev-1.13.0",
    "google/gemma-3-27b-it",
    "meta-llama/llama-4-maverick",
    "openai/gpt-4o-mini",
    "meta-llama/llama-3.1-8b-instruct",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--judge", action="append", help="repeatable; default is the list above")
    ap.add_argument("--per-grade", type=int, default=50)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dataset", default="esci-us")
    args = ap.parse_args()

    with connect() as con:
        pairs = con.execute(f"""
            SELECT query_id, doc_id, q.text,
                   concat_ws('. ', d.title, d.brand, d.color, left(d.body, 800)), grade
            FROM (
                SELECT j.*, row_number() OVER (PARTITION BY j.grade ORDER BY hash(j.query_id || j.doc_id)) AS rn
                FROM judgments j JOIN queries q USING (dataset, query_id)
                WHERE j.dataset = ? AND j.source = 'human' AND NOT q.sampled
            ) j
            JOIN queries q USING (dataset, query_id)
            JOIN docs d USING (dataset, doc_id)
            WHERE rn <= {args.per_grade}
            ORDER BY grade, query_id, doc_id
        """, [args.dataset]).fetchall()
    print(f"{len(pairs)} human-labeled pairs, {args.per_grade} per grade\n")

    if not pairs:
        raise SystemExit("No held-out labeled pairs. Run scripts/02_prepare_esci.py first.")

    summary = []
    for judge in args.judge or JUDGES:
        out = grade_pairs(judge, pairs, args.workers)
        ok = [(p[4], g) for p, g in zip(pairs, out) if g.grade is not None]
        if not ok:
            first_error = next((g.error for g in out if g.error), "no grades")
            print(f"{judge:<44} failed: {first_error[:100]}")
            continue

        source = f"{judge}@bakeoff"
        with connect() as con:
            con.execute("DELETE FROM judgments WHERE dataset = ? AND source = ?", [args.dataset, source])
            con.executemany(
                "INSERT INTO judgments (dataset, query_id, doc_id, source, grade, prob, expected) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                [(args.dataset, p[0], p[1], source, g.grade, g.prob, g.expected)
                 for p, g in zip(pairs, out) if g.grade is not None],
            )

        human = [h for h, _ in ok]
        graded = [g.grade for _, g in ok]
        lat = sorted(g.latency_ms for _, g in ok)
        costs = [g.cost for g in out if g.cost is not None]
        summary.append({
            "judge": judge,
            "answered": f"{len(ok)}/{len(pairs)}",
            "with probs": round(sum(g.prob is not None for g in out) / len(out), 2),
            "exact": round(sum(h == m for h, m in zip(human, graded)) / len(ok), 3),
            "within 1": round(sum(abs(h - m) <= 1 for h, m in zip(human, graded)) / len(ok), 3),
            "kappa": round(weighted_kappa(human, graded), 3),
            "r(expected)": round(pearson([float(h) for h in human], [g.expected for _, g in ok]), 3),
            "p50 ms": round(lat[len(lat) // 2]),
            # OpenRouter reports cost per call. TypeSafe doesn't, so Jev shows NULL;
            # at $0.042 per 1M input tokens it's a fraction of a cent here.
            "cost $": round(sum(costs), 4) if costs else None,
        })
        s = summary[-1]
        print(f"{judge:<44} exact {s['exact']:.2f}  kappa {s['kappa']:.2f}  p50 {s['p50 ms']} ms")

    if summary:
        print()
        bakeoff = pa.Table.from_pylist(summary)  # noqa: F841  (DuckDB reads it by name)
        with connect() as con:
            con.sql("SELECT * FROM bakeoff ORDER BY kappa DESC").show(max_width=250)


if __name__ == "__main__":
    main()
