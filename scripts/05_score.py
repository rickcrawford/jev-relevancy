"""Step 5: score runs against the judgments and print a comparison table.

  uv run scripts/05_score.py                 # every run on esci-us, human labels
  uv run scripts/05_score.py --gain linear   # E=3, S=2, C=1 instead of 2^grade - 1
  uv run scripts/05_score.py --by-slice      # break nDCG@10 down by query slice

Metric definitions live in src/relevancy/metrics.py.
"""

import argparse

from relevancy.metrics import score
from relevancy.warehouse import connect


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="esci-us")
    ap.add_argument("--judge", default="human", help="judgment source: human, jev-v1, ...")
    ap.add_argument("--gain", choices=["exp", "linear"], default="exp")
    ap.add_argument("--run", action="append", help="run_id to score (repeatable); default all")
    ap.add_argument("--by-slice", action="store_true")
    args = ap.parse_args()

    con = connect()
    run_ids = args.run or [r for (r,) in con.execute(
        "SELECT run_id FROM runs WHERE dataset = ? ORDER BY started_at", [args.dataset]).fetchall()]
    if not run_ids:
        raise SystemExit("No runs yet. Run scripts/04_search.py first.")

    score(con, args.dataset, run_ids, judge=args.judge, gain=args.gain)
    judge = args.judge if args.gain == "exp" else f"{args.judge}/{args.gain}"

    # The runs and judge being reported, as a temp table both queries can join on.
    con.execute("CREATE OR REPLACE TEMP TABLE picked AS SELECT unnest(?) AS run_id, ? AS judge",
                [run_ids, judge])

    con.sql("""
        WITH m AS (SELECT * FROM metrics JOIN picked USING (run_id, judge) WHERE slice = 'all'),
        lat AS (
            SELECT run_id,
                   quantile_cont(latency_ms, 0.5)  AS p50_ms,
                   quantile_cont(latency_ms, 0.99) AS p99_ms,
                   quantile_cont(server_ms, 0.5)   AS server_p50_ms
            FROM (SELECT DISTINCT run_id, query_id, latency_ms, server_ms FROM results)
            GROUP BY run_id
        )
        SELECT r.retriever, r.reranker,
               printf('%.3f [%.3f, %.3f]', n.value, n.ci_low, n.ci_high) AS "nDCG@10 [95% CI]",
               round(c.value, 3) AS "nDCG@10 condensed",
               round(rc.value, 3) AS "Recall@50",
               round(mr.value, 3) AS "MRR@10",
               round(lat.p50_ms, 1) AS "p50 ms",
               round(lat.p99_ms, 1) AS "p99 ms",
               round(lat.server_p50_ms, 1) AS "server p50 ms",
               r.run_id
        FROM runs r
        JOIN m n  ON n.run_id = r.run_id AND n.metric = 'ndcg10'
        JOIN m c  ON c.run_id = r.run_id AND c.metric = 'ndcg10_condensed'
        JOIN m rc ON rc.run_id = r.run_id AND rc.metric = 'recall50'
        JOIN m mr ON mr.run_id = r.run_id AND mr.metric = 'mrr10'
        LEFT JOIN lat ON lat.run_id = r.run_id
        ORDER BY n.value DESC
    """).show(max_width=250)

    if args.by_slice:
        print("nDCG@10 by query slice")
        con.sql("""
            PIVOT (
                SELECT r.retriever || '/' || r.reranker AS system, m.slice, round(m.value, 3) AS v
                FROM metrics m JOIN picked USING (run_id, judge) JOIN runs r USING (run_id)
                WHERE m.metric = 'ndcg10'
            ) ON slice USING first(v) ORDER BY system
        """).show(max_width=250)


if __name__ == "__main__":
    main()
