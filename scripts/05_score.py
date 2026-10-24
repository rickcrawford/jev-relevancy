"""Step 5: score runs against the judgments and print a comparison table.

  uv run scripts/05_score.py                 # every run on esci-us, human labels
  uv run scripts/05_score.py --gain linear   # E=3, S=2, C=1 instead of 2^grade - 1
  uv run scripts/05_score.py --by-slice      # break nDCG@10 down by query slice

The "vs base" column is the paired difference in nDCG@10 from the plain
hybrid-rrf run, query by query, with a 95% bootstrap interval. If that
interval excludes 0, the difference is unlikely to be noise. That's a much
sharper test than eyeballing whether two runs' own intervals overlap.

Metric definitions live in src/relevancy/metrics.py.
"""

import argparse

from relevancy.metrics import paired_delta, per_query, score
from relevancy.warehouse import connect, sample_id


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="esci-us")
    ap.add_argument("--judge", default="human", help="judgment source: human, jev:jev-1.13.0, ...")
    ap.add_argument("--gain", choices=["exp", "linear"], default="exp")
    ap.add_argument("--run", action="append",
                    help="run_id to score (repeatable); default is every run on the current sample")
    ap.add_argument("--by-slice", action="store_true")
    ap.add_argument("--base", help="run_id to compare against; default is the latest plain hybrid-rrf run")
    args = ap.parse_args()

    con = connect()
    # By default, only runs made on the current query sample. A run from an
    # earlier sample would be scored against queries it never searched.
    sample = sample_id(con, args.dataset)
    if args.run:
        run_ids = args.run
        stale = [r for (r,) in con.execute(
            "SELECT run_id FROM runs WHERE run_id IN (SELECT unnest(?)) "
            "AND coalesce(config->>'sample', '') <> ?", [run_ids, sample]).fetchall()]
        for r in stale:
            print(f"warning: {r} was made on a different query sample; queries it never searched score 0")
    else:
        run_ids = [r for (r,) in con.execute(
            "SELECT run_id FROM runs WHERE dataset = ? AND config->>'sample' = ? ORDER BY started_at",
            [args.dataset, sample]).fetchall()]
    if not run_ids:
        raise SystemExit("No runs on the current query sample. Run scripts/04_search.py first.")

    score(con, args.dataset, run_ids, judge=args.judge, gain=args.gain)
    judge = args.judge if args.gain == "exp" else f"{args.judge}/{args.gain}"

    # The runs and judge being reported, as a temp table both queries can join on.
    con.execute("CREATE OR REPLACE TEMP TABLE picked AS SELECT unnest(?) AS run_id, ? AS judge",
                [run_ids, judge])

    # Paired nDCG@10 difference from the base run, query by query.
    base = args.base or next((r for (r,) in con.execute(
        "SELECT run_id FROM runs WHERE run_id IN (SELECT unnest(?)) "
        "AND retriever = 'hybrid-rrf' AND reranker = 'none' ORDER BY started_at DESC",
        [run_ids]).fetchall()), None)
    con.execute("CREATE OR REPLACE TEMP TABLE deltas (run_id VARCHAR, vs_base VARCHAR)")
    if base:
        pq = per_query(con, args.dataset, run_ids, judge=args.judge, gain=args.gain)
        by_run: dict[str, dict[str, float]] = {}
        for (run_id, qid), m in pq.items():
            by_run.setdefault(run_id, {})[qid] = m["ndcg10"]
        for run_id in run_ids:
            if run_id == base:
                continue
            d, lo, hi, _n = paired_delta(by_run.get(run_id, {}), by_run[base])
            con.execute("INSERT INTO deltas VALUES (?, ?)", [run_id, f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]"])

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
               coalesce(d.vs_base, 'base') AS "nDCG@10 vs base [95% CI]",
               r.run_id
        FROM runs r
        LEFT JOIN m n  ON n.run_id = r.run_id AND n.metric = 'ndcg10'
        LEFT JOIN m c  ON c.run_id = r.run_id AND c.metric = 'ndcg10_condensed'
        LEFT JOIN m rc ON rc.run_id = r.run_id AND rc.metric = 'recall50'
        LEFT JOIN m mr ON mr.run_id = r.run_id AND mr.metric = 'mrr10'
        LEFT JOIN lat ON lat.run_id = r.run_id
        LEFT JOIN deltas d ON d.run_id = r.run_id
        WHERE r.run_id IN (SELECT run_id FROM picked)
        ORDER BY n.value DESC NULLS LAST
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
