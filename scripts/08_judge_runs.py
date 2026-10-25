"""Step 8: can a cheap judge rank search systems the way humans do?

Pools the top 10 from the latest run of each configuration on the current
sample, has each judge grade every pooled (query, result) pair against the
dataset's rubric, then scores every run twice: once with the human labels and
once with the judge's. If the judge orders the runs the same way the humans
do (Kendall's tau near 1), it's good enough to pick a winner on data that has
no human labels.

On your own data with no human labels, the same command grades the pool and
reports each run's nDCG@10 under the judge, and skips the comparison.

  uv run scripts/08_judge_runs.py
  uv run scripts/08_judge_runs.py --judge jev-batched:jev-1.14.0 --judge openai/gpt-4o-mini
  uv run scripts/08_judge_runs.py --dataset mydocs --regrade   # after editing the rubric

Two things keep the comparison honest:
  - A run reranked by a judge is never scored by that same judge. A Jev-
    reranked run would grade its own homework, so it's left out of that
    judge's tau.
  - ESCI labels only ~20 products per query, so many pooled products have no
    human label. Tau is reported against both human nDCG (unlabeled counts as
    irrelevant) and condensed nDCG (unlabeled dropped).

Pairs already graded by a judge are skipped, so reruns only pay for new ones.
"""

import argparse

import pyarrow as pa

from relevancy.agreement import kendall_tau, weighted_kappa
from relevancy.judges import grade_pairs, model_of
from relevancy.metrics import per_query, score
from relevancy.rubric import for_dataset
from relevancy.warehouse import connect, sample_id

DEFAULT_JUDGES = ["jev-batched:jev-1.13.0", "google/gemma-3-27b-it"]
DEPTH = 10  # matches the nDCG@10 cutoff; metrics.py builds a model judge's ideal from the same top 10
CHECKPOINT = 25  # queries per write, so a crash halfway keeps what was already paid for


def grades_own_homework(judge: str, reranker: str) -> bool:
    """True if this run was reranked by the judge's own model (any Jev counts as Jev)."""
    if "jev" in judge:
        return "jev" in reranker
    return model_of(judge) == model_of(reranker)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--judge", action="append", help=f"repeatable; default {DEFAULT_JUDGES}")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dataset", default="esci-us")
    ap.add_argument("--regrade", action="store_true",
                    help="drop these judges' grades on this dataset first, e.g. after editing the rubric")
    args = ap.parse_args()
    judges = args.judge or DEFAULT_JUDGES

    with connect() as con:
        sample = sample_id(con, args.dataset)
        rubric = for_dataset(con, args.dataset)
        if args.regrade:
            con.execute("DELETE FROM judgments WHERE dataset = ? AND source IN (SELECT unnest(?))",
                        [args.dataset, judges])
        # Human labels on the sampled queries, if there are any. ESCI has
        # them; your own data usually won't, and then the judge is the label.
        has_human = con.execute("""
            SELECT count(*) > 0 FROM judgments j JOIN queries q USING (dataset, query_id)
            WHERE j.dataset = ? AND j.source = 'human' AND q.sampled
        """, [args.dataset]).fetchone()[0]
        # The latest run of each retriever/reranker combination on this
        # sample, so a configuration that was run twice doesn't count twice.
        runs = con.execute("""
            SELECT run_id, retriever, reranker FROM runs
            WHERE dataset = ? AND config->>'sample' = ?
            QUALIFY row_number() OVER (PARTITION BY retriever, reranker ORDER BY started_at DESC) = 1
            ORDER BY started_at
        """, [args.dataset, sample]).fetchall()
        # Ranking runs against the humans needs a few runs to rank.
        if len(runs) < (3 if has_human else 1):
            raise SystemExit("Not enough runs on the current sample. Run steps 4 and 6 first.")
        run_ids = [r[0] for r in runs]

        todo = {}
        for judge in judges:
            todo[judge] = con.execute(f"""
                SELECT DISTINCT r.query_id, r.doc_id, q.text,
                       concat_ws('. ', d.title, d.brand, d.color, left(d.body, 800))
                FROM results r
                JOIN queries q ON q.dataset = ? AND q.query_id = r.query_id
                JOIN docs d ON d.dataset = q.dataset AND d.doc_id = r.doc_id
                WHERE r.run_id IN (SELECT unnest(?)) AND r.rank <= {DEPTH}
                  AND NOT EXISTS (SELECT 1 FROM judgments j
                                  WHERE j.dataset = q.dataset AND j.query_id = r.query_id
                                    AND j.doc_id = r.doc_id AND j.source = ?)
                ORDER BY r.query_id, r.doc_id
            """, [args.dataset, run_ids, judge]).fetchall()

    for judge, pairs in todo.items():
        print(f"{judge}: grading {len(pairs):,} new pooled pairs")
        if not pairs:
            continue
        query_ids = list(dict.fromkeys(p[0] for p in pairs))
        failed = 0
        for i in range(0, len(query_ids), CHECKPOINT):
            chunk_ids = set(query_ids[i:i + CHECKPOINT])
            chunk = [p for p in pairs if p[0] in chunk_ids]
            out = grade_pairs(judge, chunk, rubric, args.workers)
            failed += sum(g.grade is None for g in out)
            with connect() as con:
                con.executemany(
                    "INSERT OR REPLACE INTO judgments "
                    "(dataset, query_id, doc_id, source, grade, prob, expected) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    [(args.dataset, p[0], p[1], judge, g.grade, g.prob, g.expected)
                     for p, g in zip(chunk, out) if g.grade is not None])
        if failed:
            # An ungraded product would count as irrelevant and drag the run down.
            print(f"  warning: {failed} pairs came back without a grade; rerun to retry just those")

    con = connect()
    for judge in judges:
        score(con, args.dataset, run_ids, judge=judge)

    if not has_human:
        # Nothing to check the judges against, so report what they say.
        table = con.execute("""
            SELECT ru.retriever || '/' || ru.reranker AS run, m.judge, round(m.value, 3) AS ndcg10,
                   '[' || round(m.ci_low, 3) || ', ' || round(m.ci_high, 3) || ']' AS ci95
            FROM metrics m JOIN runs ru USING (run_id)
            WHERE m.run_id IN (SELECT unnest(?)) AND m.judge IN (SELECT unnest(?))
              AND m.metric = 'ndcg10' AND m.slice = 'all'
            ORDER BY m.judge, m.value DESC
        """, [run_ids, judges]).fetchall()
        for row in table:
            print(f"  {row[0]:<40} {row[1]:<28} nDCG@10 {row[2]:.3f} {row[3]}")
        # nDCG is undefined for a query with nothing relevant to find, so
        # those queries drop out of the averages. Say how many.
        for judge in judges:
            empty, total = con.execute("""
                SELECT count(*) FILTER (WHERE best = 0), count(*) FROM (
                    SELECT max(grade) AS best FROM judgments j JOIN queries q USING (dataset, query_id)
                    WHERE j.dataset = ? AND j.source = ? AND q.sampled GROUP BY query_id)
            """, [args.dataset, judge]).fetchall()[0]
            if empty:
                print(f"  {judge}: {empty} of {total} queries had nothing graded above 0 in any "
                      "run's top 10, so nDCG leaves them out")
        print(f"\nNo human labels on {args.dataset}, so the judge's grades are the labels.")
        print("For paired differences between runs, score with one judge:")
        print(f"  uv run scripts/05_score.py --dataset {args.dataset} --judge {judges[0]}")
        return

    score(con, args.dataset, run_ids, judge="human")

    # Per-query scores under every judge. A query can score NULL under one
    # judge and not another (say, the humans marked every product Irrelevant),
    # so each comparison averages over the queries both sides could score.
    pq = {j: per_query(con, args.dataset, run_ids, judge=j) for j in ["human", *judges]}

    def ndcg(run_id: str, judge: str, metric: str = "ndcg10", against: str | None = None) -> float:
        vals = []
        for (rid, qid), m in pq[judge].items():
            if rid != run_id or m[metric] is None:
                continue
            if against and pq[against][(rid, qid)]["ndcg10"] is None:
                continue
            vals.append(m[metric])
        return sum(vals) / len(vals) if vals else float("nan")

    # Per-run scores, side by side.
    table = [{"run": f"{ret}/{rr}", "human": round(ndcg(rid, "human"), 3),
              "human condensed": round(ndcg(rid, "human", "ndcg10_condensed"), 3),
              **{j: round(ndcg(rid, j), 3) for j in judges}}
             for rid, ret, rr in runs]
    per_run = pa.Table.from_pylist(sorted(table, key=lambda r: -r["human"]))  # noqa: F841
    print("\nnDCG@10 for each run, under each judge")
    con.sql("SELECT * FROM per_run").show(max_width=250)

    # Does each judge put the runs in the same order as the humans?
    summary = []
    for judge in judges:
        fair = [(rid, rr) for rid, _, rr in runs if not grades_own_homework(judge, rr)]
        ids = [rid for rid, _ in fair]
        (n, exact, h, g) = con.execute(f"""
            SELECT count(*), avg((h.grade = j.grade)::INT), list(h.grade), list(j.grade)
            FROM judgments h JOIN judgments j USING (dataset, query_id, doc_id)
            WHERE h.dataset = ? AND h.source = 'human' AND j.source = ?
              -- only pairs in this pool, so every judge is compared on the same pairs
              AND (h.query_id, h.doc_id) IN (SELECT query_id, doc_id FROM results
                                             WHERE run_id IN (SELECT unnest(?)) AND rank <= {DEPTH})
        """, [args.dataset, judge, run_ids]).fetchall()[0]
        summary.append({
            "judge": judge,
            "runs ranked": len(ids),
            "left out (own reranker)": len(runs) - len(ids),
            "tau vs human": round(kendall_tau(
                [ndcg(r, "human", against=judge) for r in ids],
                [ndcg(r, judge, against="human") for r in ids]), 3),
            "tau vs human condensed": round(kendall_tau(
                [ndcg(r, "human", "ndcg10_condensed", against=judge) for r in ids],
                [ndcg(r, judge, against="human") for r in ids]), 3),
            "pairs with a human label": n,
            "exact agreement": round(exact, 3) if exact is not None else None,
            "kappa": round(weighted_kappa(h, g), 3) if n else None,
        })
    agreement = pa.Table.from_pylist(summary)  # noqa: F841
    print("\nAgreement with the human labels")
    con.sql("SELECT * FROM agreement").show(max_width=250)


if __name__ == "__main__":
    main()
