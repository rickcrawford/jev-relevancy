"""Per-query relevance metrics in SQL, plus bootstrap confidence intervals.

Definitions (k = 10 for nDCG and MRR, 50 for recall):
  nDCG@10            DCG over the top 10 / best possible DCG for that query.
                     Unjudged results count as gain 0.
  nDCG@10 condensed  Same, after dropping unjudged results and closing the gaps.
                     ESCI only labels ~20 products per query, so this is the
                     fairer number when retrieval surfaces lots of unlabeled items.
  Recall@50          Relevant products in the top 50 / all relevant judged products.
  MRR@10             1 / rank of the first relevant result in the top 10.
"Relevant" (for recall and MRR) means grade >= 2, Exact or Substitute.
A query whose judged products are all Irrelevant has no ideal ranking, so its
nDCG is NULL; one with nothing Exact or Substitute gets NULL recall and MRR.
NULLs drop out of the averages.
"""

import random

import duckdb

GAINS = {
    "exp": "(pow(2, {g}) - 1)",   # the usual TREC / KDD form; rewards Exact heavily
    "linear": "({g})",            # E=3, S=2, C=1, I=0
}

METRICS = ["ndcg10", "ndcg10_condensed", "recall50", "mrr10"]


def per_query_sql(gain: str = "exp") -> str:
    g = GAINS[gain].format(g="grade")
    return f"""
    WITH sample AS (
        SELECT query_id, slice FROM queries WHERE dataset = $dataset AND sampled
    ),
    run_queries AS (
        SELECT r.run_id, s.query_id, s.slice
        FROM runs r CROSS JOIN sample s
        WHERE r.dataset = $dataset AND r.run_id IN (SELECT unnest($run_ids))
    ),
    j AS (
        -- Human labels cover a fixed set of products per query. A model judge
        -- only grades what was pooled, so its ideal ranking is built from the
        -- top 10 of the runs being scored and nothing else; grades left over
        -- from other runs or samples would shift the scores.
        SELECT query_id, doc_id, grade FROM judgments jj
        WHERE dataset = $dataset AND source = $judge
          AND ($judge = 'human' OR EXISTS (
                SELECT 1 FROM results p
                WHERE p.run_id IN (SELECT unnest($run_ids)) AND p.rank <= 10
                  AND p.query_id = jj.query_id AND p.doc_id = jj.doc_id))
    ),
    graded AS (
        SELECT res.run_id, res.query_id, res.rank, j.grade
        FROM results res
        LEFT JOIN j USING (query_id, doc_id)
        WHERE res.run_id IN (SELECT unnest($run_ids))
    ),
    dcg AS (
        SELECT run_id, query_id,
               sum({GAINS[gain].format(g="coalesce(grade, 0)")} / log2(rank + 1))
                   FILTER (WHERE rank <= 10)                 AS dcg10,
               count(*) FILTER (WHERE rank <= 50 AND grade >= 2) AS rel50,
               min(rank) FILTER (WHERE rank <= 10 AND grade >= 2) AS first_rel
        FROM graded GROUP BY ALL
    ),
    condensed AS (
        SELECT run_id, query_id,
               sum({g} / log2(crank + 1)) FILTER (WHERE crank <= 10) AS cdcg10
        FROM (
            SELECT *, row_number() OVER (PARTITION BY run_id, query_id ORDER BY rank) AS crank
            FROM graded WHERE grade IS NOT NULL
        ) GROUP BY ALL
    ),
    ideal AS (
        SELECT query_id,
               sum({g} / log2(rn + 1)) FILTER (WHERE rn <= 10) AS idcg10,
               count(*) FILTER (WHERE grade >= 2) AS n_rel
        FROM (
            SELECT query_id, grade,
                   row_number() OVER (PARTITION BY query_id ORDER BY grade DESC) AS rn
            FROM j
        ) GROUP BY ALL
    )
    SELECT rq.run_id, rq.query_id, rq.slice,
           coalesce(d.dcg10, 0) / nullif(i.idcg10, 0)          AS ndcg10,
           coalesce(c.cdcg10, 0) / nullif(i.idcg10, 0)         AS ndcg10_condensed,
           coalesce(d.rel50, 0) / nullif(i.n_rel, 0)           AS recall50,
           CASE WHEN i.n_rel > 0 THEN coalesce(1.0 / d.first_rel, 0) END AS mrr10
    FROM run_queries rq
    LEFT JOIN dcg d       USING (run_id, query_id)
    LEFT JOIN condensed c USING (run_id, query_id)
    LEFT JOIN ideal i     USING (query_id)
    """


def bootstrap_ci(values: list[float], draws: int = 1000, seed: int = 0) -> tuple[float, float]:
    """95% percentile interval for the mean, resampling queries with replacement."""
    if len(values) < 2:
        return (float("nan"), float("nan"))
    values = sorted(values)  # SQL row order isn't stable; sorting makes the draw reproducible
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choices(values, k=n)) / n for _ in range(draws))
    # With 1,000 draws: the 25th smallest and the 975th smallest.
    return means[round(0.025 * draws) - 1], means[round(0.975 * draws) - 1]


def per_query(con: duckdb.DuckDBPyConnection, dataset: str, run_ids: list[str],
              judge: str = "human", gain: str = "exp") -> dict[tuple[str, str], dict[str, float | None]]:
    """{(run_id, query_id): {metric: value}} for every sampled query."""
    rows = con.execute(per_query_sql(gain), {"dataset": dataset, "run_ids": run_ids, "judge": judge}).fetchall()
    return {(run_id, qid): dict(zip(METRICS, vals)) for run_id, qid, _slice, *vals in rows}


def paired_delta(a: dict[str, float], b: dict[str, float], draws: int = 1000,
                 seed: int = 0) -> tuple[float, float, float, int]:
    """Mean of (a - b) over queries both runs scored, with a 95% bootstrap interval.

    Comparing two runs on the same queries this way is much tighter than
    checking whether their separate intervals overlap, because most of the
    query-to-query variation cancels out.
    """
    diffs = [a[q] - b[q] for q in sorted(a.keys() & b.keys())
             if a[q] is not None and b[q] is not None]
    if not diffs:
        return (float("nan"), float("nan"), float("nan"), 0)
    lo, hi = bootstrap_ci(diffs, draws, seed)
    return (sum(diffs) / len(diffs), lo, hi, len(diffs))


def score(con: duckdb.DuckDBPyConnection, dataset: str, run_ids: list[str],
          judge: str = "human", gain: str = "exp") -> None:
    """Compute metrics for each run and write them to the metrics table."""
    judge_label = judge if gain == "exp" else f"{judge}/{gain}"
    rows = con.execute(
        per_query_sql(gain),
        {"dataset": dataset, "run_ids": run_ids, "judge": judge},
    ).fetchall()

    by_key: dict[tuple[str, str, str], list[float]] = {}
    for run_id, _qid, slice_, *vals in rows:
        for metric, v in zip(METRICS, vals):
            if v is None:
                continue
            by_key.setdefault((run_id, metric, "all"), []).append(v)
            by_key.setdefault((run_id, metric, slice_), []).append(v)

    con.execute(
        "DELETE FROM metrics WHERE judge = ? AND run_id IN (SELECT unnest(?))",
        [judge_label, run_ids],
    )
    for (run_id, metric, slice_), vals in by_key.items():
        lo, hi = bootstrap_ci(vals) if slice_ == "all" else (None, None)
        con.execute(
            "INSERT INTO metrics VALUES (?, ?, ?, ?, ?, ?, ?)",
            [run_id, judge_label, metric, slice_, sum(vals) / len(vals), lo, hi],
        )
