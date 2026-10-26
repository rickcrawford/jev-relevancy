"""A second catalog: does Jev's agreement with human labels hold on Wayfair's WANDS?

  download  Free. WANDS's three tab-separated files from GitHub into data/wands/.
  prepare   Free. Converts them and loads dataset `wands` through
            02_prepare_custom.py with --sample 380, so 100 queries stay out of the
            sample for step 7's bake-off. Exact/Partial/Irrelevant become grades
            3/2/0 (rubrics/wands.toml).
  report    Free. After the pipeline steps in study.toml: Kendall's tau between the
            runs' nDCG@10 under human and under Jev labels, with a bootstrap
            interval over queries, and Jev's pooled kappa against the humans,
            checked against the pass marks fixed in the README.

  uv run experiments/judge-wands/wands.py download
"""

import argparse
import random
import subprocess
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

import duckdb

from relevancy.agreement import kendall_tau, weighted_kappa
from relevancy.metrics import per_query
from relevancy.warehouse import ROOT, read_only, sample_id

OUT = Path(__file__).resolve().parent / "results"
DATA = ROOT / "data" / "wands"
URL = "https://raw.githubusercontent.com/wayfair/WANDS/main/dataset"
DATASET, JUDGE = "wands", "jev-batched:jev-1.13.0"
GRADES = {"Exact": 3, "Partial": 2, "Irrelevant": 0}
HOLDS, FAILS = (0.71, 0.35), (0.60, 0.30)  # (tau, kappa), fixed in the README before the run


def run_download(args):
    DATA.mkdir(parents=True, exist_ok=True)
    for f in ("product.csv", "query.csv", "label.csv"):
        if not (DATA / f).exists():
            urllib.request.urlretrieve(f"{URL}/{f}", DATA / f)
        print(f"  {f}: {(DATA / f).stat().st_size / 1e6:.1f} MB")


def run_prepare(args):
    con = duckdb.connect()
    read = "read_csv(?, delim='\\t', header=true, all_varchar=true, quote='\"')"
    con.execute(f"""COPY (SELECT product_id AS id, product_name AS title,
                         concat_ws(' ', product_description, product_features) AS body,
                         NULL AS brand, NULL AS color
                  FROM {read}) TO '{DATA}/docs.parquet'""", [str(DATA / "product.csv")])
    con.execute(f"""COPY (SELECT query_id, query AS text, query_class AS slice FROM {read})
                  TO '{DATA}/queries.csv' (HEADER)""", [str(DATA / "query.csv")])
    con.execute(f"""COPY (SELECT query_id, product_id AS doc_id,
                         CASE label WHEN 'Exact' THEN 3 WHEN 'Partial' THEN 2 ELSE 0 END AS grade
                  FROM {read}) TO '{DATA}/labels.csv' (HEADER)""", [str(DATA / "label.csv")])
    cmd = [sys.executable, str(ROOT / "scripts" / "02_prepare_custom.py"), "--dataset", DATASET,
           "--docs", str(DATA / "docs.parquet"), "--queries", str(DATA / "queries.csv"),
           "--judgments", str(DATA / "labels.csv"), "--rubric", "wands", "--sample", "380",
           "--description", "Wayfair WANDS (MIT)"]
    raise SystemExit(subprocess.call(cmd, cwd=ROOT))


def fair_runs(con) -> list[str]:
    """Step 8's selection, so tau here is the same statistic as on ESCI: the latest run of each
    retriever and reranker on the current sample, minus runs reranked by any Jev model (a Jev
    judge would be grading its own homework)."""
    rows = con.execute("""
        SELECT run_id, reranker FROM runs WHERE dataset = ? AND config->>'sample' = ?
        QUALIFY row_number() OVER (PARTITION BY retriever, reranker ORDER BY started_at DESC) = 1
        ORDER BY run_id""", [DATASET, sample_id(con, DATASET)]).fetchall()
    return [run_id for run_id, reranker in rows if "jev" not in reranker]


def tau_with_interval(con, runs, draws=1000):
    human = per_query(con, DATASET, runs, judge="human")
    judged = per_query(con, DATASET, runs, judge=JUDGE)
    queries = sorted({q for _, q in human})

    def run_mean(scores, r, qs):
        # Like step 8: average over the queries both sides could score, not counting NULL as 0.
        vals = [scores[(r, q)]["ndcg10"] for q in qs
                if human[(r, q)]["ndcg10"] is not None and judged[(r, q)]["ndcg10"] is not None]
        return sum(vals) / len(vals) if vals else float("nan")

    def tau(qs):
        return kendall_tau([run_mean(human, r, qs) for r in runs], [run_mean(judged, r, qs) for r in runs])

    rnd = random.Random(0)
    taus = sorted(tau(rnd.choices(queries, k=len(queries))) for _ in range(draws))
    return tau(queries), taus[24], taus[974]


def run_report(args):
    with read_only() as con:
        runs = fair_runs(con)
        if len(runs) < 3:
            raise SystemExit("Fewer than 3 runs on wands. Run the pipeline steps first (relevancy show judge-wands).")
        tau, lo, hi = tau_with_interval(con, runs)
        pairs = con.execute("""
            SELECT h.grade, j.grade, h.query_id FROM judgments h JOIN judgments j USING (dataset, query_id, doc_id)
            WHERE h.dataset = ? AND h.source = 'human' AND j.source = ?""", [DATASET, JUDGE]).fetchall()
    kappa = weighted_kappa([p[0] for p in pairs], [p[1] for p in pairs])
    by_q = defaultdict(list)
    for h, j, q in pairs:
        by_q[q].append((h, j))
    rnd, qs = random.Random(0), sorted(by_q)
    ks = sorted(weighted_kappa(*map(list, zip(*[x for q in rnd.choices(qs, k=len(qs)) for x in by_q[q]])))
                for _ in range(500))
    exact = sum(h == j for h, j, _ in pairs) / len(pairs)
    print(f"  {len(runs)} runs on wands")
    print(f"  tau (human vs Jev nDCG@10 ordering of runs): {tau:.3f} [{lo:.3f}, {hi:.3f}]")
    print(f"  pooled kappa: {kappa:.3f} [{ks[12]:.3f}, {ks[487]:.3f}], exact {exact:.3f}, on {len(pairs):,} labeled pairs")
    if tau >= HOLDS[0] and kappa >= HOLDS[1]:
        print(f"  HOLDS: tau >= {HOLDS[0]} and kappa >= {HOLDS[1]}")
    elif tau < FAILS[0] or kappa < FAILS[1]:
        print(f"  FAILS: tau < {FAILS[0]} or kappa < {FAILS[1]}")
    else:
        print("  PARTIAL: between the pass and fail marks")


if __name__ == "__main__":
    commands = {"download": run_download, "prepare": run_prepare, "report": run_report}
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=commands)
    commands[ap.parse_args().command](None)
