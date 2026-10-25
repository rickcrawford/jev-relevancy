"""Where does turbopuffer's time go, and how much recall does ANN give up?

The main runs log wall-clock time and turbopuffer's server_total_ms. A query's
response has more than that: how long the server spent embedding the query
text, how long it spent executing, how hot the cache was, and how many
documents it had to scan because they weren't indexed yet. This script reruns
the first-stage retrievers on the 500 sampled queries, twice each in a row,
and records all of it. Then it asks turbopuffer's recall endpoint how ANN
compares with exhaustive search on the same namespace.

  uv run experiments/turbopuffer-perf/turbopuffer_perf.py            # about 10 minutes, under a cent
  uv run experiments/turbopuffer-perf/turbopuffer_perf.py --report   # summarize results/turbopuffer_perf.csv

Queries run one at a time, so server time never includes waiting on the
namespace's concurrency limit.
"""

import argparse
import csv
import json
import time
from pathlib import Path

import duckdb

from relevancy.tpuf import client, requests_by_query, search

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "results"
NAMESPACE = "esci-us-full"
RETRIEVERS = ["bm25", "ann", "hybrid-rrf"]
FIELDS = ["pass", "retriever", "query_id", "wall_ms", "server_total_ms", "query_execution_ms",
          "embedding_ms", "embedding_tokens", "cache_temperature", "cache_hit_ratio",
          "exhaustive_search_count", "billable_bytes", "top50"]


def run() -> None:
    con = duckdb.connect(str(ROOT / "warehouse" / "relevancy.duckdb"), read_only=True)
    queries = con.execute("SELECT query_id, text FROM queries WHERE dataset = 'esci-us' AND sampled "
                          "ORDER BY query_id").fetchall()
    con.close()
    tp = client()

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "turbopuffer_perf.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for pass_ in (1, 2):
            for retriever in RETRIEVERS:
                run_id, top50, wall = f"perf-{pass_}-{retriever}", {}, {}
                for qid, text in queries:
                    started = time.perf_counter()
                    hits = search(tp, NAMESPACE, retriever, text, run_id=run_id, query_id=qid)
                    wall[qid] = (time.perf_counter() - started) * 1000
                    top50[qid] = " ".join(hits["id"].to_pylist())
                # Everything else turbopuffer reported is in tpuf-arrow's request log.
                for qid, r in requests_by_query(tp, run_id).items():
                    w.writerow({"pass": pass_, "retriever": retriever, "query_id": qid,
                                "wall_ms": round(wall[qid], 1), "server_total_ms": r["server_ms"],
                                "query_execution_ms": r["query_execution_ms"], "embedding_ms": r["embedding_ms"],
                                "embedding_tokens": r["embedding_tokens"],
                                "cache_temperature": r["cache_temperature"],
                                "cache_hit_ratio": r["cache_hit_ratio"],
                                "exhaustive_search_count": r["exhaustive_search_count"],
                                "billable_bytes": r["billable_bytes_queried"], "top50": top50[qid]})
                print(f"pass {pass_} {retriever}: done")

    # Recall of ANN against exhaustive search. The endpoint picks query
    # vectors from the namespace itself; it doesn't accept a text query.
    # tpuf-arrow doesn't wrap it, so this uses the SDK client it holds.
    ns = tp.tpuf.namespace(NAMESPACE)
    started = time.perf_counter()
    r = ns.recall(top_k=50, num=100)
    recall = {"top_k": 50, "num": 100, "avg_recall": r.avg_recall, "avg_ann_count": r.avg_ann_count,
              "avg_exhaustive_count": r.avg_exhaustive_count,
              "seconds": round(time.perf_counter() - started, 1)}
    r10 = ns.recall(top_k=10, num=100)
    recall["top10_avg_recall"] = r10.avg_recall
    (OUT / "turbopuffer_recall.json").write_text(json.dumps(recall, indent=2) + "\n")
    print(recall)


def q(values: list[float], p: float) -> float:
    values = sorted(values)
    return values[min(len(values) - 1, int(p * len(values)))]


def report() -> None:
    with open(OUT / "turbopuffer_perf.csv") as f:
        rows = list(csv.DictReader(f))
    print(f"{'retriever':<11}{'pass':>5}{'wall p50':>10}{'p99':>7}{'server p50':>12}{'p99':>7}"
          f"{'embed p50':>11}{'exec p50':>10}{'hot':>6}{'unindexed':>11}")
    for retriever in RETRIEVERS:
        for pass_ in ("1", "2"):
            rs = [r for r in rows if r["retriever"] == retriever and r["pass"] == pass_]
            def num(k, rs=rs):
                return [float(r[k]) for r in rs if r[k] not in ("", "None")]
            emb = num("embedding_ms")
            print(f"{retriever:<11}{pass_:>5}{q(num('wall_ms'), .5):10.0f}{q(num('wall_ms'), .99):7.0f}"
                  f"{q(num('server_total_ms'), .5):12.0f}{q(num('server_total_ms'), .99):7.0f}"
                  f"{(q(emb, .5) if emb else 0):11.0f}{q(num('query_execution_ms'), .5):10.0f}"
                  f"{sum(r['cache_temperature'] == 'hot' for r in rs) / len(rs):6.0%}"
                  f"{q(num('exhaustive_search_count'), .5):11.0f}")
    # Are results stable between passes and against the stored runs?
    same = {}
    for retriever in RETRIEVERS:
        a = {r["query_id"]: r["top50"].split() for r in rows if r["retriever"] == retriever and r["pass"] == "1"}
        b = {r["query_id"]: r["top50"].split() for r in rows if r["retriever"] == retriever and r["pass"] == "2"}
        same[retriever] = sum(a[k][:10] == b[k][:10] for k in a) / len(a)
    print("top 10 identical between passes: " + ", ".join(f"{k} {v:.1%}" for k, v in same.items()))
    path = OUT / "turbopuffer_recall.json"
    if path.exists():
        print("recall:", path.read_text().strip())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", action="store_true")
    if ap.parse_args().report:
        report()
    else:
        run()
        report()
