"""Four retrieval questions the main runs leave open, on the full 482k namespace.

  bm25      Why does title+body BM25 take ~300 ms of server time on long queries?
            Runs four BM25 variants on the 500 sampled queries, one at a time.
  onecall   Hybrid search and Voyage reranking in one multi_query request
            (rerank_by on the outer request), with 50 and 100 candidates per
            subquery, against the two-request pipeline the main runs use.
  filters   On queries that name a brand, does filtering to that brand help?
            The brand comes from the query's human-Exact products, so this is
            an upper bound: a perfect brand detector.
  judged    KDD Cup Task 1 style: rerank only each query's judged products, so
            the numbers sit next to published ESCI reranking results.

  uv run experiments/retrieval-followups/retrieval_followups.py bm25
  uv run experiments/retrieval-followups/retrieval_followups.py report

Each writes results/followup_<name>.csv; `report` scores them with the human
labels in the warehouse (read-only). Total cost is well under $1.
"""

import csv
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from relevancy.judges import grade_pairs
from relevancy.metrics import paired_delta, per_query
from relevancy.rubric import load
from relevancy.tpuf import (
    bm25_rank_by,
    client,
    hybrid_queries,
    requests_by_query,
    rerank,
)
from relevancy.warehouse import read_only

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "results"
DS, NAMESPACE = "esci-us", "esci-us-full"
TEXT = "concat_ws('. ', d.title, d.brand, d.color, left(d.body, 800))"
BM25 = {
    "title2x+body": bm25_rank_by,  # what the main runs use
    "title+body": lambda q: ("Sum", (("title", "BM25", q), ("body", "BM25", q))),
    "body": lambda q: ("body", "BM25", q),
    "title": lambda q: ("title", "BM25", q),
}


def warehouse():
    return read_only()


def sampled(con):
    return con.execute("SELECT query_id, text FROM queries WHERE dataset = ? AND sampled ORDER BY query_id",
                       [DS]).fetchall()


def timed(call, *args, **kwargs):
    started = time.perf_counter()
    result = call(*args, **kwargs)
    return result, (time.perf_counter() - started) * 1000


def joined_ids(hits) -> str:
    return " ".join(hits["id"].to_pylist())


def write(name, rows, fields):
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"followup_{name}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to results/followup_{name}.csv")


def run_bm25():
    with warehouse() as con:
        queries = sampled(con)
    tp, rows = client(), []
    for variant, rank_by in BM25.items():
        run = []
        for qid, text in queries:
            hits, wall = timed(tp.query, NAMESPACE, rank_by(text), top_k=50, run_id=f"bm25-{variant}", query_id=qid)
            run.append({"variant": variant, "query_id": qid, "words": len(text.split()), "wall_ms": round(wall, 1),
                        "ids": joined_ids(hits)})
        logged = requests_by_query(tp, f"bm25-{variant}")
        rows += [r | {"server_ms": logged[r["query_id"]]["server_ms"],
                      "exec_ms": logged[r["query_id"]]["query_execution_ms"]} for r in run]
        print(f"  {variant} done")
    write("bm25", rows, ["variant", "query_id", "words", "wall_ms", "server_ms", "exec_ms", "ids"])


def run_onecall():
    with warehouse() as con:
        queries = sampled(con)
    tp, rows = client(), []
    for k in (50, 100):
        run = []
        for qid, text in queries:
            (hits,), wall = timed(
                tp.multi_query, NAMESPACE, hybrid_queries(text, k),
                rerank_by={"model": "voyage/rerank-3", "attribute": "text", "query": text},
                limit=50, run_id=f"onecall-{k}", query_id=qid)
            run.append({"variant": f"onecall-{k}", "query_id": qid, "wall_ms": round(wall, 1), "ids": joined_ids(hits)})
        logged = requests_by_query(tp, f"onecall-{k}")
        rows += [r | {"server_ms": logged[r["query_id"]]["server_ms"],
                      "billable_bytes": logged[r["query_id"]]["billable_bytes_queried"]} for r in run]
        print(f"  top_k {k} done")
    write("onecall", rows, ["variant", "query_id", "wall_ms", "server_ms", "billable_bytes", "ids"])


def run_filters():
    with warehouse() as con:
        # The brand an Exact product carries and the query names; the most
        # common one if there are several.
        branded = con.execute("""
            SELECT q.query_id, q.text, mode(d.brand) AS brand
            FROM queries q JOIN judgments h USING (dataset, query_id) JOIN docs d USING (dataset, doc_id)
            WHERE q.dataset = ? AND q.sampled AND h.source = 'human' AND h.grade = 3
              AND length(d.brand) >= 3 AND contains(lower(q.text), lower(d.brand))
            GROUP BY ALL ORDER BY q.query_id""", [DS]).fetchall()
    tp, rows = client(), []
    for qid, text, brand in branded:
        for variant, filters in (("unfiltered", None), ("brand filter", ("brand", "Eq", brand))):
            extra = {"filters": filters} if filters else {}
            (hits,), wall = timed(tp.multi_query, NAMESPACE, hybrid_queries(text, 50, **extra),
                                  rerank_by=("RRF", {}), limit=50, run_id=f"filters-{variant}", query_id=qid)
            rows.append({"variant": variant, "query_id": qid, "brand": brand, "wall_ms": round(wall, 1),
                         "ids": joined_ids(hits)})
    logged = {v: requests_by_query(tp, f"filters-{v}") for v in ("unfiltered", "brand filter")}
    for r in rows:
        r["server_ms"] = logged[r["variant"]][r["query_id"]]["server_ms"]
    write("filters", rows, ["variant", "query_id", "brand", "wall_ms", "server_ms", "ids"])


def run_judged():
    with warehouse() as con:
        cands = con.execute(f"""
            SELECT q.query_id, q.text, list(d.doc_id ORDER BY d.doc_id), list({TEXT} ORDER BY d.doc_id)
            FROM queries q JOIN judgments h USING (dataset, query_id) JOIN docs d USING (dataset, doc_id)
            WHERE q.dataset = ? AND q.sampled AND h.source = 'human'
            GROUP BY ALL ORDER BY q.query_id""", [DS]).fetchall()
    tp, rubric = client(), load("esci")

    def voyage(c):
        qid, text, doc_ids, _ = c
        return {"variant": "voyage", "query_id": qid, "ids": joined_ids(rerank(tp, NAMESPACE, "voyage/rerank-3", text, doc_ids))}

    def jev(c):
        qid, text, ids, texts = c
        grades = grade_pairs("jev-batched:jev-1.13.0", [(qid, d, text, t) for d, t in zip(ids, texts)], rubric)
        order = sorted(range(len(ids)), key=lambda i: -(grades[i].expected or 0))
        return {"variant": "jev", "query_id": qid, "ids": " ".join(ids[i] for i in order)}

    with ThreadPoolExecutor(8) as pool:
        rows = list(pool.map(voyage, cands)) + list(pool.map(jev, cands))
    # The same products sorted by id, an arbitrary order, as the floor.
    rows += [{"variant": "id order", "query_id": c[0], "ids": " ".join(c[2])} for c in cands]
    write("judged", rows, ["variant", "query_id", "ids"])


# -- scoring ------------------------------------------------------------------

def ndcg(ranked, labels, gain=lambda g: 2 ** g - 1):
    ideal = sorted((gain(g) for g in labels.values()), reverse=True)[:10]
    idcg = sum(x / math.log2(i + 2) for i, x in enumerate(ideal))
    if not idcg:
        return None
    return sum(gain(labels.get(d, 0)) / math.log2(i + 2) for i, d in enumerate(ranked[:10])) / idcg


def q(vals, p):
    vals = sorted(vals)
    return vals[min(len(vals) - 1, int(p * len(vals)))]


def report():
    con = warehouse()
    labels = {}
    for qid, d, g in con.execute("SELECT query_id, doc_id, grade FROM judgments WHERE dataset = ? AND source = 'human'",
                                 [DS]).fetchall():
        labels.setdefault(qid, {})[d] = g

    def read(name):
        with open(OUT / f"followup_{name}.csv") as f:
            return list(csv.DictReader(f))

    def scores(rows, variant, gain=lambda g: 2 ** g - 1):
        out = {}
        for r in rows:
            if r["variant"] == variant:
                v = ndcg(r["ids"].split(), labels.get(r["query_id"], {}), gain)
                if v is not None:
                    out[r["query_id"]] = v
        return out

    # The stored two-request pipeline: the latest Voyage rerank of hybrid on the full namespace.
    voyage_run = con.execute("""
        SELECT run_id FROM runs WHERE dataset = ? AND reranker = 'voyage/rerank-3' AND config->>'namespace' = ?
        ORDER BY started_at DESC LIMIT 1""", [DS, NAMESPACE]).fetchone()
    if voyage_run is None:
        raise SystemExit(f"No voyage/rerank-3 run on {NAMESPACE}. Run scripts/06_rerank.py --model voyage/rerank-3 first.")
    voyage_run = voyage_run[0]
    voyage2 = {q_: m["ndcg10"] for (_, q_), m in per_query(con, DS, [voyage_run]).items()
               if m["ndcg10"] is not None}

    if (OUT / "followup_bm25.csv").exists():
        rows = read("bm25")
        print("\n## BM25 variants: server time by query length, and quality")
        print(f"{'variant':<14}{'nDCG@10':>8}{'vs main':>24}{'server p50':>12}{'p99':>6}  median server ms by words 1/2/3/4/5/6+")
        base = scores(rows, "title2x+body")
        for v in BM25:
            s = scores(rows, v)
            m, lo, hi, _ = paired_delta(s, base)
            rs = [r for r in rows if r["variant"] == v]
            srv = [float(r["server_ms"]) for r in rs]
            by = [q([float(r["server_ms"]) for r in rs if min(int(r["words"]), 6) == w] or [0], .5) for w in range(1, 7)]
            print(f"{v:<14}{sum(s.values()) / len(s):8.3f}{f'{m:+.3f} [{lo:+.3f}, {hi:+.3f}]':>24}"
                  f"{q(srv, .5):12.0f}{q(srv, .99):6.0f}  " + "/".join(f"{x:.0f}" for x in by))

    if (OUT / "followup_onecall.csv").exists():
        rows = read("onecall")
        print("\n## Hybrid + Voyage in one request, against the two-request pipeline")
        # A reranked run stores latency and server time already summed over both requests.
        stored = con.execute("""
            SELECT quantile_cont(latency_ms, 0.5), quantile_cont(server_ms, 0.5)
            FROM (SELECT DISTINCT query_id, latency_ms, server_ms FROM results WHERE run_id = ?)
            """, [voyage_run]).fetchone()
        print(f"  two requests (stored runs): nDCG@10 {sum(voyage2.values()) / len(voyage2):.3f}, "
              f"wall p50 {stored[0]:.0f} ms, server p50 {stored[1]:.0f} ms (both requests)")
        for v in ("onecall-50", "onecall-100"):
            s = scores(rows, v)
            m, lo, hi, _ = paired_delta(s, voyage2)
            rs = [r for r in rows if r["variant"] == v]
            print(f"  {v}: nDCG@10 {sum(s.values()) / len(s):.3f}, vs two requests {m:+.3f} [{lo:+.3f}, {hi:+.3f}], "
                  f"wall p50 {q([float(r['wall_ms']) for r in rs], .5):.0f} ms, "
                  f"server p50 {q([float(r['server_ms']) for r in rs], .5):.0f} ms, "
                  f"{float(rs[0]['billable_bytes']) / 1e9:.2f} GB billed per query")

    if (OUT / "followup_filters.csv").exists():
        rows = read("filters")
        print("\n## Brand filter on branded queries (brand from the human-Exact products: an upper bound)")
        a, b = scores(rows, "brand filter"), scores(rows, "unfiltered")
        m, lo, hi, n = paired_delta(a, b)
        for v in ("unfiltered", "brand filter"):
            rs = [r for r in rows if r["variant"] == v]
            print(f"  {v}: nDCG@10 {sum(scores(rows, v).values()) / n:.3f}, "
                  f"server p50 {q([float(r['server_ms']) for r in rs], .5):.0f} ms, "
                  f"{sum(not r['ids'] for r in rs)} queries returned nothing")
        print(f"  filter minus unfiltered: {m:+.3f} [{lo:+.3f}, {hi:+.3f}], n={n}")

    if (OUT / "followup_judged.csv").exists():
        rows = read("judged")
        kdd = {3: 1.0, 2: 0.1, 1: 0.01, 0: 0.0}
        print("\n## Reranking only the judged products (KDD Cup Task 1 style)")
        for v in ("voyage", "jev", "id order"):
            s, k = scores(rows, v), scores(rows, v, kdd.get)
            print(f"  {v:<9} nDCG@10 {sum(s.values()) / len(s):.3f} (exp gain), "
                  f"{sum(k.values()) / len(k):.3f} (KDD gain E=1, S=0.1, C=0.01), n={len(s)}")
        m, lo, hi, _ = paired_delta(scores(rows, "jev"), scores(rows, "voyage"))
        print(f"  jev minus voyage, exp gain: {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        m, lo, hi, _ = paired_delta(scores(rows, "jev", kdd.get), scores(rows, "voyage", kdd.get))
        print(f"  jev minus voyage, KDD gain: {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")


if __name__ == "__main__":
    commands = {"bm25": run_bm25, "onecall": run_onecall, "filters": run_filters, "judged": run_judged,
                "report": report}
    if len(sys.argv) != 2 or sys.argv[1] not in commands:
        raise SystemExit(__doc__)
    commands[sys.argv[1]]()
