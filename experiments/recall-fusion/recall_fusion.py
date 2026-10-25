"""Where does recall go, and how much can first-stage fusion get back?

  fetch     Pull each retriever's top 1,000 for the 500 sampled queries, one
            request per query: ann, bm25 (title 2x + body), and title (BM25 on
            the title alone). Saved to data/recall/, which is git-ignored, one
            file per list, so a rerun skips lists it already has.
  diagnose  For every product humans rated Exact or Substitute, its best rank
            under any retriever: top 10, 11-50, 51-100, 101-1,000, or not
            found. Products ranked past 50 are a depth and fusion problem; ones
            nothing finds in 1,000 are a vocabulary problem, and their word
            overlap with the query says how much.
  check     Fuse the stored lists the way turbopuffer does (RRF, k=60, top 50
            of each) and compare with the stored hybrid-rrf run. If they
            don't match, the sweep below doesn't describe the server.
  sweep     Reciprocal rank fusion, done here instead of on the server, over a
            grid of which lists, how deep, RRF k, and the vector list's
            weight. Configs are picked on half the queries and reported on the
            other half, so the winner isn't just the luckiest of 180.
  rerank    voyage/rerank-3 over three candidate sets: the baseline's top 50,
            the baseline at depth 100 (its top 100), and the sweep's best
            recall setting (its top 100). Does more recall turn into nDCG@10,
            and is it depth or fusion that does it?

  uv run experiments/recall-fusion/recall_fusion.py fetch
  uv run experiments/recall-fusion/recall_fusion.py diagnose

Costs: 1,500 turbopuffer queries for fetch and 1,500 Voyage reranks for
rerank. The sweep makes no requests. The test plan and decision rules are in
README.md, next to this script.
"""

import argparse
import csv
import hashlib
import itertools
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from relevancy.metrics import paired_delta
from relevancy.study import fuse
from relevancy.tpuf import RANK_BY, client, requests_by_query, rerank
from relevancy.warehouse import read_only

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "results"
DATA = ROOT / "data" / "recall"
DS, NAMESPACE = "esci-us", "esci-us-full"
DEPTH = 1_000
LISTS = {"ann": "ann", "bm25": "bm25", "title": "bm25-title"}  # our name: tpuf.RANK_BY name
BASELINE = {"lists": ("ann", "bm25"), "depth": 50, "k": 60, "w_ann": 0.5}  # what hybrid-rrf does
GRID = {
    "lists": [("ann", "bm25"), ("ann", "title"), ("ann", "bm25", "title")],
    "depth": [50, 100, 200, 1_000],
    "k": [20, 60, 120],
    "w_ann": [0.3, 0.4, 0.5, 0.6, 0.7],  # the BM25 lists split the rest equally
}
BUCKETS = ["top 10", "11-50", "51-100", "101-1000", "not found"]


def warehouse():
    return read_only()


def sampled(con):
    return con.execute("SELECT query_id, text FROM queries WHERE dataset = ? AND sampled ORDER BY query_id",
                       [DS]).fetchall()


def labels(con) -> dict[str, dict[str, int]]:
    out = defaultdict(dict)
    for qid, d, g in con.execute("SELECT query_id, doc_id, grade FROM judgments WHERE dataset = ? AND source = 'human'",
                                 [DS]).fetchall():
        out[qid][d] = g
    return out


def tune_half(qid: str) -> bool:
    return int(hashlib.md5(qid.encode()).hexdigest(), 16) % 2 == 0


def write(name, rows, fields):
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"recall_{name}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to results/recall_{name}.csv")


# -- fetch --------------------------------------------------------------------

def run_fetch():
    with warehouse() as con:
        queries = sampled(con)
    DATA.mkdir(parents=True, exist_ok=True)
    for name, retriever in LISTS.items():
        if (DATA / f"lists-{name}.parquet").exists():
            print(f"  {name} already fetched", flush=True)
            continue
        tp, tables = client(), []
        for qid, text in queries:
            hits = tp.query(NAMESPACE, RANK_BY[retriever](text), top_k=DEPTH, run_id=f"recall-{name}", query_id=qid)
            n = hits.num_rows
            tables.append(pa.table({"list": [name] * n, "query_id": [qid] * n,
                                    "doc_id": hits["id"], "rank": hits["rank"]}))
        pq.write_table(pa.concat_tables(tables), DATA / f"lists-{name}.parquet")
        rows = [{"list": name, "query_id": qid, "server_ms": r["server_ms"], "wall_ms": round(r["wall_ms"], 1)}
                for qid, r in requests_by_query(tp, f"recall-{name}").items()]
        write(f"fetch_latency_{name}", rows, ["list", "query_id", "server_ms", "wall_ms"])
        print(f"  {name} done", flush=True)


def lists() -> dict[str, dict[str, list[str]]]:
    """{list name: {query_id: doc ids in rank order}}."""
    out = defaultdict(dict)
    for name, qid, ids in duckdb.sql(f"""
            SELECT list, query_id, list(doc_id ORDER BY rank) FROM '{DATA}/lists-*.parquet' GROUP BY ALL""").fetchall():
        out[name][qid] = ids
    return out


# -- diagnose -----------------------------------------------------------------

def words(text: str | None) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def bucket(rank: int | None) -> str:
    if rank is None:
        return "not found"
    return "top 10" if rank <= 10 else "11-50" if rank <= 50 else "51-100" if rank <= 100 else "101-1000"


def run_diagnose():
    with warehouse() as con:
        queries, human = dict(sampled(con)), labels(con)
        titles = dict(con.execute("""
            SELECT d.doc_id, d.title FROM docs d JOIN judgments h USING (dataset, doc_id)
            JOIN queries q USING (dataset, query_id)
            WHERE d.dataset = ? AND q.sampled AND h.source = 'human' AND h.grade >= 2""", [DS]).fetchall())
    ranked = lists()
    rank_of = {name: {qid: {d: i + 1 for i, d in enumerate(ids)} for qid, ids in by_q.items()}
               for name, by_q in ranked.items()}

    rows = []
    for qid, text in queries.items():
        for doc_id, grade in human.get(qid, {}).items():
            if grade < 2:
                continue
            ranks = {name: rank_of[name].get(qid, {}).get(doc_id) for name in LISTS}
            found = [r for r in ranks.values() if r is not None]
            q_words = words(text)
            overlap = len(q_words & words(titles.get(doc_id))) / len(q_words) if q_words else 0.0
            rows.append({"query_id": qid, "doc_id": doc_id, "grade": grade, **{f"{n}_rank": r for n, r in ranks.items()},
                         "best": min(found) if found else None, "bucket": bucket(min(found) if found else None),
                         "title_overlap": round(overlap, 3)})
    write("diagnose", rows, list(rows[0]))

    for grade, label in ((3, "Exact"), (2, "Substitute")):
        rs = [r for r in rows if r["grade"] == grade]
        counts = Counter(r["bucket"] for r in rs)
        print(f"\n## {label}: {len(rs):,} products, best rank under any of {', '.join(LISTS)}")
        for b in BUCKETS:
            share = [r["title_overlap"] for r in rs if r["bucket"] == b]
            mean = sum(share) / len(share) if share else float("nan")
            print(f"  {b:<10} {counts[b]:>6,}  {counts[b] / len(rs):6.1%}   query words in the title: {mean:.2f}")

    print("\n## Exact and Substitute products in a list's top 50 that no other list has in its top 50")
    for name in LISTS:
        alone = sum(1 for r in rows if (r[f"{name}_rank"] or DEPTH + 1) <= 50
                    and all((r[f"{o}_rank"] or DEPTH + 1) > 50 for o in LISTS if o != name))
        print(f"  {name:<6} {alone:>6,}")

    # Averaged per query, like the stored Recall@50 (0.471 for hybrid), not pooled over products.
    by_query = defaultdict(list)
    for r in rows:
        by_query[r["query_id"]].append(r)

    def recall_at(found):
        return sum(sum(map(found, rs)) / len(rs) for rs in by_query.values()) / len(by_query)

    print("\n## Recall of Exact and Substitute products, averaged per query")
    hybrid50 = {q: set(fuse(ranked, q, BASELINE, top=50)) for q in by_query}
    print(f"  hybrid-rrf top 50 (fused here): {recall_at(lambda r: r['doc_id'] in hybrid50[r['query_id']]):.3f}")
    for depth in (50, 100, 200, 1_000):
        per = [recall_at(lambda r, n=n, d=depth: (r[f"{n}_rank"] or DEPTH + 1) <= d) for n in LISTS]
        union = recall_at(lambda r, d=depth: r["best"] is not None and r["best"] <= d)
        print(f"  each list's top {depth:>5}: " + "  ".join(f"{n} {p:.3f}" for n, p in zip(LISTS, per))
              + f"  any list {union:.3f}")


# -- fusion -------------------------------------------------------------------

def gain(g):
    return 2 ** g - 1


def ndcg(ranked, labels_, condensed=False):
    ideal = sorted((gain(g) for g in labels_.values()), reverse=True)[:10]
    idcg = sum(x / math.log2(i + 2) for i, x in enumerate(ideal))
    if not idcg:
        return None
    if condensed:
        ranked = [d for d in ranked if d in labels_]
    return sum(gain(labels_.get(d, 0)) / math.log2(i + 2) for i, d in enumerate(ranked[:10])) / idcg


def recall(ranked, labels_, at):
    rel = {d for d, g in labels_.items() if g >= 2}
    return len(rel & set(ranked[:at])) / len(rel) if rel else None


METRICS = {
    "nDCG@10": lambda r, l: ndcg(r, l),
    "condensed": lambda r, l: ndcg(r, l, True),
    "R@50": lambda r, l: recall(r, l, 50),
    "R@100": lambda r, l: recall(r, l, 100),
}


def per_query(fused: dict[str, list[str]], human, metric) -> dict[str, float]:
    out = {}
    for qid, ids in fused.items():
        v = METRICS[metric](ids, human.get(qid, {}))
        if v is not None:
            out[qid] = v
    return out


def name(cfg: dict) -> str:
    return f"{'+'.join(cfg['lists'])} depth={cfg['depth']} k={cfg['k']} w_ann={cfg['w_ann']}"


def run_check():
    with warehouse() as con:
        run_id = con.execute("""SELECT run_id FROM runs WHERE dataset = ? AND retriever = 'hybrid-rrf'
                                AND reranker = 'none' AND config->>'namespace' = ?
                                ORDER BY started_at DESC LIMIT 1""", [DS, NAMESPACE]).fetchone()[0]
        stored = {qid: ids for qid, ids in con.execute(
            "SELECT query_id, list(doc_id ORDER BY rank) FROM results WHERE run_id = ? GROUP BY ALL", [run_id]).fetchall()}
    ranked = lists()
    overlap10, overlap50, same_order = [], [], 0
    for qid, ids in stored.items():
        local = fuse(ranked, qid, BASELINE, top=50)
        overlap10.append(len(set(local[:10]) & set(ids[:10])) / 10)
        overlap50.append(len(set(local) & set(ids)) / max(len(ids), 1))
        same_order += local[:10] == ids[:10]
    n = len(stored)
    print(f"local RRF against stored run {run_id}, {n} queries:")
    print(f"  top-10 overlap {sum(overlap10) / n:.3f}, top-50 overlap {sum(overlap50) / n:.3f}, "
          f"identical top 10 on {same_order} queries")
    print("  Below about 0.95 top-50 overlap, the sweep isn't modelling the server; look at how it ranks ties first.")


def run_sweep():
    with warehouse() as con:
        human, queries = labels(con), [q for q, _ in sampled(con)]
    ranked = lists()
    tune = [q for q in queries if tune_half(q)]
    held = [q for q in queries if not tune_half(q)]
    configs = [dict(zip(GRID, values)) for values in itertools.product(*GRID.values())]
    configs = [c for c in configs if "ann" in c["lists"]]

    def scores(cfg, qids):
        fused = {q: fuse(ranked, q, cfg) for q in qids}
        return {m: per_query(fused, human, m) for m in METRICS}

    base_tune, base_held = scores(BASELINE, tune), scores(BASELINE, held)
    rows = []
    for cfg in configs:
        s = scores(cfg, tune)
        rows.append({"config": name(cfg), **{f"tune_{m}": round(sum(v.values()) / len(v), 4) for m, v in s.items()},
                     **{f"tune_d_{m}": round(paired_delta(v, base_tune[m])[0], 4) for m, v in s.items()}})
    write("sweep", rows, list(rows[0]))

    print(f"\n{len(configs)} configs, picked on {len(tune)} queries, reported on the other {len(held)}")
    print(f"baseline: {name(BASELINE)}")
    # Standard nDCG counts every unlabeled product as irrelevant, which favors settings that
    # return what Amazon's search returned when the labels were made. Condensed nDCG doesn't.
    for pick in ("nDCG@10", "condensed", "R@100"):
        best_row = max(rows, key=lambda r: r[f"tune_d_{pick}"])
        cfg = next(c for c in configs if name(c) == best_row["config"])
        s = scores(cfg, held)
        print(f"\n## Best on the tuning half by {pick}: {name(cfg)}")
        for m in METRICS:
            mean, lo, hi, n = paired_delta(s[m], base_held[m])
            print(f"  held-out {m:<10} {sum(s[m].values()) / len(s[m]):.3f}  vs baseline {mean:+.3f} [{lo:+.3f}, {hi:+.3f}]  n={n}")


def run_rerank():
    """Voyage over three candidate sets, against Voyage over the baseline's top 50."""
    with warehouse() as con:
        human, queries = labels(con), sampled(con)
    with open(OUT / "recall_sweep.csv") as f:
        best = max(csv.DictReader(f), key=lambda r: float(r["tune_d_R@100"]))["config"]
    configs = [dict(zip(GRID, values)) for values in itertools.product(*GRID.values())]
    cfg = next(c for c in configs if name(c) == best)
    ranked, tp = lists(), client()
    # "depth 100" separates the gain from more candidates alone from the gain from better fusion.
    variants = {"baseline top 50": (BASELINE, 50), "depth 100 top 100": ({**BASELINE, "depth": 100}, 100),
                "best top 100": (cfg, 100)}
    out, rows = {}, []
    for v, (c, n) in variants.items():
        out[v], short = {}, 0
        for qid, text in queries:
            ids = fuse(ranked, qid, c, top=n)
            hits = rerank(tp, NAMESPACE, "voyage/rerank-3", text, ids, run_id=f"recall-{v}", query_id=qid)
            out[v][qid] = hits["id"].to_pylist()
            short += len(out[v][qid]) < len(ids)
            rows.append({"variant": v, "query_id": qid, "ids": " ".join(out[v][qid])})
        print(f"  {v} done" + (f", {short} queries came back with fewer products than were sent" if short else ""),
              flush=True)
    write("rerank", rows, ["variant", "query_id", "ids"])

    print(f"\nbest candidate config (by tuning-half R@100): {best}")
    held = {q for q, _ in queries if not tune_half(q)}
    for m in ("nDCG@10", "condensed"):
        base = per_query({q: ids for q, ids in out["baseline top 50"].items() if q in held}, human, m)
        for v in variants:
            s = per_query({q: ids for q, ids in out[v].items() if q in held}, human, m)
            mean, lo, hi, n = paired_delta(s, base)
            print(f"  {v:<18} held-out {m:<9} {sum(s.values()) / len(s):.3f}  "
                  f"vs baseline top 50 {mean:+.3f} [{lo:+.3f}, {hi:+.3f}]  n={n}")
        # The primary comparison: better fusion against depth alone, both at 100 candidates.
        s = per_query({q: ids for q, ids in out["best top 100"].items() if q in held}, human, m)
        d = per_query({q: ids for q, ids in out["depth 100 top 100"].items() if q in held}, human, m)
        mean, lo, hi, n = paired_delta(s, d)
        print(f"  best top 100 vs depth 100 top 100, {m}: {mean:+.3f} [{lo:+.3f}, {hi:+.3f}]  n={n}")


if __name__ == "__main__":
    commands = {"fetch": run_fetch, "diagnose": run_diagnose, "check": run_check, "sweep": run_sweep, "rerank": run_rerank}
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=commands)
    commands[ap.parse_args().command]()
