"""Shared helpers for the studies in experiments/.

Splitting queries into a tuning half and a held-out half, reading labels and
stored runs from the warehouse, scoring ranked lists, paired comparisons
(including an equivalence test), weighted reciprocal rank fusion, resumable
part files for long labeling jobs, and calibration error.
"""

import csv
import hashlib
import heapq
import math
import random
from collections import defaultdict
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from relevancy.metrics import paired_delta

DS, NAMESPACE = "esci-us", "esci-us-full"
# What rerankers and judges read for a product: the same text the main runs use.
TEXT = "concat_ws('. ', d.title, d.brand, d.color, left(d.body, 800))"


# -- queries and labels -------------------------------------------------------

def tune_half(qid: str) -> bool:
    """True for the half of the queries settings are picked on; report the other half."""
    return int(hashlib.md5(qid.encode()).hexdigest(), 16) % 2 == 0


def sampled(con: duckdb.DuckDBPyConnection, dataset: str = DS) -> list[tuple[str, str]]:
    return con.execute("SELECT query_id, text FROM queries WHERE dataset = ? AND sampled ORDER BY query_id",
                       [dataset]).fetchall()


def labels(con: duckdb.DuckDBPyConnection, dataset: str = DS, source: str = "human") -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = defaultdict(dict)
    for qid, d, g in con.execute("SELECT query_id, doc_id, grade FROM judgments WHERE dataset = ? AND source = ?",
                                 [dataset, source]).fetchall():
        out[qid][d] = g
    return out


def doc_texts(con: duckdb.DuckDBPyConnection, ids, dataset: str = DS) -> dict[str, str]:
    con.register("_ids", pa.table({"doc_id": pa.array(sorted(set(ids)), pa.string())}))
    rows = con.execute(f"SELECT d.doc_id, {TEXT} FROM docs d JOIN _ids USING (doc_id) WHERE d.dataset = ?",
                       [dataset]).fetchall()
    con.unregister("_ids")
    return dict(rows)


# -- stored runs --------------------------------------------------------------

def latest_run(con: duckdb.DuckDBPyConnection, retriever: str = "hybrid-rrf", reranker: str = "none",
               dataset: str = DS, namespace: str = NAMESPACE) -> str:
    row = con.execute("""SELECT run_id FROM runs WHERE dataset = ? AND retriever = ? AND reranker = ?
                         AND config->>'namespace' = ? ORDER BY started_at DESC LIMIT 1""",
                      [dataset, retriever, reranker, namespace]).fetchone()
    if row is None:
        raise SystemExit(f"No {retriever} run with reranker {reranker} on {namespace}. "
                         "Build the base first: uv run relevancy base")
    return row[0]


def run_lists(con: duckdb.DuckDBPyConnection, run_id: str) -> dict[str, list[tuple[str, float]]]:
    """{query_id: [(doc_id, score), ...] in rank order} for one stored run."""
    out: dict[str, list[tuple[str, float]]] = {}
    for qid, ids, scores in con.execute("""
            SELECT query_id, list(doc_id ORDER BY rank), list(score ORDER BY rank)
            FROM results WHERE run_id = ? GROUP BY query_id""", [run_id]).fetchall():
        out[qid] = list(zip(ids, scores))
    return out


# -- metrics ------------------------------------------------------------------

def gain(g: int) -> float:
    return 2 ** g - 1


def ndcg(ranked: list[str], labels_: dict[str, int], condensed: bool = False, k: int = 10) -> float | None:
    """nDCG@k with exponential gain. Condensed drops unlabeled results first."""
    ideal = sorted((gain(g) for g in labels_.values()), reverse=True)[:k]
    idcg = sum(x / math.log2(i + 2) for i, x in enumerate(ideal))
    if not idcg:
        return None
    if condensed:
        ranked = [d for d in ranked if d in labels_]
    return sum(gain(labels_.get(d, 0)) / math.log2(i + 2) for i, d in enumerate(ranked[:k])) / idcg


def recall(ranked: list[str], labels_: dict[str, int], at: int) -> float | None:
    rel = {d for d, g in labels_.items() if g >= 2}
    return len(rel & set(ranked[:at])) / len(rel) if rel else None


METRICS = {
    "nDCG@10": lambda r, l: ndcg(r, l),
    "condensed": lambda r, l: ndcg(r, l, condensed=True),
    "R@50": lambda r, l: recall(r, l, 50),
}


def per_query(ranked: dict[str, list[str]], labels_: dict[str, dict[str, int]], metric: str = "nDCG@10",
              only=None) -> dict[str, float]:
    fn = METRICS[metric]
    out = {}
    for qid, ids in ranked.items():
        if only is not None and qid not in only:
            continue
        v = fn(ids, labels_.get(qid, {}))
        if v is not None:
            out[qid] = v
    return out


def mean(d: dict) -> float:
    return sum(d.values()) / len(d) if d else float("nan")


def compare(a: dict[str, float], b: dict[str, float]) -> str:
    """'0.452 vs 0.437: +0.015 [+0.004, +0.026] n=250' for two per-query score dicts."""
    m, lo, hi, n = paired_delta(a, b)
    return f"{mean(a):.3f} vs {mean(b):.3f}: {m:+.3f} [{lo:+.3f}, {hi:+.3f}] n={n}"


def interval(diffs: list[float], level: float = 0.90, draws: int = 1000, seed: int = 0) -> tuple[float, float]:
    """Bootstrap percentile interval for the mean of paired differences."""
    if len(diffs) < 2:
        return (float("nan"), float("nan"))
    diffs = sorted(diffs)
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(diffs, k=len(diffs))) / len(diffs) for _ in range(draws))
    tail = (1 - level) / 2
    return means[max(0, round(tail * draws) - 1)], means[min(draws - 1, round((1 - tail) * draws) - 1)]


def tost(a: dict[str, float], b: dict[str, float], margin: float) -> tuple[float, float, float, int, bool]:
    """Two one-sided tests at 5%: equivalent if the 90% interval of (a - b) sits inside ±margin."""
    diffs = [a[q] - b[q] for q in sorted(a.keys() & b.keys())]
    lo, hi = interval(diffs, 0.90)
    m = sum(diffs) / len(diffs) if diffs else float("nan")
    return m, lo, hi, len(diffs), (-margin < lo and hi < margin)


# -- fusion -------------------------------------------------------------------

def rrf(lists: dict[str, list[str]], weights: dict[str, float] | None = None, k: int = 60,
        depth: int | None = None, top: int = 50) -> list[str]:
    """Weighted reciprocal rank fusion: score = sum of weight / (k + rank), rank from 1.
    Ties break on doc id, descending, so the order is reproducible."""
    scores: dict[str, float] = defaultdict(float)
    for name, ids in lists.items():
        w = 1.0 if weights is None else weights.get(name, 0.0)
        for i, d in enumerate(ids[:depth]):
            scores[d] += w / (k + i + 1)
    return [d for d, _ in heapq.nlargest(top, scores.items(), key=lambda x: (x[1], x[0]))]


def fuse(ranked: dict[str, dict[str, list[str]]], qid: str, cfg: dict, top: int = 200) -> list[str]:
    """recall-fusion's settings: lists, depth, k, and the vector list's weight (BM25 lists split the rest)."""
    bm25_lists = [n for n in cfg["lists"] if n != "ann"]
    weights = {n: (1 - cfg["w_ann"]) / len(bm25_lists) for n in bm25_lists}
    if "ann" in cfg["lists"]:
        weights["ann"] = cfg["w_ann"]
    return rrf({n: ranked[n].get(qid, []) for n in weights}, weights, cfg["k"], cfg["depth"], top)


# -- files --------------------------------------------------------------------

def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or (list(rows[0]) if rows else [])
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows):,} rows to {path.name}")


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"{path} is missing. Run the step that writes it first.")
    with open(path) as f:
        return list(csv.DictReader(f))


def part_ids(folder: Path, column: str = "id") -> set[str]:
    """Ids already written to a folder of parquet parts, so a long job can resume."""
    if not any(folder.glob("*.parquet")):
        return set()
    return {r[0] for r in duckdb.sql(f"SELECT {column} FROM '{folder}/*.parquet'").fetchall()}


def write_part(folder: Path, rows: list[dict], schema: pa.Schema) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    n = len(list(folder.glob("*.parquet")))
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), folder / f"part-{n:05d}.parquet")


# -- calibration --------------------------------------------------------------

def ece(probs: list[float], truth: list[bool], bins: int = 10) -> float:
    """Expected calibration error of yes-probabilities against yes/no truth."""
    total, err = len(probs), 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, p in enumerate(probs) if lo <= p < hi or (b == bins - 1 and p == 1.0)]
        if idx:
            conf = sum(probs[i] for i in idx) / len(idx)
            acc = sum(truth[i] for i in idx) / len(idx)
            err += len(idx) / total * abs(conf - acc)
    return err
