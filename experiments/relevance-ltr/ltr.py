"""Learning to rank: does a model trained on every signal beat any single reranker?

  train-queries  Free. 5,000 queries from ESCI's train split, stratified by the
                 same length and brand slices as the test sample, with their
                 human labels. Read from the raw ESCI parquet.
  features       Paid, resumable. For every train query and the 500 test queries:
                 hybrid's top 100, then each candidate's vector distance, three
                 BM25 scores (title x2 + body, title, body), RRF rank, Voyage's
                 score, and Jev's four grade probabilities, plus brand, color, and
                 title-word matches. A list that didn't return a candidate gives
                 a missing-value flag, not a zero.
  train          Free. LightGBM LambdaRank on the train queries, hyperparameters by
                 3-fold cross-validation on train only. Ablations without Jev,
                 without Voyage, without both, and a model trained on Jev's grades
                 instead of the human labels.
  report         Free. Each model against Voyage over the same top 100 on the 500
                 test queries, nDCG@10 and condensed.

  uv run experiments/relevance-ltr/ltr.py train-queries
"""

import argparse
import itertools
import json
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from relevancy import jev
from relevancy.rubric import load
from relevancy.study import (
    NAMESPACE,
    compare,
    doc_texts,
    labels,
    part_ids,
    per_query,
    read_csv,
    sampled,
    write_csv,
    write_part,
)
from relevancy.tpuf import ann_rank_by, bm25_rank_by, client, hybrid_queries, rerank
from relevancy.warehouse import ROOT, read_only

OUT = Path(__file__).resolve().parent / "results"
DATA = ROOT / "data" / "ltr"
RAW = ROOT / "data" / "raw" / "esci"
DEPTH = 100
GROUPS = {
    "retrieval": ["rrf_rank", "ann_dist", "ann_missing", "bm25", "bm25_missing", "title", "title_missing",
                  "body", "body_missing"],
    "match": ["brand_match", "color_match", "title_overlap"],
    "voyage": ["voyage", "voyage_missing"],
    "jev": ["jev_p0", "jev_p1", "jev_p2", "jev_p3", "jev_expected"],
}
FEATURES = [f for g in GROUPS.values() for f in g]
SCHEMA = pa.schema([("split", pa.string()), ("query_id", pa.string()), ("doc_id", pa.string())]
                   + [(f, pa.float64()) for f in FEATURES])
GRID = {"num_leaves": [15, 31], "learning_rate": [0.05, 0.1], "min_child_samples": [20]}


def run_train_queries(args):
    con = duckdb.connect()
    con.execute(f"""
        CREATE VIEW examples AS SELECT CAST(query_id AS VARCHAR) AS query_id, query, product_id, esci_label, split
        FROM '{RAW}/shopping_queries_dataset_examples.parquet' WHERE product_locale = 'us' AND small_version = 1;
        CREATE VIEW products AS SELECT * FROM '{RAW}/shopping_queries_dataset_products.parquet' WHERE product_locale = 'us';
    """)
    picked = con.execute(f"""
        WITH q AS (SELECT DISTINCT query_id, query FROM examples WHERE split = 'train'),
        branded AS (
            SELECT DISTINCT e.query_id FROM examples e JOIN products p USING (product_id)
            WHERE e.split = 'train' AND e.esci_label = 'E' AND length(p.product_brand) >= 3
              AND contains(lower(e.query), lower(p.product_brand))),
        sliced AS (
            SELECT q.query_id, q.query,
                   CASE WHEN len(regexp_split_to_array(trim(q.query), '\\s+')) <= 2 THEN 'short'
                        WHEN len(regexp_split_to_array(trim(q.query), '\\s+')) <= 5 THEN 'medium' ELSE 'long' END
                   || '|' || CASE WHEN b.query_id IS NULL THEN 'no-brand' ELSE 'brand' END AS slice
            FROM q LEFT JOIN branded b USING (query_id))
        SELECT query_id, query, slice FROM (
            SELECT *, row_number() OVER (PARTITION BY slice ORDER BY hash(query_id || 'ltr'))
                      / count(*) OVER (PARTITION BY slice) AS frac FROM sliced)
        ORDER BY frac, hash(query_id || 'ltr') LIMIT {args.train_queries}""").fetchall()
    DATA.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"query_id": [p[0] for p in picked], "text": [p[1] for p in picked],
                             "slice": [p[2] for p in picked]}), DATA / "train_queries.parquet")
    con.register("picked", pa.table({"query_id": [p[0] for p in picked]}))
    labels_ = con.execute("""
        SELECT query_id, product_id, CASE esci_label WHEN 'E' THEN 3 WHEN 'S' THEN 2 WHEN 'C' THEN 1 ELSE 0 END
        FROM examples JOIN picked USING (query_id) WHERE split = 'train'""").fetchall()
    pq.write_table(pa.table({"query_id": [r[0] for r in labels_], "doc_id": [r[1] for r in labels_],
                             "grade": pa.array([r[2] for r in labels_], pa.int64())}), DATA / "train_labels.parquet")
    print(f"  {len(picked):,} train queries, {len(labels_):,} labels")


def words(text):
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def all_queries():
    train = pq.read_table(DATA / "train_queries.parquet").to_pylist()
    with read_only() as con:
        test = sampled(con)
    return [("train", r["query_id"], r["text"]) for r in train] + [("test", q, t) for q, t in test]


def scores_by_id(table, key="score"):
    return dict(zip(table["id"].to_pylist(), table[key].to_pylist()))


def query_features(tp, judge, rubric, split, qid, text, ids, docs):
    """One row per candidate, in hybrid's order."""
    if not ids:
        return []
    only = ("id", "In", ids)
    lists = {
        "ann_dist": scores_by_id(tp.query(NAMESPACE, ann_rank_by(text), top_k=len(ids), filters=only)),
        "bm25": scores_by_id(tp.query(NAMESPACE, bm25_rank_by(text), top_k=len(ids), filters=only)),
        "title": scores_by_id(tp.query(NAMESPACE, ("title", "BM25", text), top_k=len(ids), filters=only)),
        "body": scores_by_id(tp.query(NAMESPACE, ("body", "BM25", text), top_k=len(ids), filters=only)),
        "voyage": scores_by_id(rerank(tp, NAMESPACE, "voyage/rerank-3", text, ids)),
    }
    texts = {d: docs[d]["text"] for d in ids if d in docs}
    grades = jev.grade_products(judge, text, texts, rubric) if texts else {}
    qwords, rows = words(text), []
    for rank, d in enumerate(ids, 1):
        row = {"split": split, "query_id": qid, "doc_id": d, "rrf_rank": float(rank)}
        for name, got in lists.items():
            row[name] = got.get(d)
            row[f"{name.split('_')[0]}_missing"] = float(d not in got)
        g = grades.get(d)
        probs = (g.probs if g else None) or {}
        for k in range(4):
            row[f"jev_p{k}"] = probs.get(k)
        row["jev_expected"] = g.expected if g else None
        info = docs.get(d, {})
        row["brand_match"] = float(bool(info.get("brand")) and info["brand"].lower() in text.lower())
        row["color_match"] = float(bool(info.get("color")) and info["color"].lower() in text.lower())
        row["title_overlap"] = len(qwords & words(info.get("title"))) / len(qwords) if qwords else 0.0
        rows.append(row)
    return rows


def run_features(args):
    folder = DATA / "features"
    done = part_ids(folder, "query_id")
    todo = [q for q in all_queries() if q[1] not in done]
    print(f"  {len(done):,} queries done, {len(todo):,} to go", flush=True)
    tp, judge, rubric = client(), jev.client(), load("esci")
    for i in range(0, len(todo), 100):
        chunk = todo[i:i + 100]

        def one(q):
            split, qid, text = q
            (hits,) = tp.multi_query(NAMESPACE, hybrid_queries(text, DEPTH), rerank_by=("RRF", {}), limit=DEPTH,
                                     run_id="ltr-candidates", query_id=qid)
            with read_only() as con:
                ids = hits["id"].to_pylist()
                texts = doc_texts(con, ids)
                con.register("_ids", pa.table({"doc_id": pa.array(ids, pa.string())}))
                info = {d: {"title": t, "brand": b, "color": c} for d, t, b, c in con.execute(
                    "SELECT doc_id, title, brand, color FROM docs JOIN _ids USING (doc_id) WHERE dataset = 'esci-us'"
                ).fetchall()}
            docs = {d: info.get(d, {}) | {"text": texts.get(d, "")} for d in ids}
            return query_features(tp, judge, rubric, split, qid, text, ids, docs)

        with ThreadPoolExecutor(4) as pool:
            rows = [r for part in pool.map(one, chunk) for r in part]
        write_part(folder, rows, SCHEMA)
        print(f"  {len(done) + i + len(chunk):,} queries", flush=True)


def load_features():
    t = duckdb.sql(f"SELECT * FROM '{DATA}/features/*.parquet' ORDER BY split, query_id, rrf_rank").arrow()
    return t.read_all() if hasattr(t, "read_all") else t


def matrix(rows, cols):
    import numpy as np
    return np.array([[np.nan if r[c] is None else r[c] for c in cols] for r in rows], dtype=float)


def groups_of(rows):
    sizes, last = [], None
    for r in rows:
        if r["query_id"] != last:
            sizes.append(0)
            last = r["query_id"]
        sizes[-1] += 1
    return sizes


def fit(rows, y, cols, params):
    """LightGBM's native LambdaRank (the sklearn wrapper would need scikit-learn too)."""
    import lightgbm as lgb
    data = lgb.Dataset(matrix(rows, cols), label=y, group=groups_of(rows), feature_name=cols)
    return lgb.train({"objective": "lambdarank", "metric": "ndcg", "verbosity": -1, **params}, data,
                     num_boost_round=300)


def cv_params(rows, y, cols):
    """3-fold cross-validation over train queries, by mean nDCG@10 of the held fold."""
    from relevancy.study import ndcg
    qids = sorted({r["query_id"] for r in rows})
    folds = [set(qids[i::3]) for i in range(3)]
    best, best_score = None, -1.0
    for values in itertools.product(*GRID.values()):
        params = dict(zip(GRID, values))
        scores = []
        for fold in folds:
            tr = [i for i, r in enumerate(rows) if r["query_id"] not in fold]
            te = [i for i, r in enumerate(rows) if r["query_id"] in fold]
            model = fit([rows[i] for i in tr], [y[i] for i in tr], cols, params)
            pred = model.predict(matrix([rows[i] for i in te], cols))
            by_q = defaultdict(list)
            for i, p in zip(te, pred):
                by_q[rows[i]["query_id"]].append((p, rows[i]["doc_id"], y[i]))
            for items in by_q.values():
                ranked = [d for _, d, _ in sorted(items, key=lambda x: -x[0])]
                v = ndcg(ranked, {d: g for _, d, g in items})
                if v is not None:
                    scores.append(v)
        mean_score = sum(scores) / len(scores) if scores else 0
        if mean_score > best_score:
            best, best_score = params, mean_score
    return best


def run_train(args):
    feats = load_features().to_pylist()
    train = [r for r in feats if r["split"] == "train"]
    test = [r for r in feats if r["split"] == "test"]
    human = defaultdict(dict)
    for r in pq.read_table(DATA / "train_labels.parquet").to_pylist():
        human[r["query_id"]][r["doc_id"]] = r["grade"]
    y_human = [human[r["query_id"]].get(r["doc_id"], 0) for r in train]
    # Jev's own grade for each candidate: the most likely of its four probabilities.
    y_jev = [max(range(4), key=lambda k: r[f"jev_p{k}"] or 0) for r in train]
    variants = {
        "ltr": (FEATURES, y_human),
        "ltr without jev": ([f for f in FEATURES if f not in GROUPS["jev"]], y_human),
        "ltr without voyage": ([f for f in FEATURES if f not in GROUPS["voyage"]], y_human),
        "ltr without both": (GROUPS["retrieval"] + GROUPS["match"], y_human),
        "ltr on jev labels": (FEATURES, y_jev),
    }
    out, importance = [], []
    for name, (cols, y) in variants.items():
        params = cv_params(train, y, cols)
        model = fit(train, y, cols, params)
        pred = model.predict(matrix(test, cols))
        by_q = defaultdict(list)
        for r, p in zip(test, pred):
            by_q[r["query_id"]].append((p, r["doc_id"]))
        out += [{"variant": name, "query_id": q, "ids": " ".join(d for _, d in sorted(v, key=lambda x: -x[0]))}
                for q, v in by_q.items()]
        if name == "ltr":
            importance = sorted(zip(cols, model.feature_importance("gain")), key=lambda x: -x[1])
        print(f"  {name:<20} params {json.dumps(params)}")
    voyage = defaultdict(list)
    for r in test:
        voyage[r["query_id"]].append((r["voyage"] if r["voyage"] is not None else -1e9, r["doc_id"]))
    out += [{"variant": "voyage top 100", "query_id": q, "ids": " ".join(d for _, d in sorted(v, key=lambda x: -x[0]))}
            for q, v in voyage.items()]
    write_csv(OUT / "ltr_runs.csv", out)
    write_csv(OUT / "importance.csv", [{"feature": f, "gain": round(float(g), 1)} for f, g in importance])


def run_report(args):
    with read_only() as con:
        human = labels(con)
    by = defaultdict(dict)
    for r in read_csv(OUT / "ltr_runs.csv"):
        by[r["variant"]][r["query_id"]] = r["ids"].split()
    base = by["voyage top 100"]
    print(f"## Against Voyage over the same top 100, {len(base)} test queries")
    for v in [x for x in by if x != "voyage top 100"]:
        print(f"  {v:<20} nDCG@10   {compare(per_query(by[v], human), per_query(base, human))}")
        print(f"  {'':<20} condensed {compare(per_query(by[v], human, 'condensed'), per_query(base, human, 'condensed'))}")
    top = read_csv(OUT / "importance.csv")[:8]
    print("  most important features (gain): " + ", ".join(f"{r['feature']} {r['gain']}" for r in top))


if __name__ == "__main__":
    commands = {"train-queries": run_train_queries, "features": run_features, "train": run_train, "report": run_report}
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=commands)
    ap.add_argument("--train-queries", type=int, default=5_000)
    args = ap.parse_args()
    commands[args.command](args)
