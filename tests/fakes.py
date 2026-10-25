"""Offline stand-ins for turbopuffer, Jev, and OpenRouter, plus a synthetic ESCI warehouse.

The studies can't run here without API keys and a warehouse, so the tests run
every step, paid ones included, against these. The fakes rank deterministically
(a hash of the query and the document), honor the filters the studies use, and
log requests the way tpuf-arrow does, so the scripts' own logic is what gets
tested. They say nothing about how well anything works on real data.
"""

import hashlib
import json
import random
from types import SimpleNamespace

import pyarrow as pa

from relevancy.warehouse import connect

DS, NS = "esci-us", "esci-us-full"
WORDS = ["red", "shoe", "running", "trail", "kid", "replacement", "filter", "coffee", "maker", "steel", "pan", "cast", "iron", "wood", "battery", "charger", "phone", "case", "usb", "cable", "lamp", "desk", "chair", "toy", "blue", "green", "pack"]


def h(*parts) -> float:
    """A stable number in [0, 1) from any values."""
    return int(hashlib.md5("|".join(map(str, parts)).encode()).hexdigest()[:12], 16) / 16 ** 12


# -- the warehouse ------------------------------------------------------------

def build_warehouse(path, n_queries=60, n_sampled=40, n_docs=600, seed=1):
    """A small ESCI-shaped warehouse: queries, products, human labels, and the base runs."""
    rnd = random.Random(seed)
    con = connect(path)
    con.execute("INSERT INTO datasets VALUES (?, 'esci', 'synthetic')", [DS])
    docs = [f"P{i:04d}" for i in range(n_docs)]
    con.executemany("INSERT INTO docs VALUES (?, ?, ?, ?, ?, ?)", [
        (DS, d, " ".join(rnd.sample(WORDS, 4)), " ".join(rnd.sample(WORDS, 8)), rnd.choice(["Acme", "Tascam", "Zed"]),
         rnd.choice(["red", "blue"])) for d in docs])
    queries, qrows, jrows = [], [], []
    for q in range(n_queries):
        qid = f"{1000 + q}"
        text = " ".join(rnd.sample(WORDS, 3)) + (" without wood" if q % 7 == 0 else "")
        if q % 5 == 0:
            text += " replacement"
        queries.append((qid, text))
        qrows.append((DS, qid, text, "test", rnd.choice(["short|brand", "short|nobrand", "medium|nobrand"]), q < n_sampled))
        jrows += [(DS, qid, d, "human", rnd.choice([0, 1, 2, 3, 3]), None, None) for d in rnd.sample(docs, 15)]
    labeled = {}
    for _, qid, d, *_ in jrows:
        labeled.setdefault(qid, []).append(d)
    sampled = queries[:n_sampled]
    config = json.dumps({"namespace": NS, "sample": "x"})
    runs, rrows = [], []
    for run_id, retriever, reranker in [("hyb", "hybrid-rrf", "none"), ("bm25", "bm25", "none"), ("ann", "ann", "none"),
                                        ("title", "bm25-title", "none"), ("voy", "hybrid-rrf", "voyage/rerank-3"),
                                        ("jev", "hybrid-rrf", "jev:jev-1.13.0"),
                                        ("coh", "hybrid-rrf", "cohere/rerank-v3.5")]:
        runs.append((run_id, DS, retriever, reranker, config))
        for qid, _ in sampled:
            pool = sorted(set(labeled[qid]) | set(rnd.sample(docs, 60)), key=lambda d, r=run_id, q=qid: h(r, q, d))[:50]
            rrows += [(run_id, qid, d, rank, 1 - rank / 100 if reranker != "jev:jev-1.13.0" else 3 * h(qid, d),
                       100.0, 50.0, 0) for rank, d in enumerate(pool, 1)]
    jev_rows = {(qid, d) for run_id, qid, d, rank, *_ in rrows if rank <= 10}
    jrows += [(DS, qid, d, "jev-batched:jev-1.13.0", int(h("g", qid, d) * 4), 0.6, 3 * h("e", qid, d))
              for qid, d in sorted(jev_rows)]

    def insert(table, cols, rows):
        con.register("_rows", pa.table({c: [r[i] for r in rows] for i, c in enumerate(cols)}))
        con.execute(f"INSERT INTO {table} ({', '.join(cols)}) SELECT * FROM _rows")
        con.unregister("_rows")

    insert("queries", ["dataset", "query_id", "text", "split", "slice", "sampled"], qrows)
    insert("judgments", ["dataset", "query_id", "doc_id", "source", "grade", "prob", "expected"], jrows)
    insert("runs", ["run_id", "dataset", "retriever", "reranker", "config"], runs)
    insert("results", ["run_id", "query_id", "doc_id", "rank", "score", "latency_ms", "server_ms", "billable_bytes"], rrows)
    con.close()
    return queries, docs


# -- turbopuffer --------------------------------------------------------------

LOG_FIELDS = ["run_id", "query_id", "wall_ms", "server_ms", "query_execution_ms", "embedding_ms",
              "billable_bytes_queried", "exhaustive_search_count", "embedding_tokens"]


class FakeNamespace:
    def __init__(self, tp, name):
        self.tp, self.name = tp, name

    def write(self, patch_columns=None, upsert_columns=None, schema=None, **_):
        cols = patch_columns or upsert_columns
        attrs = self.tp.attrs.setdefault(self.name, {})
        for i, doc in enumerate(cols["id"]):
            attrs.setdefault(doc, {}).update({k: v[i] for k, v in cols.items() if k != "id"})
        return SimpleNamespace(rows_affected=len(cols["id"]))


class FakeTp:
    """Enough of tpuf_arrow.Client for the studies."""

    def __init__(self, docs):
        self.docs = list(docs)
        self.attrs: dict[str, dict[str, dict]] = {}
        self.rows: list[dict] = []
        self.tpuf = SimpleNamespace(namespace=lambda name: FakeNamespace(self, name))

    @property
    def log(self):
        return pa.table({f: [r.get(f) for r in self.rows] for f in LOG_FIELDS})

    def _log(self, run_id, query_id):
        self.rows.append({"run_id": run_id, "query_id": query_id, "wall_ms": 100.0, "server_ms": 60.0,
                          "query_execution_ms": 20.0, "embedding_ms": 40.0, "billable_bytes_queried": 1000,
                          "exhaustive_search_count": 0, "embedding_tokens": 5})

    def _match(self, ns, doc, filters):
        if filters is None:
            return True
        if isinstance(filters[0], str) and filters[0] in ("And", "Or"):
            parts = [self._match(ns, doc, f) for f in filters[1]]
            return all(parts) if filters[0] == "And" else any(parts)
        attr, op, value = filters
        got = doc if attr == "id" else self.attrs.get(ns, {}).get(doc, {}).get(attr)
        return {"In": lambda: got in value, "NotIn": lambda: got not in value, "Eq": lambda: got == value, "NotEq": lambda: got != value,
                "Contains": lambda: value in (got or []), "Gte": lambda: got is not None and got >= value,
                }[op]()

    def _rank(self, ns, rank_by, filters, k, salt=""):
        pool = list(self.attrs[ns]) if ns in self.attrs and not ns.startswith("esci") else self.docs
        hits = [d for d in pool if self._match(ns, d, filters)]
        key = json.dumps(rank_by, default=str)
        if "BM25" in key:
            # Real BM25 only returns documents that share a term with the query.
            hits = [d for d in hits if h("terms", key, d) < 0.6]
        hits.sort(key=lambda d: h(key + salt, d))
        hits = hits[:k]
        n = len(hits) + 1
        # turbopuffer's score is $dist: a distance for ANN (lower is better), a BM25 score otherwise.
        ann = isinstance(rank_by, (list, tuple)) and len(rank_by) > 1 and rank_by[1] == "ANN"
        scores = [i / n for i in range(1, n)] if ann else [1 - i / n for i in range(len(hits))]
        return pa.table({"id": pa.array(hits, pa.string()), "rank": pa.array(range(1, len(hits) + 1), pa.int64()),
                         "score": pa.array(scores, pa.float64())})

    def query(self, ns, rank_by, top_k=10, filters=None, rerank_by=None, include=None, run_id=None, query_id=None):
        self._log(run_id, query_id)
        return self._rank(ns, rank_by, filters, top_k, json.dumps(rerank_by, default=str))

    def multi_query(self, ns, queries, rerank_by=None, limit=None, include=None, run_id=None, query_id=None):
        # The same checks tpuf-arrow makes before sending (client.py: max_concurrency, limit without rerank_by).
        if len(queries) > 8:
            raise ValueError(f"multi_query got {len(queries)} queries; the client allows at most 8")
        if limit is not None and rerank_by is None:
            raise ValueError("limit only applies with rerank_by")
        self._log(run_id, query_id)
        tables = [self._rank(ns, q["rank_by"], q.get("filters"), q.get("top_k", 10)) for q in queries]
        if rerank_by is None:
            return tables  # one table per subquery, unfused
        if isinstance(rerank_by, dict):  # a hosted model over the union
            ids = sorted({d for t in tables for d in t["id"].to_pylist()}, key=lambda d: h(rerank_by.get("model"), d))
            ids = ids[:limit]
            return [pa.table({"id": pa.array(ids, pa.string()), "rank": pa.array(range(1, len(ids) + 1), pa.int64()),
                              "score": pa.array([1 - i / (len(ids) + 1) for i in range(len(ids))], pa.float64())})]
        params = rerank_by[1] if len(rerank_by) > 1 else {}
        weights = params.get("weights") or [1.0] * len(tables)
        assert len(weights) == len(tables), "RRF needs one weight per subquery"
        k = params.get("rank_constant", 60)
        fused: dict[str, float] = {}
        for w, t in zip(weights, tables):
            for i, d in enumerate(t["id"].to_pylist()):
                fused[d] = fused.get(d, 0) + w / (k + i + 1)
        ids = sorted(fused, key=lambda d: (fused[d], d), reverse=True)[:limit]
        return [pa.table({"id": pa.array(ids, pa.string()), "rank": pa.array(range(1, len(ids) + 1), pa.int64()),
                          "score": pa.array([fused[d] for d in ids], pa.float64())})]

    def upsert(self, ns, data, schema=None, distance_metric=None, run_id=None, **_):
        table = data if isinstance(data, pa.Table) else data.read_all()
        cols = table.to_pydict()
        FakeNamespace(self, ns).write(upsert_columns=cols)
        return SimpleNamespace(rows=table.num_rows, billable_bytes=0)

    def scan(self, ns, columns=None, run_id=None, **_):
        ids = self.docs if ns.startswith("esci") else list(self.attrs.get(ns, {}))
        return SimpleNamespace(read_all=lambda: pa.table({"id": pa.array(ids, pa.string())}))


# -- Jev ----------------------------------------------------------------------

class FakeJev:
    """TypeSafeClient.system_one, answering Score, Choice, and Noul questions."""

    def __init__(self):
        self.requests = 0

    def system_one(self, state, questions, model=None):
        self.requests += 1
        assert len(questions) <= 50, "Jev takes at most 50 questions a request"
        answers = {}
        for key, q in questions.items():
            d = q.model_dump() if hasattr(q, "model_dump") else dict(q)
            seed = h(state, d.get("instructions"))
            if d["type"] == "score":
                n = len(d["criteria"])
                w = [h(seed, i) + 0.05 for i in range(n)]
                probs = {i: x / sum(w) for i, x in enumerate(w)}  # the SDK keys Score probabilities by int
                answers[key] = {"type": "score", "probabilities": probs,
                                "score": sum(i * p for i, p in enumerate(probs.values()))}
            elif d["type"] == "choice":
                names = list(d["criteria"])
                w = [h(seed, c) + 0.05 for c in names]
                probs = {c: x / sum(w) for c, x in zip(names, w)}
                best = max(probs, key=probs.get)
                answers[key] = {"type": "choice", "choice": best, "confidence": probs[best], "probabilities": probs}
            else:
                answers[key] = {"type": "noul", "noul": seed}
        return SimpleNamespace(model_dump=lambda: {"answers": answers, "model": "fake-jev"})


# -- OpenRouter ---------------------------------------------------------------

def fake_complete(http, model, system, user, max_tokens=400, json_mode=False):
    """Plausible JSON for each prompt the studies send."""
    s = system.lower()
    if "constraint" in s:
        return json.dumps({"constraints": [{"type": "brand", "value": "Acme"}, {"type": "excludes", "value": "wood"}]})
    if "rephras" in s:
        return json.dumps({"rewrites": [f"{user} variant {i}" for i in range(3)]})
    if "would type" in s or "expansion" in s:
        return json.dumps({"queries": [f"query {i} for {user[:20]}" for i in range(5)]})
    if "yes or no" in s or "label" in s:
        return json.dumps({"answer": h(model, user) > 0.6})
    if "relevance" in s or "rate" in s:
        return json.dumps({"score": round(h(model, user) * 3, 2)})
    return "ok"
