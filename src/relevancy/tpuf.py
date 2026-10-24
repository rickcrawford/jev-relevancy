"""turbopuffer client, namespace schema, and the retrieval configs we compare."""

import os
import time
from dataclasses import dataclass

from dotenv import load_dotenv
from turbopuffer import Turbopuffer

from relevancy.warehouse import ROOT

load_dotenv(ROOT / ".env")

# gcp-us-central1 is the default in a lot of turbopuffer examples, but the docs
# ask new workloads to go elsewhere. Model reranking with Jev is US-only.
REGION = os.environ.get("TURBOPUFFER_REGION", "aws-us-east-1")

# turbopuffer embeds the `text` attribute on write and the query string on
# search, so no separate embedding API key is needed.
EMBED_MODEL = os.environ.get("EMBED_MODEL", "voyage/voyage-4-lite")

TOP_K = 50  # candidates per query; every reranker sees the same 50


def client() -> Turbopuffer:
    if not os.environ.get("TURBOPUFFER_API_KEY"):
        raise SystemExit("TURBOPUFFER_API_KEY is not set. Copy .env.example to .env and add your key.")
    return Turbopuffer(region=REGION)


def namespace_name(dataset: str, scope: str) -> str:
    return f"{dataset}-{scope}"


def schema() -> dict:
    english = {"language": "english", "stemming": True}
    return {
        # Full-text search fields. Turning on FTS makes them non-filterable,
        # which is what we want: we never filter on free text.
        "title": {"type": "string", "full_text_search": english},
        "body": {"type": "string", "full_text_search": english},
        # Short text that gets embedded, and that model rerankers read.
        "text": {"type": "string", "embed": EMBED_MODEL, "filterable": False},
        # Plain attributes stay filterable by default.
        "brand": {"type": "string"},
        "color": {"type": "string"},
    }


def bm25_rank_by(q: str) -> tuple:
    """BM25 over title and body, with title counted twice."""
    return ("Sum", (("Product", 2, ("title", "BM25", q)), ("body", "BM25", q)))


def ann_rank_by(q: str) -> tuple:
    return ("text", "ANN", ("Embed", q))


@dataclass
class Hits:
    ids: list[str]
    scores: list[float]
    latency_ms: float        # wall clock, as the caller sees it
    server_ms: float | None
    billable_bytes: int | None


def _hits(resp, rows, started: float) -> Hits:
    perf, bill = resp.performance, resp.billing
    return Hits(
        ids=[str(r.id) for r in rows],
        scores=[float(getattr(r, "$dist")) for r in rows],
        latency_ms=(time.perf_counter() - started) * 1000,
        server_ms=getattr(perf, "server_total_ms", None),
        billable_bytes=getattr(bill, "billable_logical_bytes_queried", None),
    )


def search(ns, retriever: str, q: str, k: int = TOP_K) -> Hits:
    """Run one first-stage retriever: bm25, bm25-title, ann, or hybrid-rrf."""
    started = time.perf_counter()
    if retriever == "bm25":
        resp = ns.query(rank_by=bm25_rank_by(q), top_k=k)
        return _hits(resp, resp.rows or [], started)
    if retriever == "bm25-title":
        # Deliberately weak: titles only. Project 2 needs a spread of quality
        # to test whether a judge ranks systems the way humans do.
        resp = ns.query(rank_by=("title", "BM25", q), top_k=k)
        return _hits(resp, resp.rows or [], started)
    if retriever == "ann":
        resp = ns.query(rank_by=ann_rank_by(q), top_k=k)
        return _hits(resp, resp.rows or [], started)
    if retriever == "hybrid-rrf":
        # Both subqueries run on one snapshot and turbopuffer fuses them with
        # reciprocal rank fusion (k=60) on the server.
        resp = ns.multi_query(
            queries=[{"rank_by": ann_rank_by(q), "top_k": k},
                     {"rank_by": bm25_rank_by(q), "top_k": k}],
            rerank_by=("RRF", {}),
            limit=k,
        )
        return _hits(resp, resp.results[0].rows or [], started)
    raise ValueError(f"unknown retriever {retriever!r}")


def rerank(ns, model: str, q: str, ids: list[str]) -> Hits:
    """Rerank a fixed candidate set with one of turbopuffer's hosted models.

    Filtering on the candidate ids guarantees every reranker sees exactly the
    same documents, whatever produced them.
    """
    started = time.perf_counter()
    resp = ns.query(
        rank_by=("id", "asc"),
        filters=("id", "In", ids),
        top_k=len(ids),
        include_attributes=["text"],
        rerank_by={"model": model, "attribute": "text", "query": q},
    )
    return _hits(resp, resp.rows or [], started)
