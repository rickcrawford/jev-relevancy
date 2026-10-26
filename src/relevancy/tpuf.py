"""turbopuffer client, namespace schema, and the retrieval configs we compare.

Every request goes through tpuf-arrow (https://github.com/rickcrawford/tpuf-arrow),
which returns hits as Arrow tables and logs each request with its server time
and billed bytes. Scripts tag requests with a run_id and query_id, then read
those numbers back from the log, or save it to the warehouse's tpuf_requests
table.
"""

import os

import pyarrow as pa
import pyarrow.compute as pc
import tpuf_arrow as ta
from dotenv import load_dotenv

from relevancy.warehouse import ROOT

# The judges read their keys from the environment, so load all of .env.
load_dotenv(ROOT / ".env")

# gcp-us-central1 is the default in a lot of turbopuffer examples, but the docs
# ask new workloads to go elsewhere. Model reranking with Jev is US-only.
REGION = os.environ.get("TURBOPUFFER_REGION", "aws-us-east-1")

# turbopuffer embeds the `text` attribute on write and the query string on
# search, so no separate embedding API key is needed.
EMBED_MODEL = os.environ.get("EMBED_MODEL", "voyage/voyage-4-lite")

TOP_K = 50  # candidates per query; every reranker sees the same 50


def client(max_concurrency: int = 8) -> ta.Client:
    try:
        return ta.connect(region=REGION, max_concurrency=max_concurrency)
    except ta.ConfigError:
        raise SystemExit("TURBOPUFFER_API_KEY is not set. Copy .env.example to .env and add your key.") from None


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


RANK_BY = {
    "bm25": bm25_rank_by,
    # Deliberately weak: titles only. Step 8 needs a spread of quality to
    # test whether a judge ranks systems the way humans do.
    "bm25-title": lambda q: ("title", "BM25", q),
    "ann": ann_rank_by,
}


def hybrid_queries(q: str, k: int = TOP_K, **extra) -> list[dict]:
    """The two subqueries hybrid-rrf fuses: vector search, then BM25."""
    return [{"rank_by": ann_rank_by(q), "top_k": k, **extra}, {"rank_by": bm25_rank_by(q), "top_k": k, **extra}]


def search(tp: ta.Client, namespace: str, retriever: str, q: str, k: int = TOP_K,
           *, run_id: str | None = None, query_id: str | None = None) -> pa.Table:
    """Run one first-stage retriever (bm25, bm25-title, ann, or hybrid-rrf) and return id, rank, score."""
    tags = {"run_id": run_id, "query_id": query_id}
    if retriever == "hybrid-rrf":
        # Both subqueries run on one snapshot and turbopuffer fuses them with
        # reciprocal rank fusion (k=60) on the server.
        (hits,) = tp.multi_query(namespace, hybrid_queries(q, k), rerank_by=("RRF", {}), limit=k, **tags)
        return hits
    if retriever not in RANK_BY:
        raise ValueError(f"unknown retriever {retriever!r}")
    return tp.query(namespace, RANK_BY[retriever](q), top_k=k, **tags)


def rerank(tp: ta.Client, namespace: str, model: str, q: str, ids: list[str],
           *, run_id: str | None = None, query_id: str | None = None) -> pa.Table:
    """Rerank a fixed candidate set with one of turbopuffer's hosted models.

    Filtering on the candidate ids guarantees every reranker sees exactly the
    same documents, whatever produced them.
    """
    return tp.query(namespace, ("id", "asc"), top_k=len(ids), filters=("id", "In", ids),
                    rerank_by={"model": model, "attribute": "text", "query": q},
                    run_id=run_id, query_id=query_id)


def requests_by_query(tp: ta.Client, run_id: str) -> dict[str, dict]:
    """One run's rows from the client's request log, by query_id.

    Each row has wall_ms, server_ms, billable_bytes_queried, cache details, and
    the rest of tpuf-arrow's log columns.
    """
    log = tp.log
    return {r["query_id"]: r for r in log.filter(pc.equal(log["run_id"], run_id)).to_pylist()}
