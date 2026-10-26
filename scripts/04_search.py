"""Step 4: run the first-stage retrievers and record every result.

For each sampled query, pull the top 50 from turbopuffer with each retriever:
  bm25        BM25 over title (weighted 2x) and body
  bm25-title  BM25 over the title alone, a deliberately weak baseline
  ann         vector search over the embedded `text` attribute
  hybrid-rrf  both of the above, fused with reciprocal rank fusion on the server

Each retriever becomes one row in `runs` and 50 rows per query in `results`.
Every request also lands in `tpuf_requests`, tpuf-arrow's request log, and
each result row gets its query's wall time, server time, and billed bytes by
a join against it.

  uv run scripts/04_search.py --scope smoke
  uv run scripts/04_search.py --scope full --retriever hybrid-rrf
"""

import argparse
import json
import uuid

import pyarrow as pa

from relevancy.tpuf import EMBED_MODEL, REGION, TOP_K, client, namespace_name, search
from relevancy.warehouse import connect, git_sha, sample_id

RETRIEVERS = ["bm25", "bm25-title", "ann", "hybrid-rrf"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="esci-us")
    ap.add_argument("--scope", choices=["smoke", "full"], default="smoke")
    ap.add_argument("--retriever", action="append", choices=RETRIEVERS,
                    help="repeatable; default runs all four")
    args = ap.parse_args()

    with connect() as con:
        queries = con.execute(
            "SELECT query_id, text FROM queries WHERE dataset = ? AND sampled ORDER BY query_id",
            [args.dataset]).fetchall()
        sample = sample_id(con, args.dataset)
    if not queries:
        raise SystemExit("No sampled queries. Run scripts/02_prepare_esci.py first.")

    ns_name = namespace_name(args.dataset, args.scope)
    tp = client()

    # One untimed query first, so the first retriever doesn't pay for a cold
    # cache, or for tpuf-arrow fetching the namespace schema.
    search(tp, ns_name, "bm25", queries[0][1], k=1, run_id="warmup")

    for retriever in args.retriever or RETRIEVERS:
        run_id = f"{retriever}-{args.scope}-{uuid.uuid4().hex[:6]}"
        config = {"namespace": ns_name, "region": REGION, "top_k": TOP_K, "sample": sample,
                  "embed_model": EMBED_MODEL, "bm25": "title x2 + body"}

        tables = []
        for query_id, text in queries:
            hits = search(tp, ns_name, retriever, text, run_id=run_id, query_id=query_id)
            tables.append(hits.append_column("query_id", pa.array([query_id] * hits.num_rows, pa.string())))
        hits = pa.concat_tables(tables)

        # Write once at the end, so the warehouse isn't locked while we wait on the network.
        with connect() as con:
            con.begin()
            con.execute("INSERT INTO runs (run_id, dataset, retriever, reranker, config, git_sha) "
                        "VALUES (?, ?, ?, 'none', ?, ?)",
                        [run_id, args.dataset, retriever, json.dumps(config), git_sha()])
            tp.save_log(con)
            con.register("hits", hits)
            con.execute("""
                INSERT INTO results (run_id, query_id, doc_id, rank, score, latency_ms, server_ms, billable_bytes)
                SELECT l.run_id, h.query_id, h.id, h.rank, h.score, l.wall_ms, l.server_ms, l.billable_bytes_queried
                FROM hits h JOIN tpuf_requests l ON l.run_id = ? AND l.query_id = h.query_id
            """, [run_id])
            (p50,) = con.execute("SELECT median(wall_ms) FROM tpuf_requests WHERE run_id = ?", [run_id]).fetchone()
            con.commit()
        print(f"{run_id:<28} {len(queries)} queries, p50 {p50:.0f} ms")

    flag = "" if args.dataset == "esci-us" else f" --dataset {args.dataset}"
    print(f"\nNext: uv run scripts/05_score.py{flag}")


if __name__ == "__main__":
    main()
