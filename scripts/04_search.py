"""Step 4: run the first-stage retrievers and record every result.

For each sampled query, pull the top 50 from turbopuffer with each retriever:
  bm25        BM25 over title (weighted 2x) and body
  ann         vector search over the embedded `text` attribute
  hybrid-rrf  both of the above, fused with reciprocal rank fusion on the server

Each retriever becomes one row in `runs` and 50 rows per query in `results`,
with the latency and billable bytes turbopuffer reports.

  uv run scripts/04_search.py --scope smoke
  uv run scripts/04_search.py --scope full --retriever hybrid-rrf
"""

import argparse
import json
import uuid

from relevancy.tpuf import EMBED_MODEL, REGION, TOP_K, client, namespace_name, search
from relevancy.warehouse import connect, git_sha

RETRIEVERS = ["bm25", "ann", "hybrid-rrf"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="esci-us")
    ap.add_argument("--scope", choices=["smoke", "full"], default="smoke")
    ap.add_argument("--retriever", action="append", choices=RETRIEVERS,
                    help="repeatable; default runs all three")
    args = ap.parse_args()

    con = connect()
    queries = con.execute(
        "SELECT query_id, text FROM queries WHERE dataset = ? AND sampled ORDER BY query_id",
        [args.dataset]).fetchall()
    if not queries:
        raise SystemExit("No sampled queries. Run scripts/02_prepare_esci.py first.")

    ns_name = namespace_name(args.dataset, args.scope)
    ns = client().namespace(ns_name)

    # One untimed query first, so the first retriever doesn't pay for a cold cache.
    search(ns, "bm25", queries[0][1], k=1)

    for retriever in args.retriever or RETRIEVERS:
        run_id = f"{retriever}-{args.scope}-{uuid.uuid4().hex[:6]}"
        config = {"namespace": ns_name, "region": REGION, "top_k": TOP_K,
                  "embed_model": EMBED_MODEL, "bm25": "title x2 + body"}
        con.execute("INSERT INTO runs (run_id, dataset, retriever, reranker, config, git_sha) "
                    "VALUES (?, ?, ?, 'none', ?, ?)",
                    [run_id, args.dataset, retriever, json.dumps(config), git_sha()])

        latencies = []
        for query_id, text in queries:
            hits = search(ns, retriever, text)
            latencies.append(hits.latency_ms)
            con.executemany(
                "INSERT INTO results VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [(run_id, query_id, doc_id, rank, score, hits.latency_ms,
                  hits.server_ms, hits.billable_bytes)
                 for rank, (doc_id, score) in enumerate(zip(hits.ids, hits.scores), 1)],
            )
        latencies.sort()
        print(f"{run_id:<28} {len(queries)} queries, p50 {latencies[len(latencies) // 2]:.0f} ms")

    print("\nNext: uv run scripts/05_score.py")


if __name__ == "__main__":
    main()
