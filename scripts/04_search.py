"""Step 4: run the first-stage retrievers and record every result.

For each sampled query, pull the top 50 from turbopuffer with each retriever:
  bm25        BM25 over title (weighted 2x) and body
  bm25-title  BM25 over the title alone, a deliberately weak baseline
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
    ns = client().namespace(ns_name)

    # One untimed query first, so the first retriever doesn't pay for a cold cache.
    search(ns, "bm25", queries[0][1], k=1)

    for retriever in args.retriever or RETRIEVERS:
        run_id = f"{retriever}-{args.scope}-{uuid.uuid4().hex[:6]}"
        config = {"namespace": ns_name, "region": REGION, "top_k": TOP_K, "sample": sample,
                  "embed_model": EMBED_MODEL, "bm25": "title x2 + body"}

        rows, latencies = [], []
        for query_id, text in queries:
            hits = search(ns, retriever, text)
            latencies.append(hits.latency_ms)
            rows += [(run_id, query_id, doc_id, rank, score, hits.latency_ms,
                      hits.server_ms, hits.billable_bytes)
                     for rank, (doc_id, score) in enumerate(zip(hits.ids, hits.scores), 1)]

        # Write once at the end, so the warehouse isn't locked while we wait on the network.
        with connect() as con:
            con.begin()
            con.execute("INSERT INTO runs (run_id, dataset, retriever, reranker, config, git_sha) "
                        "VALUES (?, ?, ?, 'none', ?, ?)",
                        [run_id, args.dataset, retriever, json.dumps(config), git_sha()])
            con.executemany("INSERT INTO results VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)
            con.commit()
        latencies.sort()
        print(f"{run_id:<28} {len(queries)} queries, p50 {latencies[len(latencies) // 2]:.0f} ms")

    flag = "" if args.dataset == "esci-us" else f" --dataset {args.dataset}"
    print(f"\nNext: uv run scripts/05_score.py{flag}")


if __name__ == "__main__":
    main()
