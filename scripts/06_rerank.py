"""Step 6: rerank a first-stage run with a hosted model, then score it again.

Takes the top 50 from an earlier run (hybrid-rrf by default) and asks a
turbopuffer-hosted reranker to reorder exactly those 50 for every query.
Holding the candidates fixed means any change in nDCG comes from the reranker.
Recall@50 can't move, because the set of 50 doesn't change.

  uv run scripts/06_rerank.py --model voyage/rerank-3
  uv run scripts/06_rerank.py --model typesafe/jev-1.13.0   # once Jev is enabled for your org
  uv run scripts/05_score.py

Run scripts/00_check_setup.py to see which models your org can call.
Latency recorded for a reranked run (wall clock and server) is end to end:
retrieval plus reranking.
"""

import argparse
import json
import uuid

from relevancy.tpuf import client, rerank
from relevancy.warehouse import connect, git_sha


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="e.g. voyage/rerank-3, typesafe/jev-1.13.0")
    ap.add_argument("--base-run", help="run_id to rerank; default is the latest hybrid-rrf run")
    ap.add_argument("--dataset", default="esci-us")
    args = ap.parse_args()

    con = connect()
    base = con.execute("""
        SELECT run_id, retriever, config FROM runs
        WHERE dataset = ? AND reranker = 'none'
          AND (run_id = ? OR (CAST(? AS VARCHAR) IS NULL AND retriever = 'hybrid-rrf'))
        ORDER BY started_at DESC LIMIT 1
    """, [args.dataset, args.base_run, args.base_run]).fetchone()
    if not base:
        raise SystemExit("No base run found. Run scripts/04_search.py first.")
    base_id, retriever, base_config = base
    config = json.loads(base_config) | {"base_run": base_id, "rerank_model": args.model}
    ns = client().namespace(config["namespace"])

    candidates = con.execute("""
        SELECT r.query_id, q.text, list(r.doc_id ORDER BY r.rank),
               any_value(r.latency_ms), any_value(r.server_ms)
        FROM results r JOIN queries q ON q.dataset = ? AND q.query_id = r.query_id
        WHERE r.run_id = ?
        GROUP BY ALL ORDER BY r.query_id
    """, [args.dataset, base_id]).fetchall()

    run_id = f"{args.model.split('/')[-1]}-{uuid.uuid4().hex[:6]}"
    con.execute("INSERT INTO runs (run_id, dataset, retriever, reranker, config, git_sha) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                [run_id, args.dataset, retriever, args.model, json.dumps(config), git_sha()])

    added = []
    for query_id, text, ids, base_ms, base_server_ms in candidates:
        hits = rerank(ns, args.model, text, ids)
        added.append(hits.latency_ms)
        con.executemany(
            "INSERT INTO results VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [(run_id, query_id, doc_id, rank, score, base_ms + hits.latency_ms,
              (base_server_ms or 0) + (hits.server_ms or 0), hits.billable_bytes)
             for rank, (doc_id, score) in enumerate(zip(hits.ids, hits.scores), 1)],
        )
    added.sort()
    print(f"{run_id}: reranked {len(candidates)} queries from {base_id}, "
          f"added p50 {added[len(added) // 2]:.0f} ms per query")
    print("\nNext: uv run scripts/05_score.py")


if __name__ == "__main__":
    main()
