"""Step 6: rerank a first-stage run, then score it again.

Takes the top 50 from an earlier run (hybrid-rrf by default) and has a model
reorder exactly those 50 for every query. Holding the candidates fixed means
any change in nDCG comes from the reranker. Recall@50 can't move, because the
set of 50 doesn't change.

Three kinds of reranker, picked by the --model prefix:

  voyage/rerank-3, cohere/rerank-v3.5, ...   hosted by turbopuffer (`rerank_by`)
  jev:jev-1.13.0                             Jev via TypeSafe's API (TYPESAFE_API_KEY)
  llm:openai/gpt-4o-mini                     any OpenRouter model, read Jev style
                                             through logprobs (OPENROUTER_API_KEY)

  uv run scripts/06_rerank.py --model voyage/rerank-3
  uv run scripts/06_rerank.py --model jev:jev-1.13.0
  uv run scripts/06_rerank.py --model typesafe/jev-1.13.0   # Jev inside turbopuffer, if enabled
  uv run scripts/05_score.py

Jev and LLM rerankers grade each candidate against the dataset's rubric
(rubrics/<name>.toml), sort by expected grade (sum of grade x probability) and
break ties with the original rank. Latency recorded for a reranked run (wall
clock and server) is end to end: retrieval plus reranking. Jev and LLM calls
aren't turbopuffer queries, so their server time is the base run's only.
"""

import argparse
import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

from relevancy import jev, llm_judge
from relevancy.judges import grade_pairs
from relevancy.rubric import for_dataset
from relevancy.tpuf import client, rerank
from relevancy.warehouse import connect, git_sha, sample_id


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True,
                    help="voyage/rerank-3, jev:jev-1.13.0, llm:openai/gpt-4o-mini, ...")
    ap.add_argument("--base-run", help="run_id to rerank; default is the latest hybrid-rrf run")
    ap.add_argument("--dataset", default="esci-us")
    ap.add_argument("--workers", type=int, default=1,
                    help="queries reranked in parallel. Above 1, recorded latency includes "
                         "queueing and isn't comparable with step 4, which runs one query at a time")
    args = ap.parse_args()

    with connect() as con:
        if args.base_run:
            base = con.execute("SELECT run_id, retriever, config FROM runs WHERE run_id = ?",
                               [args.base_run]).fetchone()
            if not base:
                raise SystemExit(f"No run called {args.base_run}.")
        else:
            # The latest plain hybrid run on the current query sample.
            base = con.execute("""
                SELECT run_id, retriever, config FROM runs
                WHERE dataset = ? AND reranker = 'none' AND retriever = 'hybrid-rrf'
                  AND config->>'sample' = ?
                ORDER BY started_at DESC LIMIT 1
            """, [args.dataset, sample_id(con, args.dataset)]).fetchone()
            if not base:
                raise SystemExit("No hybrid-rrf run on the current query sample. "
                                 "Run scripts/04_search.py first.")
        base_id, retriever, base_config = base
        rubric = for_dataset(con, args.dataset)

        # Per query: the candidate ids in rank order, their text, and the base timings.
        candidates = con.execute("""
            SELECT r.query_id, q.text,
                   list(r.doc_id ORDER BY r.rank),
                   list(concat_ws('. ', d.title, d.brand, d.color, left(d.body, 800)) ORDER BY r.rank),
                   any_value(r.latency_ms), any_value(r.server_ms)
            FROM results r
            JOIN queries q ON q.dataset = ? AND q.query_id = r.query_id
            JOIN docs d ON d.dataset = q.dataset AND d.doc_id = r.doc_id
            WHERE r.run_id = ?
            GROUP BY ALL ORDER BY r.query_id
        """, [args.dataset, base_id]).fetchall()

    if not candidates:
        raise SystemExit(f"{base_id} has no results to rerank.")
    config = json.loads(base_config) | {"base_run": base_id, "rerank_model": args.model}
    kind, _, name = args.model.partition(":") if ":" in args.model else ("tpuf", "", args.model)
    if kind == "tpuf":
        ns = client().namespace(config["namespace"])
    elif kind == "jev":
        jev.client()  # fail fast if TYPESAFE_API_KEY is missing
        judge = f"jev-batched:{name}"  # all 50 candidates in one request per query
        config["rubric"] = rubric.name
    elif kind == "llm":
        llm_judge.client()  # fail fast if OPENROUTER_API_KEY is missing
        judge = f"llm:{name}"
        config["rubric"] = rubric.name
    else:
        raise SystemExit(f"unknown reranker prefix {kind!r}; use jev: or llm:, or a turbopuffer model name")

    def rerank_one(cand):
        query_id, text, ids, texts, *_ = cand
        if kind == "tpuf":
            hits = rerank(ns, name, text, ids)
            if len(hits.ids) != len(ids):
                # The filter only finds ids that are still in the namespace.
                print(f"  warning: query {query_id}: reranker returned {len(hits.ids)} of {len(ids)} candidates")
            return query_id, list(zip(hits.ids, hits.scores)), hits.latency_ms, hits.server_ms

        # Jev grades all candidates in one request; an LLM takes one call per
        # candidate, 10 at a time. A failed call comes back ungraded, not as a crash.
        started = time.perf_counter()
        pairs = [(query_id, doc_id, text, t) for doc_id, t in zip(ids, texts)]
        grades = dict(zip(ids, grade_pairs(judge, pairs, rubric, workers=10)))
        elapsed = (time.perf_counter() - started) * 1000
        failed = sum(grades[i].expected is None for i in ids)
        if failed:
            print(f"  warning: query {query_id}: {failed} of {len(ids)} candidates got no grade; "
                  "they keep their place after the graded ones")
        # Highest expected grade first. Ungraded candidates sort last, in their original order.
        order = sorted(range(len(ids)), key=lambda i: (grades[ids[i]].expected is None,
                                                         -(grades[ids[i]].expected or 0), i))
        return query_id, [(ids[i], grades[ids[i]].expected) for i in order], elapsed, None

    base_timing = {c[0]: (c[4], c[5]) for c in candidates}
    with ThreadPoolExecutor(args.workers) as pool:
        reranked = list(pool.map(rerank_one, candidates))

    run_id = f"{name.split('/')[-1]}-{uuid.uuid4().hex[:6]}"
    rows, added = [], []
    for query_id, ranked, ms, server_ms in reranked:
        base_ms, base_server_ms = base_timing[query_id]
        added.append(ms)
        server = None if base_server_ms is None else base_server_ms + (server_ms or 0)
        rows += [(run_id, query_id, doc_id, rank, score, base_ms + ms, server, None)
                 for rank, (doc_id, score) in enumerate(ranked, 1)]

    with connect() as con:
        con.begin()
        con.execute("INSERT INTO runs (run_id, dataset, retriever, reranker, config, git_sha) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    [run_id, args.dataset, retriever, args.model, json.dumps(config), git_sha()])
        con.executemany("INSERT INTO results VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)
        con.commit()

    added.sort()
    print(f"{run_id}: reranked {len(candidates)} queries from {base_id}, "
          f"added p50 {added[len(added) // 2]:.0f} ms per query")
    print("\nNext: uv run scripts/05_score.py")


if __name__ == "__main__":
    main()
