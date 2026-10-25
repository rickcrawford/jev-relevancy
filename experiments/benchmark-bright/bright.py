"""BRIGHT: does Jev, as a cheap first pass, keep most of a reasoning reranker's gain?

Three domains fixed in advance: biology, economics, leetcode.

  load     Paid (turbopuffer writes, no embedding). Each domain's documents into
           a full-text namespace, bright-<domain>, and its queries to data/.
  bm25     Paid. BM25's top 100 per query, leaving out the query's excluded ids.
  jev      Paid (Jev). A Score for each of the top 100.
  llm      Paid (LLM). The reasoning model (REASONING_MODEL) scores each of the
           top 100 from 0 to 3. The costly step: 100 calls a query.
  report   Free. nDCG@10 per domain for BM25, Jev, the LLM, and the cascade
           (Jev keeps 20, the LLM orders them; computed from the two score sets,
           so it needs no extra calls), the share of the LLM's gain the cascade
           keeps, and LLM calls per query.

  uv run experiments/benchmark-bright/bright.py load
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow as pa

from relevancy import jev, llm_judge
from relevancy.rubric import Rubric
from relevancy.study import ndcg, write_csv
from relevancy.tpuf import client
from relevancy.warehouse import ROOT

OUT = Path(__file__).resolve().parent / "results"
DATA = ROOT / "data" / "bright"
DOMAINS = ("biology", "economics", "leetcode")
DEPTH, CASCADE = 100, 20
MODEL = os.environ.get("REASONING_MODEL", "openai/o4-mini")
RUBRIC = Rubric(
    name="bright", state="Question: {query}",
    question="How useful is this document for answering the question? It may help through the reasoning it "
             "supports, not just shared words.", item="Document",
    levels=("Not useful", "Related but not helpful", "Helpful for part of the answer", "Directly helps answer it"),
    llm_prompt="")
RATE = """You rate how useful a document is for answering a question. Reason about what the question needs,
then rate from 0 (not useful) to 3 (directly helps answer it). Return JSON: {"score": <0-3>}."""


def ns(domain):
    return f"bright-{domain}"


def save(name, obj):
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / f"{name}.json").write_text(json.dumps(obj))


def load_json(name):
    path = DATA / f"{name}.json"
    if not path.exists():
        raise SystemExit(f"{path} is missing. Run the step that writes it first.")
    return json.loads(path.read_text())


def run_load(args):
    from datasets import load_dataset  # the `studies` extra

    tp = client()
    for domain in DOMAINS:
        examples = load_dataset("xlangai/bright", "examples")[domain]
        documents = load_dataset("xlangai/bright", "documents")[domain]
        table = pa.table({"id": [str(d["id"]) for d in documents], "content": [d["content"] for d in documents]})
        tp.upsert(ns(domain), table, schema={"content": {"type": "string", "full_text_search": True}})
        save(f"{domain}-queries", [{"id": str(e["id"]), "query": e["query"], "gold": [str(g) for g in e["gold_ids"]],
                                    "excluded": [str(x) for x in e["excluded_ids"] if str(x) != "N/A"]}
                                   for e in examples])
        print(f"  {domain}: {table.num_rows:,} documents, {len(examples)} queries")


def run_bm25(args):
    tp = client()
    for domain in DOMAINS:
        queries = load_json(f"{domain}-queries")[:args.limit] if args.limit else load_json(f"{domain}-queries")
        out = {}
        for q in queries:
            filters = ("id", "NotIn", q["excluded"]) if q["excluded"] else None
            hits = tp.query(ns(domain), ("content", "BM25", q["query"]), top_k=DEPTH, filters=filters,
                            include=["content"], run_id=f"bright-{domain}", query_id=q["id"])
            out[q["id"]] = {"ids": hits["id"].to_pylist(),
                            "texts": [t[:3000] for t in hits["content"].to_pylist()] if "content" in hits.column_names
                            else [""] * hits.num_rows}
        save(f"{domain}-bm25", out)


def run_jev(args):
    judge = jev.client()
    for domain in DOMAINS:
        queries, cands = {q["id"]: q["query"] for q in load_json(f"{domain}-queries")}, load_json(f"{domain}-bm25")
        out = {}
        for qid, c in cands.items():
            grades = jev.grade_products(judge, queries[qid], dict(zip(c["ids"], c["texts"])), RUBRIC)
            out[qid] = {d: g.expected for d, g in grades.items()}
        save(f"{domain}-jev", out)


def run_llm(args):
    http = llm_judge.client()
    for domain in DOMAINS:
        queries, cands = {q["id"]: q["query"] for q in load_json(f"{domain}-queries")}, load_json(f"{domain}-bm25")
        out = {}
        for qid, c in cands.items():
            def one(pair, q=queries[qid]):
                raw = llm_judge.complete(http, MODEL, RATE, f"Question: {q}\n\nDocument: {pair[1]}", max_tokens=2000,
                                         json_mode=True)
                try:
                    return pair[0], float(json.loads(raw or "{}")["score"])
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    return pair[0], None
            with ThreadPoolExecutor(8) as pool:
                out[qid] = dict(pool.map(one, zip(c["ids"], c["texts"])))
        save(f"{domain}-llm", out)


def order(scores: dict[str, float | None], ids: list[str]) -> list[str]:
    return sorted(ids, key=lambda d: (-(scores.get(d) if scores.get(d) is not None else -1), ids.index(d)))


def cascade(jev_scores, llm_scores, ids):
    """Jev orders all 100; the LLM reorders Jev's top 20; the rest keep Jev's order."""
    by_jev = order(jev_scores, ids)
    return order(llm_scores, by_jev[:CASCADE]) + by_jev[CASCADE:]


def run_report(args):
    rows = []
    for domain in DOMAINS:
        queries = {q["id"]: q for q in load_json(f"{domain}-queries")}
        bm25, jev_s, llm = load_json(f"{domain}-bm25"), load_json(f"{domain}-jev"), load_json(f"{domain}-llm")
        qids = [q for q in bm25 if q in jev_s and q in llm]
        ranked = {
            "bm25": {q: bm25[q]["ids"] for q in qids},
            "jev": {q: order(jev_s[q], bm25[q]["ids"]) for q in qids},
            "llm": {q: order(llm[q], bm25[q]["ids"]) for q in qids},
            "cascade": {q: cascade(jev_s[q], llm[q], bm25[q]["ids"]) for q in qids},
        }
        score = {name: sum(ndcg(by_q[q], {g: 1 for g in queries[q]["gold"]}) or 0 for q in qids) / len(qids)
                 for name, by_q in ranked.items()}
        gain = score["llm"] - score["bm25"]
        kept = (score["cascade"] - score["bm25"]) / gain if gain else float("nan")
        print(f"  {domain:<10} " + "  ".join(f"{n} {v:.3f}" for n, v in score.items())
              + f"   cascade keeps {kept:.0%} of the LLM's gain with {CASCADE} LLM calls a query instead of {DEPTH}")
        rows.append({"domain": domain, **{n: round(v, 4) for n, v in score.items()}, "share_kept": round(kept, 4)})
    write_csv(OUT / "bright.csv", rows)


if __name__ == "__main__":
    commands = {"load": run_load, "bm25": run_bm25, "jev": run_jev, "llm": run_llm, "report": run_report}
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=commands)
    ap.add_argument("--limit", type=int, help="only the first N queries per domain")
    args = ap.parse_args()
    commands[args.command](args)
