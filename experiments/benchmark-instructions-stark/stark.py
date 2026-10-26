"""STaRK-Amazon: can per-condition checks make a cheap reranker follow multi-condition product queries?

  retrieve  Free (local). BM25 over STaRK-Amazon's product documents (bm25s),
            top 100 for each test query. Needs the stark-qa and bm25s packages,
            which the study.toml step adds with `uv run --with`.
  rerank    Paid (Jev). A Score per candidate.
  extract   Paid (LLM). Each query's conditions, as JSON.
  check     Paid (Jev). A Noul per candidate per condition; the score is
            multiplied by every P(met).
  report    Free. Hit@1 and MRR for BM25, Jev, and Jev with checks, with
            bootstrap intervals over queries.

STaRK's Amazon data terms aren't stated; say so next to any result.

  uv run --with stark-qa --with bm25s experiments/benchmark-instructions-stark/stark.py retrieve
"""

import argparse
import json
import os
import random
from pathlib import Path

from typesafe_sdk import Noul

from relevancy import jev, llm_judge
from relevancy.rubric import Rubric
from relevancy.warehouse import ROOT

OUT = Path(__file__).resolve().parent / "results"
DATA = ROOT / "data" / "stark"
DEPTH = 100
MODEL = os.environ.get("CONSTRAINT_MODEL", "openai/gpt-4o-mini")
RUBRIC = Rubric(
    name="stark", state="Product search request: {query}",
    question="How well does this product satisfy every part of the request?", item="Product",
    levels=("Irrelevant: not what was asked for", "Related: the right kind of product, but misses most conditions",
            "Close: meets most conditions", "Exact: meets every condition in the request"),
    llm_prompt="")
EXTRACT = """You list every condition a product must meet to satisfy a product search request.
Return JSON: {"constraints": [{"type": "condition", "value": "..."}]}, one entry per condition."""


def save(name, rows):
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"  wrote {len(rows):,} rows to data/stark/{name}.jsonl")


def load_rows(name):
    path = DATA / f"{name}.jsonl"
    if not path.exists():
        raise SystemExit(f"{path} is missing. Run `{name}` first.")
    return [json.loads(line) for line in path.read_text().splitlines()]


def load_stark():
    """(test queries as (qid, text, answer ids), {doc id: text}) from the stark-qa package."""
    from stark_qa import load_qa, load_skb

    qa = load_qa("amazon")
    skb = load_skb("amazon", download_processed=True, root=str(DATA / "skb"))
    test = [qa[int(i)] for i in qa.get_idx_split()["test"]]
    docs = {int(i): skb.get_doc_info(int(i), add_rel=False, compact=True) for i in skb.candidate_ids}
    return [(str(q[1]), q[0], [int(a) for a in q[2]]) for q in test], docs


def bm25_top(docs: dict[int, str], queries: list[str], k: int) -> list[list[int]]:
    import bm25s

    ids = list(docs)
    retriever = bm25s.BM25()
    retriever.index(bm25s.tokenize([docs[i] for i in ids], stopwords="en"))
    hits, _ = retriever.retrieve(bm25s.tokenize(queries, stopwords="en"), k=k)
    return [[ids[j] for j in row] for row in hits]


def run_retrieve(args):
    queries, docs = load_stark()
    if args.limit:
        queries = queries[:args.limit]
    tops = bm25_top(docs, [q[1] for q in queries], DEPTH)
    save("candidates", [{"query_id": qid, "text": text, "answers": ans, "ids": top,
                         "texts": [docs[i][:1500] for i in top]} for (qid, text, ans), top in zip(queries, tops)])


def run_rerank(args):
    judge, out = jev.client(), []
    for c in load_rows("candidates"):
        grades = jev.grade_products(judge, c["text"], {str(i): t for i, t in zip(c["ids"], c["texts"])}, RUBRIC)
        out.append({"query_id": c["query_id"], "scores": {i: g.expected for i, g in grades.items()}})
    save("jev", out)


def run_extract(args):
    http, out = llm_judge.client(), []
    for c in load_rows("candidates"):
        raw = llm_judge.complete(http, MODEL, EXTRACT, f"Request: {c['text']}", max_tokens=300, json_mode=True)
        try:
            cons = [x["value"] for x in json.loads(raw or "{}").get("constraints", []) if x.get("value")]
        except (json.JSONDecodeError, AttributeError, TypeError):
            cons = []
        out.append({"query_id": c["query_id"], "constraints": cons})
    save("constraints", out)


def run_check(args):
    cons = {r["query_id"]: r["constraints"] for r in load_rows("constraints")}
    judge, out = jev.client(), []
    for c in load_rows("candidates"):
        qs = {(str(i), k): Noul(instructions=f"Does this product meet this condition: {con}?\n\nProduct: {t}")
              for i, t in zip(c["ids"], c["texts"]) for k, con in enumerate(cons.get(c["query_id"], []))}
        answers = jev.ask(judge, "Check products against the conditions of a search request.", qs) if qs else {}
        met = {}
        for (i, _), a in answers.items():
            met[i] = met.get(i, 1.0) * jev.yes_probability(a)
        out.append({"query_id": c["query_id"], "met": met})
    save("checks", out)


def metrics(ranked: list[str], answers: set[str]) -> tuple[float, float]:
    hit1 = float(bool(ranked) and ranked[0] in answers)
    rr = next((1 / (i + 1) for i, d in enumerate(ranked) if d in answers), 0.0)
    return hit1, rr


def with_interval(vals: list[float]) -> str:
    rnd = random.Random(0)
    means = sorted(sum(rnd.choices(vals, k=len(vals))) / len(vals) for _ in range(1000))
    return f"{sum(vals) / len(vals):.3f} [{means[24]:.3f}, {means[974]:.3f}]"


def run_report(args):
    cands = {c["query_id"]: c for c in load_rows("candidates")}
    jev_s = {r["query_id"]: r["scores"] for r in load_rows("jev")}
    met = {r["query_id"]: r["met"] for r in load_rows("checks")} if (DATA / "checks.jsonl").exists() else {}
    systems = {"bm25": lambda q: [str(i) for i in cands[q]["ids"]],
               "jev": lambda q: sorted(jev_s[q], key=lambda d: -(jev_s[q][d] or 0))}
    if met:
        systems["jev + checks"] = lambda q: sorted(jev_s[q], key=lambda d: -((jev_s[q][d] or 0) * met[q].get(d, 1.0)))
    qids = [q for q in cands if q in jev_s]
    print(f"  STaRK-Amazon test, {len(qids)} queries (data terms unstated)")
    for name, fn in systems.items():
        scores = [metrics(fn(q), {str(a) for a in cands[q]["answers"]}) for q in qids]
        print(f"  {name:<14} Hit@1 {with_interval([s[0] for s in scores])}   MRR {with_interval([s[1] for s in scores])}")


if __name__ == "__main__":
    commands = {"retrieve": run_retrieve, "rerank": run_rerank, "extract": run_extract, "check": run_check,
                "report": run_report}
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=commands)
    ap.add_argument("--limit", type=int, help="only the first N test queries")
    args = ap.parse_args()
    commands[args.command](args)
