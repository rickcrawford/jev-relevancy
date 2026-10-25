"""Constraint checks: one Jev yes/no question per product per constraint in the query.

  extract  Paid (LLM). Lists each sampled query's constraints with values, as
           JSON: brand, size, color, quantity, material, excludes, other.
  check    Paid (Jev). For each query's stored hybrid top 50, one Noul per
           product per constraint: "does this product meet it?" (for an
           exclusion: "does it contain it?", inverted).
  report   Free. Voyage and Jev reranks combined with the checks, against each
           reranker alone, held-out nDCG@10. Two combinations: multiply the
           score by every P(met), or subtract lambda x sum of P(failed), with
           lambda picked on the tuning half. Broken down by constraint type;
           negation queries reported apart, since their human labels are often
           wrong (see judge-negation-audit).
  nevir    Paid (Jev). NevIR's test split: Jev scores each of the four
           (query, document) pairs; pairwise accuracy with a bootstrap interval.

ExcluIR's data is a Google Drive folder with no documented format, so it isn't
automated here.

  uv run experiments/relevance-constraints/constraints.py extract
"""

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path

from typesafe_sdk import Noul

from relevancy import jev, llm_judge
from relevancy.rubric import load
from relevancy.study import (
    compare,
    doc_texts,
    labels,
    latest_run,
    mean,
    per_query,
    read_csv,
    run_lists,
    sampled,
    tune_half,
    write_csv,
)
from relevancy.warehouse import ROOT, read_only

OUT = Path(__file__).resolve().parent / "results"
DATA = ROOT / "data" / "constraints"
MODEL = os.environ.get("CONSTRAINT_MODEL", "openai/gpt-4o-mini")
TYPES = ("brand", "size", "color", "quantity", "material", "excludes", "other")
LAMBDAS = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]
EXTRACT = f"""You list the explicit constraints in a shopping query.
Return JSON: {{"constraints": [{{"type": one of {list(TYPES)}, "value": "..."}}]}}.
Only constraints the shopper stated: a brand, a size, a color, a quantity, a material, something to exclude
("without X", "not X", "X-free"), or another hard requirement. A plain product type is not a constraint.
Return {{"constraints": []}} if there are none."""


def run_extract(args):
    with read_only() as con:
        queries = sampled(con)
    http, rows = llm_judge.client(), []
    for qid, text in queries:
        raw = llm_judge.complete(http, MODEL, EXTRACT, f"Query: {text}", max_tokens=300, json_mode=True)
        try:
            found = [c for c in json.loads(raw or "{}").get("constraints", [])
                     if c.get("type") in TYPES and str(c.get("value", "")).strip()]
        except (json.JSONDecodeError, AttributeError):
            found = None
        rows.append({"query_id": qid, "text": text, "ok": found is not None, "constraints": json.dumps(found or [])})
    write_csv(OUT / "constraints.csv", rows)
    print(f"  {sum(bool(json.loads(r['constraints'])) for r in rows)} of {len(rows)} queries have a constraint; "
          f"{sum(not r['ok'] for r in rows)} failed to parse")


def question(c: dict, product: str) -> str:
    if c["type"] == "excludes":
        return f"Does this product contain or include {c['value']}?\n\nProduct: {product}"
    return f"Does this product match this requirement from the shopper: {c['type']} {c['value']}?\n\nProduct: {product}"


def run_check(args):
    constraints = {r["query_id"]: json.loads(r["constraints"]) for r in read_csv(OUT / "constraints.csv")}
    with read_only() as con:
        hybrid = run_lists(con, latest_run(con))
        texts = doc_texts(con, [d for v in hybrid.values() for d, _ in v])
    judge, rows = jev.client(), []
    for qid, cs in constraints.items():
        if not cs or qid not in hybrid:
            continue
        qs = {(d, i): Noul(instructions=question(c, texts.get(d, ""))) for d, _ in hybrid[qid] for i, c in enumerate(cs)}
        answers = jev.ask(judge, "Check products against a shopper's stated requirements.", qs)
        for (d, i), a in answers.items():
            p = jev.yes_probability(a)
            met = 1 - p if cs[i]["type"] == "excludes" else p
            rows.append({"query_id": qid, "doc_id": d, "constraint": i, "type": cs[i]["type"], "p_met": round(met, 4)})
    DATA.mkdir(parents=True, exist_ok=True)
    write_csv(DATA / "checks.csv", rows)


def normalized(pairs):
    s = [x for _, x in pairs if x is not None]
    lo, hi = (min(s), max(s)) if s else (0.0, 1.0)
    return {d: ((x - lo) / (hi - lo) if hi > lo else 0.5) if x is not None else 0.0 for d, x in pairs}


def run_report(args):
    checks = defaultdict(lambda: defaultdict(list))
    types = defaultdict(set)
    for r in read_csv(DATA / "checks.csv"):
        checks[r["query_id"]][r["doc_id"]].append(float(r["p_met"]))
        types[r["query_id"]].add(r["type"])
    with read_only() as con:
        human = labels(con)
        bases = {"voyage": run_lists(con, latest_run(con, reranker="voyage/rerank-3")),
                 "jev": run_lists(con, latest_run(con, reranker="jev:jev-1.13.0"))}
    queries = sorted(checks)
    held, tune = {q for q in queries if not tune_half(q)}, {q for q in queries if tune_half(q)}

    def multiply(base, q):
        s = normalized(base[q])
        return sorted(s, key=lambda d: (-(s[d] * prod(checks[q].get(d, []))), d))

    def penalty(base, q, lam):
        s = normalized(base[q])
        return sorted(s, key=lambda d: (-(s[d] - lam * sum(1 - p for p in checks[q].get(d, []))), d))

    for name, base in bases.items():
        alone = {q: [d for d, _ in base[q]] for q in queries if q in base}
        lam = max(LAMBDAS, key=lambda x: mean(per_query({q: penalty(base, q, x) for q in tune if q in base}, human)))
        mult = {q: multiply(base, q) for q in queries if q in base}
        pen = {q: penalty(base, q, lam) for q in queries if q in base}
        print(f"\n## {name} + checks, held-out half ({len(held)} queries with a constraint)")
        for label, ranked in (("multiply", mult), (f"penalty, lambda {lam}", pen)):
            print(f"  {label:<20} nDCG@10   {compare(per_query(ranked, human, only=held), per_query(alone, human, only=held))}")
            print(f"  {'':<20} condensed {compare(per_query(ranked, human, 'condensed', held), per_query(alone, human, 'condensed', held))}")
        print("  by constraint type (multiply, all queries with that type):")
        for t in TYPES:
            qs = {q for q in queries if t in types[q]}
            if qs:
                note = "  (human labels unreliable on exclusions; see judge-negation-audit)" if t == "excludes" else ""
                print(f"    {t:<9} {compare(per_query(mult, human, only=qs), per_query(alone, human, only=qs))}{note}")


def prod(xs):
    out = 1.0
    for x in xs:
        out *= x
    return out


def run_nevir(args):
    from datasets import load_dataset  # the `studies` extra

    rows = load_dataset("orionweller/NevIR", "test")["test"]
    rubric, judge, out = load("generic"), jev.client(), []
    for i, r in enumerate(rows):
        # Two queries, two documents: q1 should rank doc1 first, q2 doc2.
        g1 = jev.grade_products(judge, r["q1"], {"doc1": r["doc1"], "doc2": r["doc2"]}, rubric)
        g2 = jev.grade_products(judge, r["q2"], {"doc1": r["doc1"], "doc2": r["doc2"]}, rubric)
        correct = g1["doc1"].expected > g1["doc2"].expected and g2["doc2"].expected > g2["doc1"].expected
        out.append({"pair": i, "q1_doc1": g1["doc1"].expected, "q1_doc2": g1["doc2"].expected,
                    "q2_doc1": g2["doc1"].expected, "q2_doc2": g2["doc2"].expected, "correct": correct})
    write_csv(OUT / "nevir.csv", out)
    acc = [float(r["correct"]) for r in out]
    rnd = random.Random(0)
    means = sorted(sum(rnd.choices(acc, k=len(acc))) / len(acc) for _ in range(1000))
    print(f"  NevIR test pairwise accuracy {sum(acc) / len(acc):.1%} [{means[24]:.1%}, {means[974]:.1%}], "
          f"n={len(acc)}; published: GPT-4o 70.1%, RankGPT with o3-mini 77.3%")


if __name__ == "__main__":
    commands = {"extract": run_extract, "check": run_check, "report": run_report, "nevir": run_nevir}
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=commands)
    commands[ap.parse_args().command](None)
