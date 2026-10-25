"""Reranker ensembles and cascades: more quality for the same cost, or the same quality for less?

  ensembles   Free. From the stored Voyage and Jev reranks of the same hybrid top
              50: mean rank, a score mix (weight picked on the tuning half), and a
              mean rank of every stored reranker. Compared with stored Voyage.
  candidates  Paid. Hybrid's top 100 for the 500 sampled queries.
  voyage      Paid. voyage/rerank-3 over each top 100, timed per call.
  jev         Paid. Jev over each top 100, timed per call.
  cascade     Paid. Jev over Voyage's top 20, timed per call.
  report      Free. The cascades against Voyage over the top 100: better (lower
              bound above zero) or the same for less (TOST within 0.01).

Every call is timed from this one client, so wall times compare. Voyage runs
inside turbopuffer and Jev at TypeSafe; both are measured as the wall time
they add.

  uv run experiments/relevance-reranker-cascade/cascade.py ensembles
"""

import argparse
import json
import os
import time
from pathlib import Path

from relevancy import jev
from relevancy.rubric import load
from relevancy.study import (
    NAMESPACE,
    compare,
    doc_texts,
    interval,
    labels,
    latest_run,
    mean,
    per_query,
    run_lists,
    sampled,
    tune_half,
    write_csv,
)
from relevancy.tpuf import client, hybrid_queries, rerank
from relevancy.warehouse import ROOT, read_only

OUT = Path(__file__).resolve().parent / "results"
DATA = ROOT / "data" / "cascade"
MARGIN = 0.01        # equivalence margin on nDCG@10, fixed before the run
CASCADE_TOP = 20     # how many of Voyage's top products Jev reorders
WEIGHTS = [i / 10 for i in range(1, 10)]  # 0 and 1 would just be Jev or Voyage alone
# Serving prices, for the "cheaper" verdict: dollars per Voyage rerank call and per Jev request. Without
# both, the report gives call counts only and can't call anything cheaper.
PRICES = {"voyage": os.environ.get("VOYAGE_PRICE"), "jev": os.environ.get("JEV_PRICE")}


def held_and_tune(qids):
    return {q for q in qids if not tune_half(q)}, {q for q in qids if tune_half(q)}


def verdict(name: str, a: dict, b: dict, k: int) -> None:
    """`k` variants are compared with the same baseline, so both tests are Bonferroni-corrected."""
    diffs = [a[q] - b[q] for q in sorted(a.keys() & b.keys())]
    lo95, hi95 = interval(diffs, 1 - 0.05 / k)
    lo, hi = interval(diffs, 1 - 0.10 / k)  # TOST: two one-sided tests at 5% / k each
    better = "better" if lo95 > 0 else "not shown better"
    eq = "equivalent" if -MARGIN < lo and hi < MARGIN else "not shown equivalent"
    print(f"  {name:<22} {compare(a, b)}; corrected [{lo95:+.3f}, {hi95:+.3f}] {better}; TOST {eq}")


# -- ensembles (stored runs, no requests) -------------------------------------

def normalized(pairs):
    scores = [s for _, s in pairs if s is not None]
    lo, hi = (min(scores), max(scores)) if scores else (0, 1)
    return {d: ((s - lo) / (hi - lo) if hi > lo else 0.5) if s is not None else 0.0 for d, s in pairs}


def run_ensembles(args):
    with read_only() as con:
        human, queries = labels(con), [q for q, _ in sampled(con)]
        voyage = run_lists(con, latest_run(con, reranker="voyage/rerank-3"))
        jev_run = run_lists(con, latest_run(con, reranker="jev:jev-1.13.0"))
        others = {}
        for model in ("cohere/rerank-v3.5", "zeroentropy/zerank-2"):
            try:
                others[model] = run_lists(con, latest_run(con, reranker=model))
            except SystemExit:
                pass
    held, tune = held_and_tune(queries)

    def mean_rank(runs, qid):
        ranks = {}
        for run in runs:
            for i, (d, _) in enumerate(run.get(qid, []), 1):
                ranks.setdefault(d, []).append(i)
        return sorted(ranks, key=lambda d: (sum(ranks[d]) / len(ranks[d]), d))

    def mix(w, qid):
        v, j = normalized(voyage.get(qid, [])), normalized(jev_run.get(qid, []))
        return sorted(v.keys() | j.keys(), key=lambda d: (-(w * v.get(d, 0) + (1 - w) * j.get(d, 0)), d))

    def ndcg_of(order_fn, qids):
        return per_query({q: order_fn(q) for q in qids}, human, only=qids)

    best_w = max(WEIGHTS, key=lambda w: mean(ndcg_of(lambda q: mix(w, q), tune)))
    variants = {
        "voyage (stored)": lambda q: [d for d, _ in voyage.get(q, [])],
        "jev (stored)": lambda q: [d for d, _ in jev_run.get(q, [])],
        "mean rank": lambda q: mean_rank([voyage, jev_run], q),
        f"score mix w={best_w}": lambda q: mix(best_w, q),
    }
    if others:
        variants[f"mean rank, all {2 + len(others)}"] = lambda q: mean_rank([voyage, jev_run, *others.values()], q)
    rows = [{"variant": v, "query_id": q, "ids": " ".join(fn(q))} for v, fn in variants.items() for q in queries]
    write_csv(OUT / "ensembles.csv", rows, ["variant", "query_id", "ids"])

    base = ndcg_of(variants["voyage (stored)"], held)
    others_ = [v for v in variants if v != "voyage (stored)"]
    print(f"\n## Ensembles of the stored top-50 reranks, held-out half, against Voyage (weight picked on {len(tune)} "
          f"queries; intervals corrected for {len(others_)} comparisons)")
    for v in others_:
        verdict(v, ndcg_of(variants[v], held), base, len(others_))
    print("  Serving cost: every ensemble pays for each reranker it combines.")


# -- cascades (new requests) --------------------------------------------------

def save(name, rows):
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"wrote {len(rows)} rows to data/cascade/{name}.jsonl")


def append(name, row):
    """One row per query as it finishes, so a failure keeps what's paid for and a rerun resumes."""
    DATA.mkdir(parents=True, exist_ok=True)
    with open(DATA / f"{name}.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")


def done(name) -> set[str]:
    path = DATA / f"{name}.jsonl"
    return {json.loads(line)["query_id"] for line in path.read_text().splitlines()} if path.exists() else set()


def load_rows(name):
    path = DATA / f"{name}.jsonl"
    if not path.exists():
        raise SystemExit(f"{path} is missing. Run `{name}` first.")
    return {r["query_id"]: r for r in map(json.loads, path.read_text().splitlines())}


def run_candidates(args):
    with read_only() as con:
        queries = sampled(con)
    tp, rows = client(), []
    for qid, text in queries:
        (hits,) = tp.multi_query(NAMESPACE, hybrid_queries(text, 100), rerank_by=("RRF", {}), limit=100,
                                 run_id="cascade-candidates", query_id=qid)
        rows.append({"query_id": qid, "text": text, "ids": hits["id"].to_pylist()})
    save("candidates", rows)


def timed(fn, *a, **kw):
    started = time.perf_counter()
    out = fn(*a, **kw)
    return out, (time.perf_counter() - started) * 1000


def run_voyage(args):
    cands, tp, skip = load_rows("candidates"), client(), done("voyage")
    for qid, c in cands.items():
        if qid in skip:
            continue
        hits, ms = timed(rerank, tp, NAMESPACE, "voyage/rerank-3", c["text"], c["ids"], run_id="cascade-voyage",
                         query_id=qid)
        append("voyage", {"query_id": qid, "ids": hits["id"].to_pylist(), "ms": round(ms, 1), "calls": 1})


def jev_order(client_, rubric, text, ids, texts):
    grades = jev.grade_products(client_, text, {d: texts[d] for d in ids}, rubric)
    order = sorted(ids, key=lambda d: (-(grades[d].expected or 0), ids.index(d)))
    return order, {d: grades[d].expected for d in ids}


def requests(n: int) -> int:
    return -(-n // jev.PER_REQUEST)  # Jev takes PER_REQUEST products a request


def run_jev(args):
    # Serial, like Voyage, so wall times compare (a thread pool would add contention to Jev's).
    cands, rubric, judge, skip = load_rows("candidates"), load("esci"), jev.client(), done("jev")
    with read_only() as con:
        texts = doc_texts(con, [d for c in cands.values() for d in c["ids"]])
    for qid, c in cands.items():
        if qid in skip:
            continue
        (order, expected), ms = timed(jev_order, judge, rubric, c["text"], c["ids"], texts)
        append("jev", {"query_id": qid, "ids": order, "expected": [expected[d] for d in order], "ms": round(ms, 1),
                       "calls": requests(len(order))})


def run_cascade(args):
    cands, voyage, rubric, judge = load_rows("candidates"), load_rows("voyage"), load("esci"), jev.client()
    skip = done("cascade")
    with read_only() as con:
        texts = doc_texts(con, [d for v in voyage.values() for d in v["ids"][:CASCADE_TOP]])
    for qid in voyage:
        if qid in skip:
            continue
        top = voyage[qid]["ids"][:CASCADE_TOP]
        (order, _), ms = timed(jev_order, judge, rubric, cands[qid]["text"], top, texts)
        append("cascade", {"query_id": qid, "ids": order + voyage[qid]["ids"][CASCADE_TOP:], "ms": round(ms, 1),
                           "calls": requests(len(top))})


def run_report(args):
    with read_only() as con:
        human = labels(con)
    voyage, jev_rows, cascade = load_rows("voyage"), load_rows("jev"), load_rows("cascade")
    queries = sorted(voyage.keys() & jev_rows.keys() & cascade.keys())
    held, tune = held_and_tune(queries)

    def unsure(qid, margin):
        e = jev_rows[qid]["expected"]
        return len(e) > 1 and (e[0] or 0) - (e[1] or 0) < margin

    def escalate(margin):
        return {q: voyage[q]["ids"] if unsure(q, margin) else jev_rows[q]["ids"] for q in queries}

    margins = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
    base_tune = per_query({q: voyage[q]["ids"] for q in tune}, human)
    # The smallest margin whose tuning-half nDCG is within the equivalence margin of Voyage's: cheapest that keeps quality.
    tuned = {m: mean(per_query({q: r[q] for q in tune}, human)) for m in margins for r in [escalate(m)]}
    margin = next((m for m in margins if tuned[m] >= mean(base_tune) - MARGIN), margins[-1])
    def cost(voyage_calls, jev_calls):
        if not all(PRICES.values()):
            return None
        return voyage_calls * float(PRICES["voyage"]) + jev_calls * float(PRICES["jev"])

    # (ranking, wall ms per query, (Voyage calls, Jev requests) per query)
    variants = {
        "jev, top 100": ({q: jev_rows[q]["ids"] for q in queries}, lambda q: jev_rows[q]["ms"],
                         lambda q: (0, jev_rows[q]["calls"])),
        f"cascade (Jev on Voyage's top {CASCADE_TOP})": ({q: cascade[q]["ids"] for q in queries},
                                                       lambda q: voyage[q]["ms"] + cascade[q]["ms"],
                                                       lambda q: (1, cascade[q]["calls"])),
        f"escalate, margin {margin}": (escalate(margin),
                                       lambda q: jev_rows[q]["ms"] + (voyage[q]["ms"] if unsure(q, margin) else 0),
                                       lambda q: (int(unsure(q, margin)), jev_rows[q]["calls"])),
    }
    base = per_query({q: voyage[q]["ids"] for q in held}, human)
    base_cost = cost(len(held), 0)
    print(f"\n## Against Voyage over the top 100, held-out half ({len(held)} queries; "
          f"intervals corrected for {len(variants)} comparisons)")
    print(f"  baseline Voyage top 100: nDCG@10 {mean(base):.3f}, wall p50 {pct([voyage[q]['ms'] for q in held], .5):.0f} ms"
          + (f", ${base_cost / len(held):.5f} a query" if base_cost is not None else ""))
    for name, (ranked, ms, calls) in variants.items():
        a = per_query({q: ranked[q] for q in held}, human)
        verdict(name, a, base, len(variants))
        c = per_query({q: ranked[q] for q in held}, human, "condensed")
        cb = per_query({q: voyage[q]["ids"] for q in held}, human, "condensed")
        v_calls = sum(calls(q)[0] for q in held)
        j_calls = sum(calls(q)[1] for q in held)
        total = cost(v_calls, j_calls)
        price = (f"${total / len(held):.5f} a query, {'cheaper' if total < base_cost else 'not cheaper'} than Voyage"
                 if total is not None else "set VOYAGE_PRICE and JEV_PRICE to price it")
        print(f"    condensed {compare(c, cb)}; wall p50 {pct([ms(q) for q in held], .5):.0f} ms, "
              f"p99 {pct([ms(q) for q in held], .99):.0f} ms")
        print(f"    calls: {v_calls / len(held):.2f} Voyage and {j_calls / len(held):.2f} Jev a query; {price}")


def pct(vals, p):
    vals = sorted(vals)
    return vals[min(len(vals) - 1, int(p * len(vals)))] if vals else float("nan")


if __name__ == "__main__":
    commands = {"ensembles": run_ensembles, "candidates": run_candidates, "voyage": run_voyage, "jev": run_jev,
                "cascade": run_cascade, "report": run_report}
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=commands)
    commands[ap.parse_args().command](None)
