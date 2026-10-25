"""Query routing: do different query types want different retrieval settings?

  slices    Free. Each stored first-stage run's nDCG@10 per query slice. If the
            same retriever wins every slice, there's nothing to route, so stop.
  classify  Paid. One Jev Choice per query: names a brand or model, specifies
            sizes or quantities, or neither.
  route     Free. Needs recall-fusion's fetch and sweep. Each type picks the
            best of recall-fusion's top 5 settings on the tuning half; the
            baseline is recall-fusion's single best setting.
  rerank    Paid. Voyage over the routed top 50 and the best single setting's.
  report    Free. Routed against the best single setting, held-out nDCG@10
            after Voyage, and before it.

  uv run experiments/relevance-routing/routing.py slices
"""

import argparse
import csv
import importlib.util
import itertools
import math
from collections import defaultdict
from pathlib import Path

from typesafe_sdk import Choice

from relevancy import jev
from relevancy.study import (
    NAMESPACE,
    compare,
    fuse,
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
from relevancy.tpuf import client, rerank
from relevancy.warehouse import ROOT, read_only

OUT = Path(__file__).resolve().parent / "results"
CANDIDATES = 5  # settings each type may choose from; more and the pick is noise at ~30 queries a type
TYPES = {
    "brand": "The query names a brand, a product line, or a model number",
    "specs": "The query specifies sizes, quantities, materials, or other measurable attributes, and names no brand",
    "neither": "The query describes what the product is or is for, with no brand and no specifications",
}


def _recall_fusion():
    path = ROOT / "experiments" / "recall-fusion" / "recall_fusion.py"
    spec = importlib.util.spec_from_file_location("recall_fusion", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RF = _recall_fusion()


def run_slices(args):
    with read_only() as con:
        human = labels(con)
        slices = dict(con.execute("SELECT query_id, slice FROM queries WHERE dataset = 'esci-us' AND sampled").fetchall())
        runs = {r: latest_run(con, retriever=r) for r in ("bm25", "bm25-title", "ann", "hybrid-rrf")}
        lists = {r: {q: [d for d, _ in v] for q, v in run_lists(con, rid).items()} for r, rid in runs.items()}
    by_slice = defaultdict(set)
    for q, s in slices.items():
        by_slice[s].add(q)
    winners = set()
    print("slice              n   " + "  ".join(f"{r:>10}" for r in runs))
    for s, qids in sorted(by_slice.items()):
        scores = {r: mean(per_query(lists[r], human, only=qids)) for r in runs}
        winners.add(max(scores, key=scores.get))
        print(f"{s:<16} {len(qids):>4}   " + "  ".join(f"{scores[r]:>10.3f}" for r in runs))
    if len(winners) == 1:
        print(f"\nThe same retriever ({winners.pop()}) wins every slice. Stop: there's nothing to route.")
    else:
        print(f"\nDifferent slices prefer {', '.join(sorted(winners))}, so routing might help. Go on to classify.")


def run_classify(args):
    with read_only() as con:
        queries = sampled(con)
    answers = jev.ask(jev.client(), "Classify shopping queries by what they specify.",
                      {qid: Choice(instructions=f"Which describes this shopping query?\n\nQuery: {text}", criteria=TYPES)
                       for qid, text in queries})
    rows = [{"query_id": qid, "text": text, "type": answers[qid]["choice"],
             "confidence": round(answers[qid]["confidence"], 3)} for qid, text in queries]
    write_csv(OUT / "types.csv", rows)
    counts = defaultdict(int)
    for r in rows:
        counts[r["type"]] += 1
    print("  " + ", ".join(f"{t}: {n}" for t, n in sorted(counts.items())))


def candidates():
    with open(RF.OUT / "recall_sweep.csv") as f:
        rows = sorted(csv.DictReader(f), key=lambda r: -float(r["tune_d_nDCG@10"]))
    configs = {RF.name(c): c for c in (dict(zip(RF.GRID, v)) for v in itertools.product(*RF.GRID.values()))}
    return [configs[r["config"]] for r in rows[:CANDIDATES]]


def run_route(args):
    with read_only() as con:
        human = labels(con)
    types = {r["query_id"]: r["type"] for r in read_csv(OUT / "types.csv")}
    ranked, cands = RF.lists(), candidates()
    best_single = cands[0]
    routed = {}
    for t in TYPES:
        tune = {q for q, tt in types.items() if tt == t and tune_half(q)}
        scores = [mean(per_query({q: fuse(ranked, q, c, top=50) for q in tune}, human)) for c in cands]
        # Ties (and a type with no tuning queries) go to the better-ranked setting.
        pick = cands[max(range(len(cands)), key=lambda i: (-1 if math.isnan(scores[i]) else scores[i], -i))]
        routed[t] = pick
        print(f"  {t:<8} {len(tune):>3} tuning queries -> {RF.name(pick)}")
    rows = []
    for q, t in types.items():
        rows.append({"variant": "routed", "query_id": q, "setting": RF.name(routed[t]),
                     "ids": " ".join(fuse(ranked, q, routed[t], top=50))})
        rows.append({"variant": "best single", "query_id": q, "setting": RF.name(best_single),
                     "ids": " ".join(fuse(ranked, q, best_single, top=50))})
    write_csv(OUT / "routed.csv", rows)


def run_rerank(args):
    with read_only() as con:
        texts = dict(sampled(con))
    tp, rows = client(), []
    for r in read_csv(OUT / "routed.csv"):
        hits = rerank(tp, NAMESPACE, "voyage/rerank-3", texts[r["query_id"]], r["ids"].split(),
                      run_id=f"routing-{r['variant']}", query_id=r["query_id"])
        rows.append(r | {"variant": f"{r['variant']}+voyage", "ids": " ".join(hits["id"].to_pylist())})
    write_csv(OUT / "reranked.csv", rows)


def run_report(args):
    with read_only() as con:
        human = labels(con)
    rows = read_csv(OUT / "routed.csv") + read_csv(OUT / "reranked.csv")
    by = defaultdict(dict)
    for r in rows:
        by[r["variant"]][r["query_id"]] = r["ids"].split()
    held = {q for q in by["routed"] if not tune_half(q)}
    for a, b in (("routed+voyage", "best single+voyage"), ("routed", "best single")):
        for m in ("nDCG@10", "condensed"):
            print(f"  {a} vs {b}, held-out {m}: {compare(per_query(by[a], human, m, held), per_query(by[b], human, m, held))}")


if __name__ == "__main__":
    commands = {"slices": run_slices, "classify": run_classify, "route": run_route, "rerank": run_rerank,
                "report": run_report}
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=commands)
    commands[ap.parse_args().command](None)
