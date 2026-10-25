"""Jev as a judge on TREC DL: does it order systems the way NIST's assessors do?

Research only: MS MARCO is licensed for non-commercial research.

  prepare  Free. TREC DL's judged queries, qrels, and the judged passages, from
           ir_datasets, into data/trec/<year>/. The default year is 2020.
  grade    Paid. Jev grades every judged pair with rubrics/trec-dl.toml.
  report   Free. Weighted kappa and exact agreement with NIST, per-level
           confusion, and calibration error. With --runs-dir (a folder of TREC
           run files: qid Q0 docid rank score tag), Kendall's tau between the
           systems' nDCG@10 under NIST's and Jev's labels, with a bootstrap
           interval over topics. That tau is the pass mark (lower bound >= 0.80).

  uv run experiments/judge-trec/trec.py prepare
  uv run experiments/judge-trec/trec.py report --runs-dir data/trec/2020/runs
"""

import argparse
import random
from collections import defaultdict
from pathlib import Path

from relevancy import jev
from relevancy.agreement import kendall_tau, weighted_kappa
from relevancy.rubric import load
from relevancy.study import ece, ndcg, read_csv, write_csv
from relevancy.warehouse import ROOT

OUT = Path(__file__).resolve().parent / "results"
DATA = ROOT / "data" / "trec"
DATASETS = {"2019": "msmarco-passage/trec-dl-2019/judged", "2020": "msmarco-passage/trec-dl-2020/judged"}
PASS_TAU = 0.80


def run_prepare(args):
    import ir_datasets  # the `studies` extra

    ds = ir_datasets.load(DATASETS[args.year])
    queries = {q.query_id: q.text for q in ds.queries_iter()}
    qrels = [(r.query_id, r.doc_id, int(r.relevance)) for r in ds.qrels_iter() if r.query_id in queries]
    docs = ds.docs_store().get_many({d for _, d, _ in qrels})
    folder = DATA / args.year
    write_csv(folder / "queries.csv", [{"query_id": q, "text": t} for q, t in queries.items()])
    write_csv(folder / "qrels.csv", [{"query_id": q, "doc_id": d, "grade": g} for q, d, g in qrels])
    write_csv(folder / "passages.csv", [{"doc_id": d, "text": doc.text} for d, doc in docs.items()])
    print(f"  {len(queries)} queries, {len(qrels):,} judged pairs, {len(docs):,} passages")


def run_grade(args):
    folder = DATA / args.year
    queries = {r["query_id"]: r["text"] for r in read_csv(folder / "queries.csv")}
    passages = {r["doc_id"]: r["text"] for r in read_csv(folder / "passages.csv")}
    by_q = defaultdict(dict)
    nist = {}
    for r in read_csv(folder / "qrels.csv"):
        by_q[r["query_id"]][r["doc_id"]] = passages.get(r["doc_id"], "")
        nist[(r["query_id"], r["doc_id"])] = int(r["grade"])
    rubric, judge, rows = load("trec-dl"), jev.client(), []
    for qid, docs in by_q.items():
        grades = jev.grade_products(judge, queries[qid], docs, rubric, model=args.model)
        for d, g in grades.items():
            probs = g.probs or {}
            rows.append({"query_id": qid, "doc_id": d, "nist": nist[(qid, d)], "jev": g.grade,
                         "expected": round(g.expected, 4), **{f"p{k}": round(probs.get(k, 0.0), 4) for k in range(4)}})
    write_csv(OUT / f"grades-{args.year}-{args.model}.csv", rows)


def read_runs(folder: Path) -> dict[str, dict[str, list[str]]]:
    runs: dict[str, dict[str, list[tuple[int, str]]]] = defaultdict(lambda: defaultdict(list))
    for path in sorted(folder.iterdir()):
        for line in path.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 6:
                runs[path.name][parts[0]].append((int(parts[3]), parts[2]))
    return {name: {q: [d for _, d in sorted(v)] for q, v in by_q.items()} for name, by_q in runs.items()}


def run_report(args):
    rows = read_csv(OUT / f"grades-{args.year}-{args.model}.csv")
    nist = [int(r["nist"]) for r in rows]
    jev_g = [int(r["jev"]) for r in rows]
    print(f"  {len(rows):,} pairs, Jev {args.model} against NIST, TREC DL {args.year}")
    print(f"  weighted kappa {weighted_kappa(nist, jev_g):.3f}, exact {sum(a == b for a, b in zip(nist, jev_g)) / len(rows):.3f}")
    conf = [max(float(r[f"p{k}"]) for k in range(4)) for r in rows]
    print(f"  calibration error (top-grade confidence against exact agreement): "
          f"{ece(conf, [a == b for a, b in zip(nist, jev_g)]):.3f}")
    print("  confusion (rows NIST, columns Jev):")
    for g in range(4):
        counts = [sum(1 for a, b in zip(nist, jev_g) if a == g and b == j) for j in range(4)]
        print(f"    {g}: " + " ".join(f"{c:>6}" for c in counts) + f"   right {counts[g] / max(sum(counts), 1):.0%}")
    if not args.runs_dir:
        print("  No --runs-dir, so no tau. Tau needs TREC's submitted run files (see README).")
        return
    runs = read_runs(Path(args.runs_dir))
    labels_nist, labels_jev = defaultdict(dict), defaultdict(dict)
    for r in rows:
        labels_nist[r["query_id"]][r["doc_id"]] = int(r["nist"])
        labels_jev[r["query_id"]][r["doc_id"]] = int(r["jev"])
    topics = sorted(labels_nist)

    def system_means(labels_, qs):
        return [sum(ndcg(run.get(q, []), labels_[q]) or 0 for q in qs) / len(qs) for run in runs.values()]

    def tau(qs):
        return kendall_tau(system_means(labels_nist, qs), system_means(labels_jev, qs))

    rnd = random.Random(0)
    taus = sorted(tau(rnd.choices(topics, k=len(topics))) for _ in range(1000))
    t, lo, hi = tau(topics), taus[24], taus[974]
    print(f"  tau over {len(runs)} systems and {len(topics)} topics: {t:.3f} [{lo:.3f}, {hi:.3f}]")
    print(f"  {'PASSES' if lo >= PASS_TAU else 'DOES NOT PASS'}: lower bound {'>=' if lo >= PASS_TAU else '<'} {PASS_TAU}")


if __name__ == "__main__":
    commands = {"prepare": run_prepare, "grade": run_grade, "report": run_report}
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=commands)
    ap.add_argument("--year", choices=DATASETS, default="2020")
    ap.add_argument("--model", default="jev-1.14.0")
    ap.add_argument("--runs-dir")
    args = ap.parse_args()
    commands[args.command](args)
