"""Semantic attributes at index time: label every product once, filter at query time.

  find          Free. The queries that ask for the attribute, from a pattern list
                fixed in advance. Tops up from ESCI's other test queries to 100.
  label         Paid, resumable. One Jev Noul per product in the namespace.
  check-export  Free. 100 products Jev said yes to and 100 it said no to, shuffled,
                with Jev's answer hidden, for a person to rate in the `rater` column.
  precision     Free. Precision and an estimated recall from the ratings. Stop if
                precision is under 0.9.
  patch         Paid. Writes <attribute> (bool) and <attribute>_p to the namespace.
  run           Paid. hybrid, attr-soft, and attr-filter for each found query.
  report        Free. attr-soft against hybrid on nDCG@10. The secondary attribute
                gets a Bonferroni-corrected (97.5%) interval.

  uv run experiments/relevance-attributes/attributes.py find
  uv run experiments/relevance-attributes/attributes.py label --attribute for_kids
"""

import argparse
import random
import re
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
from typesafe_sdk import Noul

from relevancy import jev
from relevancy.metrics import paired_delta
from relevancy.study import (
    DS,
    NAMESPACE,
    compare,
    doc_texts,
    interval,
    labels,
    mean,
    part_ids,
    per_query,
    read_csv,
    write_csv,
    write_part,
)
from relevancy.tpuf import client, hybrid_queries
from relevancy.warehouse import ROOT, read_only

OUT = Path(__file__).resolve().parent / "results"
DATA = ROOT / "data" / "attributes"
PRIMARY = "replacement_part"
ATTRIBUTES = {
    "replacement_part": {
        "question": "Is this a replacement part, refill, or accessory for another product, rather than the product itself?",
        "patterns": [r"\breplacement\b", r"\brefills?\b", r"\bspares?\b", r"\bcompatible with\b"],
    },
    "for_kids": {
        "question": "Is this product made for children?",
        "patterns": [r"\bfor kids\b", r"\bkids?\b", r"\btoddlers?\b", r"\bboys\b", r"\bgirls\b", r"\bbaby\b"],
    },
}
MIN_QUERIES = 100
CHUNK = 2_000
SCHEMA = pa.schema([("id", pa.string()), ("p", pa.float64())])


def attr(args) -> str:
    return args.attribute


def run_find(args):
    pats = [re.compile(p, re.IGNORECASE) for p in ATTRIBUTES[attr(args)]["patterns"]]
    with read_only() as con:
        rows = con.execute("""SELECT query_id, text, sampled FROM queries WHERE dataset = ? AND split = 'test'
                              ORDER BY NOT sampled, hash(query_id)""", [DS]).fetchall()
    hits = [(q, t, s) for q, t, s in rows if any(p.search(t) for p in pats)]
    sampled_hits = [h for h in hits if h[2]]
    chosen = sampled_hits + [h for h in hits if not h[2]][:max(0, MIN_QUERIES - len(sampled_hits))]
    write_csv(OUT / f"queries-{attr(args)}.csv", [{"query_id": q, "text": t, "sampled": s} for q, t, s in chosen])
    print(f"  {len(sampled_hits)} sampled queries match; {len(chosen)} with the top-up")


def run_label(args):
    folder, question = DATA / attr(args), ATTRIBUTES[attr(args)]["question"]
    done = part_ids(folder)
    ids = [d for d in client().scan(NAMESPACE, columns=[]).read_all()["id"].to_pylist() if d not in done]
    print(f"  {len(done):,} labeled, {len(ids):,} to go", flush=True)
    judge = jev.client()
    for i in range(0, len(ids), CHUNK):
        chunk = ids[i:i + CHUNK]
        with read_only() as con:
            texts = doc_texts(con, chunk)
        missing = [d for d in chunk if not texts.get(d)]
        if missing:
            print(f"  {len(missing)} products have no text in the warehouse; skipped", flush=True)
        answers = jev.ask(judge, "Label products in a shopping catalogue.",
                          {d: Noul(instructions=f"{question}\n\nProduct: {texts[d]}") for d in chunk if texts.get(d)})
        # Unanswered products aren't written, so the next run retries them.
        write_part(folder, [{"id": d, "p": jev.yes_probability(a)} for d, a in answers.items()], SCHEMA)
        print(f"  {len(done) + i + len(chunk):,} labeled", flush=True)


def jev_labels(a: str) -> dict[str, float]:
    import duckdb
    folder = DATA / a
    return dict(duckdb.sql(f"SELECT id, p FROM '{folder}/*.parquet'").fetchall())


def run_check_export(args):
    p = jev_labels(attr(args))
    rnd = random.Random(7)
    yes, no = sorted(d for d, v in p.items() if v >= 0.5), sorted(d for d, v in p.items() if v < 0.5)
    picked = [(d, True) for d in rnd.sample(yes, min(100, len(yes)))] + [(d, False) for d in rnd.sample(no, min(100, len(no)))]
    rnd.shuffle(picked)
    with read_only() as con:
        texts = doc_texts(con, [d for d, _ in picked])
    write_csv(OUT / f"check-{attr(args)}.csv",
              [{"id": d, "question": ATTRIBUTES[attr(args)]["question"], "product": texts.get(d, ""), "rater": ""}
               for d, _ in picked])
    write_csv(OUT / f"check-{attr(args)}-key.csv", [{"id": d, "jev_yes": y} for d, y in picked])
    print("  fill in `rater` with yes or no, without looking at the key file")


def run_precision(args):
    rated = {r["id"]: r["rater"].strip().lower() for r in read_csv(OUT / f"check-{attr(args)}.csv")}
    key = {r["id"]: r["jev_yes"] == "True" for r in read_csv(OUT / f"check-{attr(args)}-key.csv")}
    if any(v not in ("yes", "no") for v in rated.values()):
        raise SystemExit("Every row needs `rater` set to yes or no.")
    p = jev_labels(attr(args))
    n_yes, n_no = sum(v >= 0.5 for v in p.values()), sum(v < 0.5 for v in p.values())
    yes_rated = [rated[d] == "yes" for d in rated if key[d]]
    no_rated = [rated[d] == "yes" for d in rated if not key[d]]
    precision = sum(yes_rated) / len(yes_rated)
    missed_rate = sum(no_rated) / len(no_rated) if no_rated else 0.0
    est_recall = precision * n_yes / (precision * n_yes + missed_rate * n_no) if n_yes else float("nan")
    lo, hi = wilson(sum(yes_rated), len(yes_rated))
    print(f"  precision {precision:.2f} [{lo:.2f}, {hi:.2f}] on {len(yes_rated)} Jev-yes products; "
          f"{missed_rate:.0%} of {len(no_rated)} Jev-no products were rated yes; estimated recall {est_recall:.2f}")
    if precision < 0.9:
        # The README's rule. Exiting non-zero stops `relevancy run` before the paid patch.
        raise SystemExit("  Precision is under 0.9. Stop: filtering on these labels would do more harm than good.")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if not n:
        return float("nan"), float("nan")
    p = k / n
    centre, half = (p + z * z / (2 * n)), z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return (centre - half) / (1 + z * z / n), (centre + half) / (1 + z * z / n)


def run_patch(args):
    a, p = attr(args), jev_labels(attr(args))
    items, tp = sorted(p.items()), client()
    for i in range(0, len(items), 5_000):
        chunk = items[i:i + 5_000]
        tp.tpuf.namespace(NAMESPACE).write(
            patch_columns={"id": [d for d, _ in chunk], a: [v >= 0.5 for _, v in chunk], f"{a}_p": [v for _, v in chunk]},
            schema={a: {"type": "bool"}, f"{a}_p": {"type": "float"}})
    print(f"  patched {a} onto {len(items):,} products in {NAMESPACE}")


def run_run(args):
    a, tp = attr(args), client()
    queries = read_csv(OUT / f"queries-{a}.csv")
    variants = {"hybrid": lambda t: hybrid_queries(t),
                "attr-soft": lambda t: hybrid_queries(t) + hybrid_queries(t, filters=(a, "Eq", True)),
                "attr-filter": lambda t: hybrid_queries(t, filters=(a, "Eq", True))}
    rows = []
    for i, q in enumerate(queries):
        order = list(variants)[i % 3:] + list(variants)[:i % 3]  # take turns, so cache warmth hits all alike
        for v in order:
            (hits,) = tp.multi_query(NAMESPACE, variants[v](q["text"]), rerank_by=("RRF", {}), limit=50,
                                     run_id=f"attr-{a}-{v}", query_id=q["query_id"])
            rows.append({"variant": v, "query_id": q["query_id"], "ids": " ".join(hits["id"].to_pylist())})
    write_csv(OUT / f"runs-{a}.csv", rows)


def run_report(args):
    with read_only() as con:
        human = labels(con)
    for a in [PRIMARY] + [x for x in ATTRIBUTES if x != PRIMARY]:
        path = OUT / f"runs-{a}.csv"
        if not path.exists():
            continue
        by = defaultdict(dict)
        for r in read_csv(path):
            by[r["variant"]][r["query_id"]] = r["ids"].split()
        role = "primary" if a == PRIMARY else "secondary, Bonferroni 97.5%"
        print(f"\n## {a} ({role}), {len(by['hybrid'])} queries")
        for v in ("attr-soft", "attr-filter"):
            for m in ("nDCG@10", "condensed", "R@50"):
                sa, sb = per_query(by[v], human, m), per_query(by["hybrid"], human, m)
                if a == PRIMARY:
                    print(f"  {v:<12} {m:<10} {compare(sa, sb)}")
                else:
                    diffs = [sa[q] - sb[q] for q in sorted(sa.keys() & sb.keys())]
                    lo, hi = interval(diffs, 0.975)
                    print(f"  {v:<12} {m:<10} {mean(sa):.3f} vs {mean(sb):.3f}: {paired_delta(sa, sb)[0]:+.3f} "
                          f"[{lo:+.3f}, {hi:+.3f}] n={len(diffs)}")


if __name__ == "__main__":
    commands = {"find": run_find, "label": run_label, "check-export": run_check_export, "precision": run_precision,
                "patch": run_patch, "run": run_run, "report": run_report}
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=commands)
    ap.add_argument("--attribute", choices=ATTRIBUTES, default=PRIMARY)
    args = ap.parse_args()
    commands[args.command](args)
