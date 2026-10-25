"""Filter cascades on ESCI products: can Jev's yes-probability decide the clear cases?

A cascade answers a yes/no condition over a table: the cheap first stage decides
what it's sure of, and an LLM decides the rest. The goal is a recall (and
precision) target at the fewest LLM calls.

  pool      Paid (Jev). A Noul per condition for a fixed random 20,000 products.
  sample    Free. 4,000 products per condition: up to 2,000 Jev-yes, the rest at
            random from the pool, so rare conditions have positives to score.
  oracle    Paid (LLM). Ground truth from a strong model (ORACLE_MODEL).
  escalate  Paid (LLM). The model the cascade escalates to (ESCALATE_MODEL), a
            different one from the oracle.
  embed     Paid (turbopuffer). The baseline first stage: vector similarity
            between the condition and each product.
  report    Free. Jev's calibration error against the oracle, then each first
            stage's cascade: cutoffs picked on 2,000 labels, tested on the other
            2,000, over 20 random splits, at recall and precision targets of 0.9
            and 0.95. Reports the share sent to the LLM and how often the target
            is missed.

SemBench (Part A in the README) runs in SemBench's own repo and environment,
so it isn't automated here.

  uv run experiments/database-semantic-operators/filters.py pool
"""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
from typesafe_sdk import Noul

from relevancy import jev, llm_judge
from relevancy.study import NAMESPACE, doc_texts, ece, read_csv, write_csv
from relevancy.tpuf import client
from relevancy.warehouse import ROOT, read_only

OUT = Path(__file__).resolve().parent / "results"
DATA = ROOT / "data" / "filters"
POOL, SAMPLE, JEV_YES = 20_000, 4_000, 2_000
SPLITS, TARGETS = 20, (0.9, 0.95)
ORACLE = os.environ.get("ORACLE_MODEL", "openai/gpt-4o")
ESCALATE = os.environ.get("ESCALATE_MODEL", "openai/gpt-4o-mini")
CONDITIONS = {
    "replacement_part": "Is this a replacement part, refill, or accessory for another product?",
    "for_children": "Is this product made for children?",
    "has_electronics": "Does this product contain electronics?",
    "set_or_bundle": "Is this product sold as a set or bundle of several items?",
    "clothing": "Is this product clothing, shoes, or something worn?",
    "needs_batteries": "Does this product need batteries?",
}
LABEL = """You label products in a shopping catalogue. Answer the question about the product with yes or no.
Return JSON: {"answer": true} for yes or {"answer": false} for no."""


def pool_ids() -> list[str]:
    path = DATA / "pool.json"
    if path.exists():
        return json.loads(path.read_text())
    ids = sorted(client().scan(NAMESPACE, columns=[]).read_all()["id"].to_pylist())
    picked = random.Random(42).sample(ids, min(POOL, len(ids)))
    DATA.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(picked))
    return picked


def run_pool(args):
    ids = pool_ids()
    with read_only() as con:
        texts = doc_texts(con, ids)
    judge = jev.client()
    for cond, question in CONDITIONS.items():
        path = DATA / f"jev-{cond}.json"
        if path.exists():
            continue
        answers = jev.ask(judge, "Label products in a shopping catalogue.",
                          {d: Noul(instructions=f"{question}\n\nProduct: {texts.get(d, '')}") for d in ids})
        path.write_text(json.dumps({d: jev.yes_probability(a) for d, a in answers.items()}))
        print(f"  {cond}: {sum(jev.yes_probability(a) >= 0.5 for a in answers.values()):,} of {len(ids):,} Jev-yes")


def jev_p(cond):
    return json.loads((DATA / f"jev-{cond}.json").read_text())


def run_sample(args):
    for cond in CONDITIONS:
        p, rnd = jev_p(cond), random.Random(cond)
        yes = sorted(d for d, v in p.items() if v >= 0.5)
        oversampled = rnd.sample(yes, min(JEV_YES, len(yes)))
        rest = sorted(set(p) - set(oversampled))
        drawn = rnd.sample(rest, min(SAMPLE - len(oversampled), len(rest)))
        # `random` marks the second draw; calibration is measured on those only.
        write_csv(OUT / f"sample-{cond}.csv", [{"id": d, "jev_p": round(p[d], 4), "random": ""} for d in oversampled]
                  + [{"id": d, "jev_p": round(p[d], 4), "random": "yes"} for d in drawn])


def llm_labels(model, name):
    http = llm_judge.client()
    for cond, question in CONDITIONS.items():
        path = DATA / f"{name}-{cond}.json"
        if path.exists():
            continue
        ids = [r["id"] for r in read_csv(OUT / f"sample-{cond}.csv")]
        with read_only() as con:
            texts = doc_texts(con, ids)
        out = {}
        for d in ids:
            raw = llm_judge.complete(http, model, LABEL, f"Question: {question}\n\nProduct: {texts.get(d, '')}",
                                     max_tokens=20, json_mode=True)
            try:
                out[d] = bool(json.loads(raw or "{}")["answer"])
            except (json.JSONDecodeError, KeyError, TypeError):
                out[d] = None
        path.write_text(json.dumps(out))
        print(f"  {name} {cond}: {sum(v is True for v in out.values()):,} yes, {sum(v is None for v in out.values())} failed")


def run_oracle(args):
    llm_labels(ORACLE, "oracle")


def run_escalate(args):
    llm_labels(ESCALATE, "escalate")


def run_embed(args):
    tp = client()
    for cond, question in CONDITIONS.items():
        ids = [r["id"] for r in read_csv(OUT / f"sample-{cond}.csv")]
        sims = {}
        for i in range(0, len(ids), 1_000):
            chunk = ids[i:i + 1_000]
            hits = tp.query(NAMESPACE, ("text", "ANN", ("Embed", question)), top_k=len(chunk), filters=("id", "In", chunk))
            sims |= {d: 1 - s for d, s in zip(hits["id"].to_pylist(), hits["score"].to_pylist())}  # cosine distance
        (DATA / f"embed-{cond}.json").write_text(json.dumps(sims))


def cascade(score, esc, truth, cal, test, target):
    """Pick lo/hi on `cal` to minimize escalations with recall and precision >= target; score them on `test`."""
    qs = np.unique(np.quantile(score[cal], np.linspace(0, 1, 41)))
    best = None
    for lo in qs:
        for hi in qs[qs >= lo]:
            pred = np.where(score[cal] >= hi, True, np.where(score[cal] < lo, False, esc[cal]))
            tp_ = (pred & truth[cal]).sum()
            rec = tp_ / max(truth[cal].sum(), 1)
            prec = tp_ / max(pred.sum(), 1)
            share = ((score[cal] >= lo) & (score[cal] < hi)).mean()
            if rec >= target and prec >= target and (best is None or share < best[0]):
                best = (share, lo, hi)
    if best is None:  # nothing meets the target: send everything to the LLM
        best = (1.0, -np.inf, np.inf)
    _, lo, hi = best
    pred = np.where(score[test] >= hi, True, np.where(score[test] < lo, False, esc[test]))
    tp_ = (pred & truth[test]).sum()
    rec, prec = tp_ / max(truth[test].sum(), 1), tp_ / max(pred.sum(), 1)
    return ((score[test] >= lo) & (score[test] < hi)).mean(), rec, prec


def run_report(args):
    rows = []
    for cond in CONDITIONS:
        sample = read_csv(OUT / f"sample-{cond}.csv")
        oracle = json.loads((DATA / f"oracle-{cond}.json").read_text())
        esc = json.loads((DATA / f"escalate-{cond}.json").read_text())
        emb = json.loads((DATA / f"embed-{cond}.json").read_text())
        p = jev_p(cond)
        ids = [r["id"] for r in sample if oracle.get(r["id"]) is not None and esc.get(r["id"]) is not None]
        truth = np.array([oracle[d] for d in ids])
        escalated = np.array([esc[d] for d in ids])
        drawn = {r["id"] for r in sample if r["random"] == "yes"}
        rand = [i for i, d in enumerate(ids) if d in drawn]
        cal_err = ece([p[ids[i]] for i in rand], [bool(truth[i]) for i in rand]) if rand else float("nan")
        llm_rec = (escalated & truth).sum() / max(truth.sum(), 1)
        print(f"\n## {cond}: {len(ids):,} labeled, {truth.sum():,} positive")
        print(f"  Jev calibration error on the random part: {cal_err:.3f}; escalation LLM alone: recall {llm_rec:.3f}")
        stages = {"jev": np.array([p[d] for d in ids]), "embedding": np.array([emb.get(d, 0.0) for d in ids])}
        for target in TARGETS:
            for stage, score in stages.items():
                res = []
                for s in range(SPLITS):
                    order = np.random.default_rng(s).permutation(len(ids))
                    cal, test = order[:len(ids) // 2], order[len(ids) // 2:]
                    res.append(cascade(score, escalated, truth, cal, test, target))
                share = float(np.mean([r[0] for r in res]))
                missed = float(np.mean([r[1] < target or r[2] < target for r in res]))
                print(f"  target {target}: {stage:<9} sent to LLM {share:.0%}, target missed in {missed:.0%} of {SPLITS} splits")
                rows.append({"condition": cond, "target": target, "stage": stage, "llm_share": round(share, 4),
                             "missed": round(missed, 4), "calibration_error": round(cal_err, 4)})
    write_csv(OUT / "cascades.csv", rows)


if __name__ == "__main__":
    commands = {"pool": run_pool, "sample": run_sample, "oracle": run_oracle, "escalate": run_escalate,
                "embed": run_embed, "report": run_report}
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=commands)
    commands[ap.parse_args().command](None)
