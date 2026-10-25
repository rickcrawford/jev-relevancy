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
DELTAS = (0.0, 0.01, 0.02, 0.03, 0.05)  # safety margins added to the target when picking cutoffs
MISS_RATE = 0.10  # stages are compared at a margin that misses the target in at most 10% of splits
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
    """A random draw from the whole pool first, then extra Jev-yes products so rare conditions have
    positives. Each row carries its stratum's weight (pool count / sample count, by Jev yes or no):
    within a stratum the sampled products are a uniform random subset, so weighted sums estimate the
    pool, and the cascades aren't scored on a sample shaped by Jev's own answers."""
    for cond in CONDITIONS:
        p, rnd = jev_p(cond), random.Random(cond)
        pool = sorted(p)
        n_random = max(0, SAMPLE - JEV_YES)
        drawn = set(rnd.sample(pool, min(n_random, len(pool))))
        yes_left = sorted(d for d in pool if p[d] >= 0.5 and d not in drawn)
        extra = set(rnd.sample(yes_left, min(SAMPLE - len(drawn), len(yes_left))))
        picked = drawn | extra
        strata = {"yes": [d for d in pool if p[d] >= 0.5], "no": [d for d in pool if p[d] < 0.5]}
        weight = {}
        for members in strata.values():
            in_sample = [d for d in members if d in picked]
            for d in in_sample:
                weight[d] = len(members) / len(in_sample)
        write_csv(OUT / f"sample-{cond}.csv",
                  [{"id": d, "jev_p": round(p[d], 4), "random": "yes" if d in drawn else "", "weight": round(weight[d], 6)}
                   for d in sorted(picked)])
        print(f"  {cond}: {len(drawn):,} random and {len(extra):,} extra Jev-yes")


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
        missing = len(set(ids) - set(sims))
        if missing:
            print(f"  {cond}: {missing} products came back without a similarity; they score as the lowest seen")
        (DATA / f"embed-{cond}.json").write_text(json.dumps(sims))


def run_check_export(args):
    """100 products per condition for a person to label, with the oracle's answer hidden, to check the oracle."""
    rows = []
    for cond, question in CONDITIONS.items():
        ids = [r["id"] for r in read_csv(OUT / f"sample-{cond}.csv") if r["random"] == "yes"][:100]
        with read_only() as con:
            texts = doc_texts(con, ids)
        rows += [{"condition": cond, "id": d, "question": question, "product": texts.get(d, ""), "rater": ""} for d in ids]
    write_csv(OUT / "oracle-check.csv", rows)
    print("  fill in `rater` with yes or no; report then prints the oracle's agreement with the person")


def cascade(score, esc, truth, w, cal, test, target):
    """Pick lo/hi on `cal` for the fewest escalations with weighted recall >= target (the README's goal);
    return the weighted share sent to the LLM and the recall on `test`."""
    def recall_share(idx, lo, hi):
        pred = np.where(score[idx] >= hi, True, np.where(score[idx] < lo, False, esc[idx]))
        tp_ = (w[idx] * (pred & truth[idx])).sum()
        rec = tp_ / max((w[idx] * truth[idx]).sum(), 1e-12)
        share = (w[idx] * ((score[idx] >= lo) & (score[idx] < hi))).sum() / w[idx].sum()
        return rec, share

    qs = np.unique(np.quantile(score[cal], np.linspace(0, 1, 41)))
    best = None
    for lo in qs:
        for hi in qs[qs >= lo]:
            rec, share = recall_share(cal, lo, hi)
            if rec >= target and (best is None or share < best[0]):
                best = (share, lo, hi)
    if best is None:  # nothing meets the target: send everything to the LLM
        best = (1.0, -np.inf, np.inf)
    _, lo, hi = best
    rec, share = recall_share(test, lo, hi)
    return share, rec


def run_report(args):
    rows = []
    for cond in CONDITIONS:
        sample = read_csv(OUT / f"sample-{cond}.csv")
        oracle = json.loads((DATA / f"oracle-{cond}.json").read_text())
        esc = json.loads((DATA / f"escalate-{cond}.json").read_text())
        emb = json.loads((DATA / f"embed-{cond}.json").read_text())
        p = jev_p(cond)
        ids = [r["id"] for r in sample if oracle.get(r["id"]) is not None and esc.get(r["id"]) is not None]
        weight = {r["id"]: float(r["weight"]) for r in sample}
        truth = np.array([oracle[d] for d in ids])
        escalated = np.array([esc[d] for d in ids])
        w = np.array([weight[d] for d in ids])
        drawn = {r["id"] for r in sample if r["random"] == "yes"}
        rand = [i for i, d in enumerate(ids) if d in drawn]  # a plain random sample of the pool
        cal_err = ece([p[ids[i]] for i in rand], [bool(truth[i]) for i in rand]) if rand else float("nan")
        llm_rec = (w * (escalated & truth)).sum() / max((w * truth).sum(), 1e-12)
        print(f"\n## {cond}: {len(ids):,} labeled, {truth.sum():,} positive "
              f"(about {(w * truth).sum() / w.sum():.1%} of the pool)")
        print(f"  Jev calibration error on the random part: {cal_err:.3f}; escalation LLM alone: recall {llm_rec:.3f}")
        lowest = min(emb.values()) if emb else 0.0
        stages = {"jev": np.array([p[d] for d in ids]), "embedding": np.array([emb.get(d, lowest) for d in ids])}
        for target in TARGETS:
            picked = {}
            for stage, score in stages.items():
                # Cutoffs are picked on each calibration half with a safety margin; the stages are
                # compared at the smallest margin that keeps their miss rate at or under MISS_RATE.
                for delta in DELTAS:
                    res = []
                    for s in range(SPLITS):
                        order = np.random.default_rng(s).permutation(len(ids))
                        cal, test = order[:len(ids) // 2], order[len(ids) // 2:]
                        res.append(cascade(score, escalated, truth, w, cal, test, min(1.0, target + delta)))
                    share = float(np.mean([x[0] for x in res]))
                    missed = float(np.mean([x[1] < target for x in res]))
                    rows.append({"condition": cond, "target": target, "stage": stage, "margin": delta,
                                 "llm_share": round(share, 4), "missed": round(missed, 4),
                                 "calibration_error": round(cal_err, 4)})
                    if missed <= MISS_RATE and stage not in picked:
                        picked[stage] = (share, missed, delta)
                got = picked.get(stage)
                print(f"  target {target}: {stage:<9} " + (f"sent to LLM {got[0]:.0%} (missed {got[1]:.0%} of "
                      f"{SPLITS} splits, margin {got[2]})" if got else f"no margin kept misses at or under {MISS_RATE:.0%}"))
            if len(picked) == 2:
                better = picked["jev"][0] < picked["embedding"][0]
                print(f"    {'Jev sends fewer to the LLM' if better else 'Jev does not send fewer to the LLM'} "
                      "at a matched miss rate")
    write_csv(OUT / "cascades.csv", rows)
    check = OUT / "oracle-check.csv"
    if check.exists():
        rated = [r for r in read_csv(check) if r["rater"].strip().lower() in ("yes", "no")]
        if rated:
            oracle_all = {c: json.loads((DATA / f"oracle-{c}.json").read_text()) for c in CONDITIONS}
            agree = [oracle_all[r["condition"]].get(r["id"]) == (r["rater"].strip().lower() == "yes") for r in rated]
            print(f"\n  oracle agrees with the person on {sum(agree) / len(agree):.0%} of {len(rated)} checked labels")


if __name__ == "__main__":
    commands = {"pool": run_pool, "sample": run_sample, "oracle": run_oracle, "escalate": run_escalate,
                "embed": run_embed, "check-export": run_check_export, "report": run_report}
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=commands)
    commands[ap.parse_args().command](None)
