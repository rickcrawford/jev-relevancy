"""FollowIR: does Jev change its ranking the right way when the instruction changes?

Each query comes twice: with its original instruction (-og) and an altered one
(-changed) that makes some documents irrelevant. p-MRR rewards moving exactly
those documents down. Computed the way MTEB does, from the datasets MTEB pins.

  load    Free. Robust04, Core17, and News21 from Hugging Face (MTEB's copies).
  rerank  Paid (Jev). Each query's candidate documents, scored with the
          instruction in Jev's state.
  report  Free. p-MRR per collection with a bootstrap interval over queries.
          Counts if the interval sits above zero.

The text comes from licensed news collections; say so next to any result.

  uv run experiments/benchmark-instructions-stark/followir.py load
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from relevancy import jev
from relevancy.rubric import Rubric
from relevancy.warehouse import ROOT

OUT = Path(__file__).resolve().parent / "results"
DATA = ROOT / "data" / "followir"
TASKS = {
    "robust04": ("mteb/Robust04InstructionRetrieval", "0495a5cb69aa8fa5bca81bd56ac248911c19eeb6"),
    "core17": ("mteb/Core17InstructionRetrieval", "67d8733dd0691f08f34ed9dda5941578cba8d91c"),
    "news21": ("mteb/News21InstructionRetrieval", "c6fffeb9cdd95c1ffa81c62c687b48b56352872b"),
}
RUBRIC = Rubric(
    name="followir", state="Search query: {query}",
    question="How relevant is this document to the query, following the instructions exactly?", item="Document",
    levels=("Irrelevant, or ruled out by the instructions", "Related, but doesn't meet the instructions",
            "Relevant, with minor gaps against the instructions", "Relevant and meets every instruction"),
    llm_prompt="")


def _split(path, config, revision):
    from datasets import get_dataset_split_names
    splits = get_dataset_split_names(path, config_name=config, revision=revision)
    return "test" if "test" in splits else splits[0]


def _load(path, config, revision):
    from datasets import load_dataset
    return load_dataset(path, config, split=_split(path, config, revision), revision=revision)


def run_load(args):
    DATA.mkdir(parents=True, exist_ok=True)
    for task, (path, rev) in TASKS.items():
        corpus = {str(r.get("_id", r.get("id"))): (r.get("title") or "") + " " + r["text"]
                  for r in _load(path, "corpus", rev)}
        queries = {str(r.get("_id", r.get("id"))): r["text"] for r in _load(path, "queries", rev)}
        # Ids as strings, as MTEB casts them; otherwise int ids never match and p-MRR comes out 0.
        instructions = {str(r["query-id"]): r["instruction"] for r in _load(path, "instruction", rev)}
        top = {str(r["query-id"]): [str(d) for d in r["corpus-ids"]] for r in _load(path, "top_ranked", rev)}
        try:
            qrel_rows = _load(path, "default", rev)
        except Exception:  # noqa: BLE001  (older revisions name it "qrels")
            qrel_rows = _load(path, "qrels", rev)
        qrels = defaultdict(dict)
        for r in qrel_rows:
            qrels[str(r["query-id"])][str(r["corpus-id"])] = int(r["score"])
        changed = {str(r["query-id"]): [str(d) for d in r["corpus-ids"]] for r in _load(path, "qrel_diff", rev)}
        needed = {d for ids in top.values() for d in ids}
        (DATA / f"{task}.json").write_text(json.dumps({
            "queries": queries, "instructions": instructions, "top": top, "qrels": qrels, "changed": changed,
            "corpus": {d: corpus[d][:4000] for d in needed if d in corpus}}))
        print(f"  {task}: {len(queries)} queries, {len(needed):,} candidate documents")


def run_rerank(args):
    judge = jev.client()
    for task in TASKS:
        path = DATA / f"{task}.json"
        if not path.exists():
            continue
        d = json.loads(path.read_text())
        results = {}
        for qid, ids in d["top"].items():
            text = f"{d['queries'][qid]}\nInstructions: {d['instructions'].get(qid, '')}"
            grades = jev.grade_products(judge, text, {i: d["corpus"].get(i, "") for i in ids}, RUBRIC)
            results[qid] = {i: g.expected for i, g in grades.items()}
        (OUT / f"followir-{task}.json").parent.mkdir(parents=True, exist_ok=True)
        (OUT / f"followir-{task}.json").write_text(json.dumps(results))
        print(f"  {task}: reranked {len(results)} queries")


def rank_of(scores: dict[str, float], doc: str) -> int:
    """Rank with ties broken by doc id, descending, as MTEB does."""
    ordered = sorted(scores.items(), key=lambda x: (x[1], x[0]), reverse=True)
    return next((i + 1 for i, (d, _) in enumerate(ordered) if d == doc), len(ordered) + 1)


def rank_score(og: int, new: int) -> float:
    return (1 / og) / (1 / new) - 1 if og >= new else 1 - (1 / new) / (1 / og)


def pmrr_by_query(results, changed) -> dict[str, float]:
    out = {}
    for qid, docs in changed.items():
        og, new = results.get(f"{qid}-og"), results.get(f"{qid}-changed")
        if og is None or new is None or not docs:
            continue
        out[qid] = sum(rank_score(rank_of(og, d), rank_of(new, d)) for d in docs) / len(docs)
    return out


def run_report(args):
    for task in TASKS:
        path = OUT / f"followir-{task}.json"
        if not path.exists():
            continue
        changed = json.loads((DATA / f"{task}.json").read_text())["changed"]
        per_q = list(pmrr_by_query(json.loads(path.read_text()), changed).values())
        rnd = random.Random(0)
        means = sorted(sum(rnd.choices(per_q, k=len(per_q))) / len(per_q) for _ in range(1000))
        m, lo, hi = sum(per_q) / len(per_q), means[24], means[974]
        print(f"  {task:<9} p-MRR {100 * m:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}] over {len(per_q)} queries"
              f" ({'above zero' if lo > 0 else 'not above zero'})")


if __name__ == "__main__":
    commands = {"load": run_load, "rerank": run_rerank, "report": run_report}
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=commands)
    commands[ap.parse_args().command](None)
