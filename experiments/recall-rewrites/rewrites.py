"""Query rewrites and document expansion: do they find products nothing else finds?

Part A, rewrites:
  rewrite          Paid (LLM). Three rephrasings per sampled query, prompt fixed.
  drift            Paid (Jev). A Noul per rewrite: does it ask for something
                   different from the original? Those are dropped in `checked`.
  run              Paid. hybrid, rewrites (original plus all three), and checked
                   (original plus the ones that passed), each fused by weighted RRF
                   so the original query keeps half the total weight; then Voyage
                   over each top 50.
Part B, expansion (--scope smoke for plumbing; only full decides):
  expand           Paid (LLM), resumable. Five queries each product would answer.
  patch-expansion  Paid. Writes them as a full-text `expansion` field.
  run-expansion    Paid. hybrid plus BM25 over `expansion`; then Voyage.
Both:
  report           Free. Recall@50 and nDCG@10 after Voyage against hybrid, on all
                   queries and on recall-fusion's vocabulary-miss queries.

  uv run experiments/recall-rewrites/rewrites.py rewrite
"""

import argparse
import json
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow as pa
from typesafe_sdk import Noul

from relevancy import jev, llm_judge
from relevancy.metrics import paired_delta
from relevancy.study import (
    compare,
    doc_texts,
    labels,
    part_ids,
    per_query,
    read_csv,
    sampled,
    write_csv,
    write_part,
)
from relevancy.tpuf import client, hybrid_queries, namespace_name, rerank
from relevancy.warehouse import ROOT, read_only

OUT = Path(__file__).resolve().parent / "results"
DATA = ROOT / "data" / "rewrites"
MODEL = os.environ.get("REWRITE_MODEL", "openai/gpt-4o-mini")
REWRITE = """You rephrase shopping queries the way other shoppers might type them, keeping exactly what is asked for.
Use different words where you can (synonyms, the product's common names), and keep every brand, size, color, and
quantity. Return JSON: {"rewrites": ["...", "...", "..."]} with three rephrasings."""
EXPAND = """You write the queries a shopper would type to find this product. Vary the wording: common names,
what it's for, and its key attributes. Return JSON: {"queries": ["...", "...", "...", "...", "..."]} with five."""
CHUNK = 500
NON_INFERIOR = 0.01  # nDCG@10 after Voyage "doesn't fall" if the lower bound is above -0.01, fixed in advance
MISS_OVERLAP = 0.5   # a not-found product counts as a vocabulary miss when under half the query's words are in its title
SCHEMA = pa.schema([("id", pa.string()), ("expansion", pa.string())])


def run_rewrite(args):
    with read_only() as con:
        queries = sampled(con)
    http, rows = llm_judge.client(), []
    for qid, text in queries:
        started = time.perf_counter()  # the LLM call is what a rewrite adds before search
        raw = llm_judge.complete(http, MODEL, REWRITE, f"Query: {text}", max_tokens=200, json_mode=True)
        ms = (time.perf_counter() - started) * 1000
        try:
            rewrites = [str(r) for r in json.loads(raw or "{}").get("rewrites", [])][:3]
        except (json.JSONDecodeError, AttributeError):
            rewrites = []
        rows.append({"query_id": qid, "text": text, "rewrites": json.dumps(rewrites), "ok": bool(rewrites),
                     "ms": round(ms, 1)})
    write_csv(OUT / "rewrites.csv", rows)
    ms = sorted(r["ms"] for r in rows)
    print(f"  {sum(not r['ok'] for r in rows)} of {len(rows)} queries got no rewrites (they run as plain hybrid); "
          f"LLM call p50 {ms[len(ms) // 2]:.0f} ms, p99 {ms[min(len(ms) - 1, int(0.99 * len(ms)))]:.0f} ms")


def run_drift(args):
    rows = read_csv(OUT / "rewrites.csv")
    qs = {(r["query_id"], i): Noul(instructions=f"Original shopping query: {r['text']}\nRewrite: {rw}\n\n"
                                                "Does the rewrite ask for something different from the original?")
          for r in rows for i, rw in enumerate(json.loads(r["rewrites"]))}
    answers = jev.ask(jev.client(), "Compare rewritten shopping queries with the originals.", qs)
    for r in rows:
        n = len(json.loads(r["rewrites"]))
        r["kept"] = json.dumps([i for i in range(n) if jev.yes_probability(answers[(r["query_id"], i)]) < 0.5])
    write_csv(OUT / "rewrites.csv", rows)
    kept = sum(len(json.loads(r["kept"])) for r in rows)
    print(f"  kept {kept} of {sum(len(json.loads(r['rewrites'])) for r in rows)} rewrites")


def fused(text, rewrites):
    """Original's two subqueries plus two per rewrite; weights give the original half the total."""
    subs = hybrid_queries(text) + [s for rw in rewrites for s in hybrid_queries(rw)]
    w_orig = max(1.0, len(rewrites))  # 2 x w_orig == 2 x len(rewrites) x 1
    return subs, [w_orig, w_orig] + [1.0] * (2 * len(rewrites))


def rerank_all(tp, ns, text_of, ranked: dict[str, dict[str, list[str]]], prefix: str):
    rows = []
    for v, by_q in ranked.items():
        for qid, ids in by_q.items():
            rows.append({"variant": v, "query_id": qid, "ids": " ".join(ids)})
            hits = rerank(tp, ns, "voyage/rerank-3", text_of[qid], ids, run_id=f"{prefix}-{v}+voyage", query_id=qid)
            rows.append({"variant": f"{v}+voyage", "query_id": qid, "ids": " ".join(hits["id"].to_pylist())})
    return rows


def run_run(args):
    rows = read_csv(OUT / "rewrites.csv")
    tp, ns = client(), namespace_name("esci-us", args.scope)
    ranked: dict[str, dict[str, list[str]]] = defaultdict(dict)
    for i, r in enumerate(rows):
        rw = json.loads(r["rewrites"])
        kept = [rw[k] for k in json.loads(r.get("kept") or "[]")] if r.get("kept") else None
        variants = {"hybrid": [], "rewrites": rw}
        if kept is not None:
            variants["checked"] = kept
        order = list(variants)[i % len(variants):] + list(variants)[:i % len(variants)]
        for v in order:
            subs, weights = fused(r["text"], variants[v])
            (hits,) = tp.multi_query(ns, subs, rerank_by=("RRF", {"weights": weights}), limit=50,
                                     run_id=f"rewrites-{v}", query_id=r["query_id"])
            ranked[v][r["query_id"]] = hits["id"].to_pylist()
    write_csv(OUT / f"runs-rewrites-{args.scope}.csv",
              rerank_all(tp, ns, {r["query_id"]: r["text"] for r in rows}, ranked, "rewrites"))


def run_expand(args):
    folder = DATA / f"expansion-{args.scope}"
    done = part_ids(folder)
    tp, http = client(), llm_judge.client()
    ids = [d for d in tp.scan(namespace_name("esci-us", args.scope), columns=[]).read_all()["id"].to_pylist()
           if d not in done]
    print(f"  {len(done):,} expanded, {len(ids):,} to go", flush=True)
    for i in range(0, len(ids), CHUNK):
        chunk = ids[i:i + CHUNK]
        with read_only() as con:
            texts = doc_texts(con, chunk)
        def one(d, texts=texts):
            raw = llm_judge.complete(http, MODEL, EXPAND, f"Product: {texts.get(d, '')}", max_tokens=200, json_mode=True)
            try:
                qs = [str(q) for q in json.loads(raw or "{}").get("queries", [])][:5]
            except (json.JSONDecodeError, AttributeError):
                qs = []
            return {"id": d, "expansion": " | ".join(qs)} if qs else None

        with ThreadPoolExecutor(8) as pool:
            got = list(pool.map(one, chunk))
        rows = [g for g in got if g]
        if len(rows) < len(chunk):
            print(f"  {len(chunk) - len(rows)} products got no expansion; a rerun retries them", flush=True)
        write_part(folder, rows, SCHEMA)  # failures aren't written, so they're retried
        print(f"  {len(done) + i + len(chunk):,} expanded", flush=True)


def run_patch_expansion(args):
    import duckdb
    rows = duckdb.sql(f"SELECT id, expansion FROM '{DATA}/expansion-{args.scope}/*.parquet'").fetchall()
    tp, ns = client(), namespace_name("esci-us", args.scope)
    english = {"language": "english", "stemming": True}
    try:
        for i in range(0, len(rows), 5_000):
            chunk = rows[i:i + 5_000]
            tp.tpuf.namespace(ns).write(
                patch_columns={"id": [r[0] for r in chunk], "expansion": [r[1] for r in chunk]},
                schema={"expansion": {"type": "string", "full_text_search": english}})
    except Exception as e:  # noqa: BLE001  (the API's answer is the finding)
        raise SystemExit(f"turbopuffer refused the patch ({e}). A full-text field may need the namespace written "
                         "again with it; load a copy with the field instead.") from None
    print(f"  patched `expansion` onto {len(rows):,} products in {ns}")


def run_run_expansion(args):
    with read_only() as con:
        queries = sampled(con)
    tp, ns = client(), namespace_name("esci-us", args.scope)
    ranked: dict[str, dict[str, list[str]]] = defaultdict(dict)
    for i, (qid, text) in enumerate(queries):
        variants = {"hybrid": hybrid_queries(text),
                    "expansion": hybrid_queries(text) + [{"rank_by": ("expansion", "BM25", text), "top_k": 50}]}
        for v in (list(variants) if i % 2 == 0 else list(variants)[::-1]):
            (hits,) = tp.multi_query(ns, variants[v], rerank_by=("RRF", {}), limit=50, run_id=f"expansion-{v}",
                                     query_id=qid)
            ranked[v][qid] = hits["id"].to_pylist()
    write_csv(OUT / f"runs-expansion-{args.scope}.csv", rerank_all(tp, ns, dict(queries), ranked, "expansion"))


def run_report(args):
    with read_only() as con:
        human = labels(con)
    misses = set()
    diag = ROOT / "experiments" / "recall-fusion" / "results" / "recall_diagnose.csv"
    if diag.exists():
        # The README's vocabulary misses: products nothing found, whose titles share few of the query's words.
        misses = {r["query_id"] for r in read_csv(diag)
                  if r["bucket"] == "not found" and float(r["title_overlap"]) < MISS_OVERLAP}
    rw = OUT / "rewrites.csv"
    if rw.exists():
        ms = sorted(float(r["ms"]) for r in read_csv(rw) if r.get("ms"))
        if ms:
            print(f"  rewrite latency before search: p50 {ms[len(ms) // 2]:.0f} ms, "
                  f"p99 {ms[min(len(ms) - 1, int(0.99 * len(ms)))]:.0f} ms")
    for part in ("rewrites", "expansion"):
        path = OUT / f"runs-{part}-{args.scope}.csv"
        if not path.exists():
            continue
        by = defaultdict(dict)
        for r in read_csv(path):
            by[r["variant"]][r["query_id"]] = r["ids"].split()
        print(f"\n## {part} ({args.scope})")
        for v in [x for x in by if not x.endswith("+voyage") and x != "hybrid"]:
            for subset, only in (("all", None), ("vocabulary misses", misses or None)):
                if subset != "all" and not only:
                    continue
                print(f"  {v} vs hybrid, {subset}:")
                a, b = per_query(by[v + "+voyage"], human, only=only), per_query(by["hybrid+voyage"], human, only=only)
                _, lo, _, _ = paired_delta(a, b)
                print(f"    R@50              {compare(per_query(by[v], human, 'R@50', only), per_query(by['hybrid'], human, 'R@50', only))}")
                print(f"    nDCG@10 + Voyage  {compare(a, b)}  "
                      f"{'does not fall' if lo > -NON_INFERIOR else 'may fall'} (lower bound vs -{NON_INFERIOR})")
        if part == "expansion":
            print("  Caveat: the LLM may have seen ESCI's queries in training, which would flatter expansion.")


if __name__ == "__main__":
    commands = {"rewrite": run_rewrite, "drift": run_drift, "run": run_run, "expand": run_expand,
                "patch-expansion": run_patch_expansion, "run-expansion": run_run_expansion, "report": run_report}
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=commands)
    ap.add_argument("--scope", choices=["smoke", "full"], default="full")
    args = ap.parse_args()
    commands[args.command](args)
