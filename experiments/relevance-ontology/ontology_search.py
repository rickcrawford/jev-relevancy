"""Ontology-matched search: does knowing a query's product category improve hybrid retrieval?

Every product and every sampled query gets a node in the Google Product
Taxonomy (about 5,600 categories, up to 7 levels). Tagging is two steps:
turbopuffer shortlists the 8 nearest taxonomy paths by vector search, then a
Jev Choice question picks one of them, or "none of these". Jev returns a
probability per candidate, so the probability under each parent category can
be added up, and an ambiguous query gets a broad category instead of a wrong
narrow one.

Products get a `category_path` attribute: their node and every ancestor. A
query's node then works as a filter at any depth:

  ("category_path", "Contains", node)

Variants, each the top 50 for the 500 sampled queries:
  hybrid             hybrid-rrf as the main runs do it, rerun in the same session
  onto-soft          hybrid's two subqueries, plus the same two limited to the
                     query's category, fused by RRF: in-category products get
                     extra votes, nothing is dropped
  onto-filter        hybrid limited to the query's category
  oracle-soft/filter the same, with the category from half of the query's
                     human-Exact products: a perfect query tagger, so an upper
                     bound. Scored on labels without that half, so the oracle
                     can't win by filtering to the products that chose it.
  hybrid+voyage      voyage/rerank-3 over hybrid's top 50
  onto-soft+voyage   voyage/rerank-3 over onto-soft's top 50
A query the tagger abstains on (no category holds 80%) runs as plain hybrid.

  uv run experiments/relevance-ontology/ontology_search.py taxonomy     # load the taxonomy into its own namespace
  uv run experiments/relevance-ontology/ontology_search.py tag-docs     # tag every product (resumable)
  uv run experiments/relevance-ontology/ontology_search.py patch        # write category_path to the product namespace
  uv run experiments/relevance-ontology/ontology_search.py tag-queries
  uv run experiments/relevance-ontology/ontology_search.py run
  uv run experiments/relevance-ontology/ontology_search.py judge        # Jev grades top-10 products humans didn't label
  uv run experiments/relevance-ontology/ontology_search.py report

Every command takes --scope smoke|full (default full). `patch` is the one
step that changes a namespace the main runs use: it adds an attribute and
leaves title, body, and text, so BM25 and vector scores don't change. The
test plan, and what would count as a win, is in README.md, next to this script.
"""

import argparse
import csv
import hashlib
import json
import math
import time
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
from typesafe_sdk import Choice

from relevancy import jev
from relevancy.judges import grade_pairs
from relevancy.metrics import paired_delta
from relevancy.rubric import load
from relevancy.tpuf import (
    EMBED_MODEL,
    client,
    hybrid_queries,
    namespace_name,
    requests_by_query,
    rerank,
)
from relevancy.warehouse import read_only

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "results"
DATA = ROOT / "data" / "ontology"
DS = "esci-us"
TAXONOMY_URL = "https://www.google.com/basepages/producttype/taxonomy-with-ids.en-US.txt"
TAXONOMY_NS = "google-product-taxonomy"
SHORTLIST = 8          # taxonomy candidates per product or query
DOC_MASS = 0.5         # a product gets the deepest category holding this much probability
QUERY_MASS = 0.8       # a query needs more: a wrong query category costs every result
CHUNK = 2_000          # products per tagging round, and per saved part file
JUDGE = "jev-batched:jev-1.13.0"
# Short text for the shortlist (only the title says much about category), and
# a longer one for Jev, still well under its request limit at 50 products.
SHORT = "left(concat_ws('. ', d.title, d.brand), 300)"
LONG = "concat_ws('. ', d.title, d.brand, d.color, left(d.body, 300))"
STATE = "Assign products and shopping queries to categories in the Google Product Taxonomy."
NONE = "None of these categories fits"
TAG_SCHEMA = pa.schema([("id", pa.string()), ("node", pa.string()), ("mass", pa.float64()),
                        ("none", pa.float64()), ("probs", pa.string())])


# -- the taxonomy -------------------------------------------------------------

def taxonomy_file() -> Path:
    path = DATA / "taxonomy-with-ids.en-US.txt"
    if not path.exists():
        DATA.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(TAXONOMY_URL, path)
    return path


def taxonomy() -> tuple[dict[str, str], dict[str, list[str]]]:
    """{node: "A > B > C"} and {node: [its ancestors' nodes, root first, then itself]}."""
    paths, by_path = {}, {}
    for line in taxonomy_file().read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        node, path = line.split(" - ", 1)
        paths[node], by_path[path] = path, node
    lineage = {node: [by_path[" > ".join(path.split(" > ")[:i])] for i in range(1, path.count(" > ") + 2)]
               for node, path in paths.items()}
    return paths, lineage


def settle(probs: dict[str, float], lineage: dict[str, list[str]], threshold: float) -> tuple[str | None, float]:
    """The deepest category holding `threshold` of the probability, counting its descendants."""
    mass: dict[str, float] = defaultdict(float)
    for node, p in probs.items():
        for anc in lineage.get(node, []):
            mass[anc] += p
    held = [n for n, m in mass.items() if m >= threshold]
    if not held:
        return None, max(mass.values(), default=0.0)
    best = max(held, key=lambda n: (len(lineage[n]), mass[n]))
    return best, mass[best]


def run_taxonomy(args) -> None:
    paths, _ = taxonomy()
    tp = client()
    table = pa.table({"id": list(paths), "text": list(paths.values())})
    stats = tp.upsert(TAXONOMY_NS, table, distance_metric="cosine_distance", run_id="taxonomy",
                      schema={"text": {"type": "string", "embed": EMBED_MODEL, "filterable": False}})
    print(f"{stats.rows:,} taxonomy nodes in {TAXONOMY_NS}")


# -- tagging ------------------------------------------------------------------

def shortlist(tp, texts: list[str], workers: int = 8) -> list[list[str]]:
    def one(text):
        return tp.query(TAXONOMY_NS, ("text", "ANN", ("Embed", text)), top_k=SHORTLIST)["id"].to_pylist()
    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(one, texts))


def choose(client_, item: str, texts: list[str], candidates: list[list[str]],
           paths: dict[str, str], workers: int = 8) -> list[dict[str, float]]:
    """One Jev Choice question per text, 50 to a request: {node or "none": probability}."""
    def batch(start):
        questions = {
            f"p{n}": Choice(
                instructions=f"Which category does this {item.lower()} belong to?\n\n{item}: {text}",
                criteria={f"c{i}": paths[c] for i, c in enumerate(cands)} | {"none": NONE})
            for n, (text, cands) in enumerate(zip(texts[start:start + jev.PER_REQUEST],
                                                  candidates[start:start + jev.PER_REQUEST]))
        }
        data = client_.system_one(state=STATE, questions=questions, model=jev.DEFAULT_MODEL).model_dump()
        out = []
        for n, cands in enumerate(candidates[start:start + jev.PER_REQUEST]):
            probs = data["answers"][f"p{n}"]["probabilities"]
            out.append({("none" if k == "none" else cands[int(k[1:])]): v for k, v in probs.items()})
        return out
    with ThreadPoolExecutor(workers) as pool:
        return [p for chunk in pool.map(batch, range(0, len(texts), jev.PER_REQUEST)) for p in chunk]


def tagged_ids(parts: Path) -> set[str]:
    if not any(parts.glob("*.parquet")):
        return set()
    return {r[0] for r in duckdb.sql(f"SELECT id FROM '{parts}/*.parquet'").fetchall()}


def run_tag_docs(args) -> None:
    paths, lineage = taxonomy()
    ns, parts = namespace_name(DS, args.scope), DATA / f"doc_tags-{args.scope}"
    parts.mkdir(parents=True, exist_ok=True)
    have = client().scan(ns, columns=[]).read_all()
    done = tagged_ids(parts)
    con = read_only()
    con.register("have", have)
    todo = [r for r in con.execute(f"""
        SELECT d.doc_id, {SHORT}, {LONG} FROM docs d JOIN have h ON h.id = d.doc_id
        WHERE d.dataset = ? ORDER BY d.doc_id""", [DS]).fetchall() if r[0] not in done]
    con.close()
    print(f"{len(done):,} products already tagged, {len(todo):,} to go", flush=True)

    judge, started = jev.client(), time.perf_counter()
    for i in range(0, len(todo), CHUNK):
        chunk = todo[i:i + CHUNK]
        # A fresh client per chunk, so the request log doesn't grow to 482k rows.
        cands = shortlist(client(), [r[1] for r in chunk])
        probs = choose(judge, "Product", [r[2] for r in chunk], cands, paths)
        rows = []
        for (doc_id, _, _), p in zip(chunk, probs):
            node, mass = settle({k: v for k, v in p.items() if k != "none"}, lineage, DOC_MASS)
            rows.append({"id": doc_id, "node": node, "mass": mass, "none": p.get("none", 0.0),
                         "probs": json.dumps(p)})
        pq.write_table(pa.Table.from_pylist(rows, schema=TAG_SCHEMA), parts / f"part-{len(done) + i:07d}.parquet")
        print(f"  {len(done) + i + len(chunk):>8,} tagged  {(time.perf_counter() - started) / 60:5.1f} min", flush=True)


def doc_tags(scope: str, lineage) -> dict[str, list[str]]:
    rows = duckdb.sql(f"SELECT id, node FROM '{DATA}/doc_tags-{scope}/*.parquet' WHERE node IS NOT NULL").fetchall()
    return {doc_id: lineage[node] for doc_id, node in rows}


def run_patch(args) -> None:
    _, lineage = taxonomy()
    tags, ns = doc_tags(args.scope, lineage), namespace_name(DS, args.scope)
    items, tp = sorted(tags.items()), client()
    for i in range(0, len(items), 5_000):
        chunk = items[i:i + 5_000]
        # patch_columns touches only category_path, so nothing is embedded again.
        # tpuf-arrow only upserts, so this goes through the turbopuffer SDK it wraps.
        tp.tpuf.namespace(ns).write(
            patch_columns={"id": [d for d, _ in chunk], "category_path": [p for _, p in chunk]},
            schema={"category_path": {"type": "[]string"}})
    print(f"patched category_path onto {len(items):,} products in {ns}")


def run_tag_queries(args) -> None:
    paths, lineage = taxonomy()
    with read_only() as con:
        queries = con.execute("SELECT query_id, text FROM queries WHERE dataset = ? AND sampled ORDER BY query_id",
                              [DS]).fetchall()
    cands = shortlist(client(), [t for _, t in queries])
    probs = choose(jev.client(), "Shopping query", [t for _, t in queries], cands, paths)
    rows = []
    for (qid, text), p in zip(queries, probs):
        node, mass = settle({k: v for k, v in p.items() if k != "none"}, lineage, QUERY_MASS)
        rows.append({"query_id": qid, "text": text, "node": node or "", "path": paths.get(node, ""),
                     "mass": round(mass, 4), "none": round(p.get("none", 0.0), 4), "probs": json.dumps(p)})
    write(f"queries-{args.scope}", rows, list(rows[0]))
    print(f"{sum(bool(r['node']) for r in rows)} of {len(rows)} queries got a category")


# -- retrieval ----------------------------------------------------------------

def write(name, rows, fields):
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"ontology_{name}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to results/ontology_{name}.csv")


def read(name):
    with open(OUT / f"ontology_{name}.csv") as f:
        return list(csv.DictReader(f))


def oracle_half(qid: str, doc_id: str) -> bool:
    return int(hashlib.md5(f"{qid}/{doc_id}".encode()).hexdigest(), 16) % 2 == 0


def oracle_nodes(scope: str, lineage) -> dict[str, str]:
    """Per query, the deepest category that 80% of its tagged human-Exact products sit in,
    using only the oracle half of those products (see oracle_half)."""
    tags = doc_tags(scope, lineage)
    with read_only() as con:
        exact = con.execute("""
            SELECT h.query_id, list(h.doc_id) FROM judgments h JOIN queries q USING (dataset, query_id)
            WHERE q.dataset = ? AND q.sampled AND h.source = 'human' AND h.grade = 3 GROUP BY ALL""",
                            [DS]).fetchall()
    out = {}
    for qid, docs in exact:
        paths = [tags[d] for d in docs if d in tags and oracle_half(qid, d)]
        if not paths:
            continue
        counts = Counter(n for p in paths for n in p)
        held = [n for n, c in counts.items() if c / len(paths) >= QUERY_MASS]
        if held:
            out[qid] = max(held, key=lambda n: len(lineage[n]))
    return out


def in_category(node: str) -> tuple:
    return ("category_path", "Contains", node)


def subqueries(variant: str, text: str, node: str) -> list[dict]:
    if not node or variant == "hybrid":
        return hybrid_queries(text)
    if variant.endswith("-soft"):
        return hybrid_queries(text) + hybrid_queries(text, filters=in_category(node))
    return hybrid_queries(text, filters=in_category(node))


def wait_for_index(tp, ns: str, text: str, minutes: float = 30) -> None:
    """After the patch, wait until turbopuffer reports no unindexed documents, so every
    variant runs against the same, fully indexed namespace."""
    deadline = time.monotonic() + minutes * 60
    while True:
        tp.multi_query(ns, hybrid_queries(text, 1), rerank_by=("RRF", {}), limit=1, run_id="index-check")
        unindexed = tp.log["exhaustive_search_count"].to_pylist()[-1] or 0
        if unindexed == 0:
            return
        if time.monotonic() > deadline:
            raise SystemExit(f"{ns} still has {unindexed:,} unindexed documents after {minutes} minutes; try again later.")
        print(f"  waiting for {unindexed:,} unindexed documents", flush=True)
        time.sleep(60)


def run_run(args) -> None:
    _, lineage = taxonomy()
    ns = namespace_name(DS, args.scope)
    predicted = {r["query_id"]: r["node"] for r in read(f"queries-{args.scope}")}
    oracle = oracle_nodes(args.scope, lineage)
    texts = {r["query_id"]: r["text"] for r in read(f"queries-{args.scope}")}
    tp = client()
    wait_for_index(tp, ns, next(iter(texts.values())))

    # Variants take turns query by query, so cache warmth and index state hit them all alike.
    variants = ("hybrid", "onto-soft", "onto-filter", "oracle-soft", "oracle-filter")
    runs = {v: [] for v in variants}
    for i, (qid, text) in enumerate(texts.items()):
        for variant in variants[i % len(variants):] + variants[:i % len(variants)]:
            nodes = oracle if variant.startswith("oracle") else predicted
            node = "" if variant == "hybrid" else nodes.get(qid, "")
            (hits,) = tp.multi_query(ns, subqueries(variant, text, node), rerank_by=("RRF", {}), limit=50,
                                     run_id=f"onto-{variant}", query_id=qid)
            runs[variant].append({"variant": variant, "query_id": qid, "node": node,
                                  "ids": " ".join(hits["id"].to_pylist())})
    rows = []
    for variant, run in runs.items():
        logged = requests_by_query(tp, f"onto-{variant}")
        rows += [r | {"wall_ms": round(logged[r["query_id"]]["wall_ms"], 1),
                      "server_ms": logged[r["query_id"]]["server_ms"]} for r in run]
    print("  first-stage variants done", flush=True)

    # Voyage over each top 50, the way 06_rerank.py does it: a second request.
    for base in ("hybrid", "onto-soft"):
        variant, run = f"{base}+voyage", []
        for r in [r for r in rows if r["variant"] == base]:
            hits = rerank(tp, ns, "voyage/rerank-3", texts[r["query_id"]], r["ids"].split(),
                          run_id=f"onto-{variant}", query_id=r["query_id"])
            run.append(r | {"variant": variant, "ids": " ".join(hits["id"].to_pylist())})
        logged = requests_by_query(tp, f"onto-{variant}")
        rows += [r | {"wall_ms": round(r["wall_ms"] + logged[r["query_id"]]["wall_ms"], 1),
                      "server_ms": r["server_ms"] + logged[r["query_id"]]["server_ms"]} for r in run]
        print(f"  {variant} done", flush=True)
    write(f"runs-{args.scope}", rows, ["variant", "query_id", "node", "wall_ms", "server_ms", "ids"])


def run_judge(args) -> None:
    """Jev grades for top-10 products with no human label, so a new retriever
    isn't scored down just for finding products ESCI never labeled."""
    rows = read(f"runs-{args.scope}")
    with read_only() as con:
        labeled = set(con.execute("SELECT query_id, doc_id FROM judgments WHERE dataset = ? AND source = 'human'",
                                  [DS]).fetchall())
        wanted = sorted({(r["query_id"], d) for r in rows for d in r["ids"].split()[:10]} - labeled)
        con.register("wanted", pa.table({"query_id": [q for q, _ in wanted], "doc_id": [d for _, d in wanted]}))
        pairs = con.execute(f"""
            SELECT w.query_id, w.doc_id, q.text, concat_ws('. ', d.title, d.brand, d.color, left(d.body, 800))
            FROM wanted w JOIN queries q ON q.dataset = '{DS}' AND q.query_id = w.query_id
            JOIN docs d ON d.dataset = '{DS}' AND d.doc_id = w.doc_id""").fetchall()
    grades = grade_pairs(JUDGE, pairs, load("esci"))
    write(f"judged-{args.scope}", [{"query_id": p[0], "doc_id": p[1], "grade": g.grade}
                                   for p, g in zip(pairs, grades) if g.grade is not None],
          ["query_id", "doc_id", "grade"])


# -- scoring ------------------------------------------------------------------

def gain(g):
    return 2 ** g - 1


def ndcg(ranked, labels, condensed=False):
    ideal = sorted((gain(g) for g in labels.values()), reverse=True)[:10]
    idcg = sum(x / math.log2(i + 2) for i, x in enumerate(ideal))
    if not idcg:
        return None
    if condensed:
        ranked = [d for d in ranked if d in labels]
    return sum(gain(labels.get(d, 0)) / math.log2(i + 2) for i, d in enumerate(ranked[:10])) / idcg


def recall50(ranked, labels):
    rel = {d for d, g in labels.items() if g >= 2}
    return len(rel & set(ranked[:50])) / len(rel) if rel else None


def q(vals, p):
    vals = sorted(vals)
    return vals[min(len(vals) - 1, int(p * len(vals)))]


def run_report(args) -> None:
    _, lineage = taxonomy()
    con = read_only()
    human = defaultdict(dict)
    for qid, d, g in con.execute("SELECT query_id, doc_id, grade FROM judgments WHERE dataset = ? AND source = 'human'",
                                 [DS]).fetchall():
        human[qid][d] = g
    filled = None
    if (OUT / f"ontology_judged-{args.scope}.csv").exists():
        filled = defaultdict(dict)
        for r in read(f"judged-{args.scope}"):
            filled[r["query_id"]][r["doc_id"]] = int(r["grade"])
        for qid, labels in human.items():
            filled[qid] |= labels  # human labels win where both exist

    rows, queries = read(f"runs-{args.scope}"), read(f"queries-{args.scope}")
    by_variant = defaultdict(dict)
    for r in rows:
        by_variant[r["variant"]][r["query_id"]] = r

    # The oracle chose its category from half the Exact products; score it on the rest.
    held_out = {qid: {d: g for d, g in labels.items() if not (g == 3 and oracle_half(qid, d))}
                for qid, labels in human.items()}

    def metric(variant, fn, labels=human, only=None):
        out = {}
        for qid, r in by_variant[variant].items():
            if only is None or qid in only:
                v = fn(r["ids"].split(), labels.get(qid, {}))
                if v is not None:
                    out[qid] = v
        return out

    metrics = {"nDCG@10": lambda r, l: ndcg(r, l), "condensed": lambda r, l: ndcg(r, l, True), "Recall@50": recall50}
    pairs = [(v, "hybrid") for v in ("onto-soft", "onto-filter", "oracle-soft", "oracle-filter")]
    pairs.append(("onto-soft+voyage", "hybrid+voyage"))

    print("\n## Tagging")
    tags = duckdb.sql(f"SELECT count(*), count(node) FROM '{DATA}/doc_tags-{args.scope}/*.parquet'").fetchall()[0]
    print(f"  products: {tags[1]:,} of {tags[0]:,} got a category")
    depth = Counter(len(lineage[r["node"]]) for r in queries if r["node"])
    print(f"  queries: {sum(depth.values())} of {len(queries)} got a category; by depth "
          + ", ".join(f"{d}: {n}" for d, n in sorted(depth.items())))
    oracle = oracle_nodes(args.scope, lineage)
    pred = {r["query_id"]: r["node"] for r in queries if r["node"]}
    both = pred.keys() & oracle.keys()
    on_path = sum(pred[q_] in lineage[oracle[q_]] for q_ in both)
    print(f"  predicted category is the oracle's or one of its parents: {on_path} of {len(both)} "
          f"({on_path / max(len(both), 1):.0%}); oracle has a category for {len(oracle)} queries")

    print("\n## Quality, human labels (filled: Jev grades the top-10 products humans didn't label)")
    print(f"{'variant':<18}{'vs':<15}{'metric':<11}{'value':>7}{'delta [95% CI]':>26}{'n':>5}")
    for a, b in pairs:
        labels = held_out if a.startswith("oracle") else human
        for name, fn in metrics.items():
            sa, sb = metric(a, fn, labels), metric(b, fn, labels)
            m, lo, hi, n = paired_delta(sa, sb)
            print(f"{a:<18}{b:<15}{name:<11}{sum(sa.values()) / len(sa):7.3f}{f'{m:+.3f} [{lo:+.3f}, {hi:+.3f}]':>26}{n:>5}")
        if filled:
            sa, sb = metric(a, metrics["nDCG@10"], filled), metric(b, metrics["nDCG@10"], filled)
            m, lo, hi, n = paired_delta(sa, sb)
            print(f"{a:<18}{b:<15}{'filled':<11}{sum(sa.values()) / len(sa):7.3f}{f'{m:+.3f} [{lo:+.3f}, {hi:+.3f}]':>26}{n:>5}")

    print("\n## onto-soft minus hybrid, nDCG@10, by how sure the query tagger was")
    buckets = defaultdict(set)
    for r in queries:
        m_ = float(r["mass"])
        buckets["abstained" if not r["node"] else "0.95+" if m_ >= 0.95 else "0.80-0.95"].add(r["query_id"])
    for name in ("0.95+", "0.80-0.95", "abstained"):
        m, lo, hi, n = paired_delta(metric("onto-soft", metrics["nDCG@10"], only=buckets[name]),
                                    metric("hybrid", metrics["nDCG@10"], only=buckets[name]))
        print(f"  {name:<10} {m:+.3f} [{lo:+.3f}, {hi:+.3f}]  n={n}")

    print("\n## Latency (ms) and short result lists")
    for v, rs in by_variant.items():
        rs = list(rs.values())
        print(f"  {v:<18} wall p50 {q([float(r['wall_ms']) for r in rs], .5):5.0f}  p99 {q([float(r['wall_ms']) for r in rs], .99):5.0f}  "
              f"server p50 {q([float(r['server_ms']) for r in rs], .5):5.0f}  "
              f"fewer than 50 results: {sum(len(r['ids'].split()) < 50 for r in rs)}")


if __name__ == "__main__":
    commands = {"taxonomy": run_taxonomy, "tag-docs": run_tag_docs, "patch": run_patch,
                "tag-queries": run_tag_queries, "run": run_run, "judge": run_judge, "report": run_report}
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=commands)
    ap.add_argument("--scope", choices=["smoke", "full"], default="full")
    args = ap.parse_args()
    commands[args.command](args)
