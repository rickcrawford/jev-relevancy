"""Recompute every number the README and docs/ cite.

The docs quote figures from the warehouse and from the experiments in
this folder. This script derives each of them again from the data, in
numbered sections, so a claim like "Jev's gain shrinks on condensed nDCG"
can be checked against section 1 rather than taken on trust.

  uv run experiments/evidence.py | tee experiments/records/evidence.txt

It reads the warehouse read-only. Bootstrap intervals resample queries
(or pairs, for the bake-off) 1,000 times with a fixed seed, so reruns
print the same numbers.
"""

import csv
import itertools
import math
import random
from pathlib import Path

from relevancy.agreement import kendall_tau, pearson, weighted_kappa
from relevancy.metrics import GAINS, paired_delta, per_query
from relevancy.warehouse import read_only

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
RECORDS = EXP / "records"
DS = "esci-us"
RUNS = {
    "hybrid": "hybrid-rrf-full-4cea30",
    "voyage": "rerank-3-977611",
    "jev": "jev-1.13.0-b53bce",
    "zerank": "zerank-2-47e94e",
    "cohere": "rerank-v3.5-b99df4",
    "gemma": "gemma-3-27b-it-e07d2c",
    "ann": "ann-full-c9d0b4",
    "bm25-title": "bm25-title-full-4505b5",
    "bm25": "bm25-full-849929",
}
JUDGES = {"jev": "jev-batched:jev-1.13.0", "gemma": "google/gemma-3-27b-it"}
NEGATION = r"\b(without|not|no|non)\b"
DRAWS = 1000

con = read_only()
rng = random.Random(0)
inv = {v: k for k, v in RUNS.items()}


def section(title: str) -> None:
    # Reseed per section, so adding a section never shifts another one's intervals.
    rng.seed(title)
    print(f"\n## {title}\n")


def ci(values: list[float]) -> tuple[float, float]:
    values = sorted(values)
    return values[round(0.025 * len(values)) - 1], values[round(0.975 * len(values)) - 1]


def by_run(pq: dict, metric: str) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for (run_id, q), m in pq.items():
        if m[metric] is not None:
            out.setdefault(inv[run_id], {})[q] = m[metric]
    return out


def mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs)


def paired_p(a: dict[str, float], b: dict[str, float]) -> float:
    """Two-sided p-value for the mean per-query difference (normal approximation)."""
    d = [a[q] - b[q] for q in a if q in b]
    n, m = len(d), mean(d)
    sd = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1))
    z = abs(m) / (sd / math.sqrt(n)) if sd else float("inf")
    return math.erfc(z / math.sqrt(2))


human = per_query(con, DS, list(RUNS.values()), judge="human")
std, cond = by_run(human, "ndcg10"), by_run(human, "ndcg10_condensed")

# 1 ------------------------------------------------------------------
section("1. Runs: nDCG@10 standard and condensed, paired against plain hybrid")
lat = dict(con.execute("""
    SELECT run_id, quantile_cont(latency_ms, 0.5) FROM (SELECT DISTINCT run_id, query_id, latency_ms FROM results)
    WHERE run_id IN (SELECT unnest(?)) GROUP BY 1""", [list(RUNS.values())]).fetchall())
added = dict(con.execute("""
    WITH b AS (SELECT DISTINCT query_id, latency_ms FROM results WHERE run_id = ?)
    SELECT r.run_id, quantile_cont(r.latency_ms - b.latency_ms, 0.5)
    FROM (SELECT DISTINCT run_id, query_id, latency_ms FROM results) r JOIN b USING (query_id)
    WHERE r.run_id IN (SELECT unnest(?)) GROUP BY 1""", [RUNS["hybrid"], list(RUNS.values())]).fetchall())
server = {r: (p50, p99) for r, p50, p99 in con.execute("""
    SELECT run_id, quantile_cont(server_ms, 0.5), quantile_cont(server_ms, 0.99)
    FROM (SELECT DISTINCT run_id, query_id, server_ms FROM results)
    WHERE run_id IN (SELECT unnest(?)) GROUP BY 1""", [list(RUNS.values())]).fetchall()}
added_server = dict(con.execute("""
    WITH b AS (SELECT DISTINCT query_id, server_ms FROM results WHERE run_id = ?)
    SELECT r.run_id, quantile_cont(r.server_ms - b.server_ms, 0.5)
    FROM (SELECT DISTINCT run_id, query_id, server_ms FROM results) r JOIN b USING (query_id)
    WHERE r.run_id IN (SELECT unnest(?)) GROUP BY 1""", [RUNS["hybrid"], list(RUNS.values())]).fetchall())
recall = by_run(human, "recall50")
print(f"{'run':<12}{'nDCG':>7}{'cond':>7}{'R@50':>7}  {'vs hybrid, standard':<26}{'vs hybrid, condensed':<26}"
      f"{'p50 ms':>8}{'added':>7}{'server p50/p99':>16}{'server added':>13}")
for name, rid in RUNS.items():
    row = f"{name:<12}{mean(std[name].values()):7.3f}{mean(cond[name].values()):7.3f}{mean(recall[name].values()):7.3f}  "
    if name == "hybrid":
        row += f"{'base':<26}{'base':<26}"
    else:
        for d in (std, cond):
            m, lo, hi, _ = paired_delta(d[name], d["hybrid"])
            row += f"{f'{m:+.3f} [{lo:+.3f}, {hi:+.3f}]':<26}"
    sp = server.get(rid, (None, None))
    row += f"{lat[rid]:8.0f}{(added.get(rid) or 0):7.0f}" + (f"{sp[0]:>9.0f}/{sp[1]:<6.0f}" if sp[0] else "")
    row += f"{(added_server.get(rid) or 0):7.0f}"
    print(row)

for run, lo, med in con.execute("""
    SELECT run_id, min(latency_ms - server_ms), quantile_cont(latency_ms - server_ms, 0.5)
    FROM (SELECT DISTINCT run_id, query_id, latency_ms, server_ms FROM results)
    WHERE run_id IN (SELECT unnest(?)) GROUP BY 1 ORDER BY 1""",
                              [[RUNS[r] for r in ("hybrid", "ann", "bm25", "bm25-title")]]).fetchall():
    print(f"  {inv[run]}: wall minus server time, min {lo:.0f} ms, median {med:.0f} ms (the laptop's round trip)")

# 2 ------------------------------------------------------------------
section("2. Jev against Voyage, head to head")
for label, d in (("standard", std), ("condensed", cond)):
    m, lo, hi, n = paired_delta(d["jev"], d["voyage"])
    print(f"Jev minus Voyage, {label}: {m:+.3f} [{lo:+.3f}, {hi:+.3f}], n={n}, p={paired_p(d['jev'], d['voyage']):.3f}")
wins = [std["jev"][q] - std["voyage"][q] for q in std["jev"]]
print(f"Per query: Jev better {sum(w > 0 for w in wins)}, Voyage better {sum(w < 0 for w in wins)}, "
      f"tied {sum(w == 0 for w in wins)}")
m, lo, hi, _ = paired_delta(std["voyage"], std["zerank"])
print(f"Voyage minus zerank, standard: {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")
for other in ("cohere", "gemma", "zerank"):
    m, lo, hi, _ = paired_delta(std["jev"], std[other])
    print(f"Jev minus {other}, standard: {m:+.3f} [{lo:+.3f}, {hi:+.3f}], p={paired_p(std['jev'], std[other]):.3f}")

# 3 ------------------------------------------------------------------
section("3. Share of each run's top 10 with no human label")
for run, share in con.execute("""
    SELECT r.run_id, avg((h.doc_id IS NULL)::INT)
    FROM results r LEFT JOIN judgments h ON h.dataset = ? AND h.source = 'human'
         AND h.query_id = r.query_id AND h.doc_id = r.doc_id
    WHERE r.run_id IN (SELECT unnest(?)) AND r.rank <= 10 GROUP BY 1 ORDER BY 2""",
                              [DS, list(RUNS.values())]).fetchall():
    print(f"{inv[run]:<12}{share:.3f}")

# 4 ------------------------------------------------------------------
section("4. How many of the 36 run comparisons hold up (paired, standard and condensed)")
pairs36 = list(itertools.combinations(RUNS, 2))
for label, d in (("standard", std), ("condensed", cond)):
    ps = {(a, b): paired_p(d[a], d[b]) for a, b in pairs36}
    print(f"{label}: p<0.05 in {sum(p < 0.05 for p in ps.values())} of 36; "
          f"Bonferroni (p<{0.05 / 36:.4f}) in {sum(p < 0.05 / 36 for p in ps.values())}")
    weak = sorted((p, a, b) for (a, b), p in ps.items() if 0.05 / 36 <= p)
    print("  not significant after Bonferroni: " + ", ".join(f"{a}/{b} (p={p:.3f})" for p, a, b in weak))

# 5 ------------------------------------------------------------------
section("5. Power: smallest difference 500 queries can detect")
ses = []
for a, b in itertools.combinations(["voyage", "jev", "zerank", "cohere", "gemma"], 2):
    d = [std[a][q] - std[b][q] for q in std[a]]
    m = mean(d)
    ses.append(math.sqrt(sum((x - m) ** 2 for x in d) / (len(d) - 1)) / math.sqrt(len(d)))
print(f"Paired standard error between rerankers: {min(ses):.4f} to {max(ses):.4f}")
print(f"Minimum detectable difference (80% power, 5% two-sided, 2.8 x SE): {2.8 * min(ses):.3f} to {2.8 * max(ses):.3f}")

# 6 ------------------------------------------------------------------
section("6. Judge bake-off (step 7) with bootstrap intervals")
bake = {}
expected_of = {}
for source, q, d, h, g, e in con.execute("""
    SELECT j.source, j.query_id, j.doc_id, h.grade, j.grade, j.expected FROM judgments j
    JOIN judgments h ON h.dataset = j.dataset AND h.query_id = j.query_id AND h.doc_id = j.doc_id AND h.source = 'human'
    WHERE j.dataset = ? AND j.source LIKE '%@bakeoff'""", [DS]).fetchall():
    bake.setdefault(source.removesuffix("@bakeoff"), {})[(q, d)] = (h, g)
    expected_of[(source.removesuffix("@bakeoff"), q, d)] = e
for judge, rows in sorted(bake.items(), key=lambda kv: -mean(h == g for h, g in kv[1].values())):
    vals = list(rows.values())
    ex, ka = [], []
    for _ in range(DRAWS):
        s = rng.choices(vals, k=len(vals))
        ex.append(mean(h == g for h, g in s))
        ka.append(weighted_kappa([h for h, _ in s], [g for _, g in s]))
    e0 = mean(h == g for h, g in vals)
    k0 = weighted_kappa([h for h, _ in vals], [g for _, g in vals])
    (elo, ehi), (klo, khi) = ci(ex), ci(ka)
    w1 = mean(abs(h - g) <= 1 for h, g in vals)
    r = pearson([float(h) for h, _ in rows.values()], [expected_of[(judge, *k)] for k in rows])
    print(f"{judge:<34} n={len(vals)}  exact {e0:.3f} [{elo:.3f}, {ehi:.3f}]  within 1 {w1:.3f}  "
          f"kappa {k0:.3f} [{klo:.3f}, {khi:.3f}]  r {r:.3f}")
for other in ("google/gemma-3-27b-it", "meta-llama/llama-4-maverick", "jev:jev-1.14.0"):
    a, b = bake["jev:jev-1.13.0"], bake[other]
    keys = [k for k in a if k in b]
    diffs = [(a[k][0] == a[k][1]) - (b[k][0] == b[k][1]) for k in keys]
    boot = [mean(rng.choices(diffs, k=len(diffs))) for _ in range(DRAWS)]
    lo, hi = ci(boot)
    print(f"exact, jev 1.13.0 minus {other}: {mean(diffs):+.3f} [{lo:+.3f}, {hi:+.3f}]")
a, b = bake["jev:jev-1.13.0"], bake["jev-batched:jev-1.13.0"]
keys = [k for k in a if k in b]
qcount = len({q for q, _ in keys})
print(f"Test-retest: jev: and jev-batched: on the same {len(keys)} pairs ({qcount} queries, so every request held "
      f"one product) gave a different grade on {mean(a[k][1] != b[k][1] for k in keys):.1%}")

# 7 ------------------------------------------------------------------
section("7. Step 8: judges ranking runs, with bootstrap intervals on tau")
judged = {j: by_run(per_query(con, DS, list(RUNS.values()), judge=s), "ndcg10") for j, s in JUDGES.items()}
for j, own in (("jev", "jev"), ("gemma", "gemma")):
    runs = [r for r in RUNS if r != own]
    qs = [q for q in std["hybrid"] if all(q in judged[j][r] for r in runs)]
    print(f"{j}: {len(std['hybrid']) - len(qs)} of {len(std['hybrid'])} queries have no judge nDCG (nothing graded above 0)")
    for label, hm in (("standard", std), ("condensed", cond)):
        h_means = {r: mean(hm[r][q] for q in qs) for r in runs}
        j_means = {r: mean(judged[j][r][q] for q in qs) for r in runs}
        t0 = kendall_tau([h_means[r] for r in runs], [j_means[r] for r in runs])
        taus = []
        for _ in range(DRAWS):
            s = rng.choices(qs, k=len(qs))
            taus.append(kendall_tau([mean(hm[r][q] for q in s) for r in runs],
                                    [mean(judged[j][r][q] for q in s) for r in runs]))
        lo, hi = ci(taus)
        flips = [f"{a}/{b} (human gap {h_means[a] - h_means[b]:+.3f})"
                 for a, b in itertools.combinations(runs, 2)
                 if (h_means[a] - h_means[b]) * (j_means[a] - j_means[b]) < 0]
        print(f"  {label:<9} tau {t0:.3f} [{lo:.3f}, {hi:.3f}]; flipped: {', '.join(flips) or 'none'}")
    gaps = [abs(mean(std[a].values()) - mean(std[b].values())) for a, b in itertools.combinations(runs, 2)]
    print(f"  of the {len(gaps)} pairs of runs, {sum(g < 0.03 for g in gaps)} are within 0.03 on human nDCG")

pool = con.execute("""
    SELECT DISTINCT query_id, doc_id FROM results WHERE run_id IN (SELECT unnest(?)) AND rank <= 10""",
                   [list(RUNS.values())]).fetchall()
con.execute("CREATE TEMP TABLE pool (query_id VARCHAR, doc_id VARCHAR)")
con.executemany("INSERT INTO pool VALUES (?, ?)", pool)
labeled = con.execute("SELECT count(*) FROM pool JOIN judgments h USING (query_id, doc_id) "
                      "WHERE h.dataset = ? AND h.source = 'human'", [DS]).fetchone()[0]
print(f"\nPool: {len(pool):,} (query, product) pairs; {labeled:,} ({labeled / len(pool):.0%}) have a human label")
for j, s in JUDGES.items():
    rows = con.execute("""
        SELECT p.query_id, h.grade, g.grade FROM pool p
        JOIN judgments h ON h.dataset = ? AND h.source = 'human' AND h.query_id = p.query_id AND h.doc_id = p.doc_id
        JOIN judgments g ON g.dataset = ? AND g.source = ? AND g.query_id = p.query_id AND g.doc_id = p.doc_id""",
                       [DS, DS, s]).fetchall()
    byq: dict[str, list] = {}
    for q, h, g in rows:
        byq.setdefault(q, []).append((h, g))
    qlist = sorted(byq)  # join output is unordered; sort so reruns draw the same samples
    ex, ka = [], []
    for _ in range(DRAWS):
        s_ = [p for q in rng.choices(qlist, k=len(qlist)) for p in byq[q]]
        ex.append(mean(h == g for h, g in s_))
        ka.append(weighted_kappa([h for h, _ in s_], [g for _, g in s_]))
    e0 = mean(h == g for _, h, g in rows)
    k0 = weighted_kappa([h for _, h, _ in rows], [g for _, _, g in rows])
    (elo, ehi), (klo, khi) = ci(ex), ci(ka)
    print(f"  {j}: {len(rows):,} labeled pool pairs, exact {e0:.3f} [{elo:.3f}, {ehi:.3f}], "
          f"kappa {k0:.3f} [{klo:.3f}, {khi:.3f}] (intervals resample queries)")

# 8 ------------------------------------------------------------------
section("8. Why pooled agreement (0.60) beats the bake-off (0.52)")
pool_rows = con.execute("""
    SELECT h.grade, g.grade FROM pool p
    JOIN judgments h ON h.dataset = ? AND h.source = 'human' AND h.query_id = p.query_id AND h.doc_id = p.doc_id
    JOIN judgments g ON g.dataset = ? AND g.source = ? AND g.query_id = p.query_id AND g.doc_id = p.doc_id""",
                        [DS, DS, JUDGES["jev"]]).fetchall()
bake_rows = list(bake["jev-batched:jev-1.13.0"].values())
print(f"{'human grade':<12}{'share of pool':>14}{'Jev exact, pool':>17}{'Jev exact, bake-off':>21}")
recall_pool = {}
for g in range(4):
    p = [x for x in pool_rows if x[0] == g]
    b = [x for x in bake_rows if x[0] == g]
    recall_pool[g] = mean(h == j for h, j in p)
    print(f"{g:<12}{len(p) / len(pool_rows):14.1%}{recall_pool[g]:17.3f}{mean(h == j for h, j in b):21.3f}")
print(f"Pooled agreement reweighted to a balanced mix: {mean(recall_pool.values()):.3f} "
      f"(observed {mean(h == j for h, j in pool_rows):.3f}, bake-off {mean(h == j for h, j in bake_rows):.3f})")

# 9 ------------------------------------------------------------------
section("9. A judge's nDCG depends on which runs are scored together")
for subset in (["voyage", "cohere", "hybrid", "bm25"], [r for r in RUNS if r != "jev"], list(RUNS)):
    v = by_run(per_query(con, DS, [RUNS[r] for r in subset], judge=JUDGES["jev"]), "ndcg10")
    print(f"  scored with {len(subset)} runs: Voyage under Jev = {mean(v['voyage'].values()):.3f}")

# 10 -----------------------------------------------------------------
section("10. Gain function: does the run order change?")
GAINS.setdefault("binary", "(CASE WHEN {g} >= 2 THEN 1.0 ELSE 0 END)")
for gain in ("exp", "linear", "binary"):
    v = by_run(per_query(con, DS, list(RUNS.values()), judge="human", gain=gain), "ndcg10")
    order = sorted(v, key=lambda r: -mean(v[r].values()))
    print(f"  {gain:<7} " + " > ".join(order))

# 11 -----------------------------------------------------------------
section("11. Negation queries")
for neg, queries, exact, missed in con.execute(f"""
    SELECT regexp_matches(lower(q.text), '{NEGATION}') AND NOT contains(lower(q.text), 'forget me not'),
           count(DISTINCT q.query_id), count(*), count(*) FILTER (WHERE j.grade = 0)
    FROM judgments h JOIN judgments j USING (dataset, query_id, doc_id) JOIN queries q USING (dataset, query_id)
    WHERE h.dataset = ? AND h.source = 'human' AND h.grade = 3 AND j.source = ? AND q.sampled
    GROUP BY 1 ORDER BY 1""", [DS, JUDGES["jev"]]).fetchall():
    print(f"  negation={neg!s:<5}  {queries} queries, Jev called {missed} of {exact} human-Exact products "
          f"Irrelevant ({missed / exact:.1%})")
answers_path = EXP / "judge-negation-audit" / "results" / "negation_answers.csv"
if answers_path.exists():
    counts: dict[str, int] = {}
    for row in csv.DictReader(answers_path.open()):
        counts[row["has_excluded"]] = counts.get(row["has_excluded"], 0) + 1
    print(f"  blind audit of the {sum(counts.values())} exported disagreements: {counts}")
    real = sum(counts.values()) - counts.get("not_negation", 0)
    print(f"  of the {real} on real negations (\"forget me not\" removed): product has the excluded thing "
          f"{counts.get('yes', 0)} ({counts.get('yes', 0) / real:.0%}), doesn't {counts.get('no', 0)}, "
          f"text doesn't say {counts.get('unclear', 0)}")

# 12 -----------------------------------------------------------------
section("12. Batch order and description length (experiments)")
bo = EXP / "judge-batch-order" / "results" / "batch_order.csv"
if bo.exists():
    rows = list(csv.DictReader(bo.open()))
    same = mean(r["stored_grade"] == r["same_order_grade"] for r in rows)
    rev = mean(r["stored_grade"] == r["reversed_grade"] for r in rows)
    print(f"  batch order, {len(rows):,} pairs: unchanged {same:.1%} regraded as is, {rev:.1%} with products reversed")
tr = EXP / "judge-truncation" / "results" / "truncation.csv"
if tr.exists():
    rows = list(csv.DictReader(tr.open()))
    for col in ("jev_800", "jev_4000"):
        ok = [(int(r["human"]), int(r[col])) for r in rows if r[col]]
        print(f"  {col}: exact {mean(h == g for h, g in ok):.3f}, "
              f"kappa {weighted_kappa([h for h, _ in ok], [g for _, g in ok]):.3f}, n={len(ok)}")

# 13 -----------------------------------------------------------------
section("13. Query slices")
slice_of = dict(con.execute("SELECT query_id, slice FROM queries WHERE dataset = ? AND sampled", [DS]).fetchall())
print(f"{'slice':<17}{'queries':>8}{'share':>7}{'hybrid':>8}{'best run':>22}")
for s in sorted(set(slice_of.values()), key=lambda s: -sum(v == s for v in slice_of.values())):
    qs = [q for q, v in slice_of.items() if v == s]
    scores = {r: mean(std[r][q] for q in qs) for r in RUNS}
    top = max(scores, key=scores.get)
    print(f"{s:<17}{len(qs):8}{len(qs) / len(slice_of):7.1%}{scores['hybrid']:8.3f}{f'{top} {scores[top]:.3f}':>22}")

# 14 -----------------------------------------------------------------
section("14. Billing and warehouse size")
for run, n, tb in con.execute("""
    SELECT run_id, count(*), sum(billable_bytes) / 1e12 FROM (SELECT DISTINCT run_id, query_id, billable_bytes FROM results)
    WHERE run_id IN (SELECT unnest(?)) GROUP BY 1 ORDER BY 3 NULLS LAST, 1""", [list(RUNS.values())]).fetchall():
    where = f"{tb:.2f} TB queried" if tb is not None else "not recorded (reranked runs leave billable_bytes empty)"
    print(f"  {inv[run]:<12} {n} queries, {where}")
for ns, ret, gb in con.execute("""
    SELECT ru.config->>'namespace', ru.retriever, round(avg(billable_bytes) / 1e9, 2)
    FROM (SELECT DISTINCT run_id, query_id, billable_bytes FROM results) r JOIN runs ru USING (run_id)
    WHERE billable_bytes IS NOT NULL AND ru.reranker = 'none' GROUP BY ALL ORDER BY 1, 2""").fetchall():
    print(f"  {ns} {ret}: {gb} GB billed per query")
for sha, n in con.execute("SELECT git_sha, count(*) FROM runs WHERE run_id IN (SELECT unnest(?)) GROUP BY 1",
                          [list(RUNS.values())]).fetchall():
    print(f"  the 9 runs above: {n} recorded git_sha {sha}")
print(f"  warehouse file: {(ROOT / 'warehouse' / 'relevancy.duckdb').stat().st_size / 1e6:.0f} MB")
for table in ("docs", "queries", "judgments", "results", "runs"):
    print(f"  {table}: {con.execute(f'SELECT count(*) FROM {table}').fetchone()[0]:,} rows")

# 15 -----------------------------------------------------------------
section("15. Jev 1.14.0 against 1.13.0 (from experiments/judge-jev-1.14/results/)")
labels: dict[str, dict[str, int]] = {}
for q, d, g in con.execute("SELECT query_id, doc_id, grade FROM judgments WHERE dataset = ? AND source = 'human'",
                           [DS]).fetchall():
    labels.setdefault(q, {})[d] = g


def ndcg_from(ranked: dict[str, list[str]]) -> dict[str, float]:
    """Standard nDCG@10 with human labels, the same definition as metrics.py."""
    out = {}
    for q, docs in ranked.items():
        ideal = sorted(labels.get(q, {}).values(), reverse=True)[:10]
        idcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
        if idcg:
            out[q] = sum((2 ** labels[q].get(d, 0) - 1) / math.log2(i + 2) for i, d in enumerate(docs[:10])) / idcg
    return out


rerank_path, judge_path = EXP / "judge-jev-1.14" / "results" / "jev114_rerank.csv", EXP / "judge-jev-1.14" / "results" / "jev114_judge.csv"
if rerank_path.exists():
    ranked: dict[str, list[tuple[int, str]]] = {}
    for row in csv.DictReader(rerank_path.open()):
        ranked.setdefault(row["query_id"], []).append((int(row["rank"]), row["doc_id"]))
    v114 = ndcg_from({q: [d for _, d in sorted(v)] for q, v in ranked.items()})
    v113 = {q: v for q, v in std["jev"].items() if q in v114}
    m, lo, hi, n = paired_delta(v114, v113)
    print(f"  as a reranker: 1.14.0 nDCG@10 {mean(v114.values()):.3f} against 1.13.0 {mean(v113.values()):.3f}, "
          f"paired {m:+.3f} [{lo:+.3f}, {hi:+.3f}], n={n}")
if judge_path.exists():
    g114 = {(r["query_id"], r["doc_id"]): int(r["grade"]) for r in csv.DictReader(judge_path.open())}
    rows = con.execute("""
        SELECT p.query_id, p.doc_id, h.grade, g.grade FROM pool p
        JOIN judgments h ON h.dataset = ? AND h.source = 'human' AND h.query_id = p.query_id AND h.doc_id = p.doc_id
        JOIN judgments g ON g.dataset = ? AND g.source = ? AND g.query_id = p.query_id AND g.doc_id = p.doc_id""",
                       [DS, DS, JUDGES["jev"]]).fetchall()
    rows = [(q, h, g13, g114[(q, d)]) for q, d, h, g13 in rows if (q, d) in g114]
    byq: dict[str, list] = {}
    for q, h, a, b in rows:
        byq.setdefault(q, []).append((h == b) - (h == a))
    qlist = sorted(byq)  # join output is unordered; sort so reruns draw the same samples
    boot = []
    for _ in range(DRAWS):
        s_ = [x for q in rng.choices(qlist, k=len(qlist)) for x in byq[q]]
        boot.append(mean(s_))
    lo, hi = ci(boot)
    e13, e14 = mean(h == a for _, h, a, _ in rows), mean(h == b for _, h, _, b in rows)
    k14 = weighted_kappa([h for _, h, _, _ in rows], [b for _, _, _, b in rows])
    print(f"  as a judge on the same {len(rows):,} labeled pool pairs: exact {e14:.3f} against {e13:.3f}, "
          f"paired {e14 - e13:+.3f} [{lo:+.3f}, {hi:+.3f}] (resampling queries); 1.14.0 kappa {k14:.3f}")
    print(f"  1.13.0 and 1.14.0 give the same grade on {mean(a == b for _, _, a, b in rows):.1%} of those pairs")

# 16 -----------------------------------------------------------------
section("16. Worked examples quoted in the warehouse recipes")
first = con.execute("SELECT query_id, text FROM queries WHERE dataset = ? AND sampled ORDER BY query_id LIMIT 1",
                    [DS]).fetchone()
graded, jev_exact = con.execute("""
    SELECT count(h.grade), count(*) FILTER (WHERE j.grade = 3) FROM results r
    LEFT JOIN judgments h ON h.dataset = ? AND h.source = 'human' AND h.query_id = r.query_id AND h.doc_id = r.doc_id
    LEFT JOIN judgments j ON j.dataset = ? AND j.source = ? AND j.query_id = r.query_id AND j.doc_id = r.doc_id
    WHERE r.run_id = ? AND r.query_id = ? AND r.rank <= 10""",
                                [DS, DS, JUDGES["jev"], RUNS["hybrid"], first[0]]).fetchone()
print(f"  first sampled query \"{first[1]}\": hybrid top 10 has {graded} human labels; Jev called {jev_exact} Exact")
text_of = dict(con.execute("SELECT query_id, text FROM queries WHERE dataset = ? AND sampled", [DS]).fetchall())
swings = sorted(std["jev"], key=lambda q: -abs(std["jev"][q] - std["voyage"][q]))[:3]
for q in swings:
    print(f"  big Jev/Voyage swing: \"{text_of[q]}\" Jev {std['jev'][q]:.3f}, Voyage {std['voyage'][q]:.3f}")
for h, j, n in con.execute("""
    SELECT h.grade, j.grade, count(*) FROM pool p
    JOIN judgments h ON h.dataset = ? AND h.source = 'human' AND h.query_id = p.query_id AND h.doc_id = p.doc_id
    JOIN judgments j ON j.dataset = ? AND j.source = ? AND j.query_id = p.query_id AND j.doc_id = p.doc_id
    WHERE h.grade = 2 GROUP BY ALL ORDER BY 2""", [DS, DS, JUDGES["jev"]]).fetchall():
    print(f"  pool confusion, human Substitute (2) graded {j} by Jev: {n}")

# 17 -----------------------------------------------------------------
section("17. Corpus size and embedding tokens")
n_docs, chars = con.execute("""
    SELECT count(*), sum(length(concat_ws('. ', title, brand, color, left(body, 800))))
    FROM docs WHERE dataset = ?""", [DS]).fetchone()
print(f"  embedded text (title. brand. color. first 800 characters of body): {n_docs:,} products, {chars:,} characters")
per_q = [n for (n,) in con.execute("""
    SELECT count(*) FROM judgments j JOIN queries q USING (dataset, query_id)
    WHERE j.dataset = ? AND j.source = 'human' AND q.sampled GROUP BY j.query_id""", [DS]).fetchall()]
print(f"  human grades per sampled query: mean {mean(per_q):.1f}, median {sorted(per_q)[len(per_q) // 2]}, n={len(per_q)}")
# The first full load logged turbopuffer's own token count before it stopped.
last = [line for line in (RECORDS / "full_load_first_attempt.txt").read_text().splitlines() if "batches" in line][-1]
parts = last.replace(",", "").split()
rows_done, tokens_m, minutes = int(parts[2]), float(parts[4].rstrip("M")), float(parts[6])
(chars_done,) = con.execute("""
    SELECT sum(length(concat_ws('. ', title, brand, color, left(body, 800))))
    FROM (SELECT * FROM docs WHERE dataset = ? ORDER BY doc_id LIMIT ?)""", [DS, rows_done]).fetchone()
ratio = chars_done / (tokens_m * 1e6)
print(f"  first full load: {rows_done:,} products, {tokens_m}M tokens reported by turbopuffer, {minutes} min "
      f"({tokens_m / minutes:.2f}M tokens a minute)")
print(f"  those products are {chars_done:,} characters, {ratio:.2f} characters per token "
      f"(writes finish out of order, so this is approximate)")
print(f"  whole corpus at that ratio: about {chars / ratio / 1e6:.0f}M tokens, ${chars / ratio / 1e6 * 0.02:.2f} at $0.02 per 1M")
