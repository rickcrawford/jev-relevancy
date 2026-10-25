import sys
import types
from types import SimpleNamespace

import duckdb
from fakes import h


def test_wands(world, capsys):
    m = world.study("judge-wands/wands.py")
    m.DATA.mkdir(parents=True)
    (m.DATA / "product.csv").write_text("product_id\tproduct_name\tproduct_class\tcategory hierarchy\tproduct_description\t"
                                        "product_features\trating_count\taverage_rating\treview_count\n"
                                        + "".join(f"{i}\tbed {i}\tBeds\tFurniture / Beds\tnice bed {i}\tsoft\t1\t4\t1\n"
                                                  for i in range(80)))
    (m.DATA / "query.csv").write_text("query_id\tquery\tquery_class\n" + "".join(f"{q}\tbed {q}\tBeds\n" for q in range(40)))
    (m.DATA / "label.csv").write_text("id\tquery_id\tproduct_id\tlabel\n" + "".join(
        f"{q * 10 + k}\t{q}\t{(q + k) % 80}\t{['Exact', 'Partial', 'Irrelevant'][k % 3]}\n" for q in range(40) for k in range(10)))
    try:
        m.run_prepare(None)
    except SystemExit as e:
        assert e.code == 0
    con = duckdb.connect(str(world.db))
    assert con.execute("SELECT count(*) FROM queries WHERE dataset = 'wands' AND NOT sampled").fetchone()[0] == 0  # 40 < 380
    assert con.execute("SELECT rubric FROM datasets WHERE dataset = 'wands'").fetchone()[0] == "wands"
    # Stand in for the pipeline steps: four runs and Jev grades.
    qids = [r[0] for r in con.execute("SELECT query_id FROM queries WHERE dataset = 'wands'").fetchall()]
    for run in ("a", "b", "c", "d"):
        con.execute("INSERT INTO runs (run_id, dataset, retriever, reranker) VALUES (?, 'wands', 'hybrid-rrf', ?)", [run, run])
        for q in qids:
            docs = sorted(range(80), key=lambda d: h(run, q, d))[:20]
            con.executemany("INSERT INTO results (run_id, query_id, doc_id, rank) VALUES (?, ?, ?, ?)",
                            [(run, q, str(d), i + 1) for i, d in enumerate(docs)])
    con.execute("""INSERT INTO judgments (dataset, query_id, doc_id, source, grade)
                   SELECT 'wands', query_id, doc_id, 'jev-batched:jev-1.13.0', CAST(hash(query_id || doc_id) % 4 AS INTEGER)
                   FROM (SELECT DISTINCT r.query_id, r.doc_id FROM results r JOIN runs USING (run_id)
                         WHERE runs.dataset = 'wands' AND r.rank <= 10)""")
    con.close()
    m.run_report(None)
    out = capsys.readouterr().out
    assert "tau" in out and "pooled kappa" in out and any(w in out for w in ("HOLDS", "FAILS", "PARTIAL"))


def test_trec(world, capsys, monkeypatch):
    m = world.study("judge-trec/trec.py")
    queries = [SimpleNamespace(query_id=str(q), text=f"what is {q}") for q in range(12)]
    qrels = [SimpleNamespace(query_id=str(q), doc_id=f"d{q}-{k}", relevance=k % 4) for q in range(12) for k in range(8)]
    docs = {r.doc_id: SimpleNamespace(text=f"passage {r.doc_id}") for r in qrels}
    ds = SimpleNamespace(queries_iter=lambda: iter(queries), qrels_iter=lambda: iter(qrels),
                         docs_store=lambda: SimpleNamespace(get_many=lambda ids: {i: docs[i] for i in ids}))
    monkeypatch.setitem(sys.modules, "ir_datasets", types.SimpleNamespace(load=lambda name: ds))
    args = SimpleNamespace(year="2020", model="jev-1.14.0", runs_dir=None)
    m.run_prepare(args)
    m.run_grade(args)
    m.run_report(args)
    runs = world.tmp / "runs"
    runs.mkdir()
    for s in range(3):
        (runs / f"sys{s}").write_text("".join(f"{r.query_id} Q0 {r.doc_id} {int(h(s, r.doc_id) * 1000)} 1.0 sys{s}\n"
                                              for r in qrels))
    m.run_report(SimpleNamespace(year="2020", model="jev-1.14.0", runs_dir=str(runs)))
    out = capsys.readouterr().out
    assert "weighted kappa" in out and "No --runs-dir" in out and "tau over 3 systems" in out
