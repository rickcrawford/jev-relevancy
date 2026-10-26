import sys
import types


def test_constraints_end_to_end(world, capsys, monkeypatch):
    m = world.study("relevance-constraints/constraints.py")
    m.run_extract(None)
    m.run_check(None)
    rows = m.read_csv(m.DATA / "checks.csv")
    assert {r["type"] for r in rows} == {"brand", "excludes"}
    assert world.jev.requests >= 40 * 2  # 50 products x 2 constraints = 2 requests a query
    m.run_report(None)
    out = capsys.readouterr().out
    assert "voyage + checks" in out and "jev + checks" in out and "judge-negation-audit" in out


def test_nevir(world, capsys, monkeypatch):
    m = world.study("relevance-constraints/constraints.py")
    rows = [{"q1": f"q{i}", "q2": f"not q{i}", "doc1": f"doc a {i}", "doc2": f"doc b {i}"} for i in range(20)]
    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_dataset=lambda *a, **k: {"test": rows}))
    m.run_nevir(None)
    assert "pairwise accuracy" in capsys.readouterr().out
