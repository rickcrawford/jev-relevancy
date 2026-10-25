import duckdb
import pytest


def recall_fusion(world, monkeypatch):
    rf = world.study("recall-fusion/recall_fusion.py")
    rf.run_fetch()
    # Against the synthetic stored run, local fusion can't match, and `check` must stop.
    with pytest.raises(SystemExit):
        rf.run_check()
    # Make the stored hybrid run what turbopuffer would have returned, and `check` passes.
    ranked = rf.lists()
    con = duckdb.connect(str(world.db))
    con.execute("DELETE FROM results WHERE run_id = 'hyb'")
    rows = [("hyb", q, d, i + 1) for q in ranked["ann"] for i, d in enumerate(rf.fuse(ranked, q, rf.BASELINE, top=50))]
    con.executemany("INSERT INTO results (run_id, query_id, doc_id, rank) VALUES (?, ?, ?, ?)", rows)
    con.close()
    rf.run_check()
    rf.run_diagnose()
    rf.run_sweep()
    return rf


def test_recall_fusion(world, monkeypatch, capsys):
    rf = recall_fusion(world, monkeypatch)
    rf.run_rerank()
    out = capsys.readouterr().out
    assert "best top 100 vs depth 100 top 100" in out
    assert "Recall of Exact and Substitute products, averaged per query" in out
    assert "At least 0.95 top-50 overlap" in out


def test_routing(world, monkeypatch, capsys):
    rf = recall_fusion(world, monkeypatch)
    m = world.study("relevance-routing/routing.py")
    for attr in ("DATA", "OUT"):
        monkeypatch.setattr(m.RF, attr, getattr(rf, attr))
    monkeypatch.setattr(m.RF, "client", lambda *a, **k: world.tp)
    try:
        m.run_slices(None)
    except SystemExit:
        pass  # one retriever winning every slice is a legitimate stop; carry on to test the rest
    m.run_classify(None)
    m.run_route(None)
    m.run_rerank(None)
    m.run_report(None)
    out = capsys.readouterr().out
    assert "routed+voyage vs best single+voyage" in out
    assert "tuning queries ->" in out
