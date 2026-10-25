def recall_fusion(world, monkeypatch):
    rf = world.study("recall-fusion/recall_fusion.py")
    for step in ("fetch", "check", "diagnose", "sweep"):
        getattr(rf, f"run_{step}")()
    return rf


def test_recall_fusion(world, monkeypatch, capsys):
    rf = recall_fusion(world, monkeypatch)
    rf.run_rerank()
    out = capsys.readouterr().out
    assert "best top 100 vs depth 100 top 100" in out
    assert "Recall of Exact and Substitute products, averaged per query" in out


def test_routing(world, monkeypatch, capsys):
    rf = recall_fusion(world, monkeypatch)
    m = world.study("relevance-routing/routing.py")
    for attr in ("DATA", "OUT"):
        monkeypatch.setattr(m.RF, attr, getattr(rf, attr))
    monkeypatch.setattr(m.RF, "client", lambda *a, **k: world.tp)
    m.run_slices(None)
    m.run_classify(None)
    m.run_route(None)
    m.run_rerank(None)
    m.run_report(None)
    out = capsys.readouterr().out
    assert "routed+voyage vs best single+voyage" in out
    assert "tuning queries ->" in out
