def test_filter_cascades(world, monkeypatch, capsys):
    m = world.study("database-semantic-operators/filters.py")
    monkeypatch.setattr(m, "POOL", 300)
    monkeypatch.setattr(m, "SAMPLE", 200)
    monkeypatch.setattr(m, "JEV_YES", 80)
    monkeypatch.setattr(m, "SPLITS", 3)
    for step in ("pool", "sample", "oracle", "escalate", "embed", "report"):
        getattr(m, f"run_{step}")(None)
    out = capsys.readouterr().out
    assert out.count("sent to LLM") == 6 * 2 * 2  # conditions x targets x stages
    rows = m.read_csv(m.OUT / "cascades.csv")
    assert all(0 <= float(r["llm_share"]) <= 1 for r in rows)
