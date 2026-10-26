def test_filter_cascades(world, monkeypatch, capsys):
    m = world.study("database-semantic-operators/filters.py")
    monkeypatch.setattr(m, "POOL", 300)
    monkeypatch.setattr(m, "SAMPLE", 200)
    monkeypatch.setattr(m, "JEV_YES", 80)
    monkeypatch.setattr(m, "SPLITS", 3)
    for step in ("pool", "sample", "oracle", "escalate", "embed", "check_export", "report"):
        getattr(m, f"run_{step}")(None)
    out = capsys.readouterr().out
    assert out.count("target 0.9:") == 6 * 2 and out.count("target 0.95:") == 6 * 2  # conditions x stages
    sample = m.read_csv(m.OUT / "sample-replacement_part.csv")
    # Weights add back up to the pool, per stratum.
    assert abs(sum(float(r["weight"]) for r in sample) - 300) < 1e-3
    rows = m.read_csv(m.OUT / "cascades.csv")
    assert all(0 <= float(r["llm_share"]) <= 1 for r in rows)
