def test_cascade_end_to_end(world, monkeypatch, capsys):
    m = world.study("relevance-reranker-cascade/cascade.py")
    m.run_ensembles(None)
    out = capsys.readouterr().out
    assert "mean rank" in out and "score mix" in out and "corrected for" in out
    assert "score mix w=1.0" not in out and "score mix w=0.0" not in out
    for step in (m.run_candidates, m.run_voyage, m.run_jev, m.run_cascade):
        step(None)
    m.run_voyage(None)  # resumes: nothing appended twice
    lines = (m.DATA / "voyage.jsonl").read_text().splitlines()
    assert len(lines) == len({line.split('"query_id": ')[1][:8] for line in lines})
    assert len(m.load_rows("candidates")["1000"]["ids"]) == 100
    assert m.load_rows("cascade")["1000"]["ids"][20:] == m.load_rows("voyage")["1000"]["ids"][20:]
    m.run_report(None)
    out = capsys.readouterr().out
    assert "cascade" in out and "escalate" in out and "set VOYAGE_PRICE and JEV_PRICE" in out
    monkeypatch.setattr(m, "PRICES", {"voyage": "0.002", "jev": "0.0001"})
    m.run_report(None)
    out = capsys.readouterr().out
    assert "a query, cheaper than Voyage" in out or "a query, not cheaper than Voyage" in out
