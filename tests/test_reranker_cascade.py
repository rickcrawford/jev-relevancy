def test_cascade_end_to_end(world, capsys):
    m = world.study("relevance-reranker-cascade/cascade.py")
    m.run_ensembles(None)
    out = capsys.readouterr().out
    assert "mean rank" in out and "score mix" in out and "TOST" in out
    for step in (m.run_candidates, m.run_voyage, m.run_jev, m.run_cascade):
        step(None)
    assert len(m.load_rows("candidates")["1000"]["ids"]) == 100
    assert m.load_rows("cascade")["1000"]["ids"][20:] == m.load_rows("voyage")["1000"]["ids"][20:]
    m.run_report(None)
    out = capsys.readouterr().out
    assert "cascade" in out and "escalate" in out and "sent to Voyage" in out
    assert world.jev.requests > 0
