from types import SimpleNamespace


def test_rewrites_and_expansion(world, capsys):
    m = world.study("recall-rewrites/rewrites.py")
    args = SimpleNamespace(scope="full")
    m.run_rewrite(args)
    m.run_drift(args)
    m.run_run(args)
    assert m.fused("q", ["a", "b", "c"])[1] == [3.0, 3.0, 1, 1, 1, 1, 1, 1]  # the original keeps half the weight
    m.run_expand(args)
    m.run_expand(args)  # resumes
    m.run_patch_expansion(args)
    assert all("expansion" in v for v in world.tp.attrs["esci-us-full"].values())
    m.run_run_expansion(args)
    m.run_report(args)
    out = capsys.readouterr().out
    assert "kept" in out and "## rewrites" in out and "## expansion" in out and "nDCG@10 + Voyage" in out
