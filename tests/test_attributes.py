import csv
from types import SimpleNamespace

import pytest


def test_attributes_end_to_end(world, capsys):
    m = world.study("relevance-attributes/attributes.py")
    args = SimpleNamespace(attribute="replacement_part")
    m.run_find(args)
    found = m.read_csv(m.OUT / "queries-replacement_part.csv")
    assert found and all("replacement" in r["text"] for r in found)
    m.run_label(args)
    m.run_label(args)  # resumes with nothing left to do
    assert len(m.jev_labels("replacement_part")) == len(world.docs)
    m.run_check_export(args)
    rows = m.read_csv(m.OUT / "check-replacement_part.csv")
    assert "jev_yes" not in rows[0]  # the rater never sees Jev's answer
    def rate(fn):
        with open(m.OUT / "check-replacement_part.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows([r | {"rater": fn(i)} for i, r in enumerate(rows)])

    rate(lambda i: "yes" if i % 3 else "no")  # a third rated no: precision under 0.9 must stop the run
    with pytest.raises(SystemExit):
        m.run_precision(args)
    rate(lambda i: "yes")  # every Jev-yes rated yes: precision 1.0 passes
    m.run_precision(args)
    m.run_patch(args)
    assert any(v.get("replacement_part") for v in world.tp.attrs["esci-us-full"].values())
    m.run_run(args)
    m.run_report(args)
    out = capsys.readouterr().out
    assert "precision" in out and "attr-soft" in out and "(primary)" in out
