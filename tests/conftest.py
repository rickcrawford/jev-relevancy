import importlib.util
from pathlib import Path

import pytest
from fakes import FakeJev, FakeTp, build_warehouse, fake_complete

ROOT = Path(__file__).resolve().parents[1]


def load_study(rel: str):
    """Import a study script (they live in hyphenated folders, so not as packages)."""
    path = ROOT / "experiments" / rel
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def world(tmp_path, monkeypatch):
    """A synthetic warehouse, fake services, and a study loader that points the study's
    outputs into tmp_path."""
    db = tmp_path / "w.duckdb"
    queries, docs = build_warehouse(db)
    monkeypatch.setenv("RELEVANCY_DB", str(db))
    for key in ("TURBOPUFFER_API_KEY", "TYPESAFE_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.setenv(key, "fake")
    tp, jev = FakeTp(docs), FakeJev()

    import relevancy.jev
    import relevancy.llm_judge
    monkeypatch.setattr(relevancy.jev, "client", lambda: jev)
    monkeypatch.setattr(relevancy.llm_judge, "complete", fake_complete)
    monkeypatch.setattr(relevancy.llm_judge, "client", lambda: None)

    def study(rel: str):
        m = load_study(rel)
        for name in ("client",):
            if hasattr(m, name):
                monkeypatch.setattr(m, name, lambda *a, **k: tp)
        if hasattr(m, "OUT"):
            monkeypatch.setattr(m, "OUT", tmp_path / "out")
        if hasattr(m, "DATA"):
            monkeypatch.setattr(m, "DATA", tmp_path / "data")
        return m

    return type("World", (), {"tp": tp, "jev": jev, "queries": queries, "docs": docs, "tmp": tmp_path,
                              "study": staticmethod(study), "db": db})
