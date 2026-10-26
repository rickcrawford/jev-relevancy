import sys
import types
from types import SimpleNamespace


def test_stark(world, monkeypatch, capsys):
    m = world.study("benchmark-instructions-stark/stark.py")
    docs = {i: f"product {i} with features" for i in range(300)}
    queries = [(str(q), f"a product with feature {q}", [q, q + 1]) for q in range(15)]
    monkeypatch.setattr(m, "load_stark", lambda: (queries, docs))
    monkeypatch.setattr(m, "bm25_top", lambda d, qs, k: [list(range(i, i + k)) for i in range(len(qs))])
    args = SimpleNamespace(limit=None)
    for step in ("retrieve", "rerank", "extract", "check", "report"):
        getattr(m, f"run_{step}")(args)
    out = capsys.readouterr().out
    assert "jev + checks" in out and "Hit@1" in out


def test_followir_pmrr(world):
    m = world.study("benchmark-instructions-stark/followir.py")
    # The altered instruction makes the changed documents irrelevant, so moving one down
    # (rank 1 to rank 4) is right and scores positive; moving it up scores negative. MTEB's formula.
    assert m.rank_score(1, 4) > 0 > m.rank_score(4, 1)
    og = {"a": 4, "b": 3, "c": 2, "d": 1}   # a ranks first with the original instruction
    new = {"a": 1, "b": 3, "c": 2, "d": 4}  # and last with the changed one
    assert m.pmrr_by_query({"q-og": og, "q-changed": new}, {"q": ["a"]})["q"] == 0.75
    assert m.pmrr_by_query({"q-og": new, "q-changed": og}, {"q": ["a"]})["q"] < 0


def test_followir_end_to_end(world, monkeypatch, capsys):
    m = world.study("benchmark-instructions-stark/followir.py")
    rows = {
        "corpus": [{"_id": f"d{i}", "title": "", "text": f"doc {i}"} for i in range(20)],
        "queries": [{"_id": f"q{i}-{v}", "text": f"query {i}"} for i in range(4) for v in ("og", "changed")],
        "instruction": [{"query-id": f"q{i}-{v}", "instruction": f"{v} rules"} for i in range(4) for v in ("og", "changed")],
        "top_ranked": [{"query-id": f"q{i}-{v}", "corpus-ids": [f"d{j}" for j in range(10)]}
                       for i in range(4) for v in ("og", "changed")],
        "default": [{"query-id": f"q{i}-og", "corpus-id": "d1", "score": 1} for i in range(4)],
        "qrel_diff": [{"query-id": f"q{i}", "corpus-ids": ["d1", "d2"]} for i in range(4)],
    }
    fake = types.SimpleNamespace(load_dataset=lambda path, config, split=None, revision=None: rows[config],
                                 get_dataset_split_names=lambda *a, **k: ["test"])
    monkeypatch.setitem(sys.modules, "datasets", fake)
    for step in ("load", "rerank", "report"):
        getattr(m, f"run_{step}")(None)
    assert capsys.readouterr().out.count("p-MRR") == 3


def test_bright(world, monkeypatch, capsys):
    m = world.study("benchmark-bright/bright.py")
    def load_dataset(name, config):
        if config == "documents":
            return {d: [{"id": f"{d}-{i}", "content": f"text {i}"} for i in range(150)] for d in m.DOMAINS}
        return {d: [{"id": f"{d}q{i}", "query": f"why {i}", "gold_ids": [f"{d}-{i}"], "excluded_ids": ["N/A"]}
                    for i in range(6)] for d in m.DOMAINS}
    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_dataset=load_dataset))
    args = SimpleNamespace(limit=None)
    for step in ("load", "bm25", "jev", "llm", "report"):
        getattr(m, f"run_{step}")(args)
    out = capsys.readouterr().out
    assert out.count("cascade keeps") == 3
    assert m.cascade({"a": 1, "b": 2, "c": 3}, {"a": 3}, ["a", "b", "c"])[0] in {"a", "b", "c"}
