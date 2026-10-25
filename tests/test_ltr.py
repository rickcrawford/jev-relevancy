from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
from fakes import h


def raw_esci(root, docs):
    """A tiny raw ESCI: train queries over the warehouse's products."""
    raw = root / "data" / "raw" / "esci"
    raw.mkdir(parents=True)
    rows = []
    for q in range(30):
        for d in docs[q * 7:q * 7 + 12]:
            rows.append({"query_id": 5000 + q, "query": f"acme red shoe {q}" if q % 3 else f"steel pan {q}",
                         "product_id": d, "esci_label": "ESCI"[int(h(q, d) * 4)], "split": "train",
                         "product_locale": "us", "small_version": 1})
    pq.write_table(pa.Table.from_pylist(rows), raw / "shopping_queries_dataset_examples.parquet")
    pq.write_table(pa.table({"product_id": docs, "product_brand": ["Acme"] * len(docs),
                             "product_locale": ["us"] * len(docs)}), raw / "shopping_queries_dataset_products.parquet")
    return raw


def test_ltr_end_to_end(world, monkeypatch, capsys):
    m = world.study("relevance-ltr/ltr.py")
    monkeypatch.setattr(m, "RAW", raw_esci(world.tmp, world.docs))
    monkeypatch.setattr(m, "GRID", {"num_leaves": [7], "learning_rate": [0.1], "min_child_samples": [5]})
    args = SimpleNamespace(train_queries=20)
    m.run_train_queries(args)
    m.run_features(args)
    m.run_features(args)  # resumes
    feats = m.load_features().to_pylist()
    assert {r["split"] for r in feats} == {"train", "test"}
    assert all(r["jev_p0"] is not None for r in feats)
    m.run_train(args)
    m.run_report(args)
    out = capsys.readouterr().out
    assert "ltr on jev labels" in out and "most important features" in out
