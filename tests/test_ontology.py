from types import SimpleNamespace

TAXONOMY = """# Google_Product_Taxonomy_Version: 2021-09-21
1 - Home & Garden
2 - Home & Garden > Kitchen & Dining
3 - Home & Garden > Kitchen & Dining > Cookware
4 - Home & Garden > Kitchen & Dining > Cookware > Skillets
5 - Apparel & Accessories
6 - Apparel & Accessories > Shoes
7 - Electronics
8 - Electronics > Chargers
"""


def test_ontology_end_to_end(world, capsys):
    m = world.study("relevance-ontology/ontology_search.py")
    m.DATA.mkdir(parents=True)
    (m.DATA / "taxonomy-with-ids.en-US.txt").write_text(TAXONOMY)
    args = SimpleNamespace(scope="full")
    for step in ("taxonomy", "tag_docs", "patch", "tag_queries", "run", "judge", "report"):
        getattr(m, f"run_{step}")(args)
    out = capsys.readouterr().out
    assert "## Tagging" in out and "onto-soft" in out and "oracle-soft" in out
    paths = [v.get("category_path") for v in world.tp.attrs["esci-us-full"].values()]
    assert any(paths) and all(p is None or p[0] in {"1", "5", "7"} for p in paths)  # every path starts at a root
