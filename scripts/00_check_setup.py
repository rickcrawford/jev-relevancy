"""Step 0: check that everything this walkthrough needs is in place.

  uv run scripts/00_check_setup.py

Checks your turbopuffer key and region, the DuckDB warehouse, the ESCI files,
and (once a namespace is loaded) which hosted rerankers your org can call.
Nothing here costs more than a few fractions of a cent.
"""

import os

from relevancy.tpuf import REGION, client, rerank
from relevancy.warehouse import ROOT, connect

RERANKERS = ["typesafe/jev-1.13.0", "voyage/rerank-3", "voyage/rerank-3-lite",
             "cohere/rerank-v3.5", "zeroentropy/zerank-2"]


def ok(msg: str) -> None:
    print(f"  ok    {msg}")


def warn(msg: str) -> None:
    print(f"  todo  {msg}")


def main() -> None:
    print("turbopuffer")
    tp = client()
    ok(f"TURBOPUFFER_API_KEY is set, region {REGION}")
    names = tp.namespaces()["namespace"].to_pylist()
    ok(f"API reachable, {len(names)} namespace(s): {', '.join(names) or 'none yet'}")

    print("warehouse")
    con = connect()
    counts = con.execute("""
        SELECT (SELECT count(*) FROM docs), (SELECT count(*) FROM queries WHERE sampled),
               (SELECT count(*) FROM runs)
    """).fetchone()
    ok(f"DuckDB open: {counts[0]:,} docs, {counts[1]} sampled queries, {counts[2]} runs")

    print("data")
    raw = ROOT / "data" / "raw" / "esci"
    files = sorted(p.name for p in raw.glob("*")) if raw.exists() else []
    if len(files) == 3:
        ok(f"ESCI files present in {raw.relative_to(ROOT)}")
    else:
        warn("ESCI files missing. Run scripts/01_download_esci.sh")
    if counts[0] == 0:
        warn("warehouse is empty. Run scripts/02_prepare_esci.py")

    print("rerankers")
    # The rerankers read the `text` attribute, so probe a namespace step 3 loaded:
    # an ESCI one if there is one, or else the first with `text` among the others.
    loaded = [n for n in names if n.startswith("esci-us-")] or [
        n for n in names[:20] if "text" in tp.attr_types(n)][:1]
    if not loaded:
        warn("load a namespace first (scripts/03_load_turbopuffer.py), then rerun this check")
        return
    first = tp.query(loaded[0], ("id", "asc"), top_k=1)["id"].to_pylist()
    if not first:
        warn(f"{loaded[0]} is empty; rerun scripts/03_load_turbopuffer.py")
        return
    probe = first[0]
    for model in RERANKERS:
        try:
            rerank(tp, loaded[0], model, "test", [probe])
            ok(model)
        except Exception as e:  # noqa: BLE001  the API explains what's wrong; show it
            msg = str(e).split("rerank_by`:")[-1].split("', 'status'")[0].strip(" '")
            warn(f"{model}: {msg[:110]}")

    if not os.environ.get("TYPESAFE_API_KEY"):
        print("\n  Jev can also be called directly through TypeSafe's API (TYPESAFE_API_KEY);")
        print("  that path is needed for steps 7 and 8, where Jev grades results.")


if __name__ == "__main__":
    main()
