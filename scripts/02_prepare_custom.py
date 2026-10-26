"""Step 2 (your own data): load your corpus and queries into the warehouse.

Use this instead of 02_prepare_esci.py when you have your own documents and
queries. Files can be CSV, Parquet, JSON or JSONL; DuckDB detects the format.

  docs        required: id. Optional: title, body, brand, color. You need at
              least one of title or body ("text" is accepted in place of body).
  queries     required: query_id and text ("query" is accepted in place of
              text). Optional: slice, a label used to break metrics down
              (default "all").
  judgments   optional human labels: query_id, doc_id, and grade, an integer
              from 0 to 3 (3 is best). Leave it out if you have none.

Without human labels, the judge's grades are the labels. That is why steps 0
to 8 on ESCI come first: they show how far the judge can be trusted against
human grades before you rely on it for data that has none.

Run it:
  uv run scripts/02_prepare_custom.py --dataset mydocs --docs docs.parquet --queries queries.csv
  uv run scripts/02_prepare_custom.py --dataset mydocs --docs docs.parquet --queries queries.csv \\
      --judgments labels.jsonl --rubric generic --sample 200

The dataset name becomes a turbopuffer namespace prefix, so use only
lowercase letters, digits and hyphens. Names starting with "esci" belong to
02_prepare_esci.py. Re-running replaces that dataset's docs, queries and human
judgments, and leaves every other dataset alone.
"""

import argparse
import re
from pathlib import Path

from relevancy import rubric
from relevancy.warehouse import connect


def read_file(con, path: str, table: str) -> list[str]:
    """Load a CSV, Parquet, JSON or JSONL file into a temp table; return its columns."""
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"No such file: {path}")
    suffix = p.suffix.lower()
    # The path is bound, not formatted into the SQL, so quotes in it are safe.
    if suffix in (".json", ".jsonl", ".ndjson"):
        reader = "read_json_auto"
    elif suffix == ".parquet":
        reader = "read_parquet"
    else:
        reader = "read_csv"
    # CSV ids are read as text so "007" stays "007" instead of becoming 7.
    extra = ", all_varchar = true" if reader == "read_csv" else ""
    try:
        con.execute(f"CREATE OR REPLACE TEMP TABLE {table} AS SELECT * FROM {reader}(?{extra})", [str(p)])
    except Exception as e:
        raise SystemExit(f"Could not read {path}: {e}") from e
    return [r[0] for r in con.execute(f"DESCRIBE {table}").fetchall()]


def pick(cols: list[str], *names: str) -> str | None:
    """First of `names` that the file has, else None."""
    return next((n for n in names if n in cols), None)


def need(cols: list[str], what: str, path: str, *names: str) -> str:
    found = pick(cols, *names)
    if found is None:
        raise SystemExit(f"{path}: needs a {' or '.join(repr(n) for n in names)} column for {what}. "
                         f"Columns found: {', '.join(cols)}")
    return found


def scalar(con, sql: str, params: list | None = None):
    """First value of the first row. fetchall() finishes the result, so the connection stays usable."""
    return con.execute(sql, params or []).fetchall()[0][0]


def q(name: str) -> str:
    """Quote an identifier, since user files can name columns anything."""
    return '"' + name.replace('"', '""') + '"'


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="name for this dataset: lowercase letters, digits, hyphens")
    ap.add_argument("--docs", required=True, help="documents file (CSV, Parquet, JSON or JSONL)")
    ap.add_argument("--queries", required=True, help="queries file (CSV, Parquet, JSON or JSONL)")
    ap.add_argument("--judgments", help="optional human labels file: query_id, doc_id, grade (0 to 3)")
    ap.add_argument("--rubric", default="generic", help="rubric the judges grade with (rubrics/<name>.toml)")
    ap.add_argument("--sample", type=int, help="queries to evaluate on (default: all of them)")
    ap.add_argument("--seed", type=int, default=42, help="change it to draw a different sample")
    ap.add_argument("--description", help="a note about the dataset, stored with it")
    args = ap.parse_args()
    ds = args.dataset

    # Dataset names become turbopuffer namespace prefixes, so keep them plain.
    if not re.fullmatch(r"[a-z0-9-]+", ds):
        raise SystemExit(f"--dataset {ds!r}: use only lowercase letters, digits and hyphens")
    if ds.startswith("esci"):
        raise SystemExit("Dataset names starting with 'esci' are reserved for scripts/02_prepare_esci.py")
    if args.sample is not None and args.sample < 1:
        raise SystemExit("--sample must be at least 1")

    # Check everything before writing anything.
    rubric.load(args.rubric)

    con = connect()

    # Docs: id plus at least one of title/body. Missing columns become NULL.
    dcols = read_file(con, args.docs, "src_docs")
    d_id = need(dcols, "the document id", args.docs, "id")
    d_title = pick(dcols, "title")
    d_body = pick(dcols, "body", "text")
    if d_title is None and d_body is None:
        raise SystemExit(f"{args.docs}: needs a 'title' or 'body' column (or 'text' in place of body). "
                         f"Columns found: {', '.join(dcols)}")
    d_brand, d_color = pick(dcols, "brand"), pick(dcols, "color")

    def opt(col: str | None) -> str:
        return f"CAST({q(col)} AS VARCHAR)" if col else "CAST(NULL AS VARCHAR)"

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE new_docs AS
        SELECT CAST({q(d_id)} AS VARCHAR) AS doc_id, {opt(d_title)} AS title, {opt(d_body)} AS body,
               {opt(d_brand)} AS brand, {opt(d_color)} AS color
        FROM src_docs
    """)
    bad = scalar(con, "SELECT count(*) FROM new_docs WHERE doc_id IS NULL OR doc_id = ''")
    if bad:
        raise SystemExit(f"{args.docs}: {bad} rows have an empty id")
    dup = con.execute("SELECT doc_id FROM new_docs GROUP BY doc_id HAVING count(*) > 1 LIMIT 3").fetchall()
    if dup:
        raise SystemExit(f"{args.docs}: duplicate ids, for example {', '.join(r[0] for r in dup)}")

    # Queries: query_id and text (or query); slice is optional.
    qcols = read_file(con, args.queries, "src_queries")
    q_id = need(qcols, "the query id", args.queries, "query_id")
    q_text = need(qcols, "the query text", args.queries, "text", "query")
    q_slice = pick(qcols, "slice")
    slice_sql = f"coalesce(CAST({q(q_slice)} AS VARCHAR), 'all')" if q_slice else "'all'"
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE new_queries AS
        SELECT CAST({q(q_id)} AS VARCHAR) AS query_id, CAST({q(q_text)} AS VARCHAR) AS text, {slice_sql} AS slice
        FROM src_queries
    """)
    bad = scalar(con, "SELECT count(*) FROM new_queries WHERE query_id IS NULL OR query_id = '' "
                 "OR text IS NULL OR trim(text) = ''")
    if bad:
        raise SystemExit(f"{args.queries}: {bad} rows have an empty query_id or text")
    dup = con.execute("SELECT query_id FROM new_queries GROUP BY query_id HAVING count(*) > 1 LIMIT 3").fetchall()
    if dup:
        raise SystemExit(f"{args.queries}: duplicate query_id, for example {', '.join(r[0] for r in dup)}")

    # Human judgments, if any. A grade outside 0..3 is almost always a scale mix-up.
    have_judgments = args.judgments is not None
    if have_judgments:
        jcols = read_file(con, args.judgments, "src_judgments")
        j_q = need(jcols, "the query id", args.judgments, "query_id")
        j_d = need(jcols, "the document id", args.judgments, "doc_id")
        j_g = need(jcols, "the grade", args.judgments, "grade")
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE new_judgments AS
            SELECT CAST({q(j_q)} AS VARCHAR) AS query_id, CAST({q(j_d)} AS VARCHAR) AS doc_id,
                   TRY_CAST({q(j_g)} AS DOUBLE) AS g, {q(j_g)} AS raw
            FROM src_judgments
        """)
        out = con.execute("SELECT query_id, doc_id, raw FROM new_judgments "
                          "WHERE g IS NULL OR g NOT BETWEEN 0 AND 3 OR g <> floor(g) LIMIT 3").fetchall()
        if out:
            seen = "; ".join(f"{a}/{b}: {c!r}" for a, b, c in out)
            raise SystemExit(f"{args.judgments}: grade must be an integer from 0 to 3. Found {seen}")
        dup = con.execute("SELECT query_id, doc_id FROM new_judgments GROUP BY ALL HAVING count(*) > 1 LIMIT 3").fetchall()
        if dup:
            raise SystemExit(f"{args.judgments}: more than one grade for the same pair, for example {dup[0]}")

    # One transaction, so a failure partway leaves the warehouse as it was.
    con.begin()

    con.execute("INSERT OR REPLACE INTO datasets (dataset, rubric, description) VALUES (?, ?, ?)",
                [ds, args.rubric, args.description])

    con.execute("DELETE FROM docs WHERE dataset = ?", [ds])
    con.execute("INSERT INTO docs SELECT ?, doc_id, title, body, brand, color FROM new_docs", [ds])

    # Only human labels are replaced; judge grades from later steps stay put.
    con.execute("DELETE FROM judgments WHERE dataset = ? AND source = 'human'", [ds])
    if have_judgments:
        con.execute("""
            INSERT INTO judgments (dataset, query_id, doc_id, source, grade)
            SELECT ?, query_id, doc_id, 'human', CAST(g AS INTEGER) FROM new_judgments
        """, [ds])

    con.execute("DELETE FROM queries WHERE dataset = ?", [ds])
    con.execute("INSERT INTO queries SELECT ?, query_id, text, 'custom', slice, FALSE FROM new_queries", [ds])

    # Seeded sample, or every query when --sample is absent or covers them all.
    # DuckDB's hash() can change between major versions, so a DuckDB upgrade
    # can change the sample; runs record a sample fingerprint, so scoring notices.
    n_queries = scalar(con, "SELECT count(*) FROM queries WHERE dataset = ?", [ds])
    if args.sample is not None and args.sample < n_queries:
        con.execute("""
            UPDATE queries SET sampled = TRUE
            WHERE dataset = ? AND query_id IN (
                SELECT query_id FROM queries WHERE dataset = ?
                ORDER BY hash(query_id || ?) LIMIT ?
            )
        """, [ds, ds, str(args.seed), args.sample])
    else:
        con.execute("UPDATE queries SET sampled = TRUE WHERE dataset = ?", [ds])

    con.commit()

    # Labels that point at nothing are usually an id mismatch. Warn, don't fail.
    if have_judgments:
        no_doc = scalar(con, """
            SELECT count(DISTINCT doc_id) FROM judgments
            WHERE dataset = ? AND source = 'human'
              AND doc_id NOT IN (SELECT doc_id FROM docs WHERE dataset = ?)
        """, [ds, ds])
        no_query = scalar(con, """
            SELECT count(DISTINCT query_id) FROM judgments
            WHERE dataset = ? AND source = 'human'
              AND query_id NOT IN (SELECT query_id FROM queries WHERE dataset = ?)
        """, [ds, ds])
        if no_doc:
            print(f"WARNING: {no_doc} judged doc_ids are not in the docs file.")
        if no_query:
            print(f"WARNING: {no_query} judged query_ids are not in the queries file.")

    print(con.sql(f"""
        SELECT (SELECT count(*) FROM docs WHERE dataset = '{ds}') AS docs,
               (SELECT count(*) FROM queries WHERE dataset = '{ds}') AS queries,
               (SELECT count(*) FROM queries WHERE dataset = '{ds}' AND sampled) AS sampled,
               (SELECT count(*) FROM judgments WHERE dataset = '{ds}' AND source = 'human') AS human_judgments,
               '{args.rubric}' AS rubric
    """))
    if not have_judgments:
        print("No human labels: the judge's grades will be the labels (see the docstring).")
    print(f"""
Next:
  uv run scripts/03_load_turbopuffer.py --dataset {ds} --scope full
  uv run scripts/04_search.py --dataset {ds} --scope full
  uv run scripts/08_judge_runs.py --dataset {ds} --judge jev-batched:jev-1.13.0
(Step 6 adds rerankers, if you want to compare them.)""")


if __name__ == "__main__":
    main()
