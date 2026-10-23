"""Step 2: load ESCI into the warehouse and pick the evaluation queries.

Uses the reduced ("small_version") English split, the one the KDD Cup
ranking task used. Writes three tables:

  queries    every English test query, tagged with a slice; the sample is flagged
  docs       every English product in the reduced set (the retrieval corpus)
  judgments  the human ESCI labels for the test queries, as grades 3/2/1/0

Run it:
  uv run scripts/02_prepare_esci.py --sample 50    # smoke test
  uv run scripts/02_prepare_esci.py --sample 500   # the real run
"""

import argparse

from relevancy.warehouse import ROOT, connect

RAW = ROOT / "data" / "raw" / "esci"
DATASET = "esci-us"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", type=int, default=500, help="test queries to evaluate on")
    ap.add_argument("--seed", type=int, default=42, help="change it to draw a different sample")
    args = ap.parse_args()

    con = connect()
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW examples AS
        SELECT CAST(query_id AS VARCHAR) AS query_id, query, product_id, esci_label, split
        FROM '{RAW}/shopping_queries_dataset_examples.parquet'
        WHERE product_locale = 'us' AND small_version = 1;

        CREATE OR REPLACE TEMP VIEW products AS
        SELECT * FROM '{RAW}/shopping_queries_dataset_products.parquet'
        WHERE product_locale = 'us';
    """)

    # One transaction, so a failure partway leaves the warehouse as it was.
    con.begin()

    con.execute("INSERT OR REPLACE INTO datasets VALUES (?, 'esci', ?)",
                [DATASET, "Amazon Shopping Queries (ESCI), reduced English test set"])

    # Corpus: every product judged for any reduced English query, train or test.
    # Train-only products stay in as distractors, which keeps retrieval honest.
    con.execute("DELETE FROM docs WHERE dataset = ?", [DATASET])
    con.execute(f"""
        INSERT INTO docs
        SELECT '{DATASET}', p.product_id,
               p.product_title,
               trim(regexp_replace(
                   concat_ws(' ', p.product_description, p.product_bullet_point),
                   '<[^>]+>', ' ', 'g')),
               p.product_brand, p.product_color
        FROM products p
        WHERE p.product_id IN (SELECT DISTINCT product_id FROM examples)
    """)

    # Human labels for the test queries. Grades follow the ESCI order.
    con.execute("DELETE FROM judgments WHERE dataset = ? AND source = 'human'", [DATASET])
    con.execute(f"""
        INSERT INTO judgments (dataset, query_id, doc_id, source, grade)
        SELECT '{DATASET}', query_id, product_id, 'human',
               CASE esci_label WHEN 'E' THEN 3 WHEN 'S' THEN 2 WHEN 'C' THEN 1 ELSE 0 END
        FROM examples WHERE split = 'test'
    """)

    # Slice each test query by length and by whether it names a brand that
    # one of its Exact products carries. The sample is stratified on this.
    con.execute("DELETE FROM queries WHERE dataset = ?", [DATASET])
    con.execute(f"""
        INSERT INTO queries
        WITH q AS (
            SELECT DISTINCT query_id, query FROM examples WHERE split = 'test'
        ),
        branded AS (
            SELECT DISTINCT e.query_id
            FROM examples e JOIN products p USING (product_id)
            WHERE e.split = 'test' AND e.esci_label = 'E'
              AND length(p.product_brand) >= 3
              AND contains(lower(e.query), lower(p.product_brand))
        )
        SELECT '{DATASET}', q.query_id, q.query, 'test',
               CASE WHEN len(regexp_split_to_array(trim(q.query), '\\s+')) <= 2 THEN 'short'
                    WHEN len(regexp_split_to_array(trim(q.query), '\\s+')) <= 5 THEN 'medium'
                    ELSE 'long' END
               || '|' || CASE WHEN b.query_id IS NULL THEN 'no-brand' ELSE 'brand' END,
               FALSE
        FROM q LEFT JOIN branded b USING (query_id)
    """)

    # Proportional stratified sample: rank queries inside each slice by a
    # seeded hash, then take the lowest fractional ranks across all slices.
    # DuckDB's hash() can change between major versions, so a DuckDB upgrade
    # can change the sample; runs record a sample fingerprint, so scoring notices.
    con.execute(f"""
        UPDATE queries SET sampled = TRUE
        WHERE dataset = '{DATASET}' AND query_id IN (
            SELECT query_id FROM (
                SELECT query_id,
                       row_number() OVER (PARTITION BY slice ORDER BY hash(query_id || '{args.seed}'))
                         / count(*) OVER (PARTITION BY slice) AS frac
                FROM queries WHERE dataset = '{DATASET}'
            )
            ORDER BY frac, hash(query_id || '{args.seed}')  -- seeded tie-break, so slices take turns fairly
            LIMIT {args.sample}
        )
    """)

    con.commit()

    print(con.sql(f"""
        SELECT (SELECT count(*) FROM docs WHERE dataset = '{DATASET}') AS docs,
               (SELECT count(*) FROM queries WHERE dataset = '{DATASET}') AS test_queries,
               (SELECT count(*) FROM queries WHERE dataset = '{DATASET}' AND sampled) AS sampled,
               (SELECT count(*) FROM judgments WHERE dataset = '{DATASET}' AND source = 'human') AS judgments
    """))
    print(con.sql(f"""
        SELECT slice, count(*) AS test_queries, count(*) FILTER (WHERE sampled) AS sampled
        FROM queries WHERE dataset = '{DATASET}' GROUP BY slice ORDER BY slice
    """))


if __name__ == "__main__":
    main()
