"""Step 3: load the product corpus into a turbopuffer namespace.

Two scopes:
  smoke  the products judged for the sampled queries, plus random distractors
         (20,000 by default). About 21k products, 3.5M embedding tokens,
         3 minutes. Good for a first pass.
  full   all 482k English products in the reduced ESCI set, 323M characters
         of embedded text. At the 4.0 characters per token turbopuffer reported
         for the first 287k products, that's about 80M embedding tokens (~$1.60
         with voyage-4-lite). Paced under the default 2M tokens/min limit, my
         first attempt wrote 287k products in 42 minutes.

  uv run scripts/03_load_turbopuffer.py --scope smoke
  uv run scripts/03_load_turbopuffer.py --scope full

Each document gets:
  title, body   full-text indexed for BM25
  text          title + brand + color + the first 800 characters of the body,
                embedded by turbopuffer and read by model rerankers
  brand, color  filterable attributes

tpuf-arrow does the writing: 256 products per request (turbopuffer's cap
when it embeds), several requests at once. This script paces the products it
hands over so embedding stays under your token limit. The SDK's own retries
only cover a few seconds, so if requests still hit the rate limit, the script
waits a minute for the limit to reset and loads whatever is missing, up to
three more times. Each write request lands in the warehouse's
`tpuf_requests` table, with the embedding tokens turbopuffer reported for it.

Re-running is safe: writes are upserts keyed on the product id. If a load
stops partway, rerun with --resume to write only the products the namespace
doesn't have yet, instead of paying to embed everything again.
"""

import argparse
import time

import duckdb
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import tpuf_arrow as ta
from turbopuffer import NotFoundError

from relevancy.tpuf import EMBED_MODEL, REGION, client, namespace_name, schema
from relevancy.warehouse import ROOT, connect

BATCH = 256  # turbopuffer's per-request cap when a batch has embedded attributes
ROUNDS = 4  # one load, then up to three more for requests that hit the rate limit


class Pacer:
    """Spread requests out so estimated embedding tokens stay under a per-minute budget.

    New orgs get 2M embedding tokens per minute per model. Without pacing,
    eight writers blow through that in a few seconds and start getting 429s.
    """

    def __init__(self, tokens_per_minute: int):
        self.per_second = tokens_per_minute / 60
        self.next_free = time.monotonic()

    def wait(self, tokens: int) -> None:
        start = max(time.monotonic(), self.next_free)
        self.next_free = start + tokens / self.per_second
        time.sleep(max(0.0, start - time.monotonic()))


def in_namespace(tp: ta.Client, ns_name: str) -> pa.Table:
    """Ids already in the namespace. turbopuffer has no export endpoint, so
    this pages through it in id order, 10,000 ids at a time."""
    try:
        return tp.scan(ns_name, columns=[], run_id=f"load-{ns_name}").read_all()
    except NotFoundError:
        return pa.table({"id": pa.array([], pa.string())})


def open_warehouse(minutes: float = 10) -> duckdb.DuckDBPyConnection:
    """Connect, waiting while another process (the DuckDB CLI, another step) holds the file."""
    deadline = time.monotonic() + minutes * 60
    while True:
        try:
            return connect()
        except duckdb.IOException as e:
            if "lock" not in str(e).lower() or time.monotonic() > deadline:
                raise
            print("  the warehouse is open in another process; trying again in 15 s", flush=True)
            time.sleep(15)


def products_to_load(args, have: pa.Table) -> pa.Table:
    con = open_warehouse()
    con.register("have", have)
    scope_filter = ""
    if args.scope == "smoke":
        # Every product judged for a sampled query, plus a fixed random draw
        # of other products so BM25 and ANN have something to get wrong.
        scope_filter = f"""
            AND (doc_id IN (SELECT j.doc_id FROM judgments j JOIN queries q USING (dataset, query_id)
                            WHERE q.dataset = '{args.dataset}' AND q.sampled AND j.source = 'human')
                 OR doc_id IN (SELECT doc_id FROM docs WHERE dataset = '{args.dataset}'
                               ORDER BY hash(doc_id) LIMIT {args.distractors}))
        """
    products = con.sql(f"""
        SELECT doc_id AS id, coalesce(title, '') AS title, coalesce(body, '') AS body,
               concat_ws('. ', title, brand, color, left(body, 800)) AS text,
               brand, color
        FROM docs WHERE dataset = ? {scope_filter}
          AND doc_id NOT IN (SELECT id FROM have)
        ORDER BY doc_id
    """, params=[args.dataset]).arrow().read_all()
    con.close()  # release the file lock so other scripts can use the warehouse during the load
    return products


def load(tp: ta.Client, ns_name: str, args) -> None:
    empty = pa.table({"id": pa.array([], pa.string())})
    have = in_namespace(tp, ns_name) if args.resume else empty
    products = products_to_load(args, have)
    if args.resume:
        print(f"--resume: {have.num_rows:,} products already in {ns_name}, {products.num_rows:,} to go", flush=True)
        if not products.num_rows:
            return
    print(f"Loading {products.num_rows:,} products into {ns_name} ({REGION}), "
          f"embedding with {EMBED_MODEL}", flush=True)

    pacer = Pacer(args.tokens_per_minute)
    started = time.perf_counter()
    rows = billed = 0

    def paced(table: pa.Table):
        """Hand tpuf-arrow one request's worth of products at a time, no faster than the token budget."""
        sent = 0
        for i, batch in enumerate(table.to_batches(max_chunksize=BATCH), 1):
            # ~4 characters per token; turbopuffer reported 4.0 for these products.
            pacer.wait(pc.sum(pc.utf8_length(batch["text"])).as_py() // 4)
            yield batch
            sent += batch.num_rows
            if i % 20 == 0:
                print(f"  {sent:>8,} rows sent  {(time.perf_counter() - started) / 60:5.1f} min", flush=True)

    for round_ in range(1, ROUNDS + 1):
        reader = pa.RecordBatchReader.from_batches(products.schema, paced(products))
        try:
            stats = tp.upsert(ns_name, reader, schema=schema(), distance_metric="cosine_distance",
                              workers=args.workers, run_id=f"load-{ns_name}")
            rows, billed, errors = rows + stats.rows, billed + stats.billable_bytes, []
        except ta.UpsertError as e:
            rows, billed, errors = rows + e.stats.rows, billed + e.stats.billable_bytes, e.stats.errors
            if not all("RateLimitError" in line for line in errors):
                # A 4xx other than 429 stops the upsert, and loading again would fail the same
                # way. So does a 5xx that outlasted the SDK's retries.
                print(f"Stopped: {e}")
                for line in errors[:5]:
                    print(f"  {line}")
                raise SystemExit(1) from None
        if not errors:
            break
        if round_ == ROUNDS:
            print(f"{len(errors)} requests still hit the rate limit. Rerun with --resume to load the rest.")
            break
        print(f"{len(errors)} requests hit the rate limit after the SDK's retries. "
              f"Waiting a minute, then loading what's missing.", flush=True)
        time.sleep(60)
        products = products_to_load(args, in_namespace(tp, ns_name))

    tokens = pc.sum(tp.log["embedding_tokens"]).as_py() or 0
    print(f"Done in {(time.perf_counter() - started) / 60:.1f} min: {rows:,} rows, "
          f"{billed / 1e6:,.0f} MB billable writes, {tokens / 1e6:.1f}M embedding tokens")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="esci-us")
    ap.add_argument("--scope", choices=["smoke", "full"], default="smoke")
    ap.add_argument("--distractors", type=int, default=20_000, help="smoke scope only")
    ap.add_argument("--workers", type=int, default=8, help="concurrent write requests")
    ap.add_argument("--tokens-per-minute", type=int, default=1_500_000,
                    help="embedding budget to pace against; your org's limit is shown in the console")
    ap.add_argument("--resume", action="store_true",
                    help="skip products already in the namespace")
    args = ap.parse_args()

    ns_name = namespace_name(args.dataset, args.scope)
    tp = client(max_concurrency=args.workers)
    try:
        load(tp, ns_name, args)
    finally:
        # Saved even when the load stops partway, so the writes that did go
        # through, and the scan pages --resume paid for, are on record.
        if tp.log.num_rows:
            try:
                with open_warehouse(minutes=1) as con:
                    tp.save_log(con)
            except duckdb.IOException:
                path = ROOT / "warehouse" / f"tpuf_requests-{time.strftime('%Y%m%d-%H%M%S')}.parquet"
                pq.write_table(tp.log, path)
                print(f"The warehouse is locked, so the request log went to {path.relative_to(ROOT)}. "
                      f"Load it later with: INSERT INTO tpuf_requests BY NAME SELECT * FROM '{path}'")


if __name__ == "__main__":
    main()
