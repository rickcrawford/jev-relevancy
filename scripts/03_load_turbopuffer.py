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

Re-running is safe: writes are upserts keyed on the product id. If a load
stops partway (a long full load can outlast the rate-limit retries), rerun
with --resume to write only the products the namespace doesn't have yet,
instead of paying to embed everything again.
"""

import argparse
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from turbopuffer import RateLimitError

from relevancy.tpuf import EMBED_MODEL, REGION, client, namespace_name, schema
from relevancy.warehouse import connect

BATCH = 256  # turbopuffer's per-request cap when a batch has embedded attributes


class Pacer:
    """Spread requests out so estimated embedding tokens stay under a per-minute budget.

    New orgs get 2M embedding tokens per minute per model. Without pacing,
    eight writers blow through that in a few seconds and start getting 429s.
    """

    def __init__(self, tokens_per_minute: int):
        self.per_second = tokens_per_minute / 60
        self.next_free = time.monotonic()
        self.lock = threading.Lock()

    def wait(self, tokens: int) -> None:
        with self.lock:
            start = max(time.monotonic(), self.next_free)
            self.next_free = start + tokens / self.per_second
        time.sleep(max(0.0, start - time.monotonic()))


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

    con = connect()
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
    rows = con.execute(f"""
        SELECT doc_id, coalesce(title, ''), coalesce(body, ''),
               concat_ws('. ', title, brand, color, left(body, 800)),
               brand, color
        FROM docs WHERE dataset = ? {scope_filter}
        ORDER BY doc_id
    """, [args.dataset]).fetchall()
    con.close()  # release the file lock so other scripts can use the warehouse during the load

    ns_name = namespace_name(args.dataset, args.scope)
    ns = client().namespace(ns_name)

    if args.resume:
        # turbopuffer has no export endpoint; the documented way to list a
        # namespace is to page through it in id order, 10,000 ids at a time.
        have, last = set(), None
        while True:
            page = ns.query(rank_by=("id", "asc"), top_k=10_000,
                            filters=("id", "Gt", last) if last is not None else None).rows or []
            have.update(str(r.id) for r in page)
            if len(page) < 10_000:
                break
            last = page[-1].id
        rows = [r for r in rows if r[0] not in have]
        print(f"--resume: {len(have):,} products already in {ns_name}, {len(rows):,} to go", flush=True)
        if not rows:
            return
    print(f"Loading {len(rows):,} products into {ns_name} ({REGION}), embedding with {EMBED_MODEL}", flush=True)

    batches = [rows[i:i + BATCH] for i in range(0, len(rows), BATCH)]
    totals = {"rows": 0, "bytes": 0, "tokens": 0, "failed": 0}
    started = time.perf_counter()

    pacer = Pacer(args.tokens_per_minute)

    def write(batch):
        # ~4 characters per token; turbopuffer reported 4.0 for these products.
        pacer.wait(sum(len(row[3]) for row in batch) // 4)
        for attempt in range(6):
            try:
                return ns.write(
                    upsert_rows=[
                        {"id": d, "title": t, "body": b, "text": x, "brand": br, "color": c}
                        for d, t, b, x, br, c in batch
                    ],
                    distance_metric="cosine_distance",
                    schema=schema(),
                )
            except RateLimitError:
                if attempt == 5:
                    return None  # counted below; --resume picks these up later
                time.sleep(15 * 2 ** attempt)  # 15s, 30s, 60s, ... gives the minute window time to reset

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(write, b) for b in batches]
        for i, fut in enumerate(as_completed(futures), 1):
            resp = fut.result()
            if resp is None:
                totals["failed"] += 1
                continue
            totals["rows"] += resp.rows_upserted or 0
            totals["bytes"] += resp.billing.billable_logical_bytes_written or 0
            totals["tokens"] += getattr(resp.performance, "embedding_tokens", 0) or 0
            if i % 20 == 0 or i == len(batches):
                elapsed = time.perf_counter() - started
                print(f"  {i:>5}/{len(batches)} batches  {totals['rows']:>8,} rows  "
                      f"{totals['tokens'] / 1e6:6.1f}M tokens  {elapsed / 60:5.1f} min", flush=True)

    elapsed = time.perf_counter() - started
    print(f"Done in {elapsed / 60:.1f} min: {totals['rows']:,} rows, "
          f"{totals['bytes'] / 1e6:,.0f} MB billable writes, {totals['tokens'] / 1e6:.1f}M embedding tokens")
    if totals["failed"]:
        print(f"{totals['failed']} batches still hit the rate limit after retries. "
              f"Rerun with --resume to load just those.")


if __name__ == "__main__":
    main()
