# A DuckDB connector for turbopuffer

Stage 1 of this plan is built: [tpuf-arrow](https://github.com/rickcrawford/tpuf-arrow), which this repo uses for its turbopuffer requests. This doc is the design as written beforehand, so parts of it describe the repo as it was then.

I propose turning this repo's warehouse into a reusable connector in three stages: Arrow-returning Python helpers first, a shared eval schema alongside them, and a native DuckDB extension only if the first two get real use. turbopuffer is good at retrieval over data that lives on object storage. DuckDB is good at everything you do with the results afterward, and nothing today connects the two except a hand-written loop.

Status as of 2026-10-23. Every turbopuffer and DuckDB claim below comes from a page I fetched (see Sources). Anything I could not confirm is marked "Open" and collected at the end.

## Why DuckDB + turbopuffer

**What turbopuffer is good at.** Data sits in object storage, and query nodes cache a namespace on NVMe after the first query. The docs give warm p50 of 14 ms for a 10M-document vector query and cold p50 of 874 ms. A namespace is its own prefix in object storage, and the limits page says the number of namespaces is unlimited (250M+ seen). It does ANN, BM25, filters, and server-side RRF fusion. It does not do joins, and the docs advise against aggregations on latency-sensitive paths for namespaces over 1M documents.

**What DuckDB is good at.** It runs in your process, reads Arrow and Parquet, reads and writes S3-compatible storage through httpfs, and computes metrics in SQL. It is not a serving layer.

**Why the pair works.** Retrieval answers "which 50 documents?" Analysis asks "how good were they, what did they cost, and what do they join to?" The second question needs a join engine, and the first needs an index. This repo already shows three examples:

- **nDCG in SQL.** `metrics.py` computes nDCG@10, condensed nDCG, Recall@50, and MRR@10 in one query over `results` and `judgments`, with a bootstrap interval in Python. Changing the gain function from `2^g - 1` to linear is a one-word flag (`--gain linear`).
- **Joining hits to full records.** The `results` table stores only ids and ranks. Titles, brands, and human grades come from a join against `docs` and `judgments`, so a retriever run never has to carry payloads.
- **Cost and latency from billing fields.** Each query's `performance.server_total_ms` and `billing.billable_logical_bytes_queried` land in the warehouse. A query over it shows that every first-stage run in my smoke warehouse billed exactly 1,280,000,000 bytes per query, which is turbopuffer's 1.28 GB minimum, and that hybrid-rrf billed 2,560,000,000 because it is two subqueries. That is cheap at list price ($1 per PB queried), but you only see it if you log it.

**When not to use this pairing.**

- Your corpus fits in memory and you need no persistence. DuckDB alone (the community list has `faiss` and `tantivy` entries; I did not evaluate them) is simpler.
- You need joins or aggregates on the request path. turbopuffer will not do them, and DuckDB in the request path means you now operate a database server.
- You need many concurrent writers to the analytics store. I did not verify DuckDB's concurrency model for this doc (Open).
- Your queries are rare enough that cold latency (about 874 ms p50 in the docs) hurts and you cannot warm the cache.

## What exists in this repo

`warehouse/schema.sql` defines six tables: `queries`, `docs`, `judgments`, `runs`, `results`, `metrics`. `tpuf.py` wraps the SDK for four retrievers plus a model reranker call. `warehouse.py` opens the file and applies the schema. A new dataset, retriever, or judge is a new set of rows.

Limits that a connector should fix:

1. **Per-query facts are repeated on every result row.** `latency_ms`, `server_ms`, and `billable_bytes` are copied across 50 rows per query. Summing `billable_bytes` over `results` overcounts cost 50 times. A `query_log` table, one row per request, fixes it.
2. **Missing response fields.** The query response also carries `cache_temperature`, `cache_hit_ratio`, `approx_namespace_size`, `last_included_write_at`, and `embedding_tokens`. The warehouse drops them. `last_included_write_at` is what lets you say which data version a run saw.
3. **No record of namespace state.** A run stores config and git SHA but not the namespace's row count, schema, or index status, so a rerun after a reload is not provably comparable.
4. **The corpus lives twice.** `docs` holds 482,105 products in DuckDB, and the same products sit in turbopuffer. Nothing checks that they agree.
5. **Hand-written plumbing.** Pacing for embedding tokens, 429 backoff, and the rerank-by-id-filter trick are in scripts, not a library.

## Connector design

### Constraints that shape every stage

| Fact | Source | Consequence |
| --- | --- | --- |
| No export endpoint exists. Export is a loop over `query` with `rank_by: ["id","asc"]`, a `Gt` filter on the last id, and `limit` 10,000 | export.md | A full scan is many billed queries, ordered by id, serial by construction |
| `limit.total` max 10,000, and `limit + offset` must be under 10,000 | query.md | Ranked results cannot go deeper than 10k; only id-ordered scans can |
| 16 concurrent queries per unpinned namespace; the 17th waits up to 800 ms, then gets 429. kNN queries take 2 slots, aggregations 4 | limits.md | The connector needs its own per-namespace semaphore, not DuckDB's thread count |
| Multi-query: up to 16 subqueries, one snapshot, each counts against the concurrency limit | query.md | Hybrid search costs two slots and two billed queries |
| Data queried is the namespace size or 1.28 GB, whichever is larger; returned data is $0.05/GB (Launch) | pricing.md | Returned bytes dominate large scans; the minimum dominates small namespaces |
| Batch discount is `min(0.5, max(0, 0.2 * (log10(KB) - 1)))` per request: about 22% at 128 KB, 34% at 512 KB, 50% at about 3.2 MB | pricing.md | Batch by bytes, not rows |
| Embedded attributes: 256 rows per upsert batch | limits.md | Embedding writes cap the batch size, which caps the discount |
| Unindexed data over 2 GB gives HTTP 429 on upserts; `disable_backpressure` helps backfills but not patches, filter writes, or conditional writes | write.md, ingestion.md | Backfill mode is a separate code path |

A worked example of the batch tension, using row sizes I picked for illustration: 256 rows of 3 KB is 768 KB, which earns about 38% off the $2/GB write price. Writing the same rows without embedding in 4 MB batches earns the full 50%. The repo's README prices writes with no discount and calls the result an upper bound, which is right.

### Stage A: Python package (`tpuf_duck`)

I could not find a documented way to define a table function from Python in DuckDB's core docs. The Python function page covers scalar UDFs only. A community extension, `python_udf`, offers `py_scan` (8 GitHub stars at fetch time, so I would not build on it). So stage A uses Arrow, which DuckDB queries by variable name.

```python
import duckdb, tpuf_duck as td

con = duckdb.connect("lab.duckdb")
tp = td.connect(region="aws-us-east-1")        # reads TURBOPUFFER_API_KEY

# 1. Ranked search -> Arrow table (pushdown is explicit parameters)
hits = tp.query("products", rank_by=("title", "BM25", "red shoes"),
                filters=("brand", "Eq", "Nike"), limit=50,
                include=["title", "brand"], log=con, run_id="r42")
con.sql("SELECT h.id, h.title, j.grade FROM hits h LEFT JOIN judgments j ON j.doc_id = h.id")

# 2. Full scan -> RecordBatchReader, keyset-paged on id
rows = tp.export("products", where=("brand", "Eq", "Nike"), columns=["title", "brand"])
con.sql("COPY (SELECT * FROM rows) TO 's3://bucket/products.parquet'")

# 3. Write a relation into a namespace, batched by bytes
tp.upsert(con.sql("SELECT doc_id AS id, title, brand FROM docs"),
          "products", target_batch_mb=8, embed_attr="text")
```

Notes on behavior:

- **Pushdown is the caller's job.** DuckDB pushes filters into Arrow datasets but reads RecordBatchReaders, tables, and record batches "in full and filtered afterwards." A reader also scans only once. So `export` takes `where` and `columns` arguments that become the `filters` and `include_attributes` params, and DuckDB's own `WHERE` on top is a second filter, not a pushdown.
- **A scalar UDF covers batch search.** `con.create_function("tpuf_search", ..., type="arrow", side_effects=True)` receives up to 2,048 rows per call, so `SELECT query_id, tpuf_search('products', text, 50) FROM queries` runs a whole query set. `side_effects=True` is required, since DuckDB assumes functions are pure. The UDF must size its thread pool from the 16-slot limit, not from rows per batch.
- **Pagination:** keyset on `id`, stop on a short page. Documents inserted mid-export are included, so an export is not a snapshot (Open: whether a stable export needs a pinned `last_included_write_at` check).
- **Rate limits:** per-namespace semaphore below 16, exponential backoff on 429 with the 800 ms wait in mind. The repo's token pacing for embeddings moves into the library.
- **Auth:** API key from the environment or a DuckDB secret. Do not put it in a connection string; the Postgres extension docs warn that errors can expose credentials.
- **Cost:** `log=con` writes one `query_log` row per request. `tp.estimate_export("products")` should read namespace metadata (zero-row query, p50 11 ms, billed as a query) and print pages times returned bytes.

Export cost example under an assumption I have not tested: a 10 GB, 10M-document namespace takes 1,000 pages. If each page bills the full namespace size, that is 10 TB queried, about $0.01 at $1/PB. Returning all 10 GB costs about $0.50. If filtered pages bill less than the full size, the first number shrinks (Open).

### Stage B: native extension

Aim: the same operations in plain SQL, usable from the CLI, Python, or any client.

```sql
INSTALL turbopuffer FROM community;   -- hypothetical
CREATE SECRET tpuf (TYPE turbopuffer, API_KEY '...', REGION 'aws-us-east-1');

SELECT id, "$dist" FROM tpuf_query('products', rank_by := 'title BM25 "red shoes"', top_k := 50);

ATTACH 'aws-us-east-1' AS tp (TYPE turbopuffer, READ_ONLY);
SELECT title FROM tp.products WHERE brand = 'Nike' LIMIT 100;   -- id-ordered scan
```

Mapping rules:

| SQL | turbopuffer param |
| --- | --- |
| `WHERE col = x`, `>=`, `<`, `IN`, `AND` | `filters` with `Eq`, `Gte`, `Lt`, `In`, `And` (full operator list: Open) |
| Selected columns | `include_attributes` |
| `LIMIT n` (n under 10,000) | `limit.total` |
| `ORDER BY id` | `rank_by: ["id","asc"]` plus keyset paging |
| Plain table scan, no `LIMIT` | the export loop |
| `COUNT(*)` with `GROUP BY` | `aggregate_by` / `group_by`, warning on namespaces over 1M docs |

A bare `SELECT *` from an attached namespace is an export, never a ranked query, and the extension should say so in `EXPLAIN`. Ranked search stays in `tpuf_query`, because SQL has no ANN operator.

Precedents I checked: the `elasticsearch_query` community extension pushes `WHERE` into Query DSL, `_source` filtering for projection, and `LIMIT`/`OFFSET`. The Postgres extension attaches remote tables with `ATTACH ... (TYPE postgres, READ_ONLY)` and a `pg_enable_filter_pushdown` switch. Iceberg attaches a REST catalog and exposes tables as `catalog.schema.table`. The `airport` extension attaches Arrow Flight servers (I could not read its pushdown details). A turbopuffer catalog would list namespaces with `GET /v1/namespaces` (page size up to 1,000, cursor paging) and take column types from the metadata `schema`.

**Language choice is open.** The community repository documents a C++ template. A Rust template exists (`duckdb/extension-template-rs`), CI tools support Rust (`enable_rust`), and `duckdb-rs` marks loadable extensions as experimental. The C API's table-function page documents projection pushdown but did not mention filter pushdown. I would prototype in Rust and fall back to C++ if filter pushdown is not reachable.

**No turbopuffer extension exists that I could find.** The community list (about 345 entries for DuckDB 2.0.0) has none, and a web search found none. A private or unlisted one is possible. Distribution has a cost: community extensions are built only for the latest stable DuckDB release. DuckDB 2.0 also allows custom signed repositories for private distribution.

### Stage C: eval and observability layer (`tpuf_eval`)

Ship the schema and SQL macros as a package that works with stage A or B (or neither, via a CSV of ids).

- Keep `queries`, `judgments`, `runs`, `metrics`. Split `results` into `results` (run, query, rank, doc, score) and `query_log` (one row per request: run_id, query_id, subquery index, wall ms, `server_total_ms`, `billable_logical_bytes_queried` and `_returned`, `cache_temperature`, `cache_hit_ratio`, `approx_namespace_size`, `last_included_write_at`, `embedding_tokens`).
- Add `namespace_snapshot` (namespace, captured_at, row count, schema hash, `index.status`) referenced from `runs`. The metadata page's own example uses different field names than its prose, so parse defensively.
- Ship `ndcg(run_id, k, gain)`, `recall(...)`, and `mrr(...)` as SQL macros and a bootstrap helper, with the repo's rules intact: unjudged counts as zero, a condensed variant, NULL for queries with no ideal.
- Views for cost per 1,000 queries by retriever, p50 and p99 by cache temperature, and a warning when a compared pair of runs have different `last_included_write_at` windows.

## Use cases beyond this repo

1. **Offline eval for any turbopuffer customer.** Export production queries to Parquet, replay them against a candidate config, and score with human or model judgments.
2. **Joining hits to warehouse tables.** Retrieve ids from turbopuffer, then join to orders, inventory, or permissions in DuckDB or Parquet on S3 before presenting.
3. **Reindex and backfill.** Export, transform in SQL, and upsert with `disable_backpressure`. For an untransformed copy, use `copy_from_namespace`, which the write docs price at 50% off and stack with batch discounts to up to 75%.
4. **Cost dashboards.** `query_log` plus the pricing formula gives spend by retriever, tenant, or day.
5. **Namespace inventory.** Metadata per namespace (row count, bytes, index status) into one table.

## Risks and open questions

- **Export semantics.** Not a snapshot; no billing text on the export page. Does a stable export need extra checks?
- **Parallel export.** Keyset paging is serial. Can id ranges be split safely without knowing the id distribution?
- **Filtered query billing.** Does a filtered or id-ranged page bill the full namespace size?
- **Full operator list.** I read the first 100,000 of 150,733 characters of query.md.
- **Filter pushdown in the Rust and C APIs.** Unconfirmed.
- **Field stability.** The query page says performance field names may change.
- **Multi-writer behavior of DuckDB** for a shared warehouse file.
- **Maintenance.** Community extensions track the latest stable DuckDB only, so a small team signs up for release-chasing.
- **Credentials.** Redaction of API keys in error messages needs a test.
- **Demand.** I have no evidence yet that anyone besides this repo wants it. Stage A is cheap enough to find out.

## Recommendation

Build stage A and the `query_log` half of stage C first. Both reuse what this repo already does. Start stage B only after stage A has users and a filter-pushdown prototype works.

## Sources

turbopuffer (fetched as raw markdown):

- https://turbopuffer.com/docs/query.md (first 100,000 of 150,733 characters)
- https://turbopuffer.com/docs/write.md (first 100,000 of 125,597 characters)
- https://turbopuffer.com/docs/export.md
- https://turbopuffer.com/docs/architecture.md
- https://turbopuffer.com/docs/namespaces.md
- https://turbopuffer.com/docs/metadata.md
- https://turbopuffer.com/docs/limits.md
- https://turbopuffer.com/docs/performance.md
- https://turbopuffer.com/docs/guarantees.md
- https://turbopuffer.com/docs/ingestion.md
- https://turbopuffer.com/pricing.md

DuckDB:

- https://duckdb.org/docs/2.0/clients/python/function.html
- https://duckdb.org/docs/2.0/guides/python/sql_on_arrow.html
- https://duckdb.org/docs/2.0/clients/python/data_ingestion.html
- https://duckdb.org/docs/2.0/extensions/overview.html
- https://duckdb.org/docs/2.0/core_extensions/postgres/overview.html
- https://duckdb.org/docs/2.0/core_extensions/httpfs/overview.html
- https://duckdb.org/docs/2.0/core_extensions/iceberg/overview.html
- https://duckdb.org/docs/2.0/clients/c/table_functions.html
- https://duckdb.org/community_extensions/list_of_extensions.html
- https://duckdb.org/community_extensions/development.html
- https://duckdb.org/community_extensions/extensions/airport.html
- https://duckdb.org/community_extensions/extensions/elasticsearch.html
- https://duckdb.org/community_extensions/extensions/python_udf.html
- https://duckdb.org/community_extensions/extensions/rusty_quack.html
- https://github.com/duckdb/extension-template
- https://github.com/duckdb/extension-ci-tools
- https://github.com/duckdb/duckdb-rs

Local: README.md, warehouse/schema.sql, src/relevancy/warehouse.py, tpuf.py, metrics.py, and a read-only query of `warehouse/relevancy.duckdb` for the billable-bytes figures.
