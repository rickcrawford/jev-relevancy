# Querying the warehouse

Every number in these docs came out of `warehouse/relevancy.duckdb`, and it answers plenty of questions the scripts don't ask. DuckDB needs nothing beyond `uv sync`: it's a Python library, so there's no server to run and no extension to install.

## Open it

From Python, with nothing else installed:

```bash
uv run python -c "import duckdb; duckdb.connect('warehouse/relevancy.duckdb', read_only=True).sql('SELECT retriever, reranker, count(*) FROM runs GROUP BY ALL').show()"
```

For poking around, the DuckDB command-line shell is nicer. Install it (`brew install duckdb` on a Mac, or a download from [duckdb.org](https://duckdb.org/docs/installation/)) and open the file read-only:

```bash
duckdb -readonly warehouse/relevancy.duckdb
```

In Python, `.show()` prints a table, `.fetchall()` returns tuples, and `.arrow()` returns a pyarrow table. `.df()` needs pandas, which this repo doesn't install (`uv add pandas` if you want it).

## Things to know before you query

- **One process writes at a time.** DuckDB locks the file while a process has it open for writing, and a second process gets an error instead of waiting, even a read-only one. The scripts work around it: they read what they need, close the connection, do the slow network work, and reopen to write. If a script is mid-write when you open the file, wait a second and retry, or copy the file and query the copy.
- **`RELEVANCY_DB` points every script at another file.** `RELEVANCY_DB=/tmp/scratch.duckdb uv run scripts/02_prepare_custom.py ...` builds a separate warehouse. That's how I tested the custom-data path without touching the real one, and it's the way to run two experiments at once.
- **The schema is applied on every connect.** `warehouse/schema.sql` uses `CREATE TABLE IF NOT EXISTS` and `ADD COLUMN IF NOT EXISTS`, so to add a column, add it there and the next script run picks it up.
- **Per-query fields repeat on every result row.** `latency_ms`, `server_ms`, and `billable_bytes` belong to the query, but they sit on each of its 50 rows. Summing them over `results` overcounts 50 times; take `DISTINCT run_id, query_id` first, as the latency query below does. Reranked runs leave `billable_bytes` empty.
- **`metrics` holds averages, not per-query scores.** Step 5 recomputes per-query nDCG with `relevancy.metrics.per_query()` and stores only the run-level numbers. For query-by-query work, call that function (the head-to-head recipe below shows how). It works on a read-only connection.
- **Runs are tied to a query sample.** Each run's `config` stores a fingerprint of the sample it searched (`config->>'sample'`). Rerunning step 2 with a new sample size changes the fingerprint, and scoring skips the old runs. The sample is drawn with DuckDB's `hash()`, which can change between major DuckDB versions, so an upgrade can change it too.

## The tables

| Table | Key | Columns worth knowing |
| --- | --- | --- |
| `datasets` | `dataset` | `rubric`, the file in `rubrics/` its judges use |
| `queries` | `dataset, query_id` | `text`, `slice` (like `short\|brand`), `sampled` |
| `docs` | `dataset, doc_id` | `title`, `body`, `brand`, `color` |
| `judgments` | `dataset, query_id, doc_id, source` | `grade` (0 to 3), `prob` and `expected` from judges that return probabilities |
| `runs` | `run_id` | `retriever`, `reranker`, `config` (JSON: namespace, region, sample, base run), `git_sha`, `started_at` |
| `results` | `run_id, query_id, rank` | `doc_id`, `score`, `latency_ms`, `server_ms`, `billable_bytes` |
| `metrics` | `run_id, judge, metric, slice` | `value`, `ci_low`, `ci_high` |
| `tpuf_requests` | none; one row per request | `kind`, `run_id`, `query_id`, `wall_ms` (which `results.latency_ms` copies), `server_ms`, `billable_bytes_queried`, `billable_bytes_written`, `embedding_tokens`, `cache_temperature`, `last_included_write_at` |

`tpuf_requests` is [tpuf-arrow](https://github.com/rickcrawford/tpuf-arrow)'s request log, saved by steps 3, 4, and 6 (for hosted rerankers). It's created the first time one of them runs, so a warehouse built before the scripts used tpuf-arrow doesn't have it, and the nine runs in [results](results.md) aren't in it. Step 4's untimed warm-up query is there too, with `run_id` `warmup`. The columns are listed in tpuf-arrow's [reference](https://github.com/rickcrawford/tpuf-arrow/blob/main/docs/reference.md#request-log).

A judge is a name, and its grades are rows. `judgments.source` is `human`, a judge name like `jev-batched:jev-1.13.0` or `openai/gpt-4o-mini`, or a bake-off grade like `jev:jev-1.13.0@bakeoff`, so the same nDCG query scores a run with any of them. After the 500-query run the file is 784 MB: 482,105 docs, 8,956 queries, about 214,000 judgments, and about 254,000 result rows.

## Recipes

Each of these runs as is against a warehouse built by the walkthrough. Swap in your own run ids from the first one.

**Runs on the current sample.** The fingerprint is the same expression `warehouse.sample_id()` uses.

```sql
SELECT run_id, retriever, reranker, config->>'namespace' AS namespace, git_sha, started_at
FROM runs
WHERE dataset = 'esci-us'
  AND config->>'sample' = (SELECT count(*) || '-' || (hash(string_agg(query_id, ',' ORDER BY query_id)) % 1000000)
                           FROM queries WHERE dataset = 'esci-us' AND sampled)
ORDER BY started_at;
```

**Every run under every judge, side by side.**

```sql
PIVOT (
  SELECT r.retriever || ' / ' || r.reranker AS run, m.judge, round(m.value, 3) AS ndcg10
  FROM metrics m JOIN runs r USING (run_id)
  WHERE m.metric = 'ndcg10' AND m.slice = 'all' AND r.config->>'namespace' = 'esci-us-full'
) ON judge USING first(ndcg10) ORDER BY run;
```

**One query's top 10, with what the humans and Jev said.** This is the query I run most. Reading ten real results next to their grades tells you more than any average.

```sql
SELECT r.rank, left(d.title, 60) AS title, h.grade AS human, j.grade AS jev, round(j.expected, 2) AS jev_expected
FROM results r
JOIN docs d ON d.dataset = 'esci-us' AND d.doc_id = r.doc_id
LEFT JOIN judgments h ON h.dataset = 'esci-us' AND h.query_id = r.query_id
                     AND h.doc_id = r.doc_id AND h.source = 'human'
LEFT JOIN judgments j ON j.dataset = 'esci-us' AND j.query_id = r.query_id
                     AND j.doc_id = r.doc_id AND j.source = 'jev-batched:jev-1.13.0'
WHERE r.run_id = 'hybrid-rrf-full-4cea30'        -- your run id
  AND r.query_id = (SELECT query_id FROM queries WHERE dataset = 'esci-us' AND sampled ORDER BY query_id LIMIT 1)
  AND r.rank <= 10
ORDER BY r.rank;
```

On my first sampled query, "t shirt dresses for women sexy", the humans had graded 2 of hybrid's top 10. Jev graded all 10 and called 6 of them Exact (§16).

**Where Jev and the humans disagree hardest.**

```sql
SELECT q.text AS query, left(d.title, 50) AS title, h.grade AS human, j.grade AS jev, round(j.prob, 2) AS jev_prob
FROM judgments h
JOIN judgments j USING (dataset, query_id, doc_id)
JOIN queries q USING (dataset, query_id)
JOIN docs d USING (dataset, doc_id)
WHERE h.dataset = 'esci-us' AND h.source = 'human' AND j.source = 'jev-batched:jev-1.13.0'
  AND abs(h.grade - j.grade) = 3
ORDER BY j.prob DESC
LIMIT 20;
```

This one found something I hadn't looked for. The top of the list was "boys swim trunks without mesh lining", "face masks without nose wire", "hand sanitizer without alcohol", and "1 12 off-road vehicle not rc": products the humans called Exact and Jev called Irrelevant with near-certainty. Splitting the human-Exact products on a negation word in the query makes it a number. The regex also catches the flower "forget me not", so the query drops it:

```sql
SELECT regexp_matches(lower(q.text), '\b(without|not|no|non)\b')
           AND NOT contains(lower(q.text), 'forget me not') AS negation,
       count(DISTINCT q.query_id) AS queries,
       count(*) AS human_exact,
       round(avg((j.grade = 0)::INT), 3) AS jev_called_irrelevant
FROM judgments h
JOIN judgments j USING (dataset, query_id, doc_id)
JOIN queries q USING (dataset, query_id)
WHERE h.dataset = 'esci-us' AND h.source = 'human' AND h.grade = 3
  AND j.source = 'jev-batched:jev-1.13.0' AND q.sampled
GROUP BY ALL;
```

On 138 human-Exact products from 27 negation queries, Jev said Irrelevant 31.2% of the time, against 6.6% of 2,121 on the other 409 queries (§11).

So who's wrong? [`experiments/negation_audit.py`](../experiments/negation_audit.py) exports every one of those disagreements with both grades hidden, for a third rater to answer one question: does the product have the thing the query excludes? I had a Claude model do the rating, given only the query and the product text. Of the 43 pairs on real negations, the product had the excluded thing in 32 (74%), so the human label was wrong and Jev was right. Jev was wrong on 4, and on 7 the product text didn't say (§11). My guess is that ESCI's graders matched on the product type and skipped the "without". That's worth knowing before you trust any human labels on negation queries, and it's the kind of slice I'd watch on my own data.

**How Jev's grades line up with the humans' (a confusion matrix).**

```sql
PIVOT (
  SELECT h.grade AS human, j.grade AS jev
  FROM judgments h JOIN judgments j USING (dataset, query_id, doc_id)
  WHERE h.dataset = 'esci-us' AND h.source = 'human' AND j.source = 'jev-batched:jev-1.13.0'
) ON jev USING count(*) ORDER BY human;
```

Rows are the human grade and columns are Jev's. This counts every pair both have graded, including the 50-query smoke run's pool. Jev's most common miss is grading a human Substitute (2) as Irrelevant (0): 216 of 1,075.

**How much of each run's top 10 has no human label.** This is the number behind the Jev and Voyage tie.

```sql
SELECT ru.retriever || ' / ' || ru.reranker AS run,
       round(avg((h.doc_id IS NULL)::INT), 3) AS unlabeled_share_top10
FROM results r
JOIN runs ru USING (run_id)
LEFT JOIN judgments h ON h.dataset = ru.dataset AND h.source = 'human'
                     AND h.query_id = r.query_id AND h.doc_id = r.doc_id
WHERE ru.config->>'namespace' = 'esci-us-full' AND r.rank <= 10
GROUP BY ALL ORDER BY 2;
```

**Latency and query cost per run.** Note the `DISTINCT`, which collapses the 50 repeated rows per query.

```sql
WITH per_query AS (
  SELECT DISTINCT run_id, query_id, latency_ms, server_ms, billable_bytes FROM results
)
SELECT ru.retriever || ' / ' || ru.reranker AS run,
       count(*) AS queries,
       round(quantile_cont(latency_ms, 0.5)) AS p50_ms,
       round(quantile_cont(latency_ms, 0.99)) AS p99_ms,
       round(sum(billable_bytes) / 1e12, 2) AS tb_queried,
       round(sum(billable_bytes) / 1e15, 4) AS usd_at_1_per_pb
FROM per_query JOIN runs ru USING (run_id)
WHERE ru.config->>'namespace' = 'esci-us-full'
GROUP BY ALL ORDER BY p50_ms;
```

BM25, title-only BM25, and vector search each queried 0.85 TB across 500 queries, and hybrid queried 1.71 TB because it runs two subqueries (§14). At list price that's under $0.002 per run.

**One run by query slice.** Intervals are stored for `all` only.

```sql
SELECT slice, round(value, 3) AS ndcg10
FROM metrics
WHERE run_id = 'jev-1.13.0-b53bce' AND judge = 'human' AND metric = 'ndcg10'
ORDER BY value;
```

**Head to head, query by query (Python).** Per-query scores come from `per_query()`, and DuckDB can join a pyarrow table by its variable name:

```python
import duckdb
import pyarrow as pa
from relevancy.metrics import per_query

con = duckdb.connect("warehouse/relevancy.duckdb", read_only=True)
a, b = "jev-1.13.0-b53bce", "rerank-3-977611"   # your run ids
pq = per_query(con, "esci-us", [a, b], judge="human")
diffs = pa.Table.from_pylist([
    {"query_id": q, "a": m["ndcg10"], "b": pq[(b, q)]["ndcg10"]}
    for (run, q), m in pq.items() if run == a and m["ndcg10"] is not None
])
con.sql("""
    SELECT q.text, round(a, 3) AS a, round(b, 3) AS b, round(a - b, 3) AS diff
    FROM diffs JOIN queries q ON q.dataset = 'esci-us' AND q.query_id = diffs.query_id
    ORDER BY abs(a - b) DESC LIMIT 10
""").show()
```

Save it as a file and run it with `uv run python yourfile.py`. For Jev against Voyage, Jev did better on 198 queries, Voyage on 202, and 100 tied. The average gap of 0.011 is a few big swings in each direction, like "rose gold series 3 apple watch", where Jev scored 0.728 and Voyage 0.

**Search from SQL.** tpuf-arrow adds `tpuf_bm25`, `tpuf_ann`, and `tpuf_search` to a connection, so you can try a retriever against the labels without writing a run. This searches titles for 20 sampled queries and shows the five whose top 10 has the lowest average human grade, counting unlabeled products as 0:

```python
import duckdb
import tpuf_arrow as ta

tp = ta.connect(env_file=".env")
con = duckdb.connect("warehouse/relevancy.duckdb", read_only=True)
ta.register(con, tp)
con.sql("""
    WITH q AS (SELECT query_id, text FROM queries WHERE dataset = 'esci-us' AND sampled ORDER BY query_id LIMIT 20),
    hits AS (SELECT query_id, text, unnest(tpuf_bm25('esci-us-full', 'title', text, 10, NULL), recursive := true) FROM q)
    SELECT text AS query, count(j.grade) AS labeled, round(avg(coalesce(j.grade, 0)), 2) AS avg_grade
    FROM hits
    LEFT JOIN judgments j ON j.dataset = 'esci-us' AND j.source = 'human'
                         AND j.query_id = hits.query_id AND j.doc_id = hits.id
    GROUP BY ALL ORDER BY avg_grade LIMIT 5
""").show()
```

Each row of `q` is one search. `tp.log` has one row per request, with its server time and billed bytes.

**Export.** One table, or the whole warehouse as Parquet:

```sql
COPY (SELECT * FROM judgments WHERE source <> 'human') TO 'judge_grades.parquet' (FORMAT parquet);
EXPORT DATABASE 'warehouse_export' (FORMAT parquet);
```

`EXPORT DATABASE` writes one Parquet file per table plus `schema.sql` and `load.sql`, and `IMPORT DATABASE 'warehouse_export'` rebuilds it. That's the easy way to share results or archive a run.

## Back up and reset

To back up, copy the file while nothing is writing to it, or use `EXPORT DATABASE`.

To start over, delete `warehouse/relevancy.duckdb` and rerun step 2. Everything else rebuilds from the scripts, except the judge grades: those cost money and time, so export `judgments` first if you might want them back.

## Why DuckDB + turbopuffer

turbopuffer answers one question well: which 50 documents match this query, with the data on object storage. Here that meant 482k products and a median of 15 to 103 ms of server time per query, depending on the retriever (§1). It doesn't do joins across tables. Everything an eval asks after retrieval is a join or an aggregate. Which of these 50 did a human grade? Which did Jev grade, and how? What did the run cost? Did branded queries get worse after the reload? DuckDB runs in your process with no server, reads Arrow and Parquet, and turns nDCG into one SQL query.

The split also keeps each side simple. turbopuffer never stores labels or run history, and DuckDB never stores vectors. `results` holds ids and ranks; titles and grades come from joins.

Logging turbopuffer's response fields paid for itself right away. On the 21k-product smoke namespace, every first-stage query billed exactly 1.28 GB (turbopuffer's per-query minimum), and hybrid-rrf billed 2.56 GB because it runs two subqueries (§14). I wouldn't have known without a `GROUP BY retriever`.

The scripts started out hand-rolling what any turbopuffer team needs for offline evals: batching embedding writes, backing off on 429s, paging through a namespace, and logging billing per request. [`duckdb-turbopuffer-connector.md`](duckdb-turbopuffer-connector.md) sketched pulling that into a package, and it became [tpuf-arrow](https://github.com/rickcrawford/tpuf-arrow), which this repo now uses for every turbopuffer request. Pacing writes under the embedding token limit is still here, in step 3.
