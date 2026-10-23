-- Search Quality Lab warehouse.
-- Every script reads from and writes to these tables, so a new dataset,
-- retriever, reranker, or judge is a new set of rows rather than new code.

-- One row per dataset: which rubric (rubrics/<name>.toml) judges grade it with.
CREATE TABLE IF NOT EXISTS datasets (
    dataset      VARCHAR PRIMARY KEY,
    rubric       VARCHAR NOT NULL DEFAULT 'generic',
    description  VARCHAR
);

CREATE TABLE IF NOT EXISTS queries (
    dataset   VARCHAR NOT NULL,   -- e.g. 'esci-us'
    query_id  VARCHAR NOT NULL,
    text      VARCHAR NOT NULL,
    split     VARCHAR,            -- train / test, as published
    slice     VARCHAR,            -- e.g. 'short|brand', used to break metrics down
    sampled   BOOLEAN DEFAULT FALSE,  -- in the evaluation sample?
    PRIMARY KEY (dataset, query_id)
);

CREATE TABLE IF NOT EXISTS docs (
    dataset     VARCHAR NOT NULL,
    doc_id      VARCHAR NOT NULL,
    title       VARCHAR,
    body        VARCHAR,          -- main text; for ESCI, description + bullet points
    brand       VARCHAR,
    color       VARCHAR,
    PRIMARY KEY (dataset, doc_id)
);

CREATE TABLE IF NOT EXISTS judgments (
    dataset   VARCHAR NOT NULL,
    query_id  VARCHAR NOT NULL,
    doc_id    VARCHAR NOT NULL,
    source    VARCHAR NOT NULL,   -- 'human', or a judge: 'jev-batched:jev-1.13.0', 'openai/gpt-4o-mini', ...
    grade     INTEGER NOT NULL,   -- 0 to 3, highest best; ESCI: 3 Exact, 2 Substitute, 1 Complement, 0 Irrelevant
    prob      DOUBLE,             -- judge confidence, when the judge gives one
    PRIMARY KEY (dataset, query_id, doc_id, source)
);
-- Expected grade (sum of grade x probability) from judges that return a
-- distribution, like Jev or an LLM read through logprobs.
ALTER TABLE judgments ADD COLUMN IF NOT EXISTS expected DOUBLE;

CREATE TABLE IF NOT EXISTS runs (
    run_id      VARCHAR PRIMARY KEY,
    dataset     VARCHAR NOT NULL,
    retriever   VARCHAR NOT NULL, -- 'bm25', 'ann', 'hybrid-rrf', ...
    reranker    VARCHAR NOT NULL, -- 'none', 'voyage/rerank-3', 'jev:jev-1.13.0', ...
    config      JSON,             -- everything needed to rerun it
    git_sha     VARCHAR,
    started_at  TIMESTAMP DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS results (
    run_id          VARCHAR NOT NULL,
    query_id        VARCHAR NOT NULL,
    doc_id          VARCHAR NOT NULL,
    rank            INTEGER NOT NULL,   -- 1-based
    score           DOUBLE,
    latency_ms      DOUBLE,             -- wall clock per query, repeated on each row
    server_ms       DOUBLE,             -- turbopuffer's server_total_ms, same
    billable_bytes  BIGINT,             -- per query, repeated on each row
    PRIMARY KEY (run_id, query_id, rank)
);

CREATE TABLE IF NOT EXISTS metrics (
    run_id   VARCHAR NOT NULL,
    judge    VARCHAR NOT NULL,
    metric   VARCHAR NOT NULL,
    slice    VARCHAR NOT NULL DEFAULT 'all',
    value    DOUBLE,
    ci_low   DOUBLE,
    ci_high  DOUBLE,
    PRIMARY KEY (run_id, judge, metric, slice)
);
