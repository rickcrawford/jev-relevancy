# Sample data

A small slice of the Shopping Queries dataset for trying the pipeline without downloading all of ESCI. It goes through `02_prepare_custom.py` like your own data would.

| File | Rows | |
| --- | --- | --- |
| `queries.csv` | 8 | `query_id`, `text`: the 8 lowest query ids in the 500-query ESCI sample |
| `docs.parquet` | 176 | `id`, `title`, `body`, `brand`, `color`: every product a person graded for those queries, with `body` cut to 300 characters |
| `labels.csv` | 176 | `query_id`, `doc_id`, `grade`: the human grades, Exact 3, Substitute 2, Complement 1, Irrelevant 0 |

The corpus holds only graded products, so every retriever scores higher here than it would on a real catalog. It's for checking that the pipeline works, not for comparing systems.

From Amazon's [Shopping Queries dataset](https://github.com/amazon-science/esci-data), Apache-2.0.
