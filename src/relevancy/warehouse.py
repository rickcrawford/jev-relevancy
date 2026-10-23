"""Open the DuckDB warehouse and make sure the schema exists."""

import os
import subprocess
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = ROOT / "warehouse" / "schema.sql"
DEFAULT_DB = ROOT / "warehouse" / "relevancy.duckdb"


def connect(path: Path | str | None = None) -> duckdb.DuckDBPyConnection:
    db = Path(path or os.environ.get("RELEVANCY_DB", DEFAULT_DB))
    db.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db))
    con.execute(SCHEMA.read_text())
    return con


def sample_id(con: duckdb.DuckDBPyConnection, dataset: str) -> str:
    """Short fingerprint of the current query sample.

    Runs record it so scoring can skip runs made on a different sample
    (re-running step 2 with a new --sample size changes it).
    """
    (fp,) = con.execute(
        "SELECT count(*) || '-' || (hash(string_agg(query_id, ',' ORDER BY query_id)) % 1000000) "
        "FROM queries WHERE dataset = ? AND sampled", [dataset]).fetchone()
    return fp


def git_sha() -> str | None:
    """Current commit, with "-dirty" if there are uncommitted changes to tracked files."""
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                               cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        return f"{sha}-dirty" if dirty else sha
    except (OSError, subprocess.CalledProcessError):
        return None
