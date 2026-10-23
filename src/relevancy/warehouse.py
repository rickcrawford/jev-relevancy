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


def git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT, capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
