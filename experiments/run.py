"""Kept so older commands work; the CLI is `uv run relevancy` (src/relevancy/cli.py)."""

import sys

from relevancy.cli import main

if __name__ == "__main__":
    sys.exit(main())
