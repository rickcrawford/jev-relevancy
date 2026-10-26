#!/usr/bin/env bash
# Step 1: download the Amazon Shopping Queries dataset (ESCI).
# Source: https://github.com/amazon-science/esci-data (Apache-2.0)
# About 1.1 GB on disk. Files that are already there are skipped.
set -euo pipefail

DEST="$(cd "$(dirname "$0")/.." && pwd)/data/raw/esci"
BASE="https://github.com/amazon-science/esci-data/raw/main/shopping_queries_dataset"
mkdir -p "$DEST"

for f in shopping_queries_dataset_examples.parquet \
         shopping_queries_dataset_products.parquet \
         shopping_queries_dataset_sources.csv; do
  if [[ -s "$DEST/$f" ]]; then
    echo "have     $f"
  else
    echo "fetching $f"
    curl -fSL --retry 3 -o "$DEST/$f.part" "$BASE/$f"
    mv "$DEST/$f.part" "$DEST/$f"
  fi
done

ls -lh "$DEST"
