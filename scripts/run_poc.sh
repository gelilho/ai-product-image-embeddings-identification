#!/usr/bin/env bash
# =============================================================================
# run_poc.sh — Run the POC: generate embeddings + evaluate
# =============================================================================
set -euo pipefail

echo "=== AI Product Image Embeddings Identification — POC ==="
echo ""

# Step 1: Generate embeddings
echo "→ Step 1: Generating embeddings (10 items)..."
uv run embedding-generation --limit 10 --force

echo ""
echo "→ Step 2: Running evaluation..."
uv run evaluate --limit 10

echo ""
echo "=== POC Complete ==="
