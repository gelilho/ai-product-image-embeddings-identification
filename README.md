# AI Product Image Embeddings Identification

Embedding-based product image matching system for internal catalog identification.

Given any product image, identifies the model/item and its characteristics (shoe vs apparel, color) by comparing against an internal embedding catalog using cosine similarity.

## Architecture

```
                    ┌─────────────────────────────────────────────┐
                    │         Embedding Generation Pipeline        │
                    │                                             │
 Golden Dataset     │  Load Catalog → Fetch Images (batched)      │
 (XLSX/CSV)    ────▶│  → Embed (Marqo-Ecommerce-L)               │───▶ embeddings.npy
                    │  → Extract Color + Category                 │     enriched_catalog.csv
                    │  → Save                                     │     metadata.json
                    └─────────────────────────────────────────────┘

                    ┌─────────────────────────────────────────────┐
                    │              Identify (Query)                │
                    │                                             │
 Query Image   ────▶│  Embed → Cosine Similarity → Top-K Matches  │───▶ Ranked results
                    │         against stored embeddings           │     (code, name, URL)
                    └─────────────────────────────────────────────┘
```

**Default backend:** [Marqo-Ecommerce-L](https://huggingface.co/Marqo/marqo-ecommerce-embeddings-L) (1024-d, Apache 2.0, runs locally)

Pluggable architecture supports swapping backends (SigLIP 2, DINOv2, Qwen3-VL) without code changes.

## Commands

The system exposes **3 CLI commands**:

| Command | Purpose |
|---|---|
| `uv run embedding-generation` | Download images, generate embeddings, save catalog |
| `uv run identify <image>` | Identify a query image against the embedding index |
| `uv run evaluate` | Self-identification accuracy test on the golden dataset |

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (package manager)

## Setup

```bash
# Clone and enter the project
cd ai-product-image-embeddings-identification

# Install all dependencies (including dev)
uv sync --all-extras

# Copy env config
cp .env.example .env

# Place golden dataset
cp /path/to/product_items_golden.xlsx data/golden/
```

## Usage

### Generate Embeddings

```bash
# POC - 5 items
uv run embedding-generation --limit 5

# 30 items (good mix of shoes + apparel + accessories)
uv run embedding-generation --limit 30 --force

# Full dataset - all 275 items (batched, anti-DoS)
uv run embedding-generation --force

# Custom batch size (default: 20 items per batch)
uv run embedding-generation --batch-size 10 --force
```

**Anti-DoS strategy:** For datasets > 5 items, images are fetched in batches (default 20) with a 5-second cooldown between batches. Combined with the rate limiter (2 req/s) and concurrency cap (3), this prevents overwhelming the image server.

**Skip logic:** If embeddings already exist, the command skips re-generation. Use `--force` to regenerate.

### Identify a Product Image

```bash
uv run identify path/to/query_image.png
uv run identify path/to/query_image.png --top-k 10
```

Output includes Rank, Score, Item Code, Item Name, and Image URL.

### Evaluate Against Golden Dataset

```bash
uv run evaluate --limit 30
uv run evaluate --top-k 5
```

### Run the Full POC

```bash
bash scripts/run_poc.sh
```

## Development

### Run Tests

```bash
uv run pytest
uv run pytest tests/unit/
uv run pytest -m integration
uv run pytest -m "not integration"
```

### Lint & Format

```bash
uv run ruff check src/ tests/
uv run ruff format src/ tests/
```

### Type Check

```bash
uv run mypy src/
```

## Configuration

All configuration via environment variables. See `.env.example` for the full list.

| Variable | Default | Description |
|---|---|---|
| `PII_EMBEDDING_BACKEND` | `marqo` | Embedding backend |
| `PII_EMBEDDING_MODEL` | `Marqo/marqo-ecommerce-embeddings-L` | Model name |
| `PII_FETCH_RATE_LIMIT` | `2.0` | Max requests/sec to image host |
| `PII_FETCH_CONCURRENCY` | `3` | Max concurrent downloads |
| `PII_LOG_LEVEL` | `INFO` | Logging level |

## Project Structure

```
src/image_identification/
├── cli/            # CLI commands (embedding-generation, identify, evaluate)
├── catalog/        # Catalog loading and schema validation
├── domain/         # Domain models and errors
├── embeddings/     # Pluggable embedding backends (Strategy pattern)
├── eval/           # Evaluation harness and metrics
├── features/       # Color extraction, category classification
├── images/         # URL builder, rate-limited fetcher, batch pagination
├── index/          # Cosine similarity search
└── storage/        # CSV and embedding persistence
```

## Troubleshooting

**Embeddings already exist:** Use `--force` to regenerate: `uv run embedding-generation --force`

**Rate limiting / slow fetches:** Adjust `PII_FETCH_RATE_LIMIT` and `PII_FETCH_CONCURRENCY` in `.env`. For large datasets, use `--batch-size`.

**GPU vs CPU:** Set `PII_EMBEDDING_DEVICE=cpu` to force CPU. GPU (cuda/mps) is auto-detected.

**First run slow:** The embedding model (~2.5GB) downloads on first use. Subsequent runs use cache.
