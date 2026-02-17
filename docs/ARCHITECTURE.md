# Architecture

## System Overview

The AI Product Image Embeddings Identification system identifies On Running products from images using embedding-based similarity search. It converts product images into high-dimensional vectors (embeddings) and finds the closest match in a pre-built catalog.

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          OFFLINE (Build Phase)                          │
│                                                                         │
│   Golden Dataset    ┌─────────────┐   ┌──────────────┐   ┌──────────┐  │
│   (XLSX/CSV)   ────▶│ Catalog     │──▶│ Image        │──▶│ Embedding│  │
│   275 products      │ Loader      │   │ Fetcher      │   │ Backend  │  │
│                     └─────────────┘   │ (batched,    │   │ (Marqo)  │  │
│                                       │  rate-limited)│   └────┬─────┘  │
│                                       └──────────────┘        │        │
│                                                               ▼        │
│   ┌──────────────┐   ┌──────────────┐   ┌────────────────────────────┐ │
│   │ Color        │   │ Category     │   │ Storage                    │ │
│   │ Extractor    │   │ Classifier   │   │  embeddings.npy            │ │
│   │ (regex)      │   │ (rules)      │   │  item_codes.json           │ │
│   └──────┬───────┘   └──────┬───────┘   │  metadata.json             │ │
│          └──────┬────────────┘           │  enriched_catalog.csv      │ │
│                 ▼                        └────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│                          ONLINE (Query Phase)                           │
│                                                                         │
│   Query Image   ┌──────────────┐   ┌──────────────┐   ┌─────────────┐  │
│   (any source)──▶│ Embedding    │──▶│ In-Memory    │──▶│ Top-K       │  │
│                  │ Backend      │   │ Index        │   │ Results     │  │
│                  │ (same model) │   │ (cosine sim) │   │ + metadata  │  │
│                  └──────────────┘   └──────────────┘   └─────────────┘  │
└──────────────────────────────────────────────────────────────────────────┘
```

## Components

### CLI Layer (`cli/main.py`)

Three commands exposed as standalone entrypoints via `pyproject.toml`:

| Command | Entrypoint | Purpose |
|---|---|---|
| `uv run embedding-generation` | `run_embedding_generation()` | Build embedding catalog from golden dataset |
| `uv run identify` | `run_identify()` | Query a single image against the catalog |
| `uv run evaluate` | `run_evaluate()` | Self-identification accuracy benchmark |

### Catalog (`catalog/`)

- **`loader.py`** - Loads XLSX/CSV golden datasets, maps D365 column names to internal names, parses gender and dates. Supports `limit` parameter for POC usage.
- **`schema.py`** - Column name mapping from raw D365 names (e.g., `dim_d365_item_vertical_name` to `vertical`). Validates required columns.

### Domain (`domain/`)

- **`models.py`** - Core data objects: `CatalogItem`, `EnrichedItem`, `EmbeddingResult`, `IdentificationMatch`, `FetchResult`, `PipelineReport`. Enums: `Category` (shoe/apparel/accessory), `Gender`.
- **`errors.py`** - Error hierarchy rooted at `PIIError` with specialized exceptions for each subsystem.

### Images (`images/`)

- **`url_builder.py`** - Deterministic URL construction: `{base_url}/{item_code}_000_001.png`. Handles both legacy (`59.98842`) and new (`1MD10060553`) item code formats.
- **`fetcher.py`** - Async HTTP fetcher with safety controls:
  - **Token bucket rate limiter** (2 req/s default)
  - **Semaphore concurrency cap** (3 concurrent default)
  - **Batched pagination** for large datasets (20 items/batch, 5s cooldown)
  - **Retry with exponential backoff + jitter** (3 retries)
  - **Content-type validation** and **max size enforcement**
  - **Local file cache** (SHA256-hashed filenames)
- **`rate_limiter.py`** - `TokenBucketRateLimiter` with async `acquire()`.
- **`validators.py`** - MIME type and PIL image validation.

### Embeddings (`embeddings/`)

**Strategy pattern** for pluggable backends:

- **`interface.py`** - `EmbeddingBackend` ABC defining `load()`, `embed_image()`, `embed_images()`. Global registry with `register_backend()` / `get_backend()`.
- **`marqo_backend.py`** - Default implementation using Marqo-Ecommerce-L (1024-d). Loads via `open_clip.create_model_and_transforms("hf-hub:Marqo/marqo-ecommerce-embeddings-L")`. Auto-registers at import time.
- **`metadata.py`** - `EmbeddingMetadata` dataclass with JSON serialization for model provenance tracking.

Adding a new backend requires:
1. Implement `EmbeddingBackend` ABC
2. Call `register_backend("name", YourBackend)` at module level
3. Set `PII_EMBEDDING_BACKEND=name` in env

### Features (`features/`)

- **`color_extractor.py`** - Regex-based color extraction from product names. Pattern: extract text after gender indicator (M/W/U/K/Jr). Example: `"Cloud 5 M Black"` yields `"Black"`.
- **`category_classifier.py`** - Rule-based classification using shoe family names (`Cloud`, `The Roger`, etc.) and keyword matching for apparel/accessories.

### Index (`index/`)

- **`similarity.py`** - `cosine_similarity()` function for vector comparison.
- **`in_memory.py`** - `InMemoryIndex` with pre-normalized embeddings for fast Top-K search. Returns `SimilarityMatch` objects sorted by score.

### Evaluation (`eval/`)

- **`metrics.py`** - `EvalMetrics` tracking Top-1/3/5 accuracy and MRR (Mean Reciprocal Rank).
- **`evaluator.py`** - Self-identification evaluation: each catalog item's embedding is queried against the full index; the item should match itself at rank 1.
- **`golden_loader.py`** - Loads golden dataset entries for evaluation.

### Storage (`storage/`)

- **`csv_store.py`** - Saves/loads enriched catalog CSV with all derived attributes.
- **`embedding_store.py`** - Saves/loads `embeddings.npy` (N x dim float32), `item_codes.json`, and `metadata.json`. Supports `exists()` check for skip logic.

### Config (`config.py`)

All configuration from environment variables via `_env()` / `_env_int()` / `_env_float()` helpers. Frozen dataclasses: `FetcherConfig`, `EmbeddingConfig`, `StorageConfig`, `LoggingConfig`, `AppConfig`.

### Logging (`logging.py`)

Loguru-based with `setup_logging()` supporting text and JSON output formats.

## Anti-DoS Strategy

Fetching 275 product images from Azure Blob Storage requires multiple safety layers:

```
Layer 1: Rate Limiter (Token Bucket)
  - 2 requests/second max throughput
  - Burst capacity matches rate

Layer 2: Concurrency Cap (Semaphore)
  - Max 3 concurrent HTTP connections

Layer 3: Batch Pagination (NEW)
  - Items split into batches of 20
  - 5-second cooldown between batches
  - Small batches (<=5) skip pagination

Layer 4: Retry with Backoff
  - 3 retries per image
  - Exponential backoff (1s, 2s, 4s) + random jitter
  - Respects HTTP 429 (rate limit) responses

Layer 5: Local Cache
  - Downloaded images cached to disk
  - Subsequent runs skip already-fetched images
  - Cache key: SHA256(item_code)[:16]
```

For the full 275-item dataset, this means:
- ~14 batches of 20 items each
- ~10s per batch (20 items at 2/s)
- ~65s cooldown (13 pauses x 5s)
- **Total fetch time: ~200s** (vs ~140s without batching)
- **Zero risk of overwhelming the server**

## Data Flow

```
product_items_golden.xlsx
    │
    ▼
CatalogItem[] (275 items)
    │
    ├──▶ ImageFetcher ──▶ FetchResult[] (bytes + metadata)
    │
    ├──▶ EmbeddingBackend ──▶ float32[N, 1024] vectors
    │
    ├──▶ ColorExtractor ──▶ "Black", "Undyed-White | Glacier", etc.
    │
    ├──▶ CategoryClassifier ──▶ shoe | apparel | accessory
    │
    ▼
EnrichedItem[]
    │
    ├──▶ enriched_catalog.csv
    ├──▶ embeddings.npy + item_codes.json + metadata.json
    └──▶ data/cache/*.png (image cache)
```

## Directory Structure

```
ai-product-image-embeddings-identification/
├── src/product_image_id/
│   ├── cli/main.py              # 3 CLI commands
│   ├── catalog/                 # XLSX/CSV loading + schema
│   ├── domain/                  # Models + errors
│   ├── embeddings/              # Pluggable backends (Strategy)
│   ├── eval/                    # Metrics + golden evaluation
│   ├── features/                # Color + category extraction
│   ├── images/                  # URL builder + safe fetcher
│   ├── index/                   # Cosine similarity search
│   ├── storage/                 # CSV + embedding persistence
│   ├── config.py                # Env-based configuration
│   └── logging.py               # Loguru setup
├── tests/
│   ├── unit/                    # 144 unit tests
│   └── integration/             # 6 integration tests
├── data/
│   ├── golden/                  # product_items_golden.xlsx
│   ├── cache/                   # Downloaded images (gitignored)
│   ├── embeddings/              # Generated embeddings (gitignored)
│   ├── output/                  # enriched_catalog.csv (gitignored)
│   └── queries/                 # Query images for identification
├── docs/                        # Architecture documentation
├── scripts/run_poc.sh           # POC runner
├── pyproject.toml               # Single source of truth
└── .env.example                 # Environment variable template
```

## Technology Choices

| Choice | Rationale |
|---|---|
| **Marqo-Ecommerce-L** | SOTA for e-commerce image search, Apache 2.0, 1024-d, runs locally |
| **open_clip** | Industry-standard CLIP implementation, supports HuggingFace Hub models |
| **Cosine similarity** | Standard metric for normalized embeddings, fast with numpy |
| **In-memory index** | Sufficient for 275 items; upgrade to FAISS/Milvus for 10K+ |
| **Token bucket rate limiter** | Smooth rate limiting vs fixed window, handles bursts |
| **Batched pagination** | Prevents sustained load on image server for large datasets |
| **uv** | Fast Python package manager, replaces pip/poetry |
| **loguru** | Structured logging with zero config, better than stdlib logging |
| **Typer** | Type-safe CLI with auto-generated help |
