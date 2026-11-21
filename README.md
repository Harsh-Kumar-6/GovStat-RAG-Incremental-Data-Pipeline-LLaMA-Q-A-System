# MoSPI Scraper + LLaMA RAG Chatbot

A production-grade pipeline that:
1. **Scrapes** public statistical publications from [MoSPI](https://mospi.gov.in)
2. **Transforms** raw content into validated, chunked text
3. **Answers questions** via a LLaMA-3-powered RAG chatbot with source citations

---

## Quick Start

```bash
# 1. Copy and configure environment
cp .env.example .env

# 2. Start all services (API + UI + Ollama)
make up

# 3. Pull the LLaMA3 model (first-time only, ~5 GB)
make pull-model

# 4. Run scraper on-demand
make crawl

# 5. Run ETL
make etl

# 6. Build vector index
make index

# 7. Open the UI
open http://localhost:8501
```

> **API docs:** http://localhost:8000/docs

---

## Project Structure

```
mospi-rag/
├── scraper/           # Web scraping & PDF parsing
│   ├── config.py      # Pydantic settings (reads .env)
│   ├── db.py          # SQLite schema + context manager
│   ├── crawl.py       # MoSPI JSON listing API client (the site is a SPA)
│   ├── parse.py       # PDF downloader + pdfplumber extractor
│   ├── logger.py      # structlog JSON logger
│   └── __main__.py    # CLI entry points
├── pipeline/
│   ├── etl.py         # Transform → chunk → validate → Parquet
│   └── run.py         # python -m pipeline.run
├── rag/
│   ├── indexer.py     # FAISS vector store + SentenceTransformer
│   ├── llm.py         # Ollama / LLaMA3 client (streaming + sync)
│   ├── api.py         # FastAPI: POST /ask, POST /ingest, GET /health
│   └── ui.py          # Streamlit web UI
├── infra/
│   ├── Dockerfile.scraper
│   ├── Dockerfile.api
│   └── Dockerfile.ui
├── tests/              # 91 tests — pytest
│   ├── conftest.py            # DB + PDF fixtures
│   ├── test_crawl.py          # API client, categories, idempotency
│   ├── test_parse.py          # PDF guard (HTML-as-200), extraction
│   ├── test_chunker.py        # Chunk windows, overlap, termination
│   ├── test_validator.py      # Validation + date normalization
│   ├── test_db.py             # Schema, migration, read-only connection
│   └── test_etl_integration.py # DB → chunks, end to end
├── docs/
│   └── schema.md       # SQL schema and the reason for each choice
├── data/
│   ├── raw/pdf/       # Downloaded PDFs
│   └── processed/     # Parquet files + FAISS index
├── datasets/
│   └── catalog.json   # ETL run manifest
├── docker-compose.yml
├── Makefile
├── requirements.txt
├── pyproject.toml
└── .env.example
```

---

## CLI Reference

```bash
# Pull MoSPI document listings (idempotent — skips unchanged content).
# --max-pages caps API listing pages PER SOURCE.
python -m scraper crawl --max-pages 5

# Note: mospi.gov.in is a client-rendered SPA — every path returns the same
# HTML shell. The scraper calls the site's own public JSON API instead, so
# --seed-url is accepted but ignored. Configure SOURCES in .env.

# Download PDFs and extract text + tables
python -m scraper parse

# Show run summary
python -m scraper report

# Run ETL (chunk, validate, write Parquet)
python -m pipeline.run
```

---

## Configuration

All settings via `.env` (see `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `SEED_URLS` | MoSPI press + reports | Comma-separated start URLs |
| `MAX_PAGES_PER_RUN` | `5` | Max listing pages per seed per run |
| `RATE_LIMIT_DELAY` | `1.5` | Seconds between requests |
| `CONCURRENCY` | `2` | Max parallel HTTP workers |
| `CHUNK_SIZE` | `1000` | Words per text chunk |
| `CHUNK_OVERLAP` | `150` | Overlap words between chunks |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | SentenceTransformer model name |
| `TOP_K` | `5` | Retrieved chunks per query |
| `OLLAMA_MODEL` | `llama3` | Ollama model tag |
| `OLLAMA_BASE_URL` | `http://ollama:11434` | Ollama server URL |

---

## Data Model

### SQLite Tables

**`documents`** — one row per scraped page
```
id, title, url, date_published, summary, category, hash, created_at
```

**`files`** — PDF/Excel/Word attachments
```
id, document_id, file_url, file_path, file_hash, file_type, pages, created_at
```

**`tables`** — extracted tabular structures from PDFs
```
id, document_id, source_file_id, table_json, n_rows, n_cols, created_at
```

**`chunks`** — text segments for RAG indexing (doc → chunk lineage preserved)
```
id, document_id, chunk_index, text, token_count, created_at
```

### Processed Artifacts

| File | Contents |
|---|---|
| `data/processed/documents.parquet` | Validated, normalized documents |
| `data/processed/chunks.parquet` | Text chunks with lineage to source docs |
| `data/processed/vector_store/` | FAISS index + metadata pickle |
| `datasets/catalog.json` | Counts by category/year, file manifest |

---

## RAG API

### `POST /ask`
```json
{
  "question": "What was India's GDP growth rate in Q2 2024-25?",
  "k": 5,
  "temperature": 0.1
}
```
Response:
```json
{
  "answer": "According to the MoSPI press release...\n\nSources:\n- [GDP Q2 2024](https://mospi.gov.in/...)",
  "citations": [
    {
      "title": "GDP Estimates Q2 2024-25",
      "url": "https://mospi.gov.in/...",
      "date_published": "2024-11-29",
      "snippet": "India's GDP grew at 7.6 percent..."
    }
  ]
}
```

### `POST /ingest`
Rebuilds the FAISS vector index from `data/processed/chunks.parquet`.

### `GET /health`
Returns `{"status": "ok", "model": "llama3"}`.

---

## Architecture

```
                   ┌─────────────────────────────────────────────┐
                   │              Docker Compose                  │
                   │                                              │
  User/Cron ──────►│  scraper (on-demand)                        │
                   │    crawl.py → parse.py → SQLite              │
                   │                  │                           │
                   │  pipeline (on-demand)                        │
                   │    etl.py → chunks.parquet                   │
                   │                  │                           │
                   │             indexer.py                       │
                   │          FAISS vector store                  │
                   │                  │                           │
  Browser ────────►│  Streamlit UI (8501)                         │
                   │        │                                     │
                   │  FastAPI (8000) ──► Ollama LLaMA3 (11434)   │
                   └─────────────────────────────────────────────┘
```

**Incremental scraping** uses SHA-256 fingerprinting of page content — unchanged pages are skipped, so re-runs are cheap and idempotent.

**Chunking** splits document text into ~1000-word segments with 150-word overlap, preserving source lineage via `document_id`. This ensures retrieved context stays within LLaMA's window while maintaining traceability.

**RAG flow:**
1. User query → embed with `all-MiniLM-L6-v2` → FAISS top-k search
2. Retrieved chunks assembled into context window
3. LLaMA-3 prompted: *"Answer strictly from provided context. Cite sources."*
4. Response streamed back with clickable source URLs

---

## Trade-offs & Assumptions

### MoSPI site structure
The MoSPI website uses a Java Portlet CMS (Liferay). The scraper uses heuristic link filtering (URL path patterns like `/press`, `/report`, `/survey`) rather than hardcoded CSS selectors. This makes it **resilient to minor HTML changes** but may miss some pages if MoSPI restructures its URL scheme. If that happens, update `SEED_URLS` and the regex in `parse_listing_page()`.

### PDF extraction quality
pdfplumber works well for text-based PDFs but struggles with:
- Scanned/image PDFs (would need OCR via `pytesseract` + `pdf2image`)
- Complex multi-column layouts
- Tables spanning multiple pages

Workaround documented in code; OCR integration is marked as a future improvement.

### LLM hallucination guard
The system prompt instructs LLaMA to answer **only from context** and say "I don't have that in my data" otherwise. This is a soft constraint — LLaMA-3 may occasionally blend training knowledge. For production, add a retrieval confidence threshold and filter out low-scoring chunks.

### Scaling
Current FAISS index is flat (exact search). For >100k chunks, switch to `IndexIVFFlat` with `nlist=100` for approximate search. For multi-user concurrency, the SQLite DB should be replaced with Postgres (Dockerfile.postgres included as a placeholder).

---

## Future Improvements

1. **OCR pipeline** — integrate `pytesseract` + `pdf2image` for scanned PDFs
2. **Scheduler** — replace manual `make crawl` with Prefect/Dagster flows with retry policies
3. **Reranking** — add a cross-encoder reranker (e.g., `ms-marco-MiniLM`) between FAISS retrieval and LLaMA generation
4. **Streaming UI** — wire SSE streaming from FastAPI to Streamlit for token-by-token display
5. **Postgres backend** — swap SQLite for Postgres for concurrent write safety
6. **Metadata filtering** — allow UI to filter by date range or category before vector search
7. **Evaluation harness** — automated RAG eval using RAGAS or custom golden-QA pairs from MoSPI FAQs
8. **CI/CD** — GitHub Actions workflow: lint → test → build Docker → push to registry