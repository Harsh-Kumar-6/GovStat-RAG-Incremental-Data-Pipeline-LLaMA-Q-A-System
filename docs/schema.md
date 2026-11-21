# Database schema

SQLite, at `data/mospi.db`. Created and upgraded by `scraper/db.py::init_db`,
which is safe to re-run: the DDL is `CREATE TABLE IF NOT EXISTS`, and columns
added after the first release are applied by inspecting `PRAGMA table_info`
(SQLite has no `ADD COLUMN IF NOT EXISTS`).

## Design principles

1. **One row per real-world thing.** A document is the published release; a file
   is a PDF attached to it; text and tables are what was extracted from a file.
   Mixing these is what caused the original chunking bug.
2. **Raw and derived data are separated.** `documents.summary` is listing-page
   metadata. `doc_texts.text` is extracted document body. They are different
   kinds of data and must never be written to the same column.
3. **Everything is traceable to a source.** Every derived row carries a
   `document_id`, so any answer the RAG or SQL agent produces can be cited.
4. **Derived data is rebuildable.** `chunks` is deleted and rebuilt on every ETL
   run, so the pipeline is idempotent and re-running it cannot duplicate rows.

## Tables

### `documents` — one published MoSPI release
| Column | Type | Notes |
|---|---|---|
| `id` | TEXT PK | UUID4 |
| `title` | TEXT NOT NULL | |
| `url` | TEXT UNIQUE NOT NULL | Natural key. `UNIQUE` makes the crawl idempotent at the DB level, not just in application logic. |
| `date_published` | TEXT | ISO `YYYY-MM-DD`. Stored as TEXT because SQLite has no date type; ISO-8601 strings sort and compare correctly as text. |
| `summary` | TEXT | Listing-page blurb **only**. Never document body text. |
| `category` | TEXT | e.g. `Price Statistics > CPI` |
| `hash` | TEXT NOT NULL | SHA-256 content fingerprint for incremental crawls: unchanged hash means skip. |
| `created_at` | TEXT | |

### `files` — a downloadable artifact of a document
| Column | Type | Notes |
|---|---|---|
| `id` | TEXT PK | |
| `document_id` | TEXT FK → documents | |
| `file_url` | TEXT NOT NULL | |
| `file_path` | TEXT | NULL until downloaded. Doubles as the work queue: `WHERE file_path IS NULL`. |
| `file_hash` | TEXT | SHA-256 of the bytes on disk, to detect a re-published PDF. |
| `file_type`, `pages` | TEXT, INTEGER | |
| `fetch_status` | TEXT | `ok` / `download_failed`. **Added in Phase 1:** without it a failed download was indistinguishable from one not yet attempted, so failures were invisible. |

### `doc_texts` — full extracted text of a file
| Column | Type | Notes |
|---|---|---|
| `id` | TEXT PK | |
| `document_id` | TEXT FK → documents | |
| `source_file_id` | TEXT FK → files, **UNIQUE** | One extraction per file; re-parsing replaces rather than appends. |
| `text` | TEXT NOT NULL | The complete text. **Not truncated.** |
| `n_chars`, `n_pages` | INTEGER | Cheap sanity checks without reading the blob. |
| `extractor` | TEXT | e.g. `pdfplumber`. Recorded so a future switch to OCR is comparable. |

**Why a separate table and not `documents.full_text`:** one document can have
several PDFs; the text is large and would be loaded by every `SELECT * FROM
documents`; and it is re-derivable, so it belongs with the file it came from.

### `tables` — tables extracted from a file
| Column | Type | Notes |
|---|---|---|
| `id` | TEXT PK | |
| `document_id`, `source_file_id` | TEXT FK | |
| `table_json` | TEXT | Raw `{page, header, rows}` JSON. **Staging only** — Phase 1's typed series tables are built from this, and SQL agents query those, not this column. |
| `page` | INTEGER | **Added in Phase 1**, so an extracted table can be cited to a page number. |
| `n_rows`, `n_cols` | INTEGER | Used to discard degenerate 1x1 extractions. |

### `chunks` — embedding units for the vector index
| Column | Type | Notes |
|---|---|---|
| `id` | TEXT PK | Matches the FAISS metadata row. |
| `document_id` | TEXT FK → documents | Retrieval evals compare these IDs against `relevant_doc_ids`. |
| `chunk_index` | INTEGER NOT NULL | Order within the document. |
| `text` | TEXT NOT NULL | ~180 words, 40-word overlap. |
| `token_count` | INTEGER | |
| `source` | TEXT | `pdf_text` or `summary`. **Added in Phase 1** so the original bug is detectable by query: `SELECT source, COUNT(*) FROM chunks GROUP BY source;` — an all-`summary` result means no PDF text was indexed. |

## Indexes

```sql
idx_documents_date     ON documents(date_published)  -- date-filtered questions
idx_documents_category ON documents(category)        -- category facets
idx_files_document     ON files(document_id)
idx_doc_texts_document ON doc_texts(document_id)     -- ETL text lookup per doc
idx_tables_document    ON tables(document_id)
idx_chunks_document    ON chunks(document_id)        -- citation lookups
```

Every index backs a query the application actually makes. At the current data
volume SQLite would scan fine; they are declared because the SQL agent writes
its own queries and should not be able to produce an accidental full scan on a
join column.

## Read-only access

`get_conn(db_path, read_only=True)` opens the file as `file:...?mode=ro` via
URI. Writes then fail inside SQLite rather than relying on the agent's prompt
to not attempt them. This is the first of the SQL agent's defense layers; the
SELECT-only parser, row limit, and timeout are the others.

## Typed series tables

Built by `pipeline/tables_to_sql.py` from the staged `tables.table_json`
blobs. This is the layer the SQL agent queries.

### `cpi_state` — state-wise CPI index and inflation

| Column | Type | Notes |
|---|---|---|
| `id` | INTEGER PK | |
| `source_document_id` | TEXT FK → documents | Every value is citable. |
| `source_table_id` | TEXT FK → tables | Traceable to the exact extracted table and page. |
| `release_date` | TEXT | When the note was published. |
| `period` | TEXT | `YYYY-MM` **reference month**, parsed from the title — *not* the release month. The September index is published in October, so using `date_published` would mislabel every row by one month. |
| `state` | TEXT NOT NULL | Newlines collapsed. |
| `sector` | TEXT NOT NULL | `Rural` / `Urban` / `Combined`. |
| `index_value` | REAL | Index, base 2012=100. |
| `inflation_yoy` | REAL | Year-on-year %. |
| | UNIQUE | `(source_document_id, state, sector)` — makes rebuilds idempotent. |

Indexes on `period`, `state`, `sector`, and `source_document_id`.

**Why long (tidy) format, one row per (release, state, sector)** — rather than
one row per state with six value columns:

- adding a sector or metric needs no schema migration;
- `WHERE sector = 'Rural'` replaces choosing between columns, which is far
  easier for an LLM to get right than remembering which of six columns holds
  rural inflation;
- `AVG`, `MAX` and `GROUP BY` work without naming every column.

The cost is more rows and a mandatory `sector` filter — worth it here, because
the query writer is a language model.

**Real queries this supports** (all verified against live data):

```sql
-- Highest rural inflation in a month
SELECT state, inflation_yoy FROM cpi_state
WHERE sector='Rural' AND period='2026-09'
ORDER BY inflation_yoy DESC LIMIT 5;

-- Month-on-month change, with the source document for the citation
SELECT a.state, a.inflation_yoy AS sep, b.inflation_yoy AS oct, d.url
FROM cpi_state a
JOIN cpi_state b ON a.state=b.state AND a.sector=b.sector
JOIN documents d ON d.id=b.source_document_id
WHERE a.period='2026-09' AND b.period='2026-10' AND a.sector='Combined'
ORDER BY (b.inflation_yoy - a.inflation_yoy) DESC;
```

### Not yet built: `iip`

The IIP press notes extract far less cleanly than CPI: the statement title
occupies the header row, real column labels are spread across three merged
rows, and there are empty spacer columns (`col_2`, `col_5`). Each IIP
statement needs its own normalizer. CPI was done first because it is clean and
tells the same design story.
