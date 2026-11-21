from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

DDL = """
CREATE TABLE IF NOT EXISTS documents (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    url         TEXT UNIQUE NOT NULL,
    date_published TEXT,
    summary     TEXT,
    category    TEXT,
    hash        TEXT NOT NULL,
    created_at  TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS files (
    id          TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(id),
    file_url    TEXT NOT NULL,
    file_path   TEXT,
    file_hash   TEXT,
    file_type   TEXT,
    pages       INTEGER,
    fetch_status TEXT,
    created_at  TEXT DEFAULT (datetime('now'))
);

-- Full extracted text of a source file.
-- Kept out of `documents` on purpose: one document can have several PDFs, a
-- PDF's text is large and re-derivable, and `documents.summary` is listing-page
-- metadata that must stay distinct from document body text.
CREATE TABLE IF NOT EXISTS doc_texts (
    id             TEXT PRIMARY KEY,
    document_id    TEXT NOT NULL REFERENCES documents(id),
    source_file_id TEXT REFERENCES files(id),
    text           TEXT NOT NULL,
    n_chars        INTEGER,
    n_pages        INTEGER,
    extractor      TEXT,
    created_at     TEXT DEFAULT (datetime('now')),
    UNIQUE(source_file_id)
);

CREATE TABLE IF NOT EXISTS tables (
    id          TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(id),
    source_file_id TEXT REFERENCES files(id),
    table_json  TEXT,
    page        INTEGER,
    n_rows      INTEGER,
    n_cols      INTEGER,
    created_at  TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS chunks (
    id          TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(id),
    chunk_index INTEGER NOT NULL,
    text        TEXT NOT NULL,
    token_count INTEGER,
    source      TEXT,
    created_at  TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_documents_date     ON documents(date_published);
CREATE INDEX IF NOT EXISTS idx_documents_category ON documents(category);
CREATE INDEX IF NOT EXISTS idx_files_document     ON files(document_id);
CREATE INDEX IF NOT EXISTS idx_doc_texts_document ON doc_texts(document_id);
CREATE INDEX IF NOT EXISTS idx_tables_document    ON tables(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_document    ON chunks(document_id);
"""

# Columns added after the first release. SQLite has no "ADD COLUMN IF NOT
# EXISTS", so existing databases are upgraded by inspecting PRAGMA output.
_MIGRATIONS: list[tuple[str, str, str]] = [
    ("files", "fetch_status", "ALTER TABLE files ADD COLUMN fetch_status TEXT"),
    ("tables", "page", "ALTER TABLE tables ADD COLUMN page INTEGER"),
    ("chunks", "source", "ALTER TABLE chunks ADD COLUMN source TEXT"),
]


def _migrate(conn: sqlite3.Connection) -> None:
    for table, column, stmt in _MIGRATIONS:
        cols = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
        if cols and column not in cols:
            conn.execute(stmt)


def init_db(db_path: str | Path) -> None:
    """Initialize database schema, upgrading an existing file in place."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(db_path)) as conn:
        conn.executescript(DDL)
        _migrate(conn)
        conn.commit()


@contextmanager
def get_conn(db_path: str | Path, read_only: bool = False) -> Generator[sqlite3.Connection, None, None]:
    """Context manager for database connections.

    `read_only=True` opens the file via a URI in `mode=ro`, so any write raises
    instead of succeeding. The SQL agent in Phase 2 uses this.
    """
    if read_only:
        conn = sqlite3.connect(f"file:{Path(db_path)}?mode=ro", uri=True)
    else:
        conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        if not read_only:
            conn.commit()
    except Exception:
        if not read_only:
            conn.rollback()
        raise
    finally:
        conn.close()
