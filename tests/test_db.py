"""Tests for schema creation, migration, and the read-only connection."""
from __future__ import annotations

import sqlite3

import pytest

from scraper.db import get_conn, init_db

EXPECTED_TABLES = {"documents", "files", "doc_texts", "tables", "chunks"}


def _names(conn, kind: str) -> set[str]:
    return {
        r[0] for r in conn.execute(f"SELECT name FROM sqlite_master WHERE type='{kind}'")
    }


def test_init_db_creates_all_tables(db_path):
    with get_conn(db_path) as conn:
        assert EXPECTED_TABLES <= _names(conn, "table")


def test_init_db_creates_indexes(db_path):
    with get_conn(db_path) as conn:
        idx = _names(conn, "index")
    assert "idx_documents_date" in idx
    assert "idx_chunks_document" in idx


def test_init_db_is_idempotent(db_path):
    init_db(db_path)
    init_db(db_path)
    with get_conn(db_path) as conn:
        assert EXPECTED_TABLES <= _names(conn, "table")


def test_migration_adds_columns_to_an_old_database(tmp_path):
    """An older DB without fetch_status/source must be upgraded, not recreated."""
    path = tmp_path / "old.db"
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE documents (id TEXT PRIMARY KEY, title TEXT NOT NULL,
                url TEXT UNIQUE NOT NULL, date_published TEXT, summary TEXT,
                category TEXT, hash TEXT NOT NULL);
            CREATE TABLE files (id TEXT PRIMARY KEY, document_id TEXT NOT NULL,
                file_url TEXT NOT NULL, file_path TEXT, file_hash TEXT,
                file_type TEXT, pages INTEGER);
            CREATE TABLE tables (id TEXT PRIMARY KEY, document_id TEXT NOT NULL,
                source_file_id TEXT, table_json TEXT, n_rows INTEGER, n_cols INTEGER);
            CREATE TABLE chunks (id TEXT PRIMARY KEY, document_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL, text TEXT NOT NULL, token_count INTEGER);
            INSERT INTO documents(id,title,url,hash) VALUES ('d1','T','u','h');
            """
        )
    init_db(path)
    with get_conn(path) as conn:
        assert "fetch_status" in {r[1] for r in conn.execute("PRAGMA table_info(files)")}
        assert "source" in {r[1] for r in conn.execute("PRAGMA table_info(chunks)")}
        assert "page" in {r[1] for r in conn.execute("PRAGMA table_info(tables)")}
        # Pre-existing data survives the upgrade.
        assert conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 1


def test_read_only_connection_can_read(seeded_db):
    with get_conn(seeded_db["db_path"], read_only=True) as conn:
        assert conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 1


def test_read_only_connection_rejects_writes(seeded_db):
    """Phase 2's SQL agent depends on this being enforced by SQLite itself."""
    with pytest.raises(sqlite3.OperationalError):
        with get_conn(seeded_db["db_path"], read_only=True) as conn:
            conn.execute("DELETE FROM documents")


def test_doc_texts_is_unique_per_source_file(seeded_db):
    db, doc_id, file_id = (
        seeded_db["db_path"], seeded_db["doc_id"], seeded_db["file_id"],
    )
    with get_conn(db) as conn:
        conn.execute(
            "INSERT INTO doc_texts(id, document_id, source_file_id, text, n_chars) "
            "VALUES ('t1',?,?,'first',5)", (doc_id, file_id),
        )
    with pytest.raises(sqlite3.IntegrityError):
        with get_conn(db) as conn:
            conn.execute(
                "INSERT INTO doc_texts(id, document_id, source_file_id, text, n_chars) "
                "VALUES ('t2',?,?,'second',6)", (doc_id, file_id),
            )
