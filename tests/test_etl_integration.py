"""End-to-end ETL tests: DB -> chunks, including the 49->49 regression guard."""
from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

from pipeline import etl as etl_mod
from pipeline.etl import get_document_text, run_etl
from scraper.db import get_conn

LONG_TEXT = " ".join(
    f"Paragraph {i} of the press note describing CPI inflation trends in detail."
    for i in range(200)
)


@pytest.fixture()
def etl_env(seeded_db, tmp_path, monkeypatch):
    """Point settings at a temp processed dir and cwd so no repo files are touched."""
    monkeypatch.setattr(etl_mod.settings, "processed_data_dir", str(tmp_path / "processed"))
    monkeypatch.setattr(etl_mod.settings, "db_path", seeded_db["db_path"])
    monkeypatch.setattr(etl_mod.settings, "chunk_size", 180)
    monkeypatch.setattr(etl_mod.settings, "chunk_overlap", 40)
    monkeypatch.setattr(etl_mod.settings, "min_chunk_words", 20)
    monkeypatch.chdir(tmp_path)
    return seeded_db


def _add_pdf_text(db_path: str, doc_id: str, file_id: str, text: str) -> None:
    with get_conn(db_path) as conn:
        conn.execute(
            "INSERT INTO doc_texts(id, document_id, source_file_id, text, n_chars, n_pages) "
            "VALUES (?,?,?,?,?,?)",
            (str(uuid.uuid4()), doc_id, file_id, text, len(text), 4),
        )


def _catalog(tmp_path: Path) -> dict:
    return json.loads((tmp_path / "datasets" / "catalog.json").read_text())


def test_get_document_text_prefers_pdf_text_over_summary(etl_env):
    db, doc_id, file_id = etl_env["db_path"], etl_env["doc_id"], etl_env["file_id"]
    _add_pdf_text(db, doc_id, file_id, LONG_TEXT)
    with get_conn(db) as conn:
        doc = dict(conn.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone())
        text, source = get_document_text(conn, doc)
    assert source == "pdf_text"
    assert text == LONG_TEXT


def test_get_document_text_falls_back_to_summary(etl_env):
    db, doc_id = etl_env["db_path"], etl_env["doc_id"]
    with get_conn(db) as conn:
        doc = dict(conn.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone())
        text, source = get_document_text(conn, doc)
    assert source == "summary"
    assert "listing-page blurb" in text


def test_one_long_document_produces_many_chunks(etl_env, tmp_path):
    """THE regression test: 1 document with a real PDF must not yield 1 chunk."""
    db, doc_id, file_id = etl_env["db_path"], etl_env["doc_id"], etl_env["file_id"]
    _add_pdf_text(db, doc_id, file_id, LONG_TEXT)

    run_etl(db)

    with get_conn(db) as conn:
        n_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        sources = {
            r[0] for r in conn.execute("SELECT DISTINCT source FROM chunks")
        }
    assert n_chunks > 5, f"expected many chunks from a long PDF, got {n_chunks}"
    assert sources == {"pdf_text"}

    cat = _catalog(tmp_path)
    assert cat["total_documents"] == 1
    assert cat["total_chunks"] == n_chunks
    assert cat["chunks_per_document"] > 5
    assert cat["chunks_by_source"] == {"pdf_text": n_chunks}


def test_catalog_records_summary_only_runs(etl_env, tmp_path):
    """Without PDF text the mix must say so -- this is the old broken state."""
    run_etl(etl_env["db_path"])
    cat = _catalog(tmp_path)
    assert list(cat["chunks_by_source"]) == ["summary"]
    assert cat["chunks_per_document"] == 1.0


def test_catalog_paths_use_forward_slashes(etl_env, tmp_path):
    run_etl(etl_env["db_path"])
    cat = _catalog(tmp_path)
    assert "\\" not in cat["files"]["chunks_parquet"]


def test_etl_is_rerunnable_without_duplicating_chunks(etl_env):
    db, doc_id, file_id = etl_env["db_path"], etl_env["doc_id"], etl_env["file_id"]
    _add_pdf_text(db, doc_id, file_id, LONG_TEXT)
    run_etl(db)
    with get_conn(db) as conn:
        first = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    run_etl(db)
    with get_conn(db) as conn:
        second = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    assert first == second


def test_invalid_documents_are_excluded(etl_env):
    db = etl_env["db_path"]
    with get_conn(db) as conn:
        conn.execute(
            "INSERT INTO documents(id,title,url,date_published,summary,category,hash) "
            "VALUES ('bad','x','https://example.test/bad.pdf',NULL,NULL,'C','h')",
        )
    run_etl(db)
    with get_conn(db) as conn:
        doc_ids = {r[0] for r in conn.execute("SELECT DISTINCT document_id FROM chunks")}
    assert "bad" not in doc_ids
