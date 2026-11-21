from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from scraper.db import get_conn, init_db


@pytest.fixture()
def db_path(tmp_path: Path) -> str:
    """A fresh, initialized SQLite database for one test."""
    path = tmp_path / "test.db"
    init_db(path)
    return str(path)


@pytest.fixture()
def seeded_db(db_path: str):
    """A database holding one document with one PDF file row."""
    doc_id = str(uuid.uuid4())
    file_id = str(uuid.uuid4())
    with get_conn(db_path) as conn:
        conn.execute(
            "INSERT INTO documents(id, title, url, date_published, summary, category, hash) "
            "VALUES (?,?,?,?,?,?,?)",
            (
                doc_id,
                "Consumer Price Index - March 2025",
                "https://example.test/cpi-march-2025.pdf",
                "2025-04-12",
                "Short listing-page blurb about CPI.",
                "Price Statistics > CPI",
                "deadbeef",
            ),
        )
        conn.execute(
            "INSERT INTO files(id, document_id, file_url, file_type) VALUES (?,?,?,?)",
            (file_id, doc_id, "https://example.test/cpi-march-2025.pdf", "pdf"),
        )
    return {"db_path": db_path, "doc_id": doc_id, "file_id": file_id}


@pytest.fixture()
def minimal_pdf_bytes() -> bytes:
    """A tiny but structurally valid single-page PDF containing known text."""
    body = (
        b"%PDF-1.4\n"
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
        b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 200 200]"
        b"/Resources<</Font<</F1 4 0 R>>>>/Contents 5 0 R>>endobj\n"
        b"4 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\n"
        b"5 0 obj<</Length 54>>stream\n"
        b"BT /F1 12 Tf 20 100 Td (CPI inflation 5.2 percent) Tj ET\n"
        b"endstream endobj\n"
        b"trailer<</Root 1 0 R>>\n"
    )
    return body
