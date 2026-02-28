"""Download PDFs and extract text + tables."""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from pathlib import Path
from typing import List, Optional, Tuple

import httpx
import pdfplumber

from scraper.config import settings
from scraper.db import get_conn
from scraper.logger import get_logger

log = get_logger(__name__)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def download_pdf(url: str, dest_dir: Path) -> Optional[Path]:
    """Download PDF to dest_dir, return local path."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    filename = hashlib.md5(url.encode()).hexdigest() + ".pdf"
    dest = dest_dir / filename

    if dest.exists():
        log.info("pdf_already_downloaded", url=url, path=str(dest))
        return dest

    try:
        with httpx.Client(headers={"User-Agent": settings.user_agent}, follow_redirects=True, timeout=60) as client:
            resp = client.get(url)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
        log.info("pdf_downloaded", url=url, size=dest.stat().st_size)
        return dest
    except Exception as exc:
        log.error("pdf_download_failed", url=url, error=str(exc))
        return None


def extract_text_and_tables(pdf_path: Path) -> Tuple[str, List[dict]]:
    """Return (full_text, list_of_table_dicts) from a PDF."""
    full_text_parts: List[str] = []
    tables: List[dict] = []

    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text() or ""
                full_text_parts.append(text)

                raw_tables = page.extract_tables()
                for t in raw_tables:
                    if not t:
                        continue
                    # Convert to list-of-dicts using first row as header
                    header = [str(c).strip() if c else f"col_{i}" for i, c in enumerate(t[0])]
                    rows = []
                    for row in t[1:]:
                        rows.append({header[i]: (str(v).strip() if v else "") for i, v in enumerate(row)})
                    tables.append({"page": page_num, "header": header, "rows": rows})
    except Exception as exc:
        log.error("pdf_extract_failed", path=str(pdf_path), error=str(exc))

    return "\n".join(full_text_parts), tables


def parse_all_pdfs(db_path: Optional[str] = None) -> None:
    """Download + parse all pending PDF files in the DB."""
    db_path = db_path or settings.db_path
    pdf_dir = Path(settings.raw_data_dir) / "pdf"

    with get_conn(db_path) as conn:
        rows = conn.execute(
            "SELECT id, document_id, file_url FROM files WHERE file_type = 'pdf' AND file_path IS NULL"
        ).fetchall()

    log.info("pdfs_to_process", count=len(rows))

    for row in rows:
        file_id = row["id"]
        doc_id = row["document_id"]
        url = row["file_url"]

        time.sleep(settings.rate_limit_delay)
        pdf_path = download_pdf(url, pdf_dir)
        if not pdf_path:
            continue

        file_hash = _sha256_file(pdf_path)
        text, tables = extract_text_and_tables(pdf_path)

        # Count pages
        try:
            with pdfplumber.open(str(pdf_path)) as pdf:
                n_pages = len(pdf.pages)
        except Exception:
            n_pages = 0

        with get_conn(db_path) as conn:
            conn.execute(
                "UPDATE files SET file_path=?, file_hash=?, pages=? WHERE id=?",
                (str(pdf_path), file_hash, n_pages, file_id),
            )
            # Store extracted text as a synthetic document summary if empty
            if text:
                conn.execute(
                    "UPDATE documents SET summary=? WHERE id=? AND (summary IS NULL OR summary='')",
                    (text[:1000], doc_id),
                )
            # Store tables
            for tbl in tables[:5]:  # limit to 5 tables per PDF
                tbl_id = str(uuid.uuid4())
                conn.execute(
                    "INSERT INTO tables(id, document_id, source_file_id, table_json, n_rows, n_cols) VALUES (?,?,?,?,?,?)",
                    (
                        tbl_id,
                        doc_id,
                        file_id,
                        json.dumps(tbl),
                        len(tbl.get("rows", [])),
                        len(tbl.get("header", [])),
                    ),
                )

        log.info("pdf_parsed", url=url, pages=n_pages, tables=len(tables))


