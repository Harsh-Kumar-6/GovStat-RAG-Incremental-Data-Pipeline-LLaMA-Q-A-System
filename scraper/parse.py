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

PDF_MAGIC = b"%PDF-"


class NotAPdfError(Exception):
    """The server returned 200 but the body is not a PDF."""


class PdfTooLargeError(Exception):
    """The PDF exceeds MAX_PDF_MB and was deliberately skipped."""


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def looks_like_pdf(content: bytes) -> bool:
    """True if the bytes begin with the PDF magic number.

    MoSPI answers dead document links with HTTP 200 and an HTML shell, so the
    status code alone cannot be trusted. Checking magic bytes is what stops an
    HTML error page from being written to disk as a .pdf and silently yielding
    zero text downstream.
    """
    return content[:5] == PDF_MAGIC


def download_pdf(url: str, dest_dir: Path) -> Optional[Path]:
    """Download a PDF to dest_dir and return its local path.

    Returns None if the download failed or the response was not a PDF.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    filename = hashlib.md5(url.encode()).hexdigest() + ".pdf"
    dest = dest_dir / filename

    if dest.exists():
        log.info("pdf_already_downloaded", url=url, path=str(dest))
        return dest

    try:
        with httpx.Client(
            headers={"User-Agent": settings.user_agent}, follow_redirects=True, timeout=60
        ) as client:
            resp = client.get(url)
            resp.raise_for_status()
            content = resp.content

        max_bytes = int(settings.max_pdf_mb * 1024 * 1024)
        if len(content) > max_bytes:
            log.warning(
                "pdf_too_large_skipped",
                url=url,
                size_mb=round(len(content) / 1024 / 1024, 1),
                limit_mb=settings.max_pdf_mb,
            )
            raise PdfTooLargeError(url)

        if not looks_like_pdf(content):
            ctype = resp.headers.get("content-type", "unknown")
            log.error(
                "pdf_download_not_a_pdf",
                url=url,
                content_type=ctype,
                size=len(content),
                head=content[:40].decode("utf-8", "replace"),
            )
            raise NotAPdfError(url)

        dest.write_bytes(content)
        log.info("pdf_downloaded", url=url, size=dest.stat().st_size)
        return dest
    except (NotAPdfError, PdfTooLargeError):
        return None
    except Exception as exc:
        log.error("pdf_download_failed", url=url, error=str(exc))
        return None


def extract_text_and_tables(
    pdf_path: Path, max_table_pages: Optional[int] = None
) -> Tuple[str, List[dict], int]:
    """Return (full_text, list_of_table_dicts, n_pages) from a PDF.

    Text is extracted from every page. Table detection is limited to the first
    `max_table_pages` pages: pdfplumber's table finder scales badly with the
    number of ruling lines, and on a 200-page yearbook it can consume minutes
    of CPU per page while adding little the press notes do not already carry.
    """
    if max_table_pages is None:
        max_table_pages = settings.max_table_pages

    full_text_parts: List[str] = []
    tables: List[dict] = []
    n_pages = 0

    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            n_pages = len(pdf.pages)
            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text() or ""
                full_text_parts.append(text)

                if page_num > max_table_pages:
                    continue
                if len(tables) >= settings.max_tables_per_pdf:
                    continue

                for t in page.extract_tables():
                    if not t or not t[0]:
                        continue
                    header = [
                        str(c).strip() if c else f"col_{i}" for i, c in enumerate(t[0])
                    ]
                    rows = []
                    for row in t[1:]:
                        rows.append(
                            {
                                header[i]: (str(v).strip() if v else "")
                                for i, v in enumerate(row)
                                if i < len(header)
                            }
                        )
                    tables.append({"page": page_num, "header": header, "rows": rows})
    except Exception as exc:
        log.error("pdf_extract_failed", path=str(pdf_path), error=str(exc))

    return "\n".join(full_text_parts), tables, n_pages


def parse_all_pdfs(db_path: Optional[str] = None) -> dict:
    """Download + parse all pending PDF files in the DB.

    Returns a counts dict so callers (and tests) can assert on the outcome
    instead of reading logs.
    """
    db_path = db_path or settings.db_path
    pdf_dir = Path(settings.raw_data_dir) / "pdf"

    with get_conn(db_path) as conn:
        rows = conn.execute(
            "SELECT id, document_id, file_url FROM files "
            "WHERE file_type = 'pdf' AND file_path IS NULL"
        ).fetchall()

    log.info("pdfs_to_process", count=len(rows))
    stats = {"attempted": 0, "downloaded": 0, "failed": 0, "empty_text": 0, "tables": 0}

    for row in rows:
        file_id, doc_id, url = row["id"], row["document_id"], row["file_url"]
        stats["attempted"] += 1

        time.sleep(settings.rate_limit_delay)
        pdf_path = download_pdf(url, pdf_dir)
        if not pdf_path:
            stats["failed"] += 1
            with get_conn(db_path) as conn:
                conn.execute(
                    "UPDATE files SET fetch_status=? WHERE id=?",
                    ("download_failed", file_id),
                )
            continue

        file_hash = _sha256_file(pdf_path)
        text, tables, n_pages = extract_text_and_tables(pdf_path)
        stats["downloaded"] += 1
        if not text.strip():
            stats["empty_text"] += 1

        with get_conn(db_path) as conn:
            conn.execute(
                "UPDATE files SET file_path=?, file_hash=?, pages=?, fetch_status=? WHERE id=?",
                (str(pdf_path), file_hash, n_pages, "ok", file_id),
            )

            # Persist the FULL extracted text. Previously this was truncated to
            # text[:1000] and written into documents.summary, which is why the
            # ETL produced exactly one chunk per document.
            if text.strip():
                conn.execute(
                    "INSERT OR REPLACE INTO doc_texts"
                    "(id, document_id, source_file_id, text, n_chars, n_pages, extractor) "
                    "VALUES (?,?,?,?,?,?,?)",
                    (
                        str(uuid.uuid4()),
                        doc_id,
                        file_id,
                        text,
                        len(text),
                        n_pages,
                        "pdfplumber",
                    ),
                )

            conn.execute("DELETE FROM tables WHERE source_file_id = ?", (file_id,))
            for tbl in tables[: settings.max_tables_per_pdf]:
                conn.execute(
                    "INSERT INTO tables"
                    "(id, document_id, source_file_id, table_json, page, n_rows, n_cols) "
                    "VALUES (?,?,?,?,?,?,?)",
                    (
                        str(uuid.uuid4()),
                        doc_id,
                        file_id,
                        json.dumps(tbl),
                        tbl.get("page"),
                        len(tbl.get("rows", [])),
                        len(tbl.get("header", [])),
                    ),
                )
                stats["tables"] += 1

        log.info(
            "pdf_parsed", url=url, pages=n_pages, chars=len(text), tables=len(tables)
        )

    log.info("parse_complete", **stats)
    return stats
