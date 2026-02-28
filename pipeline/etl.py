"""ETL pipeline: transform raw documents into chunks and validate."""
from __future__ import annotations

import json
import re
import uuid
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from scraper.config import settings
from scraper.db import get_conn, init_db
from scraper.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> List[str]:
    """Split text into overlapping token-approximate chunks."""
    if not text.strip():
        return []
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = end - overlap
    return chunks


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

class ValidationError(Exception):
    pass


def validate_document(row: Dict[str, Any]) -> List[str]:
    """Return list of validation issues (empty = valid)."""
    issues = []
    if not row.get("title") or len(str(row["title"]).strip()) < 3:
        issues.append("title_empty_or_too_short")
    if not row.get("url"):
        issues.append("url_missing")
    if row.get("date_published"):
        try:
            date.fromisoformat(str(row["date_published"]))
        except ValueError:
            issues.append(f"invalid_date: {row['date_published']}")
    if not row.get("summary"):
        issues.append("summary_empty")
    return issues


# ---------------------------------------------------------------------------
# ETL run
# ---------------------------------------------------------------------------

def run_etl(db_path: Optional[str] = None) -> None:
    db_path = db_path or settings.db_path
    init_db(db_path)

    processed_dir = Path(settings.processed_data_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    with get_conn(db_path) as conn:
        rows = conn.execute("SELECT * FROM documents").fetchall()

    docs = [dict(r) for r in rows]
    log.info("etl_start", total_docs=len(docs))

    valid_docs = []
    invalid_count = 0

    for doc in docs:
        issues = validate_document(doc)
        if issues:
            log.warning("document_invalid", doc_id=doc["id"], issues=issues)
            invalid_count += 1
            continue
        valid_docs.append(doc)

    log.info("validation_complete", valid=len(valid_docs), invalid=invalid_count)

    # Deduplicate by URL
    seen_urls = set()
    deduped = []
    for doc in valid_docs:
        if doc["url"] not in seen_urls:
            seen_urls.add(doc["url"])
            deduped.append(doc)
    log.info("deduplication_complete", before=len(valid_docs), after=len(deduped))

    # Write Parquet
    df = pd.DataFrame(deduped)
    parquet_path = processed_dir / "documents.parquet"
    df.to_parquet(str(parquet_path), index=False)
    log.info("wrote_parquet", path=str(parquet_path))

    # Chunk texts and store back to DB
    all_chunks = []
    with get_conn(db_path) as conn:
        conn.execute("DELETE FROM chunks")  # rebuild
        for doc in deduped:
            text = (doc.get("summary") or "").strip()
            if not text:
                continue
            chunks = chunk_text(text, settings.chunk_size, settings.chunk_overlap)
            for i, chunk in enumerate(chunks):
                chunk_id = str(uuid.uuid4())
                token_count = len(chunk.split())
                conn.execute(
                    "INSERT INTO chunks(id, document_id, chunk_index, text, token_count) VALUES (?,?,?,?,?)",
                    (chunk_id, doc["id"], i, chunk, token_count),
                )
                all_chunks.append({
                    "id": chunk_id,
                    "document_id": doc["id"],
                    "chunk_index": i,
                    "text": chunk,
                    "token_count": token_count,
                    "title": doc.get("title", ""),
                    "url": doc.get("url", ""),
                    "date_published": doc.get("date_published"),
                    "category": doc.get("category", ""),
                })

    # Write chunks Parquet
    chunks_df = pd.DataFrame(all_chunks)
    chunks_path = processed_dir / "chunks.parquet"
    chunks_df.to_parquet(str(chunks_path), index=False)
    log.info("wrote_chunks_parquet", path=str(chunks_path), total_chunks=len(all_chunks))

    # Catalog
    by_category = df.groupby("category").size().to_dict() if "category" in df.columns else {}
    by_date = {}
    if "date_published" in df.columns:
        df["year"] = df["date_published"].str[:4]
        by_date = df.groupby("year").size().to_dict()

    catalog = {
        "total_documents": len(deduped),
        "total_chunks": len(all_chunks),
        "by_category": by_category,
        "by_year": by_date,
        "files": {
            "documents_parquet": str(parquet_path),
            "chunks_parquet": str(chunks_path),
        },
    }
    catalog_path = Path("datasets") / "catalog.json"
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_path.write_text(json.dumps(catalog, indent=2))
    log.info("wrote_catalog", path=str(catalog_path))

    log.info("etl_complete", total_chunks=len(all_chunks))