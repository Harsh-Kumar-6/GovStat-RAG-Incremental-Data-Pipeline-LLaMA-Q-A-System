"""MoSPI crawler — reads the ministry's own JSON listing API.

Why there is no HTML parsing here
---------------------------------
mospi.gov.in is a client-rendered single-page app. Every path -- including
``/robots.txt`` and the homepage -- returns the same 2644-byte HTML shell, and
all content is fetched by JavaScript from a JSON API under ``/api/``. An
``httpx`` + BeautifulSoup crawler therefore cannot see a single document,
however many listing pages it is pointed at.

This module calls the same public endpoints the website calls:

    POST /api/latest-release/get-web-latest-release-list
    POST /api/publications-reports/get-web-publications-report-list

Both are paginated and return title, published date, and the real path of each
attached PDF under ``/uploads/``. Those PDFs are genuine (``application/pdf``,
multi-MB); the document URLs used previously were not.
"""
from __future__ import annotations

import hashlib
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from html import unescape
from typing import Dict, Iterator, List, Optional

import httpx

from scraper.config import settings
from scraper.db import get_conn, init_db
from scraper.logger import get_logger

log = get_logger(__name__)

FILE_BASE = "https://www.mospi.gov.in/"

# The listing endpoints, by source name. Both take the same request body.
ENDPOINTS: Dict[str, str] = {
    "latest-release": "latest-release/get-web-latest-release-list",
    "publications-reports": "publications-reports/get-web-publications-report-list",
}

# Category is derived from the title: the API returns no category field, but the
# plan needs category facets and the SQL agent needs a way to filter by series.
# Order matters -- the first match wins.
CATEGORY_RULES: List[tuple[str, str]] = [
    (r"\bCPI\b|consumer price", "Price Statistics > CPI"),
    (r"\bWPI\b|wholesale price", "Price Statistics > WPI"),
    (r"\bIIP\b|index of industrial production", "Industrial Statistics > IIP"),
    (r"\bASI\b|annual survey of industries", "Industrial Statistics > ASI"),
    (r"\bGDP\b|gross domestic product|national account|national income", "National Accounts > GDP"),
    (r"\bPLFS\b|labour force", "Labour Statistics > PLFS"),
    (r"\bHCES\b|consumption expenditure", "Household Surveys > HCES"),
    (r"\bASUSE\b|unincorporated", "Household Surveys > ASUSE"),
    (r"\bSDG\b|sustainable development", "Publications > SDG"),
    (r"statistical year ?book", "Publications > Statistical Year Book"),
    (r"india in figures", "Publications > India in Figures"),
    (r"women and men|youth in india|children in india", "Publications > Social Statistics"),
    (r"press note|press release", "Press Release"),
]


@dataclass
class DocumentMeta:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    url: str = ""
    date_published: Optional[str] = None
    summary: str = ""
    category: str = ""
    hash: str = ""
    file_links: List[str] = field(default_factory=list)


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def classify_category(title: str) -> str:
    """Map a release title onto a category, for facets and SQL filtering."""
    for pattern, category in CATEGORY_RULES:
        if re.search(pattern, title, re.I):
            return category
    return "Other"


def _clean_date(raw: str) -> Optional[str]:
    """Normalize a date string to ISO `YYYY-MM-DD`, or None."""
    if not raw:
        return None
    raw = raw.strip()
    # The API returns ISO-8601 timestamps; take the date part.
    m = re.match(r"(\d{4}-\d{2}-\d{2})T", raw)
    if m:
        return m.group(1)
    for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%B %d, %Y", "%d %B %Y",
                "%Y-%m-%d", "%b %d, %Y", "%d %b %Y", "%B %Y"):
        try:
            return datetime.strptime(raw, fmt).date().isoformat()
        except ValueError:
            pass
    m = re.search(r"(\d{1,2})[/-](\d{1,2})[/-](\d{4})", raw)
    if m:
        try:
            return datetime(int(m.group(3)), int(m.group(2)), int(m.group(1))).date().isoformat()
        except ValueError:
            pass
    return None


def _clean_title(raw: str) -> str:
    """Normalize a title from the API.

    The two endpoints are inconsistent: `latest-release` returns plain text
    with embedded CRLFs, while `publications-reports` returns HTML fragments
    (`<p>`, `<span class=...>`, `&nbsp;`). Both are reduced to plain text here,
    because the title is indexed, displayed, and cited.
    """
    text = raw or ""
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.I)
    text = re.sub(r"<[^>]+>", "", text)          # strip HTML tags
    text = unescape(text)                         # &nbsp; &amp; &#8211; ...
    text = text.replace("\xa0", " ").replace("\r", " ")
    return re.sub(r"\s+", " ", text).strip()


def extract_file_urls(item: Dict) -> List[str]:
    """Absolute URLs of every PDF attached to a listing item.

    The API exposes up to three attachments as file_one/file_two/file_three,
    each either null or an object with a `path` relative to the site root.
    """
    urls: List[str] = []
    for key in ("file_one", "file_two", "file_three"):
        f = item.get(key)
        if not isinstance(f, dict):
            continue
        path = (f.get("path") or "").strip()
        if not path:
            continue
        if (f.get("filemime") or "").lower() not in ("application/pdf", ""):
            continue
        urls.append(FILE_BASE + path.lstrip("/"))
    return urls


def fetch_listing_page(
    source: str,
    page_no: int,
    client: httpx.Client,
    page_size: Optional[int] = None,
) -> tuple[List[Dict], int]:
    """Fetch one page of a listing. Returns (items, total_pages)."""
    if source not in ENDPOINTS:
        raise ValueError(f"unknown source {source!r}; expected one of {sorted(ENDPOINTS)}")

    url = f"{settings.mospi_api_base.rstrip('/')}/{ENDPOINTS[source]}"
    payload = {
        "page_no": page_no,
        "page_size": page_size or settings.api_page_size,
        "search_term": "",
        "sort_field": "published_year",
        "sort_order": "DESC",
        "from_date": "",
        "to_date": "",
        "lang": settings.api_lang,
        "data_source": "web",
    }
    resp = client.post(url, json=payload)
    resp.raise_for_status()
    body = resp.json()
    items = body.get("data") or []
    total_pages = int((body.get("pagination") or {}).get("totalPages") or 1)
    log.info(
        "listing_page_fetched",
        source=source,
        page=page_no,
        items=len(items),
        total_pages=total_pages,
    )
    return items, total_pages


def iter_documents(
    sources: Optional[List[str]] = None,
    max_pages: Optional[int] = None,
) -> Iterator[DocumentMeta]:
    """Yield DocumentMeta for every listing item across the configured sources.

    `max_pages` caps listing pages **per source**, which is what makes
    MAX_PAGES_PER_RUN meaningful -- the previous implementation ignored it.
    """
    sources = sources or settings.sources
    max_pages = max_pages or settings.max_pages_per_run

    headers = {
        "User-Agent": settings.user_agent,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    with httpx.Client(headers=headers, timeout=60, follow_redirects=True) as client:
        for source in sources:
            page_no = 1
            total_pages = 1
            while page_no <= min(max_pages, total_pages):
                try:
                    items, total_pages = fetch_listing_page(source, page_no, client)
                except Exception as exc:
                    log.error("listing_page_failed", source=source, page=page_no, error=str(exc))
                    break

                for item in items:
                    title = _clean_title(item.get("title", ""))
                    file_urls = extract_file_urls(item)
                    if not title or not file_urls:
                        log.warning(
                            "listing_item_skipped",
                            source=source,
                            item_id=item.get("id"),
                            reason="no_title" if not title else "no_pdf",
                        )
                        continue

                    date_published = _clean_date(
                        item.get("published_year") or item.get("start_date") or ""
                    )
                    # The canonical URL is the first attachment: it is stable,
                    # unique, and directly fetchable, unlike the SPA route.
                    canonical = file_urls[0]
                    yield DocumentMeta(
                        id=str(uuid.uuid4()),
                        title=title,
                        url=canonical,
                        date_published=date_published,
                        # Deliberately left empty. The real summary is the PDF's
                        # extracted text, written to doc_texts by parse.py.
                        # Inventing one here is what corrupted the old dataset.
                        summary="",
                        category=classify_category(title),
                        hash=_sha256(
                            f"{title}|{date_published}|{'|'.join(file_urls)}".encode()
                        ),
                        file_links=file_urls,
                    )

                page_no += 1
                if page_no <= min(max_pages, total_pages):
                    time.sleep(settings.rate_limit_delay)


def crawl(
    seed_urls: Optional[List[str]] = None,
    max_pages: Optional[int] = None,
    db_path: Optional[str] = None,
    sources: Optional[List[str]] = None,
) -> List[DocumentMeta]:
    """Load MoSPI documents into the database.

    Idempotent: a SHA-256 fingerprint over title, date, and attachment URLs
    decides whether a row is inserted, updated, or skipped.

    `seed_urls` is accepted and ignored -- the site has no crawlable HTML. It
    remains in the signature so existing callers and the Makefile keep working.
    """
    if seed_urls:
        log.warning(
            "seed_urls_ignored",
            reason="mospi.gov.in is a client-rendered SPA; using the JSON API instead",
        )

    db_path = db_path or settings.db_path
    init_db(db_path)

    collected: List[DocumentMeta] = []
    stats = {"seen": 0, "inserted": 0, "updated": 0, "unchanged": 0}

    for doc in iter_documents(sources=sources, max_pages=max_pages):
        stats["seen"] += 1
        with get_conn(db_path) as conn:
            row = conn.execute(
                "SELECT id, hash FROM documents WHERE url = ?", (doc.url,)
            ).fetchone()

            if row and row["hash"] == doc.hash:
                stats["unchanged"] += 1
                continue

            if row:
                doc.id = row["id"]
                conn.execute(
                    "UPDATE documents SET title=?, date_published=?, category=?, hash=? "
                    "WHERE url=?",
                    (doc.title, doc.date_published, doc.category, doc.hash, doc.url),
                )
                stats["updated"] += 1
            else:
                conn.execute(
                    "INSERT INTO documents"
                    "(id, title, url, date_published, summary, category, hash) "
                    "VALUES (?,?,?,?,?,?,?)",
                    (doc.id, doc.title, doc.url, doc.date_published,
                     doc.summary, doc.category, doc.hash),
                )
                stats["inserted"] += 1

            for file_url in doc.file_links:
                exists = conn.execute(
                    "SELECT 1 FROM files WHERE document_id=? AND file_url=?",
                    (doc.id, file_url),
                ).fetchone()
                if not exists:
                    conn.execute(
                        "INSERT INTO files(id, document_id, file_url, file_type) "
                        "VALUES (?,?,?,?)",
                        (str(uuid.uuid4()), doc.id, file_url, "pdf"),
                    )

        collected.append(doc)
        log.info("loaded_document", title=doc.title[:70], category=doc.category)

    log.info("crawl_complete", **stats)
    return collected
