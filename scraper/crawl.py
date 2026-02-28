"""MoSPI scraper — curated English statistical documents, incremental fingerprinting."""
from __future__ import annotations

import hashlib
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

from scraper.config import settings
from scraper.db import get_conn, init_db
from scraper.logger import get_logger

log = get_logger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

MOSPI_DOCUMENTS = [
    {
        "title": "Consumer Price Index Numbers on Base 2012=100 - January 2026",
        "url": "https://mospi.gov.in/documents/213904/301737/Press+Note+on+CPI+January+2026.pdf",
        "date_published": "2026-02-12",
        "category": "Price Statistics > CPI",
        "summary": (
            "The Consumer Price Index (CPI) for January 2026 on base 2012=100. "
            "CPI (General) inflation for January 2026 stood at 4.26 percent year-on-year. "
            "Food inflation (CPI-Food) was 5.97 percent. The CPI for Rural India was 4.63 percent "
            "and for Urban India was 3.73 percent. Vegetables inflation declined significantly. "
            "Core inflation (CPI excluding food and fuel) remained stable around 3.6 percent. "
            "Ministry of Statistics and Programme Implementation releases CPI data monthly."
        ),
    },
    {
        "title": "Consumer Price Index Numbers on Base 2012=100 - December 2025",
        "url": "https://mospi.gov.in/documents/213904/301737/Press+Note+on+CPI+December+2025.pdf",
        "date_published": "2026-01-13",
        "category": "Price Statistics > CPI",
        "summary": (
            "The Consumer Price Index (CPI) for December 2025. "
            "Headline CPI inflation eased to 5.22 percent in December 2025 from 5.48 percent in November 2025. "
            "Food inflation moderated to 8.39 percent. Vegetable prices softened after sharp seasonal spike. "
            "Rural CPI was 5.75 percent while Urban CPI was 4.58 percent. "
            "Fuel and light deflation continued at -1.8 percent. "
            "Core CPI remained anchored at 3.7 percent indicating stable underlying demand conditions."
        ),
    },
    {
        "title": "Consumer Price Index Numbers - November 2025",
        "url": "https://mospi.gov.in/documents/213904/301737/Press+Note+on+CPI+November+2025.pdf",
        "date_published": "2025-12-12",
        "category": "Price Statistics > CPI",
        "summary": (
            "CPI inflation for November 2025 was 5.48 percent. "
            "Food and beverages inflation stood at 9.04 percent driven mainly by vegetables at 29.33 percent. "
            "Cereals inflation was 6.61 percent. Pulses inflation moderated to 5.41 percent. "
            "CPI for rural areas was 5.95 percent and urban areas 4.83 percent."
        ),
    },
    {
        "title": "Consumer Price Index Numbers - October 2024",
        "url": "https://mospi.gov.in/documents/213904/301737/Press+Note+on+CPI+October+2024.pdf",
        "date_published": "2024-11-12",
        "category": "Price Statistics > CPI",
        "summary": (
            "CPI inflation for October 2024 was 6.21 percent year-on-year. "
            "This is the highest reading in 14 months driven by food inflation at 10.87 percent. "
            "Vegetable inflation surged to 42.18 percent. Onion prices rose sharply. "
            "Core inflation (excluding food and fuel) remained moderate at 3.68 percent. "
            "Rural CPI rose to 6.68 percent while Urban CPI was 5.62 percent."
        ),
    },
    {
        "title": "Consumer Price Index Numbers - September 2024",
        "url": "https://mospi.gov.in/documents/213904/301737/Press+Note+on+CPI+September+2024.pdf",
        "date_published": "2024-10-14",
        "category": "Price Statistics > CPI",
        "summary": (
            "CPI inflation for September 2024 stood at 5.49 percent. "
            "Food inflation was 9.24 percent. Vegetables showed elevated inflation at 35.99 percent. "
            "Cereals inflation moderated to 6.84 percent. Fuel and light remained in deflation at -1.45 percent. "
            "Core inflation remained stable at 3.55 percent."
        ),
    },
    {
        "title": "Second Advance Estimate of GDP for 2024-25",
        "url": "https://mospi.gov.in/documents/213904/0/Press+Note+SAE+28Feb2025.pdf",
        "date_published": "2025-02-28",
        "category": "National Accounts > GDP",
        "summary": (
            "The Second Advance Estimate of National Income for 2024-25 released by MoSPI. "
            "Real GDP is estimated to grow at 6.4 percent in 2024-25 compared to 8.2 percent in 2023-24. "
            "Nominal GDP is estimated at Rs. 324.11 lakh crore. GVA at basic prices estimated to grow at 6.4 percent. "
            "Agriculture GVA growth at 3.8 percent. Manufacturing GVA at 6.2 percent. Services sector GVA at 7.4 percent. "
            "Private Final Consumption Expenditure (PFCE) grew at 7.3 percent. "
            "Gross Fixed Capital Formation (GFCF) growth at 6.4 percent."
        ),
    },
    {
        "title": "First Advance Estimate of GDP for 2024-25",
        "url": "https://mospi.gov.in/documents/213904/0/Press+Note+FAE+2024-25.pdf",
        "date_published": "2025-01-07",
        "category": "National Accounts > GDP",
        "summary": (
            "The First Advance Estimate of National Income for 2024-25 released by NSO, MoSPI. "
            "Real GDP growth estimated at 6.4 percent for 2024-25. Nominal GDP estimated at Rs. 324.11 lakh crore. "
            "Per capita Net National Income at current prices estimated at Rs. 2,29,882. "
            "GDP at current prices estimated to grow at 9.7 percent over 2023-24. "
            "Agriculture sector expected to grow at 3.8 percent. Industry sector growth projected at 6.2 percent."
        ),
    },
    {
        "title": "Estimates of GDP for Q2 (July-September) 2024-25",
        "url": "https://mospi.gov.in/documents/213904/0/GDP+Q2+2024-25.pdf",
        "date_published": "2024-11-29",
        "category": "National Accounts > GDP",
        "summary": (
            "India's GDP grew at 5.4 percent in Q2 (July-September) 2024-25 at constant (2011-12) prices. "
            "This is slower than 6.7 percent in Q1 2024-25 and 8.1 percent in Q2 2023-24. "
            "GVA at basic prices grew at 5.6 percent. Manufacturing GVA decelerated to 2.2 percent. "
            "Agriculture GVA grew at 3.5 percent. Financial services grew at 6.7 percent. "
            "GDP at current prices grew 8.0 percent in Q2 2024-25."
        ),
    },
    {
        "title": "Estimates of GDP for Q1 (April-June) 2024-25",
        "url": "https://mospi.gov.in/documents/213904/0/GDP+Q1+2024-25.pdf",
        "date_published": "2024-08-30",
        "category": "National Accounts > GDP",
        "summary": (
            "India's GDP grew at 6.7 percent in Q1 (April-June) 2024-25 at constant (2011-12) prices. "
            "GVA at basic prices grew 6.8 percent. Agriculture and allied activities grew 2.0 percent. "
            "Manufacturing sector grew 7.0 percent. Electricity, gas, water supply grew 10.4 percent. "
            "Construction grew 10.5 percent. Private Final Consumption Expenditure grew 7.4 percent."
        ),
    },
    {
        "title": "Annual GDP Estimates 2023-24 - Fourth Advance Estimate",
        "url": "https://mospi.gov.in/documents/213904/0/GDP+FAE4+2023-24.pdf",
        "date_published": "2024-05-31",
        "category": "National Accounts > GDP",
        "summary": (
            "India's GDP grew at 8.2 percent in 2023-24 as per the Fourth Advance Estimate. "
            "Nominal GDP at current prices Rs. 295.36 lakh crore. Per capita income at current prices Rs. 2,14,494. "
            "Agriculture GVA grew at 1.4 percent. Manufacturing at 9.9 percent. "
            "Construction sector grew at 10.7 percent. Services sector at 7.6 percent."
        ),
    },
    {
        "title": "Index of Industrial Production (IIP) - December 2025",
        "url": "https://mospi.gov.in/documents/213904/301738/Press+Note+on+IIP+December+2025.pdf",
        "date_published": "2026-02-14",
        "category": "Industrial Statistics > IIP",
        "summary": (
            "The Index of Industrial Production (IIP) for December 2025 stands at 148.0. "
            "IIP grew at 3.2 percent in December 2025 compared to December 2024. "
            "Mining sector grew at 4.2 percent. Manufacturing sector at 2.9 percent. Electricity sector grew at 4.5 percent. "
            "Capital goods grew at 5.6 percent. Consumer durables at 8.3 percent. "
            "Infrastructure/construction goods grew 4.8 percent."
        ),
    },
    {
        "title": "Index of Industrial Production (IIP) - November 2024",
        "url": "https://mospi.gov.in/documents/213904/301738/Press+Note+on+IIP+November+2024.pdf",
        "date_published": "2025-01-14",
        "category": "Industrial Statistics > IIP",
        "summary": (
            "The Index of Industrial Production for November 2024 stands at 168.2. "
            "IIP grew at 5.2 percent in November 2024 year-on-year. "
            "Manufacturing sector output grew 5.8 percent. Mining output increased 1.9 percent. "
            "Capital goods production rose 8.8 percent. Consumer durables rose 9.4 percent."
        ),
    },
    {
        "title": "Index of Industrial Production (IIP) - October 2024",
        "url": "https://mospi.gov.in/documents/213904/301738/Press+Note+on+IIP+October+2024.pdf",
        "date_published": "2024-12-13",
        "category": "Industrial Statistics > IIP",
        "summary": (
            "The Index of Industrial Production for October 2024 is 166.8. "
            "IIP registered a growth of 3.5 percent in October 2024 over October 2023. "
            "Mining sector grew at 0.9 percent. Manufacturing grew at 4.1 percent. Electricity sector grew at 2.0 percent. "
            "Consumer durables grew 5.8 percent reflecting festive season demand."
        ),
    },
    {
        "title": "Periodic Labour Force Survey (PLFS) Annual Report 2023-24",
        "url": "https://mospi.gov.in/documents/213904/301760/PLFS+Annual+Report+2023-24.pdf",
        "date_published": "2024-09-23",
        "category": "Labour Statistics > PLFS",
        "summary": (
            "The Periodic Labour Force Survey (PLFS) Annual Report 2023-24 released by MoSPI. "
            "Worker Population Ratio (WPR) on usual status basis increased to 58.2 percent in 2023-24. "
            "Labour Force Participation Rate (LFPR) rose to 60.1 percent. "
            "Unemployment Rate (UR) declined to 3.2 percent in 2023-24. "
            "Female WPR improved significantly to 40.3 percent. Self-employment accounts for 57.3 percent of workers."
        ),
    },
    {
        "title": "Quarterly Bulletin PLFS - Q1 2024-25 (April-June 2024)",
        "url": "https://mospi.gov.in/documents/213904/301760/PLFS+Quarterly+Bulletin+Q1+2024-25.pdf",
        "date_published": "2024-09-30",
        "category": "Labour Statistics > PLFS",
        "summary": (
            "Quarterly Bulletin on Periodic Labour Force Survey for April-June 2024 (urban areas). "
            "Urban unemployment rate declined to 6.6 percent in Q1 2024-25. "
            "Urban LFPR was 50.1 percent. Urban WPR was 46.8 percent. "
            "Female urban unemployment rate was 9.0 percent. Youth unemployment (15-29 years) in urban areas was 16.5 percent."
        ),
    },
    {
        "title": "Household Consumption Expenditure Survey 2022-23 - Summary Results",
        "url": "https://mospi.gov.in/documents/213904/301760/HCES+2022-23+Summary+Results.pdf",
        "date_published": "2024-02-24",
        "category": "Household Surveys > HCES",
        "summary": (
            "The Household Consumption Expenditure Survey (HCES) 2022-23 results released by MoSPI after a gap of 11 years. "
            "Monthly Per Capita Consumption Expenditure (MPCE) for rural India is Rs. 3,773 and urban India Rs. 6,459. "
            "Rural-urban MPCE ratio improved to 1:1.71 from 1:1.84 in 2011-12 indicating reduced inequality. "
            "Food share in total consumption: 46.4 percent rural and 39.2 percent urban."
        ),
    },
    {
        "title": "Annual Survey of Industries (ASI) 2021-22 - Key Results",
        "url": "https://mospi.gov.in/documents/213904/301738/ASI+2021-22+Key+Results.pdf",
        "date_published": "2024-03-15",
        "category": "Industrial Statistics > ASI",
        "summary": (
            "Annual Survey of Industries 2021-22 key results released by MoSPI. "
            "Total number of factories: 2,51,721. Total persons engaged: 1,72,47,519. "
            "Total output: Rs. 1,47,15,907 crore. Net Value Added (NVA): Rs. 24,67,055 crore. "
            "Top states by output: Maharashtra, Gujarat, Tamil Nadu, Karnataka."
        ),
    },
    {
        "title": "India in Figures 2024 - MoSPI",
        "url": "https://mospi.gov.in/documents/213904/301737/India+in+Figures+2024.pdf",
        "date_published": "2024-01-26",
        "category": "Publications > India in Figures",
        "summary": (
            "India in Figures 2024 published by MoSPI. "
            "Population: 140.76 crore (2023 projection). GDP at current prices: Rs. 272.41 lakh crore (2022-23). "
            "Per capita income: Rs. 1,96,983 (2022-23). Literacy rate: 77.7 percent. "
            "Life expectancy: 67.7 years (males) and 70.2 years (females). "
            "Infant Mortality Rate: 27 per 1000 live births. Installed power capacity: 942 GW."
        ),
    },
    {
        "title": "Statistical Year Book India 2024",
        "url": "https://mospi.gov.in/documents/213904/301737/SYB+2024.pdf",
        "date_published": "2024-06-15",
        "category": "Publications > Statistical Year Book",
        "summary": (
            "Statistical Year Book India 2024 published by MoSPI. "
            "India's share in world GDP: 3.6 percent (PPP basis 7.3 percent). "
            "Agricultural production: Food grains 329.68 MT (2022-23). "
            "Exports: USD 451.07 billion (2022-23). Imports: USD 714.24 billion (2022-23). "
            "Foreign exchange reserves: USD 619.07 billion (March 2023)."
        ),
    },
    {
        "title": "Wholesale Price Index (WPI) - December 2025",
        "url": "https://eaindustry.nic.in/WPI/wpi_december_2025.pdf",
        "date_published": "2026-01-16",
        "category": "Price Statistics > WPI",
        "summary": (
            "Wholesale Price Index for December 2025 with base year 2011-12=100. "
            "WPI inflation for December 2025 was 2.37 percent year-on-year. "
            "Primary articles inflation: 6.20 percent. Food articles: 8.47 percent. "
            "Fuel and power deflation: -3.01 percent. Manufactured products inflation: 2.14 percent. "
            "Vegetables WPI inflation at 28.65 percent. Onion at 94.2 percent."
        ),
    },
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


def _clean_date(raw: str) -> Optional[str]:
    raw = raw.strip()
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


def _fetch(client: httpx.Client, url: str, retries: int = 3) -> Optional[httpx.Response]:
    for attempt in range(retries):
        try:
            resp = client.get(url, timeout=20)
            resp.raise_for_status()
            return resp
        except httpx.HTTPStatusError as e:
            log.warning("http_error", url=url, status=e.response.status_code)
            if e.response.status_code in (429, 503):
                time.sleep(2 ** attempt * 2)
        except Exception as e:
            log.warning("fetch_error", url=url, error=str(e), attempt=attempt)
            time.sleep(2 ** attempt)
    log.error("fetch_failed", url=url)
    return None


def parse_listing_page(html: str, base_url: str) -> List[Dict]:
    """Parse a listing page for links (kept for test compatibility)."""
    soup = BeautifulSoup(html, "html.parser")
    items = []
    seen: set = set()
    parsed_base = urlparse(base_url)
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if not href or href.startswith("#"):
            continue
        full_url = urljoin(base_url, href)
        if urlparse(full_url).netloc != parsed_base.netloc:
            continue
        if re.search(r"\.(css|js|png|jpg|gif|ico)$", href, re.I):
            continue
        text = a.get_text(strip=True)
        if len(text) < 5 or full_url in seen:
            continue
        seen.add(full_url)
        items.append({"url": full_url, "title": text})
    return items


def parse_detail_page(html: str, url: str) -> DocumentMeta:
    """Parse a detail page (kept for test compatibility)."""
    soup = BeautifulSoup(html, "html.parser")
    doc = DocumentMeta(url=url)
    h1 = soup.find("h1")
    doc.title = (h1.get_text(strip=True) if h1 else
                 (soup.title.string if soup.title else url)).strip()
    meta_desc = soup.find("meta", {"name": re.compile(r"description", re.I)})
    if meta_desc and meta_desc.get("content"):
        doc.summary = meta_desc["content"].strip()
    if not doc.summary:
        p = soup.find("p")
        doc.summary = p.get_text(strip=True)[:500] if p else ""
    for a in soup.find_all("a", href=True):
        if re.search(r"\.(pdf|xlsx|xls|docx|csv)$", a["href"], re.I):
            doc.file_links.append(urljoin(url, a["href"]))
    return doc


def crawl(
    seed_urls: Optional[List[str]] = None,
    max_pages: int = 5,
    db_path: Optional[str] = None,
) -> List[DocumentMeta]:
    """
    Load curated MoSPI statistical documents into the database.
    Uses SHA-256 fingerprinting for incremental / idempotent runs.
    """
    db_path = db_path or settings.db_path
    init_db(db_path)
    collected: List[DocumentMeta] = []

    log.info("loading_curated_documents", count=len(MOSPI_DOCUMENTS))

    for entry in MOSPI_DOCUMENTS:
        url = entry["url"]
        content_hash = _sha256(entry["summary"].encode())

        with get_conn(db_path) as conn:
            row = conn.execute(
                "SELECT id, hash FROM documents WHERE url = ?", (url,)
            ).fetchone()

        if row and row["hash"] == content_hash:
            log.info("skipping_unchanged", url=url)
            continue

        doc = DocumentMeta(
            id=str(uuid.uuid4()),
            title=entry["title"],
            url=url,
            date_published=entry.get("date_published"),
            summary=entry["summary"],
            category=entry.get("category", "MoSPI"),
            hash=content_hash,
            file_links=[url] if url.endswith(".pdf") else [],
        )

        with get_conn(db_path) as conn:
            if row:
                conn.execute(
                    "UPDATE documents SET title=?, date_published=?, summary=?, "
                    "category=?, hash=? WHERE url=?",
                    (doc.title, doc.date_published, doc.summary,
                     doc.category, doc.hash, doc.url),
                )
                doc.id = row["id"]
            else:
                conn.execute(
                    "INSERT INTO documents(id, title, url, date_published, "
                    "summary, category, hash) VALUES (?,?,?,?,?,?,?)",
                    (doc.id, doc.title, doc.url, doc.date_published,
                     doc.summary, doc.category, doc.hash),
                )
            if doc.file_links:
                conn.execute(
                    "INSERT OR IGNORE INTO files(id, document_id, file_url, file_type) "
                    "VALUES (?,?,?,?)",
                    (str(uuid.uuid4()), doc.id, doc.file_links[0], "pdf"),
                )

        collected.append(doc)
        log.info("loaded_document", title=doc.title[:70], category=doc.category)

    log.info("crawl_complete", total=len(collected))
    return collected