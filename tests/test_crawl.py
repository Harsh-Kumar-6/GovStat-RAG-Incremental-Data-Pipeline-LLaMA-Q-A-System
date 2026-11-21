"""Tests for the MoSPI JSON-API crawler."""
from __future__ import annotations

import json

import httpx
import pytest

from scraper import crawl as crawl_mod
from scraper.crawl import (
    ENDPOINTS,
    classify_category,
    crawl,
    extract_file_urls,
    fetch_listing_page,
    iter_documents,
)
from scraper.db import get_conn


def _item(i: int, title: str, path: str | None = "uploads/latestReleases/f%d.pdf") -> dict:
    return {
        "id": str(i),
        "title": title,
        "published_year": "2026-11-16",
        "start_date": "2026-11-16T00:00:00.000Z",
        "file_one": None
        if path is None
        else {
            "path": path % i if "%d" in path else path,
            "filemime": "application/pdf",
            "filename": f"f{i}.pdf",
            "filesize": 1234,
        },
        "file_two": None,
        "file_three": None,
        "is_active": True,
    }


def _stub_api(monkeypatch, pages: dict[int, list[dict]], total_pages: int | None = None):
    """Patch httpx.Client so POSTs return canned listing pages."""
    total = total_pages if total_pages is not None else len(pages)
    calls: list[dict] = []

    class StubClient:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, url, json=None):
            calls.append({"url": url, "body": json})
            page = (json or {}).get("page_no", 1)
            body = {
                "status": "success",
                "code": 200,
                "data": pages.get(page, []),
                "pagination": {
                    "currentPage": page,
                    "totalItems": sum(len(v) for v in pages.values()),
                    "totalPages": total,
                    "pageSize": (json or {}).get("page_size"),
                },
            }
            return httpx.Response(
                200, json=body, request=httpx.Request("POST", url)
            )

    monkeypatch.setattr(crawl_mod.httpx, "Client", StubClient)
    monkeypatch.setattr(crawl_mod.settings, "rate_limit_delay", 0)
    monkeypatch.setattr(crawl_mod.settings, "sources_str", "latest-release")
    return calls


# --------------------------------------------------------------------------
# Category classification
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "title,expected",
    [
        ("Press Note on CPI for October 2026", "Price Statistics > CPI"),
        ("Consumer Price Index Numbers", "Price Statistics > CPI"),
        ("Wholesale Price Index December", "Price Statistics > WPI"),
        ("Press Note on IIP September 2026", "Industrial Statistics > IIP"),
        ("Index of Industrial Production", "Industrial Statistics > IIP"),
        ("Second Advance Estimates of GDP", "National Accounts > GDP"),
        ("Press Note on Periodic Labour Force Survey (PLFS)", "Labour Statistics > PLFS"),
        ("Household Consumption Expenditure Survey", "Household Surveys > HCES"),
        ("Annual Survey of Industries 2023-24", "Industrial Statistics > ASI"),
        ("SDG National Indicator Framework Progress Report", "Publications > SDG"),
        ("Statistical Year Book India", "Publications > Statistical Year Book"),
        ("Press Release on something unrelated", "Press Release"),
        ("Minutes of the 42nd meeting", "Other"),
    ],
)
def test_classify_category(title, expected):
    assert classify_category(title) == expected


def test_classify_category_is_case_insensitive():
    assert classify_category("press note on cpi") == "Price Statistics > CPI"


# --------------------------------------------------------------------------
# File URL extraction
# --------------------------------------------------------------------------

def test_extract_file_urls_builds_absolute_urls():
    urls = extract_file_urls(_item(1, "CPI"))
    assert urls == ["https://www.mospi.gov.in/uploads/latestReleases/f1.pdf"]


def test_extract_file_urls_handles_leading_slash():
    item = _item(1, "CPI", path="/uploads/x.pdf")
    assert extract_file_urls(item) == ["https://www.mospi.gov.in/uploads/x.pdf"]


def test_extract_file_urls_collects_all_three_attachments():
    item = _item(1, "CPI")
    item["file_two"] = {"path": "uploads/b.pdf", "filemime": "application/pdf"}
    item["file_three"] = {"path": "uploads/c.pdf", "filemime": "application/pdf"}
    assert len(extract_file_urls(item)) == 3


def test_extract_file_urls_skips_nulls_and_non_pdfs():
    item = _item(1, "CPI", path=None)
    item["file_two"] = {"path": "uploads/sheet.xlsx", "filemime": "application/vnd.ms-excel"}
    assert extract_file_urls(item) == []


# --------------------------------------------------------------------------
# Listing fetch
# --------------------------------------------------------------------------

def test_fetch_listing_page_posts_expected_body(monkeypatch):
    calls = _stub_api(monkeypatch, {1: [_item(1, "CPI")]})
    with crawl_mod.httpx.Client() as client:
        items, total = fetch_listing_page("latest-release", 1, client, page_size=50)
    assert len(items) == 1 and total == 1
    body = calls[0]["body"]
    assert body["page_no"] == 1 and body["page_size"] == 50
    assert body["sort_field"] == "published_year" and body["data_source"] == "web"
    assert calls[0]["url"].endswith(ENDPOINTS["latest-release"])


def test_fetch_listing_page_rejects_unknown_source():
    with pytest.raises(ValueError):
        fetch_listing_page("not-a-source", 1, client=None)


# --------------------------------------------------------------------------
# Document iteration
# --------------------------------------------------------------------------

def test_iter_documents_maps_fields(monkeypatch):
    _stub_api(monkeypatch, {1: [_item(1, "Press Note on CPI  for\r\n October 2026")]})
    docs = list(iter_documents(max_pages=1))
    assert len(docs) == 1
    d = docs[0]
    assert d.title == "Press Note on CPI for October 2026"  # whitespace collapsed
    assert d.date_published == "2026-11-16"
    assert d.category == "Price Statistics > CPI"
    assert d.url == "https://www.mospi.gov.in/uploads/latestReleases/f1.pdf"
    assert d.hash


def test_iter_documents_leaves_summary_empty(monkeypatch):
    """The crawler must never invent summary prose -- that corrupted the old data."""
    _stub_api(monkeypatch, {1: [_item(1, "Press Note on CPI")]})
    assert list(iter_documents(max_pages=1))[0].summary == ""


def test_iter_documents_skips_items_without_a_pdf(monkeypatch):
    _stub_api(monkeypatch, {1: [_item(1, "No file here", path=None), _item(2, "CPI")]})
    docs = list(iter_documents(max_pages=1))
    assert [d.title for d in docs] == ["CPI"]


def test_max_pages_is_honoured(monkeypatch):
    """MAX_PAGES_PER_RUN was ignored entirely by the previous implementation."""
    pages = {n: [_item(n, f"Press Note {n}")] for n in range(1, 6)}
    _stub_api(monkeypatch, pages, total_pages=5)
    assert len(list(iter_documents(max_pages=2))) == 2
    assert len(list(iter_documents(max_pages=5))) == 5


def test_pagination_stops_at_total_pages(monkeypatch):
    _stub_api(monkeypatch, {1: [_item(1, "A")]}, total_pages=1)
    assert len(list(iter_documents(max_pages=99))) == 1


def test_listing_failure_is_contained(monkeypatch):
    class BoomClient:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def post(self, url, json=None): raise httpx.ConnectError("offline")

    monkeypatch.setattr(crawl_mod.httpx, "Client", BoomClient)
    monkeypatch.setattr(crawl_mod.settings, "sources_str", "latest-release")
    assert list(iter_documents(max_pages=1)) == []


# --------------------------------------------------------------------------
# crawl() persistence and idempotency
# --------------------------------------------------------------------------

def test_crawl_inserts_documents_and_files(monkeypatch, db_path):
    _stub_api(monkeypatch, {1: [_item(1, "Press Note on CPI"), _item(2, "Press Note on IIP")]})
    crawl(max_pages=1, db_path=db_path)
    with get_conn(db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 2
        assert conn.execute("SELECT COUNT(*) FROM files").fetchone()[0] == 2
        cats = {r[0] for r in conn.execute("SELECT category FROM documents")}
    assert cats == {"Price Statistics > CPI", "Industrial Statistics > IIP"}


def test_crawl_is_idempotent(monkeypatch, db_path):
    _stub_api(monkeypatch, {1: [_item(1, "Press Note on CPI")]})
    crawl(max_pages=1, db_path=db_path)
    second = crawl(max_pages=1, db_path=db_path)
    assert second == []  # unchanged hash -> skipped
    with get_conn(db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM files").fetchone()[0] == 1


def test_crawl_updates_a_changed_document(monkeypatch, db_path):
    _stub_api(monkeypatch, {1: [_item(1, "Press Note on CPI")]})
    crawl(max_pages=1, db_path=db_path)
    # Same URL, new title -> new hash -> update in place, not a duplicate row.
    _stub_api(monkeypatch, {1: [_item(1, "Press Note on CPI (revised)")]})
    crawl(max_pages=1, db_path=db_path)
    with get_conn(db_path) as conn:
        rows = conn.execute("SELECT title FROM documents").fetchall()
    assert len(rows) == 1 and "revised" in rows[0][0]


def test_crawl_ignores_seed_urls_without_failing(monkeypatch, db_path):
    _stub_api(monkeypatch, {1: [_item(1, "Press Note on CPI")]})
    crawl(seed_urls=["https://mospi.gov.in/web/mospi/press-releases"],
          max_pages=1, db_path=db_path)
    with get_conn(db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 1


def test_no_fabricated_document_list_remains():
    """Guard against the hardcoded MOSPI_DOCUMENTS list coming back."""
    assert not hasattr(crawl_mod, "MOSPI_DOCUMENTS")


# --------------------------------------------------------------------------
# Title normalization
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw,expected",
    [
        ("<p>ASI Summary Results 2024-2025</p>", "ASI Summary Results 2024-2025"),
        ("<p>Report of 4th ISI review committee&nbsp;</p>", "Report of 4th ISI review committee"),
        ('<p><span class="x_110font">Monthly Flash Report</span></p>', "Monthly Flash Report"),
        ("Press Note on CPI\r\nOctober 2026", "Press Note on CPI October 2026"),
        ("A &amp; B", "A & B"),
        ("Line one<br/>Line two", "Line one Line two"),
        ("  padded  title  ", "padded title"),
    ],
)
def test_clean_title_strips_html_and_whitespace(raw, expected):
    assert crawl_mod._clean_title(raw) == expected


def test_iter_documents_cleans_html_titles(monkeypatch):
    _stub_api(monkeypatch, {1: [_item(1, "<p>ASI Summary Results 2024-2025</p>")]})
    doc = list(iter_documents(max_pages=1))[0]
    assert doc.title == "ASI Summary Results 2024-2025"
    assert "<" not in doc.title


def test_category_is_classified_after_html_stripping(monkeypatch):
    _stub_api(monkeypatch, {1: [_item(1, "<p>Press Note on <b>CPI</b> October 2026</p>")]})
    assert list(iter_documents(max_pages=1))[0].category == "Price Statistics > CPI"
