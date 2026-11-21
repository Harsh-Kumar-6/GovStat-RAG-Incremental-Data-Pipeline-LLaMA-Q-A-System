"""Tests for PDF download guarding and text/table extraction."""
from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from scraper import parse as parse_mod
from scraper.parse import download_pdf, extract_text_and_tables, looks_like_pdf

HTML_ERROR_PAGE = b"<!DOCTYPE html><html><head><title>MoSPI</title></head><body>Not found</body></html>"


def test_looks_like_pdf_accepts_pdf_magic(minimal_pdf_bytes):
    assert looks_like_pdf(minimal_pdf_bytes)


@pytest.mark.parametrize("content", [HTML_ERROR_PAGE, b"", b"junk"])
def test_looks_like_pdf_rejects_non_pdf(content):
    assert not looks_like_pdf(content)


def _patch_client(monkeypatch, content: bytes, content_type: str, status: int = 200):
    """Replace httpx.Client with a stub returning a fixed response."""

    class StubClient:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, url):
            return httpx.Response(
                status_code=status,
                content=content,
                headers={"content-type": content_type},
                request=httpx.Request("GET", url),
            )

    monkeypatch.setattr(parse_mod.httpx, "Client", StubClient)


def test_html_served_with_200_is_not_saved_as_pdf(monkeypatch, tmp_path):
    """MoSPI returns 200 + HTML for dead document links.

    raise_for_status() passes, so without a magic-byte check the HTML would be
    written to disk as a .pdf and silently yield zero text.
    """
    _patch_client(monkeypatch, HTML_ERROR_PAGE, "text/html; charset=utf-8")
    result = download_pdf("https://example.test/missing.pdf", tmp_path)
    assert result is None
    assert list(tmp_path.glob("*.pdf")) == []


def test_real_pdf_is_saved(monkeypatch, tmp_path, minimal_pdf_bytes):
    _patch_client(monkeypatch, minimal_pdf_bytes, "application/pdf")
    result = download_pdf("https://example.test/real.pdf", tmp_path)
    assert result is not None and result.exists()
    assert result.read_bytes()[:5] == b"%PDF-"


def test_download_is_idempotent(monkeypatch, tmp_path, minimal_pdf_bytes):
    _patch_client(monkeypatch, minimal_pdf_bytes, "application/pdf")
    url = "https://example.test/real.pdf"
    first = download_pdf(url, tmp_path)
    calls = {"n": 0}

    class ExplodingClient:
        def __init__(self, *a, **k):
            calls["n"] += 1
            raise AssertionError("should not re-download an existing file")

    monkeypatch.setattr(parse_mod.httpx, "Client", ExplodingClient)
    second = download_pdf(url, tmp_path)
    assert first == second
    assert calls["n"] == 0


def test_http_error_returns_none(monkeypatch, tmp_path):
    _patch_client(monkeypatch, b"nope", "text/html", status=404)
    assert download_pdf("https://example.test/x.pdf", tmp_path) is None


def test_extract_from_corrupt_file_returns_empty_without_raising(tmp_path):
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"%PDF-1.4 truncated garbage")
    text, tables, pages = extract_text_and_tables(bad)
    assert text == "" and tables == [] and pages == 0


def test_extract_returns_text_and_page_count(tmp_path, minimal_pdf_bytes):
    pdf = tmp_path / "ok.pdf"
    pdf.write_bytes(minimal_pdf_bytes)
    text, tables, pages = extract_text_and_tables(pdf)
    if pages == 0:
        pytest.skip("pdfplumber could not open the synthetic fixture PDF")
    assert pages == 1
    assert "CPI" in text


# --------------------------------------------------------------------------
# Size and cost guards
# --------------------------------------------------------------------------

def test_oversized_pdf_is_skipped(monkeypatch, tmp_path, minimal_pdf_bytes):
    """A 200-page yearbook once burned 9+ min of CPU; size is capped first."""
    big = minimal_pdf_bytes + b"\0" * (3 * 1024 * 1024)
    _patch_client(monkeypatch, big, "application/pdf")
    monkeypatch.setattr(parse_mod.settings, "max_pdf_mb", 1.0)
    assert download_pdf("https://example.test/big.pdf", tmp_path) is None
    assert list(tmp_path.glob("*.pdf")) == []


def test_pdf_within_the_size_limit_is_kept(monkeypatch, tmp_path, minimal_pdf_bytes):
    _patch_client(monkeypatch, minimal_pdf_bytes, "application/pdf")
    monkeypatch.setattr(parse_mod.settings, "max_pdf_mb", 20.0)
    assert download_pdf("https://example.test/small.pdf", tmp_path) is not None


def test_table_extraction_is_limited_by_page_count(tmp_path, minimal_pdf_bytes):
    """max_table_pages must not suppress text extraction, only tables."""
    pdf = tmp_path / "ok.pdf"
    pdf.write_bytes(minimal_pdf_bytes)
    text, tables, pages = extract_text_and_tables(pdf, max_table_pages=0)
    if pages == 0:
        pytest.skip("pdfplumber could not open the synthetic fixture PDF")
    assert tables == []
    assert "CPI" in text, "text extraction must still cover every page"
