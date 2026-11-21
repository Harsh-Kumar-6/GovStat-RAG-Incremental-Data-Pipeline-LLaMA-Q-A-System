"""Tests for document validation and date normalization."""
from __future__ import annotations

import pytest

from pipeline.etl import validate_document
from scraper.crawl import _clean_date

VALID = {
    "title": "CPI March 2025",
    "url": "https://example.test/a.pdf",
    "date_published": "2025-04-12",
    "summary": "Something substantive.",
}


def test_valid_document_has_no_issues():
    assert validate_document(VALID) == []


@pytest.mark.parametrize(
    "field,value,expected",
    [
        ("title", "", "title_empty_or_too_short"),
        ("title", "ab", "title_empty_or_too_short"),
        ("url", None, "url_missing"),
    ],
)
def test_missing_fields_are_reported(field, value, expected):
    row = dict(VALID)
    row[field] = value
    assert expected in validate_document(row)


@pytest.mark.parametrize("value", ["", None])
def test_empty_summary_is_allowed(value):
    """Real MoSPI API documents have no summary; their content is PDF text.

    Requiring one would reject every crawled document, and is what pressured
    the old pipeline into fabricating summaries.
    """
    row = dict(VALID)
    row["summary"] = value
    assert validate_document(row) == []


def test_malformed_date_is_reported():
    row = dict(VALID)
    row["date_published"] = "12/04/2025"
    issues = validate_document(row)
    assert any(i.startswith("invalid_date") for i in issues)


def test_absent_date_is_allowed():
    row = dict(VALID)
    row["date_published"] = None
    assert validate_document(row) == []


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("12/04/2025", "2025-04-12"),
        ("12-04-2025", "2025-04-12"),
        ("2025-04-12", "2025-04-12"),
        ("12 April 2025", "2025-04-12"),
        ("April 12, 2025", "2025-04-12"),
        ("  12/04/2025  ", "2025-04-12"),
        ("Released on 12/04/2025 by MoSPI", "2025-04-12"),
    ],
)
def test_dates_normalize_to_iso(raw, expected):
    assert _clean_date(raw) == expected


@pytest.mark.parametrize("raw", ["", "not a date", "32/13/2025"])
def test_unparseable_dates_return_none(raw):
    assert _clean_date(raw) is None
