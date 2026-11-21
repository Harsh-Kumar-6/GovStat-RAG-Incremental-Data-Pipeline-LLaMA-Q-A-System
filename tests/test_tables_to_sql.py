"""Tests for normalizing staged PDF tables into the typed cpi_state table."""
from __future__ import annotations

import json
import uuid

import pytest

from pipeline.tables_to_sql import (
    _to_float,
    build_cpi_state,
    has_devanagari,
    is_cpi_state_table,
    normalize_cpi_state_table,
    parse_period,
)
from scraper.db import get_conn

# A faithful miniature of the real MoSPI state-wise CPI statement: a two-row
# header, a roman-numeral index row, and newline-split state names.
EN_HEADER = ["S.No.", "State", "CPI", "col_3", "col_4", "Inflation", "col_6", "col_7"]
EN_ROWS = [
    {"S.No.": "", "State": "", "CPI": "Rural", "col_3": "Urban", "col_4": "Combined",
     "Inflation": "Rural", "col_6": "Urban", "col_7": "Combd."},
    {"S.No.": "(i)", "State": "(ii)", "CPI": "(iii)", "col_3": "(iv)", "col_4": "(v)",
     "Inflation": "(vi)", "col_6": "(vii)", "col_7": "(viii)"},
    {"S.No.": "1.", "State": "Andaman And Nicobar\nIslands", "CPI": "111.31",
     "col_3": "108.23", "col_4": "109.80", "Inflation": "5.01", "col_6": "3.56", "col_7": "4.30"},
    {"S.No.": "2.", "State": "Andhra Pradesh", "CPI": "108.82", "col_3": "108.65",
     "col_4": "108.75", "Inflation": "6.63", "col_6": "5.74", "col_7": "6.29"},
    {"S.No.": "3.", "State": "Delhi", "CPI": "-", "col_3": "107.10",
     "col_4": "107.10", "Inflation": "-", "col_6": "4.11", "col_7": "4.11"},
]
HI_HEADER = ["क्रम सां.", "राज्य", "सीपीआई", "col_3", "col_4", "मुद्रास्फीसत", "col_6", "col_7"]

EN_TABLE = {"page": 9, "header": EN_HEADER, "rows": EN_ROWS}
HI_TABLE = {"page": 9, "header": HI_HEADER, "rows": EN_ROWS}


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("111.31", 111.31), ("1,234.5", 1234.5), ("5.01*", 5.01),
    ("-", None), ("--", None), ("NA", None), ("", None),
    ("-2.4", -2.4), ("abc", None), (None, None),
])
def test_to_float(raw, expected):
    assert _to_float(raw) == expected


def test_has_devanagari():
    assert has_devanagari("राज्य")
    assert not has_devanagari("State")


@pytest.mark.parametrize("title,expected", [
    ("Press release of CPI for the month of September 2026", "2026-09"),
    ("Press release of CPI for the month of October 2026", "2026-10"),
    ("Consumer Price Index Numbers for January, 2025", "2025-01"),
    ("CPI press note", None),
])
def test_parse_period(title, expected):
    assert parse_period(title) == expected


def test_parse_period_uses_reference_month_not_release_month():
    """The September index is published in October; the period must say 09."""
    assert parse_period("Press release of CPI for the month of September 2026") == "2026-09"


# --------------------------------------------------------------------------
# Table detection
# --------------------------------------------------------------------------

def test_detects_the_english_state_table():
    assert is_cpi_state_table(EN_HEADER, EN_ROWS)


def test_rejects_the_hindi_duplicate():
    """Every MoSPI table appears twice; counting both would double the rows."""
    assert not is_cpi_state_table(HI_HEADER, EN_ROWS)


def test_rejects_unrelated_tables():
    assert not is_cpi_state_table(["Group code", "Group name", "Weights"], EN_ROWS)
    assert not is_cpi_state_table(EN_HEADER, EN_ROWS[:2])  # too few rows


# --------------------------------------------------------------------------
# Normalization
# --------------------------------------------------------------------------

def _norm_rows():
    return normalize_cpi_state_table(EN_TABLE, "doc1", "tbl1", "2026-10-12", "2026-09")


def test_normalization_is_long_format():
    rows = _norm_rows()
    # 3 states x 3 sectors, minus Delhi/Rural which has no values at all.
    assert len(rows) == 8
    assert {r["sector"] for r in rows} == {"Rural", "Urban", "Combined"}


def test_normalization_maps_values_to_the_right_sector():
    rows = {(r["state"], r["sector"]): r for r in _norm_rows()}
    a = rows[("Andaman And Nicobar Islands", "Rural")]
    assert a["index_value"] == 111.31 and a["inflation_yoy"] == 5.01
    u = rows[("Andaman And Nicobar Islands", "Urban")]
    assert u["index_value"] == 108.23 and u["inflation_yoy"] == 3.56
    c = rows[("Andaman And Nicobar Islands", "Combined")]
    assert c["index_value"] == 109.80 and c["inflation_yoy"] == 4.30


def test_newlines_in_state_names_are_collapsed():
    assert any(r["state"] == "Andaman And Nicobar Islands" for r in _norm_rows())


def test_roman_numeral_and_subheader_rows_are_dropped():
    states = {r["state"] for r in _norm_rows()}
    assert not any(s.startswith("(") for s in states)
    assert "Rural" not in states


def test_rows_with_no_values_are_skipped():
    """Delhi has no rural CPI; that pair must be absent, not stored as null."""
    pairs = {(r["state"], r["sector"]) for r in _norm_rows()}
    assert ("Delhi", "Rural") not in pairs
    assert ("Delhi", "Urban") in pairs


def test_citation_fields_are_carried_through():
    r = _norm_rows()[0]
    assert r["source_document_id"] == "doc1"
    assert r["source_table_id"] == "tbl1"
    assert r["period"] == "2026-09"
    assert r["release_date"] == "2026-10-12"


def test_empty_table_yields_nothing():
    assert normalize_cpi_state_table({"header": [], "rows": []}, "d", "t", None, None) == []


# --------------------------------------------------------------------------
# build_cpi_state against a real database
# --------------------------------------------------------------------------

@pytest.fixture()
def cpi_db(db_path):
    doc_id, file_id = str(uuid.uuid4()), str(uuid.uuid4())
    with get_conn(db_path) as conn:
        conn.execute(
            "INSERT INTO documents(id,title,url,date_published,summary,category,hash) "
            "VALUES (?,?,?,?,?,?,?)",
            (doc_id, "Press release of CPI for the month of September 2026",
             "https://example.test/cpi.pdf", "2026-10-12", "",
             "Price Statistics > CPI", "h1"),
        )
        conn.execute(
            "INSERT INTO files(id,document_id,file_url,file_type) VALUES (?,?,?,?)",
            (file_id, doc_id, "https://example.test/cpi.pdf", "pdf"),
        )
        for tid, table in (("t_en", EN_TABLE), ("t_hi", HI_TABLE)):
            conn.execute(
                "INSERT INTO tables(id,document_id,source_file_id,table_json,page,n_rows,n_cols) "
                "VALUES (?,?,?,?,?,?,?)",
                (tid, doc_id, file_id, json.dumps(table), 9, len(EN_ROWS), len(EN_HEADER)),
            )
    return {"db_path": db_path, "doc_id": doc_id}


def test_build_populates_typed_table(cpi_db):
    stats = build_cpi_state(cpi_db["db_path"])
    assert stats["rows"] == 8
    assert stats["tables_used"] == 1  # the Hindi copy was skipped
    with get_conn(cpi_db["db_path"]) as conn:
        row = conn.execute(
            "SELECT index_value, inflation_yoy FROM cpi_state "
            "WHERE state='Andhra Pradesh' AND sector='Combined'"
        ).fetchone()
    assert row["index_value"] == 108.75 and row["inflation_yoy"] == 6.29


def test_build_is_idempotent(cpi_db):
    first = build_cpi_state(cpi_db["db_path"])
    second = build_cpi_state(cpi_db["db_path"])
    assert first["rows"] == second["rows"]
    with get_conn(cpi_db["db_path"]) as conn:
        assert conn.execute("SELECT COUNT(*) FROM cpi_state").fetchone()[0] == 8


def test_typed_table_is_queryable_and_citable(cpi_db):
    """The point of the whole module: real SQL, joined back to its source."""
    build_cpi_state(cpi_db["db_path"])
    with get_conn(cpi_db["db_path"], read_only=True) as conn:
        row = conn.execute(
            "SELECT cs.state, cs.inflation_yoy, d.url FROM cpi_state cs "
            "JOIN documents d ON d.id = cs.source_document_id "
            "WHERE cs.sector='Rural' ORDER BY cs.inflation_yoy DESC LIMIT 1"
        ).fetchone()
    assert row["state"] == "Andhra Pradesh"
    assert row["inflation_yoy"] == 6.63
    assert row["url"].endswith(".pdf")
