"""Turn extracted PDF tables into typed, queryable SQL tables.

Why this module exists
----------------------
`tables.table_json` is a raw staging blob: `{page, header, rows}` exactly as
pdfplumber found it. An LLM cannot write useful SQL against a JSON column, so
the series that matter are normalized into typed tables with real columns,
real types, and a foreign key back to the document they came from.

Design: long (tidy) format
--------------------------
`cpi_state` stores ONE ROW PER MEASUREMENT -- (release, state, sector) -- not
one row per state with six value columns. Long format means:

  * adding a sector or a metric never requires a schema migration,
  * `WHERE sector = 'Rural'` replaces picking between columns,
  * aggregates (`AVG`, `MAX`, `GROUP BY`) work without naming every column,

which is what makes it tractable for an LLM-written query.

Quirks of the real MoSPI tables that this handles
-------------------------------------------------
1. **Every table appears twice** -- once in English, once in Hindi, with
   identical numbers. The Hindi copy is detected by Devanagari in its header
   and skipped, so values are not double-counted.
2. **The header spans two rows.** pdfplumber returns
   `['S.No.','State','CPI','col_3','col_4','Inflation','col_6','col_7']`
   and the Rural/Urban/Combined labels arrive as the first *data* row. The
   metric group is forward-filled across columns and paired with that row.
3. **A roman-numeral column-index row** `(i) (ii) (iii)...` follows the
   sub-header and is dropped.
4. **State names contain embedded newlines** ("Andaman And Nicobar\\nIslands").
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from scraper.config import settings
from scraper.db import get_conn, init_db
from scraper.logger import get_logger

log = get_logger(__name__)

DDL = """
CREATE TABLE IF NOT EXISTS cpi_state (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    source_document_id TEXT NOT NULL REFERENCES documents(id),
    source_table_id    TEXT REFERENCES tables(id),
    release_date       TEXT,
    period             TEXT,
    state              TEXT NOT NULL,
    sector             TEXT NOT NULL,
    index_value        REAL,
    inflation_yoy      REAL,
    created_at         TEXT DEFAULT (datetime('now')),
    UNIQUE(source_document_id, state, sector)
);

CREATE INDEX IF NOT EXISTS idx_cpi_state_period ON cpi_state(period);
CREATE INDEX IF NOT EXISTS idx_cpi_state_state  ON cpi_state(state);
CREATE INDEX IF NOT EXISTS idx_cpi_state_sector ON cpi_state(sector);
CREATE INDEX IF NOT EXISTS idx_cpi_state_doc    ON cpi_state(source_document_id);
"""

DEVANAGARI = re.compile(r"[ऀ-ॿ]")
ROMAN_ROW = re.compile(r"^\(?[ivx]+\)?$", re.I)

MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12,
}

SECTOR_ALIASES = {
    "rural": "Rural",
    "urban": "Urban",
    "combined": "Combined",
    "combd.": "Combined",
    "combd": "Combined",
}


def _norm(value: Any) -> str:
    """Collapse newlines and repeated spaces in a cell."""
    return re.sub(r"\s+", " ", str(value or "")).strip()


def has_devanagari(text: str) -> bool:
    return bool(DEVANAGARI.search(text or ""))


def parse_period(title: str) -> Optional[str]:
    """Extract the reference month from a CPI press-note title as `YYYY-MM`.

    The release *date* is not the reference period: the September 2026 index is
    published in October 2026, so using `date_published` would mislabel every
    row by a month.
    """
    t = (title or "").lower()
    m = re.search(
        r"(january|february|march|april|may|june|july|august|september|october|"
        r"november|december)[,\s]+(\d{4})",
        t,
    )
    if m:
        return f"{int(m.group(2)):04d}-{MONTHS[m.group(1)]:02d}"
    return None


def _to_float(value: Any) -> Optional[float]:
    """Parse a numeric cell, tolerating commas, footnote marks, and dashes."""
    s = _norm(value)
    if not s:
        return None
    s = s.replace(",", "")
    s = re.sub(r"[*#@†‡]+$", "", s).strip()
    if s in {"-", "--", "—", "NA", "N.A.", "na", "nil"}:
        return None
    m = re.match(r"^-?\d+(?:\.\d+)?$", s)
    return float(s) if m else None


def is_cpi_state_table(header: List[str], rows: List[Dict]) -> bool:
    """True for the state-wise CPI statement, in English."""
    joined = " ".join(_norm(h) for h in header).lower()
    if has_devanagari(joined):
        return False  # the Hindi duplicate of the same table
    if "state" not in joined:
        return False
    if not ("cpi" in joined or "index" in joined):
        return False
    return "inflation" in joined and len(rows) >= 5


def _column_labels(header: List[str], subheader: Dict) -> List[Tuple[str, str]]:
    """Pair each column with its (metric_group, sector).

    The metric group is forward-filled because pdfplumber only reports it on
    the first column of each merged span.
    """
    labels: List[Tuple[str, str]] = []
    group = ""
    for key in header:
        h = _norm(key)
        if h and not h.lower().startswith("col_"):
            group = h
        sector = SECTOR_ALIASES.get(_norm(subheader.get(key)).lower(), "")
        labels.append((group, sector))
    return labels


def normalize_cpi_state_table(
    table: Dict, document_id: str, table_id: str, release_date: Optional[str],
    period: Optional[str],
) -> List[Dict]:
    """Convert one raw CPI state table into long-format measurement rows."""
    header: List[str] = table.get("header") or []
    rows: List[Dict] = table.get("rows") or []
    if not header or not rows:
        return []

    labels = _column_labels(header, rows[0])
    # Column 1 by convention is the state name; find it by header text instead
    # of position so a leading S.No. column being absent does not break it.
    state_key = next(
        (k for k in header if "state" in _norm(k).lower()), header[1] if len(header) > 1 else header[0]
    )

    out: List[Dict] = []
    seen: set[tuple[str, str]] = set()
    for row in rows[1:]:
        state = _norm(row.get(state_key))
        if not state or ROMAN_ROW.match(state):
            continue
        if state.lower() in {"state", "all india", "s.no.", ""}:
            # "All India" is a total, not a state; keep states only so that
            # SUM/AVG over this table is not silently double-counted.
            if state.lower() != "all india":
                continue

        # Collect index and inflation per sector for this state.
        per_sector: Dict[str, Dict[str, Optional[float]]] = {}
        for key, (group, sector) in zip(header, labels):
            if not sector:
                continue
            value = _to_float(row.get(key))
            slot = per_sector.setdefault(sector, {"index_value": None, "inflation_yoy": None})
            field = "inflation_yoy" if "inflation" in group.lower() else "index_value"
            if slot[field] is None:
                slot[field] = value

        for sector, vals in per_sector.items():
            if vals["index_value"] is None and vals["inflation_yoy"] is None:
                continue
            if (state, sector) in seen:
                continue
            seen.add((state, sector))
            out.append({
                "source_document_id": document_id,
                "source_table_id": table_id,
                "release_date": release_date,
                "period": period,
                "state": state,
                "sector": sector,
                "index_value": vals["index_value"],
                "inflation_yoy": vals["inflation_yoy"],
            })
    return out


def build_cpi_state(db_path: Optional[str] = None) -> Dict[str, int]:
    """Populate `cpi_state` from the staged CPI tables. Idempotent."""
    db_path = db_path or settings.db_path
    init_db(db_path)

    stats = {"documents": 0, "tables_considered": 0, "tables_used": 0, "rows": 0}

    with get_conn(db_path) as conn:
        conn.executescript(DDL)
        candidates = conn.execute(
            """SELECT t.id AS table_id, t.table_json, d.id AS document_id,
                      d.title, d.date_published
               FROM tables t
               JOIN documents d ON d.id = t.document_id
               WHERE d.category LIKE '%CPI%'
               ORDER BY d.date_published DESC""",
        ).fetchall()

        by_doc: Dict[str, List] = {}
        for row in candidates:
            by_doc.setdefault(row["document_id"], []).append(row)

        for document_id, rows in by_doc.items():
            title = rows[0]["title"]
            period = parse_period(title)
            release_date = rows[0]["date_published"]
            # Rebuild this document's rows so re-runs do not duplicate.
            conn.execute("DELETE FROM cpi_state WHERE source_document_id = ?", (document_id,))
            doc_rows: List[Dict] = []

            for row in rows:
                stats["tables_considered"] += 1
                try:
                    table = json.loads(row["table_json"])
                except (TypeError, ValueError):
                    continue
                if not is_cpi_state_table(table.get("header") or [], table.get("rows") or []):
                    continue
                norm = normalize_cpi_state_table(
                    table, document_id, row["table_id"], release_date, period
                )
                if norm:
                    stats["tables_used"] += 1
                    doc_rows.extend(norm)

            inserted = 0
            for r in doc_rows:
                cur = conn.execute(
                    "INSERT OR IGNORE INTO cpi_state"
                    "(source_document_id, source_table_id, release_date, period,"
                    " state, sector, index_value, inflation_yoy) "
                    "VALUES (:source_document_id,:source_table_id,:release_date,:period,"
                    ":state,:sector,:index_value,:inflation_yoy)",
                    r,
                )
                inserted += cur.rowcount or 0
            if inserted:
                stats["documents"] += 1
                stats["rows"] += inserted
            log.info("cpi_state_document", title=title[:60], period=period, rows=inserted)

    log.info("cpi_state_built", **stats)
    return stats


def main() -> None:
    import sys

    stats = build_cpi_state()
    json.dump(stats, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()
