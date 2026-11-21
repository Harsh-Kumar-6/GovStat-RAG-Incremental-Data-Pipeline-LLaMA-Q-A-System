"""The schema description handed to the SQL agent.

Why this is written by hand and not generated
---------------------------------------------
`PRAGMA table_info` gives names and types. It does not say that `period` is
the month the statistics *describe* while `release_date` is when they were
published -- and that distinction is the single most likely cause of a
confidently wrong answer in this dataset. An LLM cannot infer semantics from
a column name, so the semantics have to be stated.

The live part (which tables exist, how many rows, what values a column
actually holds) is introspected, because a hand-written claim about the data
goes stale the moment the data changes. Hand-write meaning, introspect facts.
"""
from __future__ import annotations

from typing import List, Optional

from scraper.config import settings
from scraper.db import get_conn

# Only these tables are exposed. documents/files/doc_texts are joinable for
# citations; chunks and tables are not -- chunk text belongs to the RAG agent,
# and raw table_json is unqueryable JSON blobs.
EXPOSED_TABLES = ("cpi_state", "documents")

SEMANTICS = """
TABLE cpi_state -- Consumer Price Index by state, one row per state x sector
  period        TEXT  'YYYY-MM'. The month the index DESCRIBES, parsed from the
                      press-release title. THIS is what a question about
                      "September inflation" means.
  release_date  TEXT  'YYYY-MM-DD'. When MoSPI published it, typically the
                      middle of the FOLLOWING month. Never use this to answer a
                      question about a statistical month: the September index
                      is published in October, so filtering release_date for
                      September silently returns August's numbers.
  state         TEXT  State or union territory name, as printed in the release.
  sector        TEXT  Exactly one of 'Rural', 'Urban', 'Combined'.
                      'Combined' is rural+urban together -- it is NOT a total to
                      be summed, and the three sectors must never be averaged.
  index_value   REAL  The CPI index level (e.g. 194.2). Not a percentage.
  inflation_yoy REAL  Year-on-year inflation in PERCENT (e.g. 7.31 means 7.31%).
                      Already a percentage: do not multiply by 100.
  source_document_id TEXT  -> documents.id. Join to cite the source.
  source_table_id    TEXT  -> the extracted table the row came from.

TABLE documents -- one row per MoSPI publication
  id             TEXT  Primary key.
  title          TEXT  Release title, e.g. 'Press release of CPI for the month
                       of October 2026'. The statistical month is in here.
  url            TEXT  Direct link to the source PDF. Use for citations.
  date_published TEXT  'YYYY-MM-DD' publication date.
  category       TEXT  Derived from the title, e.g. 'Price Statistics > CPI'.
  summary        TEXT  Always empty. The API provides none; the extracted PDF
                       text lives elsewhere and is the RAG agent's job.

RULES
  - SELECT only, one statement. No writes of any kind.
  - Always filter sector explicitly; omitting it mixes Rural, Urban and
    Combined rows into one result set, which looks like data and is nonsense.
  - To cite a figure, join cpi_state.source_document_id = documents.id and
    select documents.title and documents.url.
  - If the question asks for a month or a series that is not in the data,
    return a query that will come back empty rather than substituting a
    different period. An empty result is a correct answer to an unanswerable
    question.
"""


def _column_names(conn, table: str) -> List[str]:
    return [r[1] for r in conn.execute(f"PRAGMA table_info({table})")]


def build_schema_doc(db_path: Optional[str] = None,
                     sample_values: bool = True) -> str:
    """Hand-written semantics plus live facts about what is really in the DB."""
    db_path = db_path or settings.db_path
    parts = [SEMANTICS.strip(), "", "LIVE DATA (introspected, not assumed):"]

    with get_conn(db_path, read_only=True) as conn:
        present = {
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        for table in EXPOSED_TABLES:
            if table not in present:
                parts.append(f"  {table}: NOT PRESENT -- do not query it")
                continue
            n = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            parts.append(f"  {table}: {n} rows")
            if not sample_values or n == 0:
                continue
            cols = _column_names(conn, table)
            # Enumerate the small-cardinality columns in full. Guessing that a
            # sector is called 'rural' when the data says 'Rural' produces an
            # empty result set that looks like "no data" instead of a typo.
            for col in ("period", "sector", "category"):
                if col not in cols:
                    continue
                vals = [
                    str(r[0]) for r in conn.execute(
                        f"SELECT DISTINCT {col} FROM {table} "
                        f"WHERE {col} IS NOT NULL ORDER BY {col} LIMIT 25"
                    )
                ]
                if vals:
                    parts.append(f"    {col} values: {', '.join(vals)}")
            if "state" in cols:
                n_states = conn.execute(
                    f"SELECT COUNT(DISTINCT state) FROM {table}"
                ).fetchone()[0]
                sample = [
                    r[0] for r in conn.execute(
                        f"SELECT DISTINCT state FROM {table} ORDER BY state LIMIT 6"
                    )
                ]
                parts.append(
                    f"    state: {n_states} distinct, e.g. {', '.join(sample)}"
                )

    return "\n".join(parts)
