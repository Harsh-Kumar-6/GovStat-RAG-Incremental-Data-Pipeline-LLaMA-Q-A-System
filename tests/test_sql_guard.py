"""The guard is the security boundary for LLM-written SQL, so it gets the
nastiest tests in the suite. A guard that only blocks a bare DROP is theatre.
"""
from __future__ import annotations

import pytest

from agents.sql_guard import UnsafeSQL, guard_sql, strip_comments


# ---------------------------------------------------------------------------
# Legitimate queries must survive
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM cpi_state",
        "select state, inflation_yoy from cpi_state where sector='Rural'",
        "SELECT state FROM cpi_state ORDER BY inflation_yoy DESC LIMIT 5",
        "WITH t AS (SELECT * FROM cpi_state) SELECT state FROM t",
        "SELECT * FROM cpi_state WHERE state = 'Jammu; Kashmir'",   # quoted ;
        "SELECT COUNT(*) FROM documents WHERE title LIKE '%CPI%'",
        "SELECT created_at FROM documents",                          # not CREATE
    ],
)
def test_valid_read_queries_are_allowed(sql):
    assert guard_sql(sql).sql


def test_a_limit_is_added_when_missing():
    r = guard_sql("SELECT * FROM cpi_state", max_rows=50)
    assert r.limit_added is True and r.sql.endswith("LIMIT 50")


def test_an_existing_limit_is_respected():
    r = guard_sql("SELECT * FROM cpi_state LIMIT 3")
    assert r.limit_added is False and r.sql.count("LIMIT") == 1


def test_trailing_semicolon_is_fine():
    assert guard_sql("SELECT 1;").sql.startswith("SELECT 1")


# ---------------------------------------------------------------------------
# Writes, in every disguise
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE documents",
        "DELETE FROM cpi_state",
        "UPDATE cpi_state SET inflation_yoy = 0",
        "INSERT INTO cpi_state (state) VALUES ('x')",
        "ALTER TABLE documents ADD COLUMN x TEXT",
        "CREATE TABLE evil (a TEXT)",
    ],
    ids=["drop", "delete", "update", "insert", "alter", "create"],
)
def test_plain_writes_are_blocked(sql):
    with pytest.raises(UnsafeSQL):
        guard_sql(sql)


def test_a_write_hidden_behind_a_cte_is_blocked():
    """SQLite allows WITH ... INSERT, so 'starts with WITH' is not enough."""
    with pytest.raises(UnsafeSQL, match="INSERT"):
        guard_sql("WITH t AS (SELECT 1) INSERT INTO cpi_state SELECT * FROM t")


def test_a_second_statement_after_a_semicolon_is_blocked():
    with pytest.raises(UnsafeSQL, match="one statement"):
        guard_sql("SELECT 1; DROP TABLE documents")


def test_a_statement_smuggled_past_a_line_comment_is_blocked():
    with pytest.raises(UnsafeSQL):
        guard_sql("SELECT 1 -- harmless\n; DROP TABLE documents")


def test_a_statement_smuggled_inside_a_block_comment_is_blocked():
    with pytest.raises(UnsafeSQL):
        guard_sql("SELECT 1 /* ; still a comment */ ; DELETE FROM documents")


def test_a_query_that_is_only_a_comment_is_blocked():
    with pytest.raises(UnsafeSQL, match="only comments"):
        guard_sql("-- SELECT 1")


# ---------------------------------------------------------------------------
# The ones a denylist of DROP|DELETE|UPDATE would miss
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "sql",
    [
        "ATTACH DATABASE '/tmp/evil.db' AS evil",
        "PRAGMA table_info(documents)",
        "VACUUM",
        "BEGIN; SELECT 1",
        "REINDEX documents",
        "DETACH DATABASE main",
    ],
    ids=["attach", "pragma", "vacuum", "begin", "reindex", "detach"],
)
def test_the_non_obvious_dangerous_statements_are_blocked(sql):
    """This is the argument for an allowlist: none of these are DROP."""
    with pytest.raises(UnsafeSQL):
        guard_sql(sql)


def test_empty_and_whitespace_are_rejected():
    for sql in ("", "   ", "\n\t"):
        with pytest.raises(UnsafeSQL, match="empty"):
            guard_sql(sql)


def test_explain_is_not_allowed():
    """Not dangerous, but not in the allowlist: fails closed, by design."""
    with pytest.raises(UnsafeSQL):
        guard_sql("EXPLAIN QUERY PLAN SELECT * FROM documents")


# ---------------------------------------------------------------------------
# Comment stripping
# ---------------------------------------------------------------------------
def test_strip_comments_removes_both_styles():
    out = strip_comments("SELECT 1 -- a\n/* b */ , 2")
    assert "--" not in out and "/*" not in out and "1" in out and "2" in out


# ---------------------------------------------------------------------------
# The rejection message feeds the ReAct loop
# ---------------------------------------------------------------------------
def test_the_rejection_says_what_was_wrong():
    """The agent retries using this text as its observation, so a bare
    'rejected' would give it nothing to act on."""
    # The opener allowlist fires first, so the message names the verb it saw.
    with pytest.raises(UnsafeSQL) as exc:
        guard_sql("DELETE FROM cpi_state")
    msg = str(exc.value)
    assert "DELETE" in msg and "allowed" in msg

    # A write behind a CTE gets past the opener check and hits the keyword
    # check, which names the keyword instead.
    with pytest.raises(UnsafeSQL) as exc2:
        guard_sql("WITH t AS (SELECT 1) DELETE FROM cpi_state")
    assert "DELETE" in str(exc2.value) and "not allowed" in str(exc2.value)
