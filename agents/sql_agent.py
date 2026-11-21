"""SQL agent: a ReAct loop over a read-only SQLite connection.

What ReAct is, concretely
-------------------------
Reason -> Act -> Observe, repeated. Here:

  Reason   the model reads the schema and the question and decides on a query
  Act      the guard checks it, then it runs
  Observe  the rows come back -- or an error does

The observation is the interesting half. A plain "generate SQL" call gets one
shot and fails on a column it misremembered. A ReAct loop feeds the database's
own error message back in ("no such column: inflation") and lets the model
correct itself. The error text is not noise to be hidden; it is the signal the
loop runs on.

Bounded on purpose: three attempts. An unbounded loop against a rate-limited
API is how you spend a daily quota on one malformed question.
"""
from __future__ import annotations

import sqlite3
import time
from typing import Any, Dict, List, Optional, Tuple

from agents.llm_provider import LLMError, complete_json
from agents.schema_doc import build_schema_doc
from agents.sql_guard import UnsafeSQL, guard_sql
from scraper.config import settings
from scraper.db import get_conn
from scraper.logger import get_logger

log = get_logger(__name__)

MAX_ATTEMPTS = 3
MAX_ROWS = 200
TIMEOUT_S = 10.0

SYSTEM = (
    "You write SQLite queries against a database of Indian government "
    "statistics published by MoSPI. You answer only with JSON. You never "
    "write a query that modifies data."
)

PROMPT = """{schema}

QUESTION: {question}

Write one SQLite SELECT query that answers the question.

Reply with JSON only:
{{"sql": "<the query>", "reason": "<one sentence on what it returns>"}}
{feedback}"""

FEEDBACK_HEADER = """
YOUR PREVIOUS ATTEMPTS FAILED. Fix the specific problem and try again.
"""


def execute_sql(
    sql: str,
    db_path: Optional[str] = None,
    timeout_s: float = TIMEOUT_S,
) -> Tuple[List[Dict[str, Any]], float]:
    """Run a validated query on a read-only connection, with a wall clock.

    The timeout uses SQLite's progress handler, which is called every N
    virtual-machine instructions and aborts the query when it returns non-zero.
    A Python-side timer cannot interrupt a query already executing inside the C
    library, so a cartesian join would otherwise hang the whole request with no
    way to stop it.
    """
    db_path = db_path or settings.db_path
    started = time.perf_counter()
    deadline = started + timeout_s

    with get_conn(db_path, read_only=True) as conn:
        def _watchdog() -> int:
            return 1 if time.perf_counter() > deadline else 0

        conn.set_progress_handler(_watchdog, 10_000)
        try:
            cur = conn.execute(sql)
            rows = [dict(r) for r in cur.fetchmany(MAX_ROWS)]
        finally:
            conn.set_progress_handler(None, 0)

    return rows, (time.perf_counter() - started) * 1000.0


def _format_feedback(attempts: List[Dict[str, Any]]) -> str:
    if not attempts:
        return ""
    lines = [FEEDBACK_HEADER]
    for i, a in enumerate(attempts, 1):
        lines.append(f"Attempt {i}: {a.get('query', '')}")
        lines.append(f"  -> ERROR: {a.get('error', 'unknown')}")
    return "\n".join(lines)


def run_sql_agent(
    question: str,
    db_path: Optional[str] = None,
    span: Any = None,
    max_attempts: int = MAX_ATTEMPTS,
) -> Dict[str, Any]:
    """Answer a question with SQL. Returns a partial AgentState.

    Every attempt is recorded, successes and failures alike: the recovery is
    the thing worth showing, and it is invisible if only the final query is
    kept.
    """
    schema = build_schema_doc(db_path)
    attempts: List[Dict[str, Any]] = []
    llm_calls: List[Dict[str, Any]] = []

    for attempt_no in range(1, max_attempts + 1):
        prompt = PROMPT.format(
            schema=schema,
            question=question,
            feedback=_format_feedback(attempts),
        )

        try:
            obj, resp = complete_json(
                prompt, system=SYSTEM, required_keys=["sql"], temperature=0.0
            )
        except LLMError as exc:
            log.error("sql_agent_llm_failed", attempt=attempt_no, error=str(exc))
            return {
                "sql_attempts": attempts,
                "sql_error": f"could not generate SQL: {exc}",
                "llm_calls": llm_calls,
            }

        llm_calls.append(
            {
                "node": "sql_agent",
                "attempt": attempt_no,
                "model": resp.model,
                "input_tokens": resp.input_tokens,
                "output_tokens": resp.output_tokens,
                "thinking_tokens": resp.thinking_tokens,
                "cached": resp.cached,
                "latency_ms": resp.latency_ms,
            }
        )
        if span is not None:
            span.record_llm(resp)

        raw_sql = (obj.get("sql") or "").strip()

        # --- Act: guard, then run -------------------------------------
        try:
            guarded = guard_sql(raw_sql, max_rows=MAX_ROWS)
        except UnsafeSQL as exc:
            attempts.append(
                {"query": raw_sql, "ok": False, "error": f"rejected: {exc}",
                 "row_count": 0}
            )
            log.warning("sql_rejected", attempt=attempt_no, error=str(exc))
            continue

        try:
            rows, elapsed_ms = execute_sql(guarded.sql, db_path)
        except sqlite3.Error as exc:
            # The database's own message is the observation the loop needs:
            # "no such column: inflation" tells the model exactly what to fix.
            attempts.append(
                {"query": guarded.sql, "ok": False,
                 "error": f"{type(exc).__name__}: {exc}", "row_count": 0}
            )
            log.warning("sql_failed", attempt=attempt_no, error=str(exc))
            continue

        attempts.append(
            {"query": guarded.sql, "ok": True, "error": None,
             "row_count": len(rows)}
        )
        log.info(
            "sql_ok",
            attempt=attempt_no,
            rows=len(rows),
            ms=round(elapsed_ms, 1),
            limit_added=guarded.limit_added,
        )
        return {
            "sql_query": guarded.sql,
            "sql_result": rows,
            "sql_attempts": attempts,
            "sql_error": None,
            "llm_calls": llm_calls,
        }

    # Out of attempts. An empty result and a failure are different things, and
    # the writer has to be able to tell them apart.
    last = attempts[-1]["error"] if attempts else "no attempts made"
    return {
        "sql_attempts": attempts,
        "sql_error": f"all {max_attempts} attempts failed; last error: {last}",
        "llm_calls": llm_calls,
    }
