"""A small tracer: one trace per request, one span per agent or tool call.

What tracing actually is
------------------------
A trace is one request. A span is one unit of work inside it, with a start, an
end, a parent, and attributes. Spans nest, so the parent/child chain *is* the
agent path -- which is why routing accuracy in Phase 3 can be measured from
traces without the graph reporting anything extra.

Why build it rather than install Langfuse
-----------------------------------------
Three reasons, in order: it costs nothing, it needs no service running beside
the app, and writing it is how you learn what a span is. The data model here is
deliberately the same shape as OpenTelemetry's (trace_id, span_id, parent_id,
name, start, end, attributes), so moving to a real backend later is a matter of
changing the exporter, not the instrumentation.

Two sinks on purpose:
  - JSON lines: append-only, cheap, survives a crash mid-request, easy to diff.
  - SQLite: queryable. "p95 latency by agent" is a GROUP BY, not a log scrape.
"""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from scraper.config import settings
from scraper.logger import get_logger
from tracing.pricing import cost_usd

log = get_logger(__name__)

TRACE_DDL = """
CREATE TABLE IF NOT EXISTS traces (
    trace_id    TEXT PRIMARY KEY,
    question    TEXT,
    route       TEXT,
    final_answer TEXT,
    ok          INTEGER,
    error       TEXT,
    started_at  TEXT,
    duration_ms REAL,
    n_spans     INTEGER,
    total_input_tokens  INTEGER,
    total_output_tokens INTEGER,
    total_thinking_tokens INTEGER,
    total_cost_usd REAL,
    created_at  TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS spans (
    span_id     TEXT PRIMARY KEY,
    trace_id    TEXT NOT NULL,
    parent_id   TEXT,
    name        TEXT NOT NULL,
    kind        TEXT,
    started_at  TEXT,
    duration_ms REAL,
    ok          INTEGER,
    error       TEXT,
    model       TEXT,
    input_tokens  INTEGER,
    output_tokens INTEGER,
    thinking_tokens INTEGER,
    cost_usd    REAL,
    cached      INTEGER,
    attributes  TEXT
);

CREATE INDEX IF NOT EXISTS idx_spans_trace  ON spans(trace_id);
CREATE INDEX IF NOT EXISTS idx_spans_name   ON spans(name);
CREATE INDEX IF NOT EXISTS idx_traces_start ON traces(started_at);
"""


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())


class Span:
    """One unit of work. Mutable while open, frozen once recorded."""

    def __init__(self, trace_id: str, name: str, kind: str, parent_id: Optional[str]):
        self.span_id = uuid.uuid4().hex[:16]
        self.trace_id = trace_id
        self.parent_id = parent_id
        self.name = name
        self.kind = kind
        self.started_at = _now_iso()
        self._t0 = time.perf_counter()
        self.duration_ms: float = 0.0
        self.ok = True
        self.error: Optional[str] = None
        self.model: Optional[str] = None
        self.input_tokens = 0
        self.output_tokens = 0
        self.thinking_tokens = 0
        self.cost_usd: Optional[float] = None
        self.cached = False
        self.attributes: Dict[str, Any] = {}

    def set(self, **kv: Any) -> "Span":
        """Attach attributes. Keep these small: a span is not a log sink."""
        self.attributes.update(kv)
        return self

    def record_llm(self, resp: Any) -> "Span":
        """Fold an LLMResponse into this span's usage figures."""
        self.model = getattr(resp, "model", None)
        self.input_tokens += int(getattr(resp, "input_tokens", 0) or 0)
        self.output_tokens += int(getattr(resp, "output_tokens", 0) or 0)
        self.thinking_tokens += int(getattr(resp, "thinking_tokens", 0) or 0)
        self.cached = bool(getattr(resp, "cached", False))
        if self.model:
            # Thinking tokens are billed as output, so they belong in the
            # output side of the cost calculation.
            self.cost_usd = cost_usd(
                self.model, self.input_tokens, self.output_tokens + self.thinking_tokens
            )
        return self

    def to_row(self) -> Dict[str, Any]:
        return {
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "parent_id": self.parent_id,
            "name": self.name,
            "kind": self.kind,
            "started_at": self.started_at,
            "duration_ms": round(self.duration_ms, 2),
            "ok": int(self.ok),
            "error": self.error,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "thinking_tokens": self.thinking_tokens,
            "cost_usd": self.cost_usd,
            "cached": int(self.cached),
            "attributes": json.dumps(self.attributes, ensure_ascii=False, default=str),
        }


class Trace:
    """One request. Holds its spans and knows how to persist them."""

    def __init__(self, question: str = "", trace_id: Optional[str] = None,
                 db_path: Optional[str] = None, jsonl_dir: Optional[str] = None):
        self.trace_id = trace_id or uuid.uuid4().hex[:16]
        self.question = question
        self.started_at = _now_iso()
        self._t0 = time.perf_counter()
        self.spans: List[Span] = []
        self._stack: List[Span] = []
        self.ok = True
        self.error: Optional[str] = None
        self.db_path = db_path or settings.db_path
        self.jsonl_dir = Path(jsonl_dir or Path(settings.processed_data_dir) / "traces")

    # -- span creation ---------------------------------------------------
    @contextmanager
    def span(self, name: str, kind: str = "agent", **attrs: Any) -> Iterator[Span]:
        """Open a span. Nesting is implicit: the open span is the parent.

        The `finally` block is what makes a trace trustworthy. If a node
        raises, the span must still be closed and recorded with the error --
        otherwise the only traces you keep are the successful ones, and you
        have built a tool that cannot see failures.
        """
        parent = self._stack[-1].span_id if self._stack else None
        sp = Span(self.trace_id, name, kind, parent)
        sp.set(**attrs)
        self._stack.append(sp)
        try:
            yield sp
        except Exception as exc:
            sp.ok = False
            sp.error = f"{type(exc).__name__}: {exc}"[:500]
            self.ok = False
            self.error = sp.error
            raise
        finally:
            sp.duration_ms = (time.perf_counter() - sp._t0) * 1000.0
            self._stack.pop()
            self.spans.append(sp)

    # -- totals ----------------------------------------------------------
    @property
    def duration_ms(self) -> float:
        return (time.perf_counter() - self._t0) * 1000.0

    def totals(self) -> Dict[str, Any]:
        costs = [s.cost_usd for s in self.spans if s.cost_usd is not None]
        return {
            "n_spans": len(self.spans),
            "total_input_tokens": sum(s.input_tokens for s in self.spans),
            "total_output_tokens": sum(s.output_tokens for s in self.spans),
            "total_thinking_tokens": sum(s.thinking_tokens for s in self.spans),
            # None, not 0, when no span had a known price: 0 would read as free.
            "total_cost_usd": sum(costs) if costs else None,
        }

    def path(self) -> List[str]:
        """The agent path, in the order the work happened.

        This is what Phase 3's routing metric compares against the expected
        route -- the reason tracing has to exist before the evals, not after.
        """
        return [s.name for s in self.spans if s.kind == "agent"]

    # -- persistence -----------------------------------------------------
    def save(self, route: Optional[List[str]] = None,
             final_answer: Optional[str] = None) -> str:
        totals = self.totals()
        header = {
            "trace_id": self.trace_id,
            "question": self.question,
            "route": json.dumps(route or [], ensure_ascii=False),
            "final_answer": (final_answer or "")[:4000],
            "ok": int(self.ok),
            "error": self.error,
            "started_at": self.started_at,
            "duration_ms": round(self.duration_ms, 2),
            **totals,
        }

        # JSON lines first: it is append-only and cannot fail a run.
        try:
            self.jsonl_dir.mkdir(parents=True, exist_ok=True)
            out = self.jsonl_dir / f"{self.started_at[:10]}.jsonl"
            with out.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(
                    {"trace": header, "spans": [s.to_row() for s in self.spans]},
                    ensure_ascii=False, default=str) + "\n")
        except Exception as exc:
            log.warning("trace_jsonl_failed", error=str(exc))

        try:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(self.db_path) as conn:
                conn.executescript(TRACE_DDL)
                conn.execute(
                    "INSERT OR REPLACE INTO traces "
                    "(trace_id, question, route, final_answer, ok, error, started_at,"
                    " duration_ms, n_spans, total_input_tokens, total_output_tokens,"
                    " total_thinking_tokens, total_cost_usd) "
                    "VALUES (:trace_id,:question,:route,:final_answer,:ok,:error,"
                    ":started_at,:duration_ms,:n_spans,:total_input_tokens,"
                    ":total_output_tokens,:total_thinking_tokens,:total_cost_usd)",
                    header,
                )
                for sp in self.spans:
                    row = sp.to_row()
                    conn.execute(
                        "INSERT OR REPLACE INTO spans "
                        "(span_id,trace_id,parent_id,name,kind,started_at,duration_ms,"
                        " ok,error,model,input_tokens,output_tokens,thinking_tokens,"
                        " cost_usd,cached,attributes) "
                        "VALUES (:span_id,:trace_id,:parent_id,:name,:kind,:started_at,"
                        ":duration_ms,:ok,:error,:model,:input_tokens,:output_tokens,"
                        ":thinking_tokens,:cost_usd,:cached,:attributes)",
                        row,
                    )
                conn.commit()
        except Exception as exc:
            # A tracing failure must never fail the request it is observing.
            log.warning("trace_db_failed", error=str(exc))

        return self.trace_id


def init_trace_tables(db_path: Optional[str] = None) -> None:
    with sqlite3.connect(db_path or settings.db_path) as conn:
        conn.executescript(TRACE_DDL)
        conn.commit()
