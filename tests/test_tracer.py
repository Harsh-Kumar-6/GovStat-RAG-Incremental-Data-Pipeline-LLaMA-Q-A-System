"""Tracer tests.

The one that matters most is the failure case. A tracer that only records
successful spans is worse than none: it gives you confidence in a view that
hides exactly the requests you need to look at.
"""
from __future__ import annotations

import json
import sqlite3

import pytest

from tracing.pricing import cost_usd
from tracing.tracer import Trace


@pytest.fixture
def trace(tmp_path):
    return Trace(question="q", db_path=str(tmp_path / "t.db"),
                 jsonl_dir=str(tmp_path / "traces"))


class FakeResp:
    model = "m1"
    input_tokens = 100
    output_tokens = 20
    thinking_tokens = 300
    cached = False


def test_spans_nest_by_parent(trace):
    with trace.span("supervisor", "agent"):
        with trace.span("llm", "llm"):
            pass
    by_name = {s.name: s for s in trace.spans}
    assert by_name["llm"].parent_id == by_name["supervisor"].span_id
    assert by_name["supervisor"].parent_id is None


def test_a_failing_span_is_still_recorded(trace):
    with pytest.raises(ValueError):
        with trace.span("critic", "agent"):
            raise ValueError("boom")
    span = trace.spans[-1]
    assert span.ok is False and "boom" in span.error
    assert trace.ok is False


def test_an_exception_propagates(trace):
    """Tracing observes; it must not swallow the error it observed."""
    with pytest.raises(KeyError):
        with trace.span("x"):
            raise KeyError("k")


def test_duration_is_recorded(trace):
    with trace.span("x"):
        pass
    assert trace.spans[0].duration_ms >= 0


def test_agent_path_excludes_non_agent_spans(trace):
    with trace.span("supervisor", "agent"):
        with trace.span("gemini", "llm"):
            pass
    with trace.span("writer", "agent"):
        pass
    assert trace.path() == ["supervisor", "writer"]


def test_token_usage_rolls_up(trace):
    with trace.span("writer", "agent") as sp:
        sp.record_llm(FakeResp())
    t = trace.totals()
    assert t["total_input_tokens"] == 100
    assert t["total_output_tokens"] == 20
    assert t["total_thinking_tokens"] == 300


def test_unpriced_model_reports_none_not_zero(trace):
    """Zero reads as 'free'. None reads as 'unknown', which is the truth."""
    with trace.span("writer", "agent") as sp:
        sp.record_llm(FakeResp())
    assert trace.totals()["total_cost_usd"] is None
    assert cost_usd("m1", 100, 20) is None


def test_thinking_tokens_are_billed_as_output(monkeypatch, trace):
    """A reasoning model's thinking tokens dominate the bill, so leaving them
    out of the cost calculation understates it by an order of magnitude."""
    import tracing.pricing as pricing

    monkeypatch.setattr(
        pricing, "_cache",
        {"m1": {"input_per_1m": 1_000_000.0, "output_per_1m": 1_000_000.0}},
    )
    with trace.span("writer", "agent") as sp:
        sp.record_llm(FakeResp())
    # 100 in + (20 visible + 300 thinking) out, at $1 per token.
    assert trace.spans[0].cost_usd == pytest.approx(420.0)


def test_save_writes_both_sinks(trace, tmp_path):
    with trace.span("supervisor", "agent") as sp:
        sp.set(route=["sql"])
    trace.save(route=["sql"], final_answer="ans")

    conn = sqlite3.connect(str(tmp_path / "t.db"))
    assert conn.execute("SELECT COUNT(*) FROM traces").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM spans").fetchone()[0] == 1

    files = list((tmp_path / "traces").glob("*.jsonl"))
    assert len(files) == 1
    rec = json.loads(files[0].read_text(encoding="utf-8").strip())
    assert rec["trace"]["question"] == "q"
    assert rec["spans"][0]["name"] == "supervisor"


def test_span_attributes_survive_the_round_trip(trace, tmp_path):
    with trace.span("rag_agent", "agent") as sp:
        sp.set(k=5, doc_ids=["d1", "d2"])
    trace.save()
    conn = sqlite3.connect(str(tmp_path / "t.db"))
    attrs = json.loads(
        conn.execute("SELECT attributes FROM spans").fetchone()[0]
    )
    assert attrs["k"] == 5 and attrs["doc_ids"] == ["d1", "d2"]


def test_a_tracing_failure_does_not_fail_the_request(tmp_path):
    """Observability must never be the thing that breaks the system."""
    t = Trace(question="q", db_path="/nonexistent/dir/x.db",
              jsonl_dir="/nonexistent/dir/traces")
    with t.span("x", "agent"):
        pass
    assert t.save()      # returns the trace id instead of raising


def test_per_agent_latency_is_queryable(trace, tmp_path):
    """The reason spans go to SQL and not only to a log file: 'p95 latency by
    agent' should be a GROUP BY, not a log scrape."""
    for name in ("supervisor", "sql_agent", "writer"):
        with trace.span(name, "agent"):
            pass
    trace.save()
    conn = sqlite3.connect(str(tmp_path / "t.db"))
    rows = conn.execute(
        "SELECT name, AVG(duration_ms) FROM spans WHERE kind='agent' "
        "GROUP BY name ORDER BY name"
    ).fetchall()
    assert [r[0] for r in rows] == ["sql_agent", "supervisor", "writer"]
