"""Whole-graph tests with a stubbed model.

These are the tests that would have caught the mistakes that matter: a
revision loop that never terminates, a parallel fan-out where one agent's
state overwrites the other's, a refusal that still pays for a critic call.
"""
from __future__ import annotations

import sqlite3
import uuid

import pytest

from agents import llm_provider as P
from agents.graph import ask, build_graph
from scraper.config import settings


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "t.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE documents (id TEXT PRIMARY KEY, title TEXT, url TEXT,
            date_published TEXT, summary TEXT, category TEXT, hash TEXT);
        CREATE TABLE cpi_state (id TEXT PRIMARY KEY, source_document_id TEXT,
            source_table_id TEXT, release_date TEXT, period TEXT, state TEXT,
            sector TEXT, index_value REAL, inflation_yoy REAL);
        """
    )
    conn.execute("INSERT INTO documents VALUES ('d1','CPI Sept 2026','http://x',"
                 "'2026-10-13','','Price Statistics > CPI','h')")
    conn.execute(
        "INSERT INTO cpi_state VALUES (?,?,?,?,?,?,?,?,?)",
        (uuid.uuid4().hex, "d1", "t1", "2026-10-13", "2026-09", "Kerala",
         "Rural", 190.0, 6.22),
    )
    conn.commit()
    conn.close()
    return str(path)


@pytest.fixture(autouse=True)
def stub(tmp_path, monkeypatch, db):
    monkeypatch.setattr(settings, "llm_cache_dir", str(tmp_path / "cache"))
    monkeypatch.setattr(settings, "gemini_model", "test-model")
    monkeypatch.setattr(settings, "llm_provider", "gemini")
    monkeypatch.setattr(settings, "db_path", db)
    monkeypatch.setattr(settings, "processed_data_dir", str(tmp_path / "proc"))
    P.reset_budget(100)
    yield


def script(monkeypatch, *replies):
    """Return canned replies in order, and record the prompts that got them."""
    queue = list(replies)
    seen = []

    def fake(prompt, system, temperature, model, json_mode, max_output_tokens):
        seen.append(prompt)
        text = queue.pop(0) if queue else '{"verdict": "approve", "issues": []}'
        return P.LLMResponse(text=text, model=model, provider="gemini",
                             input_tokens=10, output_tokens=4, thinking_tokens=3)

    monkeypatch.setattr(P, "_gemini_complete", fake)
    return seen


def no_retrieval(monkeypatch, passages=None):
    """Stub retrieval, so these tests need no FAISS index or torch import."""
    import agents.graph as G

    monkeypatch.setattr(
        G, "run_rag_agent",
        lambda q, span=None, **k: {"passages": passages or [], "rag_error": None},
    )


# ---------------------------------------------------------------------------
# Paths through the graph
# ---------------------------------------------------------------------------
def test_sql_only_path(monkeypatch):
    script(
        monkeypatch,
        '{"route": ["sql"], "reason": "numeric"}',
        '{"sql": "SELECT state, inflation_yoy FROM cpi_state", "reason": "r"}',
        "Kerala rural inflation was 6.22%. [1]",
        '{"verdict": "approve", "issues": []}',
    )
    out = ask("rural inflation in Kerala?", save_trace=False)
    assert out["agent_path"] == ["supervisor", "sql_agent", "writer", "critic"]
    assert "6.22" in out["final_answer"]
    assert out["sql_query"].startswith("SELECT")


def test_rag_only_path(monkeypatch):
    no_retrieval(monkeypatch, [{"document_id": "d1", "title": "CPI Sept",
                                "url": "u", "text": "methodology note"}])
    script(
        monkeypatch,
        '{"route": ["rag"], "reason": "textual"}',
        "MoSPI described the methodology. [1]",
        '{"verdict": "approve", "issues": []}',
    )
    out = ask("what methodology did MoSPI describe?", save_trace=False)
    assert out["agent_path"] == ["supervisor", "rag_agent", "writer", "critic"]
    assert out["sql_query"] is None


def test_both_agents_run_and_neither_erases_the_other(monkeypatch):
    """The real risk in a parallel fan-out: two nodes returning partial state
    that merges badly, so whichever finished last wins."""
    no_retrieval(monkeypatch, [{"document_id": "d1", "title": "t", "url": "u",
                               "text": "passage text"}])
    script(
        monkeypatch,
        '{"route": ["sql","rag"], "reason": "both"}',
        '{"sql": "SELECT state FROM cpi_state", "reason": "r"}',
        "Both sources agree. [1]",
        '{"verdict": "approve", "issues": []}',
    )
    out = ask("number and explanation?", save_trace=False)
    assert set(out["agent_path"][1:3]) == {"sql_agent", "rag_agent"}
    assert out["sql_result"], "sql state was lost in the merge"
    assert out["passages"], "rag state was lost in the merge"


def test_unanswerable_refuses_without_calling_the_writer_model(monkeypatch):
    seen = script(monkeypatch, '{"route": [], "reason": "a prediction"}')
    out = ask("what will GDP be in 2030?", save_trace=False)
    assert out["agent_path"] == ["supervisor", "writer"]
    assert "can't answer" in out["final_answer"]
    # One call total: the supervisor's. No writer call, no critic call.
    assert len(seen) == 1
    assert P.budget.used == 1


# ---------------------------------------------------------------------------
# The critic loop
# ---------------------------------------------------------------------------
def test_a_rejected_draft_is_rewritten(monkeypatch):
    no_retrieval(monkeypatch, [{"document_id": "d1", "title": "t", "url": "u",
                               "text": "x"}])
    script(
        monkeypatch,
        '{"route": ["rag"], "reason": "r"}',
        "Inflation was 9.9%.",                      # draft 1: unsupported
        '{"verdict": "revise", "issues": ["9.9% is not in the evidence"]}',
        "The evidence does not give a figure.",      # draft 2
        '{"verdict": "approve", "issues": []}',
    )
    out = ask("q", save_trace=False)
    assert out["agent_path"].count("writer") == 2, "the draft was not revised"
    assert out["revisions"] == 2
    assert out["final_answer"] == "The evidence does not give a figure."


def test_the_revision_loop_terminates(monkeypatch):
    """A critic can always find something, so an unbounded loop never ends."""
    no_retrieval(monkeypatch, [{"document_id": "d1", "title": "t", "url": "u",
                               "text": "x"}])
    script(
        monkeypatch,
        '{"route": ["rag"], "reason": "r"}',
        "draft 1",
        '{"verdict": "revise", "issues": ["no"]}',
        "draft 2",
        '{"verdict": "revise", "issues": ["still no"]}',
        "draft 3",
        '{"verdict": "revise", "issues": ["never happy"]}',
    )
    out = ask("q", save_trace=False)
    assert out["revisions"] == 2, "the revision cap did not hold"
    assert out["final_answer"], "the graph ended without an answer"


def test_the_no_critic_configuration_skips_the_critic(monkeypatch):
    """One of the four Phase 3 configurations, not a debug flag."""
    no_retrieval(monkeypatch, [{"document_id": "d1", "title": "t", "url": "u",
                               "text": "x"}])
    script(monkeypatch, '{"route": ["rag"], "reason": "r"}', "the draft")
    out = ask("q", with_critic=False, save_trace=False)
    assert "critic" not in out["agent_path"]
    assert out["final_answer"] == "the draft"
    assert P.budget.used == 2       # supervisor + writer only


def test_with_critic_costs_more_requests_than_without(monkeypatch):
    """The quality/cost trade-off Phase 3 has to quantify."""
    no_retrieval(monkeypatch, [{"document_id": "d1", "title": "t", "url": "u",
                               "text": "x"}])
    script(monkeypatch, '{"route": ["rag"], "reason": "r"}', "d",
           '{"verdict": "approve", "issues": []}')
    ask("q", with_critic=True, save_trace=False)
    with_critic = P.budget.used

    P.reset_budget(100)
    script(monkeypatch, '{"route": ["rag"], "reason": "r"}', "d")
    ask("q", with_critic=False, save_trace=False)
    assert with_critic > P.budget.used


# ---------------------------------------------------------------------------
# Tracing
# ---------------------------------------------------------------------------
def test_the_trace_records_the_agent_path(monkeypatch):
    """Phase 3's routing metric reads the path from here, which is why
    tracing had to exist before the evals rather than after."""
    script(
        monkeypatch,
        '{"route": ["sql"], "reason": "r"}',
        '{"sql": "SELECT state FROM cpi_state", "reason": "r"}',
        "answer [1]",
        '{"verdict": "approve", "issues": []}',
    )
    out = ask("q", save_trace=False)
    assert out["agent_path"] == ["supervisor", "sql_agent", "writer", "critic"]
    assert out["trace_id"]


def test_the_trace_accumulates_tokens_across_nodes(monkeypatch):
    script(
        monkeypatch,
        '{"route": ["sql"], "reason": "r"}',
        '{"sql": "SELECT state FROM cpi_state", "reason": "r"}',
        "answer",
        '{"verdict": "approve", "issues": []}',
    )
    out = ask("q", save_trace=False)
    u = out["usage"]
    assert u["total_input_tokens"] > 0
    assert u["total_thinking_tokens"] > 0, "thinking tokens were not counted"
    # No prices configured, so cost is None rather than a fabricated 0.
    assert u["total_cost_usd"] is None


def test_a_saved_trace_is_queryable(monkeypatch, db):
    script(monkeypatch, '{"route": [], "reason": "unanswerable"}')
    out = ask("q", save_trace=True)
    conn = sqlite3.connect(db)
    row = conn.execute(
        "SELECT question, route, n_spans FROM traces WHERE trace_id = ?",
        (out["trace_id"],),
    ).fetchone()
    assert row is not None, "the trace was not persisted"
    spans = conn.execute(
        "SELECT name FROM spans WHERE trace_id = ? ORDER BY started_at",
        (out["trace_id"],),
    ).fetchall()
    assert {s[0] for s in spans} >= {"supervisor", "writer"}


def test_graph_compiles_both_ways():
    assert "critic" in build_graph(with_critic=True).get_graph().nodes
    assert "critic" not in build_graph(with_critic=False).get_graph().nodes
