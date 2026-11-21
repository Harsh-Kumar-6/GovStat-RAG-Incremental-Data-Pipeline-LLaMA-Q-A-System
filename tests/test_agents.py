"""Agent and graph tests. Every LLM call is stubbed.

The design goal of the whole agent layer was that its logic -- routing, the
ReAct recovery, the revision cap, the refusal path -- is testable without a
model. If these needed an API key they could not run in CI, and each push
would spend part of a daily free-tier quota.
"""
from __future__ import annotations

import sqlite3
import uuid

import pytest

from agents import llm_provider as P
from agents.critic import run_critic
from agents.state import has_evidence, initial_state
from agents.supervisor import _sanitize, run_supervisor
from agents.writer import build_citations, run_writer
from scraper.config import settings


@pytest.fixture(autouse=True)
def no_real_calls(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "llm_cache_dir", str(tmp_path / "cache"))
    monkeypatch.setattr(settings, "gemini_model", "test-model")
    monkeypatch.setattr(settings, "llm_provider", "gemini")
    P.reset_budget(100)
    yield


def reply_with(monkeypatch, *texts):
    """Queue canned model replies, one per call."""
    queue = list(texts)
    seen = []

    def fake(prompt, system, temperature, model, json_mode, max_output_tokens):
        seen.append(prompt)
        text = queue.pop(0) if queue else (texts[-1] if texts else "")
        return P.LLMResponse(text=text, model=model, provider="gemini",
                             input_tokens=10, output_tokens=5, thinking_tokens=7)

    monkeypatch.setattr(P, "_gemini_complete", fake)
    return seen


# ---------------------------------------------------------------------------
# Supervisor
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "raw,expected",
    [
        ('{"route": ["sql"], "reason": "numeric"}', ["sql"]),
        ('{"route": ["rag"], "reason": "textual"}', ["rag"]),
        ('{"route": ["sql","rag"], "reason": "both"}', ["sql", "rag"]),
        ('{"route": [], "reason": "unanswerable"}', []),
        ('{"route": "sql", "reason": "a bare string"}', ["sql"]),
        ('{"route": ["sql_agent"], "reason": "suffixed"}', ["sql"]),
        ('{"route": ["rag","sql"], "reason": "order"}', ["sql", "rag"]),
        ('{"route": ["mongodb"], "reason": "hallucinated tool"}', []),
    ],
)
def test_supervisor_normalizes_whatever_the_model_returns(monkeypatch, raw, expected):
    reply_with(monkeypatch, raw)
    assert run_supervisor("q")["route"] == expected


def test_supervisor_falls_back_to_rag_when_the_model_fails(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("400 bad request")

    monkeypatch.setattr(P, "_gemini_complete", boom)
    out = run_supervisor("q")
    assert out["route"] == ["rag"]
    assert "fallback" in out["route_reason"]


def test_supervisor_is_told_which_periods_exist(monkeypatch):
    """Without this the supervisor routes a 2024 question to a table that
    only holds 2026 and gets an empty result that reads as 'no data exists'."""
    seen = reply_with(monkeypatch, '{"route": ["sql"], "reason": "x"}')
    run_supervisor("q", periods="2026-09, 2026-10", n_docs=48)
    assert "2026-09, 2026-10" in seen[0] and "48" in seen[0]


# ---------------------------------------------------------------------------
# SQL agent: the ReAct loop
# ---------------------------------------------------------------------------
@pytest.fixture
def sql_db(tmp_path):
    """A tiny cpi_state, so the agent runs against real SQLite behaviour."""
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
    for st, val in (("Kerala", 6.22), ("Odisha", 7.24)):
        conn.execute(
            "INSERT INTO cpi_state VALUES (?,?,?,?,?,?,?,?,?)",
            (uuid.uuid4().hex, "d1", "t1", "2026-10-13", "2026-09", st,
             "Rural", 190.0, val),
        )
    conn.commit()
    conn.close()
    return str(path)


def test_sql_agent_answers_on_the_first_try(monkeypatch, sql_db):
    from agents.sql_agent import run_sql_agent

    reply_with(monkeypatch, '{"sql": "SELECT state, inflation_yoy FROM cpi_state '
                            'WHERE sector = \'Rural\'", "reason": "r"}')
    out = run_sql_agent("which state", db_path=sql_db)
    assert out["sql_error"] is None
    assert len(out["sql_result"]) == 2
    assert len(out["sql_attempts"]) == 1


def test_sql_agent_recovers_from_a_bad_column(monkeypatch, sql_db):
    """This is the ReAct story: the database's error becomes the observation
    the model corrects from."""
    from agents.sql_agent import run_sql_agent

    seen = reply_with(
        monkeypatch,
        '{"sql": "SELECT inflation FROM cpi_state", "reason": "wrong column"}',
        '{"sql": "SELECT inflation_yoy FROM cpi_state", "reason": "fixed"}',
    )
    out = run_sql_agent("q", db_path=sql_db)

    assert out["sql_error"] is None, "the agent did not recover"
    assert len(out["sql_attempts"]) == 2
    assert out["sql_attempts"][0]["ok"] is False
    assert "no such column" in out["sql_attempts"][0]["error"]
    # The error text must reach the second prompt, or the retry is a blind
    # re-roll rather than a correction.
    assert "no such column" in seen[1]


def test_sql_agent_feeds_a_guard_rejection_back_to_the_model(monkeypatch, sql_db):
    from agents.sql_agent import run_sql_agent

    seen = reply_with(
        monkeypatch,
        '{"sql": "DROP TABLE cpi_state", "reason": "malicious or confused"}',
        '{"sql": "SELECT state FROM cpi_state", "reason": "fixed"}',
    )
    out = run_sql_agent("q", db_path=sql_db)
    assert out["sql_error"] is None
    assert "rejected" in out["sql_attempts"][0]["error"]
    assert "DROP" in seen[1]


def test_sql_agent_gives_up_after_three_attempts(monkeypatch, sql_db):
    """Bounded: an unbounded loop spends a daily quota on one bad question."""
    from agents.sql_agent import run_sql_agent

    reply_with(monkeypatch, '{"sql": "SELECT nope FROM cpi_state", "reason": "r"}')
    out = run_sql_agent("q", db_path=sql_db)
    assert out["sql_error"] is not None
    assert len(out["sql_attempts"]) == 3
    assert out.get("sql_result") is None


def test_sql_agent_records_every_attempt_not_just_the_winner(monkeypatch, sql_db):
    from agents.sql_agent import run_sql_agent

    reply_with(
        monkeypatch,
        '{"sql": "SELECT bad FROM cpi_state", "reason": "r"}',
        '{"sql": "SELECT worse FROM cpi_state", "reason": "r"}',
        '{"sql": "SELECT state FROM cpi_state", "reason": "r"}',
    )
    out = run_sql_agent("q", db_path=sql_db)
    assert [a["ok"] for a in out["sql_attempts"]] == [False, False, True]


def test_sql_execution_is_read_only(sql_db):
    from agents.sql_agent import execute_sql

    with pytest.raises(sqlite3.Error, match="readonly"):
        execute_sql("CREATE TABLE evil (a TEXT)", db_path=sql_db)


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------
def test_writer_refuses_without_evidence_and_spends_nothing(monkeypatch):
    """The refusal is enforced in code, before the model is called. No prompt
    wording reliably stops a model inventing a figure when given nothing."""
    calls = reply_with(monkeypatch, "should never be called")
    out = run_writer("q", route=[])
    assert "can't answer" in out["final_answer"]
    assert calls == [], "the writer called the model with no evidence"
    assert P.budget.used == 0


def test_writer_reports_a_failed_query_rather_than_answering_anyway(monkeypatch):
    reply_with(monkeypatch, "unused")
    out = run_writer("q", sql_error="all 3 attempts failed")
    assert "did not succeed" in out["final_answer"]


def test_writer_puts_the_evidence_in_the_prompt(monkeypatch):
    seen = reply_with(monkeypatch, "Kerala was 6.22%. [1]")
    run_writer(
        "q",
        sql_result=[{"state": "Kerala", "inflation_yoy": 6.22}],
        sql_query="SELECT ...",
        passages=[{"title": "CPI Sept", "url": "u", "text": "body text here",
                   "document_id": "d1"}],
    )
    assert "Kerala" in seen[0] and "6.22" in seen[0] and "body text here" in seen[0]


def test_writer_passes_the_critique_into_the_revision(monkeypatch):
    seen = reply_with(monkeypatch, "draft one", "draft two")
    run_writer("q", passages=[{"title": "t", "url": "u", "text": "x",
                               "document_id": "d1"}])
    run_writer("q", passages=[{"title": "t", "url": "u", "text": "x",
                               "document_id": "d1"}],
               critique="the 5.2% figure is not in the evidence", revisions=1)
    assert "5.2% figure is not in the evidence" in seen[1]


def test_writer_increments_the_revision_counter(monkeypatch):
    reply_with(monkeypatch, "d")
    out = run_writer("q", passages=[{"title": "t", "url": "u", "text": "x",
                                     "document_id": "d1"}], revisions=1)
    assert out["revisions"] == 2


def test_citations_cover_sql_answers_too():
    """A number without a source is the failure this project exists to fix."""
    cites = build_citations(
        [], [{"source_document_id": "d9", "title": "CPI Oct", "url": "u9"}]
    )
    assert cites == [{"document_id": "d9", "title": "CPI Oct", "url": "u9",
                      "source": "sql"}]


def test_citations_are_deduplicated_by_document():
    cites = build_citations(
        [{"document_id": "d1", "title": "t", "url": "u"},
         {"document_id": "d1", "title": "t", "url": "u"}],
        [{"source_document_id": "d1"}],
    )
    assert len(cites) == 1


# ---------------------------------------------------------------------------
# Critic
# ---------------------------------------------------------------------------
def test_critic_approves(monkeypatch):
    reply_with(monkeypatch, '{"verdict": "approve", "issues": [], "reason": "ok"}')
    out = run_critic("q", "draft")
    assert out["critic_verdict"] == "approve" and out["critique"] is None


def test_critic_rejects_and_explains(monkeypatch):
    reply_with(monkeypatch, '{"verdict": "revise", "issues": ["7.9% is not in '
                            'the evidence"], "reason": "unsupported number"}')
    out = run_critic("q", "inflation was 7.9%")
    assert out["critic_verdict"] == "revise"
    assert "7.9%" in out["critique"]


@pytest.mark.parametrize("verdict", ['"maybe"', '"APPROVE"', '"reject"', "null"])
def test_an_unrecognised_verdict_defaults_to_approve(monkeypatch, verdict):
    """Shipping a grounded draft beats blocking it on a malformed verdict."""
    reply_with(monkeypatch, f'{{"verdict": {verdict}, "issues": []}}')
    assert run_critic("q", "d")["critic_verdict"] == "approve"


def test_critic_failure_does_not_block_the_answer(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("400 invalid")

    monkeypatch.setattr(P, "_gemini_complete", boom)
    out = run_critic("q", "d")
    assert out["critic_verdict"] == "approve"
    assert "unavailable" in out["critique"]


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
def test_initial_state_defines_every_key():
    """A node reading a key another node has not written yet must get an empty
    value, not a KeyError."""
    s = initial_state("q", "t")
    for key in ("question", "route", "sql_query", "passages", "draft_answer",
                "citations", "revisions", "final_answer", "llm_calls",
                "sql_attempts", "trace_id"):
        assert key in s


def test_has_evidence_distinguishes_nothing_from_something():
    s = initial_state("q", "t")
    assert has_evidence(s) is False
    assert has_evidence({**s, "sql_result": [{"a": 1}]}) is True
    assert has_evidence({**s, "passages": [{"id": "c"}]}) is True
    # An empty result set is not evidence: it is the absence of it.
    assert has_evidence({**s, "sql_result": []}) is False
