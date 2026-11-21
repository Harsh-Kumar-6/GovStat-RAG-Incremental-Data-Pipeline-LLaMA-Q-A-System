"""The LangGraph wiring.

    supervisor -> (sql_agent | rag_agent | both | neither) -> writer -> critic
                                                               ^         |
                                                               |  revise |
                                                               +---------+
                                                                         -> finalize -> END

Two things worth understanding here.

**Parallel fan-out.** The router returns a *list* of node names. LangGraph then
runs those nodes in the same superstep and waits for all of them before moving
to the writer. That is why `sql_agent` and `rag_agent` can both feed one answer
without any explicit join: the graph's edges express the join.

**Why state is merged, not mutated.** Each node returns a partial dict.
`sql_attempts` and `llm_calls` are annotated with `operator.add` in state.py, so
parallel nodes appending to them merge instead of overwriting -- without that,
whichever node finished last would silently erase the other's record.

The tracer is passed through a ContextVar rather than through the state,
because a Trace holds open spans and a live SQLite handle: it is not
serialisable, and anything in the state has to be.
"""
from __future__ import annotations

import contextvars
import sqlite3
import uuid
from typing import Any, Dict, List, Optional

from langgraph.graph import END, START, StateGraph

from agents.critic import MAX_REVISIONS, run_critic
from agents.rag_agent import run_rag_agent
from agents.state import AgentState, initial_state
from agents.supervisor import run_supervisor
from agents.writer import run_writer
from scraper.config import settings
from scraper.logger import get_logger
from tracing.tracer import Trace

log = get_logger(__name__)

# The trace for the request currently being handled. A ContextVar rather than a
# global so concurrent requests cannot write into each other's traces.
_current_trace: contextvars.ContextVar[Optional[Trace]] = contextvars.ContextVar(
    "current_trace", default=None
)


class _NullSpan:
    """Stand-in when a node runs outside a trace, so nodes need no None checks."""

    def set(self, **kv: Any) -> "_NullSpan":
        return self

    def record_llm(self, resp: Any) -> "_NullSpan":
        return self


class _NullTrace:
    from contextlib import contextmanager as _cm

    @_cm
    def span(self, name: str, kind: str = "agent", **attrs: Any):
        yield _NullSpan()


def _trace() -> Any:
    return _current_trace.get() or _NullTrace()


# ---------------------------------------------------------------------------
# Data facts for the supervisor prompt
# ---------------------------------------------------------------------------
def _data_facts(db_path: Optional[str] = None) -> Dict[str, Any]:
    """What the data actually covers, read from the database.

    The supervisor has to know the SQL table holds only two months, or it
    confidently routes a 2024 question to SQL and gets an empty result that
    reads as "no inflation data exists".
    """
    db_path = db_path or settings.db_path
    facts = {"periods": "none", "n_docs": 0}
    try:
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
            rows = [r[0] for r in conn.execute(
                "SELECT DISTINCT period FROM cpi_state ORDER BY period"
            )]
            if rows:
                facts["periods"] = ", ".join(rows)
            facts["n_docs"] = conn.execute(
                "SELECT COUNT(*) FROM documents"
            ).fetchone()[0]
    except Exception as exc:
        log.warning("data_facts_failed", error=str(exc))
    return facts


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
def node_supervisor(state: AgentState) -> Dict[str, Any]:
    facts = _data_facts()
    with _trace().span("supervisor", "agent") as sp:
        out = run_supervisor(
            state["question"], periods=str(facts["periods"]),
            n_docs=int(facts["n_docs"]), span=sp,
        )
        sp.set(route=out.get("route"), reason=out.get("route_reason", "")[:200])
    return out


def node_sql_agent(state: AgentState) -> Dict[str, Any]:
    from agents.sql_agent import run_sql_agent

    with _trace().span("sql_agent", "agent") as sp:
        out = run_sql_agent(state["question"], span=sp)
        sp.set(
            attempts=len(out.get("sql_attempts") or []),
            rows=len(out.get("sql_result") or []),
            recovered=bool(
                out.get("sql_result") is not None
                and len(out.get("sql_attempts") or []) > 1
            ),
        )
    return out


def node_rag_agent(state: AgentState) -> Dict[str, Any]:
    with _trace().span("rag_agent", "agent") as sp:
        return run_rag_agent(state["question"], span=sp)


def node_writer(state: AgentState) -> Dict[str, Any]:
    with _trace().span("writer", "agent") as sp:
        out = run_writer(
            state["question"],
            sql_result=state.get("sql_result"),
            sql_query=state.get("sql_query"),
            passages=state.get("passages"),
            critique=state.get("critique"),
            revisions=int(state.get("revisions") or 0),
            route=state.get("route"),
            sql_error=state.get("sql_error"),
            span=sp,
        )
        sp.set(revision=out.get("revisions"), refused="final_answer" in out)
    return out


def node_critic(state: AgentState) -> Dict[str, Any]:
    with _trace().span("critic", "agent") as sp:
        return run_critic(
            state["question"],
            draft=state.get("draft_answer") or "",
            sql_result=state.get("sql_result"),
            sql_query=state.get("sql_query"),
            passages=state.get("passages"),
            span=sp,
        )


def node_finalize(state: AgentState) -> Dict[str, Any]:
    """Promote the approved draft to the final answer.

    A separate node so that "what shipped" is one explicit step rather than an
    implicit side effect of whichever node happened to run last.
    """
    return {"final_answer": state.get("draft_answer") or ""}


# ---------------------------------------------------------------------------
# Edges
# ---------------------------------------------------------------------------
def route_after_supervisor(state: AgentState) -> List[str]:
    """Fan out to the chosen agents, or straight to the writer to refuse."""
    route = state.get("route") or []
    targets = [f"{r}_agent" for r in route if r in ("sql", "rag")]
    return targets or ["writer"]


def route_after_writer(state: AgentState) -> str:
    """Skip the critic when the writer refused.

    There is nothing to verify: a refusal makes no factual claim, and sending
    it to the critic would spend a request to be told so.
    """
    if state.get("final_answer"):
        return END
    return "critic"


def route_after_critic(state: AgentState) -> str:
    if state.get("critic_verdict") == "revise" and \
            int(state.get("revisions") or 0) < MAX_REVISIONS:
        return "writer"
    return "finalize"


def build_graph(with_critic: bool = True) -> Any:
    """Compile the graph.

    `with_critic=False` is not a debug switch: it is one of the four
    configurations Phase 3 compares, and the comparison is the project's main
    result. Keeping it as a parameter means the no-critic run exercises the
    same code path as the real one.
    """
    g = StateGraph(AgentState)

    g.add_node("supervisor", node_supervisor)
    g.add_node("sql_agent", node_sql_agent)
    g.add_node("rag_agent", node_rag_agent)
    g.add_node("writer", node_writer)
    g.add_node("finalize", node_finalize)

    g.add_edge(START, "supervisor")
    g.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        ["sql_agent", "rag_agent", "writer"],
    )
    g.add_edge("sql_agent", "writer")
    g.add_edge("rag_agent", "writer")

    if with_critic:
        g.add_node("critic", node_critic)
        g.add_conditional_edges("writer", route_after_writer, ["critic", END])
        g.add_conditional_edges("critic", route_after_critic, ["writer", "finalize"])
    else:
        g.add_conditional_edges(
            "writer",
            lambda s: END if s.get("final_answer") else "finalize",
            ["finalize", END],
        )

    g.add_edge("finalize", END)
    return g.compile()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def ask(
    question: str,
    with_critic: bool = True,
    trace: Optional[Trace] = None,
    save_trace: bool = True,
) -> Dict[str, Any]:
    """Answer one question. Returns the final state plus trace metadata."""
    trace_id = uuid.uuid4().hex[:16]
    tr = trace or Trace(question=question, trace_id=trace_id)
    token = _current_trace.set(tr)

    try:
        graph = build_graph(with_critic=with_critic)
        state = graph.invoke(initial_state(question, tr.trace_id))
    finally:
        _current_trace.reset(token)

    result = dict(state)
    result["trace_id"] = tr.trace_id
    result["agent_path"] = tr.path()
    result["duration_ms"] = tr.duration_ms
    result["usage"] = tr.totals()

    if save_trace:
        tr.save(route=state.get("route"), final_answer=state.get("final_answer"))
    return result
