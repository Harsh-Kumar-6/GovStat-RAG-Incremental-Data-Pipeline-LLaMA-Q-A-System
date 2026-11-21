"""The state every node in the graph reads and writes.

How LangGraph state works
-------------------------
A graph is nodes plus a shared state object. Each node receives the state and
returns a *partial* dict; LangGraph merges that into the state and passes it on.
A node never mutates the state in place, which is what makes the run
reproducible and replayable -- and is why every field below is either replaced
wholesale or appended to through a reducer.

The fields are deliberately flat and JSON-serialisable: the same dict is what
gets written to the trace, so a trace is a faithful record of what each node
actually saw, not a summary of it.
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict

# Which specialist agents the supervisor may call. Kept as a Literal so a typo
# in a prompt-produced route is caught by validation, not discovered at 3am.
Route = Literal["sql", "rag"]


class Passage(TypedDict, total=False):
    """One retrieved chunk, with everything a citation needs."""

    id: str
    document_id: str
    text: str
    title: str
    url: str
    date_published: Optional[str]
    category: Optional[str]
    score: float


class SQLAttempt(TypedDict, total=False):
    """One turn of the SQL agent's ReAct loop, kept for evals and debugging.

    Storing the failures, not just the final query, is the point: the ReAct
    story in an interview is "it recovered from this error", and you cannot
    show that if only the successful query survives.
    """

    query: str
    ok: bool
    error: Optional[str]
    row_count: int


class AgentState(TypedDict, total=False):
    # --- input ---
    question: str
    trace_id: str

    # --- supervisor ---
    route: List[str]            # ["sql"], ["rag"], or ["sql", "rag"]
    route_reason: str

    # --- sql agent ---
    sql_query: Optional[str]            # the query that finally ran
    sql_result: Optional[List[Dict[str, Any]]]
    sql_attempts: Annotated[List[SQLAttempt], operator.add]
    sql_error: Optional[str]            # set when every attempt failed

    # --- rag agent ---
    passages: List[Passage]
    rag_error: Optional[str]

    # --- writer / critic ---
    draft_answer: Optional[str]
    citations: List[Dict[str, str]]
    critique: Optional[str]
    critic_verdict: Optional[str]       # "approve" | "revise"
    revisions: int                      # incremented by the writer
    final_answer: Optional[str]

    # --- accounting ---
    # Appended to by every node, so one run yields the whole call history.
    llm_calls: Annotated[List[Dict[str, Any]], operator.add]


def initial_state(question: str, trace_id: str) -> AgentState:
    """Build a fully-populated starting state.

    Every key is present from the start. A node that reads a key another node
    has not written yet gets a defined empty value rather than a KeyError, so
    a partial run still produces a usable (if incomplete) answer.
    """
    return AgentState(
        question=question,
        trace_id=trace_id,
        route=[],
        route_reason="",
        sql_query=None,
        sql_result=None,
        sql_attempts=[],
        sql_error=None,
        passages=[],
        rag_error=None,
        draft_answer=None,
        citations=[],
        critique=None,
        critic_verdict=None,
        revisions=0,
        final_answer=None,
        llm_calls=[],
    )


def has_evidence(state: AgentState) -> bool:
    """True when at least one agent produced something to ground an answer in.

    Used to decide whether the writer can answer at all. Without this check a
    question with no SQL rows and no passages still reaches the writer, which
    then has nothing to work from and invents something -- the exact failure
    the refusal metric in Phase 3 measures.
    """
    return bool(state.get("sql_result")) or bool(state.get("passages"))
