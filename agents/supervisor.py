"""Supervisor: decides which specialist agents handle the question.

Hierarchical delegation
-----------------------
One node plans, other nodes execute. The supervisor does no retrieval and no
SQL -- it only routes. That separation is what makes the decision *measurable*:
the route is a small structured value that can be compared against an expected
route, so Phase 3 can score routing accuracy independently of whether the
final answer was any good. A supervisor that answered as well as routed would
make the two inseparable.

Why the reason field matters
----------------------------
It is not decoration. When routing accuracy comes out at 70%, the reasons are
the only record of *why* the other 30% went wrong -- whether the model
misread the question or the routing rules themselves are ambiguous.
"""
from __future__ import annotations

from typing import Any, Dict, List

from agents.llm_provider import LLMError, complete_json
from scraper.config import settings
from scraper.logger import get_logger

log = get_logger(__name__)

VALID_ROUTES = ("sql", "rag")

SYSTEM = (
    "You route questions about Indian government statistics to the right "
    "tools. You answer only with JSON."
)

PROMPT = """You decide which tools answer a question about MoSPI statistics.

TOOLS
  sql  A typed table of Consumer Price Index figures by state and sector
       (Rural/Urban/Combined), with the index level and year-on-year
       inflation percent, for the months {periods}. Use for anything numeric,
       comparative, ranked, or "how much / which state / how did X change".
  rag  Full text of {n_docs} MoSPI press releases and reports, searchable.
       Use for methodology, definitions, explanations, reasons, coverage, and
       anything about what MoSPI said rather than what the numbers are.

CHOOSE
  ["sql"]        purely numeric
  ["rag"]        purely textual or explanatory
  ["sql","rag"]  a number AND an explanation (e.g. "how much did it change,
                 and why")
  []             the question cannot be answered from Indian government
                 statistics at all, or asks for a prediction about the future.
                 An empty route is a valid, correct decision -- prefer it over
                 guessing a tool for a question the data cannot answer.

QUESTION: {question}

Reply with JSON only:
{{"route": [...], "reason": "<one sentence>"}}"""


def _sanitize(route: Any) -> List[str]:
    """Keep only recognised route names, in a stable order, without repeats.

    A prompt-produced list is untrusted input: it arrives as a string, a
    nested list, or with "sql_agent" instead of "sql". Normalising here means
    a malformed route degrades the routing metric rather than crashing the
    graph.
    """
    if isinstance(route, str):
        route = [route]
    if not isinstance(route, (list, tuple)):
        return []
    seen: List[str] = []
    for item in route:
        name = str(item).strip().lower().replace("_agent", "")
        if name in VALID_ROUTES and name not in seen:
            seen.append(name)
    return [r for r in VALID_ROUTES if r in seen]


def run_supervisor(
    question: str,
    periods: str = "unknown",
    n_docs: int = 0,
    span: Any = None,
) -> Dict[str, Any]:
    """Return a partial AgentState with the route and the reason for it."""
    prompt = PROMPT.format(question=question, periods=periods, n_docs=n_docs)

    try:
        obj, resp = complete_json(
            prompt, system=SYSTEM, required_keys=["route"], temperature=0.0,
            cheap="supervisor" in settings.llm_cheap_nodes,
        )
    except LLMError as exc:
        # Fall back to retrieval rather than failing the request: RAG can
        # answer, or honestly fail to, for almost any question. Recorded as a
        # fallback so the eval does not score a guess as a real decision.
        log.error("supervisor_failed", error=str(exc))
        return {
            "route": ["rag"],
            "route_reason": f"fallback to rag: supervisor failed ({exc})",
            "llm_calls": [],
        }

    if span is not None:
        span.record_llm(resp)

    route = _sanitize(obj.get("route"))
    reason = str(obj.get("reason", ""))[:500]
    log.info("routed", route=route, reason=reason[:120])

    return {
        "route": route,
        "route_reason": reason,
        "llm_calls": [{
            "node": "supervisor", "model": resp.model,
            "input_tokens": resp.input_tokens, "output_tokens": resp.output_tokens,
            "thinking_tokens": resp.thinking_tokens, "cached": resp.cached,
            "latency_ms": resp.latency_ms,
        }],
    }
