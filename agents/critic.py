"""Critic: checks the draft against the evidence before it ships.

Self-reflection
---------------
A second model call whose only job is to find faults in the first one's
output. It never writes an answer -- it returns a verdict and a list of
problems, and the writer gets one more attempt.

Why it works at all: generating a fluent answer and verifying a specific claim
are different tasks, and the second is easier. The critic is given the
evidence and the draft and asked a narrow question ("is this number in there?"),
with no pressure to be helpful or complete.

Why it is bounded at two revisions: a critic can always find something, so an
unbounded loop never terminates. Two attempts, then ship with the critique
recorded.

Known limitation, worth stating plainly: the critic shares the writer's blind
spots because it is the same model. It reliably catches a number that is absent
from the evidence; it is much weaker at catching a subtly wrong *reading* of a
number that is present. Phase 3 measures how much it actually helps, which is
the entire reason the no-critic configuration is kept for comparison.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from agents.llm_provider import LLMError, complete_json
from scraper.config import settings
from scraper.logger import get_logger

log = get_logger(__name__)

MAX_REVISIONS = 2

SYSTEM = (
    "You verify draft answers about Indian government statistics against the "
    "evidence they were written from. You are strict and literal. You answer "
    "only with JSON."
)

PROMPT = """QUESTION: {question}

EVIDENCE (the only admissible facts)
{evidence}

DRAFT ANSWER
{draft}

Check the draft, claim by claim:
1. Is every number in the draft present in the evidence, with the same value?
2. Does every number refer to the right period, state, and sector?
   (The 'Combined' sector is not a total. 'period' is the month described,
   not the month published.)
3. Are the citations present and do they point at sources that actually
   support the claims they are attached to?
4. Does the draft answer the question that was asked?
5. Does it add anything the evidence does not support, including explanations
   of causes?

Reply with JSON only:
{{"verdict": "approve" | "revise",
  "issues": ["<specific problem>", ...],
  "reason": "<one sentence>"}}

Approve if every check passes. Do not ask for stylistic changes: only
factual, citation, or scope problems justify "revise"."""


def run_critic(
    question: str,
    draft: str,
    sql_result: Optional[List[Dict[str, Any]]] = None,
    sql_query: Optional[str] = None,
    passages: Optional[List[Dict[str, Any]]] = None,
    span: Any = None,
) -> Dict[str, Any]:
    """Return a partial AgentState with the verdict and the critique."""
    from agents.writer import _format_passages, _format_sql_evidence

    evidence = "\n".join(x for x in (
        _format_sql_evidence(sql_result or [], sql_query),
        _format_passages(passages or []),
    ) if x) or "(no evidence was available)"

    prompt = PROMPT.format(question=question, evidence=evidence, draft=draft)

    try:
        obj, resp = complete_json(
            prompt, system=SYSTEM, required_keys=["verdict"], temperature=0.0,
            cheap="critic" in settings.llm_cheap_nodes,
        )
    except LLMError as exc:
        # Approve on failure. The alternative -- blocking the answer because
        # the reviewer broke -- turns a critic outage into an outage of the
        # whole system, and the draft is already grounded by the writer's own
        # constraints.
        log.error("critic_failed", error=str(exc))
        return {
            "critic_verdict": "approve",
            "critique": f"critic unavailable ({exc}); draft approved unchecked",
            "llm_calls": [],
        }

    if span is not None:
        span.record_llm(resp)

    verdict = str(obj.get("verdict", "approve")).strip().lower()
    if verdict not in ("approve", "revise"):
        verdict = "approve"
    issues: List[str] = [str(i) for i in (obj.get("issues") or [])][:10]
    reason = str(obj.get("reason", ""))[:500]

    critique = None
    if verdict == "revise":
        critique = json.dumps({"issues": issues, "reason": reason},
                              ensure_ascii=False, indent=1)

    if span is not None:
        span.set(verdict=verdict, n_issues=len(issues))
    log.info("critic_verdict", verdict=verdict, issues=len(issues))

    return {
        "critic_verdict": verdict,
        "critique": critique,
        "llm_calls": [{
            "node": "critic", "model": resp.model,
            "input_tokens": resp.input_tokens, "output_tokens": resp.output_tokens,
            "thinking_tokens": resp.thinking_tokens, "cached": resp.cached,
            "latency_ms": resp.latency_ms,
        }],
    }
