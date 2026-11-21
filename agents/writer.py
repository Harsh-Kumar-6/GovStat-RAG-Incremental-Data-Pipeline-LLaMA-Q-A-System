"""Writer: composes the answer from evidence, and only from evidence.

Grounding
---------
The writer sees the question, the SQL rows, and the retrieved passages. It does
not see the internet and it is told, explicitly, that its own knowledge of
Indian statistics is not admissible. Everything numeric must be traceable to a
row or a passage.

That constraint is what the faithfulness metric in Phase 3 measures, and it is
why the refusal path lives here rather than in the prompt alone: if there is no
evidence at all, no prompt wording reliably stops a model from producing a
plausible number. The code refuses before the model is ever called.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from agents.llm_provider import LLMError, complete
from scraper.logger import get_logger

log = get_logger(__name__)

MAX_PASSAGE_CHARS = 1200
REFUSAL = (
    "I can't answer that from the MoSPI data I have. "
    "{detail}"
)

SYSTEM = (
    "You are a careful analyst of Indian government statistics. You answer "
    "only from the evidence provided. You never use outside knowledge, and "
    "you never estimate, extrapolate, or round a figure that is not in the "
    "evidence."
)

PROMPT = """QUESTION: {question}

{evidence}

Write the answer.

RULES
- Every number must appear in the evidence above. Do not compute a figure that
  is not there, except a difference or a ratio between two numbers that ARE
  there -- and say so when you do.
- Inflation figures are already percentages. Do not convert them.
- Cite sources as [1], [2] matching the numbered evidence items.
- If the evidence does not answer the question, say so plainly instead of
  answering a nearby question you can answer.
- Be concise: 2-5 sentences unless the question asks for a list.
{revision_note}"""


def _format_sql_evidence(rows: List[Dict[str, Any]], query: Optional[str]) -> str:
    if not rows:
        return ""
    # JSON rather than prose: it is unambiguous, and the model is less likely
    # to misread a column than a sentence describing one.
    body = json.dumps(rows[:60], ensure_ascii=False, indent=1, default=str)
    return f"SQL RESULT ({len(rows)} rows)\nquery: {query}\n{body}\n"


def _format_passages(passages: List[Dict[str, Any]]) -> str:
    if not passages:
        return ""
    out = ["RETRIEVED PASSAGES"]
    for i, p in enumerate(passages, 1):
        out.append(
            f"[{i}] {p.get('title', 'Untitled')} "
            f"({p.get('date_published') or 'undated'})\n"
            f"    url: {p.get('url', '')}\n"
            f"    {p.get('text', '')[:MAX_PASSAGE_CHARS]}"
        )
    return "\n".join(out) + "\n"


def build_citations(
    passages: List[Dict[str, Any]],
    sql_rows: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, str]]:
    """Collect the sources behind an answer, de-duplicated by document.

    SQL rows carry source_document_id precisely so a numeric answer is as
    citable as a textual one. A figure without a source is the failure mode
    this whole project exists to avoid.
    """
    seen: Dict[str, Dict[str, str]] = {}
    for p in passages or []:
        doc_id = p.get("document_id") or p.get("url", "")
        if doc_id and doc_id not in seen:
            seen[doc_id] = {
                "document_id": str(doc_id),
                "title": p.get("title", ""),
                "url": p.get("url", ""),
                "source": "rag",
            }
    for r in sql_rows or []:
        doc_id = str(r.get("source_document_id") or "")
        if doc_id and doc_id not in seen:
            seen[doc_id] = {
                "document_id": doc_id,
                "title": str(r.get("title", "")),
                "url": str(r.get("url", "")),
                "source": "sql",
            }
    return list(seen.values())


def run_writer(
    question: str,
    sql_result: Optional[List[Dict[str, Any]]] = None,
    sql_query: Optional[str] = None,
    passages: Optional[List[Dict[str, Any]]] = None,
    critique: Optional[str] = None,
    revisions: int = 0,
    route: Optional[List[str]] = None,
    sql_error: Optional[str] = None,
    span: Any = None,
) -> Dict[str, Any]:
    """Compose a grounded answer, or refuse. Returns a partial AgentState."""
    sql_result = sql_result or []
    passages = passages or []

    # --- refuse before spending a request -------------------------------
    if not sql_result and not passages:
        if route == []:
            detail = "That question is outside what these statistics cover."
        elif sql_error:
            detail = f"The database query did not succeed ({sql_error})."
        else:
            detail = "No matching figures or passages were found."
        answer = REFUSAL.format(detail=detail)
        log.info("writer_refused", reason=detail[:100])
        return {
            "draft_answer": answer,
            "final_answer": answer,
            "citations": [],
            "revisions": revisions,
            "llm_calls": [],
        }

    evidence = "\n".join(x for x in (
        _format_sql_evidence(sql_result, sql_query),
        _format_passages(passages),
    ) if x)

    revision_note = ""
    if critique:
        revision_note = (
            f"\nA REVIEWER REJECTED YOUR PREVIOUS ANSWER:\n{critique}\n"
            "Fix exactly those problems. Remove any claim you cannot support "
            "from the evidence rather than rewording it."
        )

    prompt = PROMPT.format(
        question=question, evidence=evidence, revision_note=revision_note
    )

    try:
        resp = complete(prompt, system=SYSTEM, temperature=0.1)
    except LLMError as exc:
        log.error("writer_failed", error=str(exc))
        msg = f"Could not compose an answer: {exc}"
        return {"draft_answer": msg, "final_answer": msg, "citations": [],
                "revisions": revisions, "llm_calls": []}

    if span is not None:
        span.record_llm(resp)

    return {
        "draft_answer": resp.text.strip(),
        "citations": build_citations(passages, sql_result),
        "revisions": revisions + 1,
        "llm_calls": [{
            "node": "writer", "model": resp.model,
            "input_tokens": resp.input_tokens, "output_tokens": resp.output_tokens,
            "thinking_tokens": resp.thinking_tokens, "cached": resp.cached,
            "latency_ms": resp.latency_ms,
        }],
    }
