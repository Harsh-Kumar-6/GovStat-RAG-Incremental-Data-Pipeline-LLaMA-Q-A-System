"""RAG agent: semantic search over the PDF text, returning citable passages.

This node makes no model call. It is retrieval only, and that is deliberate:
retrieval and generation are measured separately in Phase 3 (recall@k and MRR
for one, faithfulness for the other), and a node that did both would make it
impossible to tell which half failed. When an answer is wrong you need to know
whether the right passage was never retrieved or was retrieved and ignored.
"""
from __future__ import annotations

from typing import Any, Dict, List

from scraper.config import settings
from scraper.logger import get_logger

log = get_logger(__name__)


def run_rag_agent(question: str, k: int = 0, span: Any = None) -> Dict[str, Any]:
    """Return a partial AgentState with retrieved passages.

    `document_id` is carried on every passage. Phase 3's recall@k compares
    exactly that field against the golden dataset's relevant_doc_ids, so
    dropping it here would make retrieval unmeasurable.
    """
    k = k or settings.top_k

    try:
        # Imported lazily: sentence-transformers and faiss pull in torch, which
        # takes seconds to import and is not needed by the SQL-only path.
        from rag.indexer import get_store

        hits: List[Dict[str, Any]] = get_store().search(question, k=k)
    except FileNotFoundError as exc:
        log.error("rag_index_missing", error=str(exc))
        return {"passages": [], "rag_error": f"vector index not built: {exc}"}
    except Exception as exc:
        log.error("rag_failed", error=str(exc))
        return {"passages": [], "rag_error": f"retrieval failed: {exc}"}

    passages = [
        {
            "id": str(h.get("id", "")),
            "document_id": str(h.get("document_id", "")),
            "text": h.get("text", ""),
            "title": h.get("title", ""),
            "url": h.get("url", ""),
            "date_published": h.get("date_published"),
            "category": h.get("category"),
            "score": float(h.get("score", 0.0)),
        }
        for h in hits
    ]

    if span is not None:
        span.set(
            k=k,
            n_passages=len(passages),
            doc_ids=sorted({p["document_id"] for p in passages}),
        )
    log.info("retrieved", k=k, passages=len(passages))
    return {"passages": passages, "rag_error": None}
