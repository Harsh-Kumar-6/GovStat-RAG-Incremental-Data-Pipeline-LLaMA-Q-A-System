"""FastAPI application for the RAG chatbot."""
from __future__ import annotations

from typing import List, Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from rag.indexer import get_store
from rag.llm import ask_llm, ask_llm_stream
from scraper.config import settings
from scraper.logger import get_logger

log = get_logger(__name__)

app = FastAPI(title="MoSPI RAG API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    question: str
    k: int = 5
    temperature: float = 0.1
    stream: bool = False
    # "agents" = supervisor -> sql/rag -> writer -> critic (the Phase 2 graph)
    # "basic"  = the original single-step RAG: retrieve, stuff, answer.
    # The basic path is kept deliberately. It is the baseline the Phase 3 eval
    # compares against, and a comparison needs the old system still running,
    # not a description of it.
    mode: Literal["agents", "basic"] = "agents"
    with_critic: bool = True


class Citation(BaseModel):
    title: str
    url: str
    date_published: Optional[str] = None
    snippet: str


class AskResponse(BaseModel):
    answer: str
    citations: List[Citation]
    # Everything below is null in basic mode, which is itself informative:
    # the old path cannot tell you why it answered the way it did.
    mode: str = "basic"
    route: Optional[List[str]] = None
    route_reason: Optional[str] = None
    agent_path: Optional[List[str]] = None
    sql_query: Optional[str] = None
    sql_row_count: Optional[int] = None
    sql_attempts: Optional[int] = None
    revisions: Optional[int] = None
    critic_verdict: Optional[str] = None
    critique: Optional[str] = None
    trace_id: Optional[str] = None
    duration_ms: Optional[float] = None
    usage: Optional[dict] = None


class IngestResponse(BaseModel):
    status: str
    vectors: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "model": settings.ollama_model}


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    if req.mode == "agents":
        return _ask_agents(req)
    return _ask_basic(req)


def _ask_agents(req: AskRequest) -> AskResponse:
    """The multi-agent path.

    Runs synchronously inside the request. Each model call takes seconds and
    there are up to five of them, so this is slow by construction -- the
    latency is the agent pattern's real cost, and hiding it behind a
    background task would only hide it from the person measuring it.
    """
    from agents.graph import ask as agent_ask

    result = agent_ask(req.question, with_critic=req.with_critic)

    citations = [
        Citation(
            title=c.get("title", "") or "Unknown",
            url=c.get("url", "") or "",
            date_published=None,
            snippet=f"cited via {c.get('source', '')}",
        )
        for c in (result.get("citations") or [])
    ]
    return AskResponse(
        answer=result.get("final_answer") or "",
        citations=citations,
        mode="agents",
        route=result.get("route"),
        route_reason=result.get("route_reason"),
        agent_path=result.get("agent_path"),
        sql_query=result.get("sql_query"),
        sql_row_count=len(result.get("sql_result") or []) or None,
        sql_attempts=len(result.get("sql_attempts") or []) or None,
        revisions=result.get("revisions"),
        critic_verdict=result.get("critic_verdict"),
        critique=result.get("critique"),
        trace_id=result.get("trace_id"),
        duration_ms=result.get("duration_ms"),
        usage=result.get("usage"),
    )


def _ask_basic(req: AskRequest):
    """The original single-step RAG path, unchanged, as the eval baseline."""
    store = get_store()
    try:
        store.load()
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Vector index not built yet. Call POST /ingest first.")

    chunks = store.search(req.question, k=req.k)
    if not chunks:
        return AskResponse(answer="I don't have that in my data.", citations=[])

    if req.stream:
        # For streaming, return SSE
        def generate():
            for token in ask_llm_stream(req.question, chunks, req.temperature):
                yield f"data: {token}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(generate(), media_type="text/event-stream")

    answer = ask_llm(req.question, chunks, req.temperature)
    citations = [
        Citation(
            title=c.get("title", "Unknown"),
            url=c.get("url", ""),
            date_published=c.get("date_published"),
            snippet=c.get("text", "")[:300],
        )
        for c in chunks
    ]
    return AskResponse(answer=answer, citations=citations, mode="basic")


@app.post("/ingest", response_model=IngestResponse)
async def ingest():
    store = get_store()
    try:
        store.build()
        return IngestResponse(status="ok", vectors=store.index.ntotal)
    except Exception as exc:
        log.error("ingest_error", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))