"""FastAPI application for the RAG chatbot."""
from __future__ import annotations

from typing import List, Optional

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


class Citation(BaseModel):
    title: str
    url: str
    date_published: Optional[str] = None
    snippet: str


class AskResponse(BaseModel):
    answer: str
    citations: List[Citation]


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
    return AskResponse(answer=answer, citations=citations)


@app.post("/ingest", response_model=IngestResponse)
async def ingest():
    store = get_store()
    try:
        store.build()
        return IngestResponse(status="ok", vectors=store.index.ntotal)
    except Exception as exc:
        log.error("ingest_error", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))