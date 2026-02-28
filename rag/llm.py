"""LLaMA client via Ollama API."""
from __future__ import annotations

from typing import Dict, Generator, List

import httpx

from scraper.config import settings
from scraper.logger import get_logger

log = get_logger(__name__)

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions strictly from the provided context "
    "about Indian statistics published by MoSPI (Ministry of Statistics and Programme Implementation). "
    "If the answer is not found in the context, say: 'I don't have that in my data.' "
    "Always include bullet-point citations with the source title and URL at the end of your answer."
)


def _build_context(chunks: List[Dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(
            f"[{i}] Title: {c.get('title', 'Unknown')}\n"
            f"URL: {c.get('url', '')}\n"
            f"Date: {c.get('date_published', 'Unknown')}\n"
            f"Content: {c.get('text', '')}\n"
        )
    return "\n---\n".join(parts)


def ask_llm(question: str, chunks: List[Dict], temperature: float = 0.1) -> str:
    """Send question + context to Ollama, return full response text."""
    context = _build_context(chunks)
    user_msg = f"Context:\n{context}\n\nQuestion: {question}"

    payload = {
        "model": settings.ollama_model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        "stream": False,
        "options": {"temperature": temperature},
    }

    try:
        with httpx.Client(timeout=120) as client:
            resp = client.post(f"{settings.ollama_base_url}/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["message"]["content"]
    except Exception as exc:
        log.error("llm_error", error=str(exc))
        return f"Error communicating with LLM: {exc}"


def ask_llm_stream(question: str, chunks: List[Dict], temperature: float = 0.1) -> Generator[str, None, None]:
    """Stream response tokens from Ollama."""
    context = _build_context(chunks)
    user_msg = f"Context:\n{context}\n\nQuestion: {question}"

    payload = {
        "model": settings.ollama_model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        "stream": True,
        "options": {"temperature": temperature},
    }

    import json as _json

    try:
        with httpx.Client(timeout=120) as client:
            with client.stream("POST", f"{settings.ollama_base_url}/api/chat", json=payload) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    try:
                        data = _json.loads(line)
                        token = data.get("message", {}).get("content", "")
                        if token:
                            yield token
                    except _json.JSONDecodeError:
                        continue
    except Exception as exc:
        log.error("llm_stream_error", error=str(exc))
        yield f"\n[Error: {exc}]"