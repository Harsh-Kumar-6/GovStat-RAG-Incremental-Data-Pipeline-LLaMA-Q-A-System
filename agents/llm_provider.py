"""One LLM interface for every agent, over Gemini or Ollama.

Why a provider layer at all
---------------------------
Three things have to be true of every model call in this project, and none of
them belong in the agents themselves:

1. **Token counts come back with the text.** Phase 3 measures cost per request
   and tokens/sec. A provider that returns only a string makes that impossible
   to compute later, so usage is part of the response type, not an optional
   extra.
2. **Calls are budgeted.** The graph makes 4-6 model calls per question. A
   45-question eval suite is therefore 200-270 calls, which is at or past a
   free-tier *daily* quota. An unbounded loop does not fail loudly, it fails on
   request 21 of a 180-request run, halfway through, with partial results.
3. **Identical calls are free.** Re-running an eval must not re-pay for
   answers already computed, or the suite becomes something you run once.

Determinism note: the cache is keyed partly on temperature. At temperature 0
caching a response is close to free of consequence. Above 0 the model is
*supposed* to vary, and caching deliberately suppresses that variation -- which
is what you want for reproducible evals and not what you want when sampling
several answers on purpose. `use_cache=False` opts out per call.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from scraper.config import settings
from scraper.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Types and errors
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LLMResponse:
    """What every provider returns. Agents depend on this, not on an SDK."""

    text: str
    model: str
    provider: str
    input_tokens: int = 0
    output_tokens: int = 0
    # Reasoning ("thinking") tokens. Billed as output on current Gemini models
    # but reported separately, and they dominate: a one-word reply can cost
    # hundreds of thinking tokens. Leaving them out understates cost per
    # request by an order of magnitude, which is a Phase 3 metric.
    thinking_tokens: int = 0
    latency_ms: float = 0.0
    # True when served from disk: it cost no quota and no wall-clock time, so
    # perf metrics must be able to exclude it.
    cached: bool = False
    finish_reason: Optional[str] = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens + self.thinking_tokens

    @property
    def billed_output_tokens(self) -> int:
        """What the provider charges as output: visible text plus thinking."""
        return self.output_tokens + self.thinking_tokens


class LLMError(RuntimeError):
    """A model call failed in a way retrying will not fix."""


class BudgetExceeded(LLMError):
    """The per-run request ceiling was reached.

    Deliberately an exception and not a warning: a silent stop produces an
    eval report with missing rows that still looks like a result.
    """


class RateLimited(LLMError):
    """HTTP 429. Retried with backoff before this ever surfaces."""


# ---------------------------------------------------------------------------
# Request budget
# ---------------------------------------------------------------------------
class RequestBudget:
    """Counts real (uncached) model calls and refuses to exceed a ceiling."""

    def __init__(self, limit: Optional[int] = None) -> None:
        self.limit = settings.llm_max_requests_per_run if limit is None else limit
        self.used = 0
        self.cache_hits = 0

    def spend(self, n: int = 1) -> None:
        if self.used + n > self.limit:
            raise BudgetExceeded(
                f"request budget exhausted: {self.used}/{self.limit} used. "
                f"Raise LLM_MAX_REQUESTS_PER_RUN, or enable the cache."
            )
        self.used += n

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.used)

    def summary(self) -> Dict[str, int]:
        return {
            "requests_used": self.used,
            "requests_limit": self.limit,
            "cache_hits": self.cache_hits,
        }


# Module-level budget: one process run, one ceiling.
budget = RequestBudget()


def reset_budget(limit: Optional[int] = None) -> RequestBudget:
    """Start a fresh count. Called by the eval runner and by tests."""
    global budget
    budget = RequestBudget(limit)
    return budget


# ---------------------------------------------------------------------------
# Disk cache
# ---------------------------------------------------------------------------
def _cache_key(
    provider: str,
    model: str,
    prompt: str,
    system: Optional[str],
    temperature: float,
    json_mode: bool,
) -> str:
    """Hash everything that can change the answer -- and nothing else.

    Leaving a field out of the key means two genuinely different calls collide
    and the second silently receives the first one's answer. `system` and
    `json_mode` are easy to forget and both change the output completely.
    """
    payload = json.dumps(
        {
            "provider": provider,
            "model": model,
            "prompt": prompt,
            "system": system or "",
            "temperature": round(float(temperature), 4),
            "json_mode": bool(json_mode),
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cache_path(key: str) -> Path:
    # Two-character shard: a flat directory with tens of thousands of files
    # gets slow to list and unpleasant to inspect by hand.
    d = Path(settings.llm_cache_dir) / key[:2]
    return d / f"{key}.json"


def _cache_read(key: str) -> Optional[LLMResponse]:
    path = _cache_path(key)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return LLMResponse(**{**data, "cached": True})
    except Exception as exc:
        # A corrupt cache entry must never break a run.
        log.warning("llm_cache_unreadable", path=str(path), error=str(exc))
        return None


def _cache_write(key: str, resp: LLMResponse) -> None:
    path = _cache_path(key)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(resp)
        data["cached"] = False  # store as fetched; the reader marks it cached
        # Write to a temp file and rename: an interrupted write would otherwise
        # leave a half-written JSON file that every later run has to skip.
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)
    except Exception as exc:
        log.warning("llm_cache_unwritable", path=str(path), error=str(exc))


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------
_gemini_client: Any = None


def _get_gemini_client() -> Any:
    global _gemini_client
    if _gemini_client is None:
        key = settings.gemini_api_key.get_secret_value()
        if not key:
            raise LLMError(
                "GEMINI_API_KEY is empty. Put the key in .env (gitignored); "
                "see .env.example."
            )
        try:
            from google import genai
        except ImportError as exc:
            raise LLMError("google-genai is not installed: pip install google-genai") from exc
        _gemini_client = genai.Client(api_key=key)
    return _gemini_client


def list_models() -> List[Dict[str, Any]]:
    """Ask the API which models this key can actually reach.

    Worth one request: model IDs churn, published free-tier limits are stale
    or absent, and a hardcoded ID fails at the worst moment. This is ground
    truth for *this* account.
    """
    client = _get_gemini_client()
    budget.spend()
    out: List[Dict[str, Any]] = []
    for m in client.models.list():
        actions = list(getattr(m, "supported_actions", None) or [])
        out.append(
            {
                "name": getattr(m, "name", "") or "",
                "display_name": getattr(m, "display_name", "") or "",
                "input_token_limit": getattr(m, "input_token_limit", None),
                "output_token_limit": getattr(m, "output_token_limit", None),
                "supported_actions": actions,
            }
        )
    return out


def _gemini_complete(
    prompt: str,
    system: Optional[str],
    temperature: float,
    model: str,
    json_mode: bool,
    max_output_tokens: Optional[int],
) -> LLMResponse:
    client = _get_gemini_client()

    config: Dict[str, Any] = {"temperature": temperature}
    if system:
        config["system_instruction"] = system
    if json_mode:
        # Ask the API itself to constrain the output. Far more reliable than
        # asking politely in the prompt and parsing whatever arrives.
        config["response_mime_type"] = "application/json"
    if max_output_tokens:
        config["max_output_tokens"] = max_output_tokens

    started = time.perf_counter()
    resp = client.models.generate_content(model=model, contents=prompt, config=config)
    latency_ms = (time.perf_counter() - started) * 1000.0

    usage = getattr(resp, "usage_metadata", None)
    in_tok = int(getattr(usage, "prompt_token_count", 0) or 0)
    out_tok = int(getattr(usage, "candidates_token_count", 0) or 0)
    think_tok = int(getattr(usage, "thoughts_token_count", 0) or 0)

    finish = None
    candidates = getattr(resp, "candidates", None) or []
    if candidates:
        finish = str(getattr(candidates[0], "finish_reason", "") or "") or None

    text = getattr(resp, "text", None) or ""
    if not text and candidates:
        # Safety blocks and token-limit stops return candidates with no text.
        raise LLMError(f"gemini returned no text (finish_reason={finish})")

    return LLMResponse(
        text=text,
        model=model,
        provider="gemini",
        input_tokens=in_tok,
        output_tokens=out_tok,
        thinking_tokens=think_tok,
        latency_ms=latency_ms,
        finish_reason=finish,
    )


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------
def _ollama_complete(
    prompt: str,
    system: Optional[str],
    temperature: float,
    model: str,
    json_mode: bool,
    max_output_tokens: Optional[int],
) -> LLMResponse:
    import httpx

    messages: List[Dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature},
    }
    if json_mode:
        payload["format"] = "json"
    if max_output_tokens:
        payload["options"]["num_predict"] = max_output_tokens

    started = time.perf_counter()
    with httpx.Client(timeout=settings.llm_timeout) as client:
        r = client.post(f"{settings.ollama_base_url}/api/chat", json=payload)
        r.raise_for_status()
        data = r.json()
    latency_ms = (time.perf_counter() - started) * 1000.0

    return LLMResponse(
        text=(data.get("message") or {}).get("content", "") or "",
        model=model,
        provider="ollama",
        # Ollama names these differently from every hosted API; normalising
        # here is the whole point of the provider layer.
        input_tokens=int(data.get("prompt_eval_count", 0) or 0),
        output_tokens=int(data.get("eval_count", 0) or 0),
        latency_ms=latency_ms,
        finish_reason=data.get("done_reason"),
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
_RETRYABLE = ("429", "rate limit", "resource_exhausted", "500", "502", "503", "504",
              "unavailable", "deadline", "timeout")


def _is_retryable(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(tok in msg for tok in _RETRYABLE)


def resolve_model(cheap: bool = False) -> str:
    """Pick the model for this call.

    `cheap=True` asks for the high-quota, low-latency model used by the
    short structured calls (routing, verdicts). It falls back to the main
    model when none is configured, so a half-filled .env degrades to
    "works, slower" rather than to an error.
    """
    name = settings.gemini_model_cheap if cheap else settings.gemini_model
    if settings.llm_provider == "ollama":
        return settings.ollama_model
    if not name:
        name = settings.gemini_model or settings.gemini_model_cheap
    if not name:
        raise LLMError(
            "No Gemini model configured. Run `make models` to list what this "
            "key can reach, then set GEMINI_MODEL in .env."
        )
    return name


def complete(
    prompt: str,
    *,
    system: Optional[str] = None,
    temperature: float = 0.0,
    model: Optional[str] = None,
    cheap: bool = False,
    json_mode: bool = False,
    max_output_tokens: Optional[int] = None,
    use_cache: Optional[bool] = None,
) -> LLMResponse:
    """Run one model call. The only way agents talk to an LLM.

    Order of operations matters: cache first, then budget, then the call. A
    cache hit must not consume quota, so the budget check cannot come first.
    """
    provider = settings.llm_provider.strip().lower()
    model = model or resolve_model(cheap=cheap)
    caching = settings.llm_cache_enabled if use_cache is None else use_cache

    key = _cache_key(provider, model, prompt, system, temperature, json_mode)
    if caching:
        hit = _cache_read(key)
        if hit is not None:
            budget.cache_hits += 1
            log.info("llm_cache_hit", model=model, key=key[:12])
            return hit

    if provider == "gemini":
        fn = _gemini_complete
    elif provider == "ollama":
        fn = _ollama_complete
    else:
        raise LLMError(f"unknown LLM_PROVIDER {provider!r}; expected gemini or ollama")

    last: Optional[Exception] = None
    for attempt in range(1, max(1, settings.llm_max_retries) + 1):
        try:
            budget.spend()
            resp = fn(prompt, system, temperature, model, json_mode, max_output_tokens)
        except BudgetExceeded:
            raise
        except Exception as exc:  # noqa: BLE001 - classified below
            last = exc
            if not _is_retryable(exc) or attempt >= settings.llm_max_retries:
                log.error("llm_call_failed", model=model, attempt=attempt, error=str(exc))
                raise LLMError(f"{provider} call failed: {exc}") from exc
            # Exponential backoff with jitter. Without jitter, concurrent
            # retries re-collide at exactly the same moment.
            delay = min(30.0, 2.0 ** attempt) + random.uniform(0, 0.5)
            log.warning("llm_retrying", model=model, attempt=attempt,
                        sleep_s=round(delay, 2), error=str(exc)[:200])
            time.sleep(delay)
            continue

        if caching:
            _cache_write(key, resp)
        log.info(
            "llm_call",
            model=model,
            provider=provider,
            in_tokens=resp.input_tokens,
            out_tokens=resp.output_tokens,
            thinking_tokens=resp.thinking_tokens,
            latency_ms=round(resp.latency_ms, 1),
            budget_used=budget.used,
            budget_limit=budget.limit,
        )
        return resp

    raise LLMError(f"{provider} call failed after retries: {last}")


# ---------------------------------------------------------------------------
# Structured output
# ---------------------------------------------------------------------------
_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.S)


def extract_json(text: str) -> Any:
    """Parse JSON out of a model response.

    Models wrap JSON in markdown fences, prepend "Here is the JSON:", and
    append commentary -- even when asked not to. This is pure string handling
    with no model call, so it is cheap to be forgiving here and it saves a
    retry that would cost real quota.
    """
    if not text or not text.strip():
        raise ValueError("empty response")

    candidate = text.strip()
    m = _FENCE.match(candidate)
    if m:
        candidate = m.group(1).strip()

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # Fall back to the outermost {...} or [...] span.
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = candidate.find(opener), candidate.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(candidate[start : end + 1])
            except json.JSONDecodeError:
                continue

    raise ValueError(f"no JSON found in response: {text[:200]!r}")


def complete_json(
    prompt: str,
    *,
    system: Optional[str] = None,
    required_keys: Optional[List[str]] = None,
    temperature: float = 0.0,
    model: Optional[str] = None,
    cheap: bool = False,
    max_output_tokens: Optional[int] = None,
    use_cache: Optional[bool] = None,
    repair: bool = True,
) -> tuple[Any, LLMResponse]:
    """Call the model and return parsed JSON plus the raw response.

    Returns both, because the caller needs the parsed object to act on and the
    token counts to record. One repair attempt is allowed: it costs a request,
    so it is capped at one and the error text is fed back as the observation.
    """
    resp = complete(
        prompt,
        system=system,
        temperature=temperature,
        model=model,
        cheap=cheap,
        json_mode=True,
        max_output_tokens=max_output_tokens,
        use_cache=use_cache,
    )

    def _validate(obj: Any) -> Any:
        if required_keys:
            if not isinstance(obj, dict):
                raise ValueError(f"expected a JSON object, got {type(obj).__name__}")
            missing = [k for k in required_keys if k not in obj]
            if missing:
                raise ValueError(f"missing required keys: {missing}")
        return obj

    try:
        return _validate(extract_json(resp.text)), resp
    except ValueError as exc:
        if not repair:
            raise LLMError(f"invalid JSON from model: {exc}") from exc

    log.warning("llm_json_repair", error="first attempt unparseable")
    repair_prompt = (
        f"{prompt}\n\n"
        f"Your previous reply could not be parsed: {resp.text[:500]}\n"
        f"It failed because the output was not valid JSON with the required keys"
        f"{' ' + str(required_keys) if required_keys else ''}.\n"
        "Reply with ONLY a valid JSON object. No markdown fences, no commentary."
    )
    resp2 = complete(
        repair_prompt,
        system=system,
        temperature=temperature,
        model=model,
        cheap=cheap,
        json_mode=True,
        max_output_tokens=max_output_tokens,
        use_cache=use_cache,
    )
    try:
        return _validate(extract_json(resp2.text)), resp2
    except ValueError as exc:
        raise LLMError(f"invalid JSON after one repair attempt: {exc}") from exc
