"""Tests for the LLM provider layer. No network calls anywhere in this file.

The point of a provider layer is that everything around the model call --
caching, budgeting, retries, JSON repair -- is testable without a model. If
these tests needed an API key they would be useless in CI and would burn a
daily free-tier quota on every push.
"""
from __future__ import annotations

import json

import pytest

from agents import llm_provider as P
from scraper.config import settings


@pytest.fixture(autouse=True)
def isolated_provider(tmp_path, monkeypatch):
    """Fresh cache dir and budget per test, and never the real provider."""
    monkeypatch.setattr(settings, "llm_cache_dir", str(tmp_path / "cache"))
    monkeypatch.setattr(settings, "llm_cache_enabled", True)
    monkeypatch.setattr(settings, "llm_provider", "gemini")
    monkeypatch.setattr(settings, "gemini_model", "test-model")
    monkeypatch.setattr(settings, "llm_max_retries", 3)
    P.reset_budget(100)
    yield


def _stub(monkeypatch, text="hello", calls=None):
    """Replace the Gemini call with a counter. No SDK, no key, no network."""
    def fake(prompt, system, temperature, model, json_mode, max_output_tokens):
        if calls is not None:
            calls.append({"prompt": prompt, "system": system, "json_mode": json_mode})
        return P.LLMResponse(
            text=text, model=model, provider="gemini",
            input_tokens=11, output_tokens=7, latency_ms=12.5,
        )
    monkeypatch.setattr(P, "_gemini_complete", fake)


# ---------------------------------------------------------------------------
# Usage accounting
# ---------------------------------------------------------------------------
def test_response_carries_token_counts(monkeypatch):
    _stub(monkeypatch)
    r = P.complete("hi")
    assert (r.input_tokens, r.output_tokens, r.total_tokens) == (11, 7, 18)
    assert r.provider == "gemini" and r.model == "test-model"
    assert r.cached is False


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------
def test_second_identical_call_is_served_from_cache(monkeypatch):
    calls = []
    _stub(monkeypatch, calls=calls)

    first = P.complete("same question")
    second = P.complete("same question")

    assert len(calls) == 1, "the model was called twice for an identical prompt"
    assert second.cached is True and first.cached is False
    assert second.text == first.text


def test_a_cache_hit_costs_no_quota(monkeypatch):
    _stub(monkeypatch)
    P.complete("q")
    used_after_first = P.budget.used
    P.complete("q")
    assert P.budget.used == used_after_first, "a cache hit consumed request budget"
    assert P.budget.cache_hits == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"system": "you are a critic"},
        {"temperature": 0.7},
        {"json_mode": True},
        {"model": "other-model"},
    ],
    ids=["system", "temperature", "json_mode", "model"],
)
def test_anything_that_changes_the_answer_changes_the_cache_key(monkeypatch, kwargs):
    """A field left out of the key makes two different calls collide."""
    calls = []
    _stub(monkeypatch, calls=calls)
    P.complete("q")
    P.complete("q", **kwargs)
    assert len(calls) == 2, f"{kwargs} did not affect the cache key"


def test_use_cache_false_bypasses_the_cache(monkeypatch):
    calls = []
    _stub(monkeypatch, calls=calls)
    P.complete("q")
    P.complete("q", use_cache=False)
    assert len(calls) == 2


def test_a_corrupt_cache_entry_does_not_break_the_run(monkeypatch):
    _stub(monkeypatch, text="fresh")
    P.complete("q")
    key = P._cache_key("gemini", "test-model", "q", None, 0.0, False)
    P._cache_path(key).write_text("{not json", encoding="utf-8")
    assert P.complete("q").text == "fresh"


def test_cache_survives_a_new_process(monkeypatch):
    """The cache is on disk, so a second run of the eval suite is free."""
    _stub(monkeypatch, text="persisted")
    P.complete("q")
    P.reset_budget(100)          # as if a brand new process started
    r = P.complete("q")
    assert r.cached is True and r.text == "persisted"
    assert P.budget.used == 0


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------
def test_budget_stops_the_run_before_the_quota_does(monkeypatch):
    _stub(monkeypatch)
    P.reset_budget(2)
    P.complete("a")
    P.complete("b")
    with pytest.raises(P.BudgetExceeded) as exc:
        P.complete("c")
    assert "budget exhausted" in str(exc.value)


def test_budget_exceeded_is_not_swallowed_as_a_retryable_error(monkeypatch):
    """A silent stop yields an eval report with missing rows that still
    looks like a result, which is worse than a crash."""
    _stub(monkeypatch)
    P.reset_budget(0)
    with pytest.raises(P.BudgetExceeded):
        P.complete("a")
    assert P.budget.used == 0


def test_budget_reports_remaining(monkeypatch):
    _stub(monkeypatch)
    P.reset_budget(5)
    P.complete("a")
    assert P.budget.remaining == 4


# ---------------------------------------------------------------------------
# Retries
# ---------------------------------------------------------------------------
def test_rate_limit_is_retried_then_succeeds(monkeypatch):
    monkeypatch.setattr(P.time, "sleep", lambda s: None)   # no real waiting
    attempts = {"n": 0}

    def flaky(prompt, system, temperature, model, json_mode, max_output_tokens):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("429 RESOURCE_EXHAUSTED: rate limit exceeded")
        return P.LLMResponse(text="ok", model=model, provider="gemini")

    monkeypatch.setattr(P, "_gemini_complete", flaky)
    assert P.complete("q").text == "ok"
    assert attempts["n"] == 2


def test_a_retry_costs_a_second_request(monkeypatch):
    """Retries are not free, so they must count against the budget."""
    monkeypatch.setattr(P.time, "sleep", lambda s: None)
    attempts = {"n": 0}

    def flaky(prompt, system, temperature, model, json_mode, max_output_tokens):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("503 unavailable")
        return P.LLMResponse(text="ok", model=model, provider="gemini")

    monkeypatch.setattr(P, "_gemini_complete", flaky)
    P.reset_budget(100)
    P.complete("q")
    assert P.budget.used == 2


def test_a_permanent_error_is_not_retried(monkeypatch):
    """Retrying a bad API key just wastes time and quota."""
    attempts = {"n": 0}

    def broken(prompt, system, temperature, model, json_mode, max_output_tokens):
        attempts["n"] += 1
        raise RuntimeError("400 API_KEY_INVALID")

    monkeypatch.setattr(P, "_gemini_complete", broken)
    with pytest.raises(P.LLMError):
        P.complete("q")
    assert attempts["n"] == 1


def test_an_unknown_provider_is_rejected(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "openai")
    with pytest.raises(P.LLMError, match="unknown LLM_PROVIDER"):
        P.complete("q")


def test_a_missing_model_id_is_a_clear_error(monkeypatch):
    monkeypatch.setattr(settings, "gemini_model", "")
    monkeypatch.setattr(settings, "gemini_model_cheap", "")
    with pytest.raises(P.LLMError, match="No Gemini model configured"):
        P.resolve_model()


# ---------------------------------------------------------------------------
# JSON handling
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "raw",
    [
        '{"route": ["sql"]}',
        '```json\n{"route": ["sql"]}\n```',
        '```\n{"route": ["sql"]}\n```',
        'Here is the JSON:\n{"route": ["sql"]}\nHope that helps!',
        '  \n {"route": ["sql"]} \n ',
    ],
    ids=["bare", "fenced-json", "fenced-plain", "with-commentary", "whitespace"],
)
def test_json_is_extracted_from_the_shapes_models_actually_emit(raw):
    assert P.extract_json(raw) == {"route": ["sql"]}


def test_json_arrays_are_extracted_too():
    assert P.extract_json('talk\n[1, 2, 3]\nmore talk') == [1, 2, 3]


@pytest.mark.parametrize("raw", ["", "   ", "no json at all", "{broken"])
def test_unparseable_responses_raise(raw):
    with pytest.raises(ValueError):
        P.extract_json(raw)


def test_complete_json_returns_object_and_usage(monkeypatch):
    _stub(monkeypatch, text='{"route": ["sql"], "reason": "numeric"}')
    obj, resp = P.complete_json("q", required_keys=["route", "reason"])
    assert obj["route"] == ["sql"]
    assert resp.input_tokens == 11


def test_complete_json_repairs_once_then_succeeds(monkeypatch):
    replies = ["I think you should use SQL.", '{"route": ["sql"]}']

    def fake(prompt, system, temperature, model, json_mode, max_output_tokens):
        return P.LLMResponse(text=replies.pop(0), model=model, provider="gemini")

    monkeypatch.setattr(P, "_gemini_complete", fake)
    obj, _ = P.complete_json("q", required_keys=["route"])
    assert obj == {"route": ["sql"]}
    assert replies == []


def test_complete_json_gives_up_after_one_repair(monkeypatch):
    """Repairs cost quota, so the cap is one, not 'until it works'."""
    calls = []
    _stub(monkeypatch, text="still not json", calls=calls)
    with pytest.raises(P.LLMError, match="after one repair"):
        P.complete_json("q", required_keys=["route"])
    assert len(calls) == 2


def test_missing_required_key_triggers_repair(monkeypatch):
    replies = ['{"reason": "no route key"}', '{"route": ["rag"], "reason": "ok"}']

    def fake(prompt, system, temperature, model, json_mode, max_output_tokens):
        return P.LLMResponse(text=replies.pop(0), model=model, provider="gemini")

    monkeypatch.setattr(P, "_gemini_complete", fake)
    obj, _ = P.complete_json("q", required_keys=["route", "reason"])
    assert obj["route"] == ["rag"]


def test_json_mode_is_passed_to_the_provider(monkeypatch):
    """The API should constrain the output, rather than the prompt asking nicely."""
    calls = []
    _stub(monkeypatch, text='{"a": 1}', calls=calls)
    P.complete_json("q")
    assert calls[0]["json_mode"] is True


# ---------------------------------------------------------------------------
# Secret hygiene
# ---------------------------------------------------------------------------
def test_the_api_key_is_not_printable():
    """SecretStr so the key cannot reach a log line or a traceback."""
    assert "**" in repr(settings.gemini_api_key)
    assert settings.gemini_api_key.get_secret_value() not in repr(settings.gemini_api_key)
