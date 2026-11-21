"""Token prices, for cost per request.

Why this file contains no numbers
---------------------------------
Cost per request is a resume number. Inventing a price makes that number
false, and model prices change often enough that anything hardcoded here
would be wrong within months. So prices live in `tracing/prices.json`, which
you fill in from the provider's own pricing page, and an unpriced model
reports `cost_usd = None` rather than a guess.

`None` is not a gap in the implementation -- it is the honest answer to "what
did this cost?" when the price is unknown. A free-tier call genuinely costs
0, which is why `list price` is tracked separately: the resume claim is
"measured cost at list price", not "I paid this".
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

from scraper.logger import get_logger

log = get_logger(__name__)

_PRICES_FILE = Path(__file__).with_name("prices.json")
_cache: Optional[Dict[str, Dict[str, float]]] = None


def load_prices() -> Dict[str, Dict[str, float]]:
    """Read prices.json: {model: {"input_per_1m": x, "output_per_1m": y}}."""
    global _cache
    if _cache is not None:
        return _cache
    if not _PRICES_FILE.exists():
        _cache = {}
        return _cache
    try:
        raw = json.loads(_PRICES_FILE.read_text(encoding="utf-8"))
        _cache = {k: v for k, v in raw.items() if not k.startswith("_")}
    except Exception as exc:
        log.warning("prices_unreadable", error=str(exc))
        _cache = {}
    return _cache


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> Optional[float]:
    """Cost of one call at list price, or None if the model has no price.

    `output_tokens` should include thinking tokens: current reasoning models
    bill them as output, and they routinely outnumber the visible reply.
    """
    prices = load_prices()
    entry = prices.get(model) or prices.get(model.split("/")[-1])
    if not entry:
        return None
    try:
        return (
            input_tokens / 1_000_000 * float(entry["input_per_1m"])
            + output_tokens / 1_000_000 * float(entry["output_per_1m"])
        )
    except (KeyError, TypeError, ValueError):
        return None


def is_priced(model: str) -> bool:
    return cost_usd(model, 1, 1) is not None
