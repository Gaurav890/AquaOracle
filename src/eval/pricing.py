"""Static per-model $/1K-token pricing for the budget dimension's cost
estimate.

These are approximate, point-in-time figures — not a billing source of
truth. Spot-check against the provider's current pricing page before
relying on this for real budget decisions; token/latency numbers (tracked
alongside this in src/eval/service.py) are the durable ground truth and
never go stale the way a hardcoded price table does.
"""

from typing import Optional, Tuple

# (provider, model) -> {"prompt": $ per 1K prompt tokens, "completion": $ per 1K completion tokens}
_PRICING = {
    ("openai", "gpt-4o-mini"): {"prompt": 0.00015, "completion": 0.0006},
    ("openai", "gpt-4o"): {"prompt": 0.005, "completion": 0.015},
    ("anthropic", "claude-sonnet-5"): {"prompt": 0.003, "completion": 0.015},
    ("anthropic", "claude-opus-5"): {"prompt": 0.015, "completion": 0.075},
    ("anthropic", "claude-haiku-4-5"): {"prompt": 0.001, "completion": 0.005},
}


def estimate_cost_usd(
    provider: str, model: Optional[str], prompt_tokens: Optional[int], completion_tokens: Optional[int]
) -> Optional[float]:
    """None (shown as "—" in the UI) for local Ollama, an unknown model, or
    when token counts aren't available — never a fabricated number."""
    if provider == "ollama":
        return 0.0
    if not model or prompt_tokens is None or completion_tokens is None:
        return None

    rates = _PRICING.get((provider, model))
    if rates is None:
        return None

    return round((prompt_tokens / 1000) * rates["prompt"] + (completion_tokens / 1000) * rates["completion"], 6)


def known_models() -> Tuple[Tuple[str, str], ...]:
    return tuple(_PRICING.keys())
