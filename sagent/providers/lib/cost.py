"""Shared cost computation and per-model metadata for all providers."""

from __future__ import annotations

from dataclasses import dataclass

from sagent.custom_types import Pricing


__all__ = ["ModelProfile", "Pricing", "compute_cost"]


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelProfile:
    """Per-model metadata: token limits, pricing, tokenization density."""

    max_request_tokens: int
    """Context window size (max input tokens)."""

    max_response_tokens: int
    """Maximum tokens allowed in a response."""

    pricing: Pricing = Pricing()  # noqa: RUF009 -- frozen dataclass, no mutable default risk
    """Per-token price schedule."""

    supports_thinking: bool = True
    """Whether the model supports thinking/reasoning mode."""

    chars_per_token: float = 4.0
    """Estimator divisor for ``int(len(text) / chars_per_token)``. Default
    4.0 matches legacy behavior; override per model when the tokenizer
    diverges (e.g. opus-4-7 measures 2.83 on mixed code+JSON)."""


def compute_cost(
    pricing: Pricing,
    input_tokens: int,
    output_tokens: int,
    cache_creation: int = 0,
    cache_read: int = 0,
) -> tuple[float, float, float]:
    """Compute token costs in USD.

    ``input_tokens`` must be non-cached input only.  Callers whose API
    reports total input (Google, OpenAI) should subtract ``cache_read``
    before passing.

    Args:
      pricing: Per-token price schedule.
      input_tokens: Non-cached input token count.
      output_tokens: Output token count.
      cache_creation: Tokens written to prompt cache.
      cache_read: Tokens read from prompt cache.

    Returns:
      input_cost: Input cost in USD (including cache components).
      output_cost: Output cost in USD.
      total_cost: Sum of input and output costs.

    """
    input_cost = (
        input_tokens * pricing.request
        + cache_creation * pricing.cache_write
        + cache_read * pricing.cache_read
    ) / 1_000_000
    output_cost = output_tokens * pricing.response / 1_000_000
    return input_cost, output_cost, input_cost + output_cost
