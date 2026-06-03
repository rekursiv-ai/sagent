"""Shared cost computation and per-model metadata for all providers."""

from __future__ import annotations

from dataclasses import dataclass

from sagent.types.model import Pricing


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
    """Whether the model supports thinking/reasoning mode at all."""

    readable_thinking: bool = True
    """Whether the model returns readable thinking text to the client.

    ``True`` for models that stream plaintext ``thinking_delta`` events
    (measured: opus-4-6, sonnet-4-6, the 4-5 generation). ``False`` for
    models that return a signed but empty thinking block -- the model
    reasons, but the plaintext is never delivered (measured: opus-4-8,
    opus-4-7). When ``False``, every ``-show`` thinking state is
    unsatisfiable and excluded from validity."""

    adaptive_thinking_mode: bool = True
    """Whether the model accepts ``thinking.type=adaptive``.

    ``False`` for the 4-5 generation (opus-4-5, sonnet-4-5, haiku-4-5),
    which reject ``adaptive`` with HTTP 400 ('adaptive thinking is not
    supported on this model') and require ``enabled``. When ``False``, the
    ``adaptive-*`` and ``redact-hide`` states (which request ``adaptive``)
    are excluded from validity."""

    enabled_thinking_mode: bool = True
    """Whether the model accepts ``thinking.type=enabled``.

    ``False`` for opus-4-8 / opus-4-7, which reject ``enabled`` with HTTP
    400 over the API transport (the API directs callers to ``adaptive`` +
    ``output_config.effort``). When ``False``, the ``on-*`` states (which
    request ``enabled``) are excluded from validity."""

    valid_efforts: tuple[str, ...] = ()
    """Accepted ``output_config.effort`` levels; empty when none.

    Per-model (measured via API key): opus-4-8/4-7 accept
    ``low..xhigh,max``; opus-4-6/sonnet-4-6 lack ``xhigh``; opus-4-5
    accepts ``low,medium,high``; sonnet-4-5/haiku-4-5 accept none."""

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
    *,
    fast: bool = False,
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
      fast: When True, bill non-cached request/response at the
          ``fast_request`` / ``fast_response`` rates. The caller must
          gate this on the server's authoritative speed report (e.g.
          Anthropic ``usage.speed == "fast"``).

    Returns:
      input_cost: Input cost in USD (including cache components).
      output_cost: Output cost in USD.
      total_cost: Sum of input and output costs.

    """
    # Fast mode surcharges only request/response: Anthropic's fast-mode
    # pricing table lists Input/Output rates and no separate cache rates,
    # so cache write/read stay at standard rates here.
    request_rate = pricing.request
    response_rate = pricing.response
    if fast:
        # Un-priced fast rates (``fast_*`` == 0.0) fall back to standard:
        # bill what's known, never $0.
        request_rate = pricing.fast_request or request_rate
        response_rate = pricing.fast_response or response_rate
    input_cost = (
        input_tokens * request_rate
        + cache_creation * pricing.cache_write
        + cache_read * pricing.cache_read
    ) / 1_000_000
    output_cost = output_tokens * response_rate / 1_000_000
    return input_cost, output_cost, input_cost + output_cost
