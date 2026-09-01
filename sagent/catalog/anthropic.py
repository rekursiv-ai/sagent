"""Anthropic model catalog, expressed as ``ModelCapability`` rows.

Sources:
  - ModelLimits & pricing: https://docs.anthropic.com/en/docs/about-claude/models
  - Fast mode: https://docs.anthropic.com/en/docs/build-with-claude/fast-mode
  - Vision: https://platform.claude.com/docs/en/build-with-claude/vision
  - Request size: https://platform.claude.com/docs/en/api/overview

``MODELS`` is the API-key view. Other transports narrow it with ``&``
(see ``CLI``), which can only remove.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from sagent.catalog.capability import (
    ModelCapability,
    ModelLimits,
)
from sagent.catalog.cost import (
    PriceCatalog,
    PriceCatalogProduct,
    TokenPrice,
)


__all__ = ["API", "CLI", "MODELS", "SUBSCRIPTION"]


def _limits(*, request: int, response: int = 128_000, edge: int) -> ModelLimits:
    """Per-image hard limit 5 MB; per-request hard limit 32 MB.

    ``edge`` is the model's NATIVE resolution -- the long edge above
    which the server downscales for free, so pre-resizing there caps
    wire bytes and token cost without losing fidelity the model kept.
    """
    return ModelLimits(
        max_request_tokens=request,
        max_response_tokens=response,
        max_request_bytes=32 * 1024 * 1024,
        max_image_edge_px=edge,
        max_image_bytes=5 * 1024 * 1024,
    )


def _windowed(*, edge: int, default: int = 200_000) -> Mapping[str, ModelLimits]:
    """Both context tags, keyed by the id suffix that selects them.

    The 5 generation is 1M-native, so ``+1m`` is accepted but needs no
    beta header; ``default=1_000_000`` makes the untagged id 1M too.
    """
    return MappingProxyType(
        {
            "": _limits(request=default, edge=edge),
            "+1m": _limits(request=1_000_000, edge=edge),
        }
    )


def _prices(
    *,
    request: float,
    response: float,
    fast_multiple: float = 0.0,
) -> PriceCatalog:
    rows = {
        PriceCatalogProduct(): TokenPrice(
            request=request,
            response=response,
            cache_write=request * 1.25,
            cache_read=request * 0.1,
        )
    }
    if fast_multiple:
        # Fast mode surcharges request/response only: Anthropic's fast-mode
        # table lists no separate cache rates.
        rows[PriceCatalogProduct(fast=True)] = TokenPrice(
            request=request * fast_multiple,
            response=response * fast_multiple,
            cache_write=request * 1.25,
            cache_read=request * 0.1,
        )
    return PriceCatalog(rows)


# Thinking capability measured against the live API (Jun 2026). ``auto`` is
# ``thinking.type=adaptive`` (no budget); ``fixed`` is ``enabled`` +
# ``budget_tokens``. A model missing ``text`` reasons but returns a
# signed-but-empty block -- the plaintext is never delivered.
#
# ``chars_per_token`` measured via ``messages.count_tokens`` against the
# live API on 347k chars of real session text (2026-08-22). TWO tokenizer
# generations, not three: the claude-5 / opus-4-8 / opus-4-7 family
# returns 2.38, and everything 4-6 and older returns 3.12 -- sonnet-4-5
# and haiku-4-5 report counts byte-identical to opus-4-6, so the 4.83
# tier they used to carry did not exist and under-counted their tokens
# by 55%. Independently corroborated by TextKit's 2026 tokenizer survey,
# which finds the same two counting regimes.
MODELS: Mapping[str, ModelCapability] = MappingProxyType(
    {
        "claude-fable-5": ModelCapability(
            model_id="claude-fable-5",
            context_limits=_windowed(edge=2576, default=1_000_000),
            prices=_prices(request=10.0, response=50.0),
            chars_per_token=2.38,
            supported_thinking_efforts=MappingProxyType(
                {
                    "low": "low",
                    "medium": "medium",
                    "high": "high",
                    "xhigh": "xhigh",
                    "max": "max",
                }
            ),
            supported_thinking_budgets=frozenset({"auto"}),
            supported_thinking_outputs=frozenset({"redacted"}),
            fast=False,
        ),
        "claude-opus-5": ModelCapability(
            model_id="claude-opus-5",
            context_limits=_windowed(edge=2576, default=1_000_000),
            prices=_prices(request=5.0, response=25.0, fast_multiple=2.0),
            chars_per_token=2.38,
            supported_thinking_efforts=MappingProxyType(
                {
                    "low": "low",
                    "medium": "medium",
                    "high": "high",
                    "xhigh": "xhigh",
                    "max": "max",
                }
            ),
            supported_thinking_budgets=frozenset({"auto"}),
            supported_thinking_outputs=frozenset({"redacted"}),
        ),
        "claude-opus-4-8": ModelCapability(
            model_id="claude-opus-4-8",
            context_limits=_windowed(edge=2576),
            prices=_prices(request=5.0, response=25.0, fast_multiple=2.0),
            chars_per_token=2.38,
            supported_thinking_efforts=MappingProxyType(
                {
                    "low": "low",
                    "medium": "medium",
                    "high": "high",
                    "xhigh": "xhigh",
                    "max": "max",
                }
            ),
            supported_thinking_budgets=frozenset({"auto"}),
            supported_thinking_outputs=frozenset({"redacted"}),
        ),
        "claude-opus-4-7": ModelCapability(
            model_id="claude-opus-4-7",
            context_limits=_windowed(edge=2576),
            # Fast mode was removed from 4-7 on 2026-07-24; the API now
            # rejects ``speed="fast"`` rather than serving it.
            prices=_prices(request=5.0, response=25.0),
            chars_per_token=2.38,
            supported_thinking_efforts=MappingProxyType(
                {
                    "low": "low",
                    "medium": "medium",
                    "high": "high",
                    "xhigh": "xhigh",
                    "max": "max",
                }
            ),
            supported_thinking_budgets=frozenset({"auto"}),
            supported_thinking_outputs=frozenset({"redacted"}),
            fast=False,
        ),
        "claude-opus-4-6": ModelCapability(
            model_id="claude-opus-4-6",
            context_limits=_windowed(edge=1568),
            # 4-6 never shipped fast mode: a ``speed="fast"`` request runs at
            # standard speed and bills standard, reporting usage.speed
            # "standard". A fast price row would overstate the cost.
            prices=_prices(request=5.0, response=25.0),
            chars_per_token=3.12,
            supported_thinking_efforts=MappingProxyType(
                {"low": "low", "medium": "medium", "high": "high", "max": "max"}
            ),
            supported_thinking_budgets=frozenset({"auto", "fixed"}),
            supported_thinking_outputs=frozenset({"text", "redacted"}),
        ),
        "claude-opus-4-5": ModelCapability(
            model_id="claude-opus-4-5",
            context_limits=_windowed(edge=1568),
            prices=_prices(request=5.0, response=25.0),
            chars_per_token=3.12,
            supported_thinking_efforts=MappingProxyType(
                {"low": "low", "medium": "medium", "high": "high"}
            ),
            supported_thinking_budgets=frozenset({"fixed"}),
            supported_thinking_outputs=frozenset({"text", "redacted"}),
            fast=False,
        ),
        "claude-sonnet-5": ModelCapability(
            model_id="claude-sonnet-5",
            context_limits=_windowed(edge=2576, default=1_000_000),
            prices=_prices(request=3.0, response=15.0),
            chars_per_token=2.38,
            supported_thinking_efforts=MappingProxyType(
                {
                    "low": "low",
                    "medium": "medium",
                    "high": "high",
                    "xhigh": "xhigh",
                    "max": "max",
                }
            ),
            supported_thinking_budgets=frozenset({"auto"}),
            supported_thinking_outputs=frozenset({"redacted"}),
            fast=False,
        ),
        "claude-sonnet-4-6": ModelCapability(
            model_id="claude-sonnet-4-6",
            context_limits=_windowed(edge=1568),
            prices=_prices(request=3.0, response=15.0),
            chars_per_token=3.12,
            supported_thinking_efforts=MappingProxyType(
                {"low": "low", "medium": "medium", "high": "high", "max": "max"}
            ),
            supported_thinking_budgets=frozenset({"auto", "fixed"}),
            supported_thinking_outputs=frozenset({"text", "redacted"}),
            fast=False,
        ),
        "claude-sonnet-4-5": ModelCapability(
            model_id="claude-sonnet-4-5",
            context_limits=_windowed(edge=1568),
            prices=_prices(request=3.0, response=15.0),
            chars_per_token=3.12,
            supported_thinking_efforts=MappingProxyType({}),
            supported_thinking_budgets=frozenset({"fixed"}),
            supported_thinking_outputs=frozenset({"text", "redacted"}),
            fast=False,
        ),
        "claude-haiku-4-5": ModelCapability(
            model_id="claude-haiku-4-5",
            context_limits=_limits(request=200_000, response=64_000, edge=1568),
            prices=_prices(request=1.0, response=5.0),
            chars_per_token=3.12,
            supported_thinking_efforts=MappingProxyType({}),
            supported_thinking_budgets=frozenset({"fixed"}),
            supported_thinking_outputs=frozenset({"text", "redacted"}),
            fast=False,
        ),
    }
)


# A row carries only what the MODEL can do; caching, retry, and auth mode are
# transport facts. ``&`` can only remove, so a row asserting ``False`` would
# pin every transport -- these constants declare it per transport instead.

# The Messages API: prompt-cache breakpoints, internal retry, both thinking
# outputs. Server-side context management is opt-in per instance.
API: Final = ModelCapability(
    account_auth=False,
    manages_context=False,
    service_tiers=frozenset({"auto", "standard_only"}),
)

# The CLI subprocess cannot send the redact-thinking beta header, exposes no
# effort or latency knob, manages its own prompt cache, and conflicts with
# persistent retry (subprocess lifecycle). It rolls history itself.
CLI: Final = ModelCapability(
    fast=False,
    latency_modes=frozenset(),
    service_tiers=frozenset(),
    prompt_cache_breakpoints=False,
    retries_internally=False,
    supported_thinking_efforts=MappingProxyType({}),
    supported_thinking_outputs=frozenset({"text"}),
)

# OAuth billing: same wire as the API key, but account-scoped.
SUBSCRIPTION: Final = ModelCapability(
    manages_context=False,
    service_tiers=frozenset({"auto", "standard_only"}),
)
