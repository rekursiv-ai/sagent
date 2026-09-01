"""Google (Gemini) model catalog, expressed as ``ModelCapability`` rows.

Sources:
  - ModelLimits: https://ai.google.dev/gemini-api/docs/models
  - Pricing: https://ai.google.dev/gemini-api/docs/pricing

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
from sagent.catalog.thinking import ThinkingEffort


__all__ = ["API", "CLI", "MODELS", "SUBSCRIPTION"]


def _limits(*, request: int) -> ModelLimits:
    """Gemini's byte and pixel ceilings are uniform across the range.

    No per-image pixel or byte cap: larger images are tiled into 768x768
    tiles server-side, so both stay 0 (no client resize). The only
    documented ceiling is the 20 MB TOTAL inline request size.
    """
    return ModelLimits(
        max_request_tokens=request,
        max_response_tokens=65_536,
        max_request_bytes=20 * 1024 * 1024,
    )


def _prices(*, request: float, response: float, cache_read: float) -> PriceCatalog:
    return PriceCatalog(
        {
            PriceCatalogProduct(): TokenPrice(
                request=request,
                response=response,
                cache_read=cache_read,
            )
        }
    )


# Effort maps to ``thinkingConfig.thinkingBudget``, an integer token cap.
_THINKING_BUDGETS: Mapping[ThinkingEffort, str] = MappingProxyType(
    {
        "min": "1024",
        "low": "4096",
        "medium": "8192",
        "high": "16384",
        "xhigh": "20480",
        "max": "24576",
    }
)


# Gemini surfaces readable thought parts and offers no server-side redaction;
# ``thinkingBudget: -1`` is the auto budget, a positive integer the fixed one.
# A row carries only what the MODEL can do; caching, retry, auth mode, and the
# fast path are transport facts, since ``&`` can only remove and a row saying
# ``False`` would pin every transport.
MODELS: Mapping[str, ModelCapability] = MappingProxyType(
    {
        "gemini-3-flash-preview": ModelCapability(
            model_id="gemini-3-flash-preview",
            context_limits=_limits(request=1_048_576),
            prices=_prices(request=0.5, response=3.0, cache_read=0.05),
            supported_thinking_efforts=_THINKING_BUDGETS,
            supported_thinking_budgets=frozenset({"auto", "fixed"}),
            supported_thinking_outputs=frozenset({"text"}),
        ),
        "gemini-3.1-pro-preview": ModelCapability(
            model_id="gemini-3.1-pro-preview",
            context_limits=_limits(request=1_048_576),
            prices=_prices(request=2.0, response=12.0, cache_read=0.2),
            supported_thinking_efforts=_THINKING_BUDGETS,
            supported_thinking_budgets=frozenset({"auto", "fixed"}),
            supported_thinking_outputs=frozenset({"text"}),
        ),
        "gemini-2.0-flash": ModelCapability(
            model_id="gemini-2.0-flash",
            context_limits=_limits(request=1_000_000),
            prices=_prices(request=0.1, response=0.4, cache_read=0.025),
            supported_thinking_efforts=_THINKING_BUDGETS,
            supported_thinking_budgets=frozenset({"auto", "fixed"}),
            supported_thinking_outputs=frozenset({"text"}),
        ),
        "gemini-2.5-flash-lite": ModelCapability(
            model_id="gemini-2.5-flash-lite",
            context_limits=_limits(request=1_048_576),
            prices=_prices(request=0.1, response=0.4, cache_read=0.025),
            supported_thinking_efforts=_THINKING_BUDGETS,
            supported_thinking_budgets=frozenset({"auto", "fixed"}),
            supported_thinking_outputs=frozenset({"text"}),
        ),
        "gemini-2.5-flash": ModelCapability(
            model_id="gemini-2.5-flash",
            context_limits=_limits(request=1_000_000),
            prices=_prices(request=0.3, response=2.5, cache_read=0.075),
            supported_thinking_efforts=_THINKING_BUDGETS,
            supported_thinking_budgets=frozenset({"auto", "fixed"}),
            supported_thinking_outputs=frozenset({"text"}),
        ),
        "gemini-2.5-pro": ModelCapability(
            model_id="gemini-2.5-pro",
            context_limits=_limits(request=1_000_000),
            prices=_prices(request=1.25, response=10.0, cache_read=0.31),
            supported_thinking_efforts=_THINKING_BUDGETS,
            supported_thinking_budgets=frozenset({"auto", "fixed"}),
            supported_thinking_outputs=frozenset({"text"}),
        ),
        "gemini-1.5-flash": ModelCapability(
            model_id="gemini-1.5-flash",
            context_limits=_limits(request=1_000_000),
            prices=_prices(request=0.075, response=0.3, cache_read=0.01875),
            # gemini-1.5 rejects ``thinkingConfig`` outright.
            supported_thinking_efforts=MappingProxyType({}),
            supported_thinking_budgets=frozenset(),
            supported_thinking_outputs=frozenset(),
        ),
        "gemini-1.5-pro": ModelCapability(
            model_id="gemini-1.5-pro",
            context_limits=_limits(request=1_000_000),
            prices=_prices(request=1.25, response=5.0, cache_read=0.3125),
            # gemini-1.5 rejects ``thinkingConfig`` outright.
            supported_thinking_efforts=MappingProxyType({}),
            supported_thinking_budgets=frozenset(),
            supported_thinking_outputs=frozenset(),
        ),
    }
)


# The REST API exposes no prompt-cache breakpoint, no fast path, and no
# server-side context management, and does not retry internally.
API: Final = ModelCapability(
    latency_modes=frozenset(),
    service_tiers=frozenset(),
    fast=False,
    manages_context=False,
    prompt_cache_breakpoints=False,
    retries_internally=False,
    account_auth=False,
)

# ACP exposes no effort knob on ``session/prompt``, and persistent retry
# conflicts with the subprocess lifecycle. The CLI rolls history itself and
# runs on the user's subscription.
CLI: Final = ModelCapability(
    latency_modes=frozenset(),
    service_tiers=frozenset(),
    fast=False,
    prompt_cache_breakpoints=False,
    retries_internally=False,
    supported_thinking_efforts=MappingProxyType({}),
)

# OAuth billing: same wire as the API key, but account-scoped.
SUBSCRIPTION: Final = ModelCapability(
    latency_modes=frozenset(),
    service_tiers=frozenset(),
    fast=False,
    manages_context=False,
    prompt_cache_breakpoints=False,
    retries_internally=False,
)
