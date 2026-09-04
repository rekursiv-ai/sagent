"""Google (Gemini) model catalog, expressed as ``ModelCapability`` rows.

Sources:
  - Limits: https://ai.google.dev/gemini-api/docs/models
  - Pricing: https://ai.google.dev/gemini-api/docs/pricing

:func:`models` is the API-key view. Other transports narrow it with ``&``
(see :func:`cli`), which can only remove.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from types import MappingProxyType

from sagent.types.capability import (
    ContextTag,
    ModelCapability,
    ModelLimits,
    ThinkingEffort,
)
from sagent.types.cost import (
    PriceCatalog,
    PriceCatalogProduct,
    TokenPrice,
)


__all__ = ["api", "cli", "models", "subscription", "thinking_budget"]


def thinking_budget(effort: ThinkingEffort) -> str:
    """Return the ``thinkingConfig.thinkingBudget`` an effort sends.

    Args:
      effort: Selected effort level.

    Returns:
      budget: Wire value, as the string the request body carries.

    Raises:
      ValueError: ``effort`` is ``none``; a disabled request omits
          ``thinkingConfig`` rather than sending a budget of zero.

    """
    match effort:
        case "none":
            raise ValueError(f"effort {effort!r} sends no budget; omit thinkingConfig")
        case "min":
            return "1024"
        case "low":
            return "4096"
        case "medium":
            return "8192"
        case "high":
            return "16384"
        case "xhigh":
            return "20480"
        case "max":
            return "24576"


# Gemini surfaces readable thought parts and offers no server-side redaction;
# ``thinkingBudget: -1`` is the auto budget, a positive integer the fixed one.
# A row carries only what the MODEL can do; caching, retry, auth mode, and the
# fast path are transport facts, since ``&`` can only remove and a row saying
# ``False`` would pin every transport.
def models() -> Mapping[str, ModelCapability]:
    """Return every Gemini model, as the API-key transport sees it.

    Returns:
      models: Capability per base model id.

    """
    # A thinking Gemini row: every model below is this one with its window
    # and price replaced. ``thinkingBudget: -1`` is the auto budget, a
    # positive integer the fixed one.
    gemini = ModelCapability(
        context=_context(request=1_048_576),
        prices=_prices(request=0.5, response=3.0, cache_read=0.05),
        thinking_effort=_efforts(),
        thinking_budget={"none", "auto", "fixed"},
        thinking_output={"none", "text"},
    )
    # gemini-1.5 rejects ``thinkingConfig`` outright, so every thinking axis
    # offers only its off value.
    legacy = replace(
        gemini,
        context=_context(request=1_000_000),
        thinking_effort={"none"},
        thinking_budget={"none"},
        thinking_output={"none"},
    )
    rows = (
        replace(gemini, model_id="gemini-3-flash-preview"),
        replace(
            gemini,
            model_id="gemini-3.1-pro-preview",
            prices=_prices(request=2.0, response=12.0, cache_read=0.2),
        ),
        replace(
            gemini,
            model_id="gemini-2.0-flash",
            context=_context(request=1_000_000),
            prices=_prices(request=0.1, response=0.4, cache_read=0.025),
        ),
        replace(
            gemini,
            model_id="gemini-2.5-flash-lite",
            prices=_prices(request=0.1, response=0.4, cache_read=0.01),
        ),
        replace(
            gemini,
            model_id="gemini-2.5-flash",
            context=_context(request=1_000_000),
            prices=_prices(request=0.3, response=2.5, cache_read=0.03),
        ),
        replace(
            gemini,
            model_id="gemini-2.5-pro",
            context=_context(request=1_000_000),
            prices=_prices(request=1.25, response=10.0, cache_read=0.125),
        ),
        replace(
            legacy,
            model_id="gemini-1.5-flash",
            prices=_prices(request=0.075, response=0.3, cache_read=0.01875),
        ),
        replace(
            legacy,
            model_id="gemini-1.5-pro",
            prices=_prices(request=1.25, response=5.0, cache_read=0.3125),
        ),
    )
    return MappingProxyType({row.model_id: row for row in rows})


def api() -> ModelCapability:
    """Return what the REST API adds on top of a model row.

    Nothing; named so the three transports read alike at their call sites.

    One model, three ways in::

        models()["gemini-3.1-pro-preview"] & api()          # REST, key auth
        models()["gemini-3.1-pro-preview"] & subscription() # same wire, OAuth
        models()["gemini-3.1-pro-preview"] & cli()          # ACP: no effort knob

    Returns:
      capability: The API transport's restrictions.

    """
    # Every axis stated: ``&`` can only remove, and a defaulted axis is the
    # narrow value, so an omitted one would strip the model's real capability.
    return ModelCapability(
        thinking_effort={"none", "min", "low", "medium", "high", "xhigh", "max"},
        thinking_budget={"none", "auto", "fixed"},
        thinking_output={"none", "text", "redacted"},
    )


def cli() -> ModelCapability:
    """Return what the ``gemini`` subprocess narrows a model row to.

    One model, three ways in::

        models()["gemini-3.1-pro-preview"] & api()          # REST, key auth
        models()["gemini-3.1-pro-preview"] & subscription() # same wire, OAuth
        models()["gemini-3.1-pro-preview"] & cli()          # ACP: no effort knob

    Returns:
      capability: The CLI transport's restrictions.

    """
    return ModelCapability(
        thinking_effort={"none"},
        thinking_budget={"none", "auto", "fixed"},
        thinking_output={"none", "text", "redacted"},
        manage_context_server_side={True},
        account_auth=True,
    )


def subscription() -> ModelCapability:
    """Return what OAuth billing narrows a model row to.

    One model, three ways in::

        models()["gemini-3.1-pro-preview"] & api()          # REST, key auth
        models()["gemini-3.1-pro-preview"] & subscription() # same wire, OAuth
        models()["gemini-3.1-pro-preview"] & cli()          # ACP: no effort knob

    Returns:
      capability: The subscription transport's restrictions.

    """
    return ModelCapability(
        thinking_effort={"none", "min", "low", "medium", "high", "xhigh", "max"},
        thinking_budget={"none", "auto", "fixed"},
        thinking_output={"none", "text", "redacted"},
        account_auth=True,
    )


def _context(*, request: int) -> Mapping[ContextTag, ModelLimits]:
    """No per-image cap: images are tiled server-side, so only bytes bound."""
    return MappingProxyType(
        {
            "": ModelLimits(
                max_request_tokens=request,
                max_response_tokens=65_536,
                max_request_bytes=20 * 1024 * 1024,
            )
        }
    )


def _prices(*, request: float, response: float, cache_read: float) -> PriceCatalog:
    """USD per million tokens; Gemini quotes one flat tier."""
    return PriceCatalog(
        {
            PriceCatalogProduct(): TokenPrice(
                request=request,
                response=response,
                cache_read=cache_read,
            )
        }
    )


def _efforts() -> frozenset[ThinkingEffort]:
    """Every effort Gemini accepts: its wire-mapped levels, plus ``none``."""
    return frozenset({"none", "min", "low", "medium", "high", "xhigh", "max"})
