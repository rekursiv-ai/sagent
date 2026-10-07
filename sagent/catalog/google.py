"""Google (Gemini) model catalog, expressed as ``ModelCapability`` rows.

Sources:
  - Limits: https://ai.google.dev/gemini-api/docs/models
  - Pricing: https://ai.google.dev/gemini-api/docs/pricing
  - Thinking: https://ai.google.dev/gemini-api/docs/generate-content/thinking

:func:`models` is the API-key view. Other transports narrow it with ``&``
(see :func:`cli`), which can only remove.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date
from types import MappingProxyType
from typing import TYPE_CHECKING

from sagent.catalog.table import ModelTable
from sagent.types.capability import (
    ContextTag,
    ModelCapability,
    ModelLimits,
    ThinkingCapability,
    ThinkingEffort,
)
from sagent.types.cost import (
    PriceCatalog,
    PriceKey,
    TokenPrice,
)


if TYPE_CHECKING:
    from collections.abc import Mapping

    from sagent.lib.codec import MutablePlainTree
    from sagent.types.capability import ModelSettings


__all__ = [
    "api",
    "cli",
    "models",
    "subscription",
    "thinking_budget",
    "thinking_config",
]


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


def thinking_config(
    capability: ModelCapability,
    settings: ModelSettings,
) -> dict[str, MutablePlainTree] | None:
    """Return the ``generationConfig.thinkingConfig`` a selection sends.

    Budget ``none`` omits the config and effort ``none`` omits the level or
    cap, so the model's own default applies in both cases. Gemini 3 rows take
    a ``thinkingLevel`` under either budget mode, its levels being dynamic
    already; 2.5 rows take a ``thinkingBudget`` token cap, ``-1`` under
    ``auto``.

    Args:
      capability: The row the request is built against.
      settings: The selection in effect.

    Returns:
      config: The wire object, or ``None`` when the request sends none.

    """
    if settings.thinking_budget == "none":
        return None
    config: dict[str, MutablePlainTree] = {
        "includeThoughts": settings.thinking_output == "text",
    }
    effort = settings.thinking_effort
    if capability.effort_as_level:
        # Gemini 3 accepts ``thinkingBudget`` only for backwards compatibility.
        if effort != "none":
            config["thinkingLevel"] = _thinking_level(effort)
    elif settings.thinking_budget == "auto":
        config["thinkingBudget"] = -1
    elif effort != "none":
        config["thinkingBudget"] = int(thinking_budget(effort))
    return config


# Gemini surfaces readable thought parts and offers no server-side redaction.
# A row carries only what the MODEL can do; caching, retry, auth mode, and the
# fast path are transport facts, since ``&`` can only remove and a row saying
# ``False`` would pin every transport.
def models() -> ModelTable:
    """Return every Gemini model, as the API-key transport sees it.

    Returns:
      models: Capability per base model id.

    """
    # Every row keeps effort ``none``, meaning "send no level": the model's
    # default then applies, even on rows that cannot disable thinking.
    levels = ThinkingCapability(
        effort=frozenset({"none", "min", "low", "medium", "high"}),
        budget=frozenset({"none", "auto", "fixed"}),
        output=frozenset({"none", "text"}),
    )
    # "minimal is not supported and returns an error" on these.
    no_minimal = replace(levels, effort=frozenset({"none", "low", "medium", "high"}))
    budgets = replace(
        levels,
        effort=frozenset({"none", "min", "low", "medium", "high", "xhigh", "max"}),
    )
    # The reference row: every other model states only how it differs from it.
    pro = ModelCapability(
        model_id="gemini-pro-3.1",
        wire_model_id="gemini-3.1-pro-preview",
        knowledge_cutoff="January 2025",
        context=_limits(),
        prices=_prices(
            _card(request=2.0, response=12.0, cache_read=0.2),
            over_200k=_card(request=4.0, response=18.0, cache_read=0.4),
        ),
        thinking=no_minimal,
        effort_as_level=True,
    )
    # 2.5 cannot take ``thinkingLevel``; every effort maps to a token cap.
    legacy = replace(pro, thinking=budgets, effort_as_level=False)
    # The 3.6-3.8 Flash cards are introductory through 2026-12-31.
    flash = replace(
        pro,
        knowledge_cutoff=None,
        prices=_prices(
            _card(request=0.75, response=3.75, cache_read=0.075),
            regular_from_2027=_card(request=1.5, response=7.5, cache_read=0.15),
        ),
    )
    rows = (
        pro,
        replace(
            legacy,
            model_id="gemini-pro-2.5",
            wire_model_id="gemini-2.5-pro",
            prices=_prices(
                _card(request=1.25, response=10.0, cache_read=0.125),
                over_200k=_card(request=2.5, response=15.0, cache_read=0.25),
            ),
        ),
        replace(flash, model_id="gemini-flash-3.8", wire_model_id="gemini-3.8-flash"),
        replace(flash, model_id="gemini-flash-3.7", wire_model_id="gemini-3.7-flash"),
        replace(
            flash,
            model_id="gemini-flash-3.6",
            wire_model_id="gemini-3.6-flash",
            thinking=levels,
        ),
        replace(
            flash,
            model_id="gemini-flash-3.5",
            wire_model_id="gemini-3.5-flash",
            prices=_prices(_card(request=1.5, response=9.0, cache_read=0.15)),
            thinking=levels,
        ),
        # Legacy (/pricing): replaced by gemini-flash-3.6; no shutdown date yet.
        replace(
            pro,
            model_id="gemini-flash-3.0",
            wire_model_id="gemini-3-flash-preview",
            prices=_prices(_card(request=0.5, response=3.0, cache_read=0.05)),
            thinking=levels,
        ),
        replace(
            legacy,
            model_id="gemini-flash-2.5",
            wire_model_id="gemini-2.5-flash",
            prices=_prices(_card(request=0.3, response=2.5, cache_read=0.03)),
        ),
        replace(
            pro,
            model_id="gemini-flash-lite-3.5",
            wire_model_id="gemini-3.5-flash-lite",
            knowledge_cutoff=None,
            prices=_prices(_card(request=0.3, response=2.5, cache_read=0.03)),
            thinking=levels,
        ),
        # Shuts down 2027-05-07. Audio input bills $0.50, not the text rate.
        replace(
            pro,
            model_id="gemini-flash-lite-3.1",
            wire_model_id="gemini-3.1-flash-lite",
            prices=_prices(_card(request=0.25, response=1.5, cache_read=0.025)),
            thinking=levels,
        ),
        replace(
            legacy,
            model_id="gemini-flash-lite-2.5",
            wire_model_id="gemini-2.5-flash-lite",
            prices=_prices(_card(request=0.1, response=0.4, cache_read=0.01)),
        ),
    )
    # A tier is offered exactly when it is priced.
    rows = tuple(replace(row, service_tier=row.prices.service_tiers) for row in rows)
    # /models: "For any new projects, use ... 3.5 Flash-Lite or 3.8 Flash".
    return ModelTable(
        rows=rows,
        roles={"default": "gemini-pro", "utility": "gemini-flash-lite"},
    )


def api() -> ModelCapability:
    """Return what the REST API adds on top of a model row.

    Nothing; named so the three transports read alike at their call sites.

    One model, three ways in::

        models()["gemini-pro-3.1"] & api()          # REST, key auth
        models()["gemini-pro-3.1"] & subscription() # same wire, OAuth
        models()["gemini-pro-3.1"] & cli()          # ACP: no effort knob

    Returns:
      capability: The API transport's restrictions.

    """
    # Every axis stated: ``&`` can only remove, and a defaulted axis is the
    # narrow value, so an omitted one would strip the model's real capability.
    return ModelCapability(
        thinking=ThinkingCapability(
            effort=frozenset({"none", "min", "low", "medium", "high", "xhigh", "max"}),
            budget=frozenset({"none", "auto", "fixed"}),
            output=frozenset({"none", "text"}),
        ),
    )


def cli() -> ModelCapability:
    """Return what the ``gemini`` subprocess narrows a model row to.

    One model, three ways in::

        models()["gemini-pro-3.1"] & api()          # REST, key auth
        models()["gemini-pro-3.1"] & subscription() # same wire, OAuth
        models()["gemini-pro-3.1"] & cli()          # ACP: no effort knob

    Returns:
      capability: The CLI transport's restrictions.

    """
    return ModelCapability(
        thinking=ThinkingCapability(
            effort=frozenset({"none"}),
            budget=frozenset({"none", "auto", "fixed"}),
            output=frozenset({"none", "text"}),
        ),
        manage_context_server_side=frozenset({True}),
        account_auth=True,
    )


def subscription() -> ModelCapability:
    """Return what OAuth billing narrows a model row to.

    One model, three ways in::

        models()["gemini-pro-3.1"] & api()          # REST, key auth
        models()["gemini-pro-3.1"] & subscription() # same wire, OAuth
        models()["gemini-pro-3.1"] & cli()          # ACP: no effort knob

    Returns:
      capability: The subscription transport's restrictions.

    """
    return ModelCapability(
        thinking=ThinkingCapability(
            effort=frozenset({"none", "min", "low", "medium", "high", "xhigh", "max"}),
            budget=frozenset({"none", "auto", "fixed"}),
            output=frozenset({"none", "text"}),
        ),
        account_auth=True,
    )


def _thinking_level(effort: ThinkingEffort) -> str:
    """Return the Gemini 3 ``thinkingLevel`` an effort sends."""
    match effort:
        case "min":
            return "minimal"
        case "low" | "medium" | "high":
            return effort
        case "none" | "xhigh" | "max":
            raise ValueError(f"effort {effort!r} has no Gemini thinkingLevel")


def _card(*, request: float, response: float, cache_read: float) -> TokenPrice:
    """Return one published price-table row; Gemini bills no cache writes."""
    return TokenPrice(
        request=request,
        response=response,
        cache_write=0.0,
        cache_write_1h=0.0,
        cache_read=cache_read,
    )


# The REST ``generateContent`` wire this catalog serves sends no
# ``service_tier``, so only the standard tab applies; the page's flex and
# priority tabs belong to the Interactions API. Audio input is billed at a
# separate rate the single input meter cannot express; the text rate is kept.
def _prices(
    standard: TokenPrice,
    *,
    over_200k: TokenPrice | None = None,
    regular_from_2027: TokenPrice | None = None,
) -> PriceCatalog:
    """Return the standard card, plus its >200k band and 2027 reprice if published."""
    cards = {PriceKey("auto"): standard}
    if over_200k is not None:
        cards[PriceKey("auto", 200_000)] = over_200k
    if regular_from_2027 is not None:
        cards[PriceKey("auto", effective_date=date(2027, 1, 1))] = regular_from_2027
    return PriceCatalog(cards)


def _limits() -> Mapping[ContextTag, ModelLimits]:
    """No per-image cap: images are tiled server-side, so only bytes bound."""
    return MappingProxyType(
        {
            "": ModelLimits(
                max_request_tokens=1_048_576,
                max_response_tokens=65_536,
                # /file-input-methods and the changelog say 100 MB inline;
                # /image-understanding still says 20 MB.
                max_request_bytes=100 * 1024 * 1024,
            ),
        },
    )
