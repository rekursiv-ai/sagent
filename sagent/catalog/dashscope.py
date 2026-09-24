"""DashScope/Qwen model catalog, expressed as ``ModelCapability`` rows.

Source: https://help.aliyun.com/zh/model-studio/models

A row carries only what the MODEL can do; caching, retry, and auth mode
are transport facts declared on ``OpenAICompat.TRANSPORT``. These vendors
publish no image pixel or byte ceiling and preprocess images server-side,
so ``ModelLimits`` carries only the two token windows.
"""

from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType
from typing import TYPE_CHECKING

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


__all__ = ["models", "thinking_budget"]


def thinking_budget(effort: ThinkingEffort) -> str:
    """Return the ``thinking_budget`` token cap an effort sends.

    DashScope rejects ``reasoning_effort``; it takes ``enable_thinking``
    plus this cap, where ``0`` means ``enable_thinking=False``.

    Args:
      effort: Selected effort level.

    Returns:
      budget: Wire value, as the string the request body carries.

    """
    match effort:
        case "none":
            return "0"
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


def models() -> Mapping[str, ModelCapability]:
    """Return every model this vendor serves, keyed by base id.

    Returns:
      models: Capability per base model id.

    """
    # The reference row: every other model states only how it differs from it.
    default = ModelCapability(
        model_id="qwen3.6-plus",
        context=_limits(),
        prices=_prices(_card(request=0.5, response=3.0)),
        thinking=ThinkingCapability(
            effort=frozenset({"none", "min", "low", "medium", "high", "xhigh", "max"}),
            budget=frozenset({"none", "auto", "fixed"}),
            output=frozenset({"none", "text"}),
        ),
    )
    utility = replace(
        default,
        model_id="qwen3.6-flash",
        prices=_prices(_card(request=0.05, response=0.2)),
    )
    rows = (
        replace(
            default,
            model_id="qwen3.6-max-preview",
            context=_limits(max_tokens=262_144),
            prices=_prices(_card(request=1.6, response=6.4)),
        ),
        default,
        utility,
        # ``-instruct`` / ``-coder`` / ``-turbo`` reject the toggle, so every
        # thinking axis offers only its off value.
        replace(
            default,
            model_id="qwen3-235b-a22b-instruct-2507",
            context=_limits(max_tokens=262_144),
            prices=_prices(_card(request=0.7, response=2.8)),
            thinking=ThinkingCapability(),
        ),
        replace(
            default,
            model_id="qwen3-235b-a22b-thinking-2507",
            context=_limits(max_tokens=262_144, output_tokens=32_768),
            prices=_prices(_card(request=0.7, response=8.4)),
            # Everything but ``none``: these ids reject ``enable_thinking=false``.
            thinking=replace(
                default.thinking,
                effort=default.thinking.effort - {"none"},
            ),
        ),
        replace(
            default,
            model_id="qwen3-30b-a3b-instruct-2507",
            context=_limits(max_tokens=262_144),
            prices=_prices(_card(request=0.2, response=0.8)),
            thinking=ThinkingCapability(),
        ),
        replace(
            default,
            model_id="qwen3-32b",
            context=_limits(max_tokens=262_144),
            prices=_prices(_card(request=0.4, response=1.2)),
        ),
        replace(
            default,
            model_id="qwen3-coder-480b-a35b-instruct",
            context=_limits(max_tokens=262_144),
            prices=_prices(_card(request=1.0, response=5.0)),
            thinking=ThinkingCapability(),
        ),
        replace(
            default,
            model_id="qwen-plus",
            context=_limits(output_tokens=32_768),
            prices=_prices(_card(request=0.4, response=1.2)),
        ),
        replace(
            default,
            model_id="qwen-max",
            context=_limits(max_tokens=262_144),
            prices=_prices(_card(request=1.6, response=6.4)),
        ),
        replace(
            default,
            model_id="qwen-turbo",
            context=_limits(output_tokens=32_768),
            prices=_prices(_card(request=0.05, response=0.2)),
            thinking=ThinkingCapability(),
        ),
    )
    # A tier is offered exactly when it is priced.
    rows = tuple(replace(row, service_tier=row.prices.service_tiers) for row in rows)
    catalog = {row.model_id: row for row in rows}
    return MappingProxyType(
        {
            "default": catalog[default.model_id],
            "utility": catalog[utility.model_id],
            **catalog,
        },
    )


def _card(*, request: float, response: float) -> TokenPrice:
    """Return one published price-table row; DashScope reports no cache pools."""
    return TokenPrice(
        request=request,
        response=response,
        cache_write=0.0,
        cache_write_1h=0.0,
        cache_read=0.0,
    )


def _prices(standard: TokenPrice) -> PriceCatalog:
    """Return the one published card; DashScope quotes one flat tier."""
    return PriceCatalog({PriceKey("auto"): standard})


def _limits(
    *,
    max_tokens: int = 1_000_000,
    output_tokens: int = 65_536,
) -> Mapping[ContextTag, ModelLimits]:
    """One context tag: DashScope ships no window variants."""
    return MappingProxyType(
        {
            "": ModelLimits(
                max_request_tokens=max_tokens,
                max_response_tokens=output_tokens,
            ),
        },
    )
