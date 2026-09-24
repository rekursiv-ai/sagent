"""MiniMax model catalog, expressed as ``ModelCapability`` rows.

Source: https://platform.minimaxi.com/document

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
)
from sagent.types.cost import (
    PriceCatalog,
    PriceKey,
    TokenPrice,
)


if TYPE_CHECKING:
    from collections.abc import Mapping


__all__ = ["models"]


def models() -> Mapping[str, ModelCapability]:
    """Return every model this vendor serves, keyed by base id.

    Returns:
      models: Capability per base model id.

    """
    # The reference row: every other model states only how it differs from it.
    # MiniMax publishes no cache-hit rate, so each card states ``0.0``.
    default = ModelCapability(
        model_id="MiniMax-M2.7",
        context=_limits(),
        prices=_prices(_card(request=0.3, response=1.2, cache_read=0.0)),
        thinking=ThinkingCapability(output=frozenset({"none", "text"})),
    )
    utility = replace(
        default,
        model_id="MiniMax-Text-01",
        context=_limits(max_tokens=1_000_000, output_tokens=16_384),
        prices=_prices(_card(request=0.2, response=1.1, cache_read=0.0)),
    )
    rows = (
        default,
        replace(
            default,
            model_id="MiniMax-M2.7-highspeed",
            prices=_prices(_card(request=0.6, response=2.4, cache_read=0.0)),
        ),
        replace(default, model_id="MiniMax-M2.5"),
        replace(
            default,
            model_id="MiniMax-M1",
            context=_limits(max_tokens=1_000_000, output_tokens=16_384),
            prices=_prices(_card(request=0.4, response=2.2, cache_read=0.0)),
        ),
        utility,
        replace(
            default,
            model_id="abab6.5s-chat",
            context=_limits(max_tokens=245_000, output_tokens=16_384),
            prices=_prices(_card(request=0.15, response=0.15, cache_read=0.0)),
        ),
        replace(
            default,
            model_id="abab6.5-chat",
            context=_limits(max_tokens=245_000, output_tokens=16_384),
            prices=_prices(_card(request=1.5, response=1.5, cache_read=0.0)),
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


def _card(*, request: float, response: float, cache_read: float) -> TokenPrice:
    """Return one published price-table row; MiniMax bills no cache writes."""
    return TokenPrice(
        request=request,
        response=response,
        cache_write=0.0,
        cache_write_1h=0.0,
        cache_read=cache_read,
    )


def _prices(standard: TokenPrice) -> PriceCatalog:
    """Return the one published card; MiniMax quotes one flat tier."""
    return PriceCatalog({PriceKey("auto"): standard})


def _limits(
    *,
    max_tokens: int = 204_800,
    output_tokens: int = 32_768,
) -> Mapping[ContextTag, ModelLimits]:
    """One context tag: MiniMax ships no window variants."""
    return MappingProxyType(
        {
            "": ModelLimits(
                max_request_tokens=max_tokens,
                max_response_tokens=output_tokens,
            ),
        },
    )
