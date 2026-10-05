"""MiniMax model catalog, expressed as ``ModelCapability`` rows.

Sources (pay-as-you-go, 2026-10-03):
  https://platform.minimax.io/docs/guides/pricing-paygo
  https://platform.minimax.io/docs/guides/models-intro
  https://platform.minimax.io/docs/api-reference/text-chat-openai
  https://platform.minimax.io/docs/api-reference/text-openai-api

A row carries only what the MODEL can do; caching, retry, and auth mode
are transport facts declared by :func:`api`.

Thinking: M2.x always thinks ("``disabled`` is accepted but ignored"), so
its rows offer no switch. M3 thinks adaptively by default and accepts
``thinking: {"type": "disabled"}``, so its budget offers ``none`` (off)
and ``auto`` (adaptive). ``reasoning_effort`` is honored only by the
plan-only M3.1 preview, so no row offers an effort.
"""

from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType
from typing import TYPE_CHECKING

from sagent.catalog.openai import compatible
from sagent.catalog.table import ModelTable
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


__all__ = ["api", "models"]


def api() -> ModelCapability:
    """Return the chat-completions transport with MiniMax's ``priority`` tier.

    Returns:
      capability: The shared compatible transport, plus ``priority``.

    """
    transport = compatible()
    return replace(transport, service_tier=transport.service_tier | {"priority"})


def models() -> ModelTable:
    """Return every model this vendor serves, keyed by base id.

    Returns:
      models: Capability per base model id.

    """
    always_thinks = ThinkingCapability(output=frozenset({"none", "text"}))
    m2 = ModelCapability(
        model_id="minimax-2.7",
        wire_model_id="MiniMax-M2.7",
        context=_limits(max_tokens=204_800, output_tokens=204_800),
        prices=_prices(_m2_card(request=0.3, response=1.2, cache_read=0.06)),
        thinking=always_thinks,
    )
    # M2.5 and older cache hits bill $0.03 rather than M2.7's $0.06.
    legacy = _prices(_m2_card(request=0.3, response=1.2, cache_read=0.03))
    legacy_fast = _prices(_m2_card(request=0.6, response=2.4, cache_read=0.03))
    rows = (
        ModelCapability(
            model_id="minimax-3.0",
            wire_model_id="MiniMax-M3",
            # Image input: "images can be up to 10 MB, and the request body
            # can be up to 64 MB".
            context=_limits(
                max_tokens=1_000_000,
                output_tokens=524_288,
                request_bytes=64 * 1024 * 1024,
                image_bytes=10 * 1024 * 1024,
            ),
            prices=_m3_prices(),
            thinking=replace(always_thinks, budget=frozenset({"none", "auto"})),
        ),
        m2,
        replace(
            m2,
            model_id="minimax-2.5",
            wire_model_id="MiniMax-M2.5",
            prices=legacy,
        ),
        replace(
            m2,
            model_id="minimax-2.1",
            wire_model_id="MiniMax-M2.1",
            prices=legacy,
        ),
        replace(m2, model_id="minimax-2.0", wire_model_id="MiniMax-M2", prices=legacy),
        replace(
            m2,
            model_id="minimax-highspeed-2.7",
            wire_model_id="MiniMax-M2.7-highspeed",
            prices=_prices(_m2_card(request=0.6, response=2.4, cache_read=0.06)),
        ),
        replace(
            m2,
            model_id="minimax-highspeed-2.5",
            wire_model_id="MiniMax-M2.5-highspeed",
            prices=legacy_fast,
        ),
        replace(
            m2,
            model_id="minimax-highspeed-2.1",
            wire_model_id="MiniMax-M2.1-highspeed",
            prices=legacy_fast,
        ),
    )
    # A tier is offered exactly when it is priced.
    rows = tuple(replace(row, service_tier=row.prices.service_tiers) for row in rows)
    return ModelTable(
        rows=rows,
        # models-intro lists M3 first as the frontier model. With its permanent
        # 50% discount no live row is cheaper ($0.3 / $1.2 at best), so a
        # separate utility would cost as much or more.
        roles={"default": "minimax-3.0", "utility": "minimax-3.0"},
    )


# The listed rates carry a "permanent 50% off"; the cards bill the discounted rate.
# Priority is "1.5x standard" in every column.
def _m3_prices() -> PriceCatalog:
    """Return M3's standard and priority cards, each banded above 512K."""
    short = _card(request=0.3, response=1.2, cache_read=0.06)
    long = _card(request=0.6, response=2.4, cache_read=0.12)
    return PriceCatalog(
        {
            PriceKey("auto"): short,
            PriceKey("auto", 512_000): long,
            PriceKey("priority"): _scaled(short, 1.5),
            PriceKey("priority", 512_000): _scaled(long, 1.5),
        },
    )


def _card(
    *,
    request: float,
    response: float,
    cache_read: float,
    cache_write: float = 0.0,
) -> TokenPrice:
    """Return one published price-table row; M3 publishes no write column."""
    return TokenPrice(
        request=request,
        response=response,
        cache_write=cache_write,
        cache_write_1h=0.0,
        cache_read=cache_read,
    )


def _m2_card(*, request: float, response: float, cache_read: float) -> TokenPrice:
    """Return an M2.x row; every M2.x model bills a $0.375 cache write."""
    return _card(
        request=request,
        response=response,
        cache_read=cache_read,
        cache_write=0.375,
    )


def _scaled(price: TokenPrice, factor: float) -> TokenPrice:
    """Multiply every rate of ``price`` by ``factor``."""
    return replace(
        price,
        request=price.request * factor,
        response=price.response * factor,
        cache_read=price.cache_read * factor,
    )


def _prices(standard: TokenPrice) -> PriceCatalog:
    """Return the one published card; M2.x quotes one flat standard tier."""
    return PriceCatalog({PriceKey("auto"): standard})


def _limits(
    *,
    max_tokens: int,
    output_tokens: int,
    request_bytes: int = 0,
    image_bytes: int = 0,
) -> Mapping[ContextTag, ModelLimits]:
    """One context tag: MiniMax ships no window variants."""
    return MappingProxyType(
        {
            "": ModelLimits(
                max_request_tokens=max_tokens,
                max_response_tokens=output_tokens,
                max_request_bytes=request_bytes,
                max_image_bytes=image_bytes,
            ),
        },
    )
