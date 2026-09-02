"""Moonshot/Kimi model catalog, expressed as ``ModelCapability`` rows.

Source: https://platform.moonshot.cn/docs/api/chat

A row carries only what the MODEL can do; caching, retry, and auth mode
are transport facts declared on ``OpenAICompat.TRANSPORT``. These vendors
publish no image pixel or byte ceiling and preprocess images server-side,
so ``ModelLimits`` carries only the two token windows.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from types import MappingProxyType

from sagent.types.capability import ContextTag, ModelCapability, ModelLimits
from sagent.types.cost import (
    PriceCatalog,
    PriceCatalogProduct,
    TokenPrice,
)


__all__ = ["models"]


def models() -> Mapping[str, ModelCapability]:
    """Return every model this vendor serves, keyed by base id.

    Returns:
      models: Capability per base model id.

    """
    # Every Kimi row is this one with its window and price replaced.
    kimi = ModelCapability(
        context=_limits(window=256_000, response=96_000),
        prices=_prices(request=0.95, response=4.0, cache_read=0.16),
        thinking_output={"none", "text"},
    )
    rows = (
        replace(kimi, model_id="kimi-k2.6"),
        replace(
            kimi,
            model_id="kimi-k2.5",
            prices=_prices(request=0.6, response=3.0, cache_read=0.1),
        ),
        replace(
            kimi,
            model_id="kimi-k2-0905-preview",
            context=_limits(window=256_000, response=32_768),
            prices=_prices(request=0.6, response=2.5, cache_read=0.15),
        ),
        replace(
            kimi,
            model_id="kimi-k2-0711-preview",
            context=_limits(window=131_072, response=32_768),
            prices=_prices(request=0.6, response=2.5, cache_read=0.15),
        ),
        replace(
            kimi,
            model_id="kimi-k2-turbo-preview",
            context=_limits(window=256_000, response=32_768),
            prices=_prices(request=1.2, response=5.0, cache_read=0.3),
        ),
        replace(
            kimi,
            model_id="moonshot-v1-8k",
            context=_limits(window=8_000, response=16_384),
            prices=_prices(request=0.2, response=2.0),
        ),
        replace(
            kimi,
            model_id="moonshot-v1-32k",
            context=_limits(window=32_000, response=16_384),
            prices=_prices(request=0.4, response=4.0),
        ),
        replace(
            kimi,
            model_id="moonshot-v1-128k",
            context=_limits(window=128_000, response=16_384),
            prices=_prices(request=0.6, response=6.0),
        ),
    )
    return MappingProxyType({row.model_id: row for row in rows})


def _limits(*, window: int, response: int) -> Mapping[ContextTag, ModelLimits]:
    """One context tag: these vendors ship no window variants."""
    return MappingProxyType(
        {
            "": ModelLimits(
                max_request_tokens=window,
                max_response_tokens=response,
            )
        }
    )


def _prices(
    *, request: float, response: float, cache_read: float = 0.0
) -> PriceCatalog:
    """USD per million tokens; these vendors quote one flat tier."""
    return PriceCatalog(
        {
            PriceCatalogProduct(): TokenPrice(
                request=request, response=response, cache_read=cache_read
            )
        }
    )
