"""MiniMax model catalog, expressed as ``ModelCapability`` rows.

Source: https://platform.minimaxi.com/document

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
    # Every MiniMax row is this one with its window and price replaced.
    minimax = ModelCapability(
        context=_limits(window=204_800, response=32_768),
        prices=_prices(request=0.3, response=1.2),
        thinking_output={"none", "text"},
    )
    long_context = replace(minimax, context=_limits(window=1_000_000, response=16_384))
    abab = replace(minimax, context=_limits(window=245_000, response=16_384))
    rows = (
        replace(minimax, model_id="MiniMax-M2.7"),
        replace(
            minimax,
            model_id="MiniMax-M2.7-highspeed",
            prices=_prices(request=0.6, response=2.4),
        ),
        replace(minimax, model_id="MiniMax-M2.5"),
        replace(
            long_context,
            model_id="MiniMax-M1",
            prices=_prices(request=0.4, response=2.2),
        ),
        replace(
            long_context,
            model_id="MiniMax-Text-01",
            prices=_prices(request=0.2, response=1.1),
        ),
        replace(
            abab,
            model_id="abab6.5s-chat",
            prices=_prices(request=0.15, response=0.15),
        ),
        replace(
            abab,
            model_id="abab6.5-chat",
            prices=_prices(request=1.5, response=1.5),
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
