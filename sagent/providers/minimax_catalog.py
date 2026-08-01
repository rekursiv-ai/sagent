"""MiniMax model catalog, expressed as ``ModelCapability`` rows.

Source: https://platform.minimaxi.com/document

A row carries only what the MODEL can do; caching, retry, and auth mode
are transport facts declared on ``OpenAICompat.TRANSPORT``. These vendors
publish no image pixel or byte ceiling and preprocess images server-side,
so ``Limits`` carries only the two token windows.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from sagent.types.cost import (
    PriceCatalog,
    PriceCatalogProduct,
    TokenPrice,
)
from sagent.types.model import Limits, ModelCapability


__all__ = ["MODELS"]


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


MODELS: Mapping[str, ModelCapability] = MappingProxyType(
    {
        "MiniMax-M2.7": ModelCapability(
            model_id="MiniMax-M2.7",
            context_limits=Limits(
                max_request_tokens=204_800,
                max_response_tokens=32_768,
            ),
            prices=_prices(request=0.3, response=1.2),
        ),
        "MiniMax-M2.7-highspeed": ModelCapability(
            model_id="MiniMax-M2.7-highspeed",
            context_limits=Limits(
                max_request_tokens=204_800,
                max_response_tokens=32_768,
            ),
            prices=_prices(request=0.6, response=2.4),
        ),
        "MiniMax-M2.5": ModelCapability(
            model_id="MiniMax-M2.5",
            context_limits=Limits(
                max_request_tokens=204_800,
                max_response_tokens=32_768,
            ),
            prices=_prices(request=0.3, response=1.2),
        ),
        "MiniMax-M1": ModelCapability(
            model_id="MiniMax-M1",
            context_limits=Limits(
                max_request_tokens=1_000_000,
                max_response_tokens=16_384,
            ),
            prices=_prices(request=0.4, response=2.2),
        ),
        "MiniMax-Text-01": ModelCapability(
            model_id="MiniMax-Text-01",
            context_limits=Limits(
                max_request_tokens=1_000_000,
                max_response_tokens=16_384,
            ),
            prices=_prices(request=0.2, response=1.1),
        ),
        "abab6.5s-chat": ModelCapability(
            model_id="abab6.5s-chat",
            context_limits=Limits(
                max_request_tokens=245_000,
                max_response_tokens=16_384,
            ),
            prices=_prices(request=0.15, response=0.15),
        ),
        "abab6.5-chat": ModelCapability(
            model_id="abab6.5-chat",
            context_limits=Limits(
                max_request_tokens=245_000,
                max_response_tokens=16_384,
            ),
            prices=_prices(request=1.5, response=1.5),
        ),
    }
)
