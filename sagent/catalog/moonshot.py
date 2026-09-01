"""Moonshot/Kimi model catalog, expressed as ``ModelCapability`` rows.

Source: https://platform.moonshot.cn/docs/api/chat

A row carries only what the MODEL can do; caching, retry, and auth mode
are transport facts declared on ``OpenAICompat.TRANSPORT``. These vendors
publish no image pixel or byte ceiling and preprocess images server-side,
so ``ModelLimits`` carries only the two token windows.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from sagent.types.capability import ModelCapability, ModelLimits
from sagent.types.cost import (
    PriceCatalog,
    PriceCatalogProduct,
    TokenPrice,
)


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
        "kimi-k2.6": ModelCapability(
            model_id="kimi-k2.6",
            context_limits=ModelLimits(
                max_request_tokens=256_000,
                max_response_tokens=96_000,
            ),
            prices=_prices(request=0.95, response=4.0, cache_read=0.16),
        ),
        "kimi-k2.5": ModelCapability(
            model_id="kimi-k2.5",
            context_limits=ModelLimits(
                max_request_tokens=256_000,
                max_response_tokens=96_000,
            ),
            prices=_prices(request=0.6, response=3.0, cache_read=0.1),
        ),
        "kimi-k2-0905-preview": ModelCapability(
            model_id="kimi-k2-0905-preview",
            context_limits=ModelLimits(
                max_request_tokens=256_000,
                max_response_tokens=32_768,
            ),
            prices=_prices(request=0.6, response=2.5, cache_read=0.15),
        ),
        "kimi-k2-0711-preview": ModelCapability(
            model_id="kimi-k2-0711-preview",
            context_limits=ModelLimits(
                max_request_tokens=131_072,
                max_response_tokens=32_768,
            ),
            prices=_prices(request=0.6, response=2.5, cache_read=0.15),
        ),
        "kimi-k2-turbo-preview": ModelCapability(
            model_id="kimi-k2-turbo-preview",
            context_limits=ModelLimits(
                max_request_tokens=256_000,
                max_response_tokens=32_768,
            ),
            prices=_prices(request=1.2, response=5.0, cache_read=0.3),
        ),
        "moonshot-v1-8k": ModelCapability(
            model_id="moonshot-v1-8k",
            context_limits=ModelLimits(
                max_request_tokens=8_000,
                max_response_tokens=16_384,
            ),
            prices=_prices(request=0.2, response=2.0),
        ),
        "moonshot-v1-32k": ModelCapability(
            model_id="moonshot-v1-32k",
            context_limits=ModelLimits(
                max_request_tokens=32_000,
                max_response_tokens=16_384,
            ),
            prices=_prices(request=0.4, response=4.0),
        ),
        "moonshot-v1-128k": ModelCapability(
            model_id="moonshot-v1-128k",
            context_limits=ModelLimits(
                max_request_tokens=128_000,
                max_response_tokens=16_384,
            ),
            prices=_prices(request=0.6, response=6.0),
        ),
    }
)
