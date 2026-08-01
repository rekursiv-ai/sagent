"""llama.cpp (local; every rate is zero) model catalog, expressed as ``ModelCapability`` rows.

Source: n/a -- self-hosted

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


def _free() -> PriceCatalog:
    """A local server bills nothing, but a missing row would raise."""
    return PriceCatalog({PriceCatalogProduct(): TokenPrice()})


MODELS: Mapping[str, ModelCapability] = MappingProxyType(
    {
        "qwen3.6-27b-12gb": ModelCapability(
            model_id="qwen3.6-27b-12gb",
            context_limits=Limits(
                max_request_tokens=16_384,
                max_response_tokens=1_024,
            ),
            prices=_free(),
        ),
        "qwen3.6-27b-mtp-64k": ModelCapability(
            model_id="qwen3.6-27b-mtp-64k",
            context_limits=Limits(
                max_request_tokens=65_536,
                max_response_tokens=4_096,
            ),
            prices=_free(),
        ),
        "local": ModelCapability(
            model_id="local",
            context_limits=Limits(
                max_request_tokens=32_768,
                max_response_tokens=4_096,
            ),
            prices=_free(),
        ),
    }
)
