"""llama.cpp (local; every rate is zero) model catalog, expressed as ``ModelCapability`` rows.

Source: n/a -- self-hosted

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
    # A local server bills nothing, but a missing price row would raise.
    local = ModelCapability(
        context=_limits(window=16_384, response=1_024),
        prices=PriceCatalog({PriceCatalogProduct(): TokenPrice()}),
    )
    rows = (
        replace(local, model_id="qwen3.6-27b-12gb"),
        replace(
            local,
            model_id="qwen3.6-27b-mtp-64k",
            context=_limits(window=65_536, response=4_096),
        ),
        replace(
            local,
            model_id="local",
            context=_limits(window=32_768, response=4_096),
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
