"""llama.cpp (local; every rate is zero) model catalog, expressed as ``ModelCapability`` rows.

Source: n/a -- self-hosted

A row carries only what the MODEL can do; caching, retry, and auth mode
are transport facts declared on ``OpenAICompat.TRANSPORT``. These vendors
publish no image pixel or byte ceiling and preprocess images server-side,
so ``ModelLimits`` carries only the two token windows.
"""

from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType
from typing import TYPE_CHECKING

from sagent.types.capability import ContextTag, ModelCapability, ModelLimits
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
    # A local server bills nothing, but a missing price row would raise.
    default = ModelCapability(
        model_id="qwen3.6-27b-12gb",
        context=_limits(),
        prices=PriceCatalog(
            {
                PriceKey("auto"): TokenPrice(
                    request=0.0,
                    response=0.0,
                    cache_write=0.0,
                    cache_write_1h=0.0,
                    cache_read=0.0,
                ),
            },
        ),
    )
    rows = (
        default,
        replace(
            default,
            model_id="qwen3.6-27b-mtp-64k",
            context=_limits(max_tokens=65_536, output_tokens=4_096),
        ),
        replace(
            default,
            model_id="local",
            context=_limits(max_tokens=32_768, output_tokens=4_096),
        ),
    )
    # A tier is offered exactly when it is priced.
    rows = tuple(replace(row, service_tier=row.prices.service_tiers) for row in rows)
    catalog = {row.model_id: row for row in rows}
    return MappingProxyType(
        {
            "default": catalog[default.model_id],
            "utility": catalog[default.model_id],
            **catalog,
        },
    )


def _limits(
    *,
    max_tokens: int = 16_384,
    output_tokens: int = 1_024,
) -> Mapping[ContextTag, ModelLimits]:
    """One context tag: a local server ships no window variants."""
    return MappingProxyType(
        {
            "": ModelLimits(
                max_request_tokens=max_tokens,
                max_response_tokens=output_tokens,
            ),
        },
    )
