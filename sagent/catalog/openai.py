"""OpenAI model catalog, expressed as ``ModelCapability`` rows.

Sources:
  - ModelLimits: https://developers.openai.com/api/docs/models/<model>
  - Pricing: https://developers.openai.com/api/docs/pricing
  - Images: https://developers.openai.com/api/docs/guides/images-vision

:func:`models` is the API-key view. :func:`subscription_models` projects the
subscription request and response limits; transport capabilities then narrow
either view with ``&``, which can only remove.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import cache
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal

import math

from sagent.catalog.table import ModelTable
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
    ServiceTier,
    TokenPrice,
)


if TYPE_CHECKING:
    from collections.abc import Mapping


__all__ = [
    "api",
    "compatible",
    "image_tokens",
    "keeps_reasoning_across_turns",
    "models",
    "reasoning_effort",
    "served_tier",
    "subscription",
    "subscription_models",
    "tokenizer",
]


_ROLES: Final = MappingProxyType({"default": "sol", "utility": "luna"})
"""Role names both tables resolve, to the family each picks."""

# "Input modalities: text" on each model page.
_TEXT_ONLY: Final = frozenset({"o-mini-3.0", "gpt-4.0"})


_ALL_TURNS: Final = frozenset(
    {
        "astra-6.0",
        "sol-6.1",
        "sol-6.0",
        "luna-6.0",
        "sol-5.6",
        "gpt-5.6",
        "terra-5.6",
        "luna-5.6",
    },
)


@dataclass(frozen=True, slots=True, kw_only=True)
class _Patches:
    """32x32-patch image tokenization (images-vision, "Patch-based")."""

    multiplier: float
    """Billable tokens per patch."""

    budget: int = 0
    """Patch budget ``detail: auto`` resizes into; ``0`` is none."""


@dataclass(frozen=True, slots=True, kw_only=True)
class _Tiles:
    """512px-tile image tokenization (images-vision, "Tile-based")."""

    base: int
    """Tokens every image costs."""

    tile: int
    """Tokens per 512px tile after the 2048px / 768px short-side resize."""


# images-vision multiplier and tile tables, and its sizing table for
# ``detail: auto`` (2026-10-03). GPT-6 Sol/Luna and 6.1 Sol are absent from
# both tables; they take Astra's GPT-6 rule.
_VISION: Final[Mapping[str, _Patches | _Tiles]] = MappingProxyType(
    {
        "astra-6.0": _Patches(multiplier=1.2),
        "sol-6.1": _Patches(multiplier=1.2),
        "sol-6.0": _Patches(multiplier=1.2),
        "luna-6.0": _Patches(multiplier=1.2),
        "sol-5.6": _Patches(multiplier=1.2),
        "gpt-5.6": _Patches(multiplier=1.2),
        "terra-5.6": _Patches(multiplier=1.2),
        "luna-5.6": _Patches(multiplier=1.2),
        "gpt-5.5": _Patches(multiplier=1.2, budget=10_000),
        "gpt-5.4": _Patches(multiplier=1.2, budget=2_500),
        "gpt-mini-5.4": _Patches(multiplier=1.2, budget=2_500),
        "gpt-nano-5.4": _Patches(multiplier=1.2, budget=2_500),
        "gpt-5.2": _Patches(multiplier=1.2, budget=6_144),
        "gpt-mini-4.1": _Patches(multiplier=1.62, budget=6_144),
        "gpt-nano-4.1": _Patches(multiplier=2.46, budget=6_144),
        "gpt-5.1": _Tiles(base=70, tile=140),
        "gpt-4.1": _Tiles(base=85, tile=170),
        "gpt-omni-4.0": _Tiles(base=85, tile=170),
        "gpt-omni-mini-4.0": _Tiles(base=2833, tile=5667),
        "o-1.0": _Tiles(base=75, tile=150),
    },
)


def compatible() -> ModelCapability:
    """Return the common Chat Completions transport capability.

    Returns:
      capability: Restrictions shared by OpenAI-compatible transports.

    """
    return ModelCapability(
        thinking=ThinkingCapability(
            effort=frozenset({"none", "min", "low", "medium", "high", "xhigh", "max"}),
            budget=frozenset({"none", "auto", "fixed"}),
            output=frozenset({"none", "text", "redacted"}),
        ),
        service_tier=frozenset({"auto"}),
        manage_context_server_side=frozenset({False}),
    )


# A row carries only what the MODEL can do; caching, retry, and auth mode are
# transport facts declared by :func:`api` / :func:`subscription`, since ``&`` can only
# remove. Every reasoning model takes an auto budget and returns readable text.
#
# Every card, window, cutoff, and effort set is the vendor's model page and
# pricing table (standard, flex, and fast tabs), verified 2026-10-03; a model
# missing from a tab does not offer that tier. Sol 5.6's rate is promotional
# "at least through November 21, 2026" and needs rechecking then. Batch is not
# modeled. Ultrafast is priced (Astra alone, 2026-10-05) so a response reporting
# it bills correctly, but no transport offers it for selection.
#
# ``max_request_tokens`` is the page's "Maximum input tokens" where it states
# one (922,000 under a 1,050,000 context window), not the context window: a
# prompt above it is rejected even though output could still fit.
# ``approx_chars_per_token`` measured from SERVER-reported ``usage.input_tokens``
# on 347k chars of real session text (2026-08-22), differencing out the
# per-request envelope. 3.71 across the whole range -- the catalog's prior
# flat 4.0 was the dataclass default, never measured, and over-credited
# every budget by ~7%. tiktoken says 3.81 locally; the gap is request
# framing the local tokenizer never sees.
@cache
def models() -> ModelTable:
    """Return every OpenAI model, as the API-key transport sees it.

    Returns:
      models: Capability per base model id.

    """
    # The reference row: every other model states only how it differs from it.
    # Every GPT-6 rule here was measured on this id alone, so a later GPT-6
    # whose contract differs does not inherit a limit nothing verified for it.
    default = ModelCapability(
        model_id="astra-6.0",
        wire_model_id="gpt-6-astra",
        knowledge_cutoff="April 30, 2026",
        approx_chars_per_token=3.71,
        context=_limits(image_edge_px=65_535),
        prices=_prices(
            {
                "auto": _card(
                    request=10.0,
                    response=50.0,
                    cache_write=12.5,
                    cache_read=1.0,
                ),
                "flex": _card(
                    request=5.0,
                    response=25.0,
                    cache_write=6.25,
                    cache_read=0.5,
                ),
                "priority": _card(
                    request=20.0,
                    response=100.0,
                    cache_write=25.0,
                    cache_read=2.0,
                ),
                "ultrafast": _card(
                    request=60.0,
                    response=300.0,
                    cache_write=75.0,
                    cache_read=6.0,
                ),
            },
            long_context=True,
        ),
        # "`reasoning.effort` supports `low`, `medium`, `high`, `xhigh`, and
        # `max`" -- no ``none``, and no model page lists ``minimal``.
        thinking=ThinkingCapability(
            effort=frozenset({"low", "medium", "high", "xhigh", "max"}),
            budget=frozenset({"none", "auto"}),
            output=frozenset({"none", "text"}),
        ),
    )
    # GPT-6 Sol/Luna and GPT-5.6: "none, low, medium (default), high, xhigh, and max".
    full = replace(
        default.thinking,
        effort=frozenset({"none", "low", "medium", "high", "xhigh", "max"}),
    )
    # GPT-5.2 through 5.5: "none, low, medium, high and xhigh".
    legacy = replace(
        default.thinking,
        effort=frozenset({"none", "low", "medium", "high", "xhigh"}),
    )
    # The Pro models: "medium, high, xhigh".
    pro = replace(default.thinking, effort=frozenset({"medium", "high", "xhigh"}))
    rows = (
        default,
        replace(
            default,
            model_id="sol-6.1",
            wire_model_id="gpt-6.1-sol",
            # "The `none` and `minimal` reasoning efforts are not supported":
            # Astra's effort set, inherited along with its cutoff.
            prices=_prices(
                {
                    # "Cached input tokens are priced at 5% of the uncached
                    # input token rate" -- half of every other GPT-6 card.
                    "auto": _card(
                        request=2.0,
                        response=10.0,
                        cache_write=2.5,
                        cache_read=0.1,
                    ),
                    "flex": _card(
                        request=1.0,
                        response=5.0,
                        cache_write=1.25,
                        cache_read=0.05,
                    ),
                    "priority": _card(
                        request=4.0,
                        response=20.0,
                        cache_write=5.0,
                        cache_read=0.2,
                    ),
                },
                long_context=True,
            ),
        ),
        replace(
            default,
            model_id="sol-6.0",
            wire_model_id="gpt-6-sol",
            knowledge_cutoff="April 20, 2026",
            # Absent from the images-vision sizing table: no resize edge published.
            context=_limits(),
            prices=_prices(
                {
                    "auto": _card(
                        request=2.0,
                        response=10.0,
                        cache_write=2.5,
                        cache_read=0.2,
                    ),
                    "flex": _card(
                        request=1.0,
                        response=5.0,
                        cache_write=1.25,
                        cache_read=0.1,
                    ),
                    "priority": _card(
                        request=4.0,
                        response=20.0,
                        cache_write=5.0,
                        cache_read=0.4,
                    ),
                },
                long_context=True,
            ),
            thinking=full,
        ),
        replace(
            default,
            model_id="luna-6.0",
            wire_model_id="gpt-6-luna",
            knowledge_cutoff="May 18, 2026",
            # Absent from the images-vision sizing table: no resize edge published.
            context=_limits(),
            prices=_prices(
                {
                    "auto": _card(
                        request=0.1,
                        response=0.5,
                        cache_write=0.125,
                        cache_read=0.01,
                    ),
                    "flex": _card(
                        request=0.05,
                        response=0.25,
                        cache_write=0.0625,
                        cache_read=0.005,
                    ),
                    "priority": _card(
                        request=0.2,
                        response=1.0,
                        cache_write=0.25,
                        cache_read=0.02,
                    ),
                },
                long_context=True,
            ),
            thinking=full,
        ),
        replace(
            default,
            model_id="sol-5.6",
            wire_model_id="gpt-5.6-sol",
            knowledge_cutoff="February 16, 2026",
            prices=_prices(
                {
                    "auto": _card(
                        request=4.0,
                        response=20.0,
                        cache_write=5.0,
                        cache_read=0.4,
                    ),
                    "flex": _card(
                        request=2.0,
                        response=10.0,
                        cache_write=2.5,
                        cache_read=0.2,
                    ),
                    "priority": _card(
                        request=8.0,
                        response=40.0,
                        cache_write=10.0,
                        cache_read=0.8,
                    ),
                },
                long_context=True,
            ),
            thinking=full,
        ),
        # Absent from ``GET /v1/models`` yet serves ``POST /v1/responses``
        # normally. Listing is an entitlement view, not the model set, so a
        # row must be dropped only on ``model_not_found`` from a real call.
        replace(
            default,
            model_id="gpt-5.6",
            wire_model_id="gpt-5.6",
            knowledge_cutoff="February 16, 2026",
            prices=_prices(
                {
                    "auto": _card(
                        request=4.0,
                        response=20.0,
                        cache_write=5.0,
                        cache_read=0.4,
                    ),
                    "flex": _card(
                        request=2.0,
                        response=10.0,
                        cache_write=2.5,
                        cache_read=0.2,
                    ),
                    "priority": _card(
                        request=8.0,
                        response=40.0,
                        cache_write=10.0,
                        cache_read=0.8,
                    ),
                },
                long_context=True,
            ),
            thinking=full,
        ),
        replace(
            default,
            model_id="luna-5.6",
            wire_model_id="gpt-5.6-luna",
            knowledge_cutoff="February 16, 2026",
            prices=_prices(
                {
                    "auto": _card(
                        request=0.2,
                        response=1.2,
                        cache_write=0.25,
                        cache_read=0.02,
                    ),
                    "flex": _card(
                        request=0.1,
                        response=0.6,
                        cache_write=0.125,
                        cache_read=0.01,
                    ),
                    "priority": _card(
                        request=0.4,
                        response=2.4,
                        cache_write=0.5,
                        cache_read=0.04,
                    ),
                },
                long_context=True,
            ),
            thinking=full,
        ),
        replace(
            default,
            model_id="terra-5.6",
            wire_model_id="gpt-5.6-terra",
            knowledge_cutoff="February 16, 2026",
            prices=_prices(
                {
                    "auto": _card(
                        request=2.0,
                        response=12.0,
                        cache_write=2.5,
                        cache_read=0.2,
                    ),
                    "flex": _card(
                        request=1.0,
                        response=6.0,
                        cache_write=1.25,
                        cache_read=0.1,
                    ),
                    "priority": _card(
                        request=4.0,
                        response=24.0,
                        cache_write=5.0,
                        cache_read=0.4,
                    ),
                },
                long_context=True,
            ),
            thinking=full,
        ),
        replace(
            default,
            model_id="gpt-5.5",
            wire_model_id="gpt-5.5",
            knowledge_cutoff="December 1, 2025",
            context=_limits(max_tokens=1_050_000, image_edge_px=6000),
            prices=_prices(
                {
                    "auto": _card(
                        request=5.0,
                        response=30.0,
                        cache_write=0.0,
                        cache_read=0.5,
                    ),
                    "flex": _card(
                        request=2.5,
                        response=15.0,
                        cache_write=0.0,
                        cache_read=0.25,
                    ),
                    "priority": _card(
                        request=12.5,
                        response=75.0,
                        cache_write=0.0,
                        cache_read=1.25,
                    ),
                },
                long_context=True,
                short_only=frozenset({"priority"}),
            ),
            thinking=legacy,
        ),
        replace(
            default,
            model_id="gpt-pro-5.5",
            wire_model_id="gpt-5.5-pro",
            knowledge_cutoff="December 1, 2025",
            context=_limits(max_tokens=1_050_000),
            prices=_prices(
                {
                    "auto": _card(
                        request=30.0,
                        response=180.0,
                        cache_write=0.0,
                        cache_read=0.0,
                    ),
                    "flex": _card(
                        request=15.0,
                        response=90.0,
                        cache_write=0.0,
                        cache_read=0.0,
                    ),
                },
                long_context=True,
                short_only=frozenset({"flex"}),
            ),
            thinking=pro,
        ),
        replace(
            default,
            model_id="gpt-5.4",
            wire_model_id="gpt-5.4",
            knowledge_cutoff="August 31, 2025",
            # "auto uses the same sizing behavior as high": 2048px.
            context=_limits(max_tokens=1_050_000, image_edge_px=2048),
            prices=_prices(
                {
                    "auto": _card(
                        request=2.5,
                        response=15.0,
                        cache_write=0.0,
                        cache_read=0.25,
                    ),
                    "flex": _card(
                        request=1.25,
                        response=7.5,
                        cache_write=0.0,
                        cache_read=0.13,
                    ),
                    "priority": _card(
                        request=5.0,
                        response=30.0,
                        cache_write=0.0,
                        cache_read=0.5,
                    ),
                },
                long_context=True,
                short_only=frozenset({"priority"}),
                # Flex long: "$2.50 | $0.25 | - | $11.25" -- the cached rate is
                # rounded, so 2x the short card's 0.13 would bill 0.26.
                long_overrides={
                    "flex": _card(
                        request=2.5,
                        response=11.25,
                        cache_write=0.0,
                        cache_read=0.25,
                    ),
                },
            ),
            thinking=legacy,
        ),
        replace(
            default,
            model_id="gpt-pro-5.4",
            wire_model_id="gpt-5.4-pro",
            knowledge_cutoff="August 31, 2025",
            context=_limits(max_tokens=1_050_000),
            prices=_prices(
                {
                    "auto": _card(
                        request=30.0,
                        response=180.0,
                        cache_write=0.0,
                        cache_read=0.0,
                    ),
                    "flex": _card(
                        request=15.0,
                        response=90.0,
                        cache_write=0.0,
                        cache_read=0.0,
                    ),
                },
                long_context=True,
            ),
            thinking=pro,
        ),
        replace(
            default,
            model_id="gpt-mini-5.4",
            wire_model_id="gpt-5.4-mini",
            knowledge_cutoff="August 31, 2025",
            # "400,000 context window - Maximum input tokens: 272,000".
            context=_limits(max_tokens=272_000, windowed=False, image_edge_px=2048),
            prices=_prices(
                {
                    "auto": _card(
                        request=0.75,
                        response=4.5,
                        cache_write=0.0,
                        cache_read=0.075,
                    ),
                    "flex": _card(
                        request=0.375,
                        response=2.25,
                        cache_write=0.0,
                        cache_read=0.0375,
                    ),
                    "priority": _card(
                        request=1.5,
                        response=9.0,
                        cache_write=0.0,
                        cache_read=0.15,
                    ),
                },
            ),
            thinking=legacy,
        ),
        replace(
            default,
            model_id="gpt-nano-5.4",
            wire_model_id="gpt-5.4-nano",
            knowledge_cutoff="August 31, 2025",
            context=_limits(max_tokens=272_000, windowed=False, image_edge_px=2048),
            prices=_prices(
                {
                    "auto": _card(
                        request=0.2,
                        response=1.25,
                        cache_write=0.0,
                        cache_read=0.02,
                    ),
                    "flex": _card(
                        request=0.1,
                        response=0.625,
                        cache_write=0.0,
                        cache_read=0.01,
                    ),
                },
            ),
            thinking=legacy,
        ),
        # No `*-chat-latest` row. Those aliases are listed by `/v1/models` but
        # rejected by `/v1/responses` with `model_not_found` (verified for
        # gpt-5, gpt-5.2 and gpt-5.3 variants), and the Responses API is the
        # only one this provider speaks -- so a row for one is a model no
        # caller here can reach.
        replace(
            default,
            model_id="gpt-5.2",
            wire_model_id="gpt-5.2",
            knowledge_cutoff="August 31, 2025",
            context=_limits(max_tokens=400_000, windowed=False, image_edge_px=2048),
            prices=_prices(
                {
                    "auto": _card(
                        request=1.75,
                        response=14.0,
                        cache_write=0.0,
                        cache_read=0.175,
                    ),
                    "flex": _card(
                        request=0.875,
                        response=7.0,
                        cache_write=0.0,
                        cache_read=0.0875,
                    ),
                    "priority": _card(
                        request=3.5,
                        response=28.0,
                        cache_write=0.0,
                        cache_read=0.35,
                    ),
                },
            ),
            thinking=legacy,
        ),
        replace(
            default,
            model_id="gpt-pro-5.2",
            wire_model_id="gpt-5.2-pro",
            knowledge_cutoff="August 31, 2025",
            context=_limits(max_tokens=400_000, windowed=False),
            prices=_prices(
                {
                    "auto": _card(
                        request=21.0,
                        response=168.0,
                        cache_write=0.0,
                        cache_read=0.0,
                    ),
                },
            ),
            thinking=pro,
        ),
        replace(
            default,
            model_id="gpt-5.1",
            wire_model_id="gpt-5.1",
            knowledge_cutoff="September 30, 2024",
            context=_limits(max_tokens=400_000, windowed=False, image_edge_px=2048),
            prices=_prices(
                {
                    "auto": _card(
                        request=1.25,
                        response=10.0,
                        cache_write=0.0,
                        cache_read=0.125,
                    ),
                    "flex": _card(
                        request=0.625,
                        response=5.0,
                        cache_write=0.0,
                        cache_read=0.0625,
                    ),
                    "priority": _card(
                        request=2.5,
                        response=20.0,
                        cache_write=0.0,
                        cache_read=0.25,
                    ),
                },
            ),
            # "none (default), low, medium, and high".
            thinking=replace(
                default.thinking,
                effort=frozenset({"none", "low", "medium", "high"}),
            ),
        ),
        # o1, o3-mini, gpt-4.1-nano, gpt-4-turbo, and gpt-4 shut down October
        # 23, 2026 (https://developers.openai.com/api/docs/deprecations).
        replace(
            default,
            model_id="o-1.0",
            wire_model_id="o1",
            knowledge_cutoff="October 1, 2023",
            context=_limits(
                max_tokens=200_000,
                output_tokens=100_000,
                windowed=False,
                image_edge_px=2048,
            ),
            prices=_prices(
                {
                    "auto": _card(
                        request=15.0,
                        response=60.0,
                        cache_write=0.0,
                        cache_read=7.5,
                    ),
                },
            ),
            thinking=replace(
                default.thinking,
                effort=frozenset({"low", "medium", "high"}),
            ),
        ),
        replace(
            default,
            model_id="o-mini-3.0",
            wire_model_id="o3-mini",
            knowledge_cutoff="October 1, 2023",
            # "Input modalities: text" -- no image input, so no image limits.
            context=_limits(
                max_tokens=200_000,
                output_tokens=100_000,
                windowed=False,
            ),
            prices=_prices(
                {
                    "auto": _card(
                        request=1.1,
                        response=4.4,
                        cache_write=0.0,
                        cache_read=0.55,
                    ),
                },
            ),
            thinking=replace(
                default.thinking,
                effort=frozenset({"low", "medium", "high"}),
            ),
        ),
        # No reasoning knob on the 4-x generation, so every thinking axis
        # offers only its off value.
        replace(
            default,
            model_id="gpt-4.1",
            wire_model_id="gpt-4.1",
            knowledge_cutoff="June 1, 2024",
            context=_limits(
                max_tokens=1_047_576,
                output_tokens=32_768,
                windowed=False,
                image_edge_px=2048,
            ),
            prices=_prices(
                {
                    "auto": _card(
                        request=2.0,
                        response=8.0,
                        cache_write=0.0,
                        cache_read=0.5,
                    ),
                    "priority": _card(
                        request=3.5,
                        response=14.0,
                        cache_write=0.0,
                        cache_read=0.875,
                    ),
                },
            ),
            thinking=ThinkingCapability(),
        ),
        replace(
            default,
            model_id="gpt-mini-4.1",
            wire_model_id="gpt-4.1-mini",
            knowledge_cutoff="June 1, 2024",
            context=_limits(
                max_tokens=1_047_576,
                output_tokens=32_768,
                windowed=False,
                image_edge_px=2048,
            ),
            prices=_prices(
                {
                    "auto": _card(
                        request=0.4,
                        response=1.6,
                        cache_write=0.0,
                        cache_read=0.1,
                    ),
                    "priority": _card(
                        request=0.7,
                        response=2.8,
                        cache_write=0.0,
                        cache_read=0.175,
                    ),
                },
            ),
            thinking=ThinkingCapability(),
        ),
        replace(
            default,
            model_id="gpt-nano-4.1",
            wire_model_id="gpt-4.1-nano",
            knowledge_cutoff="June 1, 2024",
            context=_limits(
                max_tokens=1_047_576,
                output_tokens=32_768,
                windowed=False,
                image_edge_px=2048,
            ),
            prices=_prices(
                {
                    "auto": _card(
                        request=0.1,
                        response=0.4,
                        cache_write=0.0,
                        cache_read=0.025,
                    ),
                    "priority": _card(
                        request=0.2,
                        response=0.8,
                        cache_write=0.0,
                        cache_read=0.05,
                    ),
                },
            ),
            thinking=ThinkingCapability(),
        ),
        replace(
            default,
            model_id="gpt-omni-4.0",
            wire_model_id="gpt-4o",
            knowledge_cutoff="October 1, 2023",
            context=_limits(
                max_tokens=128_000,
                output_tokens=16_384,
                windowed=False,
                image_edge_px=2048,
            ),
            prices=_prices(
                {
                    "auto": _card(
                        request=2.5,
                        response=10.0,
                        cache_write=0.0,
                        cache_read=1.25,
                    ),
                    "priority": _card(
                        request=4.25,
                        response=17.0,
                        cache_write=0.0,
                        cache_read=2.125,
                    ),
                },
            ),
            thinking=ThinkingCapability(),
        ),
        replace(
            default,
            model_id="gpt-omni-mini-4.0",
            wire_model_id="gpt-4o-mini",
            knowledge_cutoff="October 1, 2023",
            context=_limits(
                max_tokens=128_000,
                output_tokens=16_384,
                windowed=False,
                image_edge_px=2048,
            ),
            prices=_prices(
                {
                    "auto": _card(
                        request=0.15,
                        response=0.6,
                        cache_write=0.0,
                        cache_read=0.075,
                    ),
                    "priority": _card(
                        request=0.25,
                        response=1.0,
                        cache_write=0.0,
                        cache_read=0.125,
                    ),
                },
            ),
            thinking=ThinkingCapability(),
        ),
        replace(
            default,
            model_id="gpt-turbo-4.0",
            wire_model_id="gpt-4-turbo",
            knowledge_cutoff="December 1, 2023",
            context=_limits(
                max_tokens=128_000,
                output_tokens=4_096,
                windowed=False,
                image_edge_px=2048,
            ),
            prices=_prices(
                {
                    "auto": _card(
                        request=10.0,
                        response=30.0,
                        cache_write=0.0,
                        cache_read=0.0,
                    ),
                },
            ),
            thinking=ThinkingCapability(),
        ),
        replace(
            default,
            model_id="gpt-4.0",
            wire_model_id="gpt-4",
            knowledge_cutoff="December 1, 2023",
            # Text-only, like o3-mini.
            context=_limits(
                max_tokens=8_192,
                output_tokens=8_192,
                windowed=False,
            ),
            prices=_prices(
                {
                    "auto": _card(
                        request=30.0,
                        response=60.0,
                        cache_write=0.0,
                        cache_read=0.0,
                    ),
                },
            ),
            thinking=ThinkingCapability(),
        ),
    )
    # A tier is offered exactly when it is priced.
    rows = tuple(replace(row, service_tier=row.prices.service_tiers) for row in rows)
    return ModelTable(rows=rows, roles=_ROLES)


def reasoning_effort(
    effort: ThinkingEffort,
) -> Literal["low", "medium", "high", "xhigh", "max"]:
    """Return the Responses ``reasoning.effort`` a selected level sends.

    Every level a row offers is spelled as the API spells it; the row has
    already rejected any level the model does not take.

    Args:
      effort: Selected catalog effort.

    Returns:
      wire: Responses reasoning effort.

    Raises:
      ValueError: ``effort`` is ``none``, which omits ``reasoning`` instead,
        or ``min``, which no model page lists.

    """
    match effort:
        case "none" | "min":
            raise ValueError(f"effort {effort!r} is not sent as reasoning.effort")
        case "low" | "medium" | "high" | "xhigh" | "max":
            return effort


def image_tokens(model_id: str, width: int, height: int) -> int:
    """Estimate the input tokens one image bills under ``detail: auto``.

    Args:
      model_id: Catalog name or wire id; an id no row carries is estimated too.
      width: Image width after the transport's own resize, in pixels.
      height: Image height after the transport's own resize, in pixels.

    Returns:
      tokens: The published formula's count; the GPT-4o tile estimate for a
        model that takes images but has no published formula (the Pro models,
        ``gpt-4-turbo``, other vendors' ids); ``0`` for a text-only row.

    References:
      https://developers.openai.com/api/docs/guides/images-vision

    """
    row = models().exact(model_id)
    name = row.model_id if row is not None else ""
    if name in _TEXT_ONLY:
        return 0
    match _VISION.get(name, _Tiles(base=85, tile=170)):
        case _Patches(multiplier=multiplier, budget=budget):
            width, height = _fit_patch_budget(width, height, budget)
            return math.ceil(_patch_count(width, height) * multiplier)
        case _Tiles(base=base, tile=tile):
            width, height = _fit_tiles(width, height)
            return base + tile * math.ceil(width / 512) * math.ceil(height / 512)


def tokenizer(model_id: str) -> str | None:
    """Return the ``tiktoken`` encoding a model's text tokenizes with.

    Args:
      model_id: Catalog name or wire id.

    Returns:
      encoding: ``o200k_base`` for GPT-4o and later, ``cl100k_base`` for the
        GPT-4 generation, ``None`` for an id no row carries.

    """
    row = models().exact(model_id)
    if row is None:
        return None
    return (
        "cl100k_base" if row.model_id in {"gpt-4.0", "gpt-turbo-4.0"} else "o200k_base"
    )


def keeps_reasoning_across_turns(model_id: str) -> bool:
    """Whether a request asks the model to reason over every prior turn.

    Args:
      model_id: Catalog name or wire id.

    Returns:
      all_turns: True for GPT-5.6 and GPT-6, which take
        ``reasoning.context = "all_turns"``.

    """
    row = models().exact(model_id)
    return row is not None and row.model_id in _ALL_TURNS


def api() -> ModelCapability:
    """Return what the Responses API adds on top of a model row.

    One model, two authentication modes::

        models()["gpt-5.6"] & api()          # API key: all four tiers
        models()["gpt-5.6"] & subscription() # Codex: priority only, OAuth

    Returns:
      capability: The API transport's restrictions.

    """
    return ModelCapability(
        thinking=ThinkingCapability(
            effort=frozenset({"none", "min", "low", "medium", "high", "xhigh", "max"}),
            budget=frozenset({"none", "auto", "fixed"}),
            output=frozenset({"none", "text", "redacted"}),
        ),
        service_tier=frozenset({"auto", "default", "flex", "priority", "ultrafast"}),
    )


def subscription() -> ModelCapability:
    """Return what Codex billing narrows a model row to.

    One model, two authentication modes::

        models()["gpt-5.6"] & api()          # API key: all four tiers
        models()["gpt-5.6"] & subscription() # Codex: priority only, OAuth

    Returns:
      capability: The subscription transport's restrictions.

    """
    # ``auto`` stays selectable: ``/fast`` picks ``priority``, but an
    # unqualified request still sends the default, so dropping it would make
    # ``ModelSettings()`` invalid on every Codex model.
    return ModelCapability(
        thinking=ThinkingCapability(
            effort=frozenset({"none", "min", "low", "medium", "high", "xhigh", "max"}),
            budget=frozenset({"none", "auto", "fixed"}),
            output=frozenset({"none", "text", "redacted"}),
        ),
        service_tier=frozenset({"auto", "priority"}),
        account_auth=True,
    )


@cache
def subscription_models() -> ModelTable:
    """Return catalog rows clamped to the ChatGPT backend contract.

    Returns:
      models: Subscription-visible rows with only their usable context.

    """
    rows: list[ModelCapability] = []
    for capability in models().rows:
        limits = capability.context[""]
        rows.append(
            replace(
                capability,
                context=MappingProxyType(
                    {
                        "": replace(
                            limits,
                            max_request_tokens=min(limits.max_request_tokens, 272_000),
                            max_response_tokens=min(limits.max_response_tokens, 32_000),
                        ),
                    },
                ),
            ),
        )
    return ModelTable(rows=tuple(rows), roles=_ROLES)


def served_tier(reported: str | None) -> ServiceTier | None:
    """Map the ``service_tier`` a response reports to the tier it bills at.

    Args:
      reported: The response's ``service_tier``; absent means standard.

    Returns:
      tier: Catalog tier whose card priced the request; ``None`` for a tier
        this catalog does not know. Not an error: the response is already paid
        for, so the caller decides which rate to fall back to.

    """
    match reported:
        # Scale Tier is prepaid capacity, metered at the standard card.
        case None | "auto" | "default" | "scale":
            return "auto"
        case "flex":
            return "flex"
        case "priority" | "fast":
            return "priority"
        case "ultrafast":
            return "ultrafast"
        case _:
            return None


def _card(
    *,
    request: float,
    response: float,
    cache_write: float,
    cache_read: float,
) -> TokenPrice:
    """Return one published price-table row; OpenAI has one cache-write lifetime."""
    return TokenPrice(
        request=request,
        response=response,
        cache_write=cache_write,
        cache_write_1h=0.0,
        cache_read=cache_read,
    )


# "Prompts with >272K input tokens are priced at 2x input and 1.5x output for
# the full request" -- every 1.05M-window model page states it as this ratio.
# The pricing tables publish that column per tier and leave it "-" for some
# (the fast tier of GPT-5.5 and GPT-5.4, the flex tier of GPT-5.5 Pro), so
# ``short_only`` names a tier that has no long band. ``long_overrides``
# carries a published long card the ratio does not reproduce exactly.
def _prices(
    tiers: Mapping[ServiceTier, TokenPrice],
    *,
    long_context: bool = False,
    short_only: frozenset[ServiceTier] = frozenset(),
    long_overrides: Mapping[ServiceTier, TokenPrice] | None = None,
) -> PriceCatalog:
    """Return each tier's published card, plus its >272K band when it has one."""
    cards = {PriceKey(tier): card for tier, card in tiers.items()}
    if long_context:
        cards |= {
            PriceKey(tier, 272_000): replace(
                card,
                request=card.request * 2.0,
                cache_write=card.cache_write * 2.0,
                cache_read=card.cache_read * 2.0,
                response=card.response * 1.5,
            )
            for tier, card in tiers.items()
            if tier not in short_only
        }
        cards |= {
            PriceKey(tier, 272_000): card
            for tier, card in (long_overrides or {}).items()
        }
    return PriceCatalog(cards)


# Every current model takes "Up to 512 MB total payload per request"
# (images-vision); no per-image byte cap is published. ``image_edge_px`` is the
# long edge ``detail: auto`` resizes to, per that page's "Model sizing
# behavior" table; ``0`` means the page states none for the model.
def _limits(
    *,
    max_tokens: int = 922_000,
    output_tokens: int = 128_000,
    windowed: bool = True,
    image_edge_px: int = 0,
) -> Mapping[ContextTag, ModelLimits]:
    """Serve ``max_tokens`` by default; ``+272k`` selects the cap below the surcharge."""
    limits = ModelLimits(
        max_request_tokens=max_tokens,
        max_response_tokens=output_tokens,
        max_request_bytes=512 * 1024 * 1024,
        max_image_edge_px=image_edge_px,
    )
    context: dict[ContextTag, ModelLimits] = {"": limits}
    if windowed:
        context["+272k"] = replace(limits, max_request_tokens=272_000)
    return MappingProxyType(context)


def _patch_count(width: int, height: int) -> int:
    return math.ceil(width / 32) * math.ceil(height / 32)


def _fit_patch_budget(width: int, height: int, budget: int) -> tuple[int, int]:
    """Shrink an image to the patch budget, as images-vision step B states it."""
    if budget <= 0 or _patch_count(width, height) <= budget:
        return width, height
    shrink = ((32**2 * budget) / (width * height)) ** 0.5
    adjusted = shrink * min(
        math.floor(width * shrink / 32) / (width * shrink / 32),
        math.floor(height * shrink / 32) / (height * shrink / 32),
    )
    return max(1, math.floor(width * adjusted)), max(1, math.floor(height * adjusted))


def _fit_tiles(width: int, height: int) -> tuple[int, int]:
    """Fit a 2048px square, then a 768px short side, as the tile rules state."""
    scale = min(1.0, 2048 / max(width, height))
    width, height = math.floor(width * scale), math.floor(height * scale)
    short = min(width, height)
    if short > 768:
        width, height = (
            math.floor(width * 768 / short),
            math.floor(height * 768 / short),
        )
    return max(1, width), max(1, height)
