"""OpenAI wire behavior, transport restrictions, and subscription projection.

Load model facts with ``sagent.catalog.models("openai")``.
``subscription_models`` clamps their windows to the subscription contract.
Transport capabilities narrow either view with ``&``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import cache
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal

import math

from sagent.catalog.loader import models
from sagent.catalog.table import ModelTable
from sagent.types.capability import (
    ModelCapability,
    ThinkingCapability,
    ThinkingEffort,
)


if TYPE_CHECKING:
    from collections.abc import Mapping

    from sagent.types.cost import ServiceTier


__all__ = [
    "api",
    "compatible",
    "image_tokens",
    "keeps_reasoning_across_turns",
    "reasoning_effort",
    "served_tier",
    "subscription",
    "subscription_models",
    "tokenizer",
]


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
    row = models("openai").exact(model_id)
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
    row = models("openai").exact(model_id)
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
    row = models("openai").exact(model_id)
    return row is not None and row.model_id in _ALL_TURNS


def api() -> ModelCapability:
    """Return what the Responses API adds on top of a model row.

    One model, two authentication modes::

        models("openai")["gpt-5.6"] & api()          # API key: all four tiers
        models("openai")["gpt-5.6"] & subscription() # Codex: priority only, OAuth

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

        models("openai")["gpt-5.6"] & api()          # API key: all four tiers
        models("openai")["gpt-5.6"] & subscription() # Codex: priority only, OAuth

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
    for capability in models("openai").rows:
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
    return ModelTable(rows=tuple(rows), roles=models("openai").roles)


def served_tier(reported: str | None) -> ServiceTier | None:
    """Map the ``service_tier`` a response reports to the tier it bills at.

    Args:
      reported: The response's ``service_tier``; absent means standard.

    Returns:
      tier: Catalog tier whose card priced the request; ``None`` for a tier
        this catalog does not know. Not an error: the response is already paid
        for, so the caller decides which rate to fall back to.

    """
    tiers: dict[str | None, ServiceTier] = {
        None: "auto",
        "auto": "auto",
        "default": "auto",
        # Scale Tier is prepaid capacity, metered at the standard card.
        "scale": "auto",
        "flex": "flex",
        "priority": "priority",
        "fast": "priority",
        "ultrafast": "ultrafast",
    }
    return tiers.get(reported)


def _patch_count(width: int, height: int) -> int:
    return math.ceil(width / 32) * math.ceil(height / 32)


def _fit_patch_budget(width: int, height: int, budget: int) -> tuple[int, int]:
    """Shrink an image to the patch budget, as images-vision step B states it."""
    if budget <= 0 or _patch_count(width, height) <= budget:
        return width, height
    shrink = float(((32**2 * budget) / (width * height)) ** 0.5)
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
