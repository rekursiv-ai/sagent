"""Estimate a request's tokens locally, where no server counts them.

OpenAI's models tokenize with ``tiktoken``; another vendor's text is its length
over the row's characters per token. Images cost what OpenAI's published
formula for the model says, after the transport's own shrink.

Typical usage example:

  tokens = estimate(request, capability=c, encoding="o200k_base")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import math

from treekle import json_dumps, mutable

from sagent.model import image
from sagent.providers.openai import wire
from sagent.session.types import AssistantMessage, Thinking, ToolCall, ToolResult


if TYPE_CHECKING:
    from collections.abc import Iterator

    import tiktoken

    from sagent.model.capability import ModelCapability
    from sagent.model.types import Request
    from sagent.session.types import Attachment
else:
    from wrapt import lazy_import

    tiktoken = lazy_import("tiktoken")


__all__ = [
    "attachment_tokens",
    "estimate",
    "image_tokens",
]


def estimate(
    request: Request,
    *,
    capability: ModelCapability,
    encoding: str = "",
) -> int:
    """Return about how many tokens ``request`` would send.

    Args:
      request: What the model would be asked.
      capability: The model it would be asked of, whose limits each image is
        shrunk to.
      encoding: The ``tiktoken`` encoding the model reads text with; empty
        divides by the row's characters per token instead.

    Returns:
      tokens: The text's tokens plus each image's.

    """
    text = "\n".join(_texts(request))
    if encoding:
        # Ordinary: a prompt that spells a special token reads it as text.
        tokens = len(tiktoken.get_encoding(encoding).encode_ordinary(text))
    else:
        tokens = int(len(text) / capability.approx_chars_per_token)
    return tokens + sum(
        attachment_tokens(attachment, capability=capability)
        for attachment in _attachments(request)
    )


def attachment_tokens(attachment: Attachment, *, capability: ModelCapability) -> int:
    """Return the input tokens one attachment bills.

    Args:
      attachment: What is attached.
      capability: The model, whose formula bills an image and whose limits it
        is shrunk to.

    Returns:
      tokens: The image's, by :func:`image_tokens`; ``0`` for one not an image.

    """
    if not attachment.mime_descriptor.startswith("image/"):
        return 0
    width, height = image.size(
        attachment.data,
        max_edge_px=capability.limits.max_image_edge_px,
    )
    return image_tokens(capability.model_id, width, height) if width else 0


def image_tokens(model_id: str, width: int, height: int) -> int:
    """Return the input tokens one image bills under ``detail: auto``.

    Args:
      model_id: The model's exact name; an unknown one is estimated too.
      width: The image's width after the transport's shrink, in pixels.
      height: Its height, likewise.

    Returns:
      tokens: The published formula's count; GPT-4o's tile estimate for a model
        with no published formula.

    References:
      https://developers.openai.com/api/docs/guides/images-vision

    """
    rule = _vision(model_id)
    if isinstance(rule, _Patches):
        width, height = _fit_patches(width, height, budget=rule.budget)
        return math.ceil(_patches(width, height) * rule.multiplier)
    width, height = _fit_tiles(width, height)
    return rule.base + rule.tile * math.ceil(width / 512) * math.ceil(height / 512)


@dataclass(frozen=True, slots=True, kw_only=True)
class _Patches:
    """32x32-patch image tokenization."""

    multiplier: float
    """Billable tokens per patch."""

    budget: int = 0
    """Patches ``detail: auto`` shrinks an image into; ``0`` is no budget."""


@dataclass(frozen=True, slots=True, kw_only=True)
class _Tiles:
    """512px-tile image tokenization."""

    base: int
    """Tokens every image costs."""

    tile: int
    """Tokens per 512px tile, after the 2048px and 768px short-side shrinks."""


# The images-vision guide's tables (2026-10-03). GPT-6 Sol and Luna and 6.1 Sol
# are in neither; they take Astra's GPT-6 rule.
def _vision(model_id: str) -> _Patches | _Tiles:
    """Return how ``model_id`` bills an image."""
    gpt_6 = _Patches(multiplier=1.2)
    rules: dict[str, _Patches | _Tiles] = {
        "astra-6.0": gpt_6,
        "sol-6.1": gpt_6,
        "sol-6.0": gpt_6,
        "luna-6.0": gpt_6,
        "sol-5.6": gpt_6,
        "gpt-5.6": gpt_6,
        "terra-5.6": gpt_6,
        "luna-5.6": gpt_6,
        "gpt-5.5": _Patches(multiplier=1.2, budget=10_000),
        "gpt-5.4": _Patches(multiplier=1.2, budget=2_500),
        "gpt-mini-5.4": _Patches(multiplier=1.2, budget=2_500),
        "gpt-nano-5.4": _Patches(multiplier=1.2, budget=2_500),
        "gpt-5.2": _Patches(multiplier=1.2, budget=6_144),
        "gpt-mini-4.1": _Patches(multiplier=1.62, budget=6_144),
        "gpt-5.1": _Tiles(base=70, tile=140),
        "gpt-4.1": _Tiles(base=85, tile=170),
        "gpt-omni-4.0": _Tiles(base=85, tile=170),
        "gpt-omni-mini-4.0": _Tiles(base=2833, tile=5667),
    }
    return rules.get(model_id, _Tiles(base=85, tile=170))


def _texts(request: Request) -> Iterator[str]:
    """Yield every text the request sends: prompt, tools, and conversation."""
    yield from request.system
    for tool in request.tools:
        yield tool.name
        yield tool.description
        yield json_dumps(mutable(tool.schema))
    for record in wire.flatten(request.records):
        said = wire.user_content(record)
        if said is not None:
            yield said[0]
        elif isinstance(record, AssistantMessage):
            yield record.content or ""
        elif isinstance(record, Thinking):
            yield record.summary or record.content or ""
        elif isinstance(record, ToolCall):
            yield record.name
            yield json_dumps(mutable(record.arguments))
        elif isinstance(record, ToolResult):
            yield record.text()


def _attachments(request: Request) -> Iterator[Attachment]:
    for record in wire.flatten(request.records):
        said = wire.user_content(record)
        if said is not None:
            yield from said[1]
        elif isinstance(record, ToolResult):
            yield from record.attachments


def _patches(width: int, height: int) -> int:
    return math.ceil(width / 32) * math.ceil(height / 32)


def _fit_patches(width: int, height: int, *, budget: int) -> tuple[int, int]:
    """Shrink an image into the patch budget, as the guide's step B says."""
    if budget <= 0 or _patches(width, height) <= budget:
        return width, height
    # Typeshed types ``float ** float`` as Any; STYLE bans ``math.sqrt``.
    shrink = cast(float, (32**2 * budget / (width * height)) ** 0.5)
    adjusted = shrink * min(
        math.floor(width * shrink / 32) / (width * shrink / 32),
        math.floor(height * shrink / 32) / (height * shrink / 32),
    )
    return max(1, math.floor(width * adjusted)), max(1, math.floor(height * adjusted))


def _fit_tiles(width: int, height: int) -> tuple[int, int]:
    """Fit a 2048px square, then a 768px short side, as the tile rules say."""
    scale = min(1.0, 2048 / max(width, height))
    width, height = math.floor(width * scale), math.floor(height * scale)
    short = min(width, height)
    if short > 768:
        width, height = (
            math.floor(width * 768 / short),
            math.floor(height * 768 / short),
        )
    return max(1, width), max(1, height)
