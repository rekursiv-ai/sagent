"""Wire-aware token-count walker shared across providers.

A single walker over ``ModelRequest`` so the set of counted surfaces
stays in one place. Each provider's ``Model.approx_request_tokens``
delegates here; the function in turn uses the model's primitive
``approx_text_tokens`` / ``approx_image_tokens`` heuristics. Adding a
new wire surface (e.g. a new field on ``AssistantMessage``) is a
single-file edit here, not a hunt across every provider.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import json

from sagent.lib.json import json_unfreeze
from sagent.types.runtime import (
    AssistantMessage,
    ToolResult,
    UserMessage,
)
from sagent.types.tape import TapeEvent


if TYPE_CHECKING:
    from sagent.types.model import ModelRequest


class TokenEstimator(Protocol):
    """Minimal token-estimation surface used by the walker.

    A narrower view of :class:`Model` so the walker can be exercised
    against mocks that don't bother implementing transport methods.
    """

    def approx_text_tokens(self, text: str) -> int: ...
    def approx_image_tokens(self, data: bytes) -> int: ...


def approx_request_tokens(request: ModelRequest, model: TokenEstimator) -> int:
    """Approximate input tokens for a fully-built ``ModelRequest``.

    Walks every wire-bearing surface: system prompt, each entry's
    text-bearing fields (``UserMessage.text``, ``AssistantMessage.text``,
    every ``ToolCall.args``/``name``/``id``, every thinking block's
    ``signature``/``thinking``, ``ToolResult.content``), image
    attachments, and the tools schema.

    Args:
      request: Fully-built model request.
      model: Model whose ``approx_text_tokens`` / ``approx_image_tokens``
          primitives are applied to each leaf.

    Returns:
      tokens: Approximate total input token count.

    """
    total = model.approx_text_tokens(request.system or "")
    for entry in request.messages:
        total += _entry_tokens(entry, model)
    for tool in request.tools or ():
        total += model.approx_text_tokens(tool.description or "")
        total += model.approx_text_tokens(
            json.dumps(json_unfreeze(tool.directive_schema))
        )
    return total


def _entry_tokens(entry: TapeEvent, model: TokenEstimator) -> int:
    """Approximate tokens for one history entry across every wire surface."""
    if isinstance(entry, UserMessage):
        total = model.approx_text_tokens(entry.text)
        for att in entry.attachments:
            if att.descriptor.startswith("image/"):
                total += model.approx_image_tokens(att.data)
        return total
    if isinstance(entry, AssistantMessage):
        total = model.approx_text_tokens(entry.text)
        for tc in entry.tool_calls:
            total += model.approx_text_tokens(
                json.dumps(
                    {"id": tc.id, "name": tc.name, "args": dict(tc.args)},
                    default=str,
                )
            )
        for tb in entry.thinking_blocks:
            total += model.approx_text_tokens(str(tb.get("signature") or ""))
            total += model.approx_text_tokens(str(tb.get("thinking") or ""))
        return total
    assert isinstance(entry, ToolResult)
    total = model.approx_text_tokens(entry.content)
    for att in entry.attachments:
        if att.descriptor.startswith("image/"):
            total += model.approx_image_tokens(att.data)
    return total
