"""Wire-aware token-count walker shared across providers.

A single walker over ``ModelRequest`` so the set of counted surfaces
stays in one place. Each provider's ``Model.approx_request_tokens``
delegates here; the function in turn uses the model's primitive
``approx_text_tokens`` / ``approx_image_tokens`` heuristics. Adding a
new wire surface (e.g. a new field on ``AssistantMessage``) is a
single-file edit here, not a hunt across every provider.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Protocol

import json
import logging

from sagent.lib.json import json_unfreeze
from sagent.types.runtime import (
    AgentSendMessage,
    AssistantMessage,
    ToolResult,
    UserMessage,
)
from sagent.types.tape import TapeEvent


if TYPE_CHECKING:
    from sagent.types.model import ModelRequest


logger = logging.getLogger(__name__)


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
    ``signature``/``thinking``, ``ToolResult.content``), image and PDF
    attachments (both billed as image tokens by Anthropic/Google), and
    the tools schema.

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
    if isinstance(entry, (AgentSendMessage, UserMessage)):
        total = model.approx_text_tokens(entry.text)
        for att in entry.attachments:
            total += _attachment_tokens(att.descriptor, att.data, model)
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
            total += _thinking_block_tokens(tb, model)
        return total
    assert isinstance(entry, ToolResult)
    total = model.approx_text_tokens(entry.content)
    for att in entry.attachments:
        total += _attachment_tokens(att.descriptor, att.data, model)
    return total


def _thinking_block_tokens(block: Mapping[str, object], model: TokenEstimator) -> int:
    """Sum every text-bearing field across the known thinking-block shapes.

    Anthropic emits ``{"type":"thinking","signature":...,"thinking":...}`` and
    ``{"type":"redacted_thinking"}``; OpenAI / OpenAI-subscription / chat-
    completions reasoning is stored as ``{"type":"reasoning","text":...}``. All
    re-ship on the wire in some form (or at minimum count against the model's
    output-token quota when later sent back as input on a resume), so every
    text-bearing field must contribute to the request token estimate.
    """
    total = 0
    for field in ("signature", "thinking", "text"):
        value = block.get(field)
        if isinstance(value, str) and value:
            total += model.approx_text_tokens(value)
    return total


def _attachment_tokens(descriptor: str, data: bytes, model: TokenEstimator) -> int:
    """Approximate token cost of one ``BytesMessage`` attachment.

    Images and PDFs are both shipped on the wire by Anthropic and Google
    providers; both contribute to the request token budget. New
    descriptors are logged so a silent drop -- the previous bug, where
    PDFs were filtered out and compaction fired late -- can't recur.
    """
    if descriptor.startswith("image/") or descriptor == "application/pdf":
        return model.approx_image_tokens(data)
    logger.warning("token_count: unknown attachment descriptor %s", descriptor)
    return 0
