"""Message accessors - helpers for reading fields from compound Messages."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import dataclasses

from sagent.custom_types import (
    JsonMessage,
    Message,
    MultipartMessage,
    TextMessage,
)
from sagent.lib.descriptors import is_multipart, is_user_message
from sagent.lib.json import JSON, json_freeze


def tool_call_message(
    queue_id: str,
    name: str,
    args: JSON,
) -> MultipartMessage:
    """Build a ``multipart/x-tool-call`` envelope.

    Centralizes the dynamic ``application/x-tool-{name}`` descriptor so
    callers don't need per-site type suppressions.

    Args:
      queue_id: Unique identifier tying this call to its result.
      name: Tool name (lowercased automatically for the descriptor).
      args: Frozen JSON directive for the tool.

    Returns:
      message: Two-part multipart message (queue-id + tool directive).

    """
    return MultipartMessage(
        (
            TextMessage(queue_id, "text/x-queue-id"),
            JsonMessage(args, f"application/x-tool-{name.lower()}"),
        ),
        "multipart/x-tool-call",
    )


def get_queue_id(msg: Message) -> str:
    """Return the ``text/x-queue-id`` child content; ``""`` if absent.

    Args:
      msg: Compound message to inspect.

    Returns:
      queue_id: Queue ID string, or empty string if not found.

    """
    if not is_multipart(msg.descriptor):
        return ""
    for part in cast(tuple[Message, ...], msg.content):
        if part.descriptor == "text/x-queue-id":
            return cast(str, part.content)
    return ""


def get_tool_name(msg: Message) -> str:
    """Return tool name from a ``multipart/x-tool-call`` message; ``""`` if absent.

    Args:
      msg: Tool-call message to inspect.

    Returns:
      name: Tool name extracted from the descriptor suffix.

    """
    if not is_multipart(msg.descriptor):
        return ""
    for part in cast(tuple[Message, ...], msg.content):
        if part.descriptor.startswith("application/x-tool-"):
            return part.descriptor.removeprefix("application/x-tool-")
    return ""


def get_directive(msg: Message) -> JSON:
    """Return the JSON directive from a ``multipart/x-tool-call`` message.

    Args:
      msg: Tool-call message to inspect.

    Returns:
      directive: Frozen JSON directive, or empty frozen dict if absent.

    """
    if not is_multipart(msg.descriptor):
        return json_freeze({})
    for part in cast(tuple[Message, ...], msg.content):
        if part.descriptor.startswith("application/x-tool-"):
            return cast(JSON, part.content)
    return json_freeze({})


def response_text(msg: Message) -> str:
    """Extract joined ``text/plain`` parts from a model response.

    Args:
      msg: Model response message.

    Returns:
      text: Newline-joined text content.

    """
    if not is_multipart(msg.descriptor):
        return ""
    parts = cast(tuple[Message, ...], msg.content)
    return "\n".join(str(p.content) for p in parts if p.descriptor == "text/plain")


def thinking_text(part: Message) -> str:
    """Extract display text from a thinking part (either descriptor).

    Args:
      part: Thinking message part.

    Returns:
      text: Thinking text content.

    """
    if part.descriptor == "text/x-thinking":
        # Descriptor is source of truth for content structure; no isinstance check.
        return cast(str, part.content)
    # Descriptor is source of truth for content structure; we
    # intentionally don't isinstance-check here.
    return str(cast(Mapping[str, object], part.content).get("thinking", ""))


def response_tool_calls(msg: Message) -> list[Message]:
    """Extract ``multipart/x-tool-call`` parts from a model response.

    Args:
      msg: Model response message.

    Returns:
      tool_calls: List of tool-call sub-messages.

    """
    if not is_multipart(msg.descriptor):
        return []
    parts = cast(tuple[Message, ...], msg.content)
    return [p for p in parts if p.descriptor == "multipart/x-tool-call"]


def append_to_first_user_message(
    messages: list[Message],
    text: str,
) -> None:
    """Append ``text`` to the first user message, or prepend a new one.

    Args:
      messages: Message list, mutated in place.
      text: Text to append.

    """
    for j, m in enumerate(messages):
        if is_user_message(m.descriptor):
            new_content = (
                str(m.content) + "\n\n" + text
                if m.descriptor == "text/x-user-message"
                else text
            )
            messages[j] = dataclasses.replace(m, content=new_content)
            return
    messages.insert(
        0,
        TextMessage(text, "text/x-user-message"),
    )
