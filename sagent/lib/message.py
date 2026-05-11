"""Message accessors - helpers for reading fields from compound Messages."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import dataclasses
import traceback

from sagent.custom_types import (
    JsonMessage,
    Message,
    MultipartMessage,
    TextMessage,
)
from sagent.lib.descriptors import is_multipart, is_user_message
from sagent.lib.json import JSON, MutableJSON, MutableJSONValue, json_freeze


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


_MAX_CAUSE_DEPTH: int = 5


def _exc_to_trace_dict(
    the: traceback.TracebackException,
    *,
    seen: set[int] | None = None,
    depth: int = 0,
) -> MutableJSON:
    """Build a mutable JSON dict for a ``TracebackException`` (recursive).

    Guards against cycles via ``id(the)``-tracked ``seen`` set and against
    pathological chains via ``_MAX_CAUSE_DEPTH``.
    """
    if seen is None:
        seen = set()
    frames: MutableJSONValue = [
        cast(
            "MutableJSONValue",
            {
                "file": fs.filename,
                "line": fs.lineno if fs.lineno is not None else 0,
                "function": fs.name,
                "code": (fs.line or "").strip(),
            },
        )
        for fs in the.stack
    ]
    type_name = the.exc_type.__name__ if the.exc_type else "Exception"
    seen.add(id(the))

    def _recurse(
        child: traceback.TracebackException | None,
    ) -> MutableJSON | None:
        if child is None or id(child) in seen or depth + 1 >= _MAX_CAUSE_DEPTH:
            return None
        return _exc_to_trace_dict(child, seen=seen, depth=depth + 1)

    return {
        "type": type_name,
        "message": "".join(the.format_exception_only()).strip(),
        "frames": frames,
        "cause": _recurse(the.__cause__),
        "context": _recurse(the.__context__),
    }


def _exc_to_trace_json(the: traceback.TracebackException) -> JSON:
    """Serialize a ``TracebackException`` (incl. chained) to frozen JSON."""
    return json_freeze(_exc_to_trace_dict(the))


def build_error_message(user_msg: str, exc: BaseException | None = None) -> Message:
    """Build an error Message; ``multipart/x-error`` when ``exc`` is given.

    Safe to call from inside an exception handler: if traceback extraction
    fails (cycles, recursion limit, pathological exceptions), falls back
    to a flat ``text/x-error`` so the caller never loses the event.

    Args:
      user_msg: Short user-facing error description.
      exc: Optional exception; when present, the returned Message
        includes a structured ``application/x-stack-trace`` part.

    Returns:
      message: ``text/x-error`` (flat) when ``exc`` is None or trace
        extraction fails, otherwise ``multipart/x-error`` carrying both
        the text and a structured traceback.

    """
    if exc is None:
        return TextMessage(user_msg, "text/x-error")
    try:
        the = traceback.TracebackException.from_exception(exc)
        trace = _exc_to_trace_json(the)
    except (RecursionError, RuntimeError, AttributeError, TypeError, ValueError):
        return TextMessage(user_msg, "text/x-error")
    return MultipartMessage(
        (
            TextMessage(user_msg, "text/x-error"),
            JsonMessage(trace, "application/x-stack-trace"),
        ),
        "multipart/x-error",
    )


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
