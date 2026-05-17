"""Conversation history records.

The wire-format data classes that flow between agent and model and form
the contents of ``agent.history``. Leaves: stdlib only.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import dataclasses
import itertools
import time


_id_counter: itertools.count[int] = itertools.count()


__all__ = [
    "AssistantMessage",
    "BytesMessage",
    "HistoryEntry",
    "SessionMessage",
    "ToolCall",
    "ToolResult",
    "UserMessage",
    "reset_id_counter",
]


def reset_id_counter(start: int) -> None:
    """Reset the ``SessionMessage`` id counter.

    Used by session resume: after loading history with persisted ids
    1..N, callers reset the counter to N+1 so newly created messages
    don't collide.

    Args:
      start: First id the counter will yield next.

    """
    global _id_counter  # noqa: PLW0603 -- module-level counter requires global statement
    _id_counter = itertools.count(start)


@dataclass(frozen=True, slots=True)  # check-dataclass: ignore[kw_only]
class BytesMessage:
    """Binary payload (image, PDF)."""

    data: bytes
    """Raw bytes of the payload."""

    descriptor: str
    """MIME-style content type (e.g. ``image/png``)."""


@dataclass(frozen=True, slots=True, kw_only=True)
class SessionMessage:
    """Common fields for all history entries."""

    id: int = dataclasses.field(default_factory=lambda: next(_id_counter))
    """Monotonically increasing per-session message id."""

    parent_id: int = -1
    """Id of the message this one responds to, or ``-1``."""

    timestamp: float = dataclasses.field(default_factory=time.monotonic)
    """Monotonic clock seconds when the message was created."""


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolCall:
    """One tool invocation requested by the model."""

    id: str
    """Provider-assigned call id (e.g. ``toolu_01...``)."""

    name: str
    """Tool name the model wants to invoke."""

    args: Mapping[str, object]
    """Parsed directive arguments."""


@dataclass(frozen=True, slots=True, kw_only=True)
class UserMessage(SessionMessage):
    """User or system text the model should see."""

    text: str
    """Plain-text content."""

    attachments: tuple[BytesMessage, ...] = ()
    """Image/PDF payloads sent alongside the text."""


@dataclass(frozen=True, slots=True, kw_only=True)
class AssistantMessage(SessionMessage):
    """Model response: text and/or tool calls."""

    text: str = ""
    """User-visible response text."""

    thinking_blocks: tuple[Mapping[str, object], ...] = ()
    """Provider thinking blocks (opaque dicts)."""

    tool_calls: tuple[ToolCall, ...] = ()
    """Tool invocations requested by the model."""


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolResult(SessionMessage):
    """Result of one tool invocation."""

    call_id: str
    """Id of the originating ``ToolCall``."""

    content: str
    """Result text shown to the model."""

    is_error: bool = False
    """True when the tool raised or signalled failure."""

    attachments: tuple[BytesMessage, ...] = ()
    """Image/PDF payloads produced by the tool."""

    diff: str = ""
    """Unified-diff fragment for renderers (e.g. Edit/Write)."""

    diff_file_path: str = ""
    """Absolute path the ``diff`` applies to."""

    hint: str = ""
    """Optional follow-up nudge surfaced to the model."""

    summary: str = ""
    """Optional short post-execution receipt line."""


type HistoryEntry = UserMessage | AssistantMessage | ToolResult
