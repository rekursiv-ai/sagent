"""Descriptor registry -- sole source of truth for content type.

Every known descriptor lives here as a ``Literal`` type.  Frozensets
are derived via ``get_args`` so the type and the runtime set stay in
sync automatically.

Content-type invariant:
  - ``MULTIPART_DESCRIPTORS``  →  ``tuple[Message, ...]``
  - ``BINARY_DESCRIPTORS``  →  ``bytes``
  - ``TEXT_DESCRIPTORS``  →  ``str``
  - ``JSON_DESCRIPTORS``  →  ``JSON`` (frozen dict)

Tool descriptors (``application/x-tool-<name>``) are enumerated in
``KNOWN_TOOL_DESCRIPTORS``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast, get_args

import dataclasses


if TYPE_CHECKING:
    from sagent.custom_types import Message


# -- Literal types ---------------------------------------------------------
# Use assignment (not ``type X = ...``) so ``get_args`` works at runtime.

TextDescriptor = Literal[
    "text/plain",
    "text/x-error",
    "text/x-thinking",
    "text/x-hint-tool-use-nudge",
    "text/x-queue-id",
    "text/x-user-injected",
    "text/x-user-message",
    "text/x-interrupted",
    "text/x-status",
    "text/x-agent-label",
    "text/x-tool-label",
    "text/x-signal-user-input",
    "text/x-signal-status-changed",
    "text/x-diff",
    "text/x-clear-request",
    "text/x-help-request",
    "text/x-tasks-request",
    "text/x-quit",
    # Inbox spine descriptors -- inbound signals.
    "text/x-break",
    "text/x-abort",
    "text/x-compact-request",
    "text/x-uncompact-request",
    "text/x-model-switch-request",
    "text/x-login-request",
    "text/x-status-update",
    # Inbox spine descriptors -- internal flow.
    "text/x-model-call",
    "text/x-stream-end",
    "text/x-thinking-chunk",
    "text/x-stream-tool-label",
    "text/x-wait-idle",
    "text/x-idle",
    "text/x-compact-done",
    "text/x-session-save-request",
]

ImageDescriptor = Literal[
    "image/png",
    "image/jpeg",
    "image/gif",
    "image/bmp",
    "image/webp",
    "image/svg+xml",
]

NonImageBytesDescriptor = Literal[
    "application/pdf",
    "application/octet-stream",
]

BytesDescriptor = ImageDescriptor | NonImageBytesDescriptor

MultipartDescriptor = Literal[
    "multipart/x-user-message",
    "multipart/x-model-message",
    "multipart/x-tool-call",
    "multipart/x-tool-result",
    "multipart/x-child-event",
    "multipart/mixed",
    # Inbox spine multipart envelopes -- carry inner Messages.
    "multipart/x-tool-batch",
    "multipart/x-tool-batch-result",
    "multipart/x-background-completed",
    "multipart/x-subagent-result",
]

JsonDescriptor = Literal[
    "application/json",
    "application/x-done",
    "application/x-child-done",
    "application/x-thinking-structured",
]

ToolDescriptor = Literal[
    "application/x-tool-advisor",
    "application/x-tool-agentself",
    "application/x-tool-agentsend",
    "application/x-tool-agentspawn",
    "application/x-tool-backgroundtask",
    "application/x-tool-bash",
    "application/x-tool-edit",
    "application/x-tool-glob",
    "application/x-tool-grep",
    "application/x-tool-linear",
    "application/x-tool-list",
    "application/x-tool-paperauthor",
    "application/x-tool-paperdetails",
    "application/x-tool-paperfetch",
    "application/x-tool-papersearch",
    "application/x-tool-playaudio",
    "application/x-tool-read",
    "application/x-tool-skill",
    "application/x-tool-slack",
    "application/x-tool-webfetch",
    "application/x-tool-websearch",
    "application/x-tool-wiki",
    "application/x-tool-write",
]

Descriptor = (
    TextDescriptor
    | BytesDescriptor
    | MultipartDescriptor
    | JsonDescriptor
    | ToolDescriptor
)


# -- Named sentinels --------------------------------------------------------
# Descriptors that flow through dispatch as control signals (not data)
# get a named constant here. Code comparing ``msg.descriptor`` to one
# of these reads as the intent ("is this a quit?") rather than the
# literal string. New sentinels go here, not at the use site.

QUIT_SENTINEL: Literal["text/x-quit"] = "text/x-quit"


# -- Runtime frozensets (derived from Literals) ----------------------------

TEXT_DESCRIPTORS: frozenset[str] = frozenset(get_args(TextDescriptor))
IMAGE_DESCRIPTORS: frozenset[str] = frozenset(get_args(ImageDescriptor))
BINARY_DESCRIPTORS: frozenset[str] = frozenset(get_args(ImageDescriptor)) | frozenset(
    get_args(NonImageBytesDescriptor)
)
MULTIPART_DESCRIPTORS: frozenset[str] = frozenset(get_args(MultipartDescriptor))
JSON_DESCRIPTORS: frozenset[str] = frozenset(get_args(JsonDescriptor))
KNOWN_TOOL_DESCRIPTORS: frozenset[str] = frozenset(get_args(ToolDescriptor))
ALL_DESCRIPTORS: frozenset[str] = (
    TEXT_DESCRIPTORS
    | BINARY_DESCRIPTORS
    | MULTIPART_DESCRIPTORS
    | JSON_DESCRIPTORS
    | KNOWN_TOOL_DESCRIPTORS
)

# Tools that never mutate external state and can run in parallel.
# Bash is conditionally safe - see ``is_request_read_only`` in dispatch,
# which consults ``tool_bash.is_read_only`` on the command string.
READ_ONLY_TOOLS: frozenset[str] = frozenset(
    {
        "application/x-tool-read",
        "application/x-tool-grep",
        "application/x-tool-glob",
        "application/x-tool-list",
        "application/x-tool-websearch",
        "application/x-tool-webfetch",
        "application/x-tool-papersearch",
        "application/x-tool-paperdetails",
        "application/x-tool-paperauthor",
        "application/x-tool-paperfetch",
        "application/x-tool-playaudio",
        "application/x-tool-contextdiagnostics",
        "application/x-tool-skill",
        "application/x-tool-wiki",
        # Strictly a lie - the Agent tool spawns a subagent that runs
        # full RW tools. But we classify it as RO here so sibling
        # Agent calls from one model request batch into ``asyncio.gather``,
        # enabling map-reduce. File-level races inside each subagent
        # are handled by ``get_file_write_lock`` in tools/core.py.
        "application/x-tool-agentspawn",
        "application/x-tool-agentsend",
        "application/x-tool-backgroundtask",
    },
)

# File-op tools whose ``file_path`` input triggers conditional-rule
# injection. Covers all built-in tools that read or modify a specific
# file on disk.
FILE_OP_TOOLS: frozenset[str] = frozenset(
    {"application/x-tool-read", "application/x-tool-edit", "application/x-tool-write"}
)


# -- Classification helpers ------------------------------------------------


def is_multipart(descriptor: str) -> bool:
    """Check whether ``descriptor`` denotes multipart content.

    Args:
      descriptor: Content-type descriptor string.

    Returns:
      is_multi: True if content is ``tuple[Message, ...]``.

    """
    return descriptor in MULTIPART_DESCRIPTORS


def is_image(descriptor: str) -> bool:
    """Check whether ``descriptor`` denotes an image type.

    Args:
      descriptor: Content-type descriptor string.

    Returns:
      is_img: True if content is image ``bytes``.

    """
    return descriptor in IMAGE_DESCRIPTORS


def is_binary(descriptor: str) -> bool:
    """Check whether ``descriptor`` denotes binary content.

    Args:
      descriptor: Content-type descriptor string.

    Returns:
      is_bin: True if content is ``bytes`` (images, PDFs, opaque data).

    """
    return descriptor in BINARY_DESCRIPTORS


def is_thinking(descriptor: str) -> bool:
    """Check whether ``descriptor`` denotes thinking content."""
    return descriptor in ("text/x-thinking", "application/x-thinking-structured")


def is_user_message(descriptor: str) -> bool:
    """Check whether ``descriptor`` is a user message (text or multipart).

    Args:
      descriptor: Content-type descriptor string.

    Returns:
      is_user: True for ``text/x-user-message`` or ``multipart/x-user-message``.

    """
    return descriptor in ("text/x-user-message", "multipart/x-user-message")


def has_error(msg: Message) -> bool:
    """Check whether the message tree contains a ``text/x-error`` part.

    Args:
      msg: Root message to search.

    Returns:
      found: True if any node has descriptor ``text/x-error``.

    """
    if msg.descriptor == "text/x-error":
        return True
    if msg.descriptor in MULTIPART_DESCRIPTORS:
        return any(has_error(p) for p in cast(tuple["Message", ...], msg.content))
    return False


def collect_binary(msg: Message) -> list[Message]:
    """Recursively collect all binary parts from a message tree.

    Args:
      msg: Root message to search.

    Returns:
      parts: List of leaf messages with binary descriptors.

    """
    if msg.descriptor in BINARY_DESCRIPTORS:
        return [msg]
    if msg.descriptor in MULTIPART_DESCRIPTORS:
        out: list[Message] = []
        for p in cast(tuple["Message", ...], msg.content):
            out.extend(collect_binary(p))
        return out
    return []


def strip_binary(msg: Message) -> Message | None:
    """Rebuild message tree without binary parts.

    Args:
      msg: Root message to filter.

    Returns:
      stripped: Rebuilt tree without binary nodes, or None if empty.

    """
    if msg.descriptor in BINARY_DESCRIPTORS:
        return None
    if msg.descriptor in MULTIPART_DESCRIPTORS:
        kept = [
            p
            for child in cast(tuple["Message", ...], msg.content)
            if (p := strip_binary(child)) is not None
        ]
        if not kept:
            return None
        return dataclasses.replace(msg, content=tuple(kept))
    return msg


def flat_text(msg: Message, *, include_errors: bool = False) -> str:
    """Extract text from a known-flat multipart message.

    For ``multipart/*`` messages, returns the first ``text/plain`` (or
    ``text/x-error`` when ``include_errors=True``) child's content.
    For ``text/*`` messages, returns the content directly.

    Args:
      msg: Message to extract text from.
      include_errors: Also accept ``text/x-error`` parts.

    Returns:
      text: First matching text content, or empty string.

    """
    if msg.descriptor.startswith("text/"):
        return cast(str, msg.content)
    if msg.descriptor in MULTIPART_DESCRIPTORS:
        accept = {"text/plain", "text/x-error"} if include_errors else {"text/plain"}
        for p in cast(tuple["Message", ...], msg.content):
            if p.descriptor in accept:
                return cast(str, p.content)
    return ""


def collect_text(msg: Message) -> str:
    """Recursively join all ``text/plain`` content from a message tree.

    Args:
      msg: Root message to search.

    Returns:
      text: Newline-joined text/plain content.

    """
    if msg.descriptor == "text/plain":
        return cast(str, msg.content)
    if msg.descriptor in MULTIPART_DESCRIPTORS:
        parts = [
            t
            for child in cast(tuple["Message", ...], msg.content)
            if (t := collect_text(child))
        ]
        return "\n".join(parts)
    return ""
