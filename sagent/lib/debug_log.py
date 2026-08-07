"""Structured JSON-line debug log for diagnosing wire-layer errors.

Writes to ``$SAGENT_DEBUG_LOG`` (default ``~/.sagent/debug.log``). Two
entry points:

- ``trace(event, **data)`` - verbose, gated on ``SAGENT_DEBUG=1``.
- ``trace_error(event, **data)`` - always on. Use for 400s and other
  terminal conditions so they're captured even without opting in.

Intended for bugs like "conversation must end with a user message":
we log the role sequence of every Anthropic request and the full
request shape on ``BadRequestError`` so post-mortem trace is possible
without a session replay.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

import json
import os
import time

from sagent.lib.userdirs import data_dir


_DEFAULT_PATH = data_dir("rekursiv-ai") / "sagent" / "debug.log"
_MAX_PREVIEW = 200  # config-globals: ignore -- display preview cap
_TRUTHY = frozenset({"1", "true", "yes", "on"})


def log_path() -> Path:
    """Return the debug log file path, honoring ``$SAGENT_DEBUG_LOG``.

    Returns:
      path: Resolved log file path.

    """
    override = os.environ.get("SAGENT_DEBUG_LOG")
    return Path(override) if override else _DEFAULT_PATH


def trace(event: str, **data: object) -> None:
    """Write a verbose trace line, gated on ``SAGENT_DEBUG=1``.

    Args:
      event: Event name.
      **data: Arbitrary key-value payload.

    """
    if os.environ.get("SAGENT_DEBUG", "").lower() in _TRUTHY:
        _write(event, data)


def trace_error(event: str, **data: object) -> None:
    """Write an always-on error trace line.

    Args:
      event: Event name.
      **data: Arbitrary key-value payload.

    """
    _write(event, data)


def summarize_messages(messages: Sequence[object]) -> list[dict[str, object]]:
    """Render a wire-format message list into a compact log shape.

    Args:
      messages: Raw wire-format messages.

    Returns:
      summaries: List of compact summary dicts per message.

    """
    out: list[dict[str, object]] = []
    for raw in messages:
        if not isinstance(raw, Mapping):
            out.append({"role": "?", "raw": repr(raw)[:_MAX_PREVIEW]})
            continue
        m = cast(Mapping[str, object], raw)
        role = m.get("role")
        content = m.get("content")
        if isinstance(content, str):
            out.append({"role": role, "text": content[:_MAX_PREVIEW]})
        elif isinstance(content, list):
            blocks = cast(list[object], content)
            out.append({"role": role, "blocks": [_summarize_block(b) for b in blocks]})
        else:
            out.append({"role": role, "content": None})
    return out


def role_sequence(messages: Sequence[object]) -> list[str]:
    """Extract the role of each message as a cheap fingerprint.

    Args:
      messages: Raw wire-format messages.

    Returns:
      roles: List of role strings (``"?"`` for unknown).

    """
    out: list[str] = []
    for raw in messages:
        if isinstance(raw, Mapping):
            role = cast(Mapping[str, object], raw).get("role", "?")
            out.append(role if isinstance(role, str) else "?")
        else:
            out.append("?")
    return out


_RESERVED_KEYS = frozenset({"ts", "event"})


def _write(event: str, data: dict[str, object]) -> None:
    """Append a JSON record to the debug log file.

    User-supplied keys that collide with reserved record keys (``ts``,
    ``event``) are dropped so sloppy callers cannot silently overwrite
    the timestamp or event name and corrupt downstream log analysis.
    """
    try:
        path = log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        safe = {k: v for k, v in data.items() if k not in _RESERVED_KEYS}
        record = {"ts": time.time(), "event": event, **safe}
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except Exception:  # noqa: S110 -- debug logging must never crash callers; there's no safer channel to report the failure to
        pass


def _summarize_block(raw: object) -> dict[str, object]:
    """Summarize a single content block for logging."""
    if not isinstance(raw, Mapping):
        return {"type": "?"}
    b = cast(Mapping[str, object], raw)
    t = b.get("type", "?")
    if t == "text":
        text = b.get("text") or ""
        preview = (
            text[:_MAX_PREVIEW] if isinstance(text, str) else str(text)[:_MAX_PREVIEW]
        )
        return {"type": "text", "preview": preview}
    if t == "tool_use":
        return {"type": "tool_use", "name": b.get("name"), "id": b.get("id")}
    if t == "tool_result":
        c = b.get("content")
        c_str = c if isinstance(c, str) else json.dumps(c, default=str)
        return {
            "type": "tool_result",
            "preview": c_str[:_MAX_PREVIEW],
            "is_error": b.get("is_error"),
        }
    return {"type": t}
