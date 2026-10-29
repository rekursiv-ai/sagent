"""What the OpenAI wires share: streamed events, user turns, errors, rate limits.

Both the Responses API and Chat Completions stream server-sent events, read the
same records as user turns, and fail with the same error bodies; this module
holds those, without I/O.

Typical usage example:

  async for event in events(response.aiter_lines()):
      ...
  for record in flatten(request.records):
      said = user_content(record)          # (text, attachments), or None
  raise to_exception(status, body=reply, headers=headers)
  provider.usage = rate_limits(response.headers)
"""

from __future__ import annotations

from email.utils import parsedate_to_datetime
from functools import lru_cache
from typing import TYPE_CHECKING

import base64
import hashlib
import math
import re
import time
import uuid

from treekle import ReadError, from_plain, json_loads

from sagent.model import image
from sagent.model.exceptions import (
    AuthError,
    BillingError,
    ContextOverflowError,
    OverloadedError,
    RateLimitedError,
    RefusedError,
    RequestTooLargeError,
)
from sagent.providers.types import UsageWindow
from sagent.session.types import (
    AgentToAgentMessage,
    ContextClear,
    SystemMessage,
    TurnContext,
    UserMessage,
)


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterable, Iterator, Mapping

    from sagent.model.capability import ModelLimits
    from sagent.model.types import Request
    from sagent.session.types import Attachment, SessionRecord


__all__ = [
    "cache_key",
    "events",
    "flatten",
    "image_url",
    "rate_limits",
    "to_exception",
    "user_content",
]


async def events(lines: AsyncIterator[str]) -> AsyncIterator[dict[str, object]]:
    """Yield each server-sent event of a streamed reply, as JSON, until ``[DONE]``.

    Args:
      lines: The reply's body, line by line.

    Yields:
      event: Each ``data:`` line's JSON object.

    """
    # One ``data:`` line is one event on every server these wires reach; a
    # keep-alive or a vendor's stray non-JSON line is not an event.
    async for line in lines:
        if not line.startswith("data:"):
            continue
        payload = line.removeprefix("data:").strip()
        if payload == "[DONE]":
            return
        try:
            yield json_loads(payload, dict[str, object])
        except ValueError:
            continue


def flatten(records: Iterable[SessionRecord]) -> Iterator[SessionRecord]:
    """Yield ``records``, each ``ContextClear`` replaced by its summary and history.

    Args:
      records: A conversation.

    Yields:
      record: Each record, a clear's summary as a ``UserMessage``.

    """
    for record in records:
        if isinstance(record, ContextClear):
            if record.summary is not None:
                yield UserMessage(content=record.summary)
            yield from flatten(record.history)
        else:
            yield record


def user_content(record: SessionRecord) -> tuple[str, tuple[Attachment, ...]] | None:
    """Return the text and attachments ``record`` says on the user's side.

    Args:
      record: Any session record.

    Returns:
      said: The text and its attachments; ``None`` for a record that is not a
        user turn.

    """
    match record:
        case UserMessage(content=text, attachments=attachments):
            return text or "", attachments
        case AgentToAgentMessage(content=text, sender=sender, attachments=attachments):
            body = text or ""
            return (f"[from {sender}]: {body}" if sender is not None else body), (
                attachments
            )
        # A harness note is no one's turn; it rides the user side, marked.
        case SystemMessage(content=str() as text) if text:
            return f"<system-reminder>\n{text}\n</system-reminder>", ()
        case TurnContext() if record.text():
            return f"<system-reminder>\n{record.text()}\n</system-reminder>", ()
        case _:
            return None


def image_url(attachment: Attachment, *, limits: ModelLimits) -> str | None:
    """Return an image attachment as a ``data:`` URL, shrunk to ``limits``.

    Args:
      attachment: Any attachment.
      limits: The model's image limits.

    Returns:
      url: The image's ``data:`` URL; ``None`` for an attachment not an image.

    """
    if not attachment.mime_descriptor.startswith("image/"):
        return None
    return _data_url(
        attachment.data,
        mime=attachment.mime_descriptor,
        max_edge_px=limits.max_image_edge_px,
        max_bytes=limits.max_image_bytes,
    )


def cache_key(request: Request) -> str:
    """Return the key the server routes this conversation's prompt cache by.

    Args:
      request: What the model is asked.

    Returns:
      key: A UUID from the system prompt and the first user turn, the same for
        every request of one conversation, across restarts too.

    """
    first = next(
        (said[0] for r in flatten(request.records) if (said := user_content(r))),
        "",
    )
    digest = hashlib.sha256("\n\n".join((*request.system, first)).encode()).digest()
    return str(uuid.UUID(bytes=digest[:16], version=4))


def to_exception(
    status: int,
    *,
    body: object,
    headers: Mapping[str, str] | None = None,
) -> Exception:
    """Return the exception an error reply, or an in-stream error event, means.

    Args:
      status: The HTTP status; ``0`` for an in-stream event.
      body: The reply's JSON, or its text when it was not JSON.
      headers: The reply's headers, which may advise a wait.

    Returns:
      error: The most specific exception that fits.

    """
    fields = _object(body)
    if nested := _object(fields.get("error")):
        fields = nested
    code = str(fields.get("code") or "")
    kind = str(fields.get("type") or "")
    text = str(fields.get("message") or body)
    lower = text.lower()
    if _too_large(status, lower=lower):
        return RequestTooLargeError(text)
    if code in ("context_length_exceeded", "string_above_max_length") or _overflowed(
        lower,
    ):
        return ContextOverflowError(text)
    if code in ("content_policy_violation", "content_filter") or (
        code == "invalid_prompt" and ("flagged" in lower or "policy" in lower)
    ):
        return RefusedError(text)
    if {code, kind} & {
        "insufficient_quota",
        "billing_hard_limit_reached",
        "usage_not_included",
    }:
        return BillingError(text)
    if status in (401, 403) or {code, kind} & {
        "authentication_error",
        "invalid_api_key",
        "permission_error",
    }:
        return AuthError(text)
    wait_sec = _retry_after(headers or {}, fields=fields)
    if status == 429 or {code, kind} & {
        "rate_limit",
        "rate_limit_error",
        "rate_limit_exceeded",
        "usage_limit_reached",
    }:
        return RateLimitedError(
            text,
            retry_after_sec=wait_sec,
            limit=_limit(headers or {}),
        )
    if (
        status in (408, 409)
        or status >= 500
        or {code, kind}
        & {"server_error", "api_error", "overloaded_error", "server_is_overloaded"}
        or "you can retry" in lower
    ):
        return OverloadedError(text, retry_after_sec=wait_sec)
    return RuntimeError(f"The server replied {status} {code or kind}: {text}")


def rate_limits(headers: Mapping[str, str]) -> tuple[UsageWindow, ...]:
    """Return the rate-limit windows a reply's headers report.

    Args:
      headers: The reply's headers.

    Returns:
      windows: An API key's ``requests`` and ``tokens`` windows, and a ChatGPT
        login's windows by span (``5h``, ``7d``); ``()`` when none is reported.

    """
    lower = {key.lower(): value for key, value in headers.items()}
    windows: list[UsageWindow] = []
    for resource in ("requests", "tokens"):
        limit = _number(lower.get(f"x-ratelimit-limit-{resource}"))
        remaining = _number(lower.get(f"x-ratelimit-remaining-{resource}"))
        delay = _duration(lower.get(f"x-ratelimit-reset-{resource}"))
        if limit is None and remaining is None and delay is None:
            continue
        windows.append(
            UsageWindow(
                label=resource,
                utilization=min(max(1.0 - remaining / limit, 0.0), 1.0)
                if limit is not None and limit > 0 and remaining is not None
                else None,
                resets_at=None if delay is None else time.time() + delay,
                blocked=remaining is not None and remaining <= 0,
            ),
        )
    for slot in ("primary", "secondary"):
        prefix = f"x-codex-{slot}"
        minutes = _number(lower.get(f"{prefix}-window-minutes"))
        if minutes is None or minutes <= 0:
            continue
        used = _number(lower.get(f"{prefix}-used-percent"))
        resets_at = _number(lower.get(f"{prefix}-reset-at"))
        after = _number(lower.get(f"{prefix}-reset-after-seconds"))
        if resets_at is None and after is not None:
            resets_at = time.time() + after
        windows.append(
            UsageWindow(
                label=_span(int(minutes)),
                utilization=None if used is None else min(max(used / 100, 0.0), 1.0),
                resets_at=resets_at,
                blocked=used is not None and used >= 100,
            ),
        )
    return tuple(windows)


# Every request resends the whole conversation, so without the cache each image
# would be shrunk and encoded again on every turn.
@lru_cache(maxsize=64)
def _data_url(data: bytes, *, mime: str, max_edge_px: int, max_bytes: int) -> str:
    data, mime = image.shrink(data, max_edge_px=max_edge_px, max_bytes=max_bytes)
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"


def _object(value: object) -> dict[str, object]:
    try:
        return from_plain(value, dict[str, object])
    except ReadError:
        return {}


# A byte phrase wins over a token phrase in the same body: it is the byte limit
# even when the remedy text mentions the context window. A bare 413 is the byte
# limit unless the body names the token window, which some servers reuse 413 for.
def _too_large(status: int, *, lower: str) -> bool:
    """Whether the error is the request's byte ceiling, not its token window."""
    phrases = (
        "request_too_large",
        "maximum request size",
        "number of bytes",
        "payload too large",
        "request entity too large",
    )
    if any(phrase in lower for phrase in phrases):
        return True
    return status == 413 and not _names_overflow(lower)


def _overflowed(lower: str) -> bool:
    """Whether the error names a token-window overflow."""
    # Ambiguous between bytes and tokens, so sound only once the byte case is out.
    return "exceeds the maximum" in lower or _names_overflow(lower)


# The spellings of every OpenAI-compatible server, pooled: a spelling only one
# server's code knew once reached the runtime raw, and every retry repeated it.
def _names_overflow(lower: str) -> bool:
    phrases = (
        "context_length_exceeded",
        "maximum context length",
        "input too large",
        "input too long",
        "string_above_max_length",
        "prompt is too long",
        "prompt too long",
        "too_long",
    )
    if any(phrase in lower for phrase in phrases):
        return True
    # Both appear benignly in tool-schema validation errors without a marker.
    guarded = (
        ("context window", ("exceed", "overflow", "maximum")),
        ("model context", ("exceed",)),
    )
    return any(
        phrase in lower and any(marker in lower for marker in markers)
        for phrase, markers in guarded
    )


def _retry_after(headers: Mapping[str, str], *, fields: Mapping[str, object]) -> float:
    """Return the advised wait in seconds; ``0`` when the server gave none."""
    lower = {key.lower(): value for key, value in headers.items()}
    millis = _number(lower.get("retry-after-ms"))
    raw = lower.get("retry-after")
    wait = millis / 1000 if millis is not None else _number(raw)
    if wait is None and raw is not None:
        try:
            wait = parsedate_to_datetime(raw).timestamp() - time.time()
        except (TypeError, ValueError):
            wait = None
    # A ChatGPT login's spent window says when it refills in the body.
    if wait is None:
        wait = _number(str(fields.get("resets_in_seconds", "")))
    if wait is None and (at := _number(str(fields.get("resets_at", "")))) is not None:
        wait = at - time.time()
    if wait is None:
        delays = [
            delay
            for resource in ("requests", "tokens")
            if (delay := _duration(lower.get(f"x-ratelimit-reset-{resource}")))
            is not None
        ]
        wait = max(delays, default=0.0)
    # Eight days: a spent weekly window refills up to seven days away.
    return min(max(wait, 0.0), 691_200.0)


def _number(raw: str | None) -> float | None:
    if raw is None:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _duration(raw: str | None) -> float | None:
    """Return an OpenAI reset duration (``6m0s``, ``500ms``, ``1.5``) in seconds."""
    if raw is None or not raw.strip():
        return None
    text = raw.strip().lower()
    # ``ms`` before the single letters, so ``500ms`` is not ``500m`` and ``s``.
    parts = list(re.finditer(r"(\d+(?:\.\d+)?)(ms|[dhms])", text))
    if sum(len(part.group(0)) for part in parts) != len(text):
        seconds = _number(text)
        return None if seconds is None else max(seconds, 0.0)
    unit_sec = {"d": 86_400.0, "h": 3_600.0, "m": 60.0, "s": 1.0, "ms": 0.001}
    return sum(float(part.group(1)) * unit_sec[part.group(2)] for part in parts)


def _limit(headers: Mapping[str, str]) -> str:
    """Return the name of a ChatGPT login's spent window, as ``5-hour``; else ``""``."""
    lower = {key.lower(): value for key, value in headers.items()}
    for slot in ("primary", "secondary"):
        used = _number(lower.get(f"x-codex-{slot}-used-percent"))
        minutes = _number(lower.get(f"x-codex-{slot}-window-minutes"))
        if used is None or used < 100 or minutes is None or minutes <= 0:
            continue
        span = _span(int(minutes))
        unit = {"d": "day", "h": "hour", "m": "minute"}[span[-1]]
        return {"7d": "weekly", "1d": "daily"}.get(span, f"{span[:-1]}-{unit}")
    return ""


def _span(minutes: int) -> str:
    """Return a window's span as a label: ``300`` is ``5h``, ``10080`` is ``7d``."""
    if minutes % 1440 == 0:
        return f"{minutes // 1440}d"
    if minutes % 60 == 0:
        return f"{minutes // 60}h"
    return f"{minutes}m"
