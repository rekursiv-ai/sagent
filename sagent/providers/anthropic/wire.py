"""Translate between our records and Anthropic's Messages API JSON, without I/O.

Typical usage example:

  body, betas = encode(request, capability=c, settings=s, system=system_blocks(...))
  # POST body to /v1/messages, with the header anthropic-beta: ",".join(betas).

  reader = StreamReader(capability=c, settings=s)   # The reply comes in fragments.
  async for event in events(response.aiter_lines()):
      chunk = reader.feed(event)                     # New text, to show at once.
  reader.response()                                  # The finished records and cost.

  raise to_exception(status, body=reply, headers=headers)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from functools import lru_cache
from typing import TYPE_CHECKING

import base64
import math
import re
import time

from treekle import ReadError, from_plain, immutable, json_loads, mutable

from sagent.model import image
from sagent.model.cost import TokenCount
from sagent.model.exceptions import (
    AuthError,
    BillingError,
    ContextOverflowError,
    OverloadedError,
    RateLimitedError,
    RefusedError,
    RequestTooLargeError,
)
from sagent.model.types import Response, TextChunk
from sagent.providers.types import UsageWindow
from sagent.session.types import (
    AgentToAgentMessage,
    AssistantMessage,
    ContextClear,
    SystemMessage,
    Thinking,
    ToolCall,
    ToolResult,
    TurnContext,
    UserMessage,
)


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Mapping, Sequence

    from sagent.model.capability import ModelCapability, ModelLimits, ModelSettings
    from sagent.model.types import Request
    from sagent.session.types import Attachment, SessionRecord


__all__ = [
    "StreamReader",
    "attachment_tokens",
    "content",
    "encode",
    "events",
    "rate_limits",
    "system_blocks",
    "to_exception",
]


def encode(
    request: Request,
    *,
    capability: ModelCapability,
    settings: ModelSettings,
    system: list[dict[str, object]],
    cache_expired: bool = False,
) -> tuple[dict[str, object], list[str]]:
    """Return the body of a ``/v1/messages`` request, and the betas it needs.

    Args:
      request: What the model is asked.
      capability: The model it is asked of.
      settings: The settings chosen for that model.
      system: The system blocks, as ``system_blocks`` or a provider makes them.
      cache_expired: Whether the prompt cache has lapsed since the last reply.

    Returns:
      body: The request body.
      betas: The beta features the body uses.

    """
    limits = capability.limits
    reply = request.response_tokens(limits)
    body: dict[str, object] = {
        "model": capability.wire_model_id or capability.model_id,
        "max_tokens": reply,
        "messages": _messages(request.records, settings=settings, limits=limits),
        "stream": True,
    }
    betas = list(limits.request_betas)
    if system:
        body["system"] = system
    if request.tools:
        body["tools"] = [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": mutable(tool.schema),
            }
            for tool in request.tools
        ]
    if settings.thinking_budget != "none":
        body["thinking"] = _thinking(settings, reply=reply)
    if settings.thinking_output == "redacted":
        betas.append("redact-thinking-2026-02-12")
    edits = _context_edits(settings=settings, cache_expired=cache_expired)
    if edits:
        body["context_management"] = {"edits": edits}
        betas.append("context-management-2025-06-27")
    if settings.thinking_effort != "none":
        body["output_config"] = {"effort": settings.thinking_effort}
    # The API takes only ``auto`` and ``standard_only``; fast mode is ``speed``.
    if settings.service_tier == "default":
        body["service_tier"] = "standard_only"
    if settings.service_tier == "priority":
        body["speed"] = "fast"
        betas.append("fast-mode-2026-02-01")
    return body, betas


def system_blocks(
    texts: Sequence[str],
    *,
    settings: ModelSettings,
) -> list[dict[str, object]]:
    """Return the system prompt as blocks, the last three marked for caching.

    Args:
      texts: The prompt's parts, the most stable first; empty ones are left out.
      settings: The settings whose cache lifetime the marks ask for.

    Returns:
      blocks: The system blocks.

    """
    # A request takes at most four cache marks, and the last message takes one.
    blocks: list[dict[str, object]] = [
        {"type": "text", "text": text} for text in texts if text
    ]
    return [
        {**block, "cache_control": _cache_mark(settings)}
        if i >= len(blocks) - 3
        else block
        for i, block in enumerate(blocks)
    ]


def attachment_tokens(attachment: Attachment, *, limits: ModelLimits) -> int:
    """Return about how many input tokens ``attachment`` bills.

    An image bills its pixels over 750, once shrunk to the edge the server
    allows; a PDF page bills up to 3,000 tokens of text and its page as such an
    image; what is not sent bills nothing.

    Args:
      attachment: What is attached.
      limits: The window's, whose image edge the server shrinks to.

    Returns:
      tokens: The vendor's published estimate.

    References:
      https://platform.claude.com/docs/en/build-with-claude/vision
        Image tokens are width times height over 750.
      https://platform.claude.com/docs/en/build-with-claude/pdf-support
        Each page uses 1,500-3,000 text tokens, and is billed as an image.

    """
    edge = limits.max_image_edge_px
    match _attachment_kind(attachment.mime_descriptor):
        case "image":
            width, height = image.size(attachment.data, max_edge_px=edge)
        case "document":
            return sum(
                3_000 + math.prod(image.fit(*page, max_edge_px=edge)) // 750
                for page in image.pages(attachment.data)
            )
        case _:
            return 0
    return width * height // 750


def content(
    text: str,
    attachments: Sequence[Attachment],
    *,
    limits: ModelLimits,
) -> list[dict[str, object]]:
    """Return ``text`` and its image and PDF attachments as message content.

    Args:
      text: The prose; empty adds no text block.
      attachments: Its attachments; any not an image or a PDF is left out.
      limits: The model's image limits, which each image is shrunk to.

    Returns:
      blocks: The content blocks, text first.

    """
    blocks: list[dict[str, object]] = [{"type": "text", "text": text}] if text else []
    for attachment in attachments:
        kind = _attachment_kind(attachment.mime_descriptor)
        if not kind:
            continue
        data, mime = _upload(
            attachment.data,
            mime=attachment.mime_descriptor,
            max_edge_px=limits.max_image_edge_px,
            max_bytes=limits.max_image_bytes,
        )
        blocks.append(
            {
                "type": kind,
                "source": {"type": "base64", "media_type": mime, "data": data},
            },
        )
    return blocks


async def events(lines: AsyncIterator[str]) -> AsyncIterator[dict[str, object]]:
    """Yield each server-sent event of a streamed reply, as JSON.

    Args:
      lines: The reply's body, line by line.

    Yields:
      event: Each event's JSON.

    """
    # ``data:`` lines gather until a blank line. Each payload names its own
    # ``type``, so the ``event:`` lines are redundant.
    data: list[str] = []
    async for line in lines:
        if line.startswith("data:"):
            data.append(line.removeprefix("data:").lstrip())
        elif not line and data:
            yield json_loads("\n".join(data), dict[str, object])
            data = []
    if data:
        yield json_loads("\n".join(data), dict[str, object])


class StreamReader:
    """Collects a streamed reply's fragments into records.

    Args:
      capability: The model replying, whose prices cost the reply.
      settings: The settings it was asked with.

    """

    def __init__(self, *, capability: ModelCapability, settings: ModelSettings) -> None:
        self._capability = capability
        self._settings = settings
        self._blocks: dict[int, _Block] = {}
        self._usage: dict[str, object] = {}
        self._stop_reason = ""

    @property
    def stop_reason(self) -> str:
        """Return why the model stopped; empty until it says."""
        return self._stop_reason

    def feed(self, event: Mapping[str, object]) -> TextChunk | None:
        """Take one event, and return the text it adds, if any.

        Args:
          event: One server-sent event.

        Returns:
          chunk: The text or reasoning the event adds; ``None`` for the rest.

        Raises:
          Exception: The ``to_exception`` of an in-stream ``error`` event.

        """
        kind = event.get("type")
        if kind == "message_start":
            self._usage |= _object(_object(event.get("message")).get("usage"))
        elif kind == "content_block_start":
            start = _object(event.get("content_block"))
            self._blocks[from_plain(event.get("index"), int, default=0)] = _Block(
                kind=str(start.get("type") or ""),
                call_id=str(start.get("id") or ""),
                name=str(start.get("name") or ""),
                sealed=str(start.get("data") or ""),
            )
        elif kind == "content_block_delta":
            index = from_plain(event.get("index"), int, default=0)
            block = self._blocks.get(index)
            if block is not None:
                # Earlier blocks are whole by now; one that makes no record
                # takes no index.
                return block.add(
                    _object(event.get("delta")),
                    record=sum(
                        1
                        for i, b in self._blocks.items()
                        if i < index and b.record() is not None
                    ),
                )
        elif kind == "message_delta":
            self._stop_reason = str(
                _object(event.get("delta")).get("stop_reason") or "",
            )
            self._usage |= _object(event.get("usage"))
        elif kind == "error":
            raise to_exception(0, body=event)
        return None

    def response(self) -> Response:
        """Return the finished reply.

        Returns:
          response: Its records in block order, its tokens, and their cost.

        """
        records = tuple(
            record
            for _, block in sorted(self._blocks.items())
            if (record := block.record()) is not None
        )
        tokens = self._tokens()
        # Billed as served: a fast request the server ran at standard speed
        # reports ``speed: standard`` and is charged standard rates.
        served = "priority" if self._usage.get("speed") == "fast" else "auto"
        return Response(
            records=records,
            tokens=tokens,
            spend=self._capability.prices[
                served,
                tokens.prompt,
                datetime.now(UTC).date(),
            ]
            * tokens,
            stop_reason=self._stop_reason,
        )

    def _tokens(self) -> TokenCount:
        usage = self._usage
        written = from_plain(usage.get("cache_creation_input_tokens"), int, default=0)
        split = _object(usage.get("cache_creation"))
        # A write the server did not split was made at the lifetime asked for.
        if split:
            written_1h = from_plain(
                split.get("ephemeral_1h_input_tokens"),
                int,
                default=0,
            )
        else:
            written_1h = written if self._settings.cache_ttl_sec >= 3600.0 else 0
        return TokenCount(
            request=from_plain(usage.get("input_tokens"), int, default=0),
            response=from_plain(usage.get("output_tokens"), int, default=0),
            cache_write=written - written_1h,
            cache_write_1h=written_1h,
            cache_read=from_plain(usage.get("cache_read_input_tokens"), int, default=0),
        )


def to_exception(
    status: int,
    *,
    body: object,
    headers: Mapping[str, str] | None = None,
) -> Exception:
    """Return the exception an error reply, or an in-stream ``error`` event, means.

    Args:
      status: The HTTP status; ``0`` for an in-stream event.
      body: The reply's JSON, or its text when it was not JSON.
      headers: The reply's headers, which may advise a wait.

    Returns:
      error: The most specific exception that fits.

    """
    fields = _object(body)
    if isinstance(fields.get("error"), dict):
        fields = _object(fields.get("error"))
    kind = str(fields.get("type") or "")
    text = str(fields.get("message") or body)
    lower = text.lower()
    if status == 413 or kind == "request_too_large":
        return RequestTooLargeError(text)
    if "prompt is too long" in lower:
        return ContextOverflowError(text)
    if kind == "invalid_request_error" and _blocked(lower):
        return RefusedError(text)
    if _out_of_credit(lower):
        return BillingError(text)
    if status in (401, 403) or kind in ("authentication_error", "permission_error"):
        return AuthError(text)
    wait_sec = _retry_after(headers or {}, rate_limited=status == 429)
    if status == 429 or kind == "rate_limit_error":
        claim = {key.lower(): value for key, value in (headers or {}).items()}.get(
            "anthropic-ratelimit-unified-representative-claim",
            "",
        )
        return RateLimitedError(text, retry_after_sec=wait_sec, limit=_limit(claim))
    if (
        status in (408, 409)
        or status >= 500
        or kind in ("overloaded_error", "api_error")
    ):
        return OverloadedError(text, retry_after_sec=wait_sec)
    return RuntimeError(f"Anthropic replied {status} {kind}: {text}")


def rate_limits(headers: Mapping[str, str]) -> tuple[UsageWindow, ...]:
    """Return the subscription's rate-limit windows that a reply's headers report.

    Args:
      headers: The reply's headers.

    Returns:
      windows: The 5-hour and 7-day windows reported; ``()`` for an API key.

    """
    lower = {key.lower(): value for key, value in headers.items()}
    overall = lower.get("anthropic-ratelimit-unified-status")
    windows: list[UsageWindow] = []
    for label in ("5h", "7d"):
        prefix = f"anthropic-ratelimit-unified-{label}"
        utilization = _number(lower.get(f"{prefix}-utilization"))
        resets_at = _number(lower.get(f"{prefix}-reset"))
        status = lower.get(f"{prefix}-status")
        if utilization is None and resets_at is None and status is None:
            continue
        windows.append(
            UsageWindow(
                label=label,
                utilization=None
                if utilization is None
                else min(max(utilization, 0.0), 1.0),
                resets_at=resets_at,
                blocked=_blocking(overall) or _blocking(status),
            ),
        )
    return tuple(windows)


def _messages(
    records: Sequence[SessionRecord],
    *,
    settings: ModelSettings,
    limits: ModelLimits,
) -> list[dict[str, object]]:
    """Return ``records`` as alternating user and assistant messages."""
    ids: dict[str, str] = {}
    turns: list[tuple[str, list[dict[str, object]]]] = []
    for record in records:
        for role, block in _blocks(record, ids=ids, limits=limits):
            if turns and turns[-1][0] == role:
                turns[-1][1].append(block)
            else:
                turns.append((role, [block]))
    for role, blocks in turns:
        # The API rejects an assistant turn that ends on thinking.
        if role == "assistant" and blocks[-1]["type"] in (
            "thinking",
            "redacted_thinking",
        ):
            blocks.append({"type": "text", "text": "."})
    if turns:
        blocks = turns[-1][1]
        blocks[-1] = {**blocks[-1], "cache_control": _cache_mark(settings)}
    return [{"role": role, "content": blocks} for role, blocks in turns]


def _blocks(
    record: SessionRecord,
    *,
    ids: dict[str, str],
    limits: ModelLimits,
) -> list[tuple[str, dict[str, object]]]:
    """Return the role and blocks ``record`` becomes; none for bookkeeping records."""
    match record:
        case UserMessage(content=text, attachments=attachments):
            return [
                ("user", block)
                for block in content(text or "", attachments, limits=limits)
            ]
        case AgentToAgentMessage(content=text, sender=sender, attachments=attachments):
            body = text if text is not None else ""
            labeled = f"[from {sender}]: {body}" if sender is not None else body
            return [
                ("user", block)
                for block in content(labeled, attachments, limits=limits)
            ]
        # A harness note is no one's turn; it rides the user side, marked.
        case SystemMessage(content=str() as text) if text:
            note = f"<system-reminder>\n{text}\n</system-reminder>"
            return [("user", {"type": "text", "text": note})]
        case TurnContext() if record.text():
            note = f"<system-reminder>\n{record.text()}\n</system-reminder>"
            return [("user", {"type": "text", "text": note})]
        case AssistantMessage(content=str() as text) if text:
            return [("assistant", {"type": "text", "text": text})]
        # Anthropic accepts thinking back only with its own signature, and
        # rejects the request over any other. Gemini's reasoning has no seal;
        # OpenAI seals its own beside its reasoning id. Both are left out.
        case Thinking(extra=extra) if "reasoning_id" in extra:
            return []
        case Thinking(content=str() as text, encrypted=str() as signature):
            return [
                (
                    "assistant",
                    {"type": "thinking", "thinking": text, "signature": signature},
                ),
            ]
        case Thinking(content=None, encrypted=str() as sealed):
            return [("assistant", {"type": "redacted_thinking", "data": sealed})]
        case ToolCall(call_id=call_id, name=name, arguments=arguments):
            call: dict[str, object] = {
                "type": "tool_use",
                "id": _wire_id(call_id, ids=ids),
                "name": name,
                "input": mutable(arguments),
            }
            return [("assistant", call)]
        case ToolResult():
            result: dict[str, object] = {
                "type": "tool_result",
                "tool_use_id": _wire_id(record.call_id, ids=ids),
                "is_error": record.is_error is True,
            }
            blocks = content(record.text(), record.attachments, limits=limits)
            if blocks:
                result["content"] = blocks
            return [("user", result)]
        case ContextClear(summary=summary, history=history):
            head: list[tuple[str, dict[str, object]]] = []
            if summary is not None:
                head.append(("user", {"type": "text", "text": summary}))
            return head + [
                pair
                for item in history
                for pair in _blocks(item, ids=ids, limits=limits)
            ]
        case _:
            return []


# Another vendor's ids (OpenAI's ``fc_*``) fail Anthropic's id grammar, so each
# id is renamed, consistently within one request.
def _wire_id(call_id: str, *, ids: dict[str, str]) -> str:
    return ids.setdefault(call_id, f"toolu_{len(ids)}")


def _cache_mark(settings: ModelSettings) -> dict[str, object]:
    if settings.cache_ttl_sec >= 3600.0:
        return {"type": "ephemeral", "ttl": "1h"}
    return {"type": "ephemeral"}


def _thinking(settings: ModelSettings, *, reply: int) -> dict[str, object]:
    if settings.thinking_budget == "auto":
        thinking: dict[str, object] = {"type": "adaptive"}
    else:
        # Thinking counts against ``max_tokens``; half leaves the reply room.
        budget = max(reply // 2, 1024)
        thinking = {"type": "enabled", "budget_tokens": budget}
    if settings.thinking_output == "text":
        thinking["display"] = "summarized"
    elif settings.thinking_output == "redacted":
        thinking["display"] = "omitted"
    return thinking


def _context_edits(
    *,
    settings: ModelSettings,
    cache_expired: bool,
) -> list[dict[str, object]]:
    """Return what the server should drop from the context before reading it."""
    edits: list[dict[str, object]] = []
    if settings.thinking_budget != "none":
        # Old thinking kept is cached prefix kept; once the cache has lapsed,
        # only the last turn's is worth paying to resend.
        keep = {"type": "thinking_turns", "value": 1} if cache_expired else "all"
        edits.append({"type": "clear_thinking_20251015", "keep": keep})
    pruning = settings.pruning
    if settings.manage_context_server_side and pruning is not None:
        edits.append(
            {
                "type": "clear_tool_uses_20250919",
                "trigger": {"type": "input_tokens", "value": pruning.trigger_tokens},
                "clear_at_least": {
                    "type": "input_tokens",
                    "value": max(pruning.trigger_tokens - pruning.target_tokens, 1_000),
                },
                "keep": {"type": "tool_uses", "value": pruning.keep_recent},
                "exclude_tools": sorted(pruning.keep_tools),
            },
        )
    return edits


def _attachment_kind(mime: str) -> str:
    """Return the block type Anthropic reads ``mime`` as; empty for none."""
    if mime in ("image/jpeg", "image/png", "image/gif", "image/webp"):
        return "image"
    if mime == "application/pdf":
        return "document"
    return ""


# Every request resends the whole conversation, so without the cache each image
# would be shrunk and encoded again on every turn.
@lru_cache(maxsize=64)
def _upload(
    data: bytes,
    *,
    mime: str,
    max_edge_px: int,
    max_bytes: int,
) -> tuple[str, str]:
    """Return ``data`` within the limits, base64-encoded, and its media type."""
    if mime.startswith("image/"):
        data, mime = image.shrink(data, max_edge_px=max_edge_px, max_bytes=max_bytes)
    return base64.b64encode(data).decode(), mime


@dataclass(slots=True, kw_only=True)
class _Block:
    """One content block of a reply, as its fragments arrive."""

    kind: str
    parts: list[str] = field(default_factory=list[str])
    """Text, thinking, or a tool call's argument JSON, in arrival order."""

    signature: str = ""
    sealed: str = ""
    """A redacted thinking block's opaque content."""

    call_id: str = ""
    name: str = ""

    def add(self, delta: Mapping[str, object], *, record: int) -> TextChunk | None:
        """Take one fragment of this block.

        Args:
          delta: The fragment.
          record: The index of the record this block becomes.

        Returns:
          chunk: The fragment as text when it is text or reasoning, else ``None``.

        """
        kind = delta.get("type")
        if kind == "text_delta":
            text = str(delta.get("text") or "")
            self.parts.append(text)
            return TextChunk(text=text, record=record)
        if kind == "thinking_delta":
            text = str(delta.get("thinking") or "")
            self.parts.append(text)
            return TextChunk(text=text, thinking=True, record=record)
        if kind == "signature_delta":
            self.signature += str(delta.get("signature") or "")
        elif kind == "input_json_delta":
            self.parts.append(str(delta.get("partial_json") or ""))
        return None

    def record(self) -> AssistantMessage | Thinking | ToolCall | None:
        """Return the finished block as a record.

        Returns:
          record: The record; ``None`` for a block left out of the session.

        """
        joined = "".join(self.parts)
        if self.kind == "text":
            return AssistantMessage(content=joined)
        if self.kind == "thinking":
            return Thinking(content=joined, encrypted=self.signature or None)
        if self.kind == "redacted_thinking":
            return Thinking(encrypted=self.sealed)
        # A model asked to print its context echoes the tool spec, and the
        # server parses ``$FUNCTION_NAME`` back as a call; kept, the next
        # request carries a tool use no result answers, and fails.
        if self.kind != "tool_use" or not re.fullmatch(
            r"[A-Za-z0-9_-]{1,128}",
            self.name,
        ):
            return None
        # A reply cut off by ``max_tokens`` mid-call ends inside its arguments;
        # the call is dropped, and what the model said before it is kept.
        try:
            arguments = json_loads(joined or "{}", dict[str, object])
        except ValueError:
            return None
        return ToolCall(
            call_id=self.call_id,
            name=self.name,
            arguments=immutable(arguments),
        )


def _object(value: object) -> dict[str, object]:
    try:
        return from_plain(value, dict[str, object])
    except ReadError:
        return {}


# Bare "policy" is not enough: caching and retention rejections cite policies
# too, and a false match tells the user to abandon a conversation that a
# corrected request would serve.
def _blocked(message: str) -> bool:
    signals = (
        "content filtering policy",
        "safeguards flagged",
        "usage policy",
        "cyber-related safeguards",
    )
    if any(signal in message for signal in signals):
        return True
    acted = any(act in message for act in ("block", "flag", "reject", "refus"))
    return acted and any(
        term in message for term in ("content filter", "safeguard", "safety")
    )


# Anthropic reports an empty account as a 429 too, and no wait refills it.
def _out_of_credit(message: str) -> bool:
    phrases = (
        "credit balance is too low",
        "usage credits are required",
        "insufficient credits",
        "exceeded your usage credits",
        "out of extra usage",
    )
    return any(phrase in message for phrase in phrases)


def _retry_after(headers: Mapping[str, str], *, rate_limited: bool) -> float:
    """Return the advised wait in seconds; ``0`` when the server gave none."""
    # At most eight days: a spent weekly window refills up to seven days away.
    lower = {key.lower(): value for key, value in headers.items()}
    # A spent window says when it resets to the second; ``retry-after`` rounds.
    reset = _number(lower.get("anthropic-ratelimit-unified-reset"))
    if (
        rate_limited
        and reset is not None
        and lower.get("anthropic-ratelimit-unified-status") == "rejected"
    ):
        return min(max(reset - time.time(), 0.0), 691_200.0)
    raw = lower.get("retry-after")
    seconds = _number(raw)
    if seconds is not None:
        return min(max(seconds, 0.0), 691_200.0)
    at: float | None = None
    if raw is not None:
        try:
            at = parsedate_to_datetime(raw).timestamp()
        except (TypeError, ValueError):
            at = None
    # The unified reset is on every reply, as when the window rolls over; it
    # is a wait only for a request the limit stopped.
    if at is None and rate_limited:
        at = _number(lower.get("anthropic-ratelimit-unified-reset"))
    return 0.0 if at is None else min(max(at - time.time(), 0.0), 691_200.0)


def _limit(claim: str) -> str:
    """Return a spent window's name: ``seven_day_opus`` is ``weekly Opus``; else ``""``."""
    for window, name in (("five_hour", "5-hour"), ("seven_day", "weekly")):
        if claim == window:
            return name
        if claim.startswith(f"{window}_"):
            return f"{name} {claim.removeprefix(f'{window}_').capitalize()}"
    return ""


def _number(raw: str | None) -> float | None:
    if raw is None:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _blocking(status: str | None) -> bool:
    return status is not None and status.strip().lower() not in (
        "allowed",
        "allowed_warning",
    )
