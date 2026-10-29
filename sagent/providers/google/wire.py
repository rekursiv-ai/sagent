"""Translate between our records and Gemini's ``generateContent`` JSON, without I/O.

Typical usage example:

  body = encode(request, capability=c, settings=s)
  # POST body to models/{id}:streamGenerateContent?alt=sse.

  reader = StreamReader(capability=c)              # The reply comes in fragments.
  async for event in events(response.aiter_lines()):
      for chunk in reader.feed(event):              # New text, to show at once.
          ...
  reader.response()                                 # The finished records and cost.

  raise to_exception(status, body=reply, headers=headers)
"""

from __future__ import annotations

from datetime import UTC, datetime
from functools import lru_cache
from typing import TYPE_CHECKING

import base64
import logging
import math
import uuid

from treekle import (
    ReadError,
    from_plain,
    immutable,
    json_dumps,
    json_loads,
    mutable,
)

from sagent.model import image
from sagent.model.cost import TokenCount
from sagent.model.exceptions import (
    AuthError,
    BillingError,
    ContextOverflowError,
    DisconnectedError,
    OverloadedError,
    RateLimitedError,
    RequestTooLargeError,
)
from sagent.model.types import Response, TextChunk
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
    from collections.abc import AsyncIterator, Iterator, Mapping, Sequence

    from treekle import MutablePlainTree, PlainTree

    from sagent.model.capability import (
        ModelCapability,
        ModelLimits,
        ModelSettings,
    )
    from sagent.model.types import Request
    from sagent.session.types import Attachment, SessionRecord


__all__ = [
    "StreamReader",
    "attachment_tokens",
    "encode",
    "estimate",
    "events",
    "to_exception",
]

logger = logging.getLogger(__name__)


def encode(
    request: Request,
    *,
    capability: ModelCapability,
    settings: ModelSettings,
) -> dict[str, object]:
    """Return the body of a ``generateContent`` request.

    Args:
      request: What the model is asked.
      capability: The model it is asked of.
      settings: The settings chosen for that model.

    Returns:
      body: The request body.

    """
    limits = capability.limits
    # ``maxOutputTokens`` only for a reply the window holds less of: otherwise
    # the model's own ceiling is the default, and the catalog's the same number.
    config: dict[str, object] = (
        {"maxOutputTokens": request.response_tokens(limits)}
        if request.max_response_tokens
        else {}
    )
    thinking = _thinking(capability, settings=settings)
    if thinking is not None:
        config["thinkingConfig"] = thinking
    body: dict[str, object] = {
        "contents": _contents(request.records, limits=limits),
        "generationConfig": config,
    }
    system = "\n\n".join(text for text in request.system if text)
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    if request.tools:
        body["tools"] = [
            {
                "functionDeclarations": [
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": _parameters(mutable(tool.schema)),
                    }
                    for tool in request.tools
                ],
            },
        ]
    return body


def estimate(body: Mapping[str, object], *, capability: ModelCapability) -> int:
    """Return about how many tokens ``body`` sends, for a transport that cannot count.

    Args:
      body: A ``generateContent`` request body.
      capability: The model, whose characters per token scale the text.

    Returns:
      tokens: The text's length over the model's characters per token, plus
        each inline image's or PDF's, as :func:`attachment_tokens` bills them.

    """
    media: list[tuple[str, str]] = []
    text = json_dumps(_inline_dropped(body, media=media))
    return int(len(text) / capability.approx_chars_per_token) + sum(
        _media_tokens(mime, base64.b64decode(data), max_edge_px=0)
        for mime, data in media
    )


def attachment_tokens(attachment: Attachment, *, limits: ModelLimits) -> int:
    """Return about how many input tokens ``attachment`` bills.

    Args:
      attachment: What is attached.
      limits: The window's, whose image edge the upload shrinks to.

    Returns:
      tokens: 258 per 768-pixel tile of an image, 258 per page of a PDF, and
        none for what is not sent.

    References:
      https://ai.google.dev/gemini-api/docs/tokens
        Images up to 384px are one tile; larger ones tile at 768x768.
      https://ai.google.dev/gemini-api/docs/document-processing
        Each document page is equivalent to 258 tokens.

    """
    return _media_tokens(
        attachment.mime_descriptor,
        attachment.data,
        max_edge_px=limits.max_image_edge_px,
    )


async def events(lines: AsyncIterator[str]) -> AsyncIterator[dict[str, object]]:
    """Yield each server-sent event of a streamed reply, as JSON.

    A malformed event is logged and skipped, as one bad chunk need not cost the
    reply.

    Args:
      lines: The reply's body, line by line.

    Yields:
      event: Each well-formed event's JSON.

    Raises:
      ValueError: Every event was malformed.

    """
    frames = 0
    errors: list[ValueError] = []
    async for text in _frames(lines):
        frames += 1
        try:
            event = json_loads(text, dict[str, object])
        except ValueError as error:
            errors.append(error)
            logger.warning("Gemini sent a malformed stream event: %s", error)
            continue
        yield event
    if errors and len(errors) == frames:
        raise ValueError("Gemini's stream held only malformed events.") from errors[-1]


class StreamReader:
    """Collects a streamed reply's fragments into records.

    Args:
      capability: The model replying, whose prices cost the reply.

    """

    def __init__(self, *, capability: ModelCapability) -> None:
        self._capability = capability
        self._text: list[str] = []
        self._thoughts: list[str] = []
        self._signature: str | None = None
        self._calls: list[ToolCall] = []
        self._usage: dict[str, object] = {}
        self._finish: str | None = None
        self._blocked: str | None = None

    def feed(self, event: Mapping[str, object]) -> list[TextChunk]:
        """Take one event, and return the text it adds.

        Args:
          event: One server-sent event, a partial ``GenerateContentResponse``.

        Returns:
          chunks: The text and reasoning the event adds, in order.

        """
        self._usage = _object(event.get("usageMetadata")) or self._usage
        feedback = _object(event.get("promptFeedback"))
        block = feedback.get("blockReason")
        if isinstance(block, str) and block:
            self._blocked = block
        candidates = _objects(event.get("candidates"))
        if not candidates:
            return []
        first = candidates[0]
        finish = first.get("finishReason")
        if isinstance(finish, str) and finish:
            self._finish = finish
        chunks: list[TextChunk] = []
        for part in _objects(_object(first.get("content")).get("parts")):
            chunk = self._part(part)
            if chunk is not None:
                chunks.append(chunk)
        return chunks

    def response(self) -> Response:
        """Return the finished reply.

        Returns:
          response: Its reasoning, text, and calls, its tokens, and their cost.

        Raises:
          DisconnectedError: The stream ended before the model said it was done.

        """
        if self._blocked is not None:
            stop_reason = "refusal"
        elif self._finish is not None:
            stop_reason = _stop_reason(self._finish, called=bool(self._calls))
        else:
            raise DisconnectedError("Gemini's stream ended before its reply finished.")
        signed: dict[str, PlainTree] = (
            {} if self._signature is None else {"thoughtSignature": self._signature}
        )
        text = "".join(self._text)
        records: list[AssistantMessage | Thinking | ToolCall] = []
        if self._thoughts:
            records.append(
                Thinking(content="".join(self._thoughts), extra={} if text else signed),
            )
        if text:
            records.append(AssistantMessage(content=text, extra=signed))
        records.extend(self._calls)
        tokens = self._tokens()
        return Response(
            records=tuple(records),
            tokens=tokens,
            spend=self._capability.prices[
                "auto",
                tokens.prompt,
                datetime.now(UTC).date(),
            ]
            * tokens,
            stop_reason=stop_reason,
        )

    def _part(self, part: Mapping[str, object]) -> TextChunk | None:
        """Take one part of a candidate; return it as text when it is text."""
        raw = part.get("thoughtSignature")
        signature = raw if isinstance(raw, str) and raw else None
        if "functionCall" in part:
            call = _object(part.get("functionCall"))
            name, call_id = call.get("name"), call.get("id")
            # A call with no name could never be answered; the next request
            # would carry it and fail.
            if not isinstance(name, str) or not name:
                return None
            self._calls.append(
                ToolCall(
                    # Gemini rarely names a call; ours need an id to pair results.
                    call_id=call_id
                    if isinstance(call_id, str) and call_id
                    else f"call_{uuid.uuid4().hex[:24]}",
                    name=name,
                    arguments=immutable(_object(call.get("args"))),
                    extra={} if signature is None else {"thoughtSignature": signature},
                ),
            )
            return None
        text = part.get("text")
        if not isinstance(text, str):
            return None
        # Gemini 3 signs the reply's text; echoing it back keeps the model's
        # reasoning, and a continuation without it is rejected.
        if signature is not None:
            self._signature = signature
        if not text:
            return None
        if part.get("thought") is True:
            self._thoughts.append(text)
            return TextChunk(text=text, thinking=True)
        self._text.append(text)
        # The reply's text is one record, after its reasoning's.
        return TextChunk(text=text, record=int(bool(self._thoughts)))

    def _tokens(self) -> TokenCount:
        usage = self._usage
        cached = from_plain(usage.get("cachedContentTokenCount"), int, default=0)
        # ``promptTokenCount`` includes the cached tokens; ``candidatesTokenCount``
        # excludes the reasoning, which is billed as output.
        prompt = from_plain(usage.get("promptTokenCount"), int, default=0)
        tools = from_plain(usage.get("toolUsePromptTokenCount"), int, default=0)
        reply = from_plain(usage.get("candidatesTokenCount"), int, default=0)
        thoughts = from_plain(usage.get("thoughtsTokenCount"), int, default=0)
        return TokenCount(
            request=max(prompt + tools - cached, 0),
            response=reply + thoughts,
            cache_read=cached,
        )


def to_exception(
    status: int,
    *,
    body: object,
    headers: Mapping[str, str] | None = None,
) -> Exception:
    """Return the exception an error reply means.

    Args:
      status: The HTTP status; ``0`` when there was none.
      body: The reply's JSON, or its text when it was not JSON.
      headers: The reply's headers, which may advise a wait.

    Returns:
      error: The most specific exception that fits.

    """
    # Code Assist wraps its error in a one-element list.
    items = _objects(body)
    fields = items[0] if items else _object(body)
    if isinstance(fields.get("error"), dict):
        fields = _object(fields.get("error"))
    kind = str(fields.get("status") or "")
    text = str(fields.get("message") or body)
    lower = text.lower()
    if _too_large(lower, status=status):
        return RequestTooLargeError(text)
    if _overflowed(lower):
        return ContextOverflowError(text)
    if _out_of_credit(lower):
        return BillingError(text)
    if (
        status in (401, 403)
        or kind in ("UNAUTHENTICATED", "PERMISSION_DENIED")
        or "api key not valid" in lower
    ):
        return AuthError(text)
    wait_sec = _retry_after(fields, headers=headers or {})
    if status == 429 or kind == "RESOURCE_EXHAUSTED":
        return RateLimitedError(text, retry_after_sec=wait_sec)
    if (
        status in (408, 409)
        or status >= 500
        or kind in ("UNAVAILABLE", "INTERNAL", "DEADLINE_EXCEEDED")
    ):
        return OverloadedError(text, retry_after_sec=wait_sec)
    return RuntimeError(f"Gemini replied {status} {kind}: {text}")


def _contents(
    records: Sequence[SessionRecord],
    *,
    limits: ModelLimits,
) -> list[dict[str, object]]:
    """Return ``records`` as contents, consecutive parts of one role in one content."""
    # A result names the function it answers, which only its call records.
    names = {call.call_id: call.name for call in _calls(records)}
    turns: list[tuple[str, list[dict[str, object]]]] = []
    for record in records:
        for role, part in _parts(record, names=names, limits=limits):
            if turns and turns[-1][0] == role:
                turns[-1][1].append(part)
            else:
                turns.append((role, [part]))
    return [{"role": role, "parts": parts} for role, parts in turns]


def _calls(records: Sequence[SessionRecord]) -> Iterator[ToolCall]:
    for record in records:
        if isinstance(record, ToolCall):
            yield record
        elif isinstance(record, ContextClear):
            yield from _calls(record.history)


def _parts(
    record: SessionRecord,
    *,
    names: Mapping[str, str],
    limits: ModelLimits,
) -> list[tuple[str, dict[str, object]]]:
    """Return the role and parts ``record`` becomes; none for bookkeeping records."""
    match record:
        case UserMessage(content=text, attachments=attachments):
            return [("user", p) for p in _said(text or "", attachments, limits=limits)]
        case AgentToAgentMessage(content=text, sender=sender, attachments=attachments):
            body = text if text is not None else ""
            labeled = f"[from {sender}]: {body}" if sender is not None else body
            return [("user", p) for p in _said(labeled, attachments, limits=limits)]
        # A harness note is no one's turn; it rides the user side, marked.
        case SystemMessage(content=str() as text) if text:
            note = f"<system-reminder>\n{text}\n</system-reminder>"
            return [("user", {"text": note})]
        case TurnContext() if record.text():
            note = f"<system-reminder>\n{record.text()}\n</system-reminder>"
            return [("user", {"text": note})]
        case AssistantMessage(content=str() as text) if text:
            return [("model", _signed({"text": text}, extra=record.extra))]
        case ToolCall(name=name, arguments=arguments):
            call: dict[str, object] = {
                "functionCall": {"name": name, "args": mutable(arguments)},
            }
            return [("model", _signed(call, extra=record.extra))]
        case ToolResult():
            text = record.text()
            if record.is_error is True and text:
                text = f"[Error] {text}"
            response: dict[str, object] = {
                "functionResponse": {
                    "name": names.get(record.call_id, record.call_id),
                    "response": {"content": text},
                },
            }
            # Images ride beside the response, in the same user content.
            return [
                ("user", part)
                for part in (response, *_attached(record.attachments, limits=limits))
            ]
        case ContextClear(summary=summary, history=history):
            head: list[tuple[str, dict[str, object]]] = []
            if summary is not None:
                head.append(("user", {"text": summary}))
            return head + [
                pair
                for item in history
                for pair in _parts(item, names=names, limits=limits)
            ]
        # The reasoning's text is not sent back; Gemini reads only its
        # signature, which the reply's text or call carries.
        case _:
            return []


def _said(
    text: str,
    attachments: Sequence[Attachment],
    *,
    limits: ModelLimits,
) -> list[dict[str, object]]:
    """Return a user's words and attachments as parts; one empty part for neither."""
    parts: list[dict[str, object]] = [{"text": text}] if text else []
    parts.extend(_attached(attachments, limits=limits))
    # A content with no parts is rejected.
    return parts or [{"text": ""}]


def _attached(
    attachments: Sequence[Attachment],
    *,
    limits: ModelLimits,
) -> list[dict[str, object]]:
    """Return each image and PDF as an ``inlineData`` part; others are left out."""
    parts: list[dict[str, object]] = []
    for attachment in attachments:
        mime = attachment.mime_descriptor
        if not mime.startswith("image/") and mime != "application/pdf":
            logger.warning("Gemini reads no %s attachment; leaving it out.", mime)
            continue
        data, mime = _upload(
            attachment.data,
            mime=mime,
            max_edge_px=limits.max_image_edge_px,
            max_bytes=limits.max_image_bytes,
        )
        parts.append({"inlineData": {"mimeType": mime, "data": data}})
    return parts


# Every request resends the whole conversation, so without the cache each image
# would be encoded again on every turn.
@lru_cache(maxsize=64)
def _upload(
    data: bytes,
    *,
    mime: str,
    max_edge_px: int,
    max_bytes: int,
) -> tuple[str, str]:
    """Return ``data`` within the limits, base64-encoded, and its media type."""
    # No limit, no decode: Pillow reads neither HEIC nor every format Gemini does.
    if mime.startswith("image/") and (max_edge_px or max_bytes):
        data, mime = image.shrink(data, max_edge_px=max_edge_px, max_bytes=max_bytes)
    return base64.b64encode(data).decode(), mime


def _signed(
    part: dict[str, object],
    *,
    extra: Mapping[str, PlainTree],
) -> dict[str, object]:
    """Return ``part`` with the thought signature its record carries, if any."""
    signature = extra.get("thoughtSignature")
    if isinstance(signature, str) and signature:
        return {**part, "thoughtSignature": signature}
    return part


def _parameters(schema: MutablePlainTree) -> MutablePlainTree:
    """Return ``schema`` without ``additionalProperties``, which Gemini rejects."""
    if isinstance(schema, dict):
        return {
            key: _parameters(value)
            for key, value in schema.items()
            if key != "additionalProperties"
        }
    if isinstance(schema, list):
        return [_parameters(item) for item in schema]
    return schema


# Gemini 3 rows take a ``thinkingLevel`` under every budget mode, its levels being
# dynamic already and no level stopping the thinking. On 2.5 rows budget ``none`` omits
# the config, so the model's own default applies; the rest take a ``thinkingBudget``
# token cap, ``-1`` under ``auto`` and ``0``, which stops it, for effort ``none``.
def _thinking(
    capability: ModelCapability,
    *,
    settings: ModelSettings,
) -> dict[str, object] | None:
    """Return the ``thinkingConfig`` a selection sends; ``None`` sends none."""
    # Gemini 3 cannot stop thinking: its level is the only say over how much.
    leveled = capability.effort_as_level and capability.thinking_effort != frozenset(
        {"none"},
    )
    if settings.thinking_budget == "none" and not leveled:
        return None
    config: dict[str, object] = {
        "includeThoughts": settings.thinking_output == "text",
    }
    effort = settings.thinking_effort
    # A row offering only ``none`` takes no effort, so nothing is said about it.
    if capability.thinking_effort == frozenset({"none"}):
        return config
    if capability.effort_as_level:
        levels = {"min": "minimal", "low": "low", "medium": "medium", "high": "high"}
        # Gemini 3 accepts ``thinkingBudget`` only for backwards compatibility.
        if effort not in levels:
            raise ValueError(f"Effort {effort!r} has no Gemini thinkingLevel.")
        config["thinkingLevel"] = levels[effort]
    elif settings.thinking_budget == "auto" and effort != "none":
        config["thinkingBudget"] = -1
    else:
        config["thinkingBudget"] = {
            "none": 0,
            "min": 1024,
            "low": 4096,
            "medium": 8192,
            "high": 16_384,
            "xhigh": 20_480,
            "max": 24_576,
        }[effort]
    return config


async def _frames(lines: AsyncIterator[str]) -> AsyncIterator[str]:
    """Yield each event's data: its ``data:`` lines, joined, up to a blank line."""
    data: list[str] = []
    async for line in lines:
        if line.startswith("data:"):
            data.append(line.removeprefix("data:"))
        elif not line and data:
            yield "\n".join(data)
            data = []
    if data:
        yield "\n".join(data)


def _stop_reason(finish: str, *, called: bool) -> str:
    """Return a ``finishReason`` in the words every provider shares."""
    if finish == "STOP":
        # Gemini stops the same way whether or not it called a tool.
        return "tool_use" if called else "end_turn"
    if finish == "MAX_TOKENS":
        return "max_tokens"
    if finish in (
        "SAFETY",
        "RECITATION",
        "LANGUAGE",
        "BLOCKLIST",
        "PROHIBITED_CONTENT",
        "SPII",
        "IMAGE_SAFETY",
    ):
        return "refusal"
    return finish


def _inline_dropped(tree: object, *, media: list[tuple[str, str]]) -> object:
    """Return ``tree`` without its inline data, gathering each one's type and base64."""
    fields = _object(tree)
    if fields:
        inline = _object(fields.get("inlineData"))
        if inline:
            mime, data = inline.get("mimeType"), inline.get("data")
            if isinstance(mime, str) and isinstance(data, str):
                media.append((mime, data))
            return dict[str, object]()
        return {
            key: _inline_dropped(value, media=media) for key, value in fields.items()
        }
    try:
        items = from_plain(tree, list[object])
    except ReadError:
        return tree
    return [_inline_dropped(item, media=media) for item in items]


def _media_tokens(mime: str, data: bytes, *, max_edge_px: int) -> int:
    """Return the tokens one image's tiles or one PDF's pages bill."""
    if mime == "application/pdf":
        return 258 * len(image.pages(data))
    if not mime.startswith("image/"):
        return 0
    width, height = image.size(data, max_edge_px=max_edge_px)
    return 258 * math.ceil(width / 768) * math.ceil(height / 768)


# A byte phrase names the request's size, never the token window, so it wins
# over a window phrase in the same message; a bare 413 is bytes only when no
# window phrase says otherwise.
def _too_large(message: str, *, status: int) -> bool:
    phrases = (
        "request_too_large",
        "maximum request size",
        "number of bytes",
        "payload too large",
        "request entity too large",
    )
    if any(phrase in message for phrase in phrases):
        return True
    return status == 413 and not _names_tokens(message)


# Sound only once the byte limit is ruled out: ``exceeds the maximum`` also
# prefixes a byte error, so it may not veto one.
def _overflowed(message: str) -> bool:
    return "exceeds the maximum" in message or _names_tokens(message)


# Each names the token window, never a byte size. The guarded pairs appear
# harmlessly in tool-schema errors without their markers.
def _names_tokens(message: str) -> bool:
    phrases = (
        "maximum number of tokens",
        "input token count",
        "maximum context length",
        "context_length_exceeded",
        "input too large",
        "input too long",
        "prompt is too long",
        "prompt too long",
        "too_long",
        # A single value over its cap: shedding attachment bytes cannot clear
        # it, but shrinking the oversized tool result can.
        "string_above_max_length",
    )
    if any(phrase in message for phrase in phrases):
        return True
    guarded = (
        ("context window", ("exceed", "overflow", "maximum")),
        ("model context", ("exceed",)),
    )
    return any(
        phrase in message and any(marker in message for marker in markers)
        for phrase, markers in guarded
    )


# A quota that resets is a rate limit, and its message mentions billing too;
# only these phrases mean no wait will help.
def _out_of_credit(message: str) -> bool:
    phrases = (
        "billing account",
        "prepayment credits",
        "credits are depleted",
        "billing to be enabled",
    )
    return any(phrase in message for phrase in phrases)


def _retry_after(fields: Mapping[str, object], *, headers: Mapping[str, str]) -> float:
    """Return the advised wait in seconds; ``0`` when the server gave none."""
    # At most eight days: a spent weekly quota refills up to seven days away.
    for detail in _objects(fields.get("details")):
        kind, delay = detail.get("@type"), detail.get("retryDelay")
        if (
            isinstance(kind, str)
            and kind.endswith("RetryInfo")
            and isinstance(delay, str)
        ):
            seconds = _number(delay.removesuffix("s"))
            if seconds is not None:
                return min(max(seconds, 0.0), 691_200.0)
    lower = {key.lower(): value for key, value in headers.items()}
    seconds = _number(lower.get("retry-after"))
    return 0.0 if seconds is None else min(max(seconds, 0.0), 691_200.0)


def _number(raw: str | None) -> float | None:
    if raw is None:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _object(value: object) -> dict[str, object]:
    try:
        return from_plain(value, dict[str, object])
    except ReadError:
        return {}


def _objects(value: object) -> list[dict[str, object]]:
    try:
        return from_plain(value, list[dict[str, object]])
    except ReadError:
        return []
