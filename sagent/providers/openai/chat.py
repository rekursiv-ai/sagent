"""Translate between our records and Chat Completions JSON, without I/O.

The wire every OpenAI-compatible server speaks. A tool result cannot carry an
image there, so its images follow in a user message of their own.

Typical usage example:

  body = encode(request, capability=c, settings=s, reasoning_field="")
  reader = StreamReader(capability=c, settings=s, reasoning_field="")
  async for event in wire.events(response.aiter_lines()):
      chunk = reader.feed(event)                     # New text, to show at once.
  reader.response()                                  # The records and their cost.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from treekle import (
    ReadError,
    from_plain,
    immutable,
    json_dumps,
    json_loads,
    mutable,
)

from sagent.model.cost import TokenCount
from sagent.model.exceptions import DisconnectedError
from sagent.model.types import Response, TextChunk
from sagent.providers.openai import wire
from sagent.session.types import AssistantMessage, Thinking, ToolCall, ToolResult


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from sagent.model.capability import ModelCapability, ModelLimits, ModelSettings
    from sagent.model.types import Request
    from sagent.session.types import Attachment


__all__ = [
    "StreamReader",
    "encode",
]


def encode(
    request: Request,
    *,
    capability: ModelCapability,
    settings: ModelSettings,
    reasoning_field: str,
    temperature: float | None = None,
) -> dict[str, object]:
    """Return the body of a streamed ``/chat/completions`` request.

    Args:
      request: What the model is asked.
      capability: The model it is asked of.
      settings: The settings chosen for that model.
      reasoning_field: The assistant message field the server reads the model's
        earlier reasoning back from; empty leaves reasoning out.
      temperature: The sampling temperature; ``None`` sends none.

    Returns:
      body: The request body.

    """
    limits = capability.limits
    body: dict[str, object] = {
        "model": capability.wire_model_id or capability.model_id,
        "messages": _messages(
            request,
            limits=limits,
            reasoning_field=reasoning_field,
        ),
    }
    if temperature is not None:
        body["temperature"] = temperature
    if reply := request.response_tokens(limits):
        # A reasoning model rejects ``max_tokens`` with a 400; a row that
        # offers an effort is one.
        cap = (
            "max_tokens"
            if capability.thinking_effort == frozenset({"none"})
            else "max_completion_tokens"
        )
        body[cap] = reply
    body["stream"] = True
    body["stream_options"] = {"include_usage": True}
    effort = settings.thinking_effort
    # ``none`` is said, not left out: left out, the model reasons at its default.
    if capability.thinking_effort != frozenset({"none"}):
        body["reasoning_effort"] = (
            "minimal"
            if effort == "min"
            else "high"
            if effort in ("xhigh", "max")
            else effort
        )
    if settings.service_tier != "auto":
        body["service_tier"] = settings.service_tier
    if request.tools:
        body["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": mutable(tool.schema),
                },
            }
            for tool in request.tools
        ]
    return body


class StreamReader:
    """Collects a streamed reply's chunks into records.

    Args:
      capability: The model replying, whose prices cost the reply.
      settings: The settings it was asked with.
      reasoning_field: The delta field the server streams reasoning in; empty
        for a server that streams none.

    """

    def __init__(
        self,
        *,
        capability: ModelCapability,
        settings: ModelSettings,
        reasoning_field: str,
    ) -> None:
        self._capability = capability
        self._settings = settings
        self._reasoning_field = reasoning_field
        self._text: list[str] = []
        self._thinking: list[str] = []
        self._calls: dict[int, _Call] = {}
        self._usage: dict[str, object] = {}
        self._finish = ""
        self._refused = False

    def feed(self, event: Mapping[str, object]) -> TextChunk | None:
        """Take one chunk, and return the text it adds, if any.

        Args:
          event: One server-sent chunk.

        Returns:
          chunk: The text or reasoning the chunk adds; ``None`` for the rest.

        Raises:
          Exception: The ``wire.to_exception`` of an in-stream error.

        """
        # A chunk may carry ``"error": null``; only a value there is an error.
        if event.get("error") is not None or event.get("object") == "error":
            raise wire.to_exception(0, body=event)
        # Most chunks carry ``"usage": null``; the last carries the counts.
        if usage := _object(event.get("usage")):
            self._usage = usage
        choices = _objects(event.get("choices"))
        if not choices:
            return None
        choice = choices[0]
        self._finish = str(choice.get("finish_reason") or "") or self._finish
        delta = _object(choice.get("delta"))
        for call in _objects(delta.get("tool_calls")):
            self._call(call)
        text = str(delta.get("content") or "")
        refusal = str(delta.get("refusal") or "")
        thinking = (
            str(delta.get(self._reasoning_field) or "") if self._reasoning_field else ""
        )
        self._refused = self._refused or bool(refusal)
        self._text.extend(part for part in (text, refusal) if part)
        if thinking:
            self._thinking.append(thinking)
            return TextChunk(text=thinking, thinking=True)
        # The reply's text is one record, after its reasoning's.
        if text or refusal:
            return TextChunk(text=text + refusal, record=int(bool(self._thinking)))
        return None

    def response(self) -> Response:
        """Return the finished reply.

        Returns:
          response: Its reasoning, text, and calls, its tokens, and their cost.

        Raises:
          DisconnectedError: The stream ended before the model finished.

        """
        if not self._finish:
            raise DisconnectedError("The reply ended before it was finished.")
        records: list[AssistantMessage | Thinking | ToolCall] = []
        if self._thinking:
            records.append(Thinking(content="".join(self._thinking)))
        if self._text:
            records.append(AssistantMessage(content="".join(self._text)))
        records.extend(
            ToolCall(
                call_id=call.call_id,
                name=call.name,
                arguments=immutable(_arguments("".join(call.parts))),
            )
            for _, call in sorted(self._calls.items())
            if call.call_id
        )
        tokens = _tokens(self._usage)
        return Response(
            records=tuple(records),
            tokens=tokens,
            spend=self._capability.prices[
                self._settings.service_tier,
                tokens.prompt,
                datetime.now(UTC).date(),
            ]
            * tokens,
            stop_reason=_stop_reason(
                self._finish,
                refused=self._refused,
                called=any(isinstance(record, ToolCall) for record in records),
            ),
        )

    def _call(self, fragment: Mapping[str, object]) -> None:
        """Take one fragment of a tool call, which arrive sparse, by index."""
        index = from_plain(fragment.get("index"), int, default=0)
        call = self._calls.setdefault(index, _Call())
        call.call_id = str(fragment.get("id") or "") or call.call_id
        function = _object(fragment.get("function"))
        call.name = str(function.get("name") or "") or call.name
        if arguments := str(function.get("arguments") or ""):
            call.parts.append(arguments)


@dataclass(slots=True, kw_only=True)
class _Call:
    """One tool call of a reply, as its fragments arrive."""

    call_id: str = ""
    name: str = ""
    parts: list[str] = field(default_factory=list[str])
    """The argument JSON, in arrival order."""


def _messages(
    request: Request,
    *,
    limits: ModelLimits,
    reasoning_field: str,
) -> list[dict[str, object]]:
    """Return the system prompt and ``records`` as chat messages."""
    # Another vendor's ids may not fit this server's, so each is renamed,
    # consistently within one request.
    ids: dict[str, str] = {}
    messages: list[dict[str, object]] = []
    system = "\n\n".join(text for text in request.system if text)
    if system:
        messages.append({"role": "system", "content": system})
    images: list[dict[str, object]] = []
    turn: _Turn | None = None
    for record in wire.flatten(request.records):
        if isinstance(record, (AssistantMessage, Thinking, ToolCall)):
            if turn is None:
                _flush(messages, images=images)
                turn = _Turn()
                messages.append(turn.message)
            turn.add(record, ids=ids, reasoning_field=reasoning_field)
            continue
        said = wire.user_content(record)
        if said is not None:
            turn = None
            _flush(messages, images=images)
            messages.append(_user(*said, limits=limits))
        elif isinstance(record, ToolResult):
            turn = None
            messages.append(_result(record, ids=ids))
            images.extend(
                {"type": "image_url", "image_url": {"url": url}}
                for attachment in record.attachments
                if (url := wire.image_url(attachment, limits=limits)) is not None
            )
    _flush(messages, images=images)
    # A turn of reasoning this server is not sent says nothing, and an empty
    # assistant message is rejected.
    empty = {"role": "assistant", "content": None}
    return [message for message in messages if message != empty]


class _Turn:
    """One assistant message, gathered from the model's consecutive records."""

    def __init__(self) -> None:
        self.message: dict[str, object] = {"role": "assistant", "content": None}
        self._text: list[str] = []
        self._reasoning: list[str] = []
        self._calls: list[dict[str, object]] = []

    def add(
        self,
        record: AssistantMessage | Thinking | ToolCall,
        *,
        ids: dict[str, str],
        reasoning_field: str,
    ) -> None:
        """Add ``record`` to this message.

        Args:
          record: One of the model's records in this turn.
          ids: Each call id, renamed for this request.
          reasoning_field: Where the server reads reasoning back; empty for none.

        """
        match record:
            case AssistantMessage(content=str() as text) if text:
                self._text.append(text)
                self.message["content"] = "".join(self._text)
            # A thinking model expects its own reasoning back on a tool-call
            # turn; without it the continuation degrades or is rejected.
            case Thinking(content=str() as text) | Thinking(summary=str() as text) if (
                reasoning_field and text
            ):
                self._reasoning.append(text)
                self.message[reasoning_field] = "".join(self._reasoning)
            case ToolCall(call_id=call_id, name=name, arguments=arguments):
                self._calls.append(
                    {
                        "id": ids.setdefault(call_id, f"call_{len(ids)}"),
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": json_dumps(mutable(arguments)),
                        },
                    },
                )
                self.message["tool_calls"] = self._calls
            case _:
                pass


def _user(
    text: str,
    attachments: Sequence[Attachment],
    *,
    limits: ModelLimits,
) -> dict[str, object]:
    """Return a user message, its images first; other attachments are left out."""
    blocks: list[dict[str, object]] = [
        {"type": "image_url", "image_url": {"url": url}}
        for attachment in attachments
        if (url := wire.image_url(attachment, limits=limits)) is not None
    ]
    if not blocks:
        return {"role": "user", "content": text}
    if text:
        blocks.append({"type": "text", "text": text})
    return {"role": "user", "content": blocks}


def _result(result: ToolResult, *, ids: dict[str, str]) -> dict[str, object]:
    text = result.text()
    return {
        "role": "tool",
        "tool_call_id": ids.setdefault(result.call_id, f"call_{len(ids)}"),
        "content": f"[Error] {text}" if result.is_error is True and text else text,
    }


# Tool messages must be contiguous, so a run of results' images waits for the
# run to end and then goes out as one user message.
def _flush(
    messages: list[dict[str, object]],
    *,
    images: list[dict[str, object]],
) -> None:
    if images:
        messages.append({"role": "user", "content": list(images)})
        images.clear()


def _stop_reason(finish: str, *, refused: bool, called: bool) -> str:
    """Return ``finish`` in the vocabulary every provider's ``Response`` uses."""
    if refused and finish == "stop":
        return "refusal"
    reason = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "function_call": "tool_use",
        "content_filter": "refusal",
    }.get(finish, finish)
    # Some servers say ``stop`` for a reply that calls tools.
    return "tool_use" if called and reason == "end_turn" else reason


def _tokens(usage: Mapping[str, object]) -> TokenCount:
    """Return the reply's usage, its cached prompt split from the uncached."""
    details = _object(usage.get("prompt_tokens_details"))
    read = from_plain(details.get("cached_tokens"), int, default=0)
    written = from_plain(details.get("cache_write_tokens"), int, default=0)
    prompt = from_plain(usage.get("prompt_tokens"), int, default=0)
    return TokenCount(
        request=max(prompt - read - written, 0),
        response=from_plain(usage.get("completion_tokens"), int, default=0),
        cache_write=written,
        cache_read=read,
    )


# A call whose arguments do not parse is kept with none, so the tool reports the
# bad call to the model rather than the reply failing whole.
def _arguments(text: str) -> dict[str, object]:
    if not text:
        return {}
    try:
        return json_loads(text, dict[str, object])
    except ValueError:
        return {}


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
