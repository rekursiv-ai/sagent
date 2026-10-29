"""Translate between our records and OpenAI's Responses API JSON, without I/O.

Every request is stateless (``store: false``): the whole conversation rides
each one, and reasoning comes back sealed so the next request can replay it.

Typical usage example:

  body = encode(request, capability=c, settings=s)   # POST to /responses
  reader = StreamReader(capability=c, settings=s)
  async for event in wire.events(response.aiter_lines()):
      chunk = reader.feed(event)                      # New text, to show at once.
  reader.response()                                   # The records and their cost.
"""

from __future__ import annotations

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
    from sagent.model.cost import ServiceTier, TokenCost
    from sagent.model.types import Request
    from sagent.session.types import Attachment, SessionRecord
    from sagent.tools.types import Tool


__all__ = [
    "StreamReader",
    "encode",
]


def encode(
    request: Request,
    *,
    capability: ModelCapability,
    settings: ModelSettings,
    temperature: float | None = None,
) -> dict[str, object]:
    """Return the body of a streamed ``/responses`` request.

    Args:
      request: What the model is asked.
      capability: The model it is asked of.
      settings: The settings chosen for that model.
      temperature: The sampling temperature a model that cannot reason is sent;
        ``None`` sends none.

    Returns:
      body: The request body.

    """
    limits = capability.limits
    body: dict[str, object] = {
        "model": capability.wire_model_id or capability.model_id,
        "input": _input(request.records, limits=limits),
        "instructions": "\n\n".join(text for text in request.system if text),
        "stream": True,
        "store": False,
    }
    if request.tools:
        body["tools"] = [_tool(tool) for tool in request.tools]
        # The runtime runs a reply's independent calls at once anyway.
        body["parallel_tool_calls"] = True
        if request.allowed is not None:
            body["tool_choice"] = _allowed(request.tools, allowed=request.allowed)
    # ``none`` is said, not left out: left out, the model reasons at its default.
    if settings.thinking_effort == "none" and capability.thinking_effort != frozenset(
        {"none"},
    ):
        body["reasoning"] = {"effort": "none"}
    elif settings.thinking_effort != "none":
        reasoning: dict[str, object] = {"effort": settings.thinking_effort}
        if settings.thinking_output == "text":
            reasoning["summary"] = "auto"
        if capability.model_id in frozenset(
            {
                "astra-6.0",
                "sol-6.1",
                "sol-6.0",
                "luna-6.0",
                "sol-5.6",
                "gpt-5.6",
                "terra-5.6",
                "luna-5.6",
            },
        ):
            reasoning["context"] = "all_turns"
        body["reasoning"] = reasoning
        # Stateless, so the sealed reasoning is the only way it reaches the
        # next request.
        body["include"] = ["reasoning.encrypted_content"]
    if reply := request.response_tokens(limits):
        body["max_output_tokens"] = reply
    # A reasoning model rejects a sampling temperature.
    if temperature is not None and capability.thinking_effort == frozenset({"none"}):
        body["temperature"] = temperature
    if settings.service_tier != "auto":
        body["service_tier"] = settings.service_tier
    body["prompt_cache_key"] = wire.cache_key(request)
    return body


class StreamReader:
    """Collects a streamed reply's events into records.

    Args:
      capability: The model replying, whose prices cost the reply.
      settings: The settings it was asked with.

    """

    def __init__(self, *, capability: ModelCapability, settings: ModelSettings) -> None:
        self._capability = capability
        self._settings = settings
        self._records: dict[int, AssistantMessage | Thinking | ToolCall] = {}
        self._refusing: set[str] = set()
        self._refused = False
        self._final: dict[str, object] | None = None

    def feed(self, event: Mapping[str, object]) -> TextChunk | None:
        """Take one event, and return the text it adds, if any.

        Args:
          event: One server-sent event.

        Returns:
          chunk: The text or reasoning the event adds; ``None`` for the rest.

        Raises:
          Exception: The ``wire.to_exception`` of an error or failed reply.

        """
        kind = event.get("type")
        delta = str(event.get("delta") or "")
        # Items stream one after another, so the records before this one are done.
        at = from_plain(event.get("output_index"), int, default=len(self._records))
        record = sum(1 for i in self._records if i < at)
        if kind == "response.output_text.delta":
            return TextChunk(text=delta, record=record)
        if kind in (
            "response.reasoning_summary_text.delta",
            "response.reasoning_text.delta",
        ):
            return TextChunk(text=delta, thinking=True, record=record)
        if kind == "response.refusal.delta":
            self._refusing.add(str(event.get("item_id") or ""))
            return TextChunk(text=delta, record=record)
        if kind == "response.refusal.done":
            # A refusal sent whole, with no deltas, is shown whole.
            if str(event.get("item_id") or "") not in self._refusing:
                return TextChunk(text=str(event.get("refusal") or ""), record=record)
        elif kind == "response.output_item.done":
            self._done(
                from_plain(event.get("output_index"), int, default=len(self._records)),
                item=_object(event.get("item")),
            )
        elif kind in ("response.completed", "response.incomplete"):
            self._final = _object(event.get("response"))
        elif kind == "response.failed":
            reply = _object(event.get("response"))
            raise wire.to_exception(0, body=reply.get("error") or reply)
        elif kind == "error":
            raise wire.to_exception(0, body=event)
        return None

    def response(self) -> Response:
        """Return the finished reply.

        Returns:
          response: Its records in output order, its tokens, and their cost.

        Raises:
          DisconnectedError: The stream ended before the reply was finished.

        """
        final = self._final
        if final is None:
            raise DisconnectedError("The reply ended before it was finished.")
        records = tuple(record for _, record in sorted(self._records.items()))
        tokens = _tokens(_object(final.get("usage")))
        reason = str(_object(final.get("incomplete_details")).get("reason") or "")
        if self._refused or reason == "content_filter":
            stop_reason = "refusal"
        elif final.get("status") == "incomplete":
            stop_reason = "max_tokens"
        elif any(isinstance(record, ToolCall) for record in records):
            stop_reason = "tool_use"
        else:
            stop_reason = "end_turn"
        return Response(
            records=records,
            tokens=tokens,
            spend=self._spend(tokens, reported=final.get("service_tier")),
            stop_reason=stop_reason,
        )

    def _done(self, index: int, *, item: Mapping[str, object]) -> None:
        """Keep a finished output item as its record."""
        kind = item.get("type")
        if kind == "message":
            parts = _objects(item.get("content"))
            self._refused = self._refused or any(
                part.get("type") == "refusal" for part in parts
            )
            text = "".join(
                str(part.get("text") or part.get("refusal") or "") for part in parts
            )
            if text:
                self._records[index] = AssistantMessage(content=text)
        elif kind == "reasoning":
            summary = "\n\n".join(
                str(part.get("text") or "") for part in _objects(item.get("summary"))
            )
            text = "".join(
                str(part.get("text") or "") for part in _objects(item.get("content"))
            )
            sealed = item.get("encrypted_content")
            if summary or text or isinstance(sealed, str):
                self._records[index] = Thinking(
                    content=text or None,
                    summary=summary or None,
                    encrypted=sealed if isinstance(sealed, str) else None,
                    extra={"reasoning_id": str(item.get("id") or "")},
                )
        elif kind == "function_call":
            self._records[index] = ToolCall(
                call_id=str(item.get("call_id") or ""),
                name=str(item.get("name") or ""),
                arguments=immutable(_arguments(str(item.get("arguments") or ""))),
            )

    # The server's tier names grow without notice, and the reply is already
    # paid for, so an unknown tier is priced at the one asked for.
    def _spend(self, tokens: TokenCount, *, reported: object) -> TokenCost:
        """Price ``tokens`` at the tier the server served, else the tier asked for."""
        served = _served(reported if isinstance(reported, str) else None)
        today = datetime.now(UTC).date()
        prices = self._capability.prices
        if served is not None:
            try:
                return prices[served, tokens.prompt, today] * tokens
            except KeyError:
                pass
        return prices[self._settings.service_tier, tokens.prompt, today] * tokens


def _input(
    records: Sequence[SessionRecord],
    *,
    limits: ModelLimits,
) -> list[dict[str, object]]:
    """Return ``records`` as Responses input items; none for bookkeeping records."""
    # Another vendor's ids may not fit OpenAI's, so each is renamed,
    # consistently within one request.
    ids: dict[str, str] = {}
    items: list[dict[str, object]] = []
    for record in wire.flatten(records):
        said = wire.user_content(record)
        if said is not None:
            items.append(_user(*said, limits=limits))
            continue
        match record:
            case AssistantMessage(content=str() as text) if text:
                items.append({"role": "assistant", "content": text})
            # OpenAI accepts back only reasoning it sealed itself; another
            # vendor's thinking has no reasoning id and is left out.
            case Thinking(
                encrypted=str() as sealed,
                extra={"reasoning_id": str() as reasoning_id},
            ) if reasoning_id:
                summary = record.summary or ""
                items.append(
                    {
                        "type": "reasoning",
                        "id": reasoning_id,
                        "summary": [{"type": "summary_text", "text": summary}]
                        if summary
                        else [],
                        "encrypted_content": sealed,
                    },
                )
            case ToolCall(call_id=call_id, name=name, arguments=arguments):
                native = ids.setdefault(call_id, f"fc_{len(ids)}")
                items.append(
                    {
                        "type": "function_call",
                        "id": native,
                        "call_id": native,
                        "name": name,
                        "arguments": json_dumps(mutable(arguments)),
                        "status": "completed",
                    },
                )
            case ToolResult():
                items.append(
                    {
                        "type": "function_call_output",
                        "call_id": ids.setdefault(record.call_id, f"fc_{len(ids)}"),
                        "output": _output(record, limits=limits),
                        "status": "completed",
                    },
                )
            case _:
                pass
    return items


def _user(
    text: str,
    attachments: Sequence[Attachment],
    *,
    limits: ModelLimits,
) -> dict[str, object]:
    """Return a user turn, its images inline; other attachments are left out."""
    images = [
        {"type": "input_image", "detail": "auto", "image_url": url}
        for attachment in attachments
        if (url := wire.image_url(attachment, limits=limits)) is not None
    ]
    if not images:
        return {"role": "user", "content": text}
    blocks: list[dict[str, object]] = (
        [{"type": "input_text", "text": text}] if text else []
    )
    return {"role": "user", "content": [*blocks, *images]}


def _output(result: ToolResult, *, limits: ModelLimits) -> object:
    """Return a tool result's output: its text, or its text and images."""
    text = result.text()
    if result.is_error is True and text:
        text = f"[Error] {text}"
    images: list[dict[str, object]] = [
        {"type": "input_image", "detail": "auto", "image_url": url}
        for attachment in result.attachments
        if (url := wire.image_url(attachment, limits=limits)) is not None
    ]
    if not images:
        return text
    return [{"type": "input_text", "text": text}, *images] if text else images


def _tool(tool: Tool) -> dict[str, object]:
    # Our schemas are not written to strict mode's rules (every property
    # required, no open objects), which strict mode rejects.
    return {
        "type": "function",
        "name": tool.name,
        "description": tool.description,
        "parameters": mutable(tool.schema),
        "strict": False,
    }


def _allowed(tools: Sequence[Tool], *, allowed: frozenset[str]) -> object:
    """Return the ``tool_choice`` that lets the model call only ``allowed``."""
    # Every schema still ships, so the cached prefix is the same whatever the
    # gate; a gate naming no tool offered allows none, as an empty list fails.
    names = [tool.name for tool in tools if tool.name in allowed]
    if not names:
        return "none"
    return {
        "type": "allowed_tools",
        "mode": "auto",
        "tools": [{"type": "function", "name": name} for name in names],
    }


def _tokens(usage: Mapping[str, object]) -> TokenCount:
    """Return the reply's usage, its cached prompt split from the uncached."""
    details = _object(usage.get("input_tokens_details"))
    read = from_plain(details.get("cached_tokens"), int, default=0)
    written = from_plain(details.get("cache_write_tokens"), int, default=0)
    prompt = from_plain(usage.get("input_tokens"), int, default=0)
    return TokenCount(
        request=max(prompt - read - written, 0),
        response=from_plain(usage.get("output_tokens"), int, default=0),
        cache_write=written,
        cache_read=read,
    )


def _served(reported: str | None) -> ServiceTier | None:
    """Return the tier whose card prices a reply; ``None`` for a tier unknown."""
    tiers: dict[str | None, ServiceTier] = {
        None: "auto",
        "auto": "auto",
        "default": "auto",
        # Scale Tier is prepaid capacity, metered at the standard card.
        "scale": "auto",
        "flex": "flex",
        "priority": "priority",
        "fast": "priority",
        "ultrafast": "ultrafast",
    }
    return tiers.get(reported)


# A call whose arguments do not parse is kept with none, so the tool reports the
# bad call to the model rather than the reply failing whole.
def _arguments(text: str) -> dict[str, object]:
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
