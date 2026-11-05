"""Cohere's native v2 Chat API over the shared HTTP transport and lifecycle."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, ClassVar, NotRequired, TypedDict, override

from treekle import from_plain

from sagent.providers import cohere
from sagent.providers.openai import chat, wire
from sagent.providers.openai.compat import OpenAICompatible


if TYPE_CHECKING:
    from collections.abc import Mapping
    from types import ModuleType

    from sagent.model.types import Request, Response, TextChunk
    from sagent.providers.openai.compat import OpenAIModel, Reader


__all__ = ["Cohere"]


class Cohere(OpenAICompatible):
    """Serve Cohere's native Chat models using ``COHERE_API_KEY``."""

    section: ClassVar[str] = "cohere"
    aliases: ClassVar[tuple[str, ...]] = ("co",)
    package: ClassVar[ModuleType] = cohere
    url: ClassVar[str] = "https://api.cohere.ai/v2"
    base_url_variable: ClassVar[str] = "COHERE_BASE_URL"
    key_variable: ClassVar[str] = "COHERE_API_KEY"
    path: ClassVar[str] = "/chat"
    reasoning_field: ClassVar[str] = "tool_plan"
    max_tokens_field: ClassVar[str] = "max_tokens"
    temperature: ClassVar[float | None] = None

    @override
    def encode(self, request: Request, *, model: OpenAIModel) -> dict[str, object]:
        tools = tuple(
            tool
            for tool in request.tools
            if request.allowed is None or tool.name in request.allowed
        )
        body = super().encode(replace(request, tools=tools), model=model)
        body.pop("stream_options", None)
        messages = from_plain(body["messages"], list[dict[str, object]])
        for message in messages:
            if message.get("role") == "tool":
                message["content"] = [
                    {"type": "text", "text": message["content"]},
                ]
            elif message.get("content") is None:
                message.pop("content", None)
        body["messages"] = messages
        return body

    @override
    def reader(self, *, model: OpenAIModel) -> Reader:
        return _Reader(
            chat.StreamReader(
                capability=model.capability,
                settings=model.settings,
                reasoning_field=self.reasoning_field,
            ),
        )


class _Reader:
    """Translate native Cohere events into the shared accumulator's vocabulary."""

    def __init__(self, reader: Reader) -> None:
        self._reader = reader

    def feed(self, event: Mapping[str, object]) -> tuple[TextChunk, ...]:
        """Take one native event and return its visible text, if any.

        Args:
          event: One decoded Cohere SSE event.

        Returns:
          chunks: The new text or tool plan; empty for structural events.

        """
        kind = event.get("type")
        if kind == "error" or event.get("error") is not None:
            raise wire.to_exception(0, body=event)
        translated: dict[str, object] = {}
        if kind == "content-delta":
            content = from_plain(event["delta"], _ContentDelta)
            translated["content"] = content["message"]["content"]["text"]
        elif kind == "tool-plan-delta":
            plan = from_plain(event["delta"], _PlanDelta)
            translated["tool_plan"] = plan["message"]["tool_plan"]
        elif kind in ("tool-call-start", "tool-call-delta"):
            call = from_plain(event, _CallEvent)
            translated["tool_calls"] = [
                {
                    **call["delta"]["message"]["tool_calls"],
                    "index": call["index"],
                },
            ]
        elif kind == "message-end":
            delta = from_plain(event["delta"], _EndDelta)
            reason = delta["finish_reason"]
            finish = {
                "COMPLETE": "stop",
                "STOP_SEQUENCE": "stop",
                "TOOL_CALL": "tool_calls",
                "MAX_TOKENS": "length",
            }.get(reason, reason)
            billed = delta.get("usage", {}).get("billed_units", {})
            return self._reader.feed(
                {
                    "choices": [{"finish_reason": finish}],
                    "usage": {
                        "prompt_tokens": int(billed.get("input_tokens", 0)),
                        "completion_tokens": int(billed.get("output_tokens", 0)),
                    },
                },
            )
        return self._reader.feed({"choices": [{"delta": translated}]})

    def response(self) -> Response:
        """Return the shared accumulator's records and billed cost."""
        return self._reader.response()


class _ContentDelta(TypedDict):
    message: _ContentMessage


class _ContentMessage(TypedDict):
    content: _Content


class _Content(TypedDict):
    text: str


class _PlanDelta(TypedDict):
    message: _PlanMessage


class _PlanMessage(TypedDict):
    tool_plan: str


class _CallEvent(TypedDict):
    index: int
    delta: _CallDelta


class _CallDelta(TypedDict):
    message: _CallMessage


class _CallMessage(TypedDict):
    tool_calls: _Call


class _Function(TypedDict, total=False):
    name: str
    arguments: str


class _Call(TypedDict, total=False):
    id: str
    type: str
    function: _Function


class _EndDelta(TypedDict):
    finish_reason: str
    usage: NotRequired[_Usage]


class _Usage(TypedDict, total=False):
    billed_units: _Billed


class _Billed(TypedDict, total=False):
    input_tokens: int | float
    output_tokens: int | float
