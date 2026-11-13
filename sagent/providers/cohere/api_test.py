"""Tests for Cohere native v2 request and streamed event adaptation."""

from typing import override

import httpx2
import pytest

from sagent.model.exceptions import DisconnectedError
from sagent.model.types import Request, Response, TextChunk
from sagent.providers.cohere.api import Cohere
from sagent.providers.resources import SharedResources
from sagent.session.types import (
    AssistantMessage,
    ToolCall,
    ToolResult,
    UserMessage,
)


@pytest.mark.asyncio
async def test_endpoint_auth_and_native_request() -> None:
    provider = Cohere(api_key="key")
    assert await provider.endpoint() == "https://api.cohere.ai/v2"
    assert await provider.credentials() == {"authorization": "Bearer key"}
    assert provider.path == "/chat"
    model = provider["default"]
    assert model.capability.limits.truncates_at_context is None
    body = provider.encode(
        Request(
            records=(
                UserMessage(content="hi"),
                ToolCall(call_id="c", name="Read"),
                ToolResult(call_id="c", state="done", shown="result"),
            ),
        ),
        model=model,
    )
    assert "stream_options" not in body
    assert body["model"] == "command-r-plus-08-2024"
    assert body["messages"] == [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_0",
                    "type": "function",
                    "function": {"name": "Read", "arguments": "{}"},
                },
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_0",
            "content": [{"type": "text", "text": "result"}],
        },
    ]
    await provider.aclose()


def test_native_text_and_billed_usage() -> None:
    provider = Cohere(api_key="key")
    reader = provider.reader(model=provider["default"])
    assert reader.feed(
        {"type": "content-delta", "delta": {"message": {"content": {"text": "hi"}}}},
    ) == (TextChunk(text="hi"),)
    reader.feed(
        {
            "type": "message-end",
            "delta": {
                "finish_reason": "COMPLETE",
                "usage": {
                    "billed_units": {"input_tokens": 3, "output_tokens": 2},
                    "tokens": {"input_tokens": 20, "output_tokens": 8},
                },
            },
        },
    )
    response = reader.response()
    assert response.records == (AssistantMessage(content="hi"),)
    assert response.tokens.request == 3
    assert response.tokens.response == 2
    assert response.stop_reason == "end_turn"


def test_native_fragmented_tool_call_and_plan() -> None:
    provider = Cohere(api_key="key")
    reader = provider.reader(model=provider["default"])
    reader.feed(
        {"type": "tool-plan-delta", "delta": {"message": {"tool_plan": "Read it"}}},
    )
    reader.feed(
        {
            "type": "tool-call-start",
            "index": 1,
            "delta": {
                "message": {
                    "tool_calls": {
                        "id": "c",
                        "function": {"name": "Read", "arguments": ""},
                    },
                },
            },
        },
    )
    reader.feed(
        {
            "type": "tool-call-delta",
            "index": 1,
            "delta": {"message": {"tool_calls": {"function": {"arguments": "{}"}}}},
        },
    )
    reader.feed({"type": "message-end", "delta": {"finish_reason": "TOOL_CALL"}})
    response = reader.response()
    assert response.stop_reason == "tool_use"
    assert ToolCall(call_id="c", name="Read") in response.records
    replay = provider.encode(
        Request(records=response.records),
        model=provider["default"],
    )
    messages = replay["messages"]
    assert messages == [
        {
            "role": "assistant",
            "tool_plan": "Read it",
            "tool_calls": [
                {
                    "id": "call_0",
                    "type": "function",
                    "function": {"name": "Read", "arguments": "{}"},
                },
            ],
        },
    ]


def test_unfinished_native_stream_is_disconnected() -> None:
    provider = Cohere(api_key="key")
    reader = provider.reader(model=provider["default"])
    with pytest.raises(DisconnectedError):
        reader.response()


@pytest.mark.parametrize(
    ("native", "expected"),
    [
        ("COMPLETE", "end_turn"),
        ("STOP_SEQUENCE", "end_turn"),
        ("MAX_TOKENS", "max_tokens"),
        ("TOOL_CALL", "tool_use"),
        ("ERROR", "ERROR"),
        ("TIMEOUT", "TIMEOUT"),
    ],
)
def test_finish_reasons_and_missing_usage(native: str, expected: str) -> None:
    provider = Cohere(api_key="key")
    reader = provider.reader(model=provider["default"])
    assert (
        reader.feed({"type": "message-end", "delta": {"finish_reason": native}}) == ()
    )
    response = reader.response()
    assert response.stop_reason == expected
    assert response.tokens.request == response.tokens.response == 0
    assert response.records == ()


@pytest.mark.parametrize(
    "event",
    [
        {"type": "error", "message": "failure"},
        {"type": "debug", "error": {"message": "failure"}},
    ],
)
def test_stream_errors(event: dict[str, object]) -> None:
    provider = Cohere(api_key="key")
    reader = provider.reader(model=provider["default"])
    with pytest.raises(RuntimeError, match=r"The server replied 0 .*failure"):
        reader.feed(event)


@pytest.mark.parametrize(
    "kind",
    [
        "message-start",
        "content-start",
        "content-end",
        "tool-call-end",
        "citation-start",
        "citation-end",
        "debug",
    ],
)
def test_structural_events_do_not_finish(kind: str) -> None:
    provider = Cohere(api_key="key")
    reader = provider.reader(model=provider["default"])
    assert reader.feed({"type": kind}) == ()
    with pytest.raises(DisconnectedError):
        reader.response()


def test_parallel_calls_preserve_indices() -> None:
    provider = Cohere(api_key="key")
    reader = provider.reader(model=provider["default"])
    assert reader.feed(
        {
            "type": "tool-plan-delta",
            "delta": {"message": {"tool_plan": "Plan"}},
        },
    ) == (TextChunk(text="Plan", thinking=True),)
    for index in (2, 0):
        assert (
            reader.feed(
                {
                    "type": "tool-call-start",
                    "index": index,
                    "delta": {
                        "message": {
                            "tool_calls": {
                                "id": f"c{index}",
                                "function": {"name": f"tool{index}"},
                            },
                        },
                    },
                },
            )
            == ()
        )
    for index in (0, 2):
        assert (
            reader.feed(
                {
                    "type": "tool-call-delta",
                    "index": index,
                    "delta": {
                        "message": {
                            "tool_calls": {
                                "function": {"arguments": f'{{"value": {index}}}'},
                            },
                        },
                    },
                },
            )
            == ()
        )
    reader.feed({"type": "message-end", "delta": {"finish_reason": "TOOL_CALL"}})
    calls = tuple(
        record for record in reader.response().records if isinstance(record, ToolCall)
    )
    assert calls == (
        ToolCall(call_id="c0", name="tool0", arguments={"value": 0}),
        ToolCall(call_id="c2", name="tool2", arguments={"value": 2}),
    )


@pytest.mark.asyncio
async def test_native_stream_uses_shared_transport_and_ownership() -> None:
    async with _Resources() as resources:
        async with Cohere(resources=resources, api_key="key") as provider:
            events = [event async for event in provider["default"].stream(Request())]
        assert not resources.http.is_closed
        assert len(events) == 2
        assert events[0] == TextChunk(text="hi")
        response = events[1]
        assert isinstance(response, Response)
        assert response.records == (AssistantMessage(content="hi"),)
        assert response.stop_reason == "end_turn"
    assert resources.client.is_closed
    (sent,) = resources.requests
    assert str(sent.url) == "https://api.cohere.ai/v2/chat"
    assert sent.method == "POST"
    assert sent.headers["authorization"] == "Bearer key"
    assert sent.headers["accept"] == "text/event-stream"
    assert b'"stream_options"' not in sent.content
    owned = Cohere(api_key="key")
    client = owned.resources.http
    await owned.aclose()
    assert client.is_closed


class _Resources(SharedResources):
    def __init__(self) -> None:
        super().__init__()
        self.requests: list[httpx2.Request] = []
        self.client = httpx2.AsyncClient(transport=httpx2.MockTransport(self.answer))

    @property
    @override
    def http(self) -> httpx2.AsyncClient:
        return self.client

    @override
    async def aclose(self) -> None:
        await super().aclose()
        await self.client.aclose()

    def answer(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        return httpx2.Response(
            200,
            content=(
                b'data: {"type":"content-delta","delta":{"message":{"content":{"text":"hi"}}}}\n\n'
                b'data: {"type":"message-end","delta":{"finish_reason":"COMPLETE"}}\n\n'
            ),
        )


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
