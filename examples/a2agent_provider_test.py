"""Keyless checks of the hosted example through the real compatible transport."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, override

import asyncio
import json

import httpx2
import pytest

from examples.a2agent_provider import A2Agent, create_provider
from sagent.tools import tool
from sagent.types.capability import ModelLimits
from sagent.types.cost import TokenCount, TokenPrice
from sagent.types.model import ModelRequest, StreamInterruptedError
from sagent.types.runtime import (
    ModelResponsePartial,
    RuntimeEvent,
    ToolResult,
    UserMessage,
)


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from sagent.lib.custom_json import MutableJSON


def _prices() -> TokenPrice:
    """Synthetic fixture rates, not A2Agent pricing."""
    return TokenPrice(
        request=1.0,
        response=2.0,
        cache_read=0.25,
        cache_write=1.25,
        cache_write_1h=2.0,
    )


def _limits() -> ModelLimits:
    return ModelLimits(max_request_tokens=8_000, max_response_tokens=1_000)


@pytest.fixture
def provider(monkeypatch: pytest.MonkeyPatch) -> A2Agent:
    monkeypatch.setenv("A2AGENT_API_KEY", "test-a2agent-key")
    return create_provider("test-model", limits=_limits(), prices=_prices())


def test_catalog_and_endpoint_are_instance_specific(provider: A2Agent) -> None:
    other = create_provider(
        "other-model",
        limits=_limits(),
        prices=_prices(),
        base_url="https://gateway.invalid/prefix/v1/",
    )
    assert provider.base_url == "https://api.a2agent.me/v1"
    assert provider.model().capability.model_id == "test-model"
    assert provider.model("utility").limits == _limits()
    assert (
        other.model()._endpoint == "https://gateway.invalid/prefix/v1/chat/completions"
    )
    assert not A2Agent.catalog.rows
    with pytest.raises(ValueError, match="Unknown model"):
        provider.model("other-model")


def test_missing_key_does_not_fall_back_to_openai(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("A2AGENT_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-key")
    with pytest.raises(RuntimeError, match="A2Agent API key not configured"):
        create_provider("test-model", limits=_limits(), prices=_prices())


@pytest.mark.parametrize("model_id", ["", " ", " test-model"])
def test_empty_or_padded_model_id_is_rejected(model_id: str) -> None:
    with pytest.raises(ValueError, match="model ID"):
        create_provider(model_id, limits=_limits(), prices=_prices())


@pytest.mark.parametrize("limit", [0, -1])
def test_unknown_or_negative_limits_are_rejected(limit: int) -> None:
    with pytest.raises(ValueError, match="positive input and output"):
        create_provider(
            "test-model",
            limits=replace(_limits(), max_request_tokens=limit),
            prices=_prices(),
        )


@pytest.mark.parametrize("rate", [-1.0, float("nan"), float("inf")])
def test_invalid_rates_are_rejected(rate: float) -> None:
    with pytest.raises(ValueError, match="finite non-negative rates"):
        create_provider(
            "test-model",
            limits=_limits(),
            prices=replace(_prices(), cache_read=rate),
        )


@tool(name="Echo")
def _echo(text: str) -> str:
    """Echo text for an in-memory tool round trip.

    Args:
      text: Text to return.

    Returns:
      echoed: The supplied text.

    """
    return text


def _sse(events: list[MutableJSON], *, done: bool = True) -> httpx2.Response:
    lines = [f"data: {json.dumps(event)}\n\n" for event in events]
    if done:
        lines.append("data: [DONE]\n\n")
    return httpx2.Response(
        200,
        content="".join(lines),
        headers={"Content-Type": "text/event-stream"},
    )


@pytest.mark.asyncio
async def test_stream_tool_fragments_usage_and_result_round_trip(
    provider: A2Agent,
) -> None:
    requests: list[MutableJSON] = []

    def handle(request: httpx2.Request) -> httpx2.Response:
        assert str(request.url) == "https://api.a2agent.me/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer test-a2agent-key"
        body = json.loads(request.content)
        requests.append(body)
        assert body["model"] == "test-model"
        assert body["stream"] is True
        assert body["stream_options"] == {"include_usage": True}
        assert body["max_tokens"] == 64
        if len(requests) == 2:
            return _sse(
                [{"choices": [{"delta": {"content": "done"}, "finish_reason": "stop"}]}]
            )
        assert body["tools"][0]["function"]["name"] == "Echo"
        return _sse(
            [
                {"choices": [{"delta": {"content": "Let me "}}]},
                {"choices": [{"delta": {"content": "check."}}]},
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "gateway-call",
                                        "function": {
                                            "name": "Echo",
                                            "arguments": '{"text":',
                                        },
                                    }
                                ]
                            }
                        }
                    ]
                },
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {"index": 0, "function": {"arguments": '"hello"}'}}
                                ]
                            },
                            "finish_reason": "tool_calls",
                        }
                    ]
                },
                {
                    "choices": [],
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 5,
                        "prompt_tokens_details": {"cached_tokens": 20},
                    },
                },
            ],
        )

    model = provider.model()
    model._client = httpx2.AsyncClient(transport=httpx2.MockTransport(handle))
    client = model._client
    assert client is not None
    events: list[RuntimeEvent] = []
    message = UserMessage(text="Echo hello.")
    try:
        response = await model.stream(
            ModelRequest(messages=[message], tools=[_echo], max_response_tokens=64),
            events.append,
        )
        assert [
            event.text for event in events if isinstance(event, ModelResponsePartial)
        ] == ["Let me ", "check."]
        assert response.message.text == "Let me check."
        call = response.message.tool_calls[0]
        assert call.name == "Echo"
        assert call.args == {"text": "hello"}
        assert response.tokens == TokenCount(request=80, response=5, cache_read=20)
        assert response.total_cost == pytest.approx(95 / 1_000_000)
        followup = await model.buffer(
            ModelRequest(
                messages=[
                    message,
                    response.message,
                    ToolResult(call_id=call.id, content="hello"),
                ],
                max_response_tokens=64,
            ),
        )
        assert followup.message.text == "done"
        # A missing usage block is not fabricated from text length.
        assert followup.tokens == TokenCount()
        history = requests[1]["messages"]
        assert isinstance(history, list)
        assistant, result = history[-2:]
        assert isinstance(assistant, dict)
        assert isinstance(result, dict)
        calls = assistant["tool_calls"]
        assert isinstance(calls, list)
        wire_call = calls[0]
        assert isinstance(wire_call, dict)
        assert result["tool_call_id"] == wire_call["id"]
        assert result["content"] == "hello"
    finally:
        await model.close()
    assert client.is_closed


@pytest.mark.parametrize(
    ("status", "error"),
    [(401, "invalid_api_key"), (404, "model_not_found"), (400, "invalid_request")],
)
@pytest.mark.asyncio
async def test_gateway_errors_propagate(
    provider: A2Agent, status: int, error: str
) -> None:
    def handle(request: httpx2.Request) -> httpx2.Response:
        del request
        return httpx2.Response(status, json={"error": {"message": error}})

    model = provider.model()
    model._client = httpx2.AsyncClient(transport=httpx2.MockTransport(handle))
    try:
        with pytest.raises(httpx2.HTTPStatusError, match=error) as caught:
            await model.buffer(ModelRequest(messages=[UserMessage(text="test")]))
        assert caught.value.response.status_code == status
    finally:
        await model.close()


@pytest.mark.asyncio
async def test_truncated_stream_preserves_partial_response(provider: A2Agent) -> None:
    def handle(request: httpx2.Request) -> httpx2.Response:
        del request
        return _sse([{"choices": [{"delta": {"content": "partial"}}]}], done=False)

    model = provider.model()
    model._client = httpx2.AsyncClient(transport=httpx2.MockTransport(handle))
    try:
        with pytest.raises(StreamInterruptedError) as caught:
            await model.buffer(ModelRequest(messages=[UserMessage(text="test")]))
        assert caught.value.response.message.text == "partial"
    finally:
        await model.close()


@pytest.mark.asyncio
async def test_cancellation_closes_stream(provider: A2Agent) -> None:
    started = asyncio.Event()
    closed = asyncio.Event()

    class WaitingStream(httpx2.AsyncByteStream):
        @override
        async def __aiter__(self) -> AsyncIterator[bytes]:
            yield b'data: {"choices": [{"delta": {"content": "partial"}}]}\n\n'
            started.set()
            await asyncio.Event().wait()

        @override
        async def aclose(self) -> None:
            closed.set()

    def handle(request: httpx2.Request) -> httpx2.Response:
        del request
        return httpx2.Response(200, stream=WaitingStream())

    model = provider.model()
    model._client = httpx2.AsyncClient(transport=httpx2.MockTransport(handle))
    task = asyncio.create_task(
        model.stream(ModelRequest(messages=[UserMessage(text="test")]))
    )
    try:
        await asyncio.wait_for(started.wait(), timeout=5)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await model.close()
    assert closed.is_set()
