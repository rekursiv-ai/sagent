"""Tests for Mistral's Chat endpoint and native thinking blocks."""

from dataclasses import replace

import pytest

from sagent.model.types import Request, TextChunk
from sagent.providers.mistral.api import Mistral
from sagent.session.types import AssistantMessage, Thinking


@pytest.mark.asyncio
async def test_endpoint_auth_and_catalog() -> None:
    provider = Mistral(api_key="key")
    assert await provider.endpoint() == "https://api.mistral.ai/v1"
    assert await provider.credentials() == {"authorization": "Bearer key"}
    assert provider.name == "mistral"
    model = provider["utility"]
    assert model.capability.wire_model_id == "mistral-small-2603"
    assert model.capability.limits.max_response_tokens == 0
    assert model.capability.limits.truncates_at_context is False
    model.settings = replace(model.settings, thinking_effort="high")
    body = provider.encode(Request(max_response_tokens=123), model=model)
    assert body["max_tokens"] == 123
    assert body["reasoning_effort"] == "high"
    assert "stream_options" not in body
    await provider.aclose()


def test_thinking_blocks_are_read_and_replayed() -> None:
    provider = Mistral(api_key="key")
    model = provider["utility"]
    reader = provider.reader(model=model)
    thinking = {"type": "thinking", "thinking": [{"type": "text", "text": "hmm"}]}
    reader.feed({"choices": [{"delta": {"content": [thinking]}}]})
    reader.feed(
        {
            "choices": [
                {
                    "delta": {"content": [{"type": "text", "text": "yes"}]},
                    "finish_reason": "stop",
                },
            ],
        },
    )
    response = reader.response()
    assert isinstance(response.records[0], Thinking)
    assert response.records[0].content == "hmm"
    assert AssistantMessage(content="yes") in response.records
    body = provider.encode(Request(records=response.records), model=model)
    assert body["messages"] == [
        {"role": "assistant", "content": [thinking, {"type": "text", "text": "yes"}]},
    ]


def test_mixed_thinking_and_text_streams_both_chunks() -> None:
    provider = Mistral(api_key="key")
    model = provider["default"]
    reader = provider.reader(model=model)
    thought = {"type": "thinking", "thinking": [{"type": "text", "text": "Plan"}]}
    assert reader.feed(
        {
            "choices": [
                {
                    "delta": {"content": [thought, {"type": "text", "text": "Answer"}]},
                    "finish_reason": "stop",
                },
            ],
        },
    ) == (
        TextChunk(text="Plan", thinking=True),
        TextChunk(text="Answer", record=1),
    )
    response = reader.response()
    assert len(response.records) == 2
    assert isinstance(response.records[0], Thinking)
    assert response.records[0].content == "Plan"
    assert response.records[1] == AssistantMessage(content="Answer")
    replay = provider.encode(Request(records=response.records), model=model)
    assert replay["messages"] == [
        {
            "role": "assistant",
            "content": [thought, {"type": "text", "text": "Answer"}],
        },
    ]


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
