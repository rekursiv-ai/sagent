"""Tests for xAI's stateless Responses provider."""

from dataclasses import replace

import pytest

from sagent.model.types import Request
from sagent.providers.xai.api import XAI
from sagent.session.types import Thinking, UserMessage


@pytest.mark.asyncio
async def test_endpoint_auth_and_local_count() -> None:
    provider = XAI(api_key="key")
    assert await provider.endpoint() == "https://api.x.ai/v1"
    assert await provider.credentials() == {"authorization": "Bearer key"}
    assert provider.name == "xai"
    assert provider.path == "/responses"
    model = provider["default"]
    assert model.capability.wire_model_id == "grok-4.7"
    assert model.capability.limits.max_response_tokens == 0
    assert model.capability.limits.truncates_at_context is None
    assert await model.count(Request(records=(UserMessage(content="hello world"),))) > 0
    await provider.aclose()


def test_reasoning_and_sealed_replay() -> None:
    provider = XAI(api_key="key")
    model = provider["default"]
    model.settings = replace(model.settings, thinking_effort="xhigh")
    body = provider.encode(Request(max_response_tokens=123), model=model)
    assert body["reasoning"] == {"effort": "xhigh"}
    assert body["include"] == ["reasoning.encrypted_content"]
    assert body["store"] is False
    assert body["max_output_tokens"] == 123
    assert "temperature" not in body
    reader = provider.reader(model=model)
    reader.feed(
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                "type": "reasoning",
                "id": "rs_1",
                "summary": [],
                "encrypted_content": "sealed",
            },
        },
    )
    reader.feed({"type": "response.completed", "response": {"status": "completed"}})
    response = reader.response()
    assert isinstance(response.records[0], Thinking)
    replay = provider.encode(Request(records=response.records), model=model)
    assert replay["input"] == [
        {
            "type": "reasoning",
            "id": "rs_1",
            "summary": [],
            "encrypted_content": "sealed",
        },
    ]


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
