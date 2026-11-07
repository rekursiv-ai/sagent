"""Tests for OpenRouter's identity and reasoning configuration."""

from dataclasses import replace

import pytest

from sagent.model.types import Request
from sagent.providers.openai.compat import OpenAIModel
from sagent.providers.openrouter.api import OpenRouter
from sagent.session.types import AssistantMessage, Thinking, UserMessage


@pytest.mark.asyncio
async def test_endpoint_auth_and_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = OpenRouter(api_key="key")
    assert await provider.endpoint() == "https://openrouter.ai/api/v1"
    assert await provider.credentials() == {"authorization": "Bearer key"}
    assert provider.name == "openrouter"
    monkeypatch.setenv("OPENROUTER_BASE_URL", "http://box/v1")
    assert await OpenRouter().endpoint() == "http://box/v1"
    model = provider["default"]
    assert model.capability.wire_model_id == "mistralai/mistral-small-2603"
    assert model.capability.limits.truncates_at_context is False
    body = provider.encode(Request(max_response_tokens=123), model=model)
    assert body["reasoning"] == {"effort": "none"}
    assert body["max_tokens"] == 123
    assert "reasoning_effort" not in body
    assert await model.count(Request(records=(UserMessage(content="hello"),))) > 0
    await provider.aclose()


def test_a_nonreasoning_model_sends_no_reasoning_configuration() -> None:
    provider = OpenRouter(api_key="key")
    capability = replace(
        provider["default"].capability,
        thinking_effort=frozenset({"none"}),
    )
    model = OpenAIModel(
        provider=provider,
        capability=capability,
        settings=capability.default,
    )

    body = provider.encode(Request(), model=model)

    assert "reasoning" not in body
    assert "reasoning_effort" not in body


def test_reasoning_details_round_trip() -> None:
    provider = OpenRouter(api_key="key")
    model = provider["default"]
    reader = provider.reader(model=model)
    details = [{"type": "reasoning.encrypted", "data": "sealed", "index": 0}]
    reader.feed({"choices": [{"delta": {"reasoning_details": details}}]})
    reader.feed(
        {"choices": [{"delta": {"content": "answer"}, "finish_reason": "stop"}]},
    )
    response = reader.response()
    assert any(isinstance(record, Thinking) for record in response.records)
    assert AssistantMessage(content="answer") in response.records
    body = provider.encode(Request(records=response.records), model=model)
    assert body["messages"] == [
        {"role": "assistant", "content": "answer", "reasoning_details": details},
    ]


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
