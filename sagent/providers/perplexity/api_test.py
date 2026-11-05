"""Tests for Perplexity Orchestrator's Responses schema subset."""

import pytest

from sagent.model.types import Request
from sagent.providers.perplexity.api import Perplexity
from sagent.session.types import UserMessage
from sagent.tools.read import Read


@pytest.mark.asyncio
async def test_endpoint_auth_and_local_count() -> None:
    provider = Perplexity(api_key="key")
    assert await provider.endpoint() == "https://api.perplexity.ai/orchestrator/v1"
    assert await provider.credentials() == {"authorization": "Bearer key"}
    assert provider.name == "perplexity"
    assert provider.path == "/responses"
    model = provider["default"]
    assert model.capability.wire_model_id == "perplexity/glm-5.3"
    assert model.capability.limits.max_response_tokens == 0
    assert model.capability.limits.truncates_at_context is None
    assert await model.count(Request(records=(UserMessage(content="hello world"),))) > 0
    await provider.aclose()


def test_supported_function_schema_and_tool_filter() -> None:
    provider = Perplexity(api_key="key")
    model = provider["default"]
    tool = Read()
    request = Request(
        tools=(tool,),
        allowed=frozenset({"Read"}),
        max_response_tokens=123,
    )
    body = provider.encode(request, model=model)
    assert body["store"] is False
    assert body["max_output_tokens"] == 123
    assert body["tools"] == [
        {
            "type": "function",
            "name": tool.name,
            "description": tool.description,
            "parameters": dict(tool.schema),
        },
    ]
    assert "tool_choice" not in body
    denied = provider.encode(Request(tools=(tool,), allowed=frozenset()), model=model)
    assert "tools" not in denied
    assert denied["tool_choice"] == "none"


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
