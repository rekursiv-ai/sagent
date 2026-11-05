"""Tests for Bedrock's bearer-key Chat endpoint."""

from dataclasses import replace
from functools import partial

from treekle import json_dumps, json_loads

import httpx2
import pytest

from sagent.model.exceptions import AuthError
from sagent.model.types import Request, Response
from sagent.providers.bedrock.api import Bedrock
from sagent.providers.catalog import ModelCatalog
from sagent.providers.ollama.api import Ollama
from sagent.providers.table import ModelTable
from sagent.session.types import AssistantMessage


@pytest.mark.asyncio
async def test_bedrock_uses_bearer_keys_and_regional_chat() -> None:
    row = replace(Ollama.models()["default"], wire_model_id="openai.gpt-oss-120b-1:0")
    provider = Bedrock(
        api_key="test",
        base_url="https://bedrock-runtime.us-west-2.amazonaws.com/openai/v1",
        models=ModelTable(rows=(row,), roles={"default": row.model_id}),
    )
    assert await provider.credentials() == {"authorization": "Bearer test"}
    assert (await provider.endpoint()).endswith("/openai/v1")
    model = provider["default"]
    body = provider.encode(Request(max_response_tokens=32), model=model)
    assert body["max_completion_tokens"] == 32
    assert body["model"] == model.capability.wire_model_id


@pytest.mark.asyncio
async def test_bedrock_derives_region_and_rejects_missing_region(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AWS_REGION", "eu-west-1")
    monkeypatch.delenv("BEDROCK_BASE_URL", raising=False)
    assert (
        await Bedrock().endpoint()
        == "https://bedrock-runtime.eu-west-1.amazonaws.com/openai/v1"
    )
    monkeypatch.delenv("AWS_REGION")
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
    with pytest.raises(AuthError, match="AWS_REGION"):
        await Bedrock().endpoint()


def test_bedrock_requires_regional_rates() -> None:
    with pytest.raises(ValueError, match="models=ModelTable"):
        Bedrock()["default"]


@pytest.mark.asyncio
async def test_catalog_streams_a_caller_priced_regional_model() -> None:
    row = replace(Ollama.models()["default"], wire_model_id="openai.gpt-oss-120b-1:0")
    table = ModelTable(rows=(row,), roles={"default": row.model_id})
    seen: list[httpx2.Request] = []

    def answer(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        event = {
            "choices": [{"delta": {"content": "hello"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1},
        }
        return httpx2.Response(
            200,
            text=f"data: {json_dumps(event)}\n\ndata: [DONE]\n\n",
            headers={"content-type": "text/event-stream"},
        )

    async with ModelCatalog(
        factories={
            "bedrock": partial(
                Bedrock,
                models=table,
                api_key="test",
                base_url="https://bedrock-runtime.us-west-2.amazonaws.com/openai/v1",
            ),
        },
    ) as catalog:
        catalog._http = httpx2.AsyncClient(transport=httpx2.MockTransport(answer))
        model = catalog.model("bedrock/default", current="bedrock")
        chunks = [
            event async for event in model.stream(Request(max_response_tokens=32))
        ]
    response = chunks[-1]
    assert isinstance(response, Response)
    assert response.records == (AssistantMessage(content="hello"),)
    assert response.tokens.request == 2
    assert response.tokens.response == 1
    assert len(seen) == 1
    assert seen[0].url.path == "/openai/v1/chat/completions"
    assert seen[0].headers["authorization"] == "Bearer test"
    body = json_loads(seen[0].content, dict[str, object])
    assert body["model"] == "openai.gpt-oss-120b-1:0"
    assert body["max_completion_tokens"] == 32


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
