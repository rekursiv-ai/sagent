"""Tests for Azure's deployment-specific Responses endpoint."""

from dataclasses import replace
from functools import partial

from treekle import json_dumps, json_loads

import httpx2
import pytest

from sagent.model.types import Request, Response, TextChunk
from sagent.providers.azure.api import Azure
from sagent.providers.catalog import ModelCatalog
from sagent.providers.openai.api import OpenAI
from sagent.providers.table import ModelTable
from sagent.session.types import AssistantMessage


@pytest.mark.asyncio
async def test_azure_uses_api_key_and_caller_deployment() -> None:
    row = replace(OpenAI.models()["default"], wire_model_id="my-deployment")
    table = ModelTable(rows=(row,), roles={"default": row.model_id})
    provider = Azure(
        api_key="test",
        base_url="https://example.openai.azure.com/openai/v1",
        models=table,
    )
    assert await provider.credentials() == {"api-key": "test"}
    assert provider.path == "/responses"
    model = provider["default"]
    assert provider.encode(Request(), model=model)["model"] == "my-deployment"
    assert await model.count(Request()) >= 0


def test_azure_requires_a_priced_deployment_table() -> None:
    with pytest.raises(ValueError, match="models=ModelTable"):
        Azure()["default"]


@pytest.mark.asyncio
async def test_catalog_streams_a_caller_priced_deployment() -> None:
    row = replace(OpenAI.models()["default"], wire_model_id="my-deployment")
    table = ModelTable(rows=(row,), roles={"default": row.model_id})
    seen: list[httpx2.Request] = []
    events = [
        {"type": "response.output_text.delta", "delta": "hello"},
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {"type": "message", "content": [{"text": "hello"}]},
        },
        {
            "type": "response.completed",
            "response": {
                "status": "completed",
                "usage": {"input_tokens": 2, "output_tokens": 1},
            },
        },
    ]

    def answer(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return httpx2.Response(
            200,
            text="".join(f"data: {json_dumps(event)}\n\n" for event in events),
            headers={"content-type": "text/event-stream"},
        )

    async with ModelCatalog(
        factories={
            "azure": partial(
                Azure,
                models=table,
                api_key="test",
                base_url="https://example.openai.azure.com/openai/v1",
            ),
        },
    ) as catalog:
        catalog._http = httpx2.AsyncClient(transport=httpx2.MockTransport(answer))
        model = catalog.model("azure/default", current="azure")
        chunks = [event async for event in model.stream(Request())]
    assert chunks[0] == TextChunk(text="hello")
    response = chunks[-1]
    assert isinstance(response, Response)
    assert response.records == (AssistantMessage(content="hello"),)
    assert response.tokens.request == 2
    assert response.tokens.response == 1
    assert len(seen) == 1
    assert seen[0].url.path == "/openai/v1/responses"
    assert seen[0].headers["api-key"] == "test"
    assert json_loads(seen[0].content, dict[str, object])["model"] == "my-deployment"


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
