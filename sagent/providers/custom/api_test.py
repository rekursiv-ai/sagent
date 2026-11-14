"""Exercise the Custom OpenAI-compatible provider."""

from __future__ import annotations

import httpx2
import pytest

from sagent.model.exceptions import AuthError, UnknownModelError
from sagent.model.types import Request, Response, TextChunk
from sagent.providers.catalog import ModelCatalog
from sagent.providers.custom.api import (
    Custom,
    read_custom_config,
)
from sagent.providers.openai.compat import OpenAIModel
from sagent.providers.resources import SharedResources
from sagent.session.types import AssistantMessage, UserMessage


@pytest.fixture(autouse=True)
def clean_custom_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep Custom's process-level configuration out of every test."""
    for variable in (
        "CUSTOM_BASE_URL",
        "CUSTOM_API_KEY",
        "CUSTOM_MODEL",
        "CUSTOM_CONTEXT_WINDOW",
    ):
        monkeypatch.delenv(variable, raising=False)


@pytest.mark.asyncio
async def test_endpoint_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = Custom()
    with pytest.raises(AuthError, match="CUSTOM_BASE_URL"):
        await provider.endpoint()

    monkeypatch.setenv("OPENAI_BASE_URL", "https://unrelated-openai.example/v1")
    with pytest.raises(AuthError, match="CUSTOM_BASE_URL"):
        await provider.endpoint()

    monkeypatch.setenv("CUSTOM_BASE_URL", "https://api.experientiallabs.ai/v1")
    assert await provider.endpoint() == "https://api.experientiallabs.ai/v1"

    explicit = Custom(base_url="https://override.example.com/v1")
    assert await explicit.endpoint() == "https://override.example.com/v1"


def test_custom_config_persistence() -> None:
    assert read_custom_config() == {}

    provider = Custom()
    provider.configure(
        base_url="https://api.experientiallabs.ai/v1",
        api_key="secret-key-123",
        model="claude-fable-5.2",
    )
    saved = read_custom_config()
    assert saved["base_url"] == "https://api.experientiallabs.ai/v1"
    assert saved["api_key"] == "secret-key-123"
    assert saved["model"] == "claude-fable-5.2"


@pytest.mark.asyncio
async def test_credentials_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = Custom()
    assert await provider.credentials() == {}

    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-openai-secret")
    assert await provider.credentials() == {}

    monkeypatch.setenv("CUSTOM_API_KEY", "custom-secret")
    assert await provider.credentials() == {"authorization": "Bearer custom-secret"}

    explicit = Custom(api_key="explicit-secret")
    assert await explicit.credentials() == {"authorization": "Bearer explicit-secret"}


def test_exact_model_id_passthrough() -> None:
    provider = Custom()
    model = provider["claude-fable-5.2"]
    assert isinstance(model, OpenAIModel)
    assert model.capability.model_id == "claude-fable-5.2"
    assert model.capability.wire_model_id == "claude-fable-5.2"
    assert model.capability.limits.max_request_tokens == 0
    assert model.capability.limits.max_response_tokens == 0

    body = provider.encode(Request(), model=model)
    assert body["model"] == "claude-fable-5.2"
    assert "temperature" not in body
    assert "max_tokens" not in body
    assert "max_completion_tokens" not in body


def test_model_with_context_tag() -> None:
    provider = Custom()
    model = provider["claude-fable-5.2+200k"]
    assert model.capability.model_id == "claude-fable-5.2"
    assert model.capability.wire_model_id == "claude-fable-5.2"
    assert model.capability.limits.max_request_tokens == 200_000

    body = provider.encode(Request(), model=model)
    assert body["model"] == "claude-fable-5.2"


def test_configured_model_resolves_roles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = Custom()
    for role in ("default", "best", "everyday", "utility"):
        with pytest.raises(UnknownModelError, match="requires a model"):
            provider[role]

    monkeypatch.setenv("CUSTOM_MODEL", "my-gateway-model")
    for role in ("default", "best", "everyday", "utility"):
        assert provider[role].capability.wire_model_id == "my-gateway-model"
    assert tuple(provider) == ("my-gateway-model",)

    assert provider["explicit-model"].capability.wire_model_id == "explicit-model"
    provider.configure(base_url="https://gateway.example/v1", model="saved-model")
    assert provider["default"].capability.wire_model_id == "saved-model"
    assert tuple(provider) == ("saved-model",)

    monkeypatch.delenv("CUSTOM_MODEL")
    provider.configure(base_url="https://gateway.example/v1", model="")
    with pytest.raises(UnknownModelError, match="requires a model"):
        provider["default"]
    assert tuple(provider) == ()


def test_model_catalog_integration() -> None:
    catalog = ModelCatalog()
    assert "custom" in catalog.factories
    provider = catalog["custom"]
    assert isinstance(provider, Custom)
    assert catalog.name("custom-api") == "custom"
    assert catalog.name("proxy") == "custom"

    model = catalog.model("custom/claude-fable-5.2", current="anthropic")
    assert model.capability.wire_model_id == "claude-fable-5.2"


@pytest.mark.asyncio
async def test_stream_end_to_end(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUSTOM_BASE_URL", "https://api.experientiallabs.ai/v1")
    monkeypatch.setenv("CUSTOM_API_KEY", "explabs-test-key")

    sse_data = (
        b'data: {"choices": [{"delta": {"content": "Hello from "}, "finish_reason": null}]}\n\n'
        b'data: {"choices": [{"delta": {"content": "Experiential!"}, "finish_reason": null}]}\n\n'
        b'data: {"choices": [{"delta": {}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 12, "completion_tokens": 5}}\n\n'
        b"data: [DONE]\n\n"
    )

    requests: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(200, content=sse_data)

    class MockResources(SharedResources):
        def __init__(self) -> None:
            super().__init__()
            self._http = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))

    async with MockResources() as resources:
        provider = Custom(resources=resources)
        model = provider["claude-fable-5.2"]
        events = [
            chunk
            async for chunk in model.stream(
                Request(records=(UserMessage(content="Hi"),)),
            )
        ]

    assert len(requests) == 1
    assert str(requests[0].url) == "https://api.experientiallabs.ai/v1/chat/completions"
    assert requests[0].headers["authorization"] == "Bearer explabs-test-key"

    chunks = [e.text for e in events if isinstance(e, TextChunk)]
    assert "".join(chunks) == "Hello from Experiential!"

    response = events[-1]
    assert isinstance(response, Response)
    assert response.records == (AssistantMessage(content="Hello from Experiential!"),)
    assert (response.tokens.request, response.tokens.response) == (12, 5)


def test_custom_reader_accepts_common_reasoning_aliases() -> None:
    provider = Custom()
    model = provider["claude-fable-5.2"]
    reader = provider.reader(model=model)

    assert reader.feed(
        {"choices": [{"delta": {"reasoning": "First, inspect the request."}}]}
    ) == TextChunk(text="First, inspect the request.", thinking=True)
    assert reader.feed(
        {
            "choices": [
                {
                    "delta": {
                        "reasoning_details": [{"type": "reasoning", "text": " Done."}]
                    }
                }
            ]
        }
    ) == TextChunk(text=" Done.", thinking=True)
    reader.feed(
        {"choices": [{"delta": {"content": "Answer."}, "finish_reason": "stop"}]}
    )
    response = reader.response()
    assert response.records[0].content == "First, inspect the request. Done."


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
