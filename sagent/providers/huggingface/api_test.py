"""Exercise Hugging Face's router endpoint, token, and pinned Groq route."""

from __future__ import annotations

import pytest

from sagent.model.capability import ModelCapability, ModelSettings
from sagent.model.exceptions import AuthError
from sagent.model.types import Request
from sagent.providers.huggingface.api import HuggingFace
from sagent.providers.openai.compat import OpenAIModel


@pytest.mark.asyncio
async def test_endpoint_and_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = HuggingFace(name="huggingface")
    assert await provider.endpoint() == "https://router.huggingface.co/v1"
    monkeypatch.setenv("HF_TOKEN", "secret")
    assert await provider.credentials() == {"authorization": "Bearer secret"}
    monkeypatch.delenv("HF_TOKEN")
    with pytest.raises(AuthError, match="HF_TOKEN"):
        await provider.credentials()
    explicit = HuggingFace(api_key="explicit", base_url="http://box/v1")
    assert await explicit.endpoint() == "http://box/v1"
    assert await explicit.credentials() == {"authorization": "Bearer explicit"}


def test_the_wire_id_pins_the_priced_provider() -> None:
    provider = HuggingFace(name="huggingface")
    capability = ModelCapability(
        wire_model_id="openai/gpt-oss-120b:groq",
        thinking_effort=frozenset({"low", "medium", "high"}),
    )
    model = OpenAIModel(
        provider=provider,
        capability=capability,
        settings=ModelSettings(thinking_effort="medium"),
    )
    body = provider.encode(Request(max_response_tokens=17), model=model)
    assert body["model"] == "openai/gpt-oss-120b:groq"
    assert body["max_tokens"] == 17
    assert body["reasoning_effort"] == "medium"
    assert provider.reasoning_field == "reasoning"


def test_catalog_pins_groq_and_its_limits() -> None:
    provider = HuggingFace()
    model = provider["default"]
    assert provider.name == "huggingface"
    assert model.capability.wire_model_id == "openai/gpt-oss-120b:groq"
    assert model.capability.limits.max_request_tokens == 131_072
    assert model.capability.limits.max_response_tokens == 65_536
    assert model.settings.thinking_effort == "medium"
    assert model.capability.thinking_output == frozenset({"text"})
    assert model.capability.service_tier == frozenset({"auto"})
    assert model.capability.manage_context_server_side == frozenset({False})
    assert (
        provider.encode(Request(), model=model)["model"] == "openai/gpt-oss-120b:groq"
    )


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
