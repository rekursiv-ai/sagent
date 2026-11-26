"""Exercise DeepInfra's endpoint, credentials, and Nemotron token limits."""

from __future__ import annotations

import pytest

from sagent.model.capability import ModelCapability, ModelSettings
from sagent.model.exceptions import AuthError
from sagent.model.types import Request
from sagent.providers.deepinfra.api import DeepInfra
from sagent.providers.openai.compat import OpenAIModel


@pytest.mark.asyncio
async def test_endpoint_and_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = DeepInfra(name="deepinfra")
    assert await provider.endpoint() == "https://api.deepinfra.com/v1/openai"
    monkeypatch.setenv("DEEPINFRA_API_KEY", "secret")
    assert await provider.credentials() == {"authorization": "Bearer secret"}
    monkeypatch.delenv("DEEPINFRA_API_KEY")
    with pytest.raises(AuthError, match="DEEPINFRA_API_KEY"):
        await provider.credentials()
    explicit = DeepInfra(api_key="explicit", base_url="http://box/v1")
    assert await explicit.endpoint() == "http://box/v1"
    assert await explicit.credentials() == {"authorization": "Bearer explicit"}


def test_nemotron_uses_max_tokens_and_separate_reasoning() -> None:
    provider = DeepInfra(name="deepinfra")
    model = OpenAIModel(
        provider=provider,
        capability=ModelCapability(
            wire_model_id="nvidia/NVIDIA-Nemotron-3-Super-120B-A12B",
        ),
        settings=ModelSettings(),
    )
    body = provider.encode(Request(max_response_tokens=17), model=model)
    assert body["max_tokens"] == 17
    assert "reasoning_effort" not in body
    assert provider.reasoning_field == "reasoning_content"


def test_catalog_does_not_offer_unverified_nemotron_knobs() -> None:
    provider = DeepInfra()
    model = provider["default"]
    assert provider.name == "deepinfra"
    assert model.capability.wire_model_id == "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B"
    assert model.capability.limits.max_request_tokens == 262_144
    assert model.capability.limits.max_response_tokens == 0
    assert model.capability.limits.truncates_at_context is False
    assert model.capability.thinking_effort == frozenset({"none"})
    assert model.capability.thinking_output == frozenset({"text"})
    assert model.capability.service_tier == frozenset({"auto"})
    assert model.capability.manage_context_server_side == frozenset({False})
    assert "max_tokens" not in provider.encode(Request(), model=model)
    with pytest.raises(ValueError, match="thinking_effort"):
        model.settings = model.settings.with_value("thinking_effort", "high")


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
