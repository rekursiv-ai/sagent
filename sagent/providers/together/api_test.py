"""Exercise Together's endpoint, auth, and GPT-OSS reasoning field."""

from __future__ import annotations

import pytest

from sagent.model.capability import ModelCapability, ModelSettings
from sagent.model.exceptions import AuthError
from sagent.model.types import Request
from sagent.providers.openai.compat import OpenAIModel
from sagent.providers.together.api import Together


@pytest.mark.asyncio
async def test_endpoint_and_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = Together(name="together")
    assert await provider.endpoint() == "https://api.together.ai/v1"
    monkeypatch.setenv("TOGETHER_API_KEY", "secret")
    assert await provider.credentials() == {"authorization": "Bearer secret"}
    monkeypatch.delenv("TOGETHER_API_KEY")
    with pytest.raises(AuthError, match="TOGETHER_API_KEY"):
        await provider.credentials()
    explicit = Together(api_key="explicit", base_url="http://box/v1")
    assert await explicit.endpoint() == "http://box/v1"
    assert await explicit.credentials() == {"authorization": "Bearer explicit"}


def test_gpt_oss_uses_max_tokens_and_reasoning() -> None:
    provider = Together(name="together")
    capability = ModelCapability(
        wire_model_id="openai/gpt-oss-120b",
        thinking_effort=frozenset({"low", "medium", "high"}),
    )
    model = OpenAIModel(
        provider=provider,
        capability=capability,
        settings=ModelSettings(thinking_effort="medium"),
    )
    body = provider.encode(Request(max_response_tokens=17), model=model)
    assert body["max_tokens"] == 17
    assert "max_completion_tokens" not in body
    assert body["reasoning_effort"] == "medium"
    assert provider.reasoning_field == "reasoning"


def test_catalog_defaults_and_unknown_output_limit() -> None:
    provider = Together()
    model = provider["default"]
    assert provider.name == "together"
    assert model.capability.wire_model_id == "openai/gpt-oss-120b"
    assert model.capability.limits.max_request_tokens == 131_072
    assert model.capability.limits.max_response_tokens == 0
    assert model.capability.limits.truncates_at_context is False
    assert model.settings.thinking_effort == "medium"
    assert model.capability.thinking_output == frozenset({"text"})
    assert model.capability.service_tier == frozenset({"auto"})
    assert model.capability.manage_context_server_side == frozenset({False})
    assert "max_tokens" not in provider.encode(Request(), model=model)
    with pytest.raises(ValueError, match="thinking_effort"):
        model.settings = model.settings.with_value("thinking_effort", "none")


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
