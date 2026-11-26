"""Exercise DeepSeek's endpoint, credentials, and native reasoning controls."""

from __future__ import annotations

import pytest

from sagent.model.capability import ModelCapability, ModelSettings
from sagent.model.exceptions import AuthError
from sagent.model.types import Request
from sagent.providers.deepseek.api import DeepSeek
from sagent.providers.openai.compat import OpenAIModel


@pytest.mark.asyncio
async def test_endpoint_and_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = DeepSeek(name="deepseek")
    assert await provider.endpoint() == "https://api.deepseek.com"
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")
    assert await provider.credentials() == {"authorization": "Bearer secret"}
    monkeypatch.delenv("DEEPSEEK_API_KEY")
    with pytest.raises(AuthError, match="DEEPSEEK_API_KEY"):
        await provider.credentials()
    explicit = DeepSeek(api_key="explicit", base_url="http://box/v1")
    assert await explicit.endpoint() == "http://box/v1"
    assert await explicit.credentials() == {"authorization": "Bearer explicit"}


@pytest.mark.parametrize("effort", ["none", "low", "high", "max"])
def test_native_effort_and_token_limit(effort: str) -> None:
    provider = DeepSeek(name="deepseek")
    settings = ModelSettings().with_value("thinking_effort", effort)
    capability = ModelCapability(
        model_id="deepseek-flash-4.1",
        wire_model_id="deepseek-flash",
        thinking_effort=frozenset({"none", "low", "high", "max"}),
    )
    model = OpenAIModel(provider=provider, capability=capability, settings=settings)
    body = provider.encode(Request(max_response_tokens=17), model=model)
    assert body["max_tokens"] == 17
    assert "max_completion_tokens" not in body
    assert body["reasoning_effort"] == effort
    assert "temperature" not in body


def test_catalog_defaults_and_limits() -> None:
    provider = DeepSeek()
    assert provider.name == "deepseek"
    assert provider["default"].capability.model_id == "deepseek-flash-4.1"
    assert provider["best"].capability.wire_model_id == "deepseek-v4-pro"
    for name in provider:
        model = provider[name]
        assert model.capability.limits.max_request_tokens == 1_048_576
        assert model.capability.limits.max_response_tokens == 393_216
        assert model.capability.limits.truncates_at_context is None
        assert model.settings.thinking_effort == "high"
        assert model.capability.thinking_output == frozenset({"text"})
        assert model.capability.service_tier == frozenset({"auto"})
        assert model.capability.manage_context_server_side == frozenset({False})
        assert provider.encode(Request(), model=model)["reasoning_effort"] == "high"


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
