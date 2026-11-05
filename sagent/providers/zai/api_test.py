"""Exercise Z.ai's endpoint and always-on GLM reasoning."""

from __future__ import annotations

import pytest

from sagent.model.capability import ModelCapability, ModelSettings
from sagent.model.exceptions import AuthError
from sagent.model.types import Request
from sagent.providers.openai.compat import OpenAIModel
from sagent.providers.zai.api import ZAI


@pytest.mark.asyncio
async def test_endpoint_and_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = ZAI(name="zai")
    assert await provider.endpoint() == "https://api.z.ai/api/paas/v4"
    monkeypatch.setenv("ZAI_API_KEY", "secret")
    assert await provider.credentials() == {"authorization": "Bearer secret"}
    monkeypatch.delenv("ZAI_API_KEY")
    with pytest.raises(AuthError, match="ZAI_API_KEY"):
        await provider.credentials()
    explicit = ZAI(api_key="explicit", base_url="http://box/v1")
    assert await explicit.endpoint() == "http://box/v1"
    assert await explicit.credentials() == {"authorization": "Bearer explicit"}


@pytest.mark.parametrize("effort", ["low", "high", "max"])
def test_effort_is_not_folded_to_openai_levels(effort: str) -> None:
    provider = ZAI(name="zai")
    settings = ModelSettings().with_value("thinking_effort", effort)
    capability = ModelCapability(
        model_id="glm-5.3",
        wire_model_id="glm-5.3",
        thinking_effort=frozenset({"low", "high", "max"}),
    )
    model = OpenAIModel(provider=provider, capability=capability, settings=settings)
    body = provider.encode(Request(max_response_tokens=17), model=model)
    assert body["max_tokens"] == 17
    assert "max_completion_tokens" not in body
    assert body["reasoning_effort"] == effort
    assert body["thinking"] == {"type": "enabled", "clear_thinking": False}
    assert body["tool_stream"] is True


def test_catalog_always_thinks_and_defaults_to_max() -> None:
    provider = ZAI()
    model = provider["default"]
    assert provider.name == "zai"
    assert model.capability.wire_model_id == "glm-5.3"
    assert model.capability.limits.max_request_tokens == 1_048_576
    assert model.capability.limits.max_response_tokens == 131_072
    assert model.capability.limits.truncates_at_context is False
    assert model.settings.thinking_effort == "max"
    assert model.capability.thinking_output == frozenset({"text"})
    assert model.capability.service_tier == frozenset({"auto"})
    assert model.capability.manage_context_server_side == frozenset({False})
    with pytest.raises(ValueError, match="thinking_effort"):
        model.settings = model.settings.with_value("thinking_effort", "none")
    assert provider.encode(Request(), model=model)["reasoning_effort"] == "max"


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
