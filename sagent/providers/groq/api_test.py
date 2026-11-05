"""Exercise Groq's endpoint, auth, and GPT-OSS request dialect."""

from __future__ import annotations

import pytest

from sagent.model.capability import ModelCapability, ModelSettings
from sagent.model.exceptions import AuthError
from sagent.model.types import Request
from sagent.providers.groq.api import Groq
from sagent.providers.openai.compat import OpenAIModel


@pytest.mark.asyncio
async def test_endpoint_and_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = Groq(name="groq")
    assert await provider.endpoint() == "https://api.groq.com/openai/v1"
    monkeypatch.setenv("GROQ_API_KEY", "secret")
    assert await provider.credentials() == {"authorization": "Bearer secret"}
    monkeypatch.delenv("GROQ_API_KEY")
    with pytest.raises(AuthError, match="GROQ_API_KEY"):
        await provider.credentials()
    explicit = Groq(api_key="explicit", base_url="http://box/v1")
    assert await explicit.endpoint() == "http://box/v1"
    assert await explicit.credentials() == {"authorization": "Bearer explicit"}


def test_reasoning_is_separate_from_the_answer() -> None:
    provider = Groq(name="groq")
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
    assert body["max_completion_tokens"] == 17
    assert body["reasoning_effort"] == "medium"
    assert body["include_reasoning"] is True
    assert provider.reasoning_field == "reasoning"


def test_catalog_defaults_and_usage() -> None:
    provider = Groq()
    model = provider["default"]
    assert provider.name == "groq"
    assert model.capability.wire_model_id == "openai/gpt-oss-120b"
    assert model.capability.limits.max_request_tokens == 131_072
    assert model.capability.limits.max_response_tokens == 65_536
    assert model.settings.thinking_effort == "medium"
    assert model.capability.thinking_output == frozenset({"text"})
    assert model.capability.service_tier == frozenset({"auto"})
    assert model.capability.manage_context_server_side == frozenset({False})
    reader = provider.reader(model=model)
    reader.feed({"choices": [{"delta": {"reasoning": "Think"}}]})
    reader.feed(
        {"choices": [{"delta": {"content": "Answer"}, "finish_reason": "stop"}]},
    )
    reader.feed({"x_groq": {"usage": {"prompt_tokens": 10, "completion_tokens": 2}}})
    response = reader.response()
    assert response.tokens.request == 10
    assert response.tokens.response == 2
    assert response.spend.request == pytest.approx(0.15 * 10 / 1_000_000)


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
