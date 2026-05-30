"""Tests for ``providers.openai``: API-key dispatch + effort-model gating."""

from __future__ import annotations

import pytest

from sagent.providers.openai import OpenAI
from sagent.types.model import ModelRequest
from sagent.types.runtime import UserMessage


def test_openai_from_key_constructs() -> None:
    p = OpenAI.from_key("sk-test")
    assert isinstance(p, OpenAI)
    assert p.api_key == "sk-test"
    assert p.base_url == "https://api.openai.com/v1"


def test_openai_from_env_missing_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="API key not configured"):
        OpenAI.from_env()


def test_openai_from_env_reads_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    p = OpenAI.from_env()
    assert p.api_key == "sk-env"


def test_openai_default_model_known() -> None:
    p = OpenAI.from_key("k")
    m = p.model()  # picks DEFAULT_MODEL.
    assert m.model_id == OpenAI.DEFAULT_MODEL


def test_openai_known_model_returns_backend() -> None:
    p = OpenAI.from_key("k")
    m = p.model("gpt-4o")
    assert m.model_id == "gpt-4o"
    assert m.max_request_tokens == 128_000


def test_openai_unknown_model_raises() -> None:
    p = OpenAI.from_key("k")
    with pytest.raises(ValueError, match="Unknown model"):
        _ = p.model("not-a-real-model")


def test_openai_utility_model_default() -> None:
    p = OpenAI.from_key("k")
    m = p.utility_model()
    assert m.model_id == OpenAI.DEFAULT_UTILITY_MODEL


@pytest.mark.parametrize(
    ("model_id", "expected"),
    [
        ("o1-mini", True),
        ("o3-mini", True),
        ("o4-x", True),
        ("gpt-5.5", True),
        ("gpt-5.4-mini", True),
        ("gpt-4o", False),
        ("gpt-4.1", False),
    ],
)
def test_openai_effort_gating(model_id: str, expected: bool) -> None:
    p = OpenAI.from_key("k")
    # The internal ``_OpenAIModel`` subclass is private; reach via ``supports_effort``.
    m = p.model(model_id) if model_id in OpenAI.KNOWN_MODELS else None
    if m is None:
        # Not all hypothetical ids are mapped; check ``_is_effort_model`` directly.
        m = p.model("gpt-4o")
    is_effort = m._is_effort_model(model_id)
    assert is_effort is expected


def test_openai_pricing_attached_to_model() -> None:
    p = OpenAI.from_key("k")
    m = p.model("gpt-5.5")
    assert m.pricing.request > 0
    assert m.pricing.response > 0


def test_openai_valid_service_tiers() -> None:
    p = OpenAI.from_key("k")
    m = p.model("gpt-5.5")
    assert m.valid_service_tiers == ("auto", "default", "flex", "priority")


def test_openai_build_body_emits_service_tier() -> None:
    p = OpenAI.from_key("k")
    m = p.model("gpt-5.5")
    body = m._build_body(
        ModelRequest(messages=[UserMessage(text="x")], service_tier="priority"),
        stream=False,
    )
    assert body["service_tier"] == "priority"


def test_openai_build_body_omits_unset_service_tier() -> None:
    p = OpenAI.from_key("k")
    m = p.model("gpt-5.5")
    body = m._build_body(
        ModelRequest(messages=[UserMessage(text="x")]),
        stream=False,
    )
    assert "service_tier" not in body


def test_openai_reasoning_model_uses_max_completion_tokens() -> None:
    # gpt-5 / o-series reject ``max_tokens`` (400 unsupported_parameter);
    # they require ``max_completion_tokens``.
    p = OpenAI.from_key("k")
    m = p.model("gpt-5.5")
    body = m._build_body(
        ModelRequest(messages=[UserMessage(text="x")], max_response_tokens=42),
        stream=False,
    )
    assert body["max_completion_tokens"] == 42
    assert "max_tokens" not in body


def test_openai_valid_latency_modes_fast() -> None:
    p = OpenAI.from_key("k")
    assert p.model("gpt-5.5").valid_latency_modes == ("fast",)


def test_openai_fast_latency_maps_to_priority_tier() -> None:
    p = OpenAI.from_key("k")
    m = p.model("gpt-5.5")
    body = m._build_body(
        ModelRequest(messages=[UserMessage(text="x")], latency="fast"),
        stream=False,
    )
    assert body["service_tier"] == "priority"


def test_openai_fast_latency_overrides_explicit_service_tier() -> None:
    p = OpenAI.from_key("k")
    m = p.model("gpt-5.5")
    body = m._build_body(
        ModelRequest(
            messages=[UserMessage(text="x")], latency="fast", service_tier="flex"
        ),
        stream=False,
    )
    assert body["service_tier"] == "priority"


def test_openai_build_body_omits_unknown_service_tier() -> None:
    p = OpenAI.from_key("k")
    m = p.model("gpt-5.5")
    body = m._build_body(
        ModelRequest(messages=[UserMessage(text="x")], service_tier="bogus"),
        stream=False,
    )
    assert "service_tier" not in body


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
