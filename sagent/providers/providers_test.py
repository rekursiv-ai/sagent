"""Tests for ``providers.providers``: build_provider + infer_provider + names."""

from __future__ import annotations

import sys

import pytest

from sagent.providers import (
    PROVIDER_NAMES,
    Anthropic,
    DashScope,
    Google,
    LlamaCpp,
    MiniMax,
    Moonshot,
    OpenAI,
    OpenAICompat,
    SelfHosted,
    build_provider,
    infer_provider,
)


def test_provider_names_contains_core_providers() -> None:
    for name in ("Anthropic", "Google", "OpenAI", "DashScope", "MiniMax"):
        assert name in PROVIDER_NAMES


def test_provider_names_includes_self_hosted_and_compat() -> None:
    assert "SelfHosted" in PROVIDER_NAMES
    assert "OpenAICompat" in PROVIDER_NAMES
    assert "LlamaCpp" in PROVIDER_NAMES


@pytest.mark.parametrize(
    ("model_id", "provider"),
    [
        ("claude-sonnet-4-6", "Anthropic"),
        ("gemini-3-flash-preview", "Google"),
        ("gpt-5.5", "OpenAI"),
        ("chatgpt-4o", "OpenAI"),
        ("o1-mini", "OpenAI"),
        ("o3-mini", "OpenAI"),
        ("o4-x", "OpenAI"),
        ("codex-mini", "OpenAI"),
        ("kimi-k2.6", "Moonshot"),
        ("moonshot-v1-8k", "Moonshot"),
        ("qwen3.6-plus", "DashScope"),
        ("minimax-m2", "MiniMax"),
    ],
)
def test_infer_provider_from_prefix(model_id: str, provider: str) -> None:
    result = infer_provider(model_id, current_provider="None")
    assert result is not None
    assert result[0] == provider
    assert result[1] == "env"


def test_infer_provider_returns_none_when_already_matches() -> None:
    assert infer_provider("claude-sonnet-4-6", current_provider="Anthropic") is None


def test_infer_provider_returns_none_for_unknown_prefix() -> None:
    assert infer_provider("mystery-model", current_provider="OpenAI") is None


def test_infer_provider_local_path_returns_self_hosted() -> None:
    result = infer_provider("/opt/models/qwen", current_provider="None")
    assert result == ("SelfHosted", "/opt/models/qwen")


@pytest.mark.parametrize("path", ["./local", "../parent", "~/snapshot"])
def test_infer_provider_other_local_paths(path: str) -> None:
    result = infer_provider(path, current_provider="None")
    assert result == ("SelfHosted", path)


def test_build_provider_anthropic_from_key() -> None:
    p = build_provider("Anthropic", "sk-ant-test")
    assert isinstance(p, Anthropic)


def test_build_provider_google_from_key() -> None:
    p = build_provider("Google", "AIzaTest")
    assert isinstance(p, Google)


def test_build_provider_openai_from_key() -> None:
    p = build_provider("OpenAI", "sk-test")
    assert isinstance(p, OpenAI)


def test_build_provider_unknown_provider_raises() -> None:
    with pytest.raises(AttributeError, match="unknown provider"):
        build_provider("DoesNotExist")


def test_build_provider_from_env_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-env-key")
    p = build_provider("Anthropic", "env")
    assert isinstance(p, Anthropic)


def test_build_provider_openai_compat_from_key() -> None:
    # OpenAICompat is the base; ``from_key`` requires no ENV_VAR.
    p = build_provider("OpenAICompat", "any-key")
    assert isinstance(p, OpenAICompat)


def test_build_provider_dashscope_from_key() -> None:
    p = build_provider("DashScope", "sk-dash-test")
    assert isinstance(p, DashScope)


def test_build_provider_minimax_from_key() -> None:
    p = build_provider("MiniMax", "sk-mini-test")
    assert isinstance(p, MiniMax)


def test_build_provider_moonshot_from_key() -> None:
    p = build_provider("Moonshot", "sk-moon-test")
    assert isinstance(p, Moonshot)


def test_build_provider_llamacpp_from_key() -> None:
    p = build_provider("LlamaCpp", "local")
    assert isinstance(p, LlamaCpp)


def test_build_provider_account_kw_threaded_when_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If a provider's ``from_<auth>`` accepts ``account``, it gets forwarded."""
    # Build a stub provider class with a from_credentials(account=) signature.
    calls: list[dict[str, object]] = []

    class StubProv:
        @classmethod
        def from_credentials(cls, *, account: str | None = None) -> StubProv:
            calls.append({"account": account})
            return cls()

    providers_mod = sys.modules["sagent.providers"]
    monkeypatch.setattr(providers_mod, "StubProv", StubProv, raising=False)
    out = build_provider("StubProv", "credentials", account="work")
    assert isinstance(out, StubProv)
    assert calls == [{"account": "work"}]


def test_build_provider_falls_back_to_from_key_when_no_auth_method(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Auth strings without a ``from_<auth>`` route to ``from_key(auth)``."""
    captured: list[str] = []

    class StubProvKeyOnly:
        @classmethod
        def from_key(cls, api_key: str) -> StubProvKeyOnly:
            captured.append(api_key)
            return cls()

    providers_mod = sys.modules["sagent.providers"]
    monkeypatch.setattr(
        providers_mod, "StubProvKeyOnly", StubProvKeyOnly, raising=False
    )
    out = build_provider("StubProvKeyOnly", "literal-key-here")
    assert isinstance(out, StubProvKeyOnly)
    assert captured == ["literal-key-here"]


def test_build_provider_no_match_no_from_key_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BareProv:
        pass

    providers_mod = sys.modules["sagent.providers"]
    monkeypatch.setattr(providers_mod, "BareProv", BareProv, raising=False)
    with pytest.raises(AttributeError, match="no ``from_unsupported``"):
        build_provider("BareProv", "unsupported")


def test_self_hosted_imported_via_dispatch() -> None:
    # We exercise that ``SelfHosted`` is exposed; we don't load HF weights.
    assert SelfHosted.DEFAULT_MODEL


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
