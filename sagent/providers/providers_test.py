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


def test_infer_provider_preserves_same_family_variant() -> None:
    """A same-vendor model swap keeps the current provider variant, never
    downgrading it to the bare API-key sibling.

    An agent already on a variant provider (``AnthropicCLI``,
    ``OpenAISubscription``, ...) that swaps to another model of the SAME
    vendor family must keep its provider/auth: re-inferring to the bare
    ``Anthropic``/``OpenAI`` + ``env`` path would demand an API key the
    operator never set (they authenticate via the CLI/subscription). The
    guard is base-family prefix, so it holds regardless of which override
    entries the account table happens to carry -- the property the public
    (exported) build relies on for ``OpenAISubscription``, which its
    override table does not enumerate.
    """
    assert infer_provider("claude-haiku-4-5", current_provider="AnthropicCLI") is None
    assert infer_provider("gpt-5.5", current_provider="OpenAISubscription") is None
    # Cross-vendor from a variant caller still resolves to the new vendor.
    assert infer_provider("gpt-5.5", current_provider="AnthropicCLI") == (
        "OpenAI",
        "env",
    )


def test_infer_provider_prefers_account_variant_for_account_callers() -> None:
    """A claude-prefixed model id resolves to the ACCOUNT provider when the
    caller already runs on one.

    ``AgentSelf(model_id="claude-...")`` from an AnthropicCLI-backed
    agent must not silently build a fresh API-key ``Anthropic`` provider
    (which would demand ``ANTHROPIC_API_KEY`` even though the operator
    authenticates via the CLI subscription). Same-provider stays a
    no-op; API-key callers are unaffected.
    """
    # CLI-backed caller asking for another claude model: the override maps
    # Anthropic -> AnthropicCLI, which matches the current provider, so no
    # rebuild at all. Without the override this returned ("Anthropic",
    # "env") -- a fresh API-key provider.
    assert infer_provider("claude-haiku-4-5", current_provider="AnthropicCLI") is None
    # Cross-vendor from a CLI caller still maps to the vendor's API path.
    assert infer_provider("gpt-5.5", current_provider="AnthropicCLI") == (
        "OpenAI",
        "env",
    )


def test_infer_provider_returns_none_for_unknown_prefix() -> None:
    assert infer_provider("mystery-model", current_provider="OpenAI") is None


def test_infer_provider_local_path_returns_self_hosted() -> None:
    result = infer_provider("/opt/models/qwen", current_provider="None")
    assert result == ("SelfHosted", "/opt/models/qwen")


@pytest.mark.parametrize("path", ["./local", "../parent", "~/snapshot"])
def test_infer_provider_other_local_paths(path: str) -> None:
    result = infer_provider(path, current_provider="None")
    assert result == ("SelfHosted", path)


def test_build_provider_unknown_provider_raises() -> None:
    with pytest.raises(AttributeError, match="unknown provider"):
        build_provider("DoesNotExist")


def test_build_provider_from_env_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-env-key")
    p = build_provider("Anthropic", "env")
    assert isinstance(p, Anthropic)


@pytest.mark.parametrize(
    ("provider_name", "cls", "env_var"),
    [
        ("Google", Google, "GOOGLE_API_KEY"),
        ("OpenAI", OpenAI, "OPENAI_API_KEY"),
        ("DashScope", DashScope, "DASHSCOPE_API_KEY"),
        ("MiniMax", MiniMax, "MINIMAX_API_KEY"),
        ("Moonshot", Moonshot, "MOONSHOT_API_KEY"),
    ],
)
def test_build_provider_from_env_dispatch_per_provider(
    monkeypatch: pytest.MonkeyPatch,
    provider_name: str,
    cls: type,
    env_var: str,
) -> None:
    monkeypatch.setenv(env_var, "test-key")
    p = build_provider(provider_name, "env")
    assert isinstance(p, cls)


def test_build_provider_llamacpp_from_key_direct() -> None:
    # ``from_key`` factories take their key positionally at the class --
    # ``build_provider`` dispatches auth methods, never credentials.
    assert isinstance(LlamaCpp.from_key("local"), LlamaCpp)


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


def test_build_provider_missing_auth_method_raises_even_with_from_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class StubProvKeyOnly:
        @classmethod
        def from_key(cls, api_key: str) -> StubProvKeyOnly:
            del api_key
            return cls()

    providers_mod = sys.modules["sagent.providers"]
    monkeypatch.setattr(
        providers_mod, "StubProvKeyOnly", StubProvKeyOnly, raising=False
    )
    with pytest.raises(AttributeError, match="no ``from_literal-key-here``"):
        build_provider("StubProvKeyOnly", "literal-key-here")


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


def test_build_provider_defers_to_the_factory_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Construction knobs are the factory's; ``build_provider`` invents none."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-env-key")
    p = build_provider("Anthropic", "env")
    assert isinstance(p, Anthropic)
    assert p.server_side_context_management is False


def test_build_provider_forwards_account_only_where_declared(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A factory without an ``account`` parameter must not receive one."""
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    assert isinstance(build_provider("Google", "env", account="ignored"), Google)


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
