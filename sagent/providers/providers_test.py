"""Tests for ``providers.providers``: build_provider + infer_provider + names."""

from __future__ import annotations

import argparse
import logging
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
    collect_provider_args,
    infer_provider,
    parse_provider_arg,
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


def test_build_provider_anthropic_from_key_auth() -> None:
    p = build_provider("Anthropic", "key", api_key="sk-ant-test")
    assert isinstance(p, Anthropic)


def test_build_provider_google_from_key_auth() -> None:
    p = build_provider("Google", "key", api_key="AIzaTest")
    assert isinstance(p, Google)


def test_build_provider_openai_from_key_auth() -> None:
    p = build_provider("OpenAI", "key", api_key="sk-test")
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


def test_build_provider_openai_compat_from_key_auth() -> None:
    p = build_provider("OpenAICompat", "key", api_key="any-key")
    assert isinstance(p, OpenAICompat)


def test_build_provider_dashscope_from_key_auth() -> None:
    p = build_provider("DashScope", "key", api_key="sk-dash-test")
    assert isinstance(p, DashScope)


def test_build_provider_minimax_from_key_auth() -> None:
    p = build_provider("MiniMax", "key", api_key="sk-mini-test")
    assert isinstance(p, MiniMax)


def test_build_provider_moonshot_from_key_auth() -> None:
    p = build_provider("Moonshot", "key", api_key="sk-moon-test")
    assert isinstance(p, Moonshot)


def test_build_provider_llamacpp_from_key_auth() -> None:
    p = build_provider("LlamaCpp", "key", api_key="local")
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


# ---- ``--provider-arg`` mechanism --------------------------------------


def test_parse_provider_arg_typed_json_values() -> None:
    assert parse_provider_arg("Anthropic.flag=true") == (
        "Anthropic",
        "flag",
        True,
    )
    assert parse_provider_arg("Anthropic.n=42") == ("Anthropic", "n", 42)
    assert parse_provider_arg('Anthropic.s="hi"') == (
        "Anthropic",
        "s",
        "hi",
    )
    assert parse_provider_arg("Anthropic.opt=null") == (
        "Anthropic",
        "opt",
        None,
    )


def test_parse_provider_arg_falls_back_to_string_on_decode_failure() -> None:
    """Bare identifiers and paths pass through as strings."""
    assert parse_provider_arg("Anthropic.path=/var/foo") == (
        "Anthropic",
        "path",
        "/var/foo",
    )
    assert parse_provider_arg("Anthropic.tag=production") == (
        "Anthropic",
        "tag",
        "production",
    )


def test_parse_provider_arg_rejects_malformed() -> None:
    err = r"expected Class\.key=value"
    with pytest.raises(argparse.ArgumentTypeError, match=err):
        parse_provider_arg("no-dot=value")
    with pytest.raises(argparse.ArgumentTypeError, match=err):
        parse_provider_arg("Class.key-without-eq")
    with pytest.raises(argparse.ArgumentTypeError, match=err):
        parse_provider_arg(".key=value")


def test_collect_provider_args_walks_mro() -> None:
    """A spec keyed on a base class applies to the subclass."""
    # Use the real provider hierarchy: ``OpenAISubscription`` inherits
    # from ``OpenAI``, so a spec keyed on ``OpenAI`` must flow.
    args = [
        "OpenAI.server_side_context_management=true",
        "Google.something=ignored",
    ]
    merged = collect_provider_args(args, "OpenAISubscription")
    assert merged == {"server_side_context_management": True}


def test_collect_provider_args_leaf_wins_over_base_on_collision() -> None:
    args = [
        "OpenAI.knob=1",
        "OpenAISubscription.knob=2",
    ]
    assert collect_provider_args(args, "OpenAISubscription") == {"knob": 2}


def test_collect_provider_args_ignores_unrelated_classes() -> None:
    args = ["OpenAI.k=1", "Google.k=2"]
    assert collect_provider_args(args, "Anthropic") == {}


def test_collect_provider_args_unknown_provider_raises() -> None:
    with pytest.raises(AttributeError, match="unknown provider"):
        collect_provider_args(["X.k=1"], "DoesNotExist")


def test_build_provider_warns_and_drops_unknown_kwargs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Unknown kwargs are dropped with a warning rather than raising."""
    with caplog.at_level(logging.WARNING, logger="sagent.providers.providers"):
        p = build_provider("Anthropic", "key", api_key="sk-test", does_not_exist=42)
    assert isinstance(p, Anthropic)
    assert any("does_not_exist" in rec.message for rec in caplog.records)


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
