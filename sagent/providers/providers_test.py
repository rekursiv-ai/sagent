"""Tests for ``providers.providers``: build_provider + infer_provider + names."""

from __future__ import annotations

import inspect
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
    supported_provider_options,
)
from sagent.types.providers import ProviderOptions


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


# ---- ``ProviderOptions`` mechanism --------------------------------------


def test_provider_options_set_fields_returns_only_non_none() -> None:
    assert ProviderOptions().set_fields() == {}
    assert ProviderOptions(redact_thinking=False).set_fields() == {
        "redact_thinking": False,
    }
    assert ProviderOptions(
        redact_thinking=True,
        server_side_context_management=True,
    ).set_fields() == {
        "redact_thinking": True,
        "server_side_context_management": True,
    }


def test_supported_provider_options_declarations() -> None:
    assert supported_provider_options("Anthropic") == {
        "redact_thinking",
        "server_side_context_management",
    }
    # The CLI wrapper's ``from_credentials`` takes no construction options.
    assert supported_provider_options("AnthropicCLI") == frozenset()
    # Providers without a declaration take no options.
    assert supported_provider_options("Google") == frozenset()


def test_supported_provider_options_unknown_provider_raises() -> None:
    with pytest.raises(AttributeError, match="unknown provider"):
        supported_provider_options("DoesNotExist")


def test_build_provider_forwards_supported_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-env-key")
    p = build_provider(
        "Anthropic",
        "env",
        options=ProviderOptions(server_side_context_management=True),
    )
    assert isinstance(p, Anthropic)
    assert p.server_side_context_management is True


def test_build_provider_unset_options_defer_to_factory_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-env-key")
    p = build_provider("Anthropic", "env", options=ProviderOptions())
    assert isinstance(p, Anthropic)
    assert p.server_side_context_management is False


def test_build_provider_unsupported_option_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicitly set option on a non-supporting provider fails fast."""
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    with pytest.raises(ValueError, match="does not support option"):
        build_provider(
            "Google",
            "env",
            options=ProviderOptions(redact_thinking=True),
        )


def test_build_provider_unsupported_option_names_the_offender(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    with pytest.raises(ValueError, match="server_side_context_management"):
        build_provider(
            "Google",
            "env",
            options=ProviderOptions(server_side_context_management=True),
        )


def test_anthropic_cli_rejects_options_supported_by_base() -> None:
    """The CLI wrapper's empty declaration overrides the inherited one."""
    with pytest.raises(ValueError, match="does not support option"):
        build_provider(
            "AnthropicCLI",
            "credentials",
            options=ProviderOptions(redact_thinking=True),
        )


def test_anthropic_declarations_match_factory_signatures() -> None:
    """Every declared option is a real keyword on every ``from_*`` factory.

    Guards the declaration against drifting from the constructor
    surface -- the failure mode the deleted reflection filter used to
    paper over.
    """
    for auth in ("key", "env"):
        factory = getattr(Anthropic, f"from_{auth}")
        params = inspect.signature(factory).parameters
        missing = Anthropic.supported_options - params.keys()
        assert not missing, f"Anthropic.from_{auth} missing {missing}"


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
