"""Tests for ``ModelSwitchHandler._parse_args``."""

from __future__ import annotations

from sagent.agent.handlers.model_switch import _parse_args
from sagent.custom_types import ModelSpec


def _spec() -> ModelSpec:
    return ModelSpec(
        provider="Anthropic",
        auth="env",
        model_id="claude-sonnet-4-6",
        account=None,
    )


def test_provider_only_carries_existing_model_id() -> None:
    """``/model --provider X`` keeps the current model rather than the new provider's default.

    Regression: ``model_id`` previously defaulted to ``None`` and the
    handler then called ``provider.model(None)``, silently substituting
    each provider's ``DEFAULT_MODEL`` for what the user already had.
    """
    result = _parse_args(["--provider", "OpenAI"], _spec())
    assert isinstance(result, tuple)
    prov_name, auth, account, model_id = result
    assert prov_name == "OpenAI"
    assert auth == "env"
    assert account is None
    assert model_id == "claude-sonnet-4-6"


def test_no_arguments_returns_usage() -> None:
    result = _parse_args([], _spec())
    assert isinstance(result, str)
    assert "usage" in result


def test_bare_model_id_overrides() -> None:
    result = _parse_args(["gpt-5.5"], _spec())
    assert isinstance(result, tuple)
    _prov, _auth, _account, model_id = result
    assert model_id == "gpt-5.5"


def test_kv_form_overrides() -> None:
    result = _parse_args(["model=gpt-5.5", "auth=oauth"], _spec())
    assert isinstance(result, tuple)
    _prov, auth, _account, model_id = result
    assert model_id == "gpt-5.5"
    assert auth == "oauth"


def test_account_default_alias_clears() -> None:
    spec = ModelSpec(
        provider="Anthropic",
        auth="env",
        model_id="claude-sonnet-4-6",
        account="work",
    )
    result = _parse_args(["account=default"], spec)
    assert isinstance(result, tuple)
    _prov, _auth, account, _model = result
    assert account is None


def test_unknown_flag_returns_error() -> None:
    result = _parse_args(["--bogus", "x"], _spec())
    assert isinstance(result, str)
    assert "unknown flag" in result


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
