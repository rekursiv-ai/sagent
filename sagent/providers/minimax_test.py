"""Tests for ``providers.minimax``: MiniMax OpenAI-compat surface."""

from __future__ import annotations

import pytest

from sagent.providers.minimax import MiniMax


def test_minimax_from_key() -> None:
    p = MiniMax.from_key("sk-test")
    assert isinstance(p, MiniMax)
    assert p.api_key == "sk-test"
    assert p.base_url == "https://api.minimax.io/v1"


def test_minimax_from_env_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="API key not configured"):
        MiniMax.from_env()


def test_minimax_from_env_reads(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MINIMAX_API_KEY", "sk-mini-env")
    p = MiniMax.from_env()
    assert p.api_key == "sk-mini-env"


def test_minimax_default_model_known() -> None:
    p = MiniMax.from_key("k")
    m = p.model()
    assert m.model_id == MiniMax.DEFAULT_MODEL


def test_minimax_unknown_model_raises() -> None:
    p = MiniMax.from_key("k")
    with pytest.raises(ValueError, match="Unknown model"):
        _ = p.model("not-minimax")


def test_minimax_model_supports_thinking_via_reasoning_field() -> None:
    p = MiniMax.from_key("k")
    m = p.model("MiniMax-M2.7")
    # ``_reasoning_field = "reasoning_content"`` → supports_thinking True.
    assert m.supports_thinking is True


def test_minimax_known_models_have_pricing() -> None:
    p = MiniMax.from_key("k")
    for mid in MiniMax.KNOWN_MODELS:
        m = p.model(mid)
        assert m.pricing.request > 0
        assert m.pricing.response > 0


def test_minimax_base_url_override_via_from_key() -> None:
    p = MiniMax.from_key("k", base_url="http://localhost:8000/v1")
    assert p.base_url == "http://localhost:8000/v1"


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
