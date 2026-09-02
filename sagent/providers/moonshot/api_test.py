"""Tests for ``providers.moonshot``: Moonshot OpenAI-compat surface."""

from __future__ import annotations

import pytest

from sagent.providers.moonshot.api import Moonshot


def test_moonshot_from_key() -> None:
    p = Moonshot.from_key("sk-moon-test")
    assert isinstance(p, Moonshot)
    assert p.api_key == "sk-moon-test"
    assert p.base_url == "https://api.moonshot.ai/v1"


def test_moonshot_from_env_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="API key not configured"):
        Moonshot.from_env()


def test_moonshot_from_env_reads(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MOONSHOT_API_KEY", "sk-moon-env")
    p = Moonshot.from_env()
    assert p.api_key == "sk-moon-env"


def test_moonshot_default_model() -> None:
    p = Moonshot.from_key("k")
    m = p.model()
    assert m.capability.model_id == Moonshot.DEFAULT_MODEL
    # Kimi surfaces reasoning via ``reasoning_content``.
    assert m.capability.thinking_output == frozenset({"none", "text"})


def test_moonshot_unknown_model_raises() -> None:
    p = Moonshot.from_key("k")
    with pytest.raises(ValueError, match="Unknown model"):
        _ = p.model("not-kimi")


def test_moonshot_known_models_have_pricing_and_limits() -> None:
    p = Moonshot.from_key("k")
    for mid in Moonshot.CAPABILITIES:
        m = p.model(mid)
        assert m.max_request_tokens > 0
        assert m.limits.max_response_tokens > 0


def test_moonshot_base_url_override_via_from_key() -> None:
    p = Moonshot.from_key("k", base_url="http://localhost:8000/v1")
    assert p.base_url == "http://localhost:8000/v1"


def test_moonshot_offers_no_prompt_cache() -> None:
    p = Moonshot.from_key("k")
    m = p.model("kimi-k2.6")
    assert m.capability.cache_ttl_sec == 0.0


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
