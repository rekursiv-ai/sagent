"""Tests for sagent.providers.moonshot."""

from __future__ import annotations

import pytest

from sagent.providers.moonshot import Moonshot


class TestMoonshot:
    def test_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MOONSHOT_API_KEY", "mk-test")
        p = Moonshot.from_env()
        assert p.api_key == "mk-test"
        assert p.base_url.endswith("moonshot.ai/v1")

    def test_from_env_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="API key not configured"):
            Moonshot.from_env()

    def test_default_model(self) -> None:
        m = Moonshot.from_key("mk-x").model()
        assert m.model_id == Moonshot.DEFAULT_MODEL

    def test_thinking_surface(self) -> None:
        m = Moonshot.from_key("mk-x").model()
        assert m.supports_thinking is True

    def test_self_hosted_base_url(self) -> None:
        p = Moonshot.from_key("empty", base_url="http://localhost:8000/v1")
        m = p.model()
        assert m._endpoint == "http://localhost:8000/v1/chat/completions"


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
