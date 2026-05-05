"""Tests for sagent.providers.minimax."""

from __future__ import annotations

import pytest

from sagent.providers.minimax import MiniMax


class TestMiniMax:
    def test_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MINIMAX_API_KEY", "mm-test")
        p = MiniMax.from_env()
        assert p.api_key == "mm-test"
        assert "minimax" in p.base_url

    def test_from_env_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="API key not configured"):
            MiniMax.from_env()

    def test_default_model(self) -> None:
        m = MiniMax.from_key("mm-x").model()
        assert m.model_id == MiniMax.DEFAULT_MODEL

    def test_max_request_tokens_is_long(self) -> None:
        m = MiniMax.from_key("mm-x").model()
        assert m.max_request_tokens == 204_800

    def test_thinking_surface(self) -> None:
        m = MiniMax.from_key("mm-x").model()
        assert m.supports_thinking is True


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
