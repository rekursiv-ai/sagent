"""Tests for sagent.providers.dashscope."""

from __future__ import annotations

import pytest

from sagent.custom_types import ModelRequest, TextMessage
from sagent.providers.dashscope import DashScope


class TestDashScope:
    def test_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-q")
        p = DashScope.from_env()
        assert p.api_key == "sk-q"
        assert "dashscope" in p.base_url

    def test_from_env_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="API key not configured"):
            DashScope.from_env()

    def test_default_model(self) -> None:
        m = DashScope.from_key("sk-x").model()
        assert m.model_id == DashScope.DEFAULT_MODEL

    def test_qwen3_supports_effort(self) -> None:
        m = DashScope.from_key("sk-x").model("qwen3-235b-a22b-instruct-2507")
        assert m.supports_effort is True
        # Non-qwen3 models do not.
        legacy = DashScope.from_key("sk-x").model("qwen-turbo")
        assert legacy.supports_effort is False

    def test_effort_maps_to_enable_thinking(self) -> None:
        m = DashScope.from_key("sk-x").model("qwen3-235b-a22b-instruct-2507")
        req = ModelRequest(
            messages=[TextMessage("hi", "text/x-user-message")],
            effort="high",
        )
        body = m._build_body(req, stream=False)
        assert "reasoning_effort" not in body
        assert body.get("enable_thinking") is True

    def test_effort_none_disables_thinking(self) -> None:
        m = DashScope.from_key("sk-x").model("qwen3-235b-a22b-instruct-2507")
        req = ModelRequest(
            messages=[TextMessage("hi", "text/x-user-message")],
            effort="none",
        )
        body = m._build_body(req, stream=False)
        assert body.get("enable_thinking") is False

    def test_thinking_suffix_model_has_no_toggle(self) -> None:
        m = DashScope.from_key("sk-x").model("qwen3-235b-a22b-thinking-2507")
        req = ModelRequest(
            messages=[TextMessage("hi", "text/x-user-message")],
            effort="high",
        )
        body = m._build_body(req, stream=False)
        # Always-on thinking models don't accept the toggle.
        assert "enable_thinking" not in body
        assert "reasoning_effort" not in body

    def test_thinking_surface(self) -> None:
        m = DashScope.from_key("sk-x").model()
        assert m.supports_thinking is True


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
