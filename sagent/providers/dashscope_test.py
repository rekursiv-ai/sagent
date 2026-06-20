"""Tests for ``providers.dashscope``: DashScope (Qwen) overrides + body transform."""

from __future__ import annotations

from typing import cast

import pytest

from sagent.lib.custom_json import MutableJSON
from sagent.providers.dashscope import DashScope, _DashScopeModel
from sagent.types.model import ModelRequest


def test_dashscope_from_key() -> None:
    p = DashScope.from_key("sk-test")
    assert isinstance(p, DashScope)
    assert p.api_key == "sk-test"
    assert p.base_url.startswith("https://dashscope-intl.aliyuncs.com")


def test_dashscope_from_env_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="API key not configured"):
        DashScope.from_env()


def test_dashscope_default_model() -> None:
    p = DashScope.from_key("k")
    m = p.model()
    assert m.model_id == DashScope.DEFAULT_MODEL
    # Reasoning is surfaced via ``reasoning_content`` on Qwen3.
    assert m.supports_thinking is True


def test_dashscope_unknown_model_raises() -> None:
    p = DashScope.from_key("k")
    with pytest.raises(ValueError, match="Unknown model"):
        _ = p.model("not-qwen")


@pytest.mark.parametrize(
    ("model_id", "is_effort"),
    [
        ("qwen3-32b", True),
        ("qwen3.6-plus", False),  # qwen3.6 family doesn't start with qwen3-.
        ("qwen-plus", True),
        ("qwen-max", True),
        ("qvq-test", True),
        ("qwq-test", True),
        ("kimi-k2.6", False),
    ],
)
def test_dashscope_is_effort_model(model_id: str, is_effort: bool) -> None:
    p = DashScope.from_key("k")
    # Use a known profile id to construct the model; the predicate runs
    # over the supplied model id regardless of profile.
    m = p.model("qwen3-32b")
    assert m._is_effort_model(model_id) is is_effort


def test_dashscope_transform_body_maps_effort_to_enable_thinking() -> None:
    p = DashScope.from_key("k")
    m = cast(_DashScopeModel, p.model("qwen3-32b"))
    body: MutableJSON = {"model": "qwen3-32b", "reasoning_effort": "low"}
    out = m._transform_body(body, ModelRequest(messages=[]))
    assert "reasoning_effort" not in out
    assert out["enable_thinking"] is True


def test_dashscope_transform_body_none_effort_means_disabled() -> None:
    p = DashScope.from_key("k")
    m = cast(_DashScopeModel, p.model("qwen3-32b"))
    body: MutableJSON = {"reasoning_effort": "none"}
    out = m._transform_body(body, ModelRequest(messages=[]))
    assert out["enable_thinking"] is False


def test_dashscope_transform_body_thinking_suffix_strips_enable_thinking() -> None:
    p = DashScope.from_key("k")
    # ``*-thinking-2507`` models always reason; should not forward the flag.
    m = cast(
        _DashScopeModel,
        p.model("qwen3-235b-a22b-thinking-2507"),
    )
    body: MutableJSON = {"reasoning_effort": "low"}
    out = m._transform_body(body, ModelRequest(messages=[]))
    assert "enable_thinking" not in out


def test_dashscope_transform_body_no_effort_unchanged() -> None:
    p = DashScope.from_key("k")
    m = cast(_DashScopeModel, p.model("qwen3-32b"))
    body: MutableJSON = {"model": "qwen3-32b"}
    out = m._transform_body(body, ModelRequest(messages=[]))
    assert "enable_thinking" not in out
    assert out["model"] == "qwen3-32b"


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
