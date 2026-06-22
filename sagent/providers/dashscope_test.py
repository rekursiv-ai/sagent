"""Tests for ``providers.dashscope``: DashScope (Qwen) overrides + body transform."""

from __future__ import annotations

from typing import cast

import pytest

from sagent.agent.agent import Agent
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
        # The qwen3.6 family (incl. the DEFAULT_MODEL) is Qwen3-generation and
        # MUST expose the effort knob; the prefix has to match the dotted form.
        ("qwen3.6-plus", True),
        ("qwen3.6-max-preview", True),
        ("qwen3.6-flash", True),
        ("qwen-plus", True),
        ("qwen-max", True),
        ("qvq-test", True),
        ("qwq-test", True),
        ("kimi-k2.6", False),
        ("qwen2.5-vl", False),  # pre-Qwen3 vision model: no thinking knob.
        # ``-instruct`` / ``-coder`` qwen3 ids are NON-reasoning models (Alibaba
        # ships them without the enable_thinking toggle). A bare ``qwen3`` prefix
        # match wrongly flags them; they must be excluded like the suffix-stripped
        # ``-thinking`` ids are. All three are registered in KNOWN_MODELS.
        ("qwen3-235b-a22b-instruct-2507", False),
        ("qwen3-30b-a3b-instruct-2507", False),
        ("qwen3-coder-480b-a35b-instruct", False),
        # ``-thinking`` qwen3 ids ARE reasoning models -- must stay effort-capable.
        ("qwen3-235b-a22b-thinking-2507", True),
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
    # Effort is read from the request (the raw sagent value), and the base's
    # ``reasoning_effort`` (which DashScope rejects) is dropped.
    body: MutableJSON = {"model": "qwen3-32b", "reasoning_effort": "low"}
    out = m._transform_body(body, ModelRequest(messages=[], effort="low"))
    assert "reasoning_effort" not in out
    assert out["enable_thinking"] is True
    # The level drives a reasoning-token budget, not just an on/off flag.
    assert out["thinking_budget"] == 4_096


def test_dashscope_transform_body_effort_levels_map_distinct_budgets() -> None:
    p = DashScope.from_key("k")
    m = cast(_DashScopeModel, p.model("qwen3-32b"))
    high = m._transform_body({}, ModelRequest(messages=[], effort="high"))
    maxi = m._transform_body({}, ModelRequest(messages=[], effort="max"))
    assert high["thinking_budget"] == 16_384
    assert maxi["thinking_budget"] == 32_768
    assert high["thinking_budget"] != maxi["thinking_budget"], (
        "distinct effort levels must not collapse to the same budget"
    )


def test_dashscope_transform_body_none_effort_means_disabled() -> None:
    p = DashScope.from_key("k")
    m = cast(_DashScopeModel, p.model("qwen3-32b"))
    # ``none`` disables thinking and sets no budget, even though the base maps
    # ``none`` -> ``minimal`` into ``reasoning_effort`` for OpenAI.
    body: MutableJSON = {"reasoning_effort": "minimal"}
    out = m._transform_body(body, ModelRequest(messages=[], effort="none"))
    assert out["enable_thinking"] is False
    assert "thinking_budget" not in out


def test_dashscope_transform_body_thinking_suffix_strips_thinking_knobs() -> None:
    p = DashScope.from_key("k")
    # ``*-thinking-2507`` models always reason; neither ``enable_thinking`` nor
    # ``thinking_budget`` should be forwarded -- forwarding a budget the
    # always-on model may reject is the same wire hazard as the flag.
    m = cast(
        _DashScopeModel,
        p.model("qwen3-235b-a22b-thinking-2507"),
    )
    out = m._transform_body({}, ModelRequest(messages=[], effort="low"))
    assert "enable_thinking" not in out
    assert "thinking_budget" not in out


def test_dashscope_non_effort_model_gets_no_thinking_knobs() -> None:
    """A model that is not an effort model must never receive thinking knobs.

    ``qwen-turbo`` is registered but lacks any thinking prefix, so
    ``_is_effort_model`` is False. ``_transform_body`` must gate on the SAME
    predicate -- not a partial marker check -- or a direct caller setting effort
    ships ``enable_thinking``/``thinking_budget`` the model rejects.
    """
    p = DashScope.from_key("k")
    m = cast(_DashScopeModel, p.model("qwen-turbo"))
    assert m.supports_effort is False
    out = m._transform_body({}, ModelRequest(messages=[], effort="low"))
    assert "enable_thinking" not in out
    assert "thinking_budget" not in out


def test_dashscope_instruct_model_gets_no_thinking_knobs() -> None:
    """A non-reasoning ``-instruct`` qwen3 model must not be fed thinking knobs.

    These models reject/ignore ``enable_thinking`` and ``thinking_budget``; the
    bare ``qwen3`` prefix wrongly matched them, so ``_transform_body`` shipped the
    toggle on the wire. Neither knob may appear, and the model must not advertise
    effort at all.
    """
    p = DashScope.from_key("k")
    m = cast(_DashScopeModel, p.model("qwen3-235b-a22b-instruct-2507"))
    assert m.supports_effort is False
    out = m._transform_body({}, ModelRequest(messages=[], effort="high"))
    assert "enable_thinking" not in out
    assert "thinking_budget" not in out


def test_dashscope_default_model_supports_effort_end_to_end() -> None:
    """The documented effort knob must work on the DEFAULT model.

    ``DEFAULT_MODEL`` is a qwen3.6 model; ``Agent.effort`` must accept a level
    rather than raising "does not support effort" -- otherwise the module
    docstring's effort->enable_thinking promise is unreachable for default use.
    """
    p = DashScope.from_key("k")
    m = p.model()  # DEFAULT_MODEL
    assert m.supports_effort is True
    agent = Agent(model=m)
    agent.effort = "medium"  # must not raise
    assert agent.effort == "medium"


def test_dashscope_transform_body_no_effort_unchanged() -> None:
    p = DashScope.from_key("k")
    m = cast(_DashScopeModel, p.model("qwen3-32b"))
    body: MutableJSON = {"model": "qwen3-32b"}
    out = m._transform_body(body, ModelRequest(messages=[]))
    assert "enable_thinking" not in out
    assert "thinking_budget" not in out
    assert out["model"] == "qwen3-32b"


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
