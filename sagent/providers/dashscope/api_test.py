"""Tests for ``providers.dashscope``: DashScope (Qwen) overrides + body transform."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest

from sagent.agent.agent import Agent
from sagent.providers.dashscope.api import DashScope, _DashScopeModel
from sagent.types.capability import (
    ModelSettings,
    ThinkingBudget,
    ThinkingEffort,
)
from sagent.types.model import ModelRequest
from sagent.types.runtime import UserMessage


if TYPE_CHECKING:
    from sagent.lib.codec import MutablePlainTree


def test_dashscope_from_key() -> None:
    p = DashScope.from_key("sk-test")
    assert p.api_key == "sk-test"
    assert p.base_url.startswith("https://dashscope-intl.aliyuncs.com")


def test_dashscope_from_env_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="API key not configured"):
        DashScope.from_env()


def test_dashscope_default_model() -> None:
    p = DashScope.from_key("k")
    m = p.model()
    assert m.capability.model_id == p.catalog.resolve("default")[0].model_id
    assert m.capability.thinking.budget != frozenset({"none"})


def test_dashscope_unknown_model_raises() -> None:
    p = DashScope.from_key("k")
    with pytest.raises(ValueError, match="Unknown model"):
        _ = p.model("not-qwen")


def _model(
    model_id: str,
    *,
    budget: ThinkingBudget | None = None,
    effort: ThinkingEffort | None = None,
) -> _DashScopeModel:
    """Return a model with the given axes selected over its own narrowest."""
    m = cast(_DashScopeModel, DashScope.from_key("k").model(model_id))
    settings = ModelSettings.narrowest(m.capability)
    if budget is not None:
        settings.thinking_budget = budget
    if effort is not None:
        settings.thinking_effort = effort
    m._settings = settings
    return m


def _body(
    m: _DashScopeModel,
    body: dict[str, MutablePlainTree] | None = None,
) -> dict[str, MutablePlainTree]:
    return m._transform_body(
        body if body is not None else {},
        ModelRequest(messages=[]),
    )


def test_a_fixed_budget_caps_reasoning_with_thinking_budget() -> None:
    m = _model("qwen-plus-3.7", budget="fixed", effort="low")
    out = _body(m, {"model": "qwen3.7-plus", "reasoning_effort": "low"})
    assert "reasoning_effort" not in out
    assert out["enable_thinking"] is True
    assert out["thinking_budget"] == 4_096


def test_effort_levels_map_to_distinct_budgets() -> None:
    high = _body(_model("qwen-32b-3.0", budget="fixed", effort="high"))
    maxi = _body(_model("qwen-32b-3.0", budget="fixed", effort="max"))
    assert high["thinking_budget"] == 16_384
    assert maxi["thinking_budget"] == 24_576


def test_an_auto_budget_thinks_at_the_model_default() -> None:
    out = _body(_model("qwen-plus-3.7", budget="auto", effort="high"))
    assert out == {"enable_thinking": True}


def test_a_none_budget_disables_thinking() -> None:
    out = _body(_model("qwen-plus-3.7"), {"reasoning_effort": "minimal"})
    assert out == {"enable_thinking": False}


def test_a_level_row_sends_reasoning_effort_and_never_a_budget() -> None:
    """Qwen 3.8 errors when ``reasoning_effort`` and ``thinking_budget`` co-occur."""
    out = _body(_model("qwen-max-3.8", budget="auto", effort="xhigh"))
    assert out == {"enable_thinking": True, "reasoning_effort": "xhigh"}


def test_a_level_row_with_no_effort_leaves_the_server_default() -> None:
    out = _body(_model("qwen-flash-3.8", budget="auto"))
    assert out == {"enable_thinking": True}


def test_an_instruct_row_shares_the_qwen3_prefix_yet_gets_no_knob() -> None:
    """An id-prefix rule once sent ``enable_thinking`` to every ``qwen3*`` id."""
    m = _model("qwen-235b-a22b-instruct-3.0")
    assert m.capability.thinking.budget == frozenset({"none"})
    assert _body(m) == {}


def test_a_thinking_only_row_never_disables_thinking() -> None:
    """Model Studio rejects ``enable_thinking=false`` on ``*-thinking-2507``."""
    m = _model("qwen-235b-a22b-thinking-3.0", effort="low")
    assert "none" not in m.capability.thinking.budget
    assert _body(m)["enable_thinking"] is True
    with pytest.raises(ValueError, match="thinking_budget"):
        m.settings.thinking_budget = "none"


@pytest.mark.parametrize(
    "model_id",
    ["qwen-235b-a22b-instruct-3.0", "qwen-coder-480b-a35b-3.0"],
)
def test_a_non_reasoning_row_gets_no_thinking_knobs(model_id: str) -> None:
    m = _model(model_id)
    assert m.capability.thinking.effort == frozenset({"none"})
    out = _body(m)
    assert "enable_thinking" not in out
    assert "thinking_budget" not in out


@pytest.mark.parametrize("model_id", sorted(DashScope.catalog.models))
def test_the_wire_follows_the_row_not_the_id(model_id: str) -> None:
    """A row offering a budget sends the toggle; one that does not, never does."""
    m = _model(model_id)
    sends_toggle = "enable_thinking" in _body(m)
    assert sends_toggle is (m.capability.thinking.budget != frozenset({"none"}))


@pytest.mark.parametrize(
    ("model_id", "field"),
    [
        ("qwen-max-3.8", "max_completion_tokens"),
        ("qwen-flash-3.8", "max_completion_tokens"),
        ("qwen-plus-3.7", "max_completion_tokens"),
        ("qwen-flash-3.6", "max_completion_tokens"),
        ("qwen-27b-3.8", "max_tokens"),
        ("qwen-max-3.6", "max_tokens"),
        ("qwen-turbo-3.0", "max_tokens"),
        ("qwen-32b-3.0", "max_tokens"),
        ("qwen-235b-a22b-instruct-3.0", "max_tokens"),
    ],
)
def test_the_output_cap_uses_the_field_the_model_supports(
    model_id: str,
    field: str,
) -> None:
    """qwen-api-via-openai-chat-completions lists ``max_completion_tokens`` models.

    Elsewhere only ``max_tokens`` is honored, and it caps the answer alone.
    """
    body = _model(model_id)._build_body(
        ModelRequest(messages=[UserMessage(text="x")], max_response_tokens=42),
        stream=False,
    )
    assert {k for k in ("max_tokens", "max_completion_tokens") if k in body} == {
        field,
    }
    assert body[field] == 42


def test_the_default_model_supports_effort_end_to_end() -> None:
    m = DashScope.from_key("k").model()
    agent = Agent(model=m)
    agent.model.settings.thinking_effort = "medium"
    assert m.settings.thinking_effort == "medium"


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
