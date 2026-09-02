"""Tests for ``providers.dashscope``: DashScope (Qwen) overrides + body transform."""

from __future__ import annotations

from typing import cast

import pytest

from sagent.agent.agent import Agent
from sagent.lib.custom_json import MutableJSON
from sagent.providers.dashscope.api import DashScope, _DashScopeModel
from sagent.types.capability import ModelSettings, ThinkingEffort
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
    assert m.capability.model_id == DashScope.DEFAULT_MODEL
    # Reasoning is surfaced via ``reasoning_content`` on Qwen3.
    assert m.capability.thinking_budget != frozenset({"none"})


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


@pytest.mark.parametrize("model_id", sorted(DashScope.CAPABILITIES))
def test_the_effort_predicate_agrees_with_every_row(model_id: str) -> None:
    """The id-shape predicate and the catalog row must say the same thing.

    ``_is_effort_model`` matches id PREFIXES while the row states the axis
    directly -- two vocabularies for one fact, which is the bug class
    ``wire_conformance_test`` exists to catch. The cases above pin the
    predicate against hand-written ids; nothing pinned it against the
    catalog it actually runs on.
    """
    model = DashScope.from_key("k").model(model_id)
    offers_effort = model.capability.thinking_effort != frozenset({"none"})
    assert model._is_effort_model(model_id) is offers_effort


def _model(model_id: str, effort: ThinkingEffort | None = None) -> _DashScopeModel:
    """A model with ``effort`` selected, or its own narrowest when omitted.

    Not defaulted to ``"none"``: a ``-thinking`` row withholds that value,
    so forcing it would make the helper unusable on exactly the rows whose
    thinking behaviour these tests cover.
    """
    m = cast(_DashScopeModel, DashScope.from_key("k").model(model_id))
    if effort is not None:
        m._settings = ModelSettings(capability=m.capability, thinking_effort=effort)
    return m


def test_dashscope_transform_body_maps_effort_to_enable_thinking() -> None:
    m = _model("qwen3-32b", "low")
    # The base's ``reasoning_effort`` (which DashScope rejects) is dropped.
    body: MutableJSON = {"model": "qwen3-32b", "reasoning_effort": "low"}
    out = m._transform_body(body, ModelRequest(messages=[]))
    assert "reasoning_effort" not in out
    assert out["enable_thinking"] is True
    # The level drives a reasoning-token budget, not just an on/off flag.
    assert out["thinking_budget"] == 4_096


def test_dashscope_transform_body_effort_levels_map_distinct_budgets() -> None:
    high = _model("qwen3-32b", "high")._transform_body({}, ModelRequest(messages=[]))
    maxi = _model("qwen3-32b", "max")._transform_body({}, ModelRequest(messages=[]))
    assert high["thinking_budget"] == 16_384
    assert maxi["thinking_budget"] == 24_576


def test_dashscope_transform_body_none_effort_means_disabled() -> None:
    # ``none`` is the catalog's zero budget; Qwen spells it as a toggle, so no
    # ``thinking_budget`` accompanies it.
    out = _model("qwen3-32b")._transform_body(
        {"reasoning_effort": "minimal"}, ModelRequest(messages=[])
    )
    assert out["enable_thinking"] is False
    assert "thinking_budget" not in out


def test_dashscope_thinking_suffix_model_never_disables_thinking() -> None:
    """``*-thinking-2507`` is thinking-only: ``enable_thinking`` stays True.

    Model Studio rejects ``enable_thinking=false`` on these ids, so the row
    withholds ``none`` and the wire never sends the toggle down.
    """
    m = _model("qwen3-235b-a22b-thinking-2507", "low")
    assert "none" not in m.capability.thinking_effort
    out = m._transform_body({}, ModelRequest(messages=[]))
    assert out["enable_thinking"] is True
    assert out["thinking_budget"] == 4_096


def test_dashscope_a_thinking_only_row_cannot_select_none() -> None:
    """The capability rejects it up front rather than 400ing on the wire."""
    m = _model("qwen3-235b-a22b-thinking-2507")
    with pytest.raises(ValueError, match="thinking_effort"):
        m.settings.thinking_effort = "none"


@pytest.mark.parametrize("model_id", ["qwen-turbo", "qwen3-235b-a22b-instruct-2507"])
def test_dashscope_non_reasoning_model_gets_no_thinking_knobs(model_id: str) -> None:
    """A row offering only ``none`` claims the model REJECTS the knob."""
    m = _model(model_id)
    assert m.capability.thinking_effort == frozenset({"none"})
    out = m._transform_body({}, ModelRequest(messages=[]))
    assert "enable_thinking" not in out
    assert "thinking_budget" not in out


def test_dashscope_default_model_supports_effort_end_to_end() -> None:
    """Effort must be selectable on the DEFAULT model.

    Otherwise the module docstring's effort->enable_thinking promise is
    unreachable for default use.
    """
    m = DashScope.from_key("k").model()
    assert m.capability.thinking_effort != frozenset({"none"})
    agent = Agent(model=m)
    agent.model.settings.thinking_effort = "medium"
    assert m.settings.thinking_effort == "medium"


def test_dashscope_transform_body_no_effort_unchanged() -> None:
    m = _model("qwen3-32b")
    body: MutableJSON = {"model": "qwen3-32b"}
    out = m._transform_body(body, ModelRequest(messages=[]))
    assert out["enable_thinking"] is False
    assert "thinking_budget" not in out
    assert out["model"] == "qwen3-32b"


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
