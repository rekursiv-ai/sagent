"""Tests for ``providers.minimax``: MiniMax OpenAI-compat surface."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, cast

import pytest

from sagent.catalog.minimax import api
from sagent.catalog.table import ModelCatalog, ModelTable
from sagent.providers.minimax.api import MiniMax, _MiniMaxModel
from sagent.types.capability import (
    ModelCapability,
    ModelSettings,
    ThinkingBudget,
    ThinkingCapability,
)
from sagent.types.model import ModelRequest


if TYPE_CHECKING:
    from treekle import MutablePlainTree


@pytest.fixture(autouse=True)
def synthetic_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    row = ModelCapability(
        model_id="always-1.0",
        thinking=ThinkingCapability(output=frozenset({"none", "text"})),
    )
    switched = replace(
        row,
        model_id="switch-1.0",
        thinking=ThinkingCapability(
            budget=frozenset({"none", "auto"}),
            output=frozenset({"none", "text"}),
        ),
        service_tier=frozenset({"auto", "priority"}),
    )
    table = ModelTable(rows=(row, switched), roles={"default": "switch"})
    monkeypatch.setattr(MiniMax, "catalog", ModelCatalog(models=table, transport=api()))


def test_minimax_from_key() -> None:
    p = MiniMax.from_key("sk-test")
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
    assert m.capability.model_id == p.catalog.resolve("default")[0].model_id


def test_minimax_unknown_model_raises() -> None:
    p = MiniMax.from_key("k")
    with pytest.raises(ValueError, match="Unknown model"):
        _ = p.model("not-minimax")


def test_minimax_model_supports_thinking_via_reasoning_field() -> None:
    m = MiniMax.from_key("k").model("always-1.0")
    assert m.capability.thinking.output == frozenset({"none", "text"})


def test_minimax_base_url_override_via_from_key() -> None:
    p = MiniMax.from_key("k", base_url="http://localhost:8000/v1")
    assert p.base_url == "http://localhost:8000/v1"


def _body(
    model_id: str,
    budget: ThinkingBudget | None = None,
) -> dict[str, MutablePlainTree]:
    m = cast(_MiniMaxModel, MiniMax.from_key("k").model(model_id))
    settings = ModelSettings.narrowest(m.capability)
    if budget is not None:
        settings.thinking_budget = budget
    m._settings = settings
    return m._transform_body({}, ModelRequest(messages=[]))


def test_every_request_splits_reasoning_out_of_content() -> None:
    """Without it, thinking arrives inside ``content`` as ``<think>`` tags."""
    assert _body("always-1.0")["reasoning_split"] is True


@pytest.mark.parametrize(
    ("budget", "wire"),
    [("none", "disabled"), ("auto", "adaptive")],
)
def test_m3_sends_its_thinking_switch(budget: ThinkingBudget, wire: str) -> None:
    assert _body("switch-1.0", budget)["thinking"] == {"type": wire}


def test_m2_sends_no_thinking_switch_it_would_ignore() -> None:
    assert "thinking" not in _body("always-1.0")


def test_m3_priority_tier_survives_the_transport() -> None:
    m = MiniMax.from_key("k").model("switch-1.0")
    assert "priority" in m.capability.service_tier
    m.settings.service_tier = "priority"


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
