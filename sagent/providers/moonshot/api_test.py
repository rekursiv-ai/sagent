"""Tests for ``providers.moonshot``: Moonshot OpenAI-compat surface."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest

from sagent.providers.moonshot.api import Moonshot, _MoonshotModel
from sagent.types.capability import (
    ModelSettings,
    ThinkingBudget,
    ThinkingEffort,
)
from sagent.types.model import ModelRequest
from sagent.types.runtime import UserMessage


if TYPE_CHECKING:
    from sagent.lib.codec import MutablePlainTree


def test_moonshot_from_key() -> None:
    p = Moonshot.from_key("sk-moon-test")
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
    assert m.capability.model_id == p.catalog.resolve("default")[0].model_id
    assert m.capability.thinking.output == frozenset({"none", "text"})


def test_moonshot_unknown_model_raises() -> None:
    p = Moonshot.from_key("k")
    with pytest.raises(ValueError, match="Unknown model"):
        _ = p.model("not-kimi")


def test_moonshot_known_models_have_pricing_and_limits() -> None:
    p = Moonshot.from_key("k")
    for mid in Moonshot.catalog.models:
        m = p.model(mid)
        assert m.limits.max_request_tokens > 0
        assert m.limits.max_response_tokens > 0


def test_moonshot_base_url_override_via_from_key() -> None:
    p = Moonshot.from_key("k", base_url="http://localhost:8000/v1")
    assert p.base_url == "http://localhost:8000/v1"


def test_moonshot_offers_no_prompt_cache() -> None:
    m = Moonshot.from_key("k").model("kimi-2.6")
    assert m.capability.cache_ttl_sec == frozenset({0.0})


def _body(
    model_id: str,
    *,
    budget: ThinkingBudget | None = None,
    effort: ThinkingEffort | None = None,
) -> dict[str, MutablePlainTree]:
    m = cast(_MoonshotModel, Moonshot.from_key("k").model(model_id))
    settings = ModelSettings.narrowest(m.capability)
    if budget is not None:
        settings.thinking_budget = budget
    if effort is not None:
        settings.thinking_effort = effort
    m._settings = settings
    return m._build_body(ModelRequest(messages=[UserMessage(text="x")]), stream=False)


@pytest.mark.parametrize("model_id", sorted(Moonshot.catalog.models))
def test_no_request_carries_the_fixed_temperature(model_id: str) -> None:
    assert "temperature" not in _body(model_id)


def test_k3_sends_max_verbatim_rather_than_openais_high() -> None:
    assert _body("kimi-3.0", effort="max")["reasoning_effort"] == "max"


def test_k3_with_no_effort_leaves_the_server_default() -> None:
    body = _body("kimi-3.0")
    assert "reasoning_effort" not in body
    assert "thinking" not in body


@pytest.mark.parametrize(
    ("budget", "wire"),
    [("none", "disabled"), ("auto", "enabled")],
)
def test_k2_6_sends_its_thinking_switch(budget: ThinkingBudget, wire: str) -> None:
    assert _body("kimi-2.6", budget=budget)["thinking"] == {"type": wire}


def test_k2_7_code_is_never_sent_the_thinking_parameter() -> None:
    assert "thinking" not in _body("kimi-code-2.7")


def test_the_wire_carries_the_vendor_id() -> None:
    assert _body("kimi-code-highspeed-2.7")["model"] == "kimi-k2.7-code-highspeed"


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
