"""Tests for the MiniMax request body: split reasoning and M3's thinking switch."""

from __future__ import annotations

import pytest

from sagent.model.exceptions import AuthError
from sagent.model.types import Request
from sagent.providers.catalog import choose
from sagent.providers.minimax.api import MiniMax
from sagent.session.types import UserMessage


_REQUEST = Request(records=(UserMessage(content="hi"),))


def _body(model_id: str, **choices: object) -> dict[str, object]:
    minimax = MiniMax(api_key="k")
    model = minimax[model_id]
    for field, value in choices.items():
        model.settings = choose(model.settings, field, value)
    return minimax.encode(_REQUEST, model=model)


@pytest.mark.asyncio
async def test_the_provider_is_minimaxs_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    minimax = MiniMax()
    assert minimax.name == "minimax"
    assert await minimax.endpoint() == "https://api.minimax.io/v1"
    assert minimax.reasoning_field == "reasoning_content"
    assert minimax["default"].capability.wire_model_id == "MiniMax-M3"
    monkeypatch.setenv("MINIMAX_API_KEY", "sk-mm")
    assert await minimax.credentials() == {"authorization": "Bearer sk-mm"}
    monkeypatch.delenv("MINIMAX_API_KEY")
    with pytest.raises(AuthError, match="MINIMAX_API_KEY"):
        await minimax.credentials()


@pytest.mark.parametrize(
    "model_id",
    ["minimax-3.0", "minimax-2.7", "minimax-highspeed-2.1"],
)
def test_every_request_splits_the_reasoning_out_of_the_answer(model_id: str) -> None:
    assert _body(model_id)["reasoning_split"] is True


@pytest.mark.parametrize(
    ("budget", "kind"),
    [("none", "disabled"), ("auto", "adaptive")],
)
def test_m3_is_sent_its_thinking_switch(budget: str, kind: str) -> None:
    assert _body("minimax-3.0", thinking_budget=budget)["thinking"] == {"type": kind}


def test_m3_starts_thinking_as_its_api_does_unasked() -> None:
    # "MiniMax-M3: adaptive thinking is on by default" (Create Chat Completion).
    assert _body("minimax-3.0")["thinking"] == {"type": "adaptive"}


def test_m2_is_sent_no_switch_it_would_ignore() -> None:
    assert "thinking" not in _body("minimax-2.7")


def test_only_m3_offers_the_priority_tier() -> None:
    assert _body("minimax-3.0", service_tier="priority")["service_tier"] == "priority"
    assert sorted(MiniMax()["minimax-2.7"].capability.service_tier) == ["auto"]


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
