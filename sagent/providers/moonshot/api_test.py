"""Tests for the Moonshot request body: Kimi's fixed sampler and thinking knobs."""

from __future__ import annotations

import pytest

from sagent.model.exceptions import AuthError
from sagent.model.types import Request
from sagent.providers.catalog import choose
from sagent.providers.moonshot.api import Moonshot
from sagent.session.types import UserMessage


_REQUEST = Request(records=(UserMessage(content="hi"),))


def _body(model_id: str, **choices: object) -> dict[str, object]:
    moonshot = Moonshot(api_key="k")
    model = moonshot[model_id]
    for field, value in choices.items():
        model.settings = choose(model.settings, field, value)
    return moonshot.encode(_REQUEST, model=model)


@pytest.mark.asyncio
async def test_the_provider_is_moonshots_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    moonshot = Moonshot()
    assert moonshot.name == "moonshot"
    assert await moonshot.endpoint() == "https://api.moonshot.ai/v1"
    assert await Moonshot(base_url="http://box/v1").endpoint() == "http://box/v1"
    assert moonshot.reasoning_field == "reasoning_content"
    assert moonshot["default"].capability.wire_model_id == "kimi-k3"
    monkeypatch.setenv("MOONSHOT_API_KEY", "sk-ms")
    assert await moonshot.credentials() == {"authorization": "Bearer sk-ms"}
    monkeypatch.delenv("MOONSHOT_API_KEY")
    with pytest.raises(AuthError, match="MOONSHOT_API_KEY"):
        await moonshot.credentials()


@pytest.mark.parametrize("model_id", ["kimi-3.0", "kimi-2.6", "kimi-code-2.7"])
def test_no_request_carries_the_fixed_temperature(model_id: str) -> None:
    body = _body(model_id)
    assert "temperature" not in body
    assert body["model"] == Moonshot()[model_id].capability.wire_model_id


@pytest.mark.parametrize("effort", ["low", "high", "max"])
def test_k3_takes_its_effort_as_named(effort: str) -> None:
    # OpenAI's spelling would fold ``max`` into ``high``.
    body = _body("kimi-3.0", thinking_effort=effort)
    assert body["reasoning_effort"] == effort
    assert "thinking" not in body


def test_k3_starts_at_the_catalogs_effort() -> None:
    assert _body("kimi-3.0")["reasoning_effort"] == "high"


@pytest.mark.parametrize(
    ("budget", "kind"),
    [("none", "disabled"), ("auto", "enabled")],
)
def test_k2_6_is_sent_its_thinking_switch(budget: str, kind: str) -> None:
    body = _body("kimi-2.6", thinking_budget=budget)
    assert body["thinking"] == {"type": kind}
    assert "reasoning_effort" not in body


def test_k2_6_starts_thinking_as_its_api_does_unasked() -> None:
    # ``thinking.type`` defaults to ``enabled`` (Kimi API, "Thinking Models").
    assert _body("kimi-2.6")["thinking"] == {"type": "enabled"}


def test_k2_7_code_is_never_sent_the_thinking_parameter() -> None:
    assert "thinking" not in _body("kimi-code-2.7")


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
