"""Tests for ``tools.advisor``: ad-hoc sub-agent consultation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import override

import pytest

from sagent.testing import MockModelCaps
from sagent.tools.advisor import (
    SYSTEM_NUDGE,
    Advisor,
    _AdvisorModel,
)
from sagent.types.model import ModelRequest, ModelResponse, Pricing
from sagent.types.runtime import (
    AssistantMessage,
    ModelResponsePartial,
    RuntimeEvent,
    ToolResult,
)


@dataclass(slots=True, kw_only=True)
class StubProviderModel(MockModelCaps):
    """Configurable provider-side ``Model`` that returns scripted text."""

    model_id: str = "stub-advisor"
    max_request_tokens: int = 100_000
    received: list[ModelRequest] = field(default_factory=list)
    text: str = "advice"

    @property
    @override
    def pricing(self) -> Pricing:
        return Pricing()

    async def buffer(self, request: ModelRequest) -> ModelResponse:
        return await self.stream(request)

    async def stream(
        self,
        request: ModelRequest,
        publish: Callable[[RuntimeEvent], None] | None = None,
    ) -> ModelResponse:
        del publish
        self.received.append(request)
        return ModelResponse(message=AssistantMessage(text=self.text))


def test_metadata_basics() -> None:
    t = Advisor(model=StubProviderModel())
    assert t.name == "advisor"
    assert t.tool_id == "application/x-tool-advisor"
    assert t.summary({}) == "Advisor consulting…"
    assert t.summary_result(ToolResult(call_id="", content="")) is None


def test_prompt_returns_nudge() -> None:
    t = Advisor(model=StubProviderModel())
    assert t.prompt() == SYSTEM_NUDGE


@pytest.mark.asyncio
async def test_run_consults_model_and_returns_text() -> None:
    model = StubProviderModel(text="Try option B.")
    t = Advisor(model=model)
    result = await t.run({"prompt": "stuck on a thing"})
    assert not result.is_error
    assert result.content == "Try option B."
    # Internal counter advances.
    assert t._uses == 1
    # The stub model received exactly one request.
    assert len(model.received) == 1


@pytest.mark.asyncio
async def test_run_respects_max_uses() -> None:
    model = StubProviderModel(text="x")
    t = Advisor(model=model, max_uses=2)
    a = await t.run({"prompt": "p1"})
    b = await t.run({"prompt": "p2"})
    c = await t.run({"prompt": "p3"})
    assert not a.is_error
    assert not b.is_error
    assert c.is_error
    assert "quota exhausted" in c.content


@pytest.mark.asyncio
async def test_run_returns_empty_on_no_assistant_message() -> None:
    # Empty-text response yields an empty content result. Used to
    # exercise the "no assistant message" fallback indirectly: when
    # the runtime's history *does* contain an assistant message but
    # it's empty, the tool still returns it (content="").
    model = StubProviderModel(text="")
    t = Advisor(model=model)
    result = await t.run({"prompt": "p"})
    assert not result.is_error
    assert result.content == ""


@pytest.mark.asyncio
async def test_advisor_model_bridge_forwards_history_and_system() -> None:
    inner = StubProviderModel(text="bridged")
    bridge = _AdvisorModel(inner, system="be brief")
    captured_text: list[str] = []

    def publish(event: RuntimeEvent) -> None:
        if isinstance(event, ModelResponsePartial):
            captured_text.append(event.text)

    msg = await bridge.stream(history=[], publish=publish)
    assert msg.text == "bridged"
    assert inner.received[0].system == "be brief"


@pytest.mark.asyncio
async def test_advisor_model_bridge_blank_system_is_none() -> None:
    inner = StubProviderModel(text="ok")
    bridge = _AdvisorModel(inner, system="")
    _ = await bridge.stream(history=[], publish=lambda _event: None)
    # Blank system collapses to None at the request level.
    assert inner.received[0].system is None


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
