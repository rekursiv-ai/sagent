"""Tests for ``tools.advisor``: ad-hoc sub-agent consultation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, override

import pytest

from sagent.agent.agent import Agent
from sagent.agent.state import current_agent_var
from sagent.testing import MockModelCaps
from sagent.tools.advisor import SYSTEM_NUDGE, Advisor
from sagent.types.cost import TokenCost
from sagent.types.model import (
    ModelRequest,
    ModelResponse,
    StreamInterruptedError,
)
from sagent.types.runtime import (
    AssistantMessage,
    RuntimeEvent,
    UserMessage,
)


if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(slots=True, kw_only=True)
class StubProviderModel(MockModelCaps):
    """Configurable provider-side ``Model`` that returns scripted text."""

    model_id: str = "stub-advisor"
    max_request_tokens: int = 100_000
    received: list[ModelRequest] = field(default_factory=list)
    text: str = "advice"

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
async def test_run_returns_empty_text_verbatim() -> None:
    model = StubProviderModel(text="")
    t = Advisor(model=model)
    result = await t.run({"prompt": "p"})
    assert not result.is_error
    assert result.content == ""


@pytest.mark.asyncio
async def test_run_forwards_prompt_and_system() -> None:
    inner = StubProviderModel(text="bridged")
    result = await Advisor(model=inner, system="be brief").run({"prompt": "q"})
    assert result.content == "bridged"
    assert inner.received[0].system == "be brief"
    (message,) = inner.received[0].messages
    assert isinstance(message, UserMessage)
    assert message.text == "q"


@pytest.mark.asyncio
async def test_run_bills_the_current_agent() -> None:
    """The consult's cost lands on the agent that called the tool."""

    @dataclass(slots=True, kw_only=True)
    class _PricedModel(StubProviderModel):
        @override
        async def stream(
            self,
            request: ModelRequest,
            publish: Callable[[RuntimeEvent], None] | None = None,
        ) -> ModelResponse:
            del publish
            self.received.append(request)
            return ModelResponse(
                message=AssistantMessage(text="advice"),
                spend=TokenCost(request=0.5),
            )

    agent = Agent(model=StubProviderModel())
    agent.runtime.append_history(UserMessage(text="x" * 400_000))
    measured = tuple(agent.runtime.context().messages)
    agent._last_input_tokens = 100_000
    agent._last_measured_history = measured
    token = current_agent_var.set(agent)
    try:
        _ = await Advisor(model=_PricedModel()).run({"prompt": "p"})
    finally:
        current_agent_var.reset(token)

    assert agent.cost_tracker.spend.total == pytest.approx(0.5)
    # The consult must not stand in for the conversation's measured size.
    assert agent._last_input_tokens == 100_000
    assert agent._last_measured_history is measured


@pytest.mark.asyncio
async def test_advisor_reports_refusal_as_error() -> None:
    """A refused consult is an error result, not empty advice."""

    @dataclass(slots=True, kw_only=True)
    class _RefusingModel(StubProviderModel):
        @override
        async def stream(
            self,
            request: ModelRequest,
            publish: Callable[[RuntimeEvent], None] | None = None,
        ) -> ModelResponse:
            del publish
            self.received.append(request)
            return ModelResponse(
                message=AssistantMessage(text=""),
                stop_reason="model_refusal",
            )

    result = await Advisor(model=_RefusingModel()).run({"prompt": "p"})

    assert result.is_error
    assert "model_refusal" in result.content


@pytest.mark.asyncio
async def test_advisor_retries_transient_stream_interruption() -> None:
    """The consult goes through the shared retry path."""

    @dataclass(slots=True, kw_only=True)
    class _FlakyModel(StubProviderModel):
        calls: int = 0

        @override
        async def stream(
            self,
            request: ModelRequest,
            publish: Callable[[RuntimeEvent], None] | None = None,
        ) -> ModelResponse:
            del publish
            self.received.append(request)
            self.calls += 1
            if self.calls == 1:
                raise StreamInterruptedError(
                    ModelResponse(message=AssistantMessage(text="")),
                )
            return ModelResponse(message=AssistantMessage(text="advice"))

    model = _FlakyModel()
    result = await Advisor(model=model).run({"prompt": "p"})

    assert result.content == "advice"
    assert model.calls == 2


@pytest.mark.asyncio
async def test_a_consult_that_exhausts_its_retries_is_a_tool_error() -> None:
    """``Tool.run`` returns a failed consult; it does not raise it."""

    @dataclass(slots=True, kw_only=True)
    class _AlwaysInterrupted(StubProviderModel):
        @override
        async def stream(
            self,
            request: ModelRequest,
            publish: Callable[[RuntimeEvent], None] | None = None,
        ) -> ModelResponse:
            del request, publish
            raise StreamInterruptedError(
                ModelResponse(message=AssistantMessage(text="")),
            )

    result = await Advisor(model=_AlwaysInterrupted()).run({"prompt": "p"})
    assert result.is_error


@pytest.mark.asyncio
async def test_run_blank_system_is_empty() -> None:
    inner = StubProviderModel(text="ok")
    _ = await Advisor(model=inner, system="").run({"prompt": "p"})
    assert inner.received[0].system == ""


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
