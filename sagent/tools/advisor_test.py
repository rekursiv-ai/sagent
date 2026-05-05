"""Tests for sagent.advisor."""

from __future__ import annotations

from collections.abc import Callable
from types import MappingProxyType

import pytest

from sagent.custom_types import (
    JsonMessage,
    Message,
    ModelRequest,
    ModelResponse,
    MultipartMessage,
    TextMessage,
    TokenCount,
    Tool,
)
from sagent.lib.json import JSON, json_freeze
from sagent.testing import MockModelCaps
from sagent.tools.advisor import SYSTEM_NUDGE, Advisor


def _msg(directive: JSON) -> Message:
    """Wrap a directive in a minimal tool-call Message."""
    return MultipartMessage(
        (JsonMessage(directive, "application/x-tool-x"),),
        "multipart/x-tool-call",
    )


class _MockCaps(MockModelCaps):
    max_image_dim: int = 2000


class _MockModel(_MockCaps):
    def __init__(self, text: str = "advice") -> None:
        self._text = text
        self.requests: list[ModelRequest] = []

    @property
    def max_request_tokens(self) -> int:
        return 100_000

    @property
    def model_id(self) -> str:
        return "mock-advisor"

    async def buffer(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return ModelResponse(
            content=MultipartMessage(
                (TextMessage(self._text, "text/plain"),),
                "multipart/x-model-message",
            ),
            tokens=TokenCount(input_tokens=1, output_tokens=1),
        )

    async def stream(
        self,
        request: ModelRequest,
        on_text: Callable[[str], None] | None = None,
    ) -> ModelResponse:
        del on_text
        return await self.buffer(request=request)


class TestAdvisorShape:
    def test_conforms_to_tool_protocol(self) -> None:
        model = _MockModel()
        adv = Advisor(model=model)
        assert isinstance(adv, Tool)

    def test_schema_has_prompt(self) -> None:
        properties = Advisor.directive_schema["properties"]
        assert isinstance(properties, MappingProxyType)
        assert "prompt" in properties
        required = Advisor.directive_schema["required"]
        assert isinstance(required, tuple)
        assert list(required) == ["prompt"]

    def test_microcompaction_disabled(self) -> None:
        assert Advisor.supports_microcompaction is False

    def test_prompt_section_returns_system_nudge(self) -> None:
        # The tool self-reports its system-prompt contribution so adding
        # the Advisor to an agent auto-wires the nudge - no separate flag.
        model = _MockModel()
        adv = Advisor(model=model)
        assert adv.prompt() == SYSTEM_NUDGE


class TestSystemNudge:
    def test_mentions_tool_name(self) -> None:
        assert "`advisor`" in SYSTEM_NUDGE

    def test_lists_concrete_triggers(self) -> None:
        # The whole point of the nudge: give the executor conditions
        # to match against, not vague advice.
        assert "fails twice" in SYSTEM_NUDGE
        assert "two approaches" in SYSTEM_NUDGE
        assert "hard-to-reverse" in SYSTEM_NUDGE

    def test_states_self_contained_requirement(self) -> None:
        assert "self-contained" in SYSTEM_NUDGE


def _text(msg: Message) -> str:
    if isinstance(msg, TextMessage):
        return msg.content
    if isinstance(msg, MultipartMessage):
        for p in msg.content:
            if isinstance(p, TextMessage) and p.descriptor == "text/plain":
                return p.content
    return ""


class TestAdvisorCall:
    @pytest.mark.anyio
    async def test_returns_model_text(self) -> None:
        model = _MockModel(text="do this")
        adv = Advisor(model=model)
        resp = await adv.run(_msg(json_freeze({"prompt": "what should I do?"})))
        assert _text(resp) == "do this"

    @pytest.mark.anyio
    async def test_fresh_history_per_call(self) -> None:
        """Each consult starts from empty history."""
        model = _MockModel()
        adv = Advisor(model=model)
        await adv.run(_msg(json_freeze({"prompt": "first"})))
        await adv.run(_msg(json_freeze({"prompt": "second"})))
        # Each call makes exactly one model request (no follow-up).
        assert len(model.requests) == 2
        # Second call's request must not contain the first prompt -
        # otherwise history leaked across consults.
        second = model.requests[1]
        assert len(second.messages) == 1
        user_msg = second.messages[0]
        assert user_msg.descriptor == "text/x-user-message"
        assert user_msg.content == "second"

    @pytest.mark.anyio
    async def test_quota_enforced(self) -> None:
        model = _MockModel()
        adv = Advisor(model=model, max_uses=2)
        r1 = await adv.run(_msg(json_freeze({"prompt": "q1"})))
        r2 = await adv.run(_msg(json_freeze({"prompt": "q2"})))
        assert _text(r1) == "advice"
        assert _text(r2) == "advice"
        # Third call exceeds quota - returns error Message.
        r3 = await adv.run(_msg(json_freeze({"prompt": "q3"})))
        assert r3.descriptor == "text/x-error"
        assert "quota exhausted" in str(r3.content)
        # Third call short-circuits before model invocation.
        assert len(model.requests) == 2

    @pytest.mark.anyio
    async def test_unlimited_by_default(self) -> None:
        model = _MockModel()
        adv = Advisor(model=model)
        for _ in range(5):
            resp = await adv.run(_msg(json_freeze({"prompt": "q"})))
            assert _text(resp) == "advice"
        assert len(model.requests) == 5


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
