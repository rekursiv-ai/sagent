"""Tests for ``tools.AgentSend``."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from sagent.agent import Agent
from sagent.custom_types import (
    JsonMessage,
    Message,
    ModelRequest,
    ModelResponse,
    MultipartMessage,
    TextMessage,
    TokenCount,
)
from sagent.lib.asyncio_collections import Deque
from sagent.lib.json import JSON, json_freeze
from sagent.testing import MockModelCaps
from sagent.tools.agent_send import AgentSend, _deliver
from sagent.tools.core import (
    agent_label_var,
    agent_registry,
)


def _msg(directive: JSON) -> Message:
    return MultipartMessage(
        (JsonMessage(directive, "application/x-tool-agentsend"),),
        "multipart/x-tool-call",
    )


class _FakeAgent:
    def __init__(self, name: str = "fake") -> None:
        self.name = name
        self.inbox: Deque[Message] = Deque()


def _register(name: str) -> _FakeAgent:
    agent = _FakeAgent(name)
    agent_registry[name] = agent
    return agent


class TestAgentSendBasics:
    def test_name_and_schema(self) -> None:
        t = AgentSend()
        assert t.name == "AgentSend"
        assert t.tool_id == "application/x-tool-agentsend"

    def test_help_shows_target_and_preview(self) -> None:
        t = AgentSend()
        label = t.summary(_msg(json_freeze({"to": "Agent_0", "content": "hello"})))
        assert "Agent_0" in label
        assert "hello" in label

    def test_help_truncates_long_content(self) -> None:
        t = AgentSend()
        label = t.summary(_msg(json_freeze({"to": "X", "content": "a" * 100})))
        assert label.endswith("...")

    def test_help_bare_when_no_target(self) -> None:
        t = AgentSend()
        label = t.summary(_msg(json_freeze({})))
        assert label == "AgentSend"


class TestPrompt:
    def test_lists_other_agents(self) -> None:
        t = AgentSend()
        _register("myroot")
        _register("Agent_0")
        token = agent_label_var.set("myroot")
        try:
            p = t.prompt()
            assert "Agent_0" in p
            assert "myroot" not in p
        finally:
            agent_label_var.reset(token)
            agent_registry.clear()

    def test_empty_when_alone(self) -> None:
        t = AgentSend()
        _register("agent")
        token = agent_label_var.set("agent")
        try:
            assert t.prompt() == ""
        finally:
            agent_label_var.reset(token)
            agent_registry.clear()

    def test_empty_when_no_agents(self) -> None:
        t = AgentSend()
        agent_registry.clear()
        assert t.prompt() == ""


class TestRun:
    @pytest.mark.anyio
    async def test_deliver_message(self) -> None:
        t = AgentSend()
        target = _register("Agent_0")
        token = agent_label_var.set("agent")
        try:
            result = await t.run(
                _msg(json_freeze({"to": "Agent_0", "content": "stop editing foo.py"}))
            )
            assert result.descriptor == "text/plain"
            assert "Delivered" in str(result.content)
            drained = target.inbox.drain()
            assert len(drained) == 1
            assert "[from agent]:" in str(drained[0].content)
            assert "stop editing foo.py" in str(drained[0].content)
        finally:
            agent_label_var.reset(token)
            agent_registry.clear()

    @pytest.mark.anyio
    async def test_unknown_target_returns_error(self) -> None:
        t = AgentSend()
        agent_registry.clear()
        result = await t.run(_msg(json_freeze({"to": "Nonexistent", "content": "hi"})))
        assert result.descriptor == "text/x-error"
        assert "Unknown agent" in str(result.content)

    @pytest.mark.anyio
    async def test_missing_to_returns_error(self) -> None:
        t = AgentSend()
        result = await t.run(_msg(json_freeze({"to": "", "content": "hi"})))
        assert result.descriptor == "text/x-error"

    @pytest.mark.anyio
    async def test_missing_content_returns_error(self) -> None:
        t = AgentSend()
        _register("X")
        try:
            result = await t.run(_msg(json_freeze({"to": "X", "content": ""})))
            assert result.descriptor == "text/x-error"
        finally:
            agent_registry.clear()

    @pytest.mark.anyio
    async def test_sender_label_injected(self) -> None:
        t = AgentSend()
        target = _register("Agent1")
        token = agent_label_var.set("Agent_0")
        try:
            await t.run(_msg(json_freeze({"to": "Agent1", "content": "hey"})))
            drained = target.inbox.drain()
            assert "[from Agent_0]:" in str(drained[0].content)
        finally:
            agent_label_var.reset(token)
            agent_registry.clear()

    @pytest.mark.anyio
    async def test_delayed_message_returns_scheduled(self) -> None:
        t = AgentSend()
        target = _register("Agent_0")
        token = agent_label_var.set("agent")
        try:
            result = await t.run(
                _msg(json_freeze({"to": "Agent_0", "content": "wake up", "delay": 60}))
            )
            assert result.descriptor == "text/plain"
            assert "Scheduled" in str(result.content)
            assert target.inbox.empty()
        finally:
            agent_label_var.reset(token)
            agent_registry.clear()

    def test_deliver_callback_formats_message(self) -> None:
        target = _FakeAgent("Agent_0")
        _deliver(target, "agent", "wake up", 60)
        drained = target.inbox.drain()
        assert len(drained) == 1
        assert "[from agent, 60s ago]:" in str(drained[0].content)
        assert "wake up" in str(drained[0].content)

    def test_deliver_callback_dead_agent(self) -> None:
        _deliver(object(), "agent", "hello", 10)

    @pytest.mark.anyio
    async def test_self_send_lands_in_own_inbox(self) -> None:
        t = AgentSend()
        me = _register("agent")
        token = agent_label_var.set("agent")
        try:
            result = await t.run(
                _msg(json_freeze({"to": "agent", "content": "note to self"}))
            )
            assert result.descriptor == "text/plain"
            drained = me.inbox.drain()
            assert len(drained) == 1
            assert "[from agent]:" in str(drained[0].content)
        finally:
            agent_label_var.reset(token)
            agent_registry.clear()


class _MockModel(MockModelCaps):
    max_image_dim: int = 2000

    def __init__(self, responses: list[ModelResponse] | None = None) -> None:
        self._responses = responses or [self._text_response("done")]
        self._idx = 0

    @property
    def max_request_tokens(self) -> int:
        return 100_000

    @property
    def model_id(self) -> str:
        return "mock"

    async def buffer(self, request: ModelRequest) -> ModelResponse:
        del request
        resp = self._responses[min(self._idx, len(self._responses) - 1)]
        self._idx += 1
        return resp

    async def stream(
        self,
        request: ModelRequest,
        on_text: Callable[[str], None] | None = None,
    ) -> ModelResponse:
        del on_text
        return await self.buffer(request)

    @staticmethod
    def _text_response(text: str) -> ModelResponse:
        return ModelResponse(
            content=MultipartMessage(
                (TextMessage(text, "text/plain"),),
                "multipart/x-model-message",
            ),
            tokens=TokenCount(input_tokens=10, output_tokens=5),
        )


class TestAgentRegistration:
    @pytest.mark.anyio
    async def test_agent_registers_and_deregisters(self) -> None:
        model = _MockModel()
        agent = Agent(model=model, system="test", name="root")
        assert "root" not in agent_registry
        await agent.run(json_freeze({"prompt": "hi"}))
        assert "root" not in agent_registry

    @pytest.mark.anyio
    async def test_agent_visible_during_run(self) -> None:
        captured: list[bool] = []

        class _CaptureTool:
            name = "Capture"
            tool_id = "application/x-tool-capture"
            description = "."
            directive_schema = json_freeze(
                {"type": "object", "properties": {}, "required": []}
            )
            supports_microcompaction = False

            def summary(self, msg: Message) -> str:
                del msg
                return self.name

            def prompt(self) -> str:
                return ""

            async def run(self, msg: Message) -> Message:
                del msg
                captured.append("myagent" in agent_registry)
                return TextMessage("ok", "text/plain")

        capture = _CaptureTool()
        tc = MultipartMessage(
            (
                TextMessage("tc1", "text/x-queue-id"),
                JsonMessage(
                    json_freeze({}),
                    "application/x-tool-capture",
                ),
            ),
            "multipart/x-tool-call",
        )
        model = _MockModel(
            [
                ModelResponse(
                    content=MultipartMessage(
                        (
                            TextMessage("", "text/plain"),
                            tc,
                        ),
                        "multipart/x-model-message",
                    ),
                    stop_reason="model_tool_use",
                    tokens=TokenCount(input_tokens=10, output_tokens=5),
                ),
                _MockModel._text_response("final"),
            ]
        )
        agent = Agent(model=model, system="test", name="myagent", tools=[capture])
        await agent.run(json_freeze({"prompt": "go"}))
        assert captured == [True]


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
