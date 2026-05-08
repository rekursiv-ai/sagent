"""Tests for BackgroundTask tool and background dispatch."""

from __future__ import annotations

from collections.abc import Callable

import asyncio

import pytest

from sagent.agent.agent import Agent
from sagent.custom_types import (
    JsonMessage,
    Message,
    ModelRequest,
    ModelResponse as _ModelResponse,
    MultipartMessage,
    TextMessage,
    TokenCount,
)
from sagent.lib.json import JSON, json_freeze
from sagent.lib.message import get_directive, tool_call_message
from sagent.testing import MockModelCaps
from sagent.tools.background_task import (
    BackgroundTask,
    BackgroundTaskEntry,
)
from sagent.tools.core import current_agent_var


# -- Factories ---------------------------------------------------------


def ModelResponse(  # noqa: N802 -- PascalCase factory mimics Message constructor
    *,
    text: str = "",
    tool_calls: list[Message] | None = None,
    stop_reason: str = "model_finished",
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> _ModelResponse:
    parts: list[Message] = []
    if text:
        parts.append(TextMessage(text, "text/plain"))
    parts.extend(tool_calls or [])
    content = MultipartMessage(tuple(parts), "multipart/x-model-message")
    return _ModelResponse(
        content=content,
        stop_reason=stop_reason,
        tokens=TokenCount(input_tokens=input_tokens, output_tokens=output_tokens),
    )


class _MockCaps(MockModelCaps):
    max_image_dim: int = 2000


class _MockModel(_MockCaps):
    def __init__(self, responses: list[_ModelResponse]) -> None:
        self._responses = responses
        self._call_idx = 0
        self.requests: list[ModelRequest] = []

    @property
    def max_request_tokens(self) -> int:
        return 100_000

    @property
    def model_id(self) -> str:
        return "mock"

    async def buffer(self, request: ModelRequest) -> _ModelResponse:
        self.requests.append(request)
        resp = self._responses[min(self._call_idx, len(self._responses) - 1)]
        self._call_idx += 1
        return resp

    async def stream(
        self,
        request: ModelRequest,
        on_text: Callable[[str], None] | None = None,
        on_thinking: Callable[[str], None] | None = None,
    ) -> _ModelResponse:
        del on_text, on_thinking
        return await self.buffer(request=request)


class _MockTool:
    name = "echo"
    tool_id = "application/x-tool-echo"
    description = "Echoes input."
    supports_microcompaction = False
    directive_schema: JSON = json_freeze(
        {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        }
    )

    def summary(self, msg: Message) -> str:
        del msg
        return self.name

    def prompt(self) -> str:
        return ""

    async def run(self, msg: Message) -> Message:
        directive = get_directive(msg)
        return TextMessage(str(directive.get("text", "")), "text/plain")


class _SlowTool:
    name = "slow"
    tool_id = "application/x-tool-slow"
    description = "Sleeps then returns."
    supports_microcompaction = False
    directive_schema: JSON = json_freeze(
        {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        }
    )

    def summary(self, msg: Message) -> str:
        del msg
        return self.name

    def prompt(self) -> str:
        return ""

    async def run(self, msg: Message) -> Message:
        directive = get_directive(msg)
        await asyncio.sleep(0.05)
        return TextMessage(str(directive.get("text", "")), "text/plain")


# -- Tests: background dispatch ----------------------------------------


class TestBackgroundDispatch:
    @pytest.mark.anyio
    async def test_background_tool_returns_placeholder(self) -> None:
        model = _MockModel(
            responses=[
                ModelResponse(
                    tool_calls=[
                        tool_call_message(
                            "t1",
                            "echo",
                            json_freeze({"text": "hello", "background": True}),
                        ),
                    ],
                    stop_reason="model_tool_use",
                    input_tokens=10,
                    output_tokens=5,
                ),
                ModelResponse(text="done", input_tokens=20, output_tokens=5),
            ],
        )
        agent = Agent(
            name="test",
            model=model,
            tools=[_MockTool(), BackgroundTask()],
        )
        await agent.run(json_freeze({"prompt": "go"}))
        # Placeholder was appended to messages.
        placeholders = [
            m
            for m in agent.messages
            if m.descriptor == "multipart/x-tool-result"
            and "Running in background" in str(m.content)
        ]
        assert len(placeholders) == 1

    @pytest.mark.anyio
    async def test_background_result_lands_in_inbox(self) -> None:
        model = _MockModel(
            responses=[
                ModelResponse(
                    tool_calls=[
                        tool_call_message(
                            "t1",
                            "slow",
                            json_freeze({"text": "bg-result", "background": True}),
                        ),
                    ],
                    stop_reason="model_tool_use",
                    input_tokens=10,
                    output_tokens=5,
                ),
                ModelResponse(text="done", input_tokens=20, output_tokens=5),
            ],
        )
        agent = Agent(
            name="test",
            model=model,
            tools=[_SlowTool(), BackgroundTask()],
        )
        await agent.run(json_freeze({"prompt": "go"}))
        # Wait for the background task to finish (may already have).
        job = agent.background_tasks.get("t1")
        if job is not None:
            await job.task
        # The bg post becomes a text/x-user-message either still in
        # the inbox (if posted after run exit) or already in history
        # (if processed by the dispatch loop). Either is correct for
        # the v2 spine.
        sources = [*agent.inbox.drain(), *agent.messages]
        assert any("bg-result" in str(item.content) for item in sources)

    @pytest.mark.anyio
    async def test_delay_implies_background(self) -> None:
        model = _MockModel(
            responses=[
                ModelResponse(
                    tool_calls=[
                        tool_call_message(
                            "t1", "echo", json_freeze({"text": "delayed", "delay": 1})
                        ),
                    ],
                    stop_reason="model_tool_use",
                    input_tokens=10,
                    output_tokens=5,
                ),
                ModelResponse(text="done", input_tokens=20, output_tokens=5),
            ],
        )
        agent = Agent(
            name="test",
            model=model,
            tools=[_MockTool(), BackgroundTask()],
        )
        await agent.run(json_freeze({"prompt": "go"}))
        # delay=0 still backgrounds (delay implies background).
        placeholders = [
            m for m in agent.messages if "Running in background" in str(m.content)
        ]
        assert len(placeholders) == 1

    @pytest.mark.anyio
    async def test_foreground_tool_unchanged(self) -> None:
        model = _MockModel(
            responses=[
                ModelResponse(
                    tool_calls=[
                        tool_call_message("t1", "echo", json_freeze({"text": "sync"})),
                    ],
                    stop_reason="model_tool_use",
                    input_tokens=10,
                    output_tokens=5,
                ),
                ModelResponse(text="done", input_tokens=20, output_tokens=5),
            ],
        )
        agent = Agent(
            name="test",
            model=model,
            tools=[_MockTool(), BackgroundTask()],
        )
        result = await agent.run(json_freeze({"prompt": "go"}))
        assert str(result.content) == "done"
        # No placeholders -- tool ran synchronously.
        placeholders = [
            m for m in agent.messages if "Running in background" in str(m.content)
        ]
        assert len(placeholders) == 0


# -- Tests: BackgroundTask tool ----------------------------------------


class TestBackgroundTaskTool:
    @pytest.mark.anyio
    async def test_list_empty(self) -> None:
        model = _MockModel([ModelResponse(text="hi")])
        agent = Agent(name="test", model=model, tools=[BackgroundTask()])
        token = current_agent_var.set(agent)
        try:
            tool = BackgroundTask()
            msg = MultipartMessage(
                (
                    TextMessage("q1", "text/x-queue-id"),
                    JsonMessage(
                        json_freeze({"operation": "list"}),
                        "application/x-tool-backgroundtask",
                    ),
                ),
                "multipart/x-tool-call",
            )
            result = await tool.run(msg)
            assert "No background tasks" in str(result.content)
        finally:
            current_agent_var.reset(token)

    @pytest.mark.anyio
    async def test_cancel(self) -> None:
        model = _MockModel([ModelResponse(text="hi")])
        agent = Agent(name="test", model=model, tools=[BackgroundTask()])

        # Create a fake background job.
        async def _long_wait() -> Message:
            await asyncio.sleep(100)
            return TextMessage("", "text/plain")

        task = asyncio.create_task(_long_wait())
        agent.background_tasks["j1"] = BackgroundTaskEntry(
            task=task,
            tool_name="slow",
            queue_id="j1",
            started=0.0,
        )
        token = current_agent_var.set(agent)
        try:
            tool = BackgroundTask()
            msg = MultipartMessage(
                (
                    TextMessage("q1", "text/x-queue-id"),
                    JsonMessage(
                        json_freeze({"operation": "cancel", "id": "j1"}),
                        "application/x-tool-backgroundtask",
                    ),
                ),
                "multipart/x-tool-call",
            )
            result = await tool.run(msg)
            assert "Cancelled" in str(result.content)
            assert "j1" not in agent.background_tasks
            assert task.cancelling()
        finally:
            current_agent_var.reset(token)

    @pytest.mark.anyio
    async def test_foreground(self) -> None:
        model = _MockModel([ModelResponse(text="hi")])
        agent = Agent(name="test", model=model, tools=[BackgroundTask()])

        async def _delayed_result() -> Message:
            await asyncio.sleep(0.01)
            return TextMessage("the-result", "text/plain")

        task = asyncio.create_task(_delayed_result())
        agent.background_tasks["j1"] = BackgroundTaskEntry(
            task=task,
            tool_name="echo",
            queue_id="j1",
            started=0.0,
        )
        token = current_agent_var.set(agent)
        try:
            tool = BackgroundTask()
            msg = MultipartMessage(
                (
                    TextMessage("q1", "text/x-queue-id"),
                    JsonMessage(
                        json_freeze({"operation": "foreground", "id": "j1"}),
                        "application/x-tool-backgroundtask",
                    ),
                ),
                "multipart/x-tool-call",
            )
            result = await tool.run(msg)
            assert "the-result" in str(result.content)
            assert "j1" not in agent.background_tasks
        finally:
            current_agent_var.reset(token)

    @pytest.mark.anyio
    async def test_cancel_nonexistent(self) -> None:
        model = _MockModel([ModelResponse(text="hi")])
        agent = Agent(name="test", model=model, tools=[BackgroundTask()])
        token = current_agent_var.set(agent)
        try:
            tool = BackgroundTask()
            msg = MultipartMessage(
                (
                    TextMessage("q1", "text/x-queue-id"),
                    JsonMessage(
                        json_freeze({"operation": "cancel", "id": "nope"}),
                        "application/x-tool-backgroundtask",
                    ),
                ),
                "multipart/x-tool-call",
            )
            result = await tool.run(msg)
            assert result.descriptor == "text/x-error"
        finally:
            current_agent_var.reset(token)


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
