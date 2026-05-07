"""Tests for agent.dispatch functions."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any, Literal, cast

import asyncio
import time

import pytest

from sagent.agent import ERROR_MAX_TOOL_CALL_ROUNDS, Agent
from sagent.agent.dispatch import _drain_stream
from sagent.custom_types import (
    BytesDescriptor,
    BytesMessage,
    JsonMessage,
    Message,
    MessageContent,
    ModelRequest,
    ModelResponse as _ModelResponse,
    MultipartDescriptor,
    MultipartMessage,
    StreamingTool,
    TextDescriptor,
    TextMessage,
    TokenCount,
)
from sagent.lib.json import JSON, json_freeze
from sagent.lib.message import get_directive, tool_call_message
from sagent.testing import MockModelCaps


# -- Compatibility factories -------------------------------------------


def UserMessage(content: str) -> Message:  # noqa: N802 -- PascalCase factory mimics Message constructor
    return TextMessage(content, "text/x-user-message")


def AssistantMessage(  # noqa: N802 -- PascalCase factory mimics Message constructor
    content: str = "",
    tool_calls: list[Message] | None = None,
    message_id: str = "",
) -> Message:
    parts: list[Message] = []
    if message_id:
        parts.append(TextMessage(message_id, "text/x-queue-id"))
    if content:
        parts.append(TextMessage(content, "text/plain"))
    parts.extend(tool_calls or [])
    return MultipartMessage(tuple(parts), "multipart/x-model-message")


def _retag(p: Message, descriptor: str) -> Message:
    if isinstance(p, TextMessage):
        return TextMessage(p.content, cast(TextDescriptor, descriptor))
    if isinstance(p, MultipartMessage):
        return MultipartMessage(p.content, cast(MultipartDescriptor, descriptor))
    if isinstance(p, BytesMessage):
        return BytesMessage(p.content, cast(BytesDescriptor, descriptor))
    return JsonMessage(p.content, descriptor)


def ToolResult(  # noqa: N802 -- PascalCase factory mimics Message constructor
    *,
    queue_id: str,
    name: str,
    content: tuple[Message, ...],
    is_error: bool = False,
) -> Message:
    del name
    if is_error:
        content = tuple(
            _retag(p, "text/x-error" if p.descriptor == "text/plain" else p.descriptor)
            for p in content
        )
    return MultipartMessage(
        (TextMessage(queue_id, "text/x-queue-id"), *content),
        "multipart/x-tool-result",
    )


def Media(content: MessageContent, descriptor: str) -> Message:  # noqa: N802 -- PascalCase factory mimics Message constructor
    if isinstance(content, str):
        return TextMessage(content, cast(TextDescriptor, descriptor))
    if isinstance(content, tuple):
        return MultipartMessage(
            cast(tuple[Message, ...], content),  # pyright: ignore[reportUnnecessaryCast] -- ty needs the cast; pyright considers it redundant after isinstance
            cast(MultipartDescriptor, descriptor),
        )
    if isinstance(content, bytes):
        return BytesMessage(content, cast(BytesDescriptor, descriptor))
    return JsonMessage(content, descriptor)


def ModelResponse(  # noqa: N802 -- PascalCase factory mimics Message constructor
    *,
    text: str = "",
    tool_calls: list[Message] | None = None,
    stop_reason: str = "model_finished",
    input_tokens: int = 0,
    output_tokens: int = 0,
    message_id: str = "",
    thinking: str | None = None,
    cache_creation_input_tokens: int = 0,
    cache_read_input_tokens: int = 0,
    total_cost: float = 0.0,
) -> _ModelResponse:
    parts: list[Message] = []
    if message_id:
        parts.append(TextMessage(message_id, "text/x-queue-id"))
    if text:
        parts.append(TextMessage(text, "text/plain"))
    if thinking:
        parts.append(TextMessage(thinking, "text/x-thinking"))
    parts.extend(tool_calls or [])
    content = MultipartMessage(tuple(parts), "multipart/x-model-message")
    return _ModelResponse(
        content=content,
        stop_reason=stop_reason,
        tokens=TokenCount(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_creation_tokens=cache_creation_input_tokens,
            cache_read_tokens=cache_read_input_tokens,
        ),
        message_id=message_id,
        total_cost=total_cost,
    )


def _is_error_result(m: Message) -> bool:
    return (
        m.descriptor == "multipart/x-tool-result"
        and isinstance(m, MultipartMessage)
        and any(p.descriptor == "text/x-error" for p in m.content)
    )


def _tool_result_text(m: Message) -> str:
    if isinstance(m, MultipartMessage):
        return next(
            (
                str(p.content)
                for p in m.content
                if p.descriptor in ("text/plain", "text/x-error")
            ),
            "",
        )
    return ""


# -- Mock model --------------------------------------------------------


class _MockCaps(MockModelCaps):
    max_image_dim: int = 2000


class _MockModel(_MockCaps):
    def __init__(
        self,
        responses: list[_ModelResponse] | None = None,
    ) -> None:
        self._responses = responses or [
            ModelResponse(text="Hello!", input_tokens=10, output_tokens=5),
        ]
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
        on_text: Any = None,
    ) -> _ModelResponse:
        del on_text
        return await self.buffer(request=request)


# -- Mock tools --------------------------------------------------------


class _MockTool:
    def __init__(self) -> None:
        self.name = "echo"
        self.tool_id = "application/x-tool-echo"
        self.description = "Echoes input."
        self.supports_microcompaction = False
        self.directive_schema: JSON = json_freeze(
            {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                },
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
        text = directive.get("text", "")
        return TextMessage(str(text), "text/plain")


class _StrictTool:
    def __init__(self) -> None:
        self.name = "strict"
        self.tool_id = "application/x-tool-strict"
        self.description = "requires a value arg"
        self.supports_microcompaction = False
        self.directive_schema: JSON = json_freeze(
            {
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
                "additionalProperties": False,
            }
        )

    def summary(self, msg: Message) -> str:
        del msg
        return "Strict normal summary"

    def prompt(self) -> str:
        return ""

    async def run(self, msg: Message) -> Message:
        directive = get_directive(msg)
        value = str(directive.get("value", ""))
        return TextMessage(value, "text/plain")


class _SleepTool:
    """Tool that sleeps and records start/end timestamps."""

    def __init__(self, name: str, sleep_s: float = 0.1) -> None:
        self.name = name
        self.tool_id = f"application/x-tool-{name.lower()}"
        self.description = "sleep tool"
        self.supports_microcompaction = False
        self.directive_schema: JSON = json_freeze({"type": "object", "properties": {}})
        self.sleep_s = sleep_s
        self.starts: list[float] = []
        self.ends: list[float] = []

    def summary(self, msg: Message) -> str:
        del msg
        return self.name

    def prompt(self) -> str:
        return ""

    async def run(self, msg: Message) -> Message:
        del msg
        self.starts.append(time.monotonic())
        await asyncio.sleep(self.sleep_s)
        self.ends.append(time.monotonic())
        return TextMessage("ok", "text/plain")


# -- Tool dispatch tests -----------------------------------------------


class TestToolDispatch:
    @pytest.mark.anyio
    async def test_tool_call_and_response(self) -> None:
        model = _MockModel(
            responses=[
                ModelResponse(
                    text="Calling echo.",
                    tool_calls=[
                        tool_call_message("t1", "echo", json_freeze({"text": "hello"})),
                    ],
                    stop_reason="model_tool_use",
                    input_tokens=10,
                    output_tokens=5,
                ),
                ModelResponse(
                    text="Echo said: hello",
                    input_tokens=20,
                    output_tokens=10,
                ),
            ],
        )
        agent = Agent(
            name="test",
            description="Test agent.",
            model=model,
            tools=[_MockTool()],
        )
        response = await agent.run(json_freeze({"prompt": "echo hello"}))
        assert str(response.content) == "Echo said: hello"
        assert len(model.requests) == 2

    @pytest.mark.anyio
    async def test_unknown_tool(self) -> None:
        model = _MockModel(
            responses=[
                ModelResponse(
                    tool_calls=[
                        tool_call_message("t1", "nonexistent", json_freeze({})),
                    ],
                    stop_reason="model_tool_use",
                    input_tokens=10,
                    output_tokens=5,
                ),
                ModelResponse(
                    text="Tool not found.",
                    input_tokens=20,
                    output_tokens=10,
                ),
            ],
        )
        agent = Agent(
            name="test",
            description="Test agent.",
            model=model,
        )
        await agent.run(json_freeze({"prompt": "call missing tool"}))
        second_request = model.requests[1]
        tool_results = [
            m
            for m in second_request.messages
            if m.descriptor == "multipart/x-tool-result"
        ]
        assert any(_is_error_result(r) for r in tool_results)

    @pytest.mark.anyio
    async def test_malformed_tool_input_becomes_error(self) -> None:
        """Missing required kwargs must not kill the request.

        Regression: Python's arg-binding ``TypeError`` (e.g. when the
        model omits ``command`` for Bash) used to escape the dispatcher
        and terminate the session mid-request.
        """
        model = _MockModel(
            responses=[
                ModelResponse(
                    tool_calls=[
                        tool_call_message("t1", "strict", json_freeze({})),
                    ],
                    stop_reason="model_tool_use",
                    input_tokens=10,
                    output_tokens=5,
                ),
                ModelResponse(
                    text="Recovered.",
                    input_tokens=20,
                    output_tokens=10,
                ),
            ],
        )
        agent = Agent(
            name="test",
            description="Test agent.",
            model=model,
            tools=[_StrictTool()],
        )
        response = await agent.run(
            json_freeze({"prompt": "call strict with empty input"})
        )
        assert str(response.content) == "Recovered."
        second_request = model.requests[1]
        tool_results = [
            m
            for m in second_request.messages
            if m.descriptor == "multipart/x-tool-result"
        ]
        assert len(tool_results) == 1
        assert _is_error_result(tool_results[0])
        result_text = (
            next(
                (
                    str(p.content)
                    for p in tool_results[0].content
                    if p.descriptor == "text/x-error"
                ),
                "",
            )
            if isinstance(tool_results[0], MultipartMessage)
            else ""
        )
        assert "InputValidationError" in result_text
        assert "`value` is missing" in result_text
        assert "This tool call was not executed" in result_text
        assert "Do not repeat the same empty or incomplete call" in result_text

    @pytest.mark.anyio
    async def test_malformed_tool_label_shows_missing_keys(self) -> None:
        model = _MockModel(
            responses=[
                ModelResponse(
                    tool_calls=[
                        tool_call_message("t1", "strict", json_freeze({})),
                    ],
                    stop_reason="model_tool_use",
                    input_tokens=10,
                    output_tokens=5,
                ),
                ModelResponse(text="Recovered.", input_tokens=20, output_tokens=10),
            ],
        )
        agent = Agent(
            name="test",
            description="Test agent.",
            model=model,
            tools=[_StrictTool()],
        )

        events = [event async for event in agent.run(json_freeze({"prompt": "call"}))]

        labels = [
            str(e.content)
            for e in events
            if isinstance(e, TextMessage) and e.descriptor == "text/x-tool-label"
        ]
        assert "Invalid strict call: missing `value`" in labels
        assert "Strict normal summary" not in labels

    @pytest.mark.anyio
    async def test_multiple_malformed_tool_calls_get_batch_hint(self) -> None:
        model = _MockModel(
            responses=[
                ModelResponse(
                    tool_calls=[
                        tool_call_message("t1", "strict", json_freeze({})),
                        tool_call_message("t2", "strict", json_freeze({})),
                    ],
                    stop_reason="model_tool_use",
                    input_tokens=10,
                    output_tokens=5,
                ),
                ModelResponse(text="Recovered.", input_tokens=20, output_tokens=10),
            ],
        )
        agent = Agent(
            name="test",
            description="Test agent.",
            model=model,
            tools=[_StrictTool()],
        )

        await agent.run(json_freeze({"prompt": "call strict twice"}))

        second_request = model.requests[1]
        result_text = "\n".join(
            _tool_result_text(m)
            for m in second_request.messages
            if m.descriptor == "multipart/x-tool-result"
        )
        assert (
            "Multiple tool calls in the previous response were malformed" in result_text
        )
        assert "continue with only valid calls" in result_text

    @pytest.mark.anyio
    async def test_unexpected_tool_input_key_becomes_validation_error(self) -> None:
        model = _MockModel(
            responses=[
                ModelResponse(
                    tool_calls=[
                        tool_call_message(
                            "t1",
                            "strict",
                            json_freeze({"value": "ok", "extra": "bad"}),
                        ),
                    ],
                    stop_reason="model_tool_use",
                    input_tokens=10,
                    output_tokens=5,
                ),
                ModelResponse(text="Recovered.", input_tokens=20, output_tokens=10),
            ],
        )
        agent = Agent(
            name="test",
            description="Test agent.",
            model=model,
            tools=[_StrictTool()],
        )

        await agent.run(json_freeze({"prompt": "call strict with extra"}))

        second_request = model.requests[1]
        result_text = "\n".join(
            _tool_result_text(m)
            for m in second_request.messages
            if m.descriptor == "multipart/x-tool-result"
        )
        assert "InputValidationError" in result_text
        assert "Unexpected parameter `extra`" in result_text
        assert "strict accepts: `value`" in result_text

    @pytest.mark.anyio
    async def test_max_tool_call_rounds(self) -> None:
        model = _MockModel(
            responses=[
                ModelResponse(
                    tool_calls=[
                        tool_call_message("t1", "echo", json_freeze({"text": "loop"})),
                    ],
                    stop_reason="model_tool_use",
                    input_tokens=10,
                    output_tokens=5,
                ),
            ],
        )
        agent = Agent(
            name="test",
            description="Test agent.",
            model=model,
            tools=[_MockTool()],
            max_tool_call_rounds=3,
        )
        result = await agent.run(json_freeze({"prompt": "infinite loop"}))
        assert result.descriptor == "text/x-error"
        assert ERROR_MAX_TOOL_CALL_ROUNDS in str(result.content)


# -- Parallel tool dispatch --------------------------------------------


class TestParallelDispatch:
    """Verify agent batches read-only tool calls into a single gather."""

    @pytest.mark.anyio
    @pytest.mark.real_sleep
    async def test_read_only_tools_run_concurrently(self) -> None:
        tool = _SleepTool("Read", sleep_s=0.02)
        model = _MockModel(
            responses=[
                ModelResponse(
                    tool_calls=[
                        tool_call_message(f"t{i}", "Read", json_freeze({}))
                        for i in range(3)
                    ],
                    stop_reason="model_tool_use",
                    input_tokens=10,
                    output_tokens=5,
                ),
                ModelResponse(text="done", input_tokens=20, output_tokens=10),
            ],
        )
        agent = Agent(name="t", description="t", model=model, tools=[tool])
        t0 = time.monotonic()
        await agent.run(json_freeze({"prompt": "go"}))
        elapsed = time.monotonic() - t0
        assert elapsed < 0.15, f"expected concurrent dispatch, took {elapsed:.3f}s"
        assert len(tool.starts) == 3
        assert max(tool.starts) < min(tool.ends)

    @pytest.mark.anyio
    @pytest.mark.real_sleep
    async def test_bash_safe_commands_run_concurrently(self) -> None:
        tool = _SleepTool("Bash", sleep_s=0.02)
        model = _MockModel(
            responses=[
                ModelResponse(
                    tool_calls=[
                        tool_call_message(
                            "t1", "Bash", json_freeze({"command": "git status"})
                        ),
                        tool_call_message(
                            "t2", "Bash", json_freeze({"command": "ls -la"})
                        ),
                        tool_call_message(
                            "t3", "Bash", json_freeze({"command": "pwd"})
                        ),
                    ],
                    stop_reason="model_tool_use",
                    input_tokens=10,
                    output_tokens=5,
                ),
                ModelResponse(text="done", input_tokens=20, output_tokens=10),
            ],
        )
        agent = Agent(name="t", description="t", model=model, tools=[tool])
        t0 = time.monotonic()
        await agent.run(json_freeze({"prompt": "go"}))
        elapsed = time.monotonic() - t0
        assert elapsed < 0.15, f"expected concurrent Bash dispatch, took {elapsed:.3f}s"
        assert max(tool.starts) < min(tool.ends)

    @pytest.mark.anyio
    @pytest.mark.real_sleep
    async def test_bash_unsafe_commands_serialize(self) -> None:
        tool = _SleepTool("Bash", sleep_s=0.02)
        model = _MockModel(
            responses=[
                ModelResponse(
                    tool_calls=[
                        tool_call_message(
                            "t1", "Bash", json_freeze({"command": "rm foo"})
                        ),
                        tool_call_message(
                            "t2", "Bash", json_freeze({"command": "rm bar"})
                        ),
                    ],
                    stop_reason="model_tool_use",
                    input_tokens=10,
                    output_tokens=5,
                ),
                ModelResponse(text="done", input_tokens=20, output_tokens=10),
            ],
        )
        agent = Agent(name="t", description="t", model=model, tools=[tool])
        t0 = time.monotonic()
        await agent.run(json_freeze({"prompt": "go"}))
        elapsed = time.monotonic() - t0
        assert elapsed >= 0.035, f"expected serial dispatch, took {elapsed:.3f}s"
        assert tool.starts[1] >= tool.ends[0]

    @pytest.mark.anyio
    @pytest.mark.real_sleep
    async def test_safe_unsafe_boundary_splits_batches(self) -> None:
        tool = _SleepTool("Bash", sleep_s=0.02)
        model = _MockModel(
            responses=[
                ModelResponse(
                    tool_calls=[
                        tool_call_message("t1", "Bash", json_freeze({"command": "ls"})),
                        tool_call_message(
                            "t2", "Bash", json_freeze({"command": "pwd"})
                        ),
                        tool_call_message(
                            "t3", "Bash", json_freeze({"command": "rm x"})
                        ),
                        tool_call_message(
                            "t4", "Bash", json_freeze({"command": "git log"})
                        ),
                    ],
                    stop_reason="model_tool_use",
                    input_tokens=10,
                    output_tokens=5,
                ),
                ModelResponse(text="done", input_tokens=20, output_tokens=10),
            ],
        )
        agent = Agent(name="t", description="t", model=model, tools=[tool])
        await agent.run(json_freeze({"prompt": "go"}))
        assert tool.starts[1] - tool.starts[0] < 0.05
        assert tool.starts[2] >= tool.ends[1] - 0.01
        assert tool.starts[3] >= tool.ends[2] - 0.01


# -- Serial file mutation dispatch ------------------------------------


class TestSerialFileMutation:
    @pytest.mark.anyio
    async def test_same_file_edits_serialized(self) -> None:
        """Two Edit calls on the same file run serially, not parallel."""
        order: list[str] = []

        class _EditTool:
            def __init__(self) -> None:
                self.name = "Edit"
                self.tool_id = "application/x-tool-edit"
                self.description = "Edit."
                self.supports_microcompaction = True
                self.directive_schema: JSON = json_freeze(
                    {
                        "type": "object",
                        "properties": {},
                    }
                )

            def summary(self, msg: Message) -> str:
                del msg
                return self.name

            def prompt(self) -> str:
                return ""

            async def run(self, msg: Message) -> Message:
                directive = get_directive(msg)
                order.append(str(directive.get("file_path", "")))
                return TextMessage("ok", "text/plain")

        class _ReadTool:
            def __init__(self) -> None:
                self.name = "Read"
                self.tool_id = "application/x-tool-read"
                self.description = "Read."
                self.supports_microcompaction = True
                self.directive_schema: JSON = json_freeze(
                    {
                        "type": "object",
                        "properties": {},
                    }
                )

            def summary(self, msg: Message) -> str:
                del msg
                return self.name

            def prompt(self) -> str:
                return ""

            async def run(self, msg: Message) -> Message:
                del msg
                order.append("read")
                return TextMessage("ok", "text/plain")

        model = _MockModel(
            responses=[
                ModelResponse(
                    text="",
                    tool_calls=[
                        tool_call_message(
                            "e1", "Edit", json_freeze({"file_path": "/fake/f.py"})
                        ),
                        tool_call_message(
                            "e2", "Edit", json_freeze({"file_path": "/fake/f.py"})
                        ),
                        tool_call_message(
                            "r1", "Read", json_freeze({"file_path": "/fake/other.py"})
                        ),
                    ],
                    stop_reason="model_tool_use",
                    input_tokens=10,
                    output_tokens=5,
                ),
                ModelResponse(text="done", input_tokens=10, output_tokens=5),
            ],
        )
        agent = Agent(
            name="test",
            model=model,
            tools=[_EditTool(), _ReadTool()],
        )
        response = await agent.run(json_freeze({"prompt": "edit twice"}))
        assert str(response.content) == "done"
        assert len(order) == 3


# -- Conditional rule injection ----------------------------------------


class TestConditionalRuleInjection:
    """Matching conditional rules append to file-op tool results."""

    @pytest.mark.anyio
    async def test_rule_appended_to_read_result(self, tmp_path: Path) -> None:
        rules = tmp_path / ".sagent" / "rules"
        rules.mkdir(parents=True)
        (rules / "python.md").write_text(
            "---\npaths: ['**/*.py']\n---\n# python-rule: use ruff\n"
        )
        target = tmp_path / "foo.py"
        target.write_text("x = 1\n")

        class _ReadStub:
            def __init__(self) -> None:
                self.name = "Read"
                self.tool_id = "application/x-tool-read"
                self.description = "read"
                self.supports_microcompaction = False
                self.directive_schema: JSON = json_freeze(
                    {
                        "type": "object",
                        "properties": {"file_path": {"type": "string"}},
                    }
                )

            def summary(self, msg: Message) -> str:
                del msg
                return self.name

            def prompt(self) -> str:
                return ""

            async def run(self, msg: Message) -> Message:
                del msg
                return TextMessage("file contents\n", "text/plain")

        model = _MockModel(
            responses=[
                ModelResponse(
                    tool_calls=[
                        tool_call_message(
                            "t1", "Read", json_freeze({"file_path": str(target)})
                        ),
                    ],
                    stop_reason="model_tool_use",
                    input_tokens=10,
                    output_tokens=5,
                ),
                ModelResponse(
                    text="done",
                    stop_reason="model_finished",
                    input_tokens=12,
                    output_tokens=3,
                ),
            ],
        )
        agent = Agent(name="t", description="t", model=model, tools=[_ReadStub()])
        agent.tool_state.bash_cwd = str(tmp_path)
        await agent.run(json_freeze({"prompt": "read foo.py"}))

        tool_results = [
            m for m in agent.messages if m.descriptor == "multipart/x-tool-result"
        ]
        assert tool_results, "expected at least one tool result"
        tr0_text = _tool_result_text(tool_results[0])
        assert "python-rule: use ruff" in tr0_text
        assert "<system-reminder>" in tr0_text

    @pytest.mark.anyio
    async def test_rule_not_appended_to_bash(self, tmp_path: Path) -> None:
        rules = tmp_path / ".sagent" / "rules"
        rules.mkdir(parents=True)
        (rules / "python.md").write_text(
            "---\npaths: ['**/*.py']\n---\n# python-rule\n"
        )

        class _BashStub:
            def __init__(self) -> None:
                self.name = "Bash"
                self.tool_id = "application/x-tool-bash"
                self.description = "bash"
                self.supports_microcompaction = False
                self.directive_schema: JSON = json_freeze(
                    {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                    }
                )

            def summary(self, msg: Message) -> str:
                del msg
                return self.name

            def prompt(self) -> str:
                return ""

            async def run(self, msg: Message) -> Message:
                del msg
                return TextMessage("output\n", "text/plain")

        model = _MockModel(
            responses=[
                ModelResponse(
                    tool_calls=[
                        tool_call_message(
                            "t1", "Bash", json_freeze({"command": "echo hi"})
                        ),
                    ],
                    stop_reason="model_tool_use",
                    input_tokens=10,
                    output_tokens=5,
                ),
                ModelResponse(
                    text="done",
                    stop_reason="model_finished",
                    input_tokens=12,
                    output_tokens=3,
                ),
            ],
        )
        agent = Agent(name="t", description="t", model=model, tools=[_BashStub()])
        agent.tool_state.bash_cwd = str(tmp_path)
        await agent.run(json_freeze({"prompt": "run it"}))

        tool_results = [
            m for m in agent.messages if m.descriptor == "multipart/x-tool-result"
        ]
        assert tool_results
        tr0_text = _tool_result_text(tool_results[0])
        assert "<system-reminder>" not in tr0_text
        assert "python-rule" not in tr0_text

    @pytest.mark.anyio
    async def test_rule_deduped_within_dispatch_batch(self, tmp_path: Path) -> None:
        """N parallel file-ops emit each rule once, not N times."""
        rules = tmp_path / ".sagent" / "rules"
        rules.mkdir(parents=True)
        (rules / "python.md").write_text(
            "---\npaths: ['**/*.py']\n---\n# python-rule: use ruff\n"
        )
        for n in ("a.py", "b.py", "c.py"):
            (tmp_path / n).write_text("x = 1\n")

        class _ReadStub:
            def __init__(self) -> None:
                self.name = "Read"
                self.tool_id = "application/x-tool-read"
                self.description = "read"
                self.supports_microcompaction = False
                self.directive_schema: JSON = json_freeze(
                    {
                        "type": "object",
                        "properties": {"file_path": {"type": "string"}},
                    }
                )

            def summary(self, msg: Message) -> str:
                del msg
                return self.name

            def prompt(self) -> str:
                return ""

            async def run(self, msg: Message) -> Message:
                del msg
                return TextMessage("contents\n", "text/plain")

        model = _MockModel(
            responses=[
                ModelResponse(
                    tool_calls=[
                        tool_call_message(
                            f"t{i}",
                            "Read",
                            json_freeze({"file_path": str(tmp_path / n)}),
                        )
                        for i, n in enumerate(("a.py", "b.py", "c.py"))
                    ],
                    stop_reason="model_tool_use",
                    input_tokens=10,
                    output_tokens=5,
                ),
                ModelResponse(
                    text="done",
                    stop_reason="model_finished",
                    input_tokens=12,
                    output_tokens=3,
                ),
            ],
        )
        agent = Agent(name="t", description="t", model=model, tools=[_ReadStub()])
        agent.tool_state.bash_cwd = str(tmp_path)
        await agent.run(json_freeze({"prompt": "read the python files"}))

        tool_results = [
            m for m in agent.messages if m.descriptor == "multipart/x-tool-result"
        ]
        assert len(tool_results) == 3
        with_rule = [
            r for r in tool_results if "python-rule: use ruff" in _tool_result_text(r)
        ]
        assert len(with_rule) == 1, (
            f"rule appeared in {len(with_rule)} / 3 results; expected 1"
        )


# -- Streaming tool dispatch -------------------------------------------


class TestStreamingToolDispatch:
    """Verify that StreamingTool (async generator) is dispatched correctly."""

    @pytest.mark.anyio
    async def test_streaming_tool_last_yield_is_result(self) -> None:
        """The final yielded message becomes the tool result."""

        class _StreamTool:
            name = "stream"
            tool_id = "application/x-tool-stream"
            description = "yields events then a result"
            supports_microcompaction = False
            streaming: Literal[True] = True
            directive_schema: JSON = json_freeze({"type": "object", "properties": {}})

            def summary(self, msg: Message) -> str:
                del msg
                return self.name

            def prompt(self) -> str:
                return ""

            async def run(self, msg: Message) -> AsyncGenerator[Message, None]:
                del msg
                yield TextMessage("progress 1", "text/plain")
                yield TextMessage("progress 2", "text/plain")
                yield TextMessage("final result", "text/plain")

        tool = _StreamTool()
        # Verify it satisfies the StreamingTool protocol.
        assert isinstance(tool, StreamingTool)

        model = _MockModel(
            responses=[
                ModelResponse(
                    tool_calls=[
                        tool_call_message("s1", "stream", json_freeze({})),
                    ],
                    stop_reason="model_tool_use",
                    input_tokens=10,
                    output_tokens=5,
                ),
                ModelResponse(text="done", input_tokens=20, output_tokens=10),
            ],
        )
        agent = Agent(
            name="test",
            description="Test.",
            model=model,
            tools=[tool],
        )
        response = await agent.run(json_freeze({"prompt": "stream it"}))
        assert str(response.content) == "done"
        # The second model request should have the tool result.
        second_request = model.requests[1]
        tool_results = [
            m
            for m in second_request.messages
            if m.descriptor == "multipart/x-tool-result"
        ]
        assert len(tool_results) == 1
        result_text = _tool_result_text(tool_results[0])
        assert "final result" in result_text

    @pytest.mark.anyio
    async def test_streaming_tool_intermediate_events_emitted(self) -> None:
        """Intermediate yields are forwarded to the events queue."""

        class _StreamTool:
            name = "stream"
            tool_id = "application/x-tool-stream"
            description = "yields events then a result"
            supports_microcompaction = False
            streaming: Literal[True] = True
            directive_schema: JSON = json_freeze({"type": "object", "properties": {}})

            def summary(self, msg: Message) -> str:
                del msg
                return self.name

            def prompt(self) -> str:
                return ""

            async def run(self, msg: Message) -> AsyncGenerator[Message, None]:
                del msg
                yield TextMessage("step 1", "text/plain")
                yield TextMessage("step 2", "text/plain")
                yield TextMessage("result", "text/plain")

        model = _MockModel(
            responses=[
                ModelResponse(
                    tool_calls=[
                        tool_call_message("s1", "stream", json_freeze({})),
                    ],
                    stop_reason="model_tool_use",
                    input_tokens=10,
                    output_tokens=5,
                ),
                ModelResponse(text="ok", input_tokens=20, output_tokens=10),
            ],
        )
        agent = Agent(
            name="test",
            description="Test.",
            model=model,
            tools=[_StreamTool()],
        )
        collected = [event async for event in agent.run(json_freeze({"prompt": "go"}))]

        # Intermediate events ("step 1", "step 2") should be in the
        # event stream. The final "result" becomes the tool result, not
        # an intermediate event.
        intermediates = [
            e
            for e in collected
            if e.descriptor == "text/plain"
            and isinstance(e, TextMessage)
            and e.content.startswith("step")
        ]
        assert len(intermediates) == 2
        assert str(intermediates[0].content) == "step 1"
        assert str(intermediates[1].content) == "step 2"

    @pytest.mark.anyio
    async def test_streaming_tool_empty_generator_raises(self) -> None:
        """A streaming tool that yields nothing should raise RuntimeError."""

        class _EmptyStreamTool:
            name = "empty"
            tool_id = "application/x-tool-empty"
            description = "yields nothing"
            supports_microcompaction = False
            streaming: Literal[True] = True
            directive_schema: JSON = json_freeze({"type": "object", "properties": {}})

            def summary(self, msg: Message) -> str:
                del msg
                return self.name

            def prompt(self) -> str:
                return ""

            async def run(self, msg: Message) -> AsyncGenerator[Message, None]:
                del msg
                return
                yield  # pyright: ignore[reportUnreachable] -- generator protocol requires yield

        req = MultipartMessage(
            (
                TextMessage("q1", "text/x-queue-id"),
                JsonMessage(json_freeze({}), "application/x-tool-empty"),
            ),
            "multipart/x-tool-call",
        )
        with pytest.raises(RuntimeError, match="yielded no messages"):
            await _drain_stream(_EmptyStreamTool(), req)


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
