"""Tests for ``tools.background_task``: bash-style job control surface."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import cast, override

import asyncio
import contextlib
import json
import time

import pytest

from sagent.agent.agent import Agent
from sagent.agent.background import (
    BackgroundAwareTool,
    BackgroundTaskEntry,
)
from sagent.agent.state import agent_registry
from sagent.lib.json import json_freeze
from sagent.testing import FakeAgent, MockModelCaps, with_fake_agent
from sagent.tools.background_task import (
    BackgroundTask,
    cancel_persistent_subagent,
)
from sagent.tools.core import current_agent_var
from sagent.types.model import ModelRequest, ModelResponse, ModelSpec
from sagent.types.runtime import (
    RUNNING_PREFIX,
    AssistantMessage,
    DetachedResult,
    RuntimeEvent,
    ToolCall,
    ToolResult,
)
from sagent.types.tape import ContextSplice


class _PersistentChild(FakeAgent):
    def __init__(self) -> None:
        super().__init__()
        self.shutdown_calls: list[bool] = []

    @override
    def shutdown(self, *, force: bool = False) -> None:
        self.shutdown_calls.append(force)


class _StubModel(MockModelCaps):
    model_id: str = "stub"
    max_request_tokens: int = 100_000

    async def buffer(self, request: ModelRequest) -> ModelResponse:
        return await self.stream(request)

    async def stream(
        self,
        request: ModelRequest,
        on_text: Callable[[str], None] | None = None,
        on_thinking: Callable[[str], None] | None = None,
    ) -> ModelResponse:
        del request, on_text, on_thinking
        return ModelResponse(message=AssistantMessage(text="ok"))


class _RecordingAgent(Agent):
    def __init__(self, *, session_dir: Path | None = None) -> None:
        super().__init__(
            model=_StubModel(),
            model_spec=ModelSpec(
                provider="OpenAISubscription",
                auth="credentials",
                model_id="stub",
                account="default",
            ),
            tools=[],
            session_dir=session_dir,
        )
        self.shutdown_calls: list[bool] = []

    @override
    def shutdown(self, *, force: bool = False) -> None:
        self.shutdown_calls.append(force)


class _DummyInner:
    name: str = "Dummy"
    tool_id: str = "application/x-tool-dummy"
    description: str = "dummy"
    clearable_results: bool = False
    directive_schema = json_freeze(
        {
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "required": ["x"],
        }
    )

    def summary(self, args: Mapping[str, object]) -> str:
        return f"Dummy {args.get('x', '')}"

    def summary_result(self, result: ToolResult) -> str | None:
        del result
        return "ok"

    def prompt(self) -> str:
        return "dummy-prompt"

    def serialize_key(self, args: Mapping[str, object]) -> str | None:
        del args
        return None

    async def run(self, args: Mapping[str, object]) -> ToolResult:
        return ToolResult(call_id="", content=str(args.get("x", "")))


def test_aware_injects_background_and_delay_into_schema() -> None:
    wrapped = BackgroundAwareTool(_DummyInner())
    schema = wrapped.directive_schema
    assert isinstance(schema, Mapping)
    props = schema["properties"]
    assert isinstance(props, Mapping)
    assert "background" in props
    assert "delay" in props
    # Original properties survive.
    assert "x" in props


def test_aware_preserves_metadata_and_delegates() -> None:
    inner = _DummyInner()
    wrapped = BackgroundAwareTool(inner)
    assert wrapped.name == "Dummy"
    assert wrapped.tool_id == "application/x-tool-dummy"
    assert wrapped.description == "dummy"
    assert wrapped.summary({"x": "hi"}) == "Dummy hi"
    assert wrapped.summary_result(ToolResult(call_id="", content="")) == "ok"
    assert wrapped.prompt() == "dummy-prompt"


def test_aware_schema_without_properties_injects_background_fields() -> None:
    """B1: ``type: object`` without ``properties`` still gets BG fields.

    A wrapper that short-circuited to the unmerged schema in this
    branch silently disabled backgrounding for any tool whose schema
    declared no parameters.
    """

    class NoProps:
        name: str = "NP"
        tool_id: str = "application/x-tool-np"
        description: str = ""
        clearable_results: bool = False
        directive_schema = json_freeze({"type": "object"})

        def summary(self, args: Mapping[str, object]) -> str:
            del args
            return "np"

        def summary_result(self, result: ToolResult) -> str | None:
            del result
            return None

        def prompt(self) -> str:
            return ""

        def serialize_key(self, args: Mapping[str, object]) -> str | None:
            del args
            return None

        async def run(self, args: Mapping[str, object]) -> ToolResult:
            del args
            return ToolResult(call_id="", content="")

    wrapped = BackgroundAwareTool(NoProps())
    props = cast(Mapping[str, object], wrapped.directive_schema["properties"])
    assert "background" in props
    assert "delay" in props


@pytest.mark.asyncio
async def test_aware_run_forwards_to_inner() -> None:
    wrapped = BackgroundAwareTool(_DummyInner())
    r = await wrapped.run({"x": "hello"})
    assert r.content == "hello"


def test_metadata_basics() -> None:
    t = BackgroundTask()
    assert t.name == "BackgroundTask"
    assert t.tool_id == "application/x-tool-backgroundtask"
    assert "background:" in t.prompt()
    assert t.summary_result(ToolResult(call_id="", content="")) is None


def test_summary_with_and_without_id() -> None:
    t = BackgroundTask()
    assert t.summary({"operation": "list"}) == "BackgroundTask list"
    assert t.summary({"operation": "cancel", "id": "job-1"}) == (
        "BackgroundTask cancel job-1"
    )


async def _slow() -> ToolResult:
    """Long-running task body for foreground tests.

    Uses ``asyncio.Future`` rather than ``asyncio.sleep`` so the
    autouse ``_fast_sleep`` patcher (in ``conftest.py``) can't make
    the body return early.
    """
    await asyncio.get_running_loop().create_future()
    return ToolResult(call_id="", content="done")


@pytest.mark.asyncio
async def test_run_no_agent_errors() -> None:
    token = current_agent_var.set(None)
    try:
        t = BackgroundTask()
        result = await t.run({"operation": "list"})
    finally:
        current_agent_var.reset(token)
    assert result.is_error
    assert "no active agent" in result.content


@pytest.mark.asyncio
async def test_list_empty() -> None:
    t = BackgroundTask()
    with with_fake_agent():
        result = await t.run({"operation": "list"})
    assert result.content == "No background tasks."


@pytest.mark.asyncio
async def test_list_filters_hidden_and_reports_phases() -> None:
    t = BackgroundTask()
    now = time.time()
    with with_fake_agent() as agent:
        # Build three tasks: one running, one delayed (sleeping), one hidden.
        running_task: asyncio.Task[ToolResult] = asyncio.create_task(_slow())
        sleeping_task: asyncio.Task[ToolResult] = asyncio.create_task(_slow())
        hidden_task: asyncio.Task[ToolResult] = asyncio.create_task(_slow())
        try:
            agent.register_background(
                "j-run",
                BackgroundTaskEntry(
                    task=running_task,
                    tool_name="Dummy",
                    queue_id="j-run",
                    started=now,
                ),
            )
            agent.register_background(
                "j-sleep",
                BackgroundTaskEntry(
                    task=sleeping_task,
                    tool_name="Dummy",
                    queue_id="j-sleep",
                    started=now,
                    delay_sec=10_000.0,
                ),
            )
            agent.register_background(
                "j-hide",
                BackgroundTaskEntry(
                    task=hidden_task,
                    tool_name="HiddenInfra",
                    queue_id="j-hide",
                    started=now,
                    hidden=True,
                ),
            )
            result = await t.run({"operation": "list"})
        finally:
            for task in (running_task, sleeping_task, hidden_task):
                _ = task.cancel()
    assert "j-run" in result.content
    assert "j-sleep" in result.content
    # Hidden infra jobs are filtered out.
    assert "j-hide" not in result.content
    assert "running" in result.content
    assert "sleeping" in result.content


@pytest.mark.asyncio
async def test_list_reports_completed_and_cancelled() -> None:
    t = BackgroundTask()

    async def quick() -> ToolResult:
        return ToolResult(call_id="", content="ok")

    with with_fake_agent() as agent:
        done_task: asyncio.Task[ToolResult] = asyncio.create_task(quick())
        await done_task  # let it finish
        cancelled_task: asyncio.Task[ToolResult] = asyncio.create_task(_slow())
        # Cancel; await with suppression so .cancelled() flips before the
        # tool inspects the task state.
        _ = cancelled_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await cancelled_task
        agent.register_background(
            "j-done",
            BackgroundTaskEntry(
                task=done_task, tool_name="Dummy", queue_id="j-done", started=0.0
            ),
        )
        agent.register_background(
            "j-cancel",
            BackgroundTaskEntry(
                task=cancelled_task,
                tool_name="Dummy",
                queue_id="j-cancel",
                started=0.0,
            ),
        )
        result = await t.run({"operation": "list"})
    assert "completed" in result.content
    assert "cancelled" in result.content


@pytest.mark.asyncio
async def test_cancel_requires_id() -> None:
    t = BackgroundTask()
    with with_fake_agent():
        result = await t.run({"operation": "cancel"})
    assert result.is_error
    assert "requires an id" in result.content


@pytest.mark.asyncio
async def test_cancel_unknown_id() -> None:
    t = BackgroundTask()
    with with_fake_agent():
        result = await t.run({"operation": "cancel", "id": "ghost"})
    assert result.is_error
    assert "No such job" in result.content


@pytest.mark.asyncio
async def test_cancel_hidden_treated_as_unknown() -> None:
    t = BackgroundTask()
    with with_fake_agent() as agent:
        task: asyncio.Task[ToolResult] = asyncio.create_task(_slow())
        try:
            agent.register_background(
                "h",
                BackgroundTaskEntry(
                    task=task,
                    tool_name="HiddenInfra",
                    queue_id="h",
                    started=0.0,
                    hidden=True,
                ),
            )
            result = await t.run({"operation": "cancel", "id": "h"})
        finally:
            _ = task.cancel()
    assert result.is_error
    assert "No such job" in result.content


@pytest.mark.asyncio
async def test_cancel_success_clears_registry() -> None:
    t = BackgroundTask()
    with with_fake_agent() as agent:
        task: asyncio.Task[ToolResult] = asyncio.create_task(_slow())
        agent.register_background(
            "j",
            BackgroundTaskEntry(
                task=task, tool_name="Dummy", queue_id="j", started=0.0
            ),
        )
        result = await t.run({"operation": "cancel", "id": "j"})
        # Yield once so the cancellation is observed by the task.
        with contextlib.suppress(asyncio.CancelledError):
            await task
    assert "Cancelled" in result.content
    assert task.cancelled()
    assert "j" not in agent.background


@pytest.mark.asyncio
async def test_cancel_persistent_subagent_uses_shutdown_lifecycle() -> None:
    t = BackgroundTask()
    with with_fake_agent() as agent:
        task: asyncio.Task[ToolResult] = asyncio.create_task(_slow())
        child = _PersistentChild()
        agent_registry["child"] = child
        try:
            agent.register_background(
                "persistent:child",
                BackgroundTaskEntry(
                    task=task,
                    tool_name="persistent-agent",
                    queue_id="child",
                    started=0.0,
                    kind="persistent_subagent",
                    persistent_run_id="run-child",
                ),
            )
            result = await t.run({"operation": "cancel", "id": "persistent:child"})
        finally:
            agent_registry.pop("child", None)
            _ = task.cancel()
    assert "Cancelled" in result.content
    assert child.shutdown_calls == [True]
    assert not task.cancelled()
    assert "persistent:child" not in agent.background


@pytest.mark.asyncio
async def test_cancel_persistent_subagent_writes_cancelled_lifecycle(
    tmp_path: Path,
) -> None:
    t = BackgroundTask()
    parent = _RecordingAgent(session_dir=tmp_path)
    child = _RecordingAgent()
    task: asyncio.Task[ToolResult] = asyncio.create_task(_slow())
    agent_registry["child"] = child
    token = current_agent_var.set(parent)
    try:
        parent.register_background(
            "persistent:child",
            BackgroundTaskEntry(
                task=task,
                tool_name="persistent-agent",
                queue_id="child",
                started=0.0,
                kind="persistent_subagent",
                persistent_run_id="run-1",
                notify_on_asleep=False,
            ),
        )
        result = await t.run({"operation": "cancel", "id": "persistent:child"})
    finally:
        current_agent_var.reset(token)
        agent_registry.pop("child", None)
        _ = task.cancel()

    assert "Cancelled" in result.content
    assert child.shutdown_calls == [True]
    records = [
        json.loads(line)
        for line in (tmp_path / "session.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if "persistent_agent" in line
    ]
    assert len(records) == 1
    assert records[0]["label"] == "child"
    assert records[0]["run_id"] == "run-1"
    assert records[0]["state"] == "cancelled"
    assert records[0]["notify_on_asleep"] is False
    assert "persistent:child" not in parent.background


@pytest.mark.asyncio
async def test_cancel_persistent_subagent_helper_with_bg_entry() -> None:
    """Unified helper writes lifecycle + clears bg + shuts down the child."""
    with with_fake_agent() as agent:
        task: asyncio.Task[ToolResult] = asyncio.create_task(_slow())
        child = _PersistentChild()
        agent_registry["child"] = child
        try:
            agent.register_background(
                "persistent:child",
                BackgroundTaskEntry(
                    task=task,
                    tool_name="persistent-agent",
                    queue_id="child",
                    started=0.0,
                    kind="persistent_subagent",
                    persistent_run_id="run-child",
                ),
            )
            cancelled = cancel_persistent_subagent(agent, "child")
        finally:
            agent_registry.pop("child", None)
            _ = task.cancel()
    assert cancelled is True
    assert child.shutdown_calls == [True]
    assert "persistent:child" not in agent.background


@pytest.mark.asyncio
async def test_cancel_persistent_subagent_helper_registry_only_fallback() -> None:
    """Helper still shuts down a registered child even when no bg entry exists."""
    with with_fake_agent() as agent:
        child = _PersistentChild()
        agent_registry["child"] = child
        try:
            cancelled = cancel_persistent_subagent(agent, "child")
        finally:
            agent_registry.pop("child", None)
    assert cancelled is True
    assert child.shutdown_calls == [True]


@pytest.mark.asyncio
async def test_cancel_persistent_subagent_helper_returns_false_when_missing() -> None:
    """Helper reports failure when neither bg nor registry knows the label."""
    with with_fake_agent() as agent:
        cancelled = cancel_persistent_subagent(agent, "ghost")
    assert cancelled is False


@pytest.mark.asyncio
async def test_foreground_persistent_subagent_returns_without_detached_result() -> None:
    t = BackgroundTask()
    with with_fake_agent() as agent:
        task: asyncio.Task[ToolResult] = asyncio.create_task(_slow())
        try:
            agent.register_background(
                "persistent:child",
                BackgroundTaskEntry(
                    task=task,
                    tool_name="persistent-agent",
                    queue_id="child",
                    started=0.0,
                    kind="persistent_subagent",
                    persistent_run_id="run-child",
                ),
            )
            result = await asyncio.wait_for(
                t.run({"operation": "foreground", "id": "persistent:child"}),
                timeout=0.01,
            )
        finally:
            _ = task.cancel()
    assert result.is_error
    assert "Persistent subagent jobs cannot be foregrounded" in result.content
    assert "persistent:child" in agent.background


@pytest.mark.asyncio
async def test_foreground_requires_id() -> None:
    t = BackgroundTask()
    with with_fake_agent():
        result = await t.run({"operation": "foreground"})
    assert result.is_error
    assert "requires an id" in result.content


@pytest.mark.asyncio
async def test_foreground_unknown_id() -> None:
    t = BackgroundTask()
    with with_fake_agent():
        result = await t.run({"operation": "foreground", "id": "ghost"})
    assert result.is_error
    assert "No such job" in result.content


@pytest.mark.asyncio
async def test_foreground_success_returns_tool_result() -> None:
    """M6: foreground reads the result from the ``DetachedResult`` event
    posted by the bg task (cohort-detached and explicit-bg tasks alike
    return ``None`` from the task itself).
    """
    t = BackgroundTask()
    with with_fake_agent() as agent:
        task: asyncio.Task[None] = asyncio.create_task(asyncio.sleep(0))
        agent.register_background(
            "j",
            BackgroundTaskEntry(
                task=task, tool_name="Dummy", queue_id="j", started=0.0
            ),
        )

        async def deliver() -> None:
            await asyncio.sleep(0.01)
            agent.runtime.publish(
                DetachedResult(result=ToolResult(call_id="j", content="payload")),
            )

        delivery = asyncio.create_task(deliver())
        result = await t.run({"operation": "foreground", "id": "j"})
        await delivery
    assert result.content == "payload"
    assert "j" not in agent.background


@pytest.mark.asyncio
async def test_foreground_wait_cancellation_leaves_job_running() -> None:
    t = BackgroundTask()
    wait_forever = asyncio.Event()
    with with_fake_agent() as agent:
        task: asyncio.Task[bool] = asyncio.create_task(wait_forever.wait())
        agent.register_background(
            "j",
            BackgroundTaskEntry(
                task=task, tool_name="Dummy", queue_id="j", started=0.0
            ),
        )
        foreground = asyncio.create_task(t.run({"operation": "foreground", "id": "j"}))
        await asyncio.sleep(0)

        foreground.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await foreground

        assert not task.done()
        assert "j" in agent.background
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_foreground_cancelled_task_returns_tool_error() -> None:
    t = BackgroundTask()
    with with_fake_agent() as agent:
        task: asyncio.Task[None] = asyncio.create_task(asyncio.sleep(10.0))
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        agent.register_background(
            "j",
            BackgroundTaskEntry(
                task=task, tool_name="Dummy", queue_id="j", started=0.0
            ),
        )

        result = await t.run({"operation": "foreground", "id": "j"})

    assert result.is_error
    assert "cancelled" in result.content
    assert "j" not in agent.background


@pytest.mark.asyncio
async def test_foreground_crashed_task_returns_tool_error() -> None:
    async def fail() -> None:
        raise RuntimeError("boom")

    t = BackgroundTask()
    with with_fake_agent() as agent:
        task = asyncio.create_task(fail())
        with contextlib.suppress(RuntimeError):
            await task
        agent.register_background(
            "j",
            BackgroundTaskEntry(
                task=task, tool_name="Dummy", queue_id="j", started=0.0
            ),
        )

        result = await t.run({"operation": "foreground", "id": "j"})

    assert result.is_error
    assert "RuntimeError: boom" in result.content
    assert "j" not in agent.background


@pytest.mark.asyncio
async def test_foreground_running_background_job_returns_result_stub_stays() -> None:
    foreground_waiting = asyncio.Event()
    finish_background = asyncio.Event()

    t = BackgroundTask()
    with with_fake_agent() as agent:
        agent.runtime.append_history(
            AssistantMessage(
                tool_calls=(ToolCall(id="j-running", name="Dummy", args={}),),
            ),
        )
        agent.runtime.append_history(
            ToolResult(
                call_id="j-running",
                content=f"{RUNNING_PREFIX}Dummy]",
            ),
        )

        async def finish() -> None:
            await finish_background.wait()
            agent.runtime.inbox.push_back(
                DetachedResult(
                    result=ToolResult(call_id="j-running", content="queued payload")
                )
            )
            agent.cancel_background("j-running")

        task = asyncio.create_task(finish())
        agent.register_background(
            "j-running",
            BackgroundTaskEntry(
                task=task,
                tool_name="Dummy",
                queue_id="j-running",
                started=0.0,
            ),
        )

        class ForegroundAwareObservers(list[Callable[[RuntimeEvent], None]]):
            @override
            def append(self, observer: Callable[[RuntimeEvent], None]) -> None:
                super().append(observer)
                foreground_waiting.set()

        observers = ForegroundAwareObservers(agent.runtime.observers)
        agent.runtime.observers = observers
        foreground = asyncio.create_task(
            t.run({"operation": "foreground", "id": "j-running"})
        )
        try:
            await asyncio.wait_for(foreground_waiting.wait(), timeout=1.0)
            finish_background.set()
            result = await asyncio.wait_for(foreground, timeout=1.0)
        finally:
            agent.runtime.observers = list(observers)
            if not foreground.done():
                foreground.cancel()
    messages = agent.runtime.context().messages
    # The foregrounder receives the real result as its own tool answer.
    assert result.content == "queued payload"
    # The original ``[Running in background]`` placeholder STAYS (it is the
    # honest record that the call was backgrounded; the result arrived via
    # foreground). It is not silently back-patched in-slot.
    assert any(
        isinstance(message, ToolResult)
        and message.call_id == "j-running"
        and message.content == f"{RUNNING_PREFIX}Dummy]"
        for message in messages
    )
    # No foreground back-patch splice was appended.
    assert not any(
        isinstance(record, ContextSplice)
        and record.strategy == "foreground_detached_splice"
        for record in agent.runtime.tape
    )
    assert not agent.events_of(DetachedResult)
    assert "j-running" not in agent.background


@pytest.mark.asyncio
async def test_foreground_reads_pre_existing_spliced_result() -> None:
    """M6: when the splice has already landed in history, foreground
    returns the spliced result without waiting for another event.
    """
    t = BackgroundTask()
    with with_fake_agent() as agent:
        task: asyncio.Task[None] = asyncio.create_task(asyncio.sleep(0))
        agent.register_background(
            "j",
            BackgroundTaskEntry(
                task=task, tool_name="Dummy", queue_id="j", started=0.0
            ),
        )
        agent.runtime.append_history(
            ToolResult(call_id="j", content="prior result"),
        )
        result = await t.run({"operation": "foreground", "id": "j"})
    assert result.content == "prior result"
    assert "j" not in agent.background


@pytest.mark.asyncio
async def test_foreground_propagates_error_via_detached_result() -> None:
    """M6: a failing bg task posts an ``is_error=True`` ``DetachedResult``;
    foreground surfaces it as a tool-error result.
    """
    t = BackgroundTask()
    with with_fake_agent() as agent:
        task: asyncio.Task[None] = asyncio.create_task(asyncio.sleep(0))
        agent.register_background(
            "j",
            BackgroundTaskEntry(
                task=task, tool_name="Dummy", queue_id="j", started=0.0
            ),
        )

        async def deliver() -> None:
            await asyncio.sleep(0.01)
            agent.runtime.publish(
                DetachedResult(
                    result=ToolResult(
                        call_id="j",
                        content="RuntimeError: boom",
                        is_error=True,
                    ),
                ),
            )

        delivery = asyncio.create_task(deliver())
        result = await t.run({"operation": "foreground", "id": "j"})
        await delivery
    assert result.is_error
    assert "RuntimeError" in result.content
    assert "boom" in result.content
    assert "j" not in agent.background


@pytest.mark.asyncio
async def test_run_unknown_operation() -> None:
    t = BackgroundTask()
    with with_fake_agent():
        result = await t.run({"operation": "bogus"})
    assert result.is_error
    assert "Unknown operation" in result.content


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
