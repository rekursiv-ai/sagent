"""Tests for ``tools.background_task``: bash-style job control surface."""

from __future__ import annotations

from collections.abc import Mapping

import asyncio
import contextlib
import time

import pytest

from sagent.agent.background import (
    BackgroundAwareTool,
    BackgroundTaskEntry,
)
from sagent.agent.runtime import DetachedResult, ToolResult
from sagent.lib.json import json_freeze
from sagent.testing import with_fake_agent
from sagent.tools.background_task import BackgroundTask
from sagent.tools.core import current_agent_var


class _DummyInner:
    name: str = "Dummy"
    tool_id: str = "application/x-tool-dummy"
    description: str = "dummy"
    supports_microcompaction: bool = False
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
    assert wrapped.supports_microcompaction is False
    assert wrapped.summary({"x": "hi"}) == "Dummy hi"
    assert wrapped.summary_result(ToolResult(call_id="", content="")) == "ok"
    assert wrapped.prompt() == "dummy-prompt"


def test_aware_schema_without_properties_passes_through() -> None:
    class NoProps:
        name: str = "NP"
        tool_id: str = "application/x-tool-np"
        description: str = ""
        supports_microcompaction: bool = False
        directive_schema = json_freeze({"type": "object"})

        def summary(self, args: Mapping[str, object]) -> str:
            del args
            return "np"

        def summary_result(self, result: ToolResult) -> str | None:
            del result
            return None

        def prompt(self) -> str:
            return ""

        async def run(self, args: Mapping[str, object]) -> ToolResult:
            del args
            return ToolResult(call_id="", content="")

    wrapped = BackgroundAwareTool(NoProps())
    # Schema unchanged - no properties key to merge into.
    assert wrapped.directive_schema == {"type": "object"}


@pytest.mark.asyncio
async def test_aware_run_forwards_to_inner() -> None:
    wrapped = BackgroundAwareTool(_DummyInner())
    r = await wrapped.run({"x": "hello"})
    assert r.content == "hello"


def test_metadata_basics() -> None:
    t = BackgroundTask()
    assert t.name == "BackgroundTask"
    assert t.tool_id == "application/x-tool-backgroundtask"
    assert t.supports_microcompaction is True
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
                DetachedResult(call_id="j", content="payload"),
            )

        delivery = asyncio.create_task(deliver())
        result = await t.run({"operation": "foreground", "id": "j"})
        await delivery
    assert result.content == "payload"
    assert "j" not in agent.background


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
        agent.runtime.history.append(
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
                    call_id="j",
                    content="RuntimeError: boom",
                    is_error=True,
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
