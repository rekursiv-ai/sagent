"""Tests for ``agent.compaction``: post-compaction enrichment helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import asyncio

import pytest

from sagent.agent.background import BackgroundTaskEntry
from sagent.agent.compaction import (
    CompactionState,
    append_to_first_user,
    inject_background_status,
    post_compact_enrich,
)
from sagent.lib.json import JSON
from sagent.tools.core import ToolState
from sagent.types.model import ContextBudget
from sagent.types.runtime import (
    AssistantMessage,
    ModelContextEvent,
    ToolResult,
    UserMessage,
)
from sagent.types.tools import Tool


@dataclass(slots=True, kw_only=True)
class _StubTool:
    """Tool stub satisfying the rich ``Tool`` protocol surface."""

    name: str = "Stub"
    tool_id: str = "application/x-tool-stub"
    description: str = "Stub tool."
    directive_schema: JSON = field(default_factory=lambda: {"type": "object"})
    clearable_results: bool = False

    def summary(self, args: Mapping[str, object]) -> str:
        del args
        return ""

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


def _budget() -> ContextBudget:
    return ContextBudget(
        max_request_tokens=100_000,
        max_response_tokens=4_096,
        reattach_count=2,
        reattach_max_chars=100,
        reattach_budget=1_000,
    )


async def _noop_coro() -> None:
    return None


async def _make_bg(name: str, qid: str) -> BackgroundTaskEntry:
    """Build a ``BackgroundTaskEntry`` whose task is already running."""
    task = asyncio.create_task(_noop_coro())
    await asyncio.sleep(0)
    return BackgroundTaskEntry(
        task=task,
        tool_name=name,
        queue_id=qid,
        started=0.0,
    )


def test_compaction_state_defaults() -> None:
    s = CompactionState()
    assert s.compact_count == 0
    assert s.compact_failures == 0
    assert s.compacting is False


def test_append_to_first_user_concatenates_when_text_nonempty() -> None:
    history: list[ModelContextEvent] = [
        UserMessage(text="orig"),
        AssistantMessage(text="a"),
    ]
    append_to_first_user(history, "more")
    first = history[0]
    assert isinstance(first, UserMessage)
    assert first.text == "orig\n\nmore"


def test_append_to_first_user_replaces_when_empty_text() -> None:
    history: list[ModelContextEvent] = [UserMessage(text="")]
    append_to_first_user(history, "fresh")
    first = history[0]
    assert isinstance(first, UserMessage)
    assert first.text == "fresh"


def test_append_to_first_user_inserts_when_no_user_message() -> None:
    history: list[ModelContextEvent] = [AssistantMessage(text="hi")]
    append_to_first_user(history, "context")
    assert isinstance(history[0], UserMessage)
    assert history[0].text == "context"
    assert isinstance(history[1], AssistantMessage)


def test_inject_background_status_no_jobs_is_noop() -> None:
    orig = UserMessage(text="hi")
    history: list[ModelContextEvent] = [orig]
    inject_background_status(history, {})
    assert history[0] is orig
    after = history[0]
    assert isinstance(after, UserMessage)
    assert after.text == "hi"


@pytest.mark.asyncio
async def test_inject_background_status_appends_to_first_user() -> None:
    history: list[ModelContextEvent] = [UserMessage(text="orig")]
    bg = await _make_bg("Bash", "q1")
    inject_background_status(history, {"q1": bg})
    first = history[0]
    assert isinstance(first, UserMessage)
    assert "Active background tasks" in first.text
    assert "[q1] Bash" in first.text


@pytest.mark.asyncio
async def test_inject_background_status_wording_distinguishes_running_from_done() -> (
    None
):
    """Running tasks report 'since started'; completed ones report 'ago'."""
    gate = asyncio.Event()

    async def _wait_gate() -> None:
        await gate.wait()

    running_task = asyncio.create_task(_wait_gate())
    done_task = asyncio.create_task(_noop_coro())
    await done_task  # drive ``_noop_coro`` to completion.
    assert done_task.done()
    assert not running_task.done()
    running = BackgroundTaskEntry(
        task=running_task, tool_name="Bash", queue_id="qR", started=0.0
    )
    done = BackgroundTaskEntry(
        task=done_task, tool_name="Bash", queue_id="qD", started=0.0
    )
    history: list[ModelContextEvent] = [UserMessage(text="orig")]
    try:
        inject_background_status(history, {"qR": running, "qD": done})
        first = history[0]
        assert isinstance(first, UserMessage)
        assert "qR" in first.text
        assert "since started" in first.text
        assert "qD" in first.text
        assert "ago" in first.text
    finally:
        gate.set()
        await running_task


@pytest.mark.asyncio
async def test_post_compact_enrich_runs_restorable_tool_hook() -> None:
    """A tool implementing ``CompactRestorable`` is called."""
    calls: list[int] = []

    @dataclass(slots=True, kw_only=True)
    class Restorable(_StubTool):
        name: str = "R"

        async def post_compact_restore(
            self,
            history: list[ModelContextEvent],
            tool_state: ToolState,
            *,
            budget_chars: int = 100_000,
        ) -> None:
            del history, tool_state
            calls.append(budget_chars)

    history: list[ModelContextEvent] = [UserMessage(text="x")]
    tools_map: Mapping[str, Tool] = {"R": Restorable()}
    await post_compact_enrich(
        history=history,
        tool_state=ToolState(),
        budget=_budget(),
        tools=tools_map,
        background_tasks={},
        estimate_tokens=100,
        headroom=10,
    )
    # available = max(0, 100 - 10) = 90; chars_per_token = 4 → 360.
    assert calls == [360]


@pytest.mark.asyncio
async def test_post_compact_enrich_swallows_restorable_failures(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A raising tool hook is logged and skipped without aborting the pipeline."""
    calls: list[str] = []

    @dataclass(slots=True, kw_only=True)
    class BadRestorable(_StubTool):
        name: str = "B"

        async def post_compact_restore(
            self,
            history: list[ModelContextEvent],
            tool_state: ToolState,
            *,
            budget_chars: int = 100_000,
        ) -> None:
            del history, tool_state, budget_chars
            calls.append("B")
            raise RuntimeError("nope")

    # A second hook AFTER the failing one must still run (the failure is
    # skipped, not fatal to the pipeline).
    @dataclass(slots=True, kw_only=True)
    class GoodRestorable(_StubTool):
        name: str = "C"

        async def post_compact_restore(
            self,
            history: list[ModelContextEvent],
            tool_state: ToolState,
            *,
            budget_chars: int = 100_000,
        ) -> None:
            del history, tool_state, budget_chars
            calls.append("C")

    history: list[ModelContextEvent] = [UserMessage(text="x")]
    tools_map: Mapping[str, Tool] = {"B": BadRestorable(), "C": GoodRestorable()}
    with caplog.at_level("WARNING"):
        await post_compact_enrich(
            history=history,
            tool_state=ToolState(),
            budget=_budget(),
            tools=tools_map,
            background_tasks={},
            estimate_tokens=0,
            headroom=0,
        )
    # The failing hook ran...
    assert "B" in calls
    # ...the failure was logged (not silently swallowed)...
    assert any(
        "post_compact_restore failed" in r.getMessage() and "B" in r.getMessage()
        for r in caplog.records
    ), f"expected a logged warning naming the failed hook; got {caplog.records!r}"
    # ...and a later hook still ran (the pipeline was not aborted).
    assert "C" in calls, "a hook after the failing one must still run"


@pytest.mark.asyncio
async def test_post_compact_enrich_injects_background_status() -> None:
    history: list[ModelContextEvent] = [UserMessage(text="orig")]
    tools_map: Mapping[str, Tool] = {}
    bg = {"q9": await _make_bg("Bash", "q9")}
    await post_compact_enrich(
        history=history,
        tool_state=ToolState(),
        budget=_budget(),
        tools=tools_map,
        background_tasks=bg,
        estimate_tokens=0,
        headroom=0,
    )
    first = history[0]
    assert isinstance(first, UserMessage)
    assert "Active background tasks" in first.text


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
