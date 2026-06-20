"""Tests for ``agent.background``: ``BackgroundAwareTool`` + entry validation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import cast

import asyncio

import pytest

from sagent.agent.background import (
    BackgroundAwareTool,
    BackgroundTaskEntry,
    split_bg_args,
)
from sagent.lib.custom_json import JSON, json_freeze
from sagent.types.runtime import ToolResult


@dataclass(slots=True, kw_only=True)
class _StubTool:
    """Minimal ``Tool`` whose schema and ``clearable_results`` are configurable."""

    name: str = "Stub"
    tool_id: str = "application/x-tool-stub"
    description: str = "stub tool"
    directive_schema: JSON = field(
        default_factory=lambda: json_freeze({"type": "object"}),
    )
    clearable_results: bool = False
    calls: list[Mapping[str, object]] = field(default_factory=list)

    def summary(self, args: Mapping[str, object]) -> str:
        del args
        return "stub"

    def summary_result(self, result: ToolResult) -> str | None:
        del result
        return None

    def prompt(self) -> str | None:
        return None

    def serialize_key(self, args: Mapping[str, object]) -> str | None:
        del args
        return None

    async def run(self, args: Mapping[str, object]) -> ToolResult:
        self.calls.append(args)
        return ToolResult(call_id="", content="ok")


def _props(schema: JSON) -> Mapping[str, object]:
    return cast(Mapping[str, object], schema["properties"])


def test_background_aware_tool_injects_into_schemaless_tool() -> None:
    """B1: a ``type: object`` schema without ``properties`` still gets BG fields.

    Before the fix, ``BackgroundAwareTool`` short-circuited to the
    unmerged schema when ``properties`` wasn't a Mapping; the LLM
    never learned about ``background`` / ``delay`` and silently lost
    the ability to schedule async tool runs.
    """
    tool = _StubTool(directive_schema=json_freeze({"type": "object"}))
    wrapped = BackgroundAwareTool(tool)
    props = _props(wrapped.directive_schema)
    assert "background" in props
    assert "delay" in props


def test_background_aware_tool_preserves_existing_properties() -> None:
    """Existing ``properties`` survive the merge unchanged."""
    tool = _StubTool(
        directive_schema=json_freeze(
            {"type": "object", "properties": {"msg": {"type": "string"}}}
        ),
    )
    wrapped = BackgroundAwareTool(tool)
    props = _props(wrapped.directive_schema)
    assert "msg" in props
    assert "background" in props
    assert "delay" in props


def test_background_aware_tool_round_trips_clearable_results_true() -> None:
    """D6: ``clearable_results`` mirrors the wrapped tool (True case)."""
    tool = _StubTool(clearable_results=True)
    wrapped = BackgroundAwareTool(tool)
    assert wrapped.clearable_results is True


def test_background_aware_tool_round_trips_clearable_results_false() -> None:
    """D6: ``clearable_results`` mirrors the wrapped tool (False case)."""
    tool = _StubTool(clearable_results=False)
    wrapped = BackgroundAwareTool(tool)
    assert wrapped.clearable_results is False


def test_split_bg_args_rejects_negative_delay_by_coercion() -> None:
    """B2: negative ``delay`` coerces to ``0`` instead of waiting backwards.

    A negative integer escaping the LLM's schema-clamped ``minimum: 0``
    used to flow through as a negative ``delay_sec``. Coerce to ``0``
    so the wait either disappears or is bounded; ``background`` no
    longer flips True from a negative delay.
    """
    bg, delay, clean = split_bg_args({"msg": "hi", "delay": -5})
    assert delay == 0.0
    assert bg is False
    assert clean == {"msg": "hi"}


def test_background_task_entry_rejects_empty_persistent_run_id() -> None:
    """E1: ``kind='persistent_subagent'`` requires a non-empty run id.

    The persistent-driver bookkeeping keys off ``persistent_run_id``;
    an empty id silently aliases two distinct subagents to the same
    slot. Catch at construction.
    """
    loop = asyncio.new_event_loop()
    try:
        task = loop.create_task(asyncio.sleep(0))
        with pytest.raises(ValueError, match="persistent_run_id"):
            _ = BackgroundTaskEntry(
                task=task,
                tool_name="child",
                queue_id="q",
                started=0.0,
                kind="persistent_subagent",
                persistent_run_id="",
            )
        _ = task.cancel()
        loop.run_until_complete(asyncio.gather(task, return_exceptions=True))
    finally:
        loop.close()


def test_background_task_entry_allows_empty_run_id_for_tool_kind() -> None:
    """``kind='tool'`` doesn't need a run id (the field is persistent-only)."""
    loop = asyncio.new_event_loop()
    try:
        task = loop.create_task(asyncio.sleep(0))
        entry = BackgroundTaskEntry(
            task=task,
            tool_name="bg",
            queue_id="job-1",
            started=0.0,
        )
        assert entry.persistent_run_id == ""
        _ = task.cancel()
        loop.run_until_complete(asyncio.gather(task, return_exceptions=True))
    finally:
        loop.close()


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
