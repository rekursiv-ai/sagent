"""Tests for ``agent.compaction``: post-compaction enrichment helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import asyncio

import pytest

from sagent.agent.background import BackgroundTaskEntry
from sagent.agent.compaction import (
    CompactionState,
    append_to_first_user,
    extract_topic,
    inject_background_status,
    is_summary,
    post_compact_enrich,
)
from sagent.lib.json import JSON
from sagent.tools.core import ToolState
from sagent.types.history import (
    AssistantMessage,
    HistoryEntry,
    ToolResult,
    UserMessage,
)
from sagent.types.model import ContextBudget
from sagent.types.tools import Tool


@dataclass(slots=True, kw_only=True)
class _StubTool:
    """Tool stub satisfying the rich ``Tool`` protocol surface."""

    name: str = "Stub"
    tool_id: str = "application/x-tool-stub"
    description: str = "Stub tool."
    supports_microcompaction: bool = False
    directive_schema: JSON = field(default_factory=lambda: {"type": "object"})

    def summary(self, args: Mapping[str, object]) -> str:
        del args
        return ""

    def summary_result(self, result: ToolResult) -> str | None:
        del result
        return None

    def prompt(self) -> str:
        return ""

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
    assert s.summary_pointers == []
    assert s.compacting is False


def test_is_summary_true_when_continuation_marker_present() -> None:
    text = "This conversation is being continued from a previous turn."
    assert is_summary(text) is True


def test_is_summary_false_when_marker_absent() -> None:
    assert is_summary("compaction failed") is False


def test_is_summary_false_when_empty() -> None:
    assert is_summary("") is False


def test_extract_topic_returns_first_substantive_line() -> None:
    text = "summary heading:\n- bullet line\nReal topic line.\n"
    assert extract_topic(text) == "Real topic line."


def test_extract_topic_skips_bullets_and_colon_lines() -> None:
    assert extract_topic("- a\n- b\nfoo:\nFinal value") == "Final value"


def test_extract_topic_caps_at_120_chars() -> None:
    long = "x" * 200
    topic = extract_topic(long)
    assert len(topic) == 120
    assert topic == "x" * 120


def test_extract_topic_falls_back_to_default_string() -> None:
    assert extract_topic("- only\n- bullets\nlast:") == "(compacted context)"


def test_extract_topic_falls_back_on_empty() -> None:
    assert extract_topic("") == "(compacted context)"


def test_append_to_first_user_concatenates_when_text_nonempty() -> None:
    history: list[HistoryEntry] = [
        UserMessage(text="orig"),
        AssistantMessage(text="a"),
    ]
    append_to_first_user(history, "more")
    first = history[0]
    assert isinstance(first, UserMessage)
    assert first.text == "orig\n\nmore"


def test_append_to_first_user_replaces_when_empty_text() -> None:
    history: list[HistoryEntry] = [UserMessage(text="")]
    append_to_first_user(history, "fresh")
    first = history[0]
    assert isinstance(first, UserMessage)
    assert first.text == "fresh"


def test_append_to_first_user_inserts_when_no_user_message() -> None:
    history: list[HistoryEntry] = [AssistantMessage(text="hi")]
    append_to_first_user(history, "context")
    assert isinstance(history[0], UserMessage)
    assert history[0].text == "context"
    assert isinstance(history[1], AssistantMessage)


def test_inject_background_status_no_jobs_is_noop() -> None:
    orig = UserMessage(text="hi")
    history: list[HistoryEntry] = [orig]
    inject_background_status(history, {})
    assert history[0] is orig
    after = history[0]
    assert isinstance(after, UserMessage)
    assert after.text == "hi"


@pytest.mark.asyncio
async def test_inject_background_status_appends_to_first_user() -> None:
    history: list[HistoryEntry] = [UserMessage(text="orig")]
    bg = await _make_bg("Bash", "q1")
    inject_background_status(history, {"q1": bg})
    first = history[0]
    assert isinstance(first, UserMessage)
    assert "Active background tasks" in first.text
    assert "[q1] Bash" in first.text


@pytest.mark.asyncio
async def test_post_compact_enrich_writes_summary_file_when_real_summary(
    tmp_path: Path,
) -> None:
    state = CompactionState()
    # Topic extraction skips lines ending with ``:``; the first
    # non-bullet, non-colon-terminated line becomes the pointer topic.
    summary_text = "continued from a previous summary:\nTopic line here."
    result: list[HistoryEntry] = [UserMessage(text=summary_text)]
    history: list[HistoryEntry] = list(result)
    tools_map: Mapping[str, Tool] = {}
    await post_compact_enrich(
        result=result,
        history=history,
        state=state,
        session_dir=tmp_path,
        tool_state=ToolState(),
        budget=_budget(),
        tools=tools_map,
        background_tasks={},
        estimate_tokens=0,
        headroom=0,
    )
    # Step 1 saved a summary file + appended a pointer.
    summary_file = tmp_path / "summary_0.md"
    assert summary_file.exists()
    assert state.summary_pointers == [(str(summary_file), "Topic line here.")]


@pytest.mark.asyncio
async def test_post_compact_enrich_skips_summary_save_when_not_a_real_summary(
    tmp_path: Path,
) -> None:
    state = CompactionState()
    result: list[HistoryEntry] = [UserMessage(text="compaction failed")]
    history: list[HistoryEntry] = list(result)
    tools_map: Mapping[str, Tool] = {}
    await post_compact_enrich(
        result=result,
        history=history,
        state=state,
        session_dir=tmp_path,
        tool_state=ToolState(),
        budget=_budget(),
        tools=tools_map,
        background_tasks={},
        estimate_tokens=0,
        headroom=0,
    )
    assert not (tmp_path / "summary_0.md").exists()
    assert state.summary_pointers == []


@pytest.mark.asyncio
async def test_post_compact_enrich_no_session_dir_skips_summary_save() -> None:
    state = CompactionState()
    text = "continued from a previous turn"
    result: list[HistoryEntry] = [UserMessage(text=text)]
    history: list[HistoryEntry] = list(result)
    tools_map: Mapping[str, Tool] = {}
    await post_compact_enrich(
        result=result,
        history=history,
        state=state,
        session_dir=None,
        tool_state=ToolState(),
        budget=_budget(),
        tools=tools_map,
        background_tasks={},
        estimate_tokens=0,
        headroom=0,
    )
    assert state.summary_pointers == []


@pytest.mark.asyncio
async def test_post_compact_enrich_runs_restorable_tool_hook(tmp_path: Path) -> None:
    """A tool implementing ``CompactRestorable`` is called."""
    calls: list[int] = []

    @dataclass(slots=True, kw_only=True)
    class Restorable(_StubTool):
        name: str = "R"

        async def post_compact_restore(
            self,
            history: list[HistoryEntry],
            tool_state: ToolState,
            *,
            budget_chars: int = 100_000,
        ) -> None:
            del history, tool_state
            calls.append(budget_chars)

    state = CompactionState()
    result: list[HistoryEntry] = [UserMessage(text="x")]
    tools_map: Mapping[str, Tool] = {"R": Restorable()}
    await post_compact_enrich(
        result=result,
        history=result,
        state=state,
        session_dir=tmp_path,
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
    tmp_path: Path,
) -> None:
    """A raising tool hook is logged and skipped without aborting the pipeline."""

    @dataclass(slots=True, kw_only=True)
    class BadRestorable(_StubTool):
        name: str = "B"

        async def post_compact_restore(
            self,
            history: list[HistoryEntry],
            tool_state: ToolState,
            *,
            budget_chars: int = 100_000,
        ) -> None:
            del history, tool_state, budget_chars
            raise RuntimeError("nope")

    state = CompactionState()
    result: list[HistoryEntry] = [UserMessage(text="x")]
    tools_map: Mapping[str, Tool] = {"B": BadRestorable()}
    await post_compact_enrich(
        result=result,
        history=result,
        state=state,
        session_dir=tmp_path,
        tool_state=ToolState(),
        budget=_budget(),
        tools=tools_map,
        background_tasks={},
        estimate_tokens=0,
        headroom=0,
    )
    # Pipeline completed despite the failure.
    assert state.summary_pointers == []


@pytest.mark.asyncio
async def test_post_compact_enrich_swallows_summary_save_failure(
    tmp_path: Path,
) -> None:
    """Step 1 (summary save) failure is logged and the pipeline continues."""
    state = CompactionState()
    # Triggering a write failure: replace ``summary_0.md`` with a
    # directory so ``write_text`` raises.
    blocker = tmp_path / "summary_0.md"
    blocker.mkdir()
    result: list[HistoryEntry] = [UserMessage(text="continued from a previous turn")]
    tools_map: Mapping[str, Tool] = {}
    await post_compact_enrich(
        result=result,
        history=result,
        state=state,
        session_dir=tmp_path,
        tool_state=ToolState(),
        budget=_budget(),
        tools=tools_map,
        background_tasks={},
        estimate_tokens=0,
        headroom=0,
    )
    # Pipeline completed; pointer NOT appended because save failed.
    assert state.summary_pointers == []


@pytest.mark.asyncio
async def test_post_compact_enrich_injects_background_status(tmp_path: Path) -> None:
    state = CompactionState()
    history: list[HistoryEntry] = [UserMessage(text="orig")]
    result: list[HistoryEntry] = history
    tools_map: Mapping[str, Tool] = {}
    bg = {"q9": await _make_bg("Bash", "q9")}
    await post_compact_enrich(
        result=result,
        history=history,
        state=state,
        session_dir=tmp_path,
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
