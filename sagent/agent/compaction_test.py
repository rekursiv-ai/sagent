"""Tests for agent.compaction helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import asyncio

import pytest

from sagent.agent.compaction import (
    CompactionState,
    extract_topic,
    inject_background_status,
    post_compact_enrich,
)
from sagent.custom_types import ContextBudget, Message, TextMessage
from sagent.tools.background_task import BackgroundTaskEntry
from sagent.tools.core import ToolState


def _user_msg(text: str = "hello") -> Message:
    return TextMessage(text, "text/x-user-message")


def _bg_entry(
    *, tool_name: str = "Bash", done: bool = False, started: float = 0.0
) -> BackgroundTaskEntry:
    task: asyncio.Task[Message] = MagicMock(spec=asyncio.Task)
    task.done.return_value = done
    return BackgroundTaskEntry(
        task=task,
        tool_name=tool_name,
        queue_id="q1",
        started=started,
    )


class TestExtractTopic:
    def test_returns_first_content_line(self) -> None:
        text = "This is the topic\n- bullet\n- another"
        assert extract_topic(text) == "This is the topic"

    def test_skips_bullets_and_headers(self) -> None:
        text = "- bullet\nHeading:\nActual topic line"
        assert extract_topic(text) == "Actual topic line"

    def test_all_bullets_returns_default(self) -> None:
        text = "- a\n- b\n- c"
        assert extract_topic(text) == "(compacted context)"

    def test_all_headings_returns_default(self) -> None:
        text = "Section:\nAnother:"
        assert extract_topic(text) == "(compacted context)"

    def test_truncates_long_line(self) -> None:
        text = "x" * 200
        assert extract_topic(text) == "x" * 120

    def test_skips_blank_lines(self) -> None:
        text = "\n\n  \nReal topic"
        assert extract_topic(text) == "Real topic"


class TestInjectBackgroundStatus:
    def test_no_tasks_is_noop(self) -> None:
        msgs = [_user_msg()]
        inject_background_status(msgs, {})
        assert str(msgs[0].content) == "hello"

    @patch("sagent.agent.compaction.time")
    def test_running_task(self, mock_time: MagicMock) -> None:
        mock_time.time.return_value = 100.0
        entry = _bg_entry(tool_name="Grep", done=False, started=50.0)
        msgs = [_user_msg()]
        inject_background_status(msgs, {"abc": entry})
        text = str(msgs[0].content)
        assert "abc" in text
        assert "Grep" in text
        assert "running" in text
        assert "50s ago" in text

    @patch("sagent.agent.compaction.time")
    def test_completed_task(self, mock_time: MagicMock) -> None:
        mock_time.time.return_value = 200.0
        entry = _bg_entry(tool_name="Bash", done=True, started=190.0)
        msgs = [_user_msg()]
        inject_background_status(msgs, {"z1": entry})
        text = str(msgs[0].content)
        assert "completed" in text


class TestPostCompactEnrich:
    @staticmethod
    def _budget() -> ContextBudget:
        return ContextBudget(
            max_request_tokens=1000,
            max_response_tokens=500,
            reattach_count=2,
            reattach_max_chars=100,
            reattach_budget=200,
        )

    @staticmethod
    def _state() -> CompactionState:
        return CompactionState()

    @pytest.mark.asyncio
    async def test_saves_summary_file(self, tmp_path: Path) -> None:
        continuation = "This continued from a previous conversation about X"
        result: list[Message] = [TextMessage(continuation, "text/plain")]
        state = self._state()
        with patch(
            "sagent.agent.compaction.reattach_files",
            new_callable=AsyncMock,
        ):
            await post_compact_enrich(
                result=result,
                messages=[_user_msg()],
                state=state,
                session_dir=tmp_path,
                tool_state=ToolState(),
                budget=self._budget(),
                tools={},
                background_tasks={},
                estimate_tokens=500,
                headroom=100,
            )
        sp = tmp_path / "summary_0.md"
        assert sp.exists()
        assert sp.read_text() == continuation
        assert len(state.summary_pointers) == 1
        assert state.summary_pointers[0] == (str(sp), extract_topic(continuation))

    @pytest.mark.asyncio
    async def test_no_session_dir_skips_save(self) -> None:
        continuation = "This continued from a previous conversation"
        result: list[Message] = [TextMessage(continuation, "text/plain")]
        state = self._state()
        with patch(
            "sagent.agent.compaction.reattach_files",
            new_callable=AsyncMock,
        ):
            await post_compact_enrich(
                result=result,
                messages=[_user_msg()],
                state=state,
                session_dir=None,
                tool_state=ToolState(),
                budget=self._budget(),
                tools={},
                background_tasks={},
                estimate_tokens=500,
                headroom=100,
            )
        assert state.summary_pointers == []

    @pytest.mark.asyncio
    async def test_calls_post_compact_restore_on_tools(self) -> None:
        mock_tool = MagicMock()
        mock_tool.tool_id = "application/x-test"
        mock_tool.post_compact_restore = AsyncMock()
        with patch(
            "sagent.agent.compaction.reattach_files",
            new_callable=AsyncMock,
        ):
            await post_compact_enrich(
                result=[],
                messages=[_user_msg()],
                state=self._state(),
                session_dir=None,
                tool_state=ToolState(),
                budget=self._budget(),
                tools={"test": mock_tool},
                background_tasks={},
                estimate_tokens=500,
                headroom=100,
            )
        mock_tool.post_compact_restore.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_hook_failure_logged_not_raised(self) -> None:
        mock_tool = MagicMock()
        mock_tool.tool_id = "application/x-bad"
        mock_tool.post_compact_restore = AsyncMock(side_effect=RuntimeError("boom"))
        with patch(
            "sagent.agent.compaction.reattach_files",
            new_callable=AsyncMock,
        ):
            await post_compact_enrich(
                result=[],
                messages=[_user_msg()],
                state=self._state(),
                session_dir=None,
                tool_state=ToolState(),
                budget=self._budget(),
                tools={"bad": mock_tool},
                background_tasks={},
                estimate_tokens=500,
                headroom=100,
            )

    @pytest.mark.asyncio
    @patch("sagent.agent.compaction.time")
    async def test_resurfaces_background_tasks(self, mock_time: MagicMock) -> None:
        mock_time.time.return_value = 300.0
        entry = _bg_entry(tool_name="Read", done=False, started=290.0)
        msgs = [_user_msg()]
        with patch(
            "sagent.agent.compaction.reattach_files",
            new_callable=AsyncMock,
        ):
            await post_compact_enrich(
                result=[],
                messages=msgs,
                state=self._state(),
                session_dir=None,
                tool_state=ToolState(),
                budget=self._budget(),
                tools={},
                background_tasks={"bg1": entry},
                estimate_tokens=500,
                headroom=100,
            )
        assert "bg1" in str(msgs[0].content)
        assert "Read" in str(msgs[0].content)

    @pytest.mark.asyncio
    async def test_skips_non_restorable_tools(self) -> None:
        mock_tool = MagicMock(spec=[])
        with patch(
            "sagent.agent.compaction.reattach_files",
            new_callable=AsyncMock,
        ):
            await post_compact_enrich(
                result=[],
                messages=[_user_msg()],
                state=self._state(),
                session_dir=None,
                tool_state=ToolState(),
                budget=self._budget(),
                tools={"plain": mock_tool},
                background_tasks={},
                estimate_tokens=500,
                headroom=100,
            )


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
