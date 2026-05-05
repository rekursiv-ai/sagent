"""Tests for repl.handlers.replay_messages."""

from __future__ import annotations

from io import StringIO
from typing import Any
from unittest.mock import MagicMock

from rich.console import Console

from sagent.custom_types import (
    JsonMessage,
    MultipartMessage,
    TextMessage,
)
from sagent.lib.json import json_freeze
from sagent.repl.handlers import replay_messages


def _fake_console() -> MagicMock:
    c = MagicMock()
    c.width = 80
    return c


class TestReplayMessages:
    def _mk_agent(self, messages: list[Any]) -> MagicMock:
        a = MagicMock()
        a.messages = messages
        a.total_cost_usd = 0.0
        return a

    def test_empty_noop(self) -> None:
        agent = self._mk_agent([])
        console = _fake_console()
        out = _fake_console()
        replay_messages(agent, console, out)
        assert console.print.call_count == 0
        assert out.print.call_count == 0

    def test_renders_user_assistant_tool(self) -> None:
        msgs: list[Any] = [
            TextMessage("hi", "text/x-user-message"),
            MultipartMessage(
                (
                    TextMessage("hello back", "text/plain"),
                    MultipartMessage(
                        (
                            TextMessage("t1", "text/x-queue-id"),
                            JsonMessage(
                                json_freeze({"cmd": "ls"}),
                                "application/x-tool-bash",
                            ),
                        ),
                        "multipart/x-tool-call",
                    ),
                ),
                "multipart/x-model-message",
            ),
            MultipartMessage(
                (
                    TextMessage("t1", "text/x-queue-id"),
                    TextMessage("ok", "text/plain"),
                ),
                "multipart/x-tool-result",
            ),
            MultipartMessage(
                (TextMessage("done", "text/plain"),),
                "multipart/x-model-message",
            ),
        ]
        agent = self._mk_agent(msgs)
        console_buf, out_buf = StringIO(), StringIO()
        console = Console(file=console_buf, width=80, force_terminal=False)
        out = Console(file=out_buf, width=80, force_terminal=False)
        replay_messages(agent, console, out)
        out_text = out_buf.getvalue()
        console_text = console_buf.getvalue()
        assert "hi" in out_text
        assert "hello back" in out_text
        assert "done" in out_text
        assert "bash" in console_text.lower()
        assert "resumed" in console_text
        assert "4 messages" in console_text

    def test_error_result_and_hint(self) -> None:
        msgs: list[Any] = [
            MultipartMessage(
                (
                    TextMessage("actual error text", "text/x-error"),
                    TextMessage("prefer Grep", "text/x-hint-tool-use-nudge"),
                ),
                "multipart/x-tool-result",
            ),
        ]
        agent = self._mk_agent(msgs)
        console_buf, out_buf = StringIO(), StringIO()
        console = Console(file=console_buf, width=120, force_terminal=False)
        out = Console(file=out_buf, width=120, force_terminal=False)
        replay_messages(agent, console, out)
        console_text = console_buf.getvalue()
        assert "✗" in console_text
        assert "prefer Grep" in console_text
        assert "hint:" in console_text

    def test_cost_in_separator(self) -> None:
        agent = self._mk_agent([TextMessage("hi", "text/x-user-message")])
        agent.total_cost_usd = 1.2345
        console_buf, out_buf = StringIO(), StringIO()
        console = Console(file=console_buf, width=80, force_terminal=False)
        out = Console(file=out_buf, width=80, force_terminal=False)
        replay_messages(agent, console, out)
        assert "$1.23" in console_buf.getvalue()


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
