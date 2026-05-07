"""Tests for repl.keybindings."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from sagent.custom_types import Message, TextMessage
from sagent.lib.asyncio_collections import Deque
from sagent.repl.keybindings import build_key_bindings


def _kb_handler(kb: Any, keys: tuple[str, ...]) -> Any:
    """Return the first handler whose registered keys match ``keys``.

    ``b.keys`` is a tuple of ``Keys`` enum values whose ``.value`` is
    the string identifier ("c-c", "up", etc.). ``enter`` maps to
    ``Keys.ControlM`` at runtime.
    """
    normalized = tuple("c-m" if k == "enter" else k for k in keys)
    for b in kb.bindings:
        if tuple(k.value if hasattr(k, "value") else k for k in b.keys) == normalized:
            return b.handler
    raise AssertionError(f"no binding for {keys!r}")


def _fake_buf(text: str = "", cursor: int = 0) -> Any:
    buf = MagicMock()
    buf.text = text
    buf.cursor_position = cursor
    buf.working_index = 0
    buf.document.text_before_cursor = text[:cursor] if cursor else text
    buf.document.text = text
    return buf


def _fake_event(buf: Any = None, app: Any = None) -> Any:
    ev = MagicMock()
    ev.current_buffer = buf
    ev.app = app
    return ev


def _idle_agent() -> Any:
    a = MagicMock()
    a.active = False
    a.inbox = Deque[Message]()
    a.inflight = None
    return a


def _active_agent() -> Any:
    a = MagicMock()
    a.active = True
    a.inbox = Deque[Message]()
    a.tool_state = MagicMock()
    task = MagicMock()
    task.done.return_value = False
    a.inflight = task
    return a


class TestKeyBindings:
    """Exercise each key handler in ``build_key_bindings``."""

    def test_enter_submits_when_idle(self) -> None:
        kb = build_key_bindings(_idle_agent())
        buf = _fake_buf("hello")
        _kb_handler(kb, ("enter",))(_fake_event(buf))
        buf.validate_and_handle.assert_called_once()

    def test_enter_queues_during_request(self) -> None:
        agent = _active_agent()
        kb = build_key_bindings(agent)
        buf = _fake_buf("first msg")
        _kb_handler(kb, ("enter",))(_fake_event(buf))
        tail = agent.inbox.peek_tail()
        assert tail is not None
        assert tail.content == "first msg"
        buf.reset.assert_called_once()
        buf.validate_and_handle.assert_not_called()

    def test_enter_appends_to_existing_queue(self) -> None:
        agent = _active_agent()
        agent.inbox.put(TextMessage("prior", "text/x-user-message"))
        kb = build_key_bindings(agent)
        buf = _fake_buf("more")
        _kb_handler(kb, ("enter",))(_fake_event(buf))
        drained = agent.inbox.drain()
        assert [m.content for m in drained] == ["prior", "more"]

    def test_enter_empty_buffer_during_request_noop(self) -> None:
        agent = _active_agent()
        kb = build_key_bindings(agent)
        buf = _fake_buf("   ")
        _kb_handler(kb, ("enter",))(_fake_event(buf))
        assert not agent.inbox
        buf.reset.assert_not_called()
        buf.validate_and_handle.assert_not_called()

    def test_enter_clear_during_request_sets_clear_flag(self) -> None:
        agent = _active_agent()
        kb = build_key_bindings(agent)
        buf = _fake_buf("/clear fresh start")
        _kb_handler(kb, ("enter",))(_fake_event(buf))
        drained = agent.inbox.drain()
        assert len(drained) == 1
        assert drained[0].content == "fresh start"
        assert drained[0].descriptor == "text/x-clear-request"
        agent.tool_state.clear_requested.assert_not_called()
        buf.append_to_history.assert_called_once()
        buf.reset.assert_called_once()
        buf.validate_and_handle.assert_not_called()

    def test_enter_quit_bypasses_queue(self) -> None:
        agent = _active_agent()
        kb = build_key_bindings(agent)
        buf = _fake_buf("quit")
        _kb_handler(kb, ("enter",))(_fake_event(buf))
        buf.validate_and_handle.assert_called_once()

    def test_enter_exit_bypasses_queue(self) -> None:
        agent = _active_agent()
        kb = build_key_bindings(agent)
        buf = _fake_buf("EXIT")
        _kb_handler(kb, ("enter",))(_fake_event(buf))
        buf.validate_and_handle.assert_called_once()

    def test_alt_enter_inserts_newline(self) -> None:
        kb = build_key_bindings()
        buf = _fake_buf("abc")
        _kb_handler(kb, ("escape", "enter"))(_fake_event(buf))
        buf.insert_text.assert_called_once_with("\n")

    def test_up_lifts_queue_when_empty(self) -> None:
        agent = _idle_agent()
        agent.inbox.put(TextMessage("queued text", "text/x-user-message"))
        kb = build_key_bindings(agent)
        buf = _fake_buf("")
        _kb_handler(kb, ("up",))(_fake_event(buf))
        assert buf.text == "queued text"
        assert not agent.inbox
        buf.history_backward.assert_not_called()

    def test_up_falls_back_to_history(self) -> None:
        kb = build_key_bindings(_idle_agent())
        buf = _fake_buf("typed")
        _kb_handler(kb, ("up",))(_fake_event(buf))
        buf.history_backward.assert_called_once_with(count=1)

    def test_s_up_walks_history_until_prefix_matches(self) -> None:
        kb = build_key_bindings()

        class Buf:
            def __init__(self) -> None:
                self.working_index = 0
                self._history = ["foo bar", "frob", "foobaz"]
                self._cursor = 0
                self.document = MagicMock()
                self.document.text_before_cursor = "foo"
                self.document.text = "foo"

            def history_backward(self, count: int = 1) -> None:
                del count
                self.working_index += 1
                if self.working_index <= len(self._history):
                    self.document.text = self._history[self.working_index - 1]

        buf = Buf()
        _kb_handler(kb, ("s-up",))(_fake_event(buf))
        assert buf.working_index >= 1

    def test_s_down_walks_forward_until_stall(self) -> None:
        kb = build_key_bindings()

        class Buf:
            def __init__(self) -> None:
                self.working_index = 3
                self.document = MagicMock()
                self.document.text_before_cursor = "z"
                self.document.text = "z"

            def history_forward(self, count: int = 1) -> None:
                del count

        buf = Buf()
        _kb_handler(kb, ("s-down",))(_fake_event(buf))

    def test_open_editor_idle(self) -> None:
        kb = build_key_bindings(_idle_agent())
        buf = _fake_buf()
        _kb_handler(kb, ("c-x", "c-e"))(_fake_event(buf))
        buf.open_in_editor.assert_called_once()

    def test_open_editor_during_request(self) -> None:
        kb = build_key_bindings(_active_agent())
        buf = _fake_buf()
        _kb_handler(kb, ("c-x", "c-e"))(_fake_event(buf))
        buf.open_in_editor.assert_called_once()

    def test_ctrl_z_suspends(self) -> None:
        kb = build_key_bindings(_idle_agent())
        app = MagicMock()
        _kb_handler(kb, ("c-z",))(_fake_event(app=app))
        app.suspend_to_background.assert_called_once()

    def test_ctrl_slash_calls_undo(self) -> None:
        kb = build_key_bindings()
        buf = _fake_buf("ab")
        _kb_handler(kb, ("c-_",))(_fake_event(buf))
        buf.undo.assert_called_once()

    def test_escape_z_calls_undo(self) -> None:
        kb = build_key_bindings()
        buf = _fake_buf("ab")
        _kb_handler(kb, ("escape", "z"))(_fake_event(buf))
        buf.undo.assert_called_once()

    def test_ctrl_c_cancels_task_during_request(self) -> None:
        agent = _active_agent()
        kb = build_key_bindings(agent)
        _kb_handler(kb, ("c-c",))(_fake_event(buf=None))
        agent.inflight.cancel.assert_called_once()

    def test_ctrl_c_done_task_no_cancel(self) -> None:
        agent = _active_agent()
        agent.inflight.done.return_value = True
        kb = build_key_bindings(agent)
        _kb_handler(kb, ("c-c",))(_fake_event(buf=None))
        agent.inflight.cancel.assert_not_called()

    def test_ctrl_c_no_inflight_no_cancel(self) -> None:
        agent = _active_agent()
        agent.inflight = None
        kb = build_key_bindings(agent)
        _kb_handler(kb, ("c-c",))(_fake_event(buf=None))

    def test_ctrl_c_exits_when_idle(self) -> None:
        kb = build_key_bindings(_idle_agent())
        app = MagicMock()
        _kb_handler(kb, ("c-c",))(_fake_event(buf=None, app=app))
        app.exit.assert_called_once()


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
