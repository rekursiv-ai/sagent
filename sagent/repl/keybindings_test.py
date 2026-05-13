"""Tests for ``repl.keybindings``: prompt-toolkit key handler bindings."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import cast
from unittest.mock import MagicMock

import asyncio

from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent

import pytest

from sagent.agent.agent import Agent
from sagent.agent.runtime import UserQueuedMessage
from sagent.repl.keybindings import (
    _IDLE_COMMIT_DELAY_SEC,
    build_key_bindings,
)


@dataclass(slots=True, kw_only=True)
class _FakeInbox:
    items: list[object] = field(default_factory=list)

    def push_back(self, item: object) -> None:
        self.items.append(item)


@dataclass(slots=True, kw_only=True)
class _FakeRuntime:
    inbox: _FakeInbox = field(default_factory=_FakeInbox)
    cohort: set[str] = field(default_factory=set)


@dataclass(slots=True, kw_only=True)
class _FakeAgent:
    """Minimal stand-in for ``Agent`` matching only the surface keybindings touch."""

    work: object = None
    runtime: _FakeRuntime = field(default_factory=_FakeRuntime)
    halt_calls: int = 0

    def halt(self) -> None:
        self.halt_calls += 1


def _idle_agent() -> _FakeAgent:
    return _FakeAgent()


def _busy_agent() -> _FakeAgent:
    a = _FakeAgent()
    a.work = object()  # Truthy placeholder for a running task.
    return a


def _cohort_agent() -> _FakeAgent:
    a = _FakeAgent()
    a.runtime.cohort.add("c1")
    return a


def _handler(kb: KeyBindings, keys: tuple[str, ...]) -> Callable[[KeyPressEvent], None]:
    """Return the registered handler for ``keys``.

    ``enter`` is registered as ``Keys.ControlM`` at runtime, so callers that
    pass ``"enter"`` are normalized to ``c-m``.
    """
    normalized = tuple("c-m" if k == "enter" else k for k in keys)
    for b in kb.bindings:
        bk = tuple(getattr(k, "value", k) for k in b.keys)
        if bk == normalized:
            return cast(Callable[[KeyPressEvent], None], b.handler)
    raise AssertionError(f"no binding for {keys!r}")


def _fake_buf(text: str = "", cursor: int | None = None) -> MagicMock:
    buf = MagicMock()
    buf.text = text
    buf.cursor_position = cursor if cursor is not None else len(text)
    buf.working_index = 0
    buf.document.text_before_cursor = text[: buf.cursor_position]
    buf.document.text = text
    return buf


def _fake_event(
    buf: MagicMock | None = None, app: MagicMock | None = None
) -> MagicMock:
    ev = MagicMock()
    ev.current_buffer = buf
    ev.app = app
    return ev


def _build(agent: _FakeAgent, queued_input: list[str] | None = None) -> KeyBindings:
    """Bind the keybindings against a fake agent; ``Agent`` cast is structural."""
    return build_key_bindings(
        cast(Agent, agent), queued_input if queued_input is not None else []
    )


def test_enter_text_stages_into_queue_without_dispatch() -> None:
    """Enter on non-empty buffer appends to queue, leaves inbox untouched."""
    agent = _idle_agent()
    queued_input: list[str] = []
    kb = _build(agent, queued_input)
    buf = _fake_buf("hello")
    _handler(kb, ("enter",))(cast(KeyPressEvent, _fake_event(buf)))
    assert queued_input == ["hello"]
    assert agent.runtime.inbox.items == []
    buf.append_to_history.assert_called_once()
    buf.reset.assert_called_once()
    buf.validate_and_handle.assert_not_called()


def test_enter_text_during_model_call_stages_without_dispatch() -> None:
    """Busy-state behavior identical to idle: just stage; runtime not touched."""
    agent = _busy_agent()
    queued_input: list[str] = []
    kb = _build(agent, queued_input)
    buf = _fake_buf("first msg")
    _handler(kb, ("enter",))(cast(KeyPressEvent, _fake_event(buf)))
    assert queued_input == ["first msg"]
    assert agent.runtime.inbox.items == []
    buf.reset.assert_called_once()


def test_enter_text_during_tool_cohort_stages_without_dispatch() -> None:
    agent = _cohort_agent()
    queued_input: list[str] = []
    kb = _build(agent, queued_input)
    buf = _fake_buf("during tools")
    _handler(kb, ("enter",))(cast(KeyPressEvent, _fake_event(buf)))
    assert queued_input == ["during tools"]
    assert agent.runtime.inbox.items == []


def test_enter_text_appends_block_to_existing_queue() -> None:
    """Each Enter creates a new block; queue accumulates in order."""
    agent = _idle_agent()
    queued_input: list[str] = ["a", "b"]
    kb = _build(agent, queued_input)
    buf = _fake_buf("c")
    _handler(kb, ("enter",))(cast(KeyPressEvent, _fake_event(buf)))
    assert queued_input == ["a", "b", "c"]


def test_enter_on_empty_buf_with_queue_commits_user_queued_message() -> None:
    r"""Empty Enter with non-empty queue is the explicit commit gesture.

    Blocks join with ``\\n\\n`` into a single ``UserQueuedMessage``.
    """
    agent = _idle_agent()
    queued_input: list[str] = ["alpha", "beta"]
    kb = _build(agent, queued_input)
    buf = _fake_buf("")
    _handler(kb, ("enter",))(cast(KeyPressEvent, _fake_event(buf)))
    assert queued_input == []
    assert len(agent.runtime.inbox.items) == 1
    pushed = agent.runtime.inbox.items[0]
    assert isinstance(pushed, UserQueuedMessage)
    assert pushed.text == "alpha\n\nbeta"


def test_enter_on_empty_buf_with_empty_queue_is_noop() -> None:
    agent = _idle_agent()
    queued_input: list[str] = []
    kb = _build(agent, queued_input)
    buf = _fake_buf("")
    _handler(kb, ("enter",))(cast(KeyPressEvent, _fake_event(buf)))
    assert queued_input == []
    assert agent.runtime.inbox.items == []


def test_enter_routes_slash_command_through_pump_when_busy() -> None:
    """Slash always routes through PT so the pump's dispatcher sees it."""
    agent = _busy_agent()
    queued_input: list[str] = []
    kb = _build(agent, queued_input)
    buf = _fake_buf("/model")
    _handler(kb, ("enter",))(cast(KeyPressEvent, _fake_event(buf)))
    buf.validate_and_handle.assert_called_once()
    assert agent.runtime.inbox.items == []
    assert queued_input == []


def test_enter_whitespace_discards_buffer_does_not_commit() -> None:
    """Whitespace-only Enter resets the buffer; no commit even if queue
    is non-empty (commit requires *empty*-buffer Enter).
    """
    agent = _idle_agent()
    queued_input: list[str] = ["staged"]
    kb = _build(agent, queued_input)
    buf = _fake_buf("   ")
    _handler(kb, ("enter",))(cast(KeyPressEvent, _fake_event(buf)))
    assert queued_input == ["staged"]
    assert agent.runtime.inbox.items == []
    buf.reset.assert_called_once()


def test_down_on_non_empty_buf_acts_like_enter() -> None:
    """Down with text in input_pane stages it (same as Enter)."""
    agent = _idle_agent()
    queued_input: list[str] = []
    kb = _build(agent, queued_input)
    buf = _fake_buf("hello")
    _handler(kb, ("down",))(cast(KeyPressEvent, _fake_event(buf)))
    assert queued_input == ["hello"]
    assert agent.runtime.inbox.items == []


def test_down_on_empty_buf_is_noop() -> None:
    """Down with empty input_pane does nothing (reserved for future submenu)."""
    agent = _idle_agent()
    queued_input: list[str] = ["staged"]
    kb = _build(agent, queued_input)
    buf = _fake_buf("")
    _handler(kb, ("down",))(cast(KeyPressEvent, _fake_event(buf)))
    assert queued_input == ["staged"]
    assert agent.runtime.inbox.items == []


def test_alt_enter_inserts_newline() -> None:
    kb = _build(_idle_agent())
    buf = _fake_buf("abc")
    _handler(kb, ("escape", "enter"))(cast(KeyPressEvent, _fake_event(buf)))
    buf.insert_text.assert_called_once_with("\n")


def test_up_lifts_entire_queue_into_buffer() -> None:
    r"""Up with non-empty queue moves ALL blocks (joined ``\\n\\n``) to buf."""
    queued_input = ["a", "b", "c"]
    kb = _build(_idle_agent(), queued_input)
    buf = _fake_buf("")
    _handler(kb, ("up",))(cast(KeyPressEvent, _fake_event(buf)))
    assert buf.text == "a\n\nb\n\nc"
    assert queued_input == []
    buf.history_backward.assert_not_called()


def test_up_falls_back_to_history() -> None:
    kb = _build(_idle_agent(), [])
    buf = _fake_buf("typed")
    _handler(kb, ("up",))(cast(KeyPressEvent, _fake_event(buf)))
    buf.history_backward.assert_called_once_with(count=1)


def test_s_up_walks_history_until_prefix_matches() -> None:
    kb = _build(_idle_agent())

    class Buf:
        def __init__(self) -> None:
            self.working_index = 0
            self._history = ["foo bar", "frob", "foobaz"]
            self.document = MagicMock()
            self.document.text_before_cursor = "foo"
            self.document.text = "foo"

        def history_backward(self, count: int = 1) -> None:
            del count
            self.working_index += 1
            if self.working_index <= len(self._history):
                self.document.text = self._history[self.working_index - 1]

    buf = Buf()
    _handler(kb, ("s-up",))(cast(KeyPressEvent, _fake_event(cast(MagicMock, buf))))
    assert buf.working_index >= 1


def test_s_down_walks_forward_until_stall() -> None:
    kb = _build(_idle_agent())

    class Buf:
        def __init__(self) -> None:
            self.working_index = 3
            self.document = MagicMock()
            self.document.text_before_cursor = "z"
            self.document.text = "z"

        def history_forward(self, count: int = 1) -> None:
            del count

    buf = Buf()
    _handler(kb, ("s-down",))(cast(KeyPressEvent, _fake_event(cast(MagicMock, buf))))


def test_open_editor() -> None:
    kb = _build(_idle_agent())
    buf = _fake_buf()
    _handler(kb, ("c-x", "c-e"))(cast(KeyPressEvent, _fake_event(buf)))
    buf.open_in_editor.assert_called_once()


def test_ctrl_z_suspends() -> None:
    kb = _build(_idle_agent())
    app = MagicMock()
    _handler(kb, ("c-z",))(cast(KeyPressEvent, _fake_event(app=app)))
    app.suspend_to_background.assert_called_once()


def test_ctrl_underscore_undo() -> None:
    kb = _build(_idle_agent())
    buf = _fake_buf("ab")
    _handler(kb, ("c-_",))(cast(KeyPressEvent, _fake_event(buf)))
    buf.undo.assert_called_once()


def test_escape_z_undo() -> None:
    kb = _build(_idle_agent())
    buf = _fake_buf("ab")
    _handler(kb, ("escape", "z"))(cast(KeyPressEvent, _fake_event(buf)))
    buf.undo.assert_called_once()


def test_ctrl_c_halts_during_model_call() -> None:
    agent = _busy_agent()
    kb = _build(agent)
    buf = _fake_buf("partial line")
    _handler(kb, ("c-c",))(cast(KeyPressEvent, _fake_event(buf)))
    assert agent.halt_calls == 1
    # Buffer is not reset on halt path; only on the idle-clear path.
    buf.reset.assert_not_called()


def test_ctrl_c_halts_during_cohort() -> None:
    agent = _cohort_agent()
    kb = _build(agent)
    buf = _fake_buf("x")
    _handler(kb, ("c-c",))(cast(KeyPressEvent, _fake_event(buf)))
    assert agent.halt_calls == 1


def test_ctrl_c_idle_clears_buffer() -> None:
    agent = _idle_agent()
    kb = _build(agent)
    buf = _fake_buf("typing here")
    _handler(kb, ("c-c",))(cast(KeyPressEvent, _fake_event(buf)))
    assert agent.halt_calls == 0
    buf.reset.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.real_sleep
async def test_idle_text_enter_commits_after_delay() -> None:
    """Idle Enter on text stages, then auto-commits after the delay window.

    Sequence:
    - Type text, Enter while agent is idle.
    - Immediately: text is staged in queue; nothing pushed yet.
    - After ``_IDLE_COMMIT_DELAY_SEC``: queue committed as
      ``UserQueuedMessage``; queue cleared.
    """
    agent = _idle_agent()
    queued_input: list[str] = []
    kb = _build(agent, queued_input)
    buf = _fake_buf("hello")
    _handler(kb, ("enter",))(cast(KeyPressEvent, _fake_event(buf)))
    # Synchronously: staged, nothing pushed.
    assert queued_input == ["hello"]
    assert agent.runtime.inbox.items == []
    # Wait past the commit window plus a small margin.
    await asyncio.sleep(_IDLE_COMMIT_DELAY_SEC + 0.05)
    # After delay: queue committed.
    assert queued_input == []
    assert len(agent.runtime.inbox.items) == 1
    pushed = agent.runtime.inbox.items[0]
    assert isinstance(pushed, UserQueuedMessage)
    assert pushed.text == "hello"


@pytest.mark.asyncio
@pytest.mark.real_sleep
async def test_idle_enter_then_up_arrow_retracts_before_commit() -> None:
    """Up-arrow within the idle-commit window retracts the staging."""
    agent = _idle_agent()
    queued_input: list[str] = []
    kb = _build(agent, queued_input)
    buf = _fake_buf("hello")
    _handler(kb, ("enter",))(cast(KeyPressEvent, _fake_event(buf)))
    # User up-arrows before the commit fires.
    buf2 = _fake_buf("")
    _handler(kb, ("up",))(cast(KeyPressEvent, _fake_event(buf2)))
    assert buf2.text == "hello"
    assert queued_input == []
    # Wait past the commit window.
    await asyncio.sleep(_IDLE_COMMIT_DELAY_SEC + 0.05)
    # Commit should be a no-op (queue was emptied by retract).
    assert agent.runtime.inbox.items == []


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
