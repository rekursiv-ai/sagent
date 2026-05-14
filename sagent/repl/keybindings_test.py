"""Tests for ``repl.keybindings``: prompt-toolkit key handler bindings."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import cast
from unittest.mock import MagicMock

from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent

from sagent.agent.agent import Agent
from sagent.agent.runtime import UserMessage
from sagent.repl.keybindings import build_key_bindings


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
    r"""Return the registered handler for ``keys``.

    ``enter`` is registered as ``Keys.ControlM`` (``\r`` = Ctrl+M) and
    ``tab`` as ``Keys.ControlI`` (``\t`` = Ctrl+I) at runtime. Callers
    using the friendly names get normalized.
    """
    aliases = {"enter": "c-m", "tab": "c-i"}
    normalized = tuple(aliases.get(k, k) for k in keys)
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


def test_enter_text_pushes_user_message_does_not_touch_queued_input() -> None:
    """Enter on text dispatches ``UserMessage``; ``queued_input`` untouched.

    Under option 1: ``queued_input`` belongs exclusively to Tab staging.
    Enter is a direct-dispatch path that bypasses the queue entirely.
    """
    agent = _idle_agent()
    queued_input: list[str] = []
    kb = _build(agent, queued_input)
    buf = _fake_buf("hello")
    _handler(kb, ("enter",))(cast(KeyPressEvent, _fake_event(buf)))
    assert queued_input == [], "Enter must not touch the Tab-staging queue"
    assert len(agent.runtime.inbox.items) == 1
    pushed = agent.runtime.inbox.items[0]
    assert isinstance(pushed, UserMessage)
    assert pushed.text == "hello"
    buf.reset.assert_called_once()


def test_enter_text_during_model_call_does_not_touch_queued_input() -> None:
    """Busy-state Enter: dispatch ``UserMessage``, queue still belongs to Tab."""
    agent = _busy_agent()
    queued_input: list[str] = ["staged-by-tab"]
    kb = _build(agent, queued_input)
    buf = _fake_buf("first msg")
    _handler(kb, ("enter",))(cast(KeyPressEvent, _fake_event(buf)))
    # Tab-staged content untouched.
    assert queued_input == ["staged-by-tab"]
    assert len(agent.runtime.inbox.items) == 1
    pushed = agent.runtime.inbox.items[0]
    assert isinstance(pushed, UserMessage)
    assert pushed.text == "first msg"


def test_enter_text_during_tool_cohort_pushes_user_message_immediately() -> None:
    """Cohort-active: text Enter pushes ``UserMessage`` to preempt."""
    agent = _cohort_agent()
    queued_input: list[str] = []
    kb = _build(agent, queued_input)
    buf = _fake_buf("during tools")
    _handler(kb, ("enter",))(cast(KeyPressEvent, _fake_event(buf)))
    assert queued_input == []
    assert len(agent.runtime.inbox.items) == 1
    pushed = agent.runtime.inbox.items[0]
    assert isinstance(pushed, UserMessage)


def test_enter_on_empty_buf_is_noop() -> None:
    """Empty Enter does nothing -- no commit gesture in the new model.

    Each text Enter dispatches immediately; there's no accumulated
    queue to commit on empty Enter.
    """
    agent = _idle_agent()
    queued_input: list[str] = ["leftover"]
    kb = _build(agent, queued_input)
    buf = _fake_buf("")
    _handler(kb, ("enter",))(cast(KeyPressEvent, _fake_event(buf)))
    assert queued_input == ["leftover"]
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


def test_enter_whitespace_discards_buffer() -> None:
    """Whitespace-only Enter resets the buffer; nothing dispatches."""
    agent = _idle_agent()
    queued_input: list[str] = []
    kb = _build(agent, queued_input)
    buf = _fake_buf("   ")
    _handler(kb, ("enter",))(cast(KeyPressEvent, _fake_event(buf)))
    assert queued_input == []
    assert agent.runtime.inbox.items == []
    buf.reset.assert_called_once()


def test_enter_with_trailing_backslash_inserts_newline_no_dispatch() -> None:
    r"""Backslash continuation: trailing ``\`` + Enter swaps for ``\n``.

    Buffer becomes the same text with the trailing ``\`` replaced by
    a literal newline; nothing is dispatched. Cursor sits at end of
    the new buffer so subsequent typing continues on the new line.
    """
    agent = _idle_agent()
    queued_input: list[str] = []
    kb = _build(agent, queued_input)
    buf = _fake_buf("first line\\")
    _handler(kb, ("enter",))(cast(KeyPressEvent, _fake_event(buf)))
    assert buf.text == "first line\n"
    assert buf.cursor_position == len("first line\n")
    assert queued_input == []
    assert agent.runtime.inbox.items == []


def test_tab_stages_locally_does_not_dispatch() -> None:
    """Tab on text stages in ``queued_input``; nothing pushed to runtime.

    Under option 1: Tab is pure REPL-side staging. The runtime is
    untouched until ``make_queued_input_committer`` fires on
    ``ModelIdle``. This makes Up-arrow a true retract.
    """
    agent = _idle_agent()
    queued_input: list[str] = []
    kb = _build(agent, queued_input)
    buf = _fake_buf("for later")
    _handler(kb, ("tab",))(cast(KeyPressEvent, _fake_event(buf)))
    assert queued_input == ["for later"]
    assert agent.runtime.inbox.items == [], (
        "Tab must not push to runtime -- it stages REPL-side until ModelIdle"
    )


def test_tab_then_up_arrow_truly_retracts() -> None:
    """Reverse of the earlier bug: Tab then Up retracts cleanly.

    Under option 1: Tab stages locally; Up lifts back; nothing was
    in the runtime, so no duplicate dispatch on subsequent Enter.
    """
    agent = _idle_agent()
    queued_input: list[str] = []
    kb = _build(agent, queued_input)

    # Tab "hello" -> stage.
    buf1 = _fake_buf("hello")
    _handler(kb, ("tab",))(cast(KeyPressEvent, _fake_event(buf1)))
    assert queued_input == ["hello"]
    assert agent.runtime.inbox.items == []

    # Up -> lift back to buffer, clear local list.
    buf2 = _fake_buf("")
    _handler(kb, ("up",))(cast(KeyPressEvent, _fake_event(buf2)))
    assert buf2.text == "hello"
    assert queued_input == []
    assert agent.runtime.inbox.items == []

    # Edit + Enter -> single UserMessage, no duplicate.
    buf3 = _fake_buf("hello world")
    _handler(kb, ("enter",))(cast(KeyPressEvent, _fake_event(buf3)))
    items = agent.runtime.inbox.items
    assert len(items) == 1
    assert isinstance(items[0], UserMessage)
    assert items[0].text == "hello world"


def test_tab_on_empty_buf_is_noop() -> None:
    """Tab with empty buffer does nothing; queue untouched."""
    agent = _idle_agent()
    queued_input: list[str] = []
    kb = _build(agent, queued_input)
    buf = _fake_buf("")
    _handler(kb, ("tab",))(cast(KeyPressEvent, _fake_event(buf)))
    assert queued_input == []
    assert agent.runtime.inbox.items == []


def test_down_on_non_empty_buf_clears_input_without_staging() -> None:
    """Down with text discards the buffer; never stages, never dispatches.

    Per ``repl.input_pane``'s contract: Up navigates queue/history INTO
    the input, Down brings the input back to empty -- without staging
    or dispatching. Only Enter stages/dispatches. This guards against
    the "Up into history, then Down dispatches it" regression.
    """
    agent = _idle_agent()
    queued_input: list[str] = []
    kb = _build(agent, queued_input)
    buf = _fake_buf("hello")
    _handler(kb, ("down",))(cast(KeyPressEvent, _fake_event(buf)))
    buf.reset.assert_called_once()
    assert queued_input == []
    assert agent.runtime.inbox.items == []


def test_down_on_empty_buf_is_noop_reserved_for_future_submenu() -> None:
    """Down with empty input does nothing; queue untouched."""
    agent = _idle_agent()
    queued_input: list[str] = ["staged"]
    kb = _build(agent, queued_input)
    buf = _fake_buf("")
    _handler(kb, ("down",))(cast(KeyPressEvent, _fake_event(buf)))
    buf.reset.assert_not_called()
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


def test_up_arrow_on_tab_staging_lifts_back_truly_retracts() -> None:
    """Up-arrow lifts Tab-staged content back to buffer; runtime untouched."""
    agent = _busy_agent()
    queued_input: list[str] = []
    kb = _build(agent, queued_input)
    # Tab "draft" -> stage locally; runtime untouched.
    buf = _fake_buf("draft")
    _handler(kb, ("tab",))(cast(KeyPressEvent, _fake_event(buf)))
    assert queued_input == ["draft"]
    assert agent.runtime.inbox.items == []
    # Up-arrow lifts back -- true retract.
    buf2 = _fake_buf("")
    _handler(kb, ("up",))(cast(KeyPressEvent, _fake_event(buf2)))
    assert buf2.text == "draft"
    assert queued_input == []
    assert agent.runtime.inbox.items == []


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
