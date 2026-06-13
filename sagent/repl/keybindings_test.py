"""Tests for ``repl.keybindings``: the input_ux.md navigation + dispatch model."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import cast
from unittest.mock import MagicMock

import asyncio

from prompt_toolkit.buffer import Buffer
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent
from prompt_toolkit.key_binding.key_bindings import (
    KeyBindingsBase,
    merge_key_bindings,
)
from prompt_toolkit.keys import Keys
from prompt_toolkit.shortcuts import PromptSession

from sagent.agent import runtime as agent_runtime
from sagent.agent.agent import Agent
from sagent.repl import keybindings as keybindings_mod
from sagent.repl.input_queues import InputQueues, QueuedInputBlock
from sagent.repl.keybindings import NavState, build_key_bindings
from sagent.types.runtime import (
    AssistantMessage,
    BytesMessage,
    RuntimeEvent,
    UserDeferredMessage,
    UserMessage,
)


@dataclass(slots=True, kw_only=True)
class _ListInbox:
    """A list-backed inbox: pure storage, no dispatch logic to drift.

    The hidden-bug risk in earlier fakes was a *re-implemented predicate*
    (``accepts_user_dispatch`` etc.). An inbox has none -- it is a queue.
    So this dumb stand-in is safe: the dispatch predicates under test run
    on the REAL ``AgentRuntime`` (see ``_make_runtime``); only the queue
    is stubbed so tests can read ``items`` without draining a real
    ``GatedDeque``.
    """

    items: list[object] = field(default_factory=list)
    gate_armed: bool = False

    def push_back(self, item: object) -> None:
        self.items.append(item)

    def empty(self) -> bool:
        return not self.items


class _TrivialModel:
    """A model the runtime can hold but a keybinding test never invokes."""

    async def stream(
        self,
        history: object,
        on_text: Callable[[str], None],
        on_thinking: Callable[[str], None],
    ) -> AssistantMessage:
        del history, on_text, on_thinking
        return AssistantMessage(text="unused")


def _make_runtime() -> agent_runtime.AgentRuntime:
    """A REAL ``AgentRuntime`` with a list-backed inbox for poking state.

    The dispatch predicates (``is_idle`` / ``accepts_user_dispatch`` /
    ``accepts_deferred_dispatch``) are the real ones -- never copied --
    so a keybinding test cannot pass against a broken predicate.
    """
    runtime = agent_runtime.AgentRuntime(
        model=cast(agent_runtime.Model, _TrivialModel())
    )
    runtime.inbox = cast("agent_runtime.GatedDeque[RuntimeEvent]", _ListInbox())
    return runtime


def _sentinel_task() -> asyncio.Task[None]:
    """A stand-in ``Task`` for poking ``model_call`` / ``compact_task``.

    The dispatch predicates only test these for ``is not None``; a real
    ``Task`` is never awaited here, so an opaque sentinel typed as the
    field's declared type suffices and keeps the checkers honest.
    """
    return cast("asyncio.Task[None]", object())


@dataclass(slots=True, kw_only=True)
class _FakeAgent:
    """Minimal stand-in for ``Agent``; wraps a REAL runtime for predicates."""

    work: object = None
    runtime: agent_runtime.AgentRuntime = field(default_factory=_make_runtime)
    halt_calls: int = 0

    def halt(self) -> None:
        self.halt_calls += 1


def _inbox(agent: _FakeAgent) -> _ListInbox:
    return cast(_ListInbox, agent.runtime.inbox)


def _idle_agent() -> _FakeAgent:
    return _FakeAgent()


def _busy_agent() -> _FakeAgent:
    a = _FakeAgent()
    a.runtime.model_call = _sentinel_task()
    a.work = a.runtime.model_call
    return a


def _cohort_agent() -> _FakeAgent:
    a = _FakeAgent()
    a.runtime.cohort.add("c1")
    return a


def _handler(kb: KeyBindings, keys: tuple[str, ...]) -> Callable[[KeyPressEvent], None]:
    aliases = {"enter": "c-m", "tab": "c-i"}
    normalized = tuple(aliases.get(k, k) for k in keys)
    for b in kb.bindings:
        bk = tuple(getattr(k, "value", k) for k in b.keys)
        if bk == normalized:
            return cast(Callable[[KeyPressEvent], None], b.handler)
    raise AssertionError(f"no binding for {keys!r}")


def _fake_buf(
    text: str = "", cursor: int | None = None, history: list[str] | None = None
) -> MagicMock:
    buf = MagicMock()
    buf.text = text
    buf.cursor_position = cursor if cursor is not None else len(text)
    buf.working_index = 0
    buf.document.text_before_cursor = text[: buf.cursor_position]
    buf.document.text = text
    hist = list(history) if history else []
    buf.history.get_strings.return_value = hist
    return buf


def _fake_event(
    buf: MagicMock | None = None, app: MagicMock | None = None
) -> MagicMock:
    ev = MagicMock()
    ev.current_buffer = buf
    ev.app = app
    return ev


def _build(
    agent: _FakeAgent,
    queues: InputQueues | None = None,
    nav: NavState | None = None,
) -> KeyBindings:
    return build_key_bindings(
        cast(Agent, agent),
        queues if queues is not None else InputQueues(),
        nav,
    )


def _press(kb: KeyBindings, key: str, buf: MagicMock) -> None:
    """Press ``key`` against ``buf``; mirror prompt-toolkit's buffer mutation.

    The handlers set ``buf.text``; a real buffer would move the cursor and
    not auto-clear. The MagicMock retains assignments, so a sequence of
    presses sees the prior text -- matching live behavior closely enough
    to pin the navigation contract.
    """
    _handler(kb, (key,))(cast(KeyPressEvent, _fake_event(buf)))


# --- dispatch vs stage: pure function of is_idle ----------------------


def test_enter_idle_dispatches_user_message() -> None:
    agent = _idle_agent()
    queues = InputQueues()
    kb = _build(agent, queues)
    buf = _fake_buf("hello")
    _press(kb, "enter", buf)
    assert not queues.has_any()
    assert [type(i) for i in _inbox(agent).items] == [UserMessage]
    assert cast(UserMessage, _inbox(agent).items[0]).text == "hello"


def test_enter_busy_stages_into_queue_pane() -> None:
    agent = _busy_agent()
    queues = InputQueues()
    kb = _build(agent, queues)
    buf = _fake_buf("hello")
    _press(kb, "enter", buf)
    assert queues.queue == QueuedInputBlock(text="hello")
    assert queues.deferred is None
    assert _inbox(agent).items == []


def test_enter_busy_coalesces_into_queue_pane() -> None:
    agent = _busy_agent()
    queues = InputQueues()
    kb = _build(agent, queues)
    _press(kb, "enter", _fake_buf("alpha"))
    _press(kb, "enter", _fake_buf("beta"))
    assert queues.queue is not None
    assert queues.queue.text == "alpha\n\nbeta"


def test_enter_during_pending_halt_dispatches_not_stages() -> None:
    """REGRESSION (Bug 2 race): model_call set + Halt queued -> dispatch.

    Ctrl+C queues a Halt but the runtime has not drained it, so model_call
    is still set. Staging here would orphan the message: the imminent
    AWAIT_USER arm suppresses AgentIdle (the only commit edge), so the
    queue-staged block would never reach the model. A pending inbox item
    means mid-transition -> push directly.
    """
    agent = _busy_agent()
    _inbox(agent).items.append(object())  # a queued Halt, not drained
    queues = InputQueues()
    kb = _build(agent, queues)
    _press(kb, "enter", _fake_buf("redirect now"))
    assert not queues.has_any()
    pushed = [i for i in _inbox(agent).items if isinstance(i, UserMessage)]
    assert len(pushed) == 1
    assert pushed[0].text == "redirect now"


def test_enter_during_compaction_stages_not_dispatches() -> None:
    """Spec: busy is busy. Compaction stages, never dispatches (was a bug)."""
    agent = _idle_agent()
    agent.runtime.compact_task = _sentinel_task()
    queues = InputQueues()
    kb = _build(agent, queues)
    _press(kb, "enter", _fake_buf("during compact"))
    assert queues.queue == QueuedInputBlock(text="during compact")
    assert _inbox(agent).items == []


def test_enter_mid_cohort_dispatches_to_preempt() -> None:
    """REGRESSION: Enter while tools run dispatches (runtime detaches them).

    Mid-cohort (cohort non-empty, no model streaming) is NOT a staging
    state: a pushed UserMessage hits the runtime's preempt-and-detach
    path. Staging would never reach it, so the tools would run to
    completion -- the "type to redirect" path lost.
    """
    agent = _cohort_agent()
    queues = InputQueues()
    kb = _build(agent, queues)
    _press(kb, "enter", _fake_buf("during tools"))
    assert not queues.has_any()
    assert [type(i) for i in _inbox(agent).items] == [UserMessage]
    assert cast(UserMessage, _inbox(agent).items[0]).text == "during tools"


def test_tab_mid_cohort_stages_to_deferred_not_dispatch() -> None:
    """REGRESSION: Tab while tools run STAGES (defers), never dispatches.

    Tab's intent is "hold until the round chain goes idle." Mid-cohort is
    busy, so Tab must stage into the deferred pane -- unlike Enter, which
    dispatches mid-cohort to preempt. Dispatching a deferred message
    mid-cohort defeats the defer semantics.
    """
    agent = _cohort_agent()
    queues = InputQueues()
    kb = _build(agent, queues)
    _press(kb, "tab", _fake_buf("hold this"))
    assert queues.deferred == QueuedInputBlock(text="hold this")
    assert _inbox(agent).items == []


def test_tab_idle_dispatches_deferred_message() -> None:
    agent = _idle_agent()
    queues = InputQueues()
    kb = _build(agent, queues)
    _press(kb, "tab", _fake_buf("for later"))
    assert not queues.has_any()
    assert [type(i) for i in _inbox(agent).items] == [UserDeferredMessage]
    assert cast(UserDeferredMessage, _inbox(agent).items[0]).text == "for later"


def test_enter_when_gate_armed_dispatches_to_release() -> None:
    """REGRESSION (Bug 2): post-Halt Enter dispatches, never stages.

    An armed AWAIT_USER gate admits the user message and is released by
    it; staging would wedge (no AgentIdle fires while armed). Post-Halt
    the model call is already cancelled, so this is the realistic state:
    ``model_call is None`` with the gate armed.
    """
    agent = _idle_agent()
    _inbox(agent).gate_armed = True
    queues = InputQueues()
    kb = _build(agent, queues)
    _press(kb, "enter", _fake_buf("resume me"))
    assert not queues.has_any()
    assert [type(i) for i in _inbox(agent).items] == [UserMessage]
    assert cast(UserMessage, _inbox(agent).items[0]).text == "resume me"


def test_tab_when_gate_armed_dispatches_to_release() -> None:
    agent = _idle_agent()
    _inbox(agent).gate_armed = True
    queues = InputQueues()
    kb = _build(agent, queues)
    _press(kb, "tab", _fake_buf("resume deferred"))
    assert not queues.has_any()
    assert [type(i) for i in _inbox(agent).items] == [UserDeferredMessage]


def test_tab_busy_stages_into_deferred_pane() -> None:
    agent = _busy_agent()
    queues = InputQueues()
    kb = _build(agent, queues)
    _press(kb, "tab", _fake_buf("for later"))
    assert queues.deferred == QueuedInputBlock(text="for later")
    assert queues.queue is None
    assert _inbox(agent).items == []


def test_tab_busy_coalesces_into_deferred_pane() -> None:
    agent = _busy_agent()
    queues = InputQueues()
    kb = _build(agent, queues)
    _press(kb, "tab", _fake_buf("one"))
    _press(kb, "tab", _fake_buf("two"))
    assert queues.deferred is not None
    assert queues.deferred.text == "one\n\ntwo"


# --- empty input is a no-op on both keys ------------------------------


def test_enter_empty_is_noop() -> None:
    for agent in (_idle_agent(), _busy_agent()):
        queues = InputQueues()
        kb = _build(agent, queues)
        _press(kb, "enter", _fake_buf(""))
        assert not queues.has_any()
        assert _inbox(agent).items == []


def test_enter_whitespace_is_noop() -> None:
    for agent in (_idle_agent(), _busy_agent()):
        queues = InputQueues()
        kb = _build(agent, queues)
        _press(kb, "enter", _fake_buf("   "))
        assert not queues.has_any()
        assert _inbox(agent).items == []


def test_tab_empty_is_noop() -> None:
    for agent in (_idle_agent(), _busy_agent()):
        queues = InputQueues()
        kb = _build(agent, queues)
        _press(kb, "tab", _fake_buf(""))
        assert not queues.has_any()
        assert _inbox(agent).items == []


def test_tab_whitespace_is_noop() -> None:
    agent = _busy_agent()
    queues = InputQueues()
    kb = _build(agent, queues)
    _press(kb, "tab", _fake_buf("  \t "))
    assert not queues.has_any()
    assert _inbox(agent).items == []


# --- slash + backslash continuation -----------------------------------


def test_enter_slash_routes_through_pump() -> None:
    agent = _busy_agent()
    queues = InputQueues()
    kb = _build(agent, queues)
    buf = _fake_buf("/model")
    _press(kb, "enter", buf)
    buf.validate_and_handle.assert_called_once()
    assert _inbox(agent).items == []
    assert not queues.has_any()


def test_slash_during_navigation_restores_pane_and_ends_nav() -> None:
    """REGRESSION: a slash command mid-nav must not strand the lifted pane.

    Up lifts Q into the buffer; the user types ``/model`` and Enter. The
    command routes through the pump, Q returns to the queue pane, and
    navigation ends -- not stranded with Q lost and the cursor stuck.
    """
    agent = _busy_agent()
    queues = InputQueues(queue=QueuedInputBlock(text="Q"))
    nav = NavState()
    kb = _build(agent, queues, nav)
    buf = _fake_buf("g")
    _press(kb, "up", buf)  # lift Q
    buf.text = "/model"
    _press(kb, "enter", buf)
    buf.validate_and_handle.assert_called_once()
    assert queues.queue == QueuedInputBlock(text="Q")
    assert not nav.active()


def test_enter_trailing_backslash_inserts_newline_no_dispatch() -> None:
    agent = _idle_agent()
    queues = InputQueues()
    kb = _build(agent, queues)
    buf = _fake_buf("first line\\")
    _press(kb, "enter", buf)
    assert buf.text == "first line\n"
    assert not queues.has_any()
    assert _inbox(agent).items == []


# --- navigation: the Up/Down walk -------------------------------------


def test_up_with_nothing_is_noop() -> None:
    nav = NavState()
    kb = _build(_idle_agent(), InputQueues(), nav)
    buf = _fake_buf("typed", history=[])
    _press(kb, "up", buf)
    assert buf.text == "typed"
    assert not nav.active()


def test_up_unlifts_queue_into_input() -> None:
    """First Up lifts the queue message into the input; queue pane empties."""
    queues = InputQueues(queue=QueuedInputBlock(text="Q"))
    nav = NavState()
    kb = _build(_busy_agent(), queues, nav)
    buf = _fake_buf("g")
    _press(kb, "up", buf)
    assert buf.text == "Q"
    assert queues.queue is None
    assert nav.cursor == 1


def test_up_walks_queue_then_deferred_then_history() -> None:
    """Walk order: input -> queue -> deferred -> history (spec)."""
    queues = InputQueues(
        queue=QueuedInputBlock(text="Q"), deferred=QueuedInputBlock(text="D")
    )
    nav = NavState()
    kb = _build(_busy_agent(), queues, nav)
    buf = _fake_buf("g", history=["h1"])
    _press(kb, "up", buf)
    assert buf.text == "Q"  # queue stop
    _press(kb, "up", buf)
    assert buf.text == "D"  # deferred stop
    assert queues.queue == QueuedInputBlock(text="Q")  # restored on pass
    _press(kb, "up", buf)
    assert buf.text == "h1"  # history stop
    assert queues.deferred == QueuedInputBlock(text="D")  # restored on pass


def test_up_at_oldest_history_is_noop() -> None:
    nav = NavState()
    kb = _build(_idle_agent(), InputQueues(), nav)
    buf = _fake_buf("g", history=["old"])
    _press(kb, "up", buf)
    assert buf.text == "old"
    _press(kb, "up", buf)
    assert buf.text == "old"  # hard top -- no-op


def test_down_at_input_is_noop() -> None:
    nav = NavState()
    kb = _build(_idle_agent(), InputQueues(), nav)
    buf = _fake_buf("")
    _press(kb, "down", buf)
    assert nav.cursor == 0


def test_unedited_round_trip_restores_everything() -> None:
    """Up (unedited) then Down returns the queue pane and input verbatim."""
    queues = InputQueues(queue=QueuedInputBlock(text="Q"))
    nav = NavState()
    kb = _build(_busy_agent(), queues, nav)
    buf = _fake_buf("g", history=["h1"])
    _press(kb, "up", buf)  # input -> Q
    _press(kb, "up", buf)  # Q restored, -> h1
    assert queues.queue == QueuedInputBlock(text="Q")
    _press(kb, "down", buf)  # -> Q (queue emptied again)
    assert buf.text == "Q"
    assert queues.queue is None
    _press(kb, "down", buf)  # -> g; queue restored
    assert buf.text == "g"
    assert queues.queue == QueuedInputBlock(text="Q")
    assert nav.cursor == 0


# --- modified-test (the anti-Bug-1 invariant) -------------------------


def test_edit_at_queue_stop_then_up_does_not_requeue() -> None:
    """Modified value at the queue stop -> Q is not restored; edit rides."""
    queues = InputQueues(queue=QueuedInputBlock(text="Q"))
    nav = NavState()
    kb = _build(_busy_agent(), queues, nav)
    buf = _fake_buf("g", history=["h1"])
    _press(kb, "up", buf)  # input -> Q
    buf.text = "Q2"  # edit
    _press(kb, "up", buf)  # modified -> not restored
    assert queues.queue is None
    assert buf.text == "h1"


def test_edit_survives_down_replay() -> None:
    """Down hands back the edited value, never the original (spec)."""
    queues = InputQueues(queue=QueuedInputBlock(text="Q"))
    nav = NavState()
    kb = _build(_busy_agent(), queues, nav)
    buf = _fake_buf("g", history=["h1"])
    _press(kb, "up", buf)  # input -> Q
    buf.text = "Q2"  # edit
    _press(kb, "up", buf)  # -> h1, Q not restored
    _press(kb, "down", buf)  # -> Q2 (the edit), not Q
    assert buf.text == "Q2"
    assert queues.queue is None


def test_bug1_delete_gesture_removes_queued_message() -> None:
    """REGRESSION (Bug 1): clear queued, arrow away, it stays deleted.

    queue=Q, Up (unlift), clear, Up again -> Q gone everywhere; Down
    hands back the empty value then the original input. The deleted
    content never resurrects.
    """
    queues = InputQueues(queue=QueuedInputBlock(text="Q"))
    nav = NavState()
    kb = _build(_busy_agent(), queues, nav)
    buf = _fake_buf("g", history=["h1"])
    _press(kb, "up", buf)  # input -> Q
    buf.text = ""  # delete
    _press(kb, "up", buf)  # cleared -> Q not restored
    assert queues.queue is None
    assert buf.text == "h1"
    _press(kb, "down", buf)  # -> "" (the cleared stop's value)
    assert buf.text == ""
    assert queues.queue is None
    _press(kb, "down", buf)  # -> g
    assert buf.text == "g"
    assert queues.queue is None  # Q stays deleted everywhere


# --- Enter/Tab during navigation: replace own pane / append elsewhere -


def test_enter_at_queue_stop_replaces_queue_no_doubling() -> None:
    queues = InputQueues(queue=QueuedInputBlock(text="Q"))
    nav = NavState()
    kb = _build(_busy_agent(), queues, nav)
    buf = _fake_buf("g")
    _press(kb, "up", buf)  # -> Q (queue emptied)
    buf.text = "Q-edited"
    _press(kb, "enter", buf)  # commit: replace queue
    assert queues.queue == QueuedInputBlock(text="Q-edited")
    assert nav.cursor == 0


def test_enter_at_history_stop_appends_to_restored_queue() -> None:
    r"""Spec append rule: at a non-own stop, queue = existing + \n\n + input."""
    queues = InputQueues(queue=QueuedInputBlock(text="Q"))
    nav = NavState()
    kb = _build(_busy_agent(), queues, nav)
    buf = _fake_buf("g", history=["h1"])
    _press(kb, "up", buf)  # input -> Q
    _press(kb, "up", buf)  # Q restored, -> h1
    buf.text = "h1-edited"
    _press(kb, "enter", buf)  # commit at history stop: append to queue
    assert queues.queue is not None
    assert queues.queue.text == "Q\n\nh1-edited"
    assert nav.cursor == 0


def test_nav_commit_preserves_attachments() -> None:
    """REGRESSION: a lifted pane message's attachments survive nav-commit.

    Stage a queue message with an image, Up to lift it, edit, Enter. The
    committed queue message must keep the image (was dropped).
    """
    img = BytesMessage(data=b"png", descriptor="image/png")
    queues = InputQueues(queue=QueuedInputBlock(text="Q", attachments=(img,)))
    nav = NavState()
    kb = _build(_busy_agent(), queues, nav)
    buf = _fake_buf("g")
    _press(kb, "up", buf)  # lift Q (+image) into buffer
    buf.text = "Q-edited"
    _press(kb, "enter", buf)
    assert queues.queue is not None
    assert queues.queue.text == "Q-edited"
    assert queues.queue.attachments == (img,)


def test_nav_commit_dispatch_preserves_attachments_when_idle() -> None:
    """Idle nav-commit dispatches a UserMessage carrying the attachments."""
    img = BytesMessage(data=b"pdf", descriptor="application/pdf")
    agent = _idle_agent()
    queues = InputQueues(queue=QueuedInputBlock(text="Q", attachments=(img,)))
    nav = NavState()
    kb = _build(agent, queues, nav)
    buf = _fake_buf("g")
    _press(kb, "up", buf)
    _press(kb, "enter", buf)
    assert [type(i) for i in _inbox(agent).items] == [UserMessage]
    assert cast(UserMessage, _inbox(agent).items[0]).attachments == (img,)


def test_tab_during_navigation_moves_queue_to_deferred() -> None:
    """Unlift from queue pane, Tab -> becomes a deferred message (cross-pane)."""
    queues = InputQueues(queue=QueuedInputBlock(text="Q"))
    nav = NavState()
    kb = _build(_busy_agent(), queues, nav)
    buf = _fake_buf("g")
    _press(kb, "up", buf)  # -> Q (queue emptied)
    _press(kb, "tab", buf)  # commit as deferred
    assert queues.queue is None
    assert queues.deferred == QueuedInputBlock(text="Q")
    assert nav.cursor == 0


def test_whitespace_enter_during_navigation_ends_nav_no_restore() -> None:
    """Empty input during nav is a no-op that ends navigation (delete)."""
    agent = _busy_agent()
    queues = InputQueues(queue=QueuedInputBlock(text="Q"))
    nav = NavState()
    kb = _build(agent, queues, nav)
    buf = _fake_buf("g")
    _press(kb, "up", buf)  # -> Q (queue emptied)
    buf.text = "   "
    _press(kb, "enter", buf)
    assert nav.cursor == 0
    assert queues.queue is None  # not restored; the gesture deletes
    assert _inbox(agent).items == []


# --- idle dispatch after navigation -----------------------------------


def test_enter_after_navigation_idle_dispatches() -> None:
    """Spec: idle dispatches regardless of stop."""
    agent = _idle_agent()
    nav = NavState()
    kb = _build(agent, InputQueues(), nav)
    buf = _fake_buf("g", history=["h1"])
    _press(kb, "up", buf)  # -> h1
    buf.text = "h1-edited"
    _press(kb, "enter", buf)
    assert [type(i) for i in _inbox(agent).items] == [UserMessage]
    assert cast(UserMessage, _inbox(agent).items[0]).text == "h1-edited"
    assert nav.cursor == 0


def test_tab_after_navigation_idle_dispatches_deferred() -> None:
    """REGRESSION: idle Tab-during-navigation must dispatch (was a no-op)."""
    agent = _idle_agent()
    nav = NavState()
    kb = _build(agent, InputQueues(), nav)
    buf = _fake_buf("g", history=["h1"])
    _press(kb, "up", buf)  # -> h1
    _press(kb, "tab", buf)
    assert [type(i) for i in _inbox(agent).items] == [UserDeferredMessage]
    assert cast(UserDeferredMessage, _inbox(agent).items[0]).text == "h1"
    assert nav.cursor == 0


# --- misc keybindings -------------------------------------------------


def test_alt_enter_inserts_newline() -> None:
    kb = _build(_idle_agent())
    buf = _fake_buf("abc")
    _handler(kb, ("escape", "enter"))(cast(KeyPressEvent, _fake_event(buf)))
    buf.insert_text.assert_called_once_with("\n")


def test_down_on_non_empty_buf_clears_when_not_navigating() -> None:
    agent = _idle_agent()
    queues = InputQueues()
    kb = _build(agent, queues)
    buf = _fake_buf("hello")
    _press(kb, "down", buf)
    buf.reset.assert_called_once()
    assert not queues.has_any()
    assert _inbox(agent).items == []


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
    _press(kb, "c-_", buf)
    buf.undo.assert_called_once()


def test_escape_z_undo() -> None:
    kb = _build(_idle_agent())
    buf = _fake_buf("ab")
    _handler(kb, ("escape", "z"))(cast(KeyPressEvent, _fake_event(buf)))
    buf.undo.assert_called_once()


# --- Ctrl+C -----------------------------------------------------------


def test_ctrl_c_abandons_line_to_history() -> None:
    agent = _idle_agent()
    queues = InputQueues()
    kb = _build(agent, queues)
    buf = _fake_buf("partial line")
    _press(kb, "c-c", buf)
    buf.reset.assert_called_once_with(append_to_history=True)
    assert not queues.has_any()


def test_ctrl_c_halts_when_busy() -> None:
    agent = _busy_agent()
    kb = _build(agent)
    buf = _fake_buf("partial line")
    _press(kb, "c-c", buf)
    assert agent.halt_calls == 1
    buf.reset.assert_called_once_with(append_to_history=True)


def test_ctrl_c_halts_during_cohort() -> None:
    agent = _cohort_agent()
    kb = _build(agent)
    buf = _fake_buf("x")
    _press(kb, "c-c", buf)
    assert agent.halt_calls == 1


def test_ctrl_c_idle_does_not_halt() -> None:
    agent = _idle_agent()
    kb = _build(agent)
    buf = _fake_buf("typing here")
    _press(kb, "c-c", buf)
    assert agent.halt_calls == 0


def test_ctrl_c_leaves_committed_panes_untouched() -> None:
    queues = InputQueues(
        queue=QueuedInputBlock(text="urgent"),
        deferred=QueuedInputBlock(text="for later"),
    )
    for agent in (_busy_agent(), _idle_agent()):
        kb = _build(agent, queues)
        buf = _fake_buf("composing")
        _press(kb, "c-c", buf)
        assert queues.queue == QueuedInputBlock(text="urgent")
        assert queues.deferred == QueuedInputBlock(text="for later")


def test_ctrl_c_wins_over_prompt_session_default() -> None:
    """Our eager ``c-c`` must win the PromptSession merge (no KeyboardInterrupt)."""
    agent = _busy_agent()
    queues = InputQueues()
    session: PromptSession[str] = PromptSession(key_bindings=_build(agent, queues))
    user_bindings = session.key_bindings
    assert user_bindings is not None
    merged = merge_key_bindings(
        [user_bindings, cast(KeyBindingsBase, session._create_prompt_bindings())]
    )
    matches = merged.get_bindings_for_keys((Keys.ControlC,))
    eager = [m for m in matches if m.eager()]
    resolved = (eager or matches)[-1]
    handler = resolved.handler
    inner = getattr(handler, "func", handler)
    assert inner is keybindings_mod._kb_ctrl_c
    handler(cast(KeyPressEvent, _fake_event(_fake_buf(""))))
    assert agent.halt_calls == 1


# --- real prompt-toolkit Buffer ---------------------------------------
# The tests above drive a MagicMock buffer (fast, lets us inspect calls).
# These drive a REAL ``prompt_toolkit.buffer.Buffer`` end-to-end, so the
# handlers are validated against actual buffer semantics -- cursor moves,
# history append on reset, ``get_strings`` -- not a mock's stored fields.
# Closes the "MagicMock buffer hides behavior" gap.


def _real_buf(text: str = "", history: list[str] | None = None) -> Buffer:
    hist = InMemoryHistory()
    for entry in history or []:
        hist.append_string(entry)
    buf = Buffer(history=hist, multiline=True)
    buf.text = text
    buf.cursor_position = len(text)
    return buf


def _real_event(buf: Buffer) -> KeyPressEvent:
    ev = MagicMock()
    ev.current_buffer = buf
    return cast(KeyPressEvent, ev)


def test_real_buffer_enter_idle_dispatches() -> None:
    agent = _idle_agent()
    queues = InputQueues()
    kb = _build(agent, queues)
    buf = _real_buf("hello")
    _handler(kb, ("enter",))(_real_event(buf))
    assert [type(i) for i in _inbox(agent).items] == [UserMessage]
    assert cast(UserMessage, _inbox(agent).items[0]).text == "hello"
    assert buf.text == ""  # real buffer reset


def test_real_buffer_enter_busy_stages_and_resets() -> None:
    agent = _busy_agent()
    queues = InputQueues()
    kb = _build(agent, queues)
    buf = _real_buf("queued")
    _handler(kb, ("enter",))(_real_event(buf))
    assert queues.queue == QueuedInputBlock(text="queued")
    assert buf.text == ""


def test_real_buffer_full_nav_round_trip() -> None:
    """Real Buffer: input -> Q -> history -> back, queue restored, edits intact."""
    agent = _busy_agent()
    queues = InputQueues(queue=QueuedInputBlock(text="Q"))
    nav = NavState()
    kb = _build(agent, queues, nav)
    buf = _real_buf("typed", history=["h1", "h2", "h3"])
    up = _handler(kb, ("up",))
    down = _handler(kb, ("down",))

    up(_real_event(buf))
    assert buf.text == "Q"
    assert queues.queue is None
    up(_real_event(buf))
    assert buf.text == "h3"
    assert queues.queue == QueuedInputBlock(text="Q")  # restored on pass
    up(_real_event(buf))
    assert buf.text == "h2"
    down(_real_event(buf))
    assert buf.text == "h3"
    down(_real_event(buf))
    assert buf.text == "Q"
    assert queues.queue is None  # re-unlifted on the way down
    down(_real_event(buf))
    assert buf.text == "typed"
    assert queues.queue == QueuedInputBlock(text="Q")
    assert not nav.active()


def test_real_buffer_modified_test_delete_gesture() -> None:
    """Real Buffer: clear a lifted queue message, Up -> it stays deleted."""
    agent = _busy_agent()
    queues = InputQueues(queue=QueuedInputBlock(text="Q"))
    nav = NavState()
    kb = _build(agent, queues, nav)
    buf = _real_buf("typed", history=["h1"])
    up = _handler(kb, ("up",))
    up(_real_event(buf))  # lift Q
    assert buf.text == "Q"
    buf.text = ""  # delete
    up(_real_event(buf))  # modified -> not restored
    assert queues.queue is None
    assert buf.text == "h1"


def test_real_buffer_enter_appends_to_history_on_dispatch() -> None:
    """Real Buffer: a dispatched message lands in the buffer's history."""
    agent = _idle_agent()
    queues = InputQueues()
    kb = _build(agent, queues)
    buf = _real_buf("remember me")
    _handler(kb, ("enter",))(_real_event(buf))
    assert "remember me" in buf.history.get_strings()


def test_real_buffer_nav_commit_at_queue_stop_replaces() -> None:
    agent = _busy_agent()
    queues = InputQueues(queue=QueuedInputBlock(text="Q"))
    nav = NavState()
    kb = _build(agent, queues, nav)
    buf = _real_buf("typed")
    _handler(kb, ("up",))(_real_event(buf))
    buf.text = "Q-edited"
    _handler(kb, ("enter",))(_real_event(buf))
    assert queues.queue == QueuedInputBlock(text="Q-edited")
    assert not nav.active()


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
