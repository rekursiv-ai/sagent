"""Tests for ``repl.keybindings``: prompt-toolkit key handler bindings."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import cast
from unittest.mock import MagicMock

from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent

from sagent.agent.agent import Agent
from sagent.agent.runtime import UserMessage
from sagent.repl.keybindings import NavState, build_key_bindings


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


def _fake_buf(
    text: str = "", cursor: int | None = None, history: list[str] | None = None
) -> MagicMock:
    buf = MagicMock()
    buf.text = text
    buf.cursor_position = cursor if cursor is not None else len(text)
    buf.working_index = 0
    buf.document.text_before_cursor = text[: buf.cursor_position]
    buf.document.text = text
    # Up/Down read ``buf.history.get_strings()`` for the FileHistory walk.
    # Oldest-first list of strings.
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
    queued_input: list[str] | None = None,
    nav: NavState | None = None,
) -> KeyBindings:
    """Bind the keybindings against a fake agent; ``Agent`` cast is structural."""
    return build_key_bindings(
        cast(Agent, agent),
        queued_input if queued_input is not None else [],
        nav,
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


def test_tab_then_up_then_enter_re_queues_via_navigation_path() -> None:
    """Tab → Up → Enter exercises the navigation commit path.

    Tab stages "hello". Up lifts it into the buffer (cursor=1, snapshot
    captured). Enter at cursor>0 commits the buffer as a queued block
    and restores the snapshot's buffer (empty in this case). Net
    result: queue holds ["hello"] again, runtime untouched -- the user
    walked through navigation without losing anything.
    """
    agent = _idle_agent()
    queued_input: list[str] = []
    nav = NavState()
    kb = _build(agent, queued_input, nav)

    # Share one buffer across keystrokes; the handlers mutate buf.text.
    buf = _fake_buf("hello")
    _handler(kb, ("tab",))(cast(KeyPressEvent, _fake_event(buf)))
    assert queued_input == ["hello"]
    assert agent.runtime.inbox.items == []
    # Tab clears the buffer in the no-nav path.
    buf.text = ""

    _handler(kb, ("up",))(cast(KeyPressEvent, _fake_event(buf)))
    assert buf.text == "hello"
    assert queued_input == []  # dequeued into buffer
    assert nav.cursor == 1
    assert nav.snapshot_queue == ("hello",)
    assert nav.snapshot_input == ""

    _handler(kb, ("enter",))(cast(KeyPressEvent, _fake_event(buf)))
    # Enter at cursor==1 (Case 1, edit-mode): the lifted queue content
    # is replacing the original queue. No doubling. Snapshot input
    # (empty) is restored to the buffer. No runtime push.
    assert queued_input == ["hello"]
    assert buf.text == ""
    assert nav.cursor == 0
    assert agent.runtime.inbox.items == []


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
    r"""First Up with non-empty queue lifts all blocks (joined ``\\n\\n``) into buf.

    Per the contract figure (Case 1, t=0 -> t=1): queue empties,
    buffer gets the joined text, snapshot captures the original
    ``(queue, buffer)`` so a final Down can restore.
    """
    queued_input = ["a", "b", "c"]
    nav = NavState()
    kb = _build(_idle_agent(), queued_input, nav)
    buf = _fake_buf("typed")
    _handler(kb, ("up",))(cast(KeyPressEvent, _fake_event(buf)))
    assert buf.text == "a\n\nb\n\nc"
    assert queued_input == []
    assert nav.cursor == 1
    assert nav.snapshot_queue == ("a", "b", "c")
    assert nav.snapshot_input == "typed"


def test_up_walks_history_when_queue_empty() -> None:
    """First Up with empty queue + non-empty history pulls ``history[-1]`` in.

    Per the contract figure (Case 2, t=0 -> t=1): buffer becomes
    ``history[-1]``; snapshot captures the original input.
    """
    nav = NavState()
    kb = _build(_idle_agent(), [], nav)
    buf = _fake_buf("typed", history=["old", "newer", "latest"])
    _handler(kb, ("up",))(cast(KeyPressEvent, _fake_event(buf)))
    assert buf.text == "latest"
    assert nav.cursor == 1
    assert nav.snapshot_queue == ()
    assert nav.snapshot_input == "typed"


def test_up_with_no_queue_no_history_is_noop() -> None:
    """Up with empty queue + empty history leaves state untouched."""
    nav = NavState()
    kb = _build(_idle_agent(), [], nav)
    buf = _fake_buf("typed", history=[])
    _handler(kb, ("up",))(cast(KeyPressEvent, _fake_event(buf)))
    assert buf.text == "typed"
    assert nav.cursor == 0
    assert nav.snapshot_input == ""


def test_figure_case_1_round_trip() -> None:
    r"""Walk the Case 1 figure: queue + history, UP UP DN DN round-trip.

    Per the contract figure in :mod:`repl.input_pane`. The history
    list is the FileHistory contents (oldest-first); the visible
    ``history[-N]`` slot at any step is just the rendering window
    over this list. The test pins the *buffer*, *queue*, and *nav*
    state at each step.
    """
    nav = NavState()
    queued_input = ["f"]
    history = ["a", "b", "c", "d", "e"]  # oldest-first; history[-1] = "e"
    kb = _build(_idle_agent(), queued_input, nav)
    buf = _fake_buf("g", history=history)

    # t=0: starting state
    assert buf.text == "g"
    assert queued_input == ["f"]
    assert nav.cursor == 0

    # t=1: UP -> dequeue
    _handler(kb, ("up",))(cast(KeyPressEvent, _fake_event(buf)))
    assert buf.text == "f"
    assert queued_input == []
    assert nav.cursor == 1
    assert nav.snapshot_queue == ("f",)
    assert nav.snapshot_input == "g"

    # t=2: UP -> walk history[-1] = "e"
    _handler(kb, ("up",))(cast(KeyPressEvent, _fake_event(buf)))
    assert buf.text == "e"
    assert queued_input == []
    assert nav.cursor == 2

    # t=3: DN -> back to "edit queue" position
    _handler(kb, ("down",))(cast(KeyPressEvent, _fake_event(buf)))
    assert buf.text == "f"
    assert queued_input == []
    assert nav.cursor == 1

    # t=4: DN -> final restore (queue and buffer back to t=0)
    _handler(kb, ("down",))(cast(KeyPressEvent, _fake_event(buf)))
    assert buf.text == "g"
    assert queued_input == ["f"]
    assert nav.cursor == 0
    assert nav.snapshot_queue == ()


def test_figure_case_2_round_trip() -> None:
    r"""Walk the Case 2 figure: no queue + history, UP UP DN DN round-trip.

    No queue at t=0; the cursor walks straight into history.
    Round-trip lands exactly on t=0 (per user confirmation).
    """
    nav = NavState()
    queued_input: list[str] = []
    history = ["a", "b", "c", "d", "e", "f"]  # history[-1] = "f"
    kb = _build(_idle_agent(), queued_input, nav)
    buf = _fake_buf("g", history=history)

    # t=0
    assert buf.text == "g"
    assert queued_input == []
    assert nav.cursor == 0

    # t=1: UP -> history[-1] = "f"
    _handler(kb, ("up",))(cast(KeyPressEvent, _fake_event(buf)))
    assert buf.text == "f"
    assert nav.cursor == 1
    assert nav.snapshot_queue == ()
    assert nav.snapshot_input == "g"

    # t=2: UP -> history[-2] = "e"
    _handler(kb, ("up",))(cast(KeyPressEvent, _fake_event(buf)))
    assert buf.text == "e"
    assert nav.cursor == 2

    # t=3: DN -> back to history[-1]
    _handler(kb, ("down",))(cast(KeyPressEvent, _fake_event(buf)))
    assert buf.text == "f"
    assert nav.cursor == 1

    # t=4: DN -> final restore
    _handler(kb, ("down",))(cast(KeyPressEvent, _fake_event(buf)))
    assert buf.text == "g"
    assert queued_input == []
    assert nav.cursor == 0


def test_enter_after_navigation_commits_buffer_and_restores() -> None:
    """Enter at cursor>1: append buffer to snapshot_queue; restore typing.

    Mirrors "now: enter, appends to queued" from the contract. The
    user scrolled into history (cursor>1), then Enter commits the
    (possibly edited) history entry as a new queued block. The
    snapshot queue is preserved (NOT replaced) because the user
    "scrolled past" rather than edited the queue.
    """
    nav = NavState()
    queued_input = ["original"]
    history = ["older", "older2", "latest"]
    kb = _build(_idle_agent(), queued_input, nav)
    buf = _fake_buf("typed-before-up", history=history)

    # Up x2: dequeue then walk to history[-1].
    _handler(kb, ("up",))(cast(KeyPressEvent, _fake_event(buf)))
    _handler(kb, ("up",))(cast(KeyPressEvent, _fake_event(buf)))
    assert buf.text == "latest"
    assert nav.cursor == 2

    # Enter: commit "latest" as queued block, restore "typed-before-up".
    _handler(kb, ("enter",))(cast(KeyPressEvent, _fake_event(buf)))
    assert queued_input == ["original", "latest"]
    assert buf.text == "typed-before-up"
    assert nav.cursor == 0


def test_enter_at_cursor_zero_still_preempt_dispatches() -> None:
    """No navigation active: Enter pushes ``UserMessage`` (today's behavior)."""
    agent = _idle_agent()
    nav = NavState()
    kb = _build(agent, [], nav)
    buf = _fake_buf("hello")
    _handler(kb, ("enter",))(cast(KeyPressEvent, _fake_event(buf)))
    assert len(agent.runtime.inbox.items) == 1
    assert isinstance(agent.runtime.inbox.items[0], UserMessage)
    assert agent.runtime.inbox.items[0].text == "hello"
    assert nav.cursor == 0


def test_tab_during_navigation_commits_via_navigation_path() -> None:
    """Tab at cursor>0 mirrors Enter at cursor>0 (commit + restore).

    User scrolls history, hits Tab to "save this for later" without
    losing their pre-navigation typing.
    """
    agent = _idle_agent()
    nav = NavState()
    queued_input = ["existing"]
    history = ["latest"]
    kb = _build(agent, queued_input, nav)
    buf = _fake_buf("draft", history=history)

    # Up x2: dequeue queue, then walk to history[-1].
    _handler(kb, ("up",))(cast(KeyPressEvent, _fake_event(buf)))
    _handler(kb, ("up",))(cast(KeyPressEvent, _fake_event(buf)))
    assert buf.text == "latest"

    # Tab commits + restores draft.
    _handler(kb, ("tab",))(cast(KeyPressEvent, _fake_event(buf)))
    assert queued_input == ["existing", "latest"]
    assert buf.text == "draft"
    assert nav.cursor == 0
    assert agent.runtime.inbox.items == []


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
