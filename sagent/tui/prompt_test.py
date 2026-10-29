"""Tests for the prompt: the panes, the keys, and where a typed line goes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import asyncio
import contextlib
import signal
import weakref

from prompt_toolkit import PromptSession
from prompt_toolkit.application import Application
from prompt_toolkit.application.current import set_app
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.input import DummyInput, create_pipe_input
from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout import Layout, Window
from prompt_toolkit.layout.controls import BufferControl
from prompt_toolkit.output import DummyOutput

import pytest
import pytest_asyncio

from sagent.model.scripted import ScriptedProvider, reply
from sagent.runtime import types
from sagent.runtime.runtime import Await, Runtime
from sagent.session.types import AgentToAgentMessage, Attachment, UserMessage
from sagent.tui.prompt import (
    OPEN_TREE,
    Deferred,
    Input,
    Navigation,
    Panes,
    PromptSource,
    Staged,
    commit_panes,
    key_bindings,
    prompt_text,
    repaint_on_change,
)


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from sagent.model.types import Response


_PNG = Attachment(mime_descriptor="image/png", data=b"png")


async def _forever() -> None:
    await asyncio.Event().wait()


def _runtime() -> Runtime:
    return Runtime(model=ScriptedProvider()["scripted"])


@dataclass(slots=True, kw_only=True)
class _Prompt:
    """A real buffer in a real application, and the keys bound over them."""

    runtime: Runtime = field(default_factory=_runtime)
    panes: Panes = field(default_factory=Panes)
    nav: Navigation = field(default_factory=Navigation)
    accepted: list[str] = field(default_factory=list[str])
    buffer: Buffer = field(init=False)
    app: Application[None] = field(init=False)
    keys: KeyBindings = field(init=False)

    def __post_init__(self) -> None:
        self.buffer = Buffer(
            history=InMemoryHistory(),
            multiline=True,
            accept_handler=self._accept,
        )
        self.keys = key_bindings(self.runtime, self.panes, self.nav)
        self.app = Application(
            layout=Layout(Window(BufferControl(buffer=self.buffer))),
            key_bindings=self.keys,
            input=DummyInput(),
            output=DummyOutput(),
        )

    def _accept(self, buffer: Buffer) -> bool:
        self.accepted.append(buffer.text)
        return False

    def history(self, *entries: str) -> _Prompt:
        for entry in entries:
            self.buffer.history.append_string(entry)
        self.buffer.reset()
        return self

    async def load_history(self) -> None:
        """Load the history into the buffer, as drawing the prompt would."""
        with set_app(self.app):
            self.buffer.load_history_if_not_yet_loaded()
        loading = self.buffer._load_history_task
        assert loading is not None
        await loading

    def type(self, text: str) -> None:
        self.buffer.text = text
        self.buffer.cursor_position = len(text)

    def press(self, *keys: str | Keys) -> None:
        (binding,) = [b for b in self.keys.bindings if tuple(b.keys) == tuple(keys)]
        with set_app(self.app):
            binding.handler(
                KeyPressEvent(weakref.ref(self.app.key_processor), None, [], [], False),
            )

    def pushed(self) -> list[types.RuntimeEvent]:
        return self.runtime.inbox.drain_nowait()


# ---- panes -----------------------------------------------------------------


def test_staging_into_an_empty_pane_holds_the_message() -> None:
    panes = Panes()
    panes.stage("now", (_PNG,), deferred=False)
    panes.stage("later", deferred=True)
    assert panes.queue == Staged(text="now", attachments=(_PNG,))
    assert panes.deferred == Staged(text="later")


def test_staging_into_a_full_pane_appends_after_a_blank_line() -> None:
    panes = Panes()
    panes.stage("a", (_PNG,), deferred=False)
    panes.stage("b", (_PNG,), deferred=False)
    panes.stage("x", deferred=True)
    panes.stage("y", (_PNG,), deferred=True)
    assert panes.queue == Staged(text="a\n\nb", attachments=(_PNG, _PNG))
    assert panes.deferred == Staged(text="x\n\ny", attachments=(_PNG,))


def test_the_panes_show_the_deferred_message_above_the_queued_one() -> None:
    panes = Panes()
    assert panes.lines() == []
    panes.stage("q", deferred=False)
    assert panes.lines() == ["q"]
    panes.stage("d", deferred=True)
    assert panes.lines() == ["[deferred] d", "q"]
    panes.clear()
    assert (panes.queue, panes.deferred, panes.lines()) == (None, None, [])
    panes.stage("only d", deferred=True)
    assert panes.lines() == ["[deferred] only d"]


def test_popping_the_queue_empties_its_pane_and_keeps_the_attachments() -> None:
    panes = Panes()
    assert panes.pop_queue() is None
    panes.stage("q", (_PNG,), deferred=False)
    panes.stage("d", deferred=True)
    assert panes.pop_queue() == UserMessage(content="q", attachments=(_PNG,))
    assert panes.queue is None
    assert panes.deferred == Staged(text="d")


def test_sending_a_pane_pushes_it_and_empties_it() -> None:
    runtime = _runtime()
    panes = Panes()
    assert panes.send_queue(runtime) is False
    assert panes.send_deferred(runtime) is False
    assert runtime.inbox.drain_nowait() == []
    panes.stage("q", deferred=False)
    panes.stage("d", (_PNG,), deferred=True)
    assert panes.send_queue(runtime) is True
    assert panes.send_deferred(runtime) is True
    assert runtime.inbox.drain_nowait() == [
        UserMessage(content="q"),
        types.UserDeferredMessage(text="d", attachments=(_PNG,)),
    ]
    assert (panes.queue, panes.deferred) == (None, None)


# ---- commit_panes ----------------------------------------------------------


def test_the_queue_rides_the_next_tool_round() -> None:
    runtime, panes = _runtime(), Panes()
    commit_panes(runtime, panes)
    hook = runtime.before_tool_spawn
    assert hook is not None
    assert hook(reply("x")) is None
    panes.stage("q", deferred=False)
    panes.stage("d", deferred=True)
    assert hook(reply("x")) == UserMessage(content="q")
    assert panes.queue is None
    assert panes.deferred == Staged(text="d")


def test_an_earlier_hook_s_event_wins_and_its_silence_lets_the_queue_through() -> None:
    runtime, panes = _runtime(), Panes()
    said: list[UserMessage | None] = [UserMessage(content="first"), None]
    seen: list[Response] = []

    def earlier(turn: Response) -> UserMessage | None:
        seen.append(turn)
        return said.pop(0)

    runtime.before_tool_spawn = earlier
    commit_panes(runtime, panes)
    hook = runtime.before_tool_spawn
    assert hook is not None
    panes.stage("q", deferred=False)
    turn = reply("x")
    assert hook(turn) == UserMessage(content="first")
    assert panes.queue == Staged(text="q")
    assert hook(turn) == UserMessage(content="q")
    assert seen == [turn, turn]


@pytest.mark.parametrize("edge", [types.AgentIdle(), types.ClearComplete()])
def test_an_idle_or_a_finished_clear_sends_the_queue_first_then_the_deferred(
    edge: types.RuntimeEvent,
) -> None:
    runtime, panes = _runtime(), Panes()
    commit_panes(runtime, panes)
    panes.stage("q", deferred=False)
    panes.stage("d", deferred=True)
    runtime.publish(edge)
    assert runtime.inbox.drain_nowait() == [UserMessage(content="q")]
    runtime.publish(edge)
    assert runtime.inbox.drain_nowait() == [types.UserDeferredMessage(text="d")]
    runtime.publish(edge)
    assert runtime.inbox.drain_nowait() == []


def test_other_events_send_nothing() -> None:
    runtime, panes = _runtime(), Panes()
    commit_panes(runtime, panes)
    panes.stage("q", deferred=False)
    runtime.publish(types.ModelIdle())
    runtime.publish(types.ModelResponseComplete(response=reply("x")))
    assert runtime.inbox.drain_nowait() == []


def test_the_undo_restores_the_hook_and_removes_the_observer_once() -> None:
    runtime, panes = _runtime(), Panes()

    def earlier(turn: Response) -> UserMessage | None:
        del turn
        return None

    runtime.before_tool_spawn = earlier
    before = list(runtime.observers)
    remove = commit_panes(runtime, panes)
    assert len(runtime.observers) == len(before) + 1
    remove()
    remove()
    assert runtime.before_tool_spawn is earlier
    assert runtime.observers == before
    panes.stage("q", deferred=False)
    runtime.publish(types.AgentIdle())
    assert runtime.inbox.drain_nowait() == []


def test_the_undo_keeps_a_hook_installed_after_it() -> None:
    runtime, panes = _runtime(), Panes()
    remove = commit_panes(runtime, panes)

    def later(turn: Response) -> UserMessage | None:
        del turn
        return None

    runtime.before_tool_spawn = later
    remove()
    assert runtime.before_tool_spawn is later


# ---- prompt_text -----------------------------------------------------------


def _prompt_text(runtime: Runtime, panes: Panes) -> FormattedText:
    return prompt_text(Deferred(runtime, panes), Input())


def test_the_prompt_alone_is_a_bold_sigil_after_a_blank_line() -> None:
    assert _prompt_text(_runtime(), Panes()) == FormattedText(
        [("", "\n"), ("class:input_pane", "> ")],
    )
    assert Deferred(_runtime(), Panes()).frame() == ()
    assert Input().frame() == (("class:input_pane", "> "),)


def test_the_deferred_frame_compares_equal_until_what_waits_changes() -> None:
    runtime, panes = _runtime(), Panes()
    deferred = Deferred(runtime, panes)
    before = deferred.frame()
    assert deferred.frame() == before
    panes.stage("queued", deferred=False)
    assert deferred.frame() == (("class:queued_input_pane", "queued"), ("", "\n"))


def test_the_panes_and_pending_user_messages_show_above_the_prompt() -> None:
    runtime, panes = _runtime(), Panes()
    panes.stage("queued", deferred=False)
    panes.stage("line one\nline two", deferred=True)
    runtime._mid_stream_queue.extend(
        [
            UserMessage(content="typed mid-stream"),
            AgentToAgentMessage(content="from a child"),
        ],
    )
    assert _prompt_text(runtime, panes) == FormattedText(
        [
            ("", "\n"),
            (
                "class:queued_input_pane",
                "[deferred] line one\nline two\n\nqueued\n\npending: typed mid-stream",
            ),
            ("", "\n"),
            ("class:input_pane", "> "),
        ],
    )


def test_an_agent_s_pending_message_alone_shows_nothing() -> None:
    runtime = _runtime()
    runtime._mid_stream_queue.append(AgentToAgentMessage(content="x"))
    assert _prompt_text(runtime, Panes()) == FormattedText(
        [("", "\n"), ("class:input_pane", "> ")],
    )


# ---- Enter and Tab ---------------------------------------------------------


def test_enter_while_idle_sends_resets_and_remembers_the_line() -> None:
    prompt = _Prompt()
    prompt.type("hello")
    prompt.press(Keys.ControlM)
    assert prompt.pushed() == [UserMessage(content="hello")]
    assert prompt.buffer.text == ""
    assert list(prompt.buffer.history.get_strings()) == ["hello"]


def test_tab_while_idle_sends_the_line_deferred() -> None:
    prompt = _Prompt()
    prompt.type("later")
    prompt.press(Keys.ControlI)
    assert prompt.pushed() == [types.UserDeferredMessage(text="later")]
    assert prompt.buffer.text == ""
    assert list(prompt.buffer.history.get_strings()) == ["later"]


@pytest.mark.asyncio
async def test_enter_and_tab_while_streaming_stage_into_their_panes() -> None:
    prompt = _Prompt()
    prompt.runtime.model_call = asyncio.create_task(asyncio.sleep(0))
    for text in ("q1", "q2"):
        prompt.type(text)
        prompt.press(Keys.ControlM)
    for text in ("d1", "d2"):
        prompt.type(text)
        prompt.press(Keys.ControlI)
    assert prompt.pushed() == []
    assert prompt.panes.queue == Staged(text="q1\n\nq2")
    assert prompt.panes.deferred == Staged(text="d1\n\nd2")
    assert prompt.buffer.text == ""
    await prompt.runtime.model_call


@pytest.mark.asyncio
async def test_enter_while_compacting_stages() -> None:
    prompt = _Prompt()
    prompt.runtime.compact_task = asyncio.create_task(asyncio.sleep(0))
    prompt.type("q")
    prompt.press(Keys.ControlM)
    assert prompt.pushed() == []
    assert prompt.panes.queue == Staged(text="q")
    await prompt.runtime.compact_task


@pytest.mark.asyncio
async def test_enter_after_a_halt_is_queued_still_sends() -> None:
    """Between Ctrl+C and the cancel the call is still set; staging would orphan it."""
    prompt = _Prompt()
    prompt.runtime.model_call = asyncio.create_task(asyncio.sleep(0))
    prompt.runtime.halt()
    prompt.type("redirect")
    prompt.press(Keys.ControlM)
    assert prompt.pushed() == [types.Halt(), UserMessage(content="redirect")]
    await prompt.runtime.model_call


@pytest.mark.asyncio
@pytest.mark.parametrize("key", [Keys.ControlM, Keys.ControlI])
async def test_an_armed_gate_takes_enter_and_tab_even_mid_stream(
    busy: _Prompt,
    key: Keys,
) -> None:
    """A halt parks the inbox; the line is what releases it, so it must not stage."""
    busy.runtime.inbox.push_front(Await(types=(UserMessage,)))
    busy.type("go")
    busy.press(key)
    assert busy.pushed() == [
        UserMessage(content="go")
        if key == Keys.ControlM
        else types.UserDeferredMessage(text="go"),
    ]
    assert (busy.panes.queue, busy.panes.deferred) == (None, None)


def test_an_armed_gate_does_not_send_tab_past_a_running_cohort() -> None:
    prompt = _Prompt()
    prompt.runtime.cohort.add("c1")
    prompt.runtime.inbox.push_front(Await(types=(UserMessage,)))
    prompt.type("go")
    prompt.press(Keys.ControlI)
    assert prompt.pushed() == []
    assert prompt.panes.deferred == Staged(text="go")


def test_mid_cohort_enter_redirects_and_tab_waits() -> None:
    prompt = _Prompt()
    prompt.runtime.cohort.add("c1")
    prompt.type("redirect")
    prompt.press(Keys.ControlM)
    prompt.type("after")
    prompt.press(Keys.ControlI)
    assert prompt.pushed() == [UserMessage(content="redirect")]
    assert prompt.panes.deferred == Staged(text="after")


@pytest.mark.asyncio
async def test_tab_waits_behind_a_running_tool() -> None:
    prompt = _Prompt()
    prompt.runtime.running_tools["c1"] = asyncio.create_task(asyncio.sleep(0))
    prompt.type("after")
    prompt.press(Keys.ControlI)
    assert prompt.pushed() == []
    assert prompt.panes.deferred == Staged(text="after")
    await prompt.runtime.running_tools["c1"]


@pytest.mark.parametrize("key", [Keys.ControlM, Keys.ControlI])
@pytest.mark.parametrize("text", ["", "   ", "\n"])
def test_a_blank_line_is_cleared_and_sent_nowhere(key: Keys, text: str) -> None:
    prompt = _Prompt()
    prompt.type(text)
    prompt.press(key)
    assert prompt.pushed() == []
    assert prompt.buffer.text == ""
    assert list(prompt.buffer.history.get_strings()) == []


def test_a_trailing_backslash_continues_the_line() -> None:
    prompt = _Prompt()
    prompt.type("first \\")
    prompt.press(Keys.ControlM)
    assert prompt.buffer.text == "first \n"
    assert prompt.buffer.cursor_position == len("first \n")
    assert prompt.pushed() == []


def test_a_command_is_accepted_as_the_line() -> None:
    prompt = _Prompt()
    prompt.type("  /model x")
    prompt.press(Keys.ControlM)
    assert prompt.accepted == ["  /model x"]
    assert prompt.pushed() == []


def test_alt_enter_inserts_a_newline() -> None:
    prompt = _Prompt()
    prompt.type("a")
    prompt.press(Keys.Escape, Keys.ControlM)
    assert prompt.buffer.text == "a\n"
    assert prompt.pushed() == []


# ---- Up and Down -----------------------------------------------------------


@pytest_asyncio.fixture
async def busy() -> AsyncIterator[_Prompt]:
    """Return a prompt whose runtime streams, so a pane written back stays a pane."""
    prompt = _Prompt()
    stream = asyncio.create_task(_forever())
    prompt.runtime.model_call = stream
    yield prompt
    stream.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await stream


def test_up_with_nothing_to_walk_does_nothing() -> None:
    prompt = _Prompt()
    prompt.type("typed")
    prompt.press(Keys.Up)
    assert prompt.buffer.text == "typed"
    assert prompt.nav.cursor == 0


@pytest.mark.asyncio
async def test_down_at_the_input_clears_it(busy: _Prompt) -> None:
    busy.press(Keys.Down)
    assert busy.buffer.text == ""
    busy.type("typed")
    busy.press(Keys.Down)
    assert busy.buffer.text == ""
    assert busy.nav.cursor == 0


@pytest.mark.asyncio
async def test_up_walks_queue_then_deferred_then_history_newest_first(
    busy: _Prompt,
) -> None:
    busy.history("h1", "h2")
    busy.panes.stage("Q", deferred=False)
    busy.panes.stage("D", deferred=True)
    busy.type("typed")
    walked: list[str] = []
    for _ in range(5):
        busy.press(Keys.Up)
        walked.append(busy.buffer.text)
    assert walked == ["Q", "D", "h2", "h1", "h1"]
    assert busy.buffer.cursor_position == 2
    assert busy.panes.queue == Staged(text="Q")
    assert busy.panes.deferred == Staged(text="D")


def test_down_off_an_unedited_pane_on_an_idle_agent_sends_it() -> None:
    prompt = _Prompt()
    prompt.panes.stage("Q", deferred=False)
    prompt.type("typed")
    prompt.press(Keys.Up)
    prompt.press(Keys.Down)
    assert prompt.pushed() == [UserMessage(content="Q")]
    assert prompt.panes.queue is None
    assert prompt.buffer.text == "typed"


@pytest.mark.asyncio
async def test_committing_at_one_pane_s_stop_leaves_the_other_pane(
    busy: _Prompt,
) -> None:
    busy.panes.stage("Q", deferred=False)
    busy.panes.stage("D", deferred=True)
    busy.press(Keys.Up)
    busy.press(Keys.ControlM)
    assert (busy.panes.queue, busy.panes.deferred) == (
        Staged(text="Q"),
        Staged(text="D"),
    )
    busy.press(Keys.Up)
    busy.press(Keys.Up)
    busy.type("D2")
    busy.press(Keys.ControlI)
    assert (busy.panes.queue, busy.panes.deferred) == (
        Staged(text="Q"),
        Staged(text="D2"),
    )


def test_tab_at_an_idle_deferred_stop_sends_its_attachments() -> None:
    prompt = _Prompt()
    prompt.panes.stage("D", (_PNG,), deferred=True)
    prompt.press(Keys.Up)
    prompt.press(Keys.ControlI)
    assert prompt.pushed() == [types.UserDeferredMessage(text="D", attachments=(_PNG,))]


@pytest.mark.asyncio
async def test_a_lifted_pane_empties_and_comes_back_when_passed_unedited(
    busy: _Prompt,
) -> None:
    busy.history("h1")
    busy.panes.stage("Q", (_PNG,), deferred=False)
    busy.type("typed")
    busy.press(Keys.Up)
    assert (busy.buffer.text, busy.panes.queue) == ("Q", None)
    busy.press(Keys.Up)
    assert busy.buffer.text == "h1"
    assert busy.panes.queue == Staged(text="Q", attachments=(_PNG,))
    busy.press(Keys.Down)
    assert (busy.buffer.text, busy.panes.queue) == ("Q", None)
    busy.press(Keys.Down)
    assert busy.buffer.text == "typed"
    assert busy.buffer.cursor_position == len("typed")
    assert busy.panes.queue == Staged(text="Q", attachments=(_PNG,))
    assert busy.nav == Navigation()
    assert busy.pushed() == []


@pytest.mark.asyncio
async def test_an_edited_pane_is_not_put_back_and_its_edit_rides_the_cursor(
    busy: _Prompt,
) -> None:
    busy.history("h1")
    busy.panes.stage("Q", deferred=False)
    busy.press(Keys.Up)
    busy.type("Q edited")
    busy.press(Keys.Up)
    assert busy.buffer.text == "h1"
    assert busy.panes.queue is None
    busy.type("h1 edited")
    busy.press(Keys.Down)
    assert busy.buffer.text == "Q edited"
    busy.press(Keys.Up)
    assert busy.buffer.text == "h1 edited"
    busy.press(Keys.Down)
    busy.press(Keys.Down)
    assert busy.buffer.text == ""
    assert busy.panes.queue is None


@pytest.mark.asyncio
async def test_clearing_a_lifted_pane_deletes_its_message(busy: _Prompt) -> None:
    busy.history("h1")
    busy.panes.stage("Q", deferred=False)
    busy.press(Keys.Up)
    busy.type("")
    busy.press(Keys.Up)
    busy.press(Keys.Down)
    busy.press(Keys.Down)
    assert busy.panes.queue is None
    assert busy.buffer.text == ""


def test_a_pane_left_unedited_on_an_idle_agent_is_sent() -> None:
    """The agent may idle while the pane sits empty; written back, it would wait forever."""
    prompt = _Prompt().history("h1")
    prompt.panes.stage("Q", deferred=False)
    prompt.panes.stage("D", deferred=True)
    prompt.press(Keys.Up)
    prompt.press(Keys.Up)
    prompt.press(Keys.Up)
    assert prompt.pushed() == [
        UserMessage(content="Q"),
        types.UserDeferredMessage(text="D"),
    ]
    assert (prompt.panes.queue, prompt.panes.deferred) == (None, None)


@pytest.mark.asyncio
async def test_enter_at_a_pane_s_stop_replaces_that_pane(busy: _Prompt) -> None:
    busy.panes.stage("Q", (_PNG,), deferred=False)
    busy.press(Keys.Up)
    busy.type("Q edited")
    busy.press(Keys.ControlM)
    assert busy.panes.queue == Staged(text="Q edited", attachments=(_PNG,))
    assert busy.nav == Navigation()
    assert busy.buffer.text == ""


@pytest.mark.asyncio
async def test_enter_at_a_history_stop_appends_to_the_restored_queue(
    busy: _Prompt,
) -> None:
    busy.history("old")
    busy.panes.stage("Q", deferred=False)
    busy.press(Keys.Up)
    busy.press(Keys.Up)
    busy.press(Keys.ControlM)
    assert busy.panes.queue == Staged(text="Q\n\nold")


def test_a_committed_stop_sends_its_attachments_when_idle() -> None:
    prompt = _Prompt()
    prompt.panes.stage("Q", (_PNG,), deferred=False)
    prompt.press(Keys.Up)
    prompt.press(Keys.ControlM)
    assert prompt.pushed() == [UserMessage(content="Q", attachments=(_PNG,))]


@pytest.mark.asyncio
async def test_tab_at_the_queue_s_stop_moves_it_to_the_deferred_pane(
    busy: _Prompt,
) -> None:
    busy.panes.stage("Q", deferred=False)
    busy.press(Keys.Up)
    busy.press(Keys.ControlI)
    assert busy.panes.queue is None
    assert busy.panes.deferred == Staged(text="Q")


@pytest.mark.asyncio
async def test_tab_at_the_deferred_stop_replaces_it(busy: _Prompt) -> None:
    busy.panes.stage("D", deferred=True)
    busy.press(Keys.Up)
    busy.type("D edited")
    busy.press(Keys.ControlI)
    assert busy.panes.deferred == Staged(text="D edited")


@pytest.mark.asyncio
async def test_a_blank_enter_while_walking_ends_the_walk_and_drops_the_lifted_pane(
    busy: _Prompt,
) -> None:
    busy.panes.stage("Q", deferred=False)
    busy.press(Keys.Up)
    busy.type("  ")
    busy.press(Keys.ControlM)
    assert busy.nav == Navigation()
    assert busy.panes.queue is None
    assert busy.buffer.text == ""


@pytest.mark.asyncio
@pytest.mark.parametrize("edited", [False, True])
async def test_a_command_while_walking_leaves_the_lifted_pane_as_it_was(
    busy: _Prompt,
    edited: bool,
) -> None:
    busy.history("h1")
    busy.panes.stage("Q", deferred=False)
    busy.panes.stage("D", deferred=True)
    busy.press(Keys.Up)
    if edited:
        busy.type("Q edited")
        busy.press(Keys.Up)
        busy.press(Keys.Down)
    busy.type("/help")
    busy.press(Keys.ControlM)
    assert busy.accepted == ["/help"]
    assert busy.panes.queue == (None if edited else Staged(text="Q"))
    assert busy.panes.deferred == Staged(text="D")
    assert busy.nav == Navigation()


@pytest.mark.asyncio
async def test_a_command_at_the_deferred_stop_puts_that_pane_back(
    busy: _Prompt,
) -> None:
    busy.panes.stage("Q", deferred=False)
    busy.panes.stage("D", (_PNG,), deferred=True)
    busy.press(Keys.Up)
    busy.press(Keys.Up)
    assert busy.panes.deferred is None
    busy.type("/help")
    busy.press(Keys.ControlM)
    assert busy.panes.deferred == Staged(text="D", attachments=(_PNG,))
    assert busy.panes.queue == Staged(text="Q")
    assert busy.pushed() == []


@pytest.mark.asyncio
async def test_a_command_at_a_history_stop_changes_no_pane(busy: _Prompt) -> None:
    busy.history("/help")
    busy.panes.stage("Q", deferred=False)
    busy.press(Keys.Up)
    busy.press(Keys.Up)
    busy.press(Keys.ControlM)
    assert busy.accepted == ["/help"]
    assert busy.panes.queue == Staged(text="Q")


@pytest.mark.asyncio
async def test_the_deferred_stop_comes_back_on_the_way_down(busy: _Prompt) -> None:
    busy.panes.stage("D", deferred=True)
    busy.press(Keys.Up)
    assert busy.panes.deferred is None
    busy.press(Keys.Down)
    assert busy.panes.deferred == Staged(text="D")


# ---- other keys ------------------------------------------------------------


@pytest.mark.asyncio
async def test_shift_up_and_down_search_history_by_prefix() -> None:
    """Each press searches by the text first typed, as readline's search does."""
    prompt = _Prompt().history("ls -l", "git status", "ls", "git push", "make")
    await prompt.load_history()
    prompt.type("git")
    walked: list[str] = []
    for key in (
        Keys.ShiftUp,
        Keys.ShiftUp,
        Keys.ShiftUp,
        Keys.ShiftDown,
        Keys.ShiftDown,
    ):
        prompt.press(key)
        walked.append(prompt.buffer.text)
    assert walked == ["git push", "git status", "git status", "git push", "git"]
    assert prompt.buffer.document.text_before_cursor == "git"


@pytest.mark.asyncio
async def test_a_search_stops_at_the_nearest_match() -> None:
    prompt = _Prompt().history("go a", "go b", "go c")
    await prompt.load_history()
    prompt.type("go")
    walked: list[str] = []
    for key in (Keys.ShiftUp, Keys.ShiftUp, Keys.ShiftDown):
        prompt.press(key)
        walked.append(prompt.buffer.text)
    assert walked == ["go c", "go b", "go c"]


def test_ctrl_x_ctrl_e_opens_the_editor(monkeypatch: pytest.MonkeyPatch) -> None:
    prompt = _Prompt()
    opened: list[str] = []
    monkeypatch.setattr(
        prompt.buffer,
        "open_in_editor",
        lambda: opened.append(prompt.buffer.text),
    )
    prompt.type("draft")
    prompt.press(Keys.ControlX, Keys.ControlE)
    assert opened == ["draft"]


def test_ctrl_z_suspends(monkeypatch: pytest.MonkeyPatch) -> None:
    prompt = _Prompt()
    suspended: list[bool] = []
    monkeypatch.setattr(
        prompt.app,
        "suspend_to_background",
        lambda: suspended.append(True),
    )
    prompt.press(Keys.ControlZ)
    assert suspended == [True]


@pytest.mark.parametrize("keys", [(Keys.ControlUnderscore,), (Keys.Escape, "z")])
def test_undo_takes_back_the_last_edit(keys: tuple[str | Keys, ...]) -> None:
    prompt = _Prompt()
    prompt.buffer.insert_text("one")
    prompt.buffer.save_to_undo_stack()
    prompt.buffer.insert_text(" two")
    prompt.press(*keys)
    assert prompt.buffer.text == "one"


def test_ctrl_c_while_idle_abandons_the_line_to_history_without_a_halt() -> None:
    prompt = _Prompt()
    prompt.panes.stage("Q", deferred=False)
    prompt.type("abandoned")
    prompt.press(Keys.ControlC)
    assert prompt.buffer.text == ""
    assert list(prompt.buffer.history.get_strings()) == ["abandoned"]
    assert prompt.pushed() == []
    assert prompt.panes.queue == Staged(text="Q")


@pytest.mark.asyncio
@pytest.mark.parametrize("busy_with", ["model_call", "compact_task", "cohort"])
async def test_ctrl_c_halts_a_busy_agent(busy_with: str) -> None:
    prompt = _Prompt()
    task = asyncio.create_task(asyncio.sleep(0))
    if busy_with == "model_call":
        prompt.runtime.model_call = task
    elif busy_with == "compact_task":
        prompt.runtime.compact_task = task
    else:
        prompt.runtime.cohort.add("c1")
    prompt.press(Keys.ControlC)
    assert prompt.pushed() == [types.Halt()]
    await task


def test_the_keys_that_send_stop_once_the_prompt_is_accepted() -> None:
    prompt = _Prompt()
    with set_app(prompt.app):
        live = {tuple(b.keys): b.filter() for b in prompt.keys.bindings}
        loop = asyncio.new_event_loop()
        try:
            prompt.app.future = loop.create_future()
            prompt.app.future.set_result(None)
            done = {tuple(b.keys): b.filter() for b in prompt.keys.bindings}
        finally:
            loop.close()
    guarded = {
        (Keys.ControlM,),
        (Keys.ControlI,),
        (Keys.Up,),
        (Keys.Down,),
        (Keys.Left,),
    }
    assert all(live.values())
    assert {keys for keys, active in done.items() if not active} == guarded


# ---- PromptSource ----------------------------------------------------------


async def _line(
    keys: str,
    panes: Panes | None = None,
    *,
    bound: KeyBindings | None = None,
) -> tuple[str | None, list[str]]:
    notices: list[str] = []
    with create_pipe_input() as pipe:
        session: PromptSession[str] = PromptSession(
            input=pipe,
            output=DummyOutput(),
            key_bindings=bound,
        )
        pipe.send_text(keys)
        source = PromptSource(session, panes=panes or Panes(), notice=notices.append)
        line = await source.line()
    return line, notices


@pytest.mark.asyncio
async def test_a_typed_line_is_returned() -> None:
    assert await _line("hello\r") == ("hello", [])


_LEFT = "\x1b[D"


@pytest.mark.asyncio
async def test_left_on_an_empty_line_asks_for_the_tree() -> None:
    keys = key_bindings(_runtime(), Panes(), Navigation())
    assert await _line(_LEFT, bound=keys) == (OPEN_TREE, [])


@pytest.mark.asyncio
async def test_left_on_a_typed_line_moves_the_cursor_and_opens_nothing() -> None:
    keys = key_bindings(_runtime(), Panes(), Navigation())
    # A command, so Enter returns the line rather than sending it.
    assert await _line(f"/ac{_LEFT}b\r", bound=keys) == ("/abc", [])


def test_an_unchanged_frame_never_repaints_and_a_changed_one_does() -> None:
    frames = iter(["idle", "idle", "idle", "busy", "busy"])
    repaints: list[None] = []

    async def go() -> None:
        task = asyncio.create_task(
            repaint_on_change(
                lambda: repaints.append(None),
                lambda: next(frames, "busy"),
                interval_sec=0,
            ),
        )
        for _ in range(10):
            await asyncio.sleep(0)
        task.cancel()

    asyncio.run(go())
    assert len(repaints) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("keys", ["\x04", "/quit\r", "/exit\r", "  /QUIT \r"])
async def test_ctrl_d_quit_and_exit_end_the_session(keys: str) -> None:
    assert await _line(keys) == (None, [])


@pytest.mark.asyncio
async def test_a_stray_interrupt_reprompts_instead_of_quitting() -> None:
    assert await _line("abc\x03next\r") == ("next", [])


@pytest.mark.asyncio
async def test_our_ctrl_c_wins_over_the_session_s_own() -> None:
    """The session's own binding raises; ours halts a busy agent and keeps the line."""
    runtime, panes = _runtime(), Panes()
    runtime.cohort.add("c1")
    history = InMemoryHistory()
    keys = key_bindings(runtime, panes, Navigation())
    with create_pipe_input() as pipe:
        session: PromptSession[str] = PromptSession(
            input=pipe,
            output=DummyOutput(),
            key_bindings=keys,
            history=history,
        )
        pipe.send_text("abc\x03/after\r")
        line = await PromptSource(
            session,
            panes=panes,
            notice=list[str]().append,
        ).line()
    assert line == "/after"
    assert runtime.inbox.drain_nowait() == [types.Halt()]
    assert "abc" in list(history.get_strings())


@pytest.mark.asyncio
async def test_a_prompt_leaves_the_loop_s_exception_handler_alone() -> None:
    """Prompt-toolkit's own handler waits for Enter on any error in the loop."""
    loop = asyncio.get_running_loop()
    handler = loop.get_exception_handler()
    seen: list[object] = []
    with create_pipe_input() as pipe:
        session: PromptSession[str] = PromptSession(input=pipe, output=DummyOutput())
        source = PromptSource(session, panes=Panes(), notice=list[str]().append)
        reading = asyncio.create_task(source.line())
        await asyncio.sleep(0.02)
        seen.append(loop.get_exception_handler())
        pipe.send_text("x\r")
        assert await reading == "x"
    assert seen == [handler]


@pytest.mark.asyncio
async def test_a_prompt_leaves_the_loop_s_sigint_handler_installed() -> None:
    """Prompt-toolkit removes the handler on exit, so a later Ctrl+C did nothing."""
    loop = asyncio.get_running_loop()
    hits: list[str] = []
    loop.add_signal_handler(signal.SIGINT, hits.append, "sigint")
    try:
        assert await _line("x\r") == ("x", [])
        signal.raise_signal(signal.SIGINT)
        await asyncio.sleep(0.02)
    finally:
        loop.remove_signal_handler(signal.SIGINT)
    assert hits == ["sigint"]


@pytest.mark.asyncio
async def test_quitting_says_what_waiting_input_it_discards() -> None:
    panes = Panes()
    panes.stage("queued\n  text", deferred=False)
    assert await _line("\x04", panes) == (
        None,
        ["[discarding 1 queued message: queued text]"],
    )
    assert (panes.queue, panes.deferred) == (None, None)
    panes.stage("q", deferred=False)
    panes.stage("d", deferred=True)
    assert await _line("/quit\r", panes) == (
        None,
        ["[discarding 2 queued messages: q d]"],
    )


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
