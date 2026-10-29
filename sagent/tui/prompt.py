r"""The prompt: what the user types, and the two panes of what waits above it.

::

    [deferred] after this round     Tab while busy: sent once the agent idles
    queued text                     Enter while busy: sent at the next safe point
    > _

Each pane holds at most one message; staging into a full one appends with a
blank line. Idle, Enter and Tab send at once.

Up and Down walk one cursor over the stops ``[input, queued?, deferred?,
sent[-1], sent[-2], ...]``. The buffer holds the stop the cursor is on; each
stop keeps its own edits, so Down replays them. Leaving a pane's stop upward
unchanged puts its message back; leaving it edited or cleared empties the
pane for good, the edit riding the cursor.

Keys: Enter sends or queues (``foo\`` Enter is a newline), Tab defers,
Alt+Enter is a newline, Shift+Up/Down search history by prefix, Ctrl+X Ctrl+E
opens ``$EDITOR``, Ctrl+C halts a busy agent and clears the line, Ctrl+Z
suspends, Ctrl+_ undoes, Left on an empty line opens the team's tree.

Typical usage example:

  panes = Panes()
  keys = key_bindings(agent.runtime, panes, Navigation())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final, Literal

import asyncio
import functools

from prompt_toolkit.buffer import Buffer
from prompt_toolkit.filters import is_done
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.key_binding import KeyBindings

from sagent.runtime import types
from sagent.session.types import Attachment, UserMessage


if TYPE_CHECKING:
    from collections.abc import Callable

    from prompt_toolkit import PromptSession
    from prompt_toolkit.key_binding import KeyPressEvent

    from sagent.model.types import Response
    from sagent.runtime.runtime import Runtime
    from sagent.tui.types import DeferredPane, Fragments, InputPane


__all__ = [
    "OPEN_TREE",
    "Deferred",
    "Input",
    "Navigation",
    "Panes",
    "PromptSource",
    "Staged",
    "commit_panes",
    "key_bindings",
    "prompt_text",
    "repaint_on_change",
]


OPEN_TREE: Final = "\x00tree"
"""What the prompt returns for Left on an empty line; no typed line holds a NUL."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Staged:
    """The message one pane holds."""

    text: str
    attachments: tuple[Attachment, ...] = ()

    def then(self, text: str, attachments: tuple[Attachment, ...]) -> Staged:
        """Return this message with ``text`` appended after a blank line."""
        return Staged(
            text=f"{self.text}\n\n{text}",
            attachments=self.attachments + attachments,
        )


@dataclass(slots=True, kw_only=True)
class Panes:
    """The queue pane and the deferred pane."""

    queue: Staged | None = None
    """Sent at the next safe point."""

    deferred: Staged | None = None
    """Sent once the agent idles."""

    def stage(
        self,
        text: str,
        attachments: tuple[Attachment, ...] = (),
        *,
        deferred: bool,
    ) -> None:
        """Append ``text`` to the deferred pane, or the queue pane.

        Args:
          text: What the user typed.
          attachments: What rides with it.
          deferred: Whether it waits for the agent to idle.

        """
        current = self.deferred if deferred else self.queue
        staged = (
            Staged(text=text, attachments=attachments)
            if current is None
            else current.then(text, attachments)
        )
        if deferred:
            self.deferred = staged
        else:
            self.queue = staged

    def clear(self) -> None:
        """Empty both panes."""
        self.queue = None
        self.deferred = None

    def lines(self) -> list[str]:
        """Return what shows above the prompt.

        Returns:
          lines: The deferred message, prefixed ``[deferred]``, then the queued one.

        """
        return [
            *(
                [f"[deferred] {self.deferred.text}"]
                if self.deferred is not None
                else []
            ),
            *([self.queue.text] if self.queue is not None else []),
        ]

    def pop_queue(self) -> UserMessage | None:
        """Take the queued message out of its pane.

        Returns:
          message: It, as a ``UserMessage``; ``None`` when the pane is empty.

        """
        if self.queue is None:
            return None
        message = UserMessage(
            content=self.queue.text,
            attachments=self.queue.attachments,
        )
        self.queue = None
        return message

    def send_queue(self, runtime: Runtime) -> bool:
        """Send the queued message.

        Args:
          runtime: Where it goes.

        Returns:
          sent: Whether there was one.

        """
        message = self.pop_queue()
        if message is not None:
            runtime.push(message)
        return message is not None

    def send_deferred(self, runtime: Runtime) -> bool:
        """Send the deferred message, to wait until the agent idles.

        Args:
          runtime: Where it goes.

        Returns:
          sent: Whether there was one.

        """
        if self.deferred is None:
            return False
        runtime.push(
            types.UserDeferredMessage(
                text=self.deferred.text,
                attachments=self.deferred.attachments,
            ),
        )
        self.deferred = None
        return True


type Pane = Literal[
    "queue",
    "deferred",
]


@dataclass(slots=True, kw_only=True)
class Stop:
    """One place the cursor can sit."""

    current: str
    """What it holds now, edits included."""

    pane: Pane | None = None
    """The pane its message came from; ``None`` for the input or a history entry."""

    loaded: str = ""
    """What a pane's stop held when the cursor first came."""

    attachments: tuple[Attachment, ...] = ()
    consumed: bool = False
    """Whether a pane stop gave up its pane, having left it edited."""


@dataclass(slots=True, kw_only=True)
class Navigation:
    """The Up/Down cursor; ``0`` when not navigating."""

    cursor: int = 0
    stops: list[Stop] = field(default_factory=list[Stop])

    def end(self) -> None:
        """Stop navigating."""
        self.cursor = 0
        self.stops = []


def commit_panes(runtime: Runtime, panes: Panes) -> Callable[[], None]:
    """Send the panes when the runtime can take them; return the undo.

    The queue goes with the next reply, through ``runtime.before_tool_spawn``,
    and is answered once that reply's calls are; both go when the agent idles,
    or a clear completes, which a clear's wait for the user would otherwise
    never reach.

    Args:
      runtime: Where the panes are sent.
      panes: What waits.

    Returns:
      remove: Restores the hook and removes the observer; safe to call twice.

    """
    previous = runtime.before_tool_spawn

    def before(turn: Response) -> UserMessage | None:
        event = previous(turn) if previous is not None else None
        return event if event is not None else panes.pop_queue()

    def observe(event: types.RuntimeEvent) -> None:
        if isinstance(event, (types.AgentIdle, types.ClearComplete)) and not (
            panes.send_queue(runtime)
        ):
            panes.send_deferred(runtime)

    runtime.before_tool_spawn = before
    runtime.observers.append(observe)

    def remove() -> None:
        if observe in runtime.observers:
            runtime.observers.remove(observe)
        if runtime.before_tool_spawn is before:
            runtime.before_tool_spawn = previous

    return remove


class Deferred:
    """The deferred pane: the panes' messages, and what the runtime holds mid-stream.

    Args:
      runtime: Whose mid-stream messages show as ``pending:``.
      panes: What waits.

    """

    def __init__(self, runtime: Runtime, panes: Panes) -> None:
        self.runtime = runtime
        self.panes = panes

    def frame(self) -> Fragments:
        """Return what waits, then a newline.

        Returns:
          fragments: The pane's text, or none when nothing waits.

        """
        lines = [
            *self.panes.lines(),
            *(
                f"pending: {m.content}"
                for m in self.runtime.pending_mid_stream()
                if isinstance(m, UserMessage)
            ),
        ]
        if not lines:
            return ()
        return (("class:queued_input_pane", "\n\n".join(lines)), ("", "\n"))


class Input:
    """The input pane's ``> ``; prompt_toolkit draws what is typed after it."""

    def frame(self) -> Fragments:
        """Return the bold ``> ``."""
        return (("class:input_pane", "> "),)


def prompt_text(deferred: DeferredPane, entry: InputPane) -> FormattedText:
    """Return the prompt's message: a blank line, what waits, then ``> ``."""
    # The prompt is erased on submit, so this blank cannot stack on the one
    # ConsolePrinter.user writes above the sent message.
    return FormattedText([("", "\n"), *deferred.frame(), *entry.frame()])


# A repaint writes cursor and wrap escapes even when no cell changes, and some terminals
# drop the mouse selection on any write; so an idle screen must not repaint.
async def repaint_on_change(
    invalidate: Callable[[], None],
    frame: Callable[[], object],
    interval_sec: float = 0.2,
) -> None:
    """Call ``invalidate`` each ``interval_sec`` that ``frame()`` has changed.

    Args:
      invalidate: Repaints the screen.
      frame: What the screen would show, compared by equality.
      interval_sec: How often to compare.

    """
    last = frame()
    while True:
        await asyncio.sleep(interval_sec)
        if (now := frame()) != last:
            last = now
            invalidate()


class PromptSource:
    """Typed lines from a prompt-toolkit session.

    Ctrl+C never ends the session: a ``KeyboardInterrupt`` that slips past
    its binding re-arms the prompt. Ctrl+D, ``/quit`` and ``/exit`` end it,
    after saying what waiting input is discarded.

    Args:
      session: The prompt.
      panes: What waits, shown before it is discarded.
      notice: Where the discard notice goes.

    """

    def __init__(
        self,
        session: PromptSession[str],
        *,
        panes: Panes,
        notice: Callable[[str], None],
    ) -> None:
        self.session = session
        self.panes = panes
        self.notice = notice

    async def line(self) -> str | None:
        """Return the next typed line.

        Returns:
          line: What was typed; ``None`` on Ctrl+D, ``/quit``, or ``/exit``.

        """
        while True:
            try:
                # Not ``handle_sigint``: on exit it removes the loop's SIGINT handler,
                # so a Ctrl+C while a command ran reached nothing. Our ``c-c``
                # binding still takes the key while the prompt reads.
                text = await self.session.prompt_async(
                    set_exception_handler=False,
                    handle_sigint=False,
                )
            except EOFError:
                break
            except KeyboardInterrupt:
                continue
            if text.strip().lower() not in ("/quit", "/exit"):
                return text
            break
        waiting = [
            s.text for s in (self.panes.queue, self.panes.deferred) if s is not None
        ]
        if waiting:
            noun = "message" if len(waiting) == 1 else "messages"
            preview = " ".join("\n".join(waiting).split())
            self.notice(f"[discarding {len(waiting)} queued {noun}: {preview}]")
            self.panes.clear()
        return None


def key_bindings(runtime: Runtime, panes: Panes, nav: Navigation) -> KeyBindings:
    """Return the prompt's keys, acting on ``runtime`` and ``panes``.

    Args:
      runtime: Where Enter and Tab send, and what Ctrl+C halts.
      panes: What waits.
      nav: The Up/Down cursor.

    Returns:
      keys: The bindings.

    """
    keys = KeyBindings()
    # Not once the prompt is accepted: the buffer is no longer the user's.
    keys.add("enter", filter=~is_done)(functools.partial(_enter, runtime, panes, nav))
    keys.add("tab", filter=~is_done)(functools.partial(_tab, runtime, panes, nav))
    keys.add("up", filter=~is_done)(functools.partial(_up, runtime, panes, nav))
    keys.add("down", filter=~is_done)(functools.partial(_down, runtime, panes, nav))
    keys.add("left", filter=~is_done)(_left)
    keys.add("escape", "enter")(_newline)
    keys.add("s-up")(functools.partial(_search, step=Buffer.history_backward))
    keys.add("s-down")(functools.partial(_search, step=Buffer.history_forward))
    keys.add("c-x", "c-e")(_editor)
    # Not eager: the session merges these after its own bindings, so this one is
    # matched last and wins over the Ctrl+C that raises KeyboardInterrupt.
    keys.add("c-c")(functools.partial(_interrupt, runtime))
    keys.add("c-z")(_suspend)
    keys.add("c-_")(_undo)
    keys.add("escape", "z")(_undo)
    return keys


# Enter redirects a busy cohort, so it sends mid-cohort; Tab waits behind the round,
# so it stages there.
def _send_or_stage(
    runtime: Runtime,
    panes: Panes,
    text: str,
    *,
    deferred: bool,
    attachments: tuple[Attachment, ...] = (),
) -> None:
    """Send ``text`` when the runtime takes it now, else stage it in its pane."""
    if deferred and runtime.accepts_deferred_dispatch:
        runtime.push(types.UserDeferredMessage(text=text, attachments=attachments))
    elif not deferred and runtime.accepts_user_dispatch:
        runtime.push(UserMessage(content=text, attachments=attachments))
    else:
        panes.stage(text, attachments, deferred=deferred)


# A pane's own stop emptied that pane when the cursor came, so staging the line back
# replaces the message rather than doubling it.
def _commit(
    runtime: Runtime,
    panes: Panes,
    nav: Navigation,
    text: str,
    *,
    buf: Buffer,
    deferred: bool = False,
) -> None:
    """Send or stage ``text``, from the stop the cursor is on, and reset the line."""
    attachments = nav.stops[nav.cursor].attachments if nav.cursor else ()
    _send_or_stage(runtime, panes, text, deferred=deferred, attachments=attachments)
    nav.end()
    buf.append_to_history()
    buf.reset()


def _enter(
    runtime: Runtime,
    panes: Panes,
    nav: Navigation,
    event: KeyPressEvent,
) -> None:
    r"""Run a command, continue a ``\`` line, or send or queue the line."""
    buf = event.current_buffer
    text = buf.text
    if text.strip().startswith("/"):
        # The command leaves the pane the cursor sat on as it was.
        if nav.cursor:
            stop = nav.stops[nav.cursor]
            if stop.pane is not None and not stop.consumed:
                _restore(panes, stop, stop.loaded, runtime=None)
        nav.end()
        buf.validate_and_handle()
        return
    if text.endswith("\\"):
        buf.text = text[:-1] + "\n"
        buf.cursor_position = len(buf.text)
        return
    if not text.strip():
        nav.end()
        buf.reset()
        return
    _commit(runtime, panes, nav, text, buf=buf)


def _tab(runtime: Runtime, panes: Panes, nav: Navigation, event: KeyPressEvent) -> None:
    """Defer the line: send it once the agent idles."""
    buf = event.current_buffer
    if not buf.text.strip():
        nav.end()
        buf.reset()
        return
    _commit(runtime, panes, nav, buf.text, deferred=True, buf=buf)


def _up(runtime: Runtime, panes: Panes, nav: Navigation, event: KeyPressEvent) -> None:
    """Move the cursor to the next older stop."""
    buf = event.current_buffer
    if not nav.cursor:
        stops = [Stop(current=buf.text)]
        panes_by_name: tuple[tuple[Pane, Staged | None], ...] = (
            ("queue", panes.queue),
            ("deferred", panes.deferred),
        )
        for pane, staged in panes_by_name:
            if staged is not None:
                stops.append(
                    Stop(
                        pane=pane,
                        loaded=staged.text,
                        current=staged.text,
                        attachments=staged.attachments,
                    ),
                )
        stops.extend(
            Stop(current=entry) for entry in reversed(list(buf.history.get_strings()))
        )
        if len(stops) > 1:
            nav.stops, nav.cursor = stops, 1
            _enter_stop(panes, stops[1], buf)
        return
    stop = nav.stops[nav.cursor]
    stop.current = buf.text
    if nav.cursor + 1 >= len(nav.stops):
        return
    if stop.pane is not None and not stop.consumed:
        if stop.current != stop.loaded:
            stop.consumed = True
        else:
            _restore(panes, stop, stop.current, runtime=runtime)
    nav.cursor += 1
    _enter_stop(panes, nav.stops[nav.cursor], buf)


def _down(
    runtime: Runtime,
    panes: Panes,
    nav: Navigation,
    event: KeyPressEvent,
) -> None:
    """Move the cursor back toward the input, each stop's edits intact."""
    buf = event.current_buffer
    if not nav.cursor:
        if buf.text:
            buf.reset()
        return
    stop = nav.stops[nav.cursor]
    stop.current = buf.text
    if stop.pane is not None and not stop.consumed:
        _restore(panes, stop, stop.current, runtime=runtime)
    nav.cursor -= 1
    if nav.cursor:
        _enter_stop(panes, nav.stops[nav.cursor], buf)
        return
    buf.text = nav.stops[0].current
    buf.cursor_position = len(buf.text)
    nav.end()


def _enter_stop(panes: Panes, stop: Stop, buf: Buffer) -> None:
    """Load ``stop``; a pane still holding its message empties while the cursor sits."""
    if not stop.consumed:
        if stop.pane == "queue":
            panes.queue = None
        elif stop.pane == "deferred":
            panes.deferred = None
    buf.text = stop.current
    buf.cursor_position = len(buf.text)


# Through the runtime when it is known: the agent may have idled while the pane sat
# empty, and a pane written back would then wait for an idle that never comes.
def _restore(panes: Panes, stop: Stop, text: str, *, runtime: Runtime | None) -> None:
    """Put ``text`` back where ``stop`` came from."""
    deferred = stop.pane == "deferred"
    if runtime is not None:
        _send_or_stage(
            runtime,
            panes,
            text,
            deferred=deferred,
            attachments=stop.attachments,
        )
        return
    staged = Staged(text=text, attachments=stop.attachments)
    if deferred:
        panes.deferred = staged
    else:
        panes.queue = staged


def _left(event: KeyPressEvent) -> None:
    """Move the cursor left; on an empty line, end the prompt to open the tree."""
    buf = event.current_buffer
    if buf.text:
        buf.cursor_position += buf.document.get_cursor_left_position()
        return
    event.app.exit(result=OPEN_TREE)


def _newline(event: KeyPressEvent) -> None:
    """Insert a newline."""
    event.current_buffer.insert_text("\n")


def _search(event: KeyPressEvent, *, step: Callable[[Buffer], None]) -> None:
    """Step through history to the next entry starting with what precedes the cursor."""
    buf = event.current_buffer
    prefix = buf.document.text_before_cursor
    start = buf.working_index
    while True:
        before = buf.working_index
        step(buf)
        if buf.working_index == before:
            buf.working_index = start
            break
        if buf.document.text.startswith(prefix):
            break
    # Stepping puts the cursor at the end; left there, the next press would
    # search by the whole entry and walk past every other match to the end.
    buf.cursor_position = len(prefix)


def _editor(event: KeyPressEvent) -> None:
    """Open the line in ``$EDITOR``; the agent pauses until it exits."""
    event.current_buffer.open_in_editor()


def _interrupt(runtime: Runtime, event: KeyPressEvent) -> None:
    """Halt a busy agent, and abandon the line to history; never exit."""
    if (
        runtime.model_call is not None
        or runtime.compact_task is not None
        or runtime.cohort
    ):
        runtime.halt()
    event.current_buffer.reset(append_to_history=True)


def _suspend(event: KeyPressEvent) -> None:
    """Suspend to the background."""
    event.app.suspend_to_background()


def _undo(event: KeyPressEvent) -> None:
    """Undo the last edit."""
    event.current_buffer.undo()
