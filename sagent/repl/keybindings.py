r"""prompt-toolkit keybindings.

Wires keys to the input-pane behavior contract specified in
``docs/private/input_ux.md``. That doc is the authority; this module is
the wiring:

- ``enter``        -> :func:`_kb_submit` (dispatch or stage to queue pane)
- ``tab``          -> :func:`_kb_defer` (dispatch or stage to deferred pane)
- ``up``           -> :func:`_kb_up` (walk toward older stops)
- ``down``         -> :func:`_kb_down` (walk toward the input stop)
- ``escape enter`` -> literal newline (Alt+Enter)
- ``s-up`` / ``s-down`` -> prefix history search
- ``c-x c-e``      -> open buffer in ``$EDITOR``
- ``c-c``          -> ``agent.halt()`` when active; clear buffer when idle
- ``c-z``          -> suspend to background
- ``c-_`` / ``escape z`` -> undo

Navigation model (see ``input_ux.md`` for the full spec): Up/Down move a
single cursor over an ordered list of *stops* built when navigation
begins::

    [ input value, queue message?, deferred message?, sent[-1], sent[-2], ... ]

Each stop owns its current value. There is no snapshot. The buffer is
the only mutable truth; the stop the cursor sits on has its value in the
buffer. Up applies the *modified-test*: leaving a pane stop with the
buffer unchanged restores that pane's message; leaving it modified (or
cleared) does not, and the edited value rides the cursor. Down is a pure
replay back toward the input, handing each stop its current value with
edits intact.

Backslash continuation: ``foo\`` + Enter inserts a literal newline (no
dispatch); see ``_kb_submit``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING

import functools

from prompt_toolkit.filters import is_done
from prompt_toolkit.key_binding import KeyBindings

from sagent.repl.input_queues import InputQueues, QueuedInputBlock
from sagent.types.runtime import (
    BytesMessage,
    UserDeferredMessage,
    UserMessage,
)


if TYPE_CHECKING:
    from prompt_toolkit.buffer import Buffer
    from prompt_toolkit.key_binding import KeyPressEvent

    from sagent.agent.agent import Agent


class StopKind(Enum):
    """Which surface a navigation stop's value came from."""

    INPUT = auto()
    QUEUE = auto()
    DEFERRED = auto()
    HISTORY = auto()


@dataclass(slots=True, kw_only=True)
class Stop:
    """One position in the Up/Down walk.

    ``loaded`` is the value this stop held when the cursor first arrived
    (the modified-test compares the buffer against it). ``current`` is
    the value the stop holds now -- updated from the buffer whenever the
    cursor leaves the stop, so Down replays edits intact.

    ``consumed`` applies only to pane stops (QUEUE / DEFERRED). It starts
    ``False`` ("the pane still owns a message"). The first Up that leaves
    the stop runs the modified-test once: an edited/cleared value sets
    ``consumed = True`` permanently -- the pane is emptied for good and
    the stop behaves like a plain history edit position thereafter (no
    pane interaction in either direction). An unchanged pass-through
    leaves ``consumed = False``, so the pane keeps tracking the cursor
    (empty while sat on, restored when left).
    """

    kind: StopKind
    loaded: str
    current: str
    attachments: tuple[BytesMessage, ...] = ()
    consumed: bool = False


@dataclass(slots=True, kw_only=True)
class NavState:
    """Up/Down navigation cursor over a list of stops.

    ``cursor == 0`` means "not navigating": the buffer is live input and
    ``stops`` is empty. ``cursor > 0`` means navigation is active;
    ``stops[cursor]`` is loaded in the buffer. ``stops[0]`` is always the
    INPUT stop (the buffer value at navigation start). Pane stops (QUEUE,
    DEFERRED) follow, then HISTORY stops most-recent-first.
    """

    cursor: int = 0
    stops: list[Stop] = field(default_factory=list)

    def active(self) -> bool:
        """Return whether navigation is in progress."""
        return self.cursor > 0

    def end(self) -> None:
        """Clear navigation state."""
        self.cursor = 0
        self.stops = []


def build_key_bindings(
    agent: Agent, queues: InputQueues, nav: NavState | None = None
) -> KeyBindings:
    """Build the REPL keybindings bound to ``agent``, ``queues``, and ``nav``.

    Args:
      agent: Agent these key handlers will mutate.
      queues: The queue and deferred panes.
      nav: Up/Down navigation state. Created fresh if omitted.

    Returns:
      kb: Configured ``KeyBindings``.

    """
    if nav is None:
        nav = NavState()
    kb = KeyBindings()
    kb.add("enter", filter=~is_done)(
        functools.partial(_kb_submit, agent, queues, nav),
    )
    kb.add("tab")(functools.partial(_kb_defer, agent, queues, nav))
    kb.add("down")(functools.partial(_kb_down, queues, nav))
    kb.add("escape", "enter")(_kb_newline)
    kb.add("up")(functools.partial(_kb_up, queues, nav))
    kb.add("s-up")(_kb_history_prefix_back)
    kb.add("s-down")(_kb_history_prefix_fwd)
    kb.add("c-x", "c-e")(_kb_open_editor)
    # ``eager=True`` is load-bearing: ``PromptSession`` merges this
    # binding *before* its own default ``c-c`` -> ``_keyboard_interrupt``,
    # but the key processor invokes ``matches[-1]`` (the last match),
    # which would be prompt-toolkit's default -- raising ``KeyboardInterrupt``
    # and shadowing this handler entirely. An eager binding filters out
    # the non-eager default, so ``_kb_ctrl_c`` (halt when busy, clear the
    # line when idle) actually runs and Ctrl+C never exits the REPL.
    kb.add("c-c", eager=True)(functools.partial(_kb_ctrl_c, agent))
    kb.add("c-z")(_kb_suspend)
    kb.add("c-_")(_kb_undo)
    kb.add("escape", "z")(_kb_undo)
    return kb


def _stage_or_dispatch(
    agent: Agent,
    queues: InputQueues,
    text: str,
    *,
    deferred: bool,
    attachments: tuple[BytesMessage, ...] = (),
) -> None:
    """Dispatch ``text`` when the runtime accepts it, else stage into a pane.

    Dispatch-vs-stage consults the predicate matching the key's intent:
    Enter (``deferred=False``) uses ``accepts_user_dispatch`` -- it
    dispatches mid-cohort to redirect. Tab (``deferred=True``) uses
    ``accepts_deferred_dispatch`` -- it STAGES mid-cohort, because a defer
    must wait behind the running round chain rather than preempt it. The
    two predicates differ only in the mid-cohort state. ``attachments``
    ride the committed message (a lifted queued/deferred block may carry
    image/PDF payloads). Empty/whitespace text is a no-op (the caller
    guards this) and never reaches here.
    """
    dispatches = (
        agent.runtime.accepts_deferred_dispatch
        if deferred
        else agent.runtime.accepts_user_dispatch
    )
    if dispatches:
        message = (
            UserDeferredMessage(text=text, attachments=attachments)
            if deferred
            else UserMessage(text=text, attachments=attachments)
        )
        agent.runtime.inbox.push_back(message)
        return
    if deferred:
        queues.stage_deferred(text, attachments)
    else:
        queues.stage_queue(text, attachments)


def _commit_nav_stop(
    queues: InputQueues, nav: NavState, text: str, *, deferred: bool, agent: Agent
) -> None:
    """Commit ``text`` from the current nav stop, then end navigation.

    Placement follows the spec: at a pane's OWN stop, staging replaces
    that pane's message (no doubling); at any other stop, staging appends
    via the pane's coalesce. Idle dispatches immediately regardless of
    stop. The current stop's attachments ride the committed message so a
    lifted image/PDF payload is never silently dropped. The buffer is
    cleared by the caller via ``nav.end`` + reset.
    """
    stop = nav.stops[nav.cursor]
    own_pane = (stop.kind is StopKind.QUEUE and not deferred) or (
        stop.kind is StopKind.DEFERRED and deferred
    )
    if own_pane:
        # Replace: this stop emptied its own pane on unlift; staging the
        # edited value back must not coalesce with a phantom existing.
        if deferred:
            queues.deferred = None
        else:
            queues.queue = None
    _stage_or_dispatch(
        agent, queues, text, deferred=deferred, attachments=stop.attachments
    )


def _kb_submit(
    agent: Agent,
    queues: InputQueues,
    nav: NavState,
    event: KeyPressEvent,
) -> None:
    r"""Enter handler. See ``docs/private/input_ux.md`` for the contract.

    Branches:

    - Slash command: route through the pump via ``validate_and_handle``.
    - Buffer ends with ``\``: backslash continuation -> literal newline.
    - Empty / whitespace-only buffer: no-op; ends navigation.
    - Navigation active: commit the stop's value (replace own pane /
      append elsewhere / dispatch when idle); end navigation.
    - Not navigating: dispatch when idle, else stage into the queue pane.
    """
    buf = event.current_buffer
    text = buf.text
    stripped = text.strip()
    if stripped.startswith("/"):
        # A slash command abandons any in-flight navigation: restore the
        # sat-on pane (the buffer holds the command, not the lifted pane
        # value) before the command routes through the pump.
        _abandon_navigation(queues, nav)
        buf.validate_and_handle()
        return
    if text.endswith("\\"):
        buf.text = text[:-1] + "\n"
        buf.cursor_position = len(buf.text)
        return
    if not stripped:
        # Empty/whitespace is a no-op and ends navigation: a cleared stop
        # is the delete gesture, already realized by Up's modified-test;
        # there is nothing to commit and nothing to restore.
        nav.end()
        buf.reset()
        return
    if nav.active():
        _commit_nav_stop(queues, nav, text, deferred=False, agent=agent)
        nav.end()
        buf.append_to_history()
        buf.reset()
        return
    _stage_or_dispatch(agent, queues, text, deferred=False)
    buf.append_to_history()
    buf.reset()


def _kb_defer(
    agent: Agent,
    queues: InputQueues,
    nav: NavState,
    event: KeyPressEvent,
) -> None:
    """Tab handler. See ``docs/private/input_ux.md`` for the contract.

    Mirror of Enter with the deferred pane as the target: dispatch when
    idle (as a ``UserDeferredMessage``), else stage into the deferred
    pane. Empty/whitespace is a no-op and ends navigation. During
    navigation, commit the stop's value to the deferred pane (replace own
    pane / append elsewhere) and end navigation.
    """
    buf = event.current_buffer
    text = buf.text
    if not text.strip():
        nav.end()
        buf.reset()
        return
    if nav.active():
        _commit_nav_stop(queues, nav, text, deferred=True, agent=agent)
        nav.end()
        buf.append_to_history()
        buf.reset()
        return
    _stage_or_dispatch(agent, queues, text, deferred=True)
    buf.append_to_history()
    buf.reset()


def _kb_up(
    queues: InputQueues,
    nav: NavState,
    event: KeyPressEvent,
) -> None:
    """Up handler. Walk toward older stops; see ``input_ux.md``.

    First Up builds the stop list from the current input, the panes, and
    sent history. Each subsequent Up applies the modified-test to the
    stop being left, then advances. Up at the oldest stop is a no-op.
    """
    buf = event.current_buffer
    if not nav.active():
        _begin_navigation(queues, nav, buf)
        return
    current = nav.stops[nav.cursor]
    current.current = buf.text
    if nav.cursor + 1 >= len(nav.stops):
        return  # Oldest stop -- no-op (hard top).
    _leave_stop_upward(queues, current)
    nav.cursor += 1
    _enter_stop(queues, nav.stops[nav.cursor], buf)


def _kb_down(
    queues: InputQueues,
    nav: NavState,
    event: KeyPressEvent,
) -> None:
    """Down handler. Walk back toward the input stop; see ``input_ux.md``.

    Pure replay: hand back each stop's current value with edits intact;
    never re-derive. Leaving a pane stop downward restores that pane (the
    cursor no longer sits on it). Down at the input stop ends navigation.
    """
    buf = event.current_buffer
    if not nav.active():
        if buf.text:
            buf.reset()
        return
    current = nav.stops[nav.cursor]
    current.current = buf.text
    _leave_stop_downward(queues, current)
    nav.cursor -= 1
    if nav.cursor == 0:
        # Back at the input stop: restore its value and end navigation.
        buf.text = nav.stops[0].current
        buf.cursor_position = len(buf.text)
        nav.end()
        return
    _enter_stop(queues, nav.stops[nav.cursor], buf)


def _begin_navigation(queues: InputQueues, nav: NavState, buf: Buffer) -> None:
    """Build the stop list and load the first stop above the input."""
    stops: list[Stop] = [Stop(kind=StopKind.INPUT, loaded=buf.text, current=buf.text)]
    if queues.queue is not None:
        stops.append(
            Stop(
                kind=StopKind.QUEUE,
                loaded=queues.queue.text,
                current=queues.queue.text,
                attachments=queues.queue.attachments,
            )
        )
    if queues.deferred is not None:
        stops.append(
            Stop(
                kind=StopKind.DEFERRED,
                loaded=queues.deferred.text,
                current=queues.deferred.text,
                attachments=queues.deferred.attachments,
            )
        )
    stops.extend(
        Stop(kind=StopKind.HISTORY, loaded=entry, current=entry)
        for entry in reversed(_history_strings(buf))
    )
    if len(stops) == 1:
        return  # Nothing above the input -- no-op.
    nav.stops = stops
    nav.cursor = 1
    _enter_stop(queues, nav.stops[1], buf)


def _enter_stop(queues: InputQueues, stop: Stop, buf: Buffer) -> None:
    """Load ``stop`` into the buffer; empty its pane if it is a live pane stop.

    A pane stop that still owns its message (``not consumed``) empties
    its pane while the cursor sits on it -- its value is in the buffer.
    A consumed pane stop, or a history/input stop, owns no live pane.
    """
    if not stop.consumed:
        if stop.kind is StopKind.QUEUE:
            queues.queue = None
        elif stop.kind is StopKind.DEFERRED:
            queues.deferred = None
    buf.text = stop.current
    buf.cursor_position = len(buf.text)


def _leave_stop_upward(queues: InputQueues, stop: Stop) -> None:
    """Modified-test as the cursor leaves ``stop`` going up.

    Runs once per pane stop. Unchanged -> restore the pane (scrolling
    past). Modified or cleared -> mark the stop consumed; the pane stays
    empty and the edited value rides the cursor. No-op for non-pane or
    already-consumed stops.
    """
    if stop.kind not in (StopKind.QUEUE, StopKind.DEFERRED) or stop.consumed:
        return
    if stop.current != stop.loaded:
        stop.consumed = True
        return
    _restore_pane(queues, stop, stop.current)


def _leave_stop_downward(queues: InputQueues, stop: Stop) -> None:
    """Restore a live pane stop's pane as the cursor leaves it going down.

    Down never re-derives: it restores the pane to the stop's CURRENT
    value (edits intact). A consumed pane stop owns no pane. The
    modified-test is Up-only, so Down does not consult ``loaded``.
    """
    if stop.kind not in (StopKind.QUEUE, StopKind.DEFERRED) or stop.consumed:
        return
    _restore_pane(queues, stop, stop.current)


def _restore_pane(queues: InputQueues, stop: Stop, text: str) -> None:
    """Put ``text`` back into the pane ``stop`` represents."""
    block = QueuedInputBlock(text=text, attachments=stop.attachments)
    if stop.kind is StopKind.QUEUE:
        queues.queue = block
    elif stop.kind is StopKind.DEFERRED:
        queues.deferred = block


def _abandon_navigation(queues: InputQueues, nav: NavState) -> None:
    """Restore the sat-on live pane stop and end navigation.

    Used when the user runs a slash command mid-navigation: the buffer
    now holds the command, not the lifted pane content, so the pane the
    cursor sat on must return to its ``loaded`` value rather than vanish.
    A consumed or non-pane stop owns no live pane and needs no restore.
    """
    if nav.active():
        stop = nav.stops[nav.cursor]
        if stop.kind in (StopKind.QUEUE, StopKind.DEFERRED) and not stop.consumed:
            _restore_pane(queues, stop, stop.loaded)
    nav.end()


def _kb_newline(event: KeyPressEvent) -> None:
    """Insert a literal newline into the current buffer (Alt+Enter)."""
    event.current_buffer.insert_text("\n")


def _history_strings(buf: object) -> list[str]:
    """Return the sagent input history entries, oldest-first.

    The walk reads the REPL history file (prompt-toolkit ``FileHistory``).
    Accessed via duck-typing so tests can supply ``MagicMock`` buffers
    without a real ``History``.
    """
    history = getattr(buf, "history", None)
    if history is None:
        return []
    get_strings = getattr(history, "get_strings", None)
    if get_strings is None:
        return []
    strings = get_strings()
    return list(strings) if strings is not None else []


def _kb_history_prefix_back(event: KeyPressEvent) -> None:
    """Walk history backward to the next entry matching the pre-cursor prefix."""
    buf = event.current_buffer
    text = buf.document.text_before_cursor
    while True:
        prev_index = buf.working_index
        buf.history_backward(count=1)
        if buf.working_index == prev_index:
            break
        if buf.document.text[: len(text)] == text:
            break


def _kb_history_prefix_fwd(event: KeyPressEvent) -> None:
    """Walk history forward to the next entry matching the pre-cursor prefix."""
    buf = event.current_buffer
    text = buf.document.text_before_cursor
    while True:
        prev_index = buf.working_index
        buf.history_forward(count=1)
        if buf.working_index == prev_index:
            break
        if buf.document.text[: len(text)] == text:
            break


def _kb_open_editor(event: KeyPressEvent) -> None:
    """Open the current buffer in ``$EDITOR`` (Ctrl+X Ctrl+E).

    Blocks the prompt-toolkit event loop until the editor exits --
    prompt-toolkit's ``open_in_editor`` shells out synchronously. The
    REPL renderer and the underlying agent loop both pause for the
    editor session; this is documented prompt-toolkit behavior.
    """
    event.current_buffer.open_in_editor()


def _kb_ctrl_c(agent: Agent, event: KeyPressEvent) -> None:
    r"""Abandon the line being composed; never exit the REPL.

    One rule, matching every Unix line editor: Ctrl+C abandons the line
    you are composing and gives you a fresh prompt. The abandoned text is
    recorded in the sagent input history (Up-arrow recalls it) but is
    never carried into the next turn.

    When the agent is busy (``agent.work`` -- a model call or compaction
    -- or a non-empty ``runtime.cohort``), Ctrl+C also halts the running
    turn. Queued/deferred panes are deliberately submitted content, not
    the line being composed, so Ctrl+C leaves them untouched.

    To exit the REPL use Ctrl+D or ``/quit``.
    """
    if agent.work is not None or agent.runtime.cohort:
        agent.halt()
    event.current_buffer.reset(append_to_history=True)


def _kb_suspend(event: KeyPressEvent) -> None:
    """Suspend the REPL to background (Ctrl+Z)."""
    event.app.suspend_to_background()


def _kb_undo(event: KeyPressEvent) -> None:
    """Undo the last buffer edit (Ctrl+_ / Esc-z)."""
    event.current_buffer.undo()
