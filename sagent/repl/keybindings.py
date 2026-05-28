r"""prompt-toolkit keybindings.

Wires keys to the input-pane behavior contract. The contract itself
(Up / Down / Enter / Tab semantics, including the navigation snapshot
mechanism) is documented in :mod:`repl.input_pane`'s module docstring
-- the behavior is a property of the input zone, not of the binding.

This module is the wiring:

- ``enter``        -> :func:`_kb_submit` (preempt-dispatch OR queue-commit)
- ``tab``          -> :func:`_kb_defer` (queue-commit)
- ``down``         -> :func:`_kb_down` (navigation walk back)
- ``up``           -> :func:`_kb_up` (navigation walk older)
- ``escape enter`` -> literal newline (Alt+Enter)
- ``s-up`` / ``s-down`` -> prefix history search
- ``c-x c-e``      -> open buffer in ``$EDITOR``
- ``c-c``          -> ``agent.halt()`` when active; clear buffer when idle
- ``c-z``          -> suspend to background
- ``c-_`` / ``escape z`` -> undo

Up/Down navigation maintains a :class:`NavState` (cursor + snapshot)
so the user can scroll through queue+history and have their original
state restored on the final Down. See :mod:`repl.input_pane`'s
``Behavior contract: Up / Down navigation`` section for the figure.

Backslash continuation: ``foo\`` + Enter inserts a literal newline
(no dispatch); see ``_kb_submit``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import functools

from prompt_toolkit.filters import is_done
from prompt_toolkit.key_binding import KeyBindings

from sagent.types.runtime import UserMessage, UserQueuedMessage


if TYPE_CHECKING:
    from prompt_toolkit.key_binding import KeyPressEvent

    from sagent.agent.agent import Agent


@dataclass(slots=True, kw_only=True)
class NavState:
    """Up/Down navigation cursor + snapshot.

    ``cursor == 0`` means "not navigating": the buffer holds the user's
    typing and ``queued_input`` is shown in ``queued_input_pane`` as
    usual. ``cursor > 0`` means a navigation is in progress; ``snapshot``
    holds the ``(queued_input_copy, buffer_text)`` captured at the
    moment the first Up was pressed. Final Down restores the snapshot;
    Enter / Tab at ``cursor > 0`` commit the buffer as a queued block
    and restore the rest of the snapshot. See the contract figure in
    :mod:`repl.input_pane`.
    """

    cursor: int = 0
    snapshot_queue: tuple[str, ...] = field(default_factory=tuple)
    snapshot_input: str = ""

    def begin(self, queued: list[str], buffer_text: str) -> None:
        """Capture a snapshot at the start of navigation."""
        self.snapshot_queue = tuple(queued)
        self.snapshot_input = buffer_text
        self.cursor = 1

    def end(self) -> None:
        """Clear the snapshot when navigation completes."""
        self.cursor = 0
        self.snapshot_queue = ()
        self.snapshot_input = ""


def build_key_bindings(
    agent: Agent, queued_input: list[str], nav: NavState | None = None
) -> KeyBindings:
    """Build the REPL keybindings bound to ``agent``, ``queued_input``, and ``nav``.

    Args:
      agent: Agent these key handlers will mutate.
      queued_input: REPL-local staging buffer for queued blocks.
      nav: Up/Down navigation state. Created fresh if omitted (for
          callers that don't share state across binding builds).

    Returns:
      kb: Configured ``KeyBindings``.

    """
    if nav is None:
        nav = NavState()
    kb = KeyBindings()
    kb.add("enter", filter=~is_done)(
        functools.partial(_kb_submit, agent, queued_input, nav),
    )
    kb.add("tab")(functools.partial(_kb_defer, agent, queued_input, nav))
    kb.add("down")(functools.partial(_kb_down, queued_input, nav))
    kb.add("escape", "enter")(_kb_newline)
    kb.add("up")(functools.partial(_kb_up, queued_input, nav))
    kb.add("s-up")(_kb_history_prefix_back)
    kb.add("s-down")(_kb_history_prefix_fwd)
    kb.add("c-x", "c-e")(_kb_open_editor)
    kb.add("c-c")(functools.partial(_kb_ctrl_c, agent))
    kb.add("c-z")(_kb_suspend)
    kb.add("c-_")(_kb_undo)
    kb.add("escape", "z")(_kb_undo)
    return kb


def _commit_queued_and_restore(
    queued_input: list[str], nav: NavState, buf_text: str
) -> str:
    """Commit ``buf_text`` to queue + restore snapshot; return restored buffer.

    Shared helper for Enter and Tab at ``cursor > 0``. Two semantic
    cases:

    - Case 1 at ``cursor == 1`` (user lifted the queue and is *editing*
      it; "up once -> dequeue"): the queue is *replaced* with
      ``[buf_text]``. The lifted-then-edited content overwrites the
      original queue. Without this, hitting Enter after a no-op Up
      would duplicate the queue's content into ``[queue, queue]``.
    - All other ``cursor > 0`` states (user *scrolled past* into
      history, or Case 2 has no queue to edit): the queue becomes
      ``snapshot_queue + [buf_text]`` -- a true extension that
      preserves the original queue and adds the navigated/edited
      content as a new block.

    The snapshot is cleared and the snapshot-input is returned so the
    caller can restore the buffer to the user's pre-navigation typing.
    """
    queued_input.clear()
    if nav.cursor == 1 and nav.snapshot_queue:
        # Case 1, edit-mode: replace queue with the (possibly edited)
        # lifted content.
        queued_input.append(buf_text)
    else:
        queued_input.extend(nav.snapshot_queue)
        queued_input.append(buf_text)
    restored = nav.snapshot_input
    nav.end()
    return restored


def _flush_deferred_if_ready(agent: Agent, queued_input: list[str]) -> None:
    """Dispatch deferred input when no active round can produce an idle edge."""
    if not queued_input:
        return
    if agent.runtime.inbox.gate_armed:
        item = UserMessage(text="\n\n".join(queued_input))
    elif agent.work is None and not agent.runtime.cohort:
        item = UserQueuedMessage(text="\n\n".join(queued_input))
    else:
        return
    queued_input.clear()
    agent.runtime.inbox.push_back(item)


def _kb_submit(
    agent: Agent,
    queued_input: list[str],
    nav: NavState,
    event: KeyPressEvent,
) -> None:
    r"""Enter handler. See :mod:`repl.input_pane` for the full contract.

    Branches:

    - Slash command: route through pump via ``validate_and_handle``.
    - Buffer ends with ``\``: backslash continuation. Replace the
      trailing ``\`` with a literal newline (``\n``) and stay in the
      buffer; do not dispatch.
    - Empty / whitespace-only buffer: no-op (whitespace also resets
      the buffer to clear stale spaces).
    - ``cursor > 0`` (navigation active): commit buffer as a queued
      block, restore ``snapshot_input`` to buffer.
    - ``cursor == 0``: preempt-dispatch as a ``UserMessage``.
    """
    buf = event.current_buffer
    text = buf.text
    stripped = text.strip()
    if stripped.startswith("/"):
        buf.validate_and_handle()
        return
    if text.endswith("\\"):
        # Backslash continuation: swap trailing ``\`` for ``\n``.
        buf.text = text[:-1] + "\n"
        buf.cursor_position = len(buf.text)
        return
    if not text:
        return
    if not stripped:
        buf.reset()
        return
    if nav.cursor > 0:
        restored = _commit_queued_and_restore(queued_input, nav, text)
        _flush_deferred_if_ready(agent, queued_input)
        buf.text = restored
        buf.cursor_position = len(buf.text)
        return
    buf.append_to_history()
    buf.reset()
    agent.runtime.inbox.push_back(UserMessage(text=text))


def _kb_defer(
    agent: Agent,
    queued_input: list[str],
    nav: NavState,
    event: KeyPressEvent,
) -> None:
    """Tab handler: stage buffer in ``queued_input`` for deferred dispatch.

    At ``cursor == 0`` (no navigation): the text is appended to
    ``queued_input`` and the buffer is cleared. No runtime push.
    ``make_queued_input_committer`` in :mod:`repl.run_repl` commits the
    queue as a single ``UserQueuedMessage`` on ``ModelIdle``.

    At ``cursor > 0`` (navigation active): same commit path as Enter --
    queue gets ``snapshot_queue + [buffer]`` and the buffer is restored
    to ``snapshot_input`` so the user's pre-navigation typing isn't
    lost.
    """
    buf = event.current_buffer
    text = buf.text
    if not text.strip():
        return
    if nav.cursor > 0:
        restored = _commit_queued_and_restore(queued_input, nav, text)
        _flush_deferred_if_ready(agent, queued_input)
        buf.text = restored
        buf.cursor_position = len(buf.text)
        return
    queued_input.append(text)
    _flush_deferred_if_ready(agent, queued_input)
    buf.append_to_history()
    buf.reset()


def _kb_down(
    queued_input: list[str],
    nav: NavState,
    event: KeyPressEvent,
) -> None:
    """Down handler. See :mod:`repl.input_pane` for the full contract.

    - ``cursor == 0`` (no navigation): preserve historical behavior --
      clear the buffer if non-empty, no-op if empty.
    - ``cursor == 1``: final Down -- restore ``snapshot_queue`` and
      ``snapshot_input`` atomically (queue reappears in queue_pane,
      buffer holds original typing).
    - ``cursor >= 2``: walk back one step toward the snapshot
      (newer history; eventually back to the queue-dequeued
      position in case 1).
    """
    buf = event.current_buffer
    if nav.cursor == 0:
        if buf.text:
            buf.reset()
        return
    case_1 = bool(nav.snapshot_queue)
    if nav.cursor == 1:
        # Final Down: restore snapshot atomically.
        snapshot_input = nav.snapshot_input
        queued_input.clear()
        queued_input.extend(nav.snapshot_queue)
        buf.text = snapshot_input
        buf.cursor_position = len(buf.text)
        nav.end()
        return
    nav.cursor -= 1
    history_strings = _history_strings(buf)
    if case_1 and nav.cursor == 1:
        # Back to "queue dequeued" position.
        buf.text = "\n\n".join(nav.snapshot_queue)
    else:
        offset = nav.cursor - 1 if case_1 else nav.cursor
        buf.text = (
            history_strings[-offset] if 1 <= offset <= len(history_strings) else ""
        )
    buf.cursor_position = len(buf.text)


def _kb_newline(event: KeyPressEvent) -> None:
    """Insert a literal newline into the current buffer (Alt+Enter)."""
    event.current_buffer.insert_text("\n")


def _history_strings(buf: object) -> list[str]:
    """Return the buffer's history entries as a list, oldest-first.

    Accessed via duck-typing so tests can supply ``MagicMock`` buffers
    without wiring a real ``History`` instance.
    """
    history = getattr(buf, "history", None)
    if history is None:
        return []
    get_strings = getattr(history, "get_strings", None)
    if get_strings is None:
        return []
    strings = get_strings()
    return list(strings) if strings is not None else []


def _kb_up(
    queued_input: list[str],
    nav: NavState,
    event: KeyPressEvent,
) -> None:
    """Up handler. See :mod:`repl.input_pane` for the full contract.

    First Up: capture ``(queued_input_copy, buffer_text)`` snapshot
    and either (case 1) lift the queue into the buffer, or (case 2)
    pull ``history[-1]`` into the buffer. Subsequent Ups walk older
    history. No-op when there's neither a queue nor history available.
    """
    buf = event.current_buffer
    history_strings = _history_strings(buf)
    if nav.cursor == 0:
        if queued_input:
            # Case 1: lift queue.
            nav.begin(queued_input, buf.text)
            buf.text = "\n\n".join(queued_input)
            queued_input.clear()
        elif history_strings:
            # Case 2: walk history[-1] in.
            nav.begin(queued_input, buf.text)
            buf.text = history_strings[-1]
        else:
            # No queue, no history -- nothing to do.
            return
        buf.cursor_position = len(buf.text)
        return
    # Continuing navigation: walk older history.
    case_1 = bool(nav.snapshot_queue)
    # cursor=1 case_1 = queue content; cursor=2 = history[-1]; ...
    # cursor=1 case_2 = history[-1]; cursor=2 = history[-2]; ...
    next_offset = nav.cursor if case_1 else nav.cursor + 1
    if next_offset > len(history_strings):
        return  # No more history.
    buf.text = history_strings[-next_offset]
    buf.cursor_position = len(buf.text)
    nav.cursor += 1


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
    """Open the current buffer in ``$EDITOR`` (Ctrl+X Ctrl+E)."""
    event.current_buffer.open_in_editor()


def _kb_ctrl_c(agent: Agent, event: KeyPressEvent) -> None:
    """Halt the active turn; never exit the REPL.

    Idle path clears the input buffer (standard terminal convention --
    abandon the line you were composing). To exit the REPL use Ctrl+D
    or ``/quit``.
    """
    if agent.work is not None or agent.runtime.cohort:
        agent.halt()
        return
    event.current_buffer.reset()


def _kb_suspend(event: KeyPressEvent) -> None:
    """Suspend the REPL to background (Ctrl+Z)."""
    event.app.suspend_to_background()


def _kb_undo(event: KeyPressEvent) -> None:
    """Undo the last buffer edit (Ctrl+_ / Esc-z)."""
    event.current_buffer.undo()
