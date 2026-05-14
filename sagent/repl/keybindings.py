r"""prompt-toolkit keybindings.

Wires keys to the input-pane behavior contract. The contract itself
(what Up / Down / Enter / Tab do across queue and history states) is
documented in :mod:`repl.input_pane`'s module docstring -- the
behavior is a property of the input zone, not of the binding.

This module is the wiring:

- ``enter``        -> :func:`_kb_submit` (preempting dispatch)
- ``tab``          -> :func:`_kb_defer` (non-preempting dispatch)
- ``down``         -> :func:`_kb_down`
- ``up``           -> :func:`_kb_up`
- ``escape enter`` -> literal newline (Alt+Enter)
- ``s-up`` / ``s-down`` -> prefix history search
- ``c-x c-e``      -> open buffer in ``$EDITOR``
- ``c-c``          -> ``agent.halt()`` when active; clear buffer when idle
- ``c-z``          -> suspend to background
- ``c-_`` / ``escape z`` -> undo

Backslash continuation: ``foo\`` + Enter inserts a literal newline
(no dispatch); see ``_kb_submit``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import functools

from prompt_toolkit.filters import is_done
from prompt_toolkit.key_binding import KeyBindings

from sagent.agent.runtime import UserMessage


if TYPE_CHECKING:
    from prompt_toolkit.key_binding import KeyPressEvent

    from sagent.agent.agent import Agent


def build_key_bindings(agent: Agent, queued_input: list[str]) -> KeyBindings:
    """Build the REPL keybindings bound to ``agent`` and ``queued_input``.

    Args:
      agent: Agent these key handlers will mutate.
      queued_input: REPL-local staging buffer for queued blocks.

    Returns:
      kb: Configured ``KeyBindings``.

    """
    kb = KeyBindings()
    kb.add("enter", filter=~is_done)(functools.partial(_kb_submit, agent, queued_input))
    kb.add("tab")(functools.partial(_kb_defer, agent, queued_input))
    kb.add("down")(_kb_down)
    kb.add("escape", "enter")(_kb_newline)
    kb.add("up")(functools.partial(_kb_up, queued_input))
    kb.add("s-up")(_kb_history_prefix_back)
    kb.add("s-down")(_kb_history_prefix_fwd)
    kb.add("c-x", "c-e")(_kb_open_editor)
    kb.add("c-c")(functools.partial(_kb_ctrl_c, agent))
    kb.add("c-z")(_kb_suspend)
    kb.add("c-_")(_kb_undo)
    kb.add("escape", "z")(_kb_undo)
    return kb


def _kb_submit(
    agent: Agent,
    queued_input: list[str],
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
    - Otherwise: dispatch as a preempting ``UserMessage``.

    ``queued_input`` is *not* touched here. It belongs exclusively to
    Tab-staging (see :func:`_kb_defer`); Enter is a direct dispatch.
    """
    del queued_input  # Enter does not touch the Tab-staging queue.
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
    buf.append_to_history()
    buf.reset()
    agent.runtime.inbox.push_back(UserMessage(text=text))


def _kb_defer(
    agent: Agent,
    queued_input: list[str],
    event: KeyPressEvent,
) -> None:
    """Tab handler: stage buffer in ``queued_input`` for deferred dispatch.

    Tab is pure REPL-side staging: the text is appended to
    ``queued_input`` and the buffer is cleared. **Nothing is pushed to
    the runtime.** A ``ModelIdle`` observer (``make_queued_input_committer``
    in ``run_repl``) commits the joined queue as a single
    ``UserQueuedMessage`` when the agent's current round chain settles.

    Up-arrow lifts the queue back to the buffer for editing -- a true
    retract because nothing was ever in the runtime to begin with.
    """
    del agent  # No runtime push -- staging is REPL-local until ModelIdle.
    buf = event.current_buffer
    text = buf.text
    if not text.strip():
        return
    queued_input.append(text)
    buf.append_to_history()
    buf.reset()


def _kb_down(event: KeyPressEvent) -> None:
    """Clear input on non-empty buffer; no-op on empty.

    Per :mod:`repl.input_pane`'s contract: Down on a non-empty input
    pane discards the buffer contents (returns to empty input). Down
    on an empty buffer is reserved for a future submenu (spawned
    agents / tasks) and does nothing today.

    Down never dispatches or stages -- only Enter does.
    """
    buf = event.current_buffer
    if buf.text:
        buf.reset()


def _kb_newline(event: KeyPressEvent) -> None:
    """Insert a literal newline into the current buffer (Alt+Enter)."""
    event.current_buffer.insert_text("\n")


def _kb_up(queued_input: list[str], event: KeyPressEvent) -> None:
    r"""Lift the entire staged queue into the buffer; else PT history-back.

    The queue is treated as one draft: a single up-arrow joins all
    blocks with ``\\n\\n`` and moves them into ``input_pane`` for
    editing. Queue is emptied. A second up-arrow falls through to
    history-backward (which doesn't include staged blocks -- they
    were never dispatched).
    """
    if queued_input:
        text = "\n\n".join(queued_input)
        queued_input.clear()
        buf = event.current_buffer
        buf.text = text
        buf.cursor_position = len(buf.text)
        return
    event.current_buffer.history_backward(count=1)


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
