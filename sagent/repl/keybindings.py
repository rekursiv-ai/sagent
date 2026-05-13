r"""prompt-toolkit keybindings.

Staging model:

- ``queued_input`` is a REPL-local list of staged blocks. Each entry
  becomes one paragraph of the eventual ``UserQueuedMessage`` payload
  (blocks join with ``\\n\\n``). Nothing is dispatched to the runtime
  until commit -- so up-arrow can truly retract.
- Enter on non-empty ``input_pane``: append text to ``queued_input``,
  clear ``input_pane``.
- Down on non-empty ``input_pane``: same as Enter (staging shortcut).
- Enter on empty ``input_pane`` with non-empty queue: COMMIT -- push
  ``UserQueuedMessage`` with joined blocks, clear queue.
- Down on empty ``input_pane``: reserved (no-op for now; future
  submenu navigation).
- Up: if queue non-empty, lift the *entire* joined queue into
  ``input_pane`` for editing, clear queue. Else PT history-backward.
  Second up-arrow walks PT history (which doesn't include staged
  blocks since they were never dispatched).
- Slash commands always route through the pump via
  ``buf.validate_and_handle()``.
- Alt+Enter: literal newline (multi-line composition within a block).
- Shift-Up / Shift-Down: prefix-based history search.
- Ctrl+X Ctrl+E: open buffer in ``$EDITOR``.
- Ctrl+C: ``agent.halt()`` while active; clear input buffer when
  idle. Never exits.
- Ctrl+D / ``/quit``: exit the REPL.
- Ctrl+_ / Esc-z: undo.
- Ctrl+Z: suspend.

Auto-commit (case 2 of the staging model -- user submitted while the
agent was busy) is handled by ``make_queued_input_committer`` on the
runtime's ``ModelIdle`` event in ``run_repl``; the keybindings only
handle the manual paths above.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import asyncio
import contextlib
import functools

from prompt_toolkit.filters import is_done
from prompt_toolkit.key_binding import KeyBindings

from sagent.agent.runtime import UserQueuedMessage


if TYPE_CHECKING:
    from prompt_toolkit.key_binding import KeyPressEvent

    from sagent.agent.agent import Agent

# Idle-commit delay: after Enter on text while the agent is idle, the
# REPL waits this long before pushing the staged ``UserQueuedMessage``
# to the runtime. The window lets the user up-arrow to retract if they
# realize the message is wrong; otherwise commit fires automatically.
_IDLE_COMMIT_DELAY_SEC = 0.25


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
    kb.add("down")(functools.partial(_kb_down, agent, queued_input))
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
    r"""Stage text on Enter; commit queue on truly-empty-buffer Enter.

    Behavior matrix (the buffer text is ``text``; ``stripped`` is its
    ``.strip()``):

    - ``stripped`` starts with ``/`` (slash command): route through
      pump via ``validate_and_handle`` (pump dispatches read-mode).
    - ``text == ""`` (truly empty buffer) and queue is non-empty:
      COMMIT -- push ``UserQueuedMessage`` joining blocks on
      ``\n\n``, clear queue.
    - ``text == ""`` and queue is empty: no-op.
    - ``stripped == ""`` but ``text != ""`` (whitespace-only): discard
      the buffer contents (no commit, no stage). Avoids stale spaces.
    - Otherwise (non-empty text): append to queue, reset buffer.
    """
    buf = event.current_buffer
    text = buf.text
    stripped = text.strip()
    if stripped.startswith("/"):
        buf.validate_and_handle()
        return
    if not text:
        if queued_input:
            joined = "\n\n".join(queued_input)
            queued_input.clear()
            agent.runtime.inbox.push_back(UserQueuedMessage(text=joined))
        return
    if not stripped:
        buf.reset()
        return
    queued_input.append(text)
    buf.append_to_history()
    buf.reset()
    _schedule_idle_commit(agent, queued_input)


def _schedule_idle_commit(agent: Agent, queued_input: list[str]) -> None:
    """If the agent is idle, schedule a delayed commit of the queue.

    The delay (``_IDLE_COMMIT_DELAY_SEC``) gives the user a window to
    up-arrow and retract before the queue is dispatched. When the
    agent is busy, the staging-model's auto-committer observer
    (``make_queued_input_committer``) handles commit on ``ModelIdle``;
    we don't schedule here in that case.
    """
    if agent.work is not None or agent.runtime.cohort:
        return
    with contextlib.suppress(RuntimeError):
        # ``get_running_loop`` raises off-loop; tests that exercise
        # ``_kb_submit`` synchronously skip the scheduling path.
        loop = asyncio.get_running_loop()
        loop.call_later(
            _IDLE_COMMIT_DELAY_SEC,
            functools.partial(_commit_if_still_idle, agent, queued_input),
        )


def _commit_if_still_idle(agent: Agent, queued_input: list[str]) -> None:
    """Commit the staged queue iff the agent is still idle and the
    queue still has content (user did not retract during the window).
    """
    if not queued_input:
        return
    if agent.work is not None or agent.runtime.cohort:
        return
    joined = "\n\n".join(queued_input)
    queued_input.clear()
    agent.runtime.inbox.push_back(UserQueuedMessage(text=joined))


def _kb_down(
    agent: Agent,
    queued_input: list[str],
    event: KeyPressEvent,
) -> None:
    """Stage text on Down (same as Enter for non-empty); else no-op.

    Mirrors Enter's stage behavior for non-empty buffers so the user
    can pick whichever feels natural after editing a lifted queue:
    Down stages and schedules the idle commit just like Enter. Down
    on an empty buffer is reserved for a future submenu (spawned
    agents / tasks) and does nothing today.
    """
    buf = event.current_buffer
    text = buf.text
    if not text.strip():
        return
    queued_input.append(text)
    buf.append_to_history()
    buf.reset()
    _schedule_idle_commit(agent, queued_input)


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
