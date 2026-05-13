"""prompt-toolkit keybindings.

Translates key events into agent method calls (sync) or queued text
that the input pump consumes on the next ``next_line()`` call.

Bindings:

- Enter: queue input into the REPL-local pending list while active;
  submit when idle.
- Alt+Enter: insert a newline (multi-line composition).
- Up: pull most recent pending user message into the buffer for
  editing; else history-backward.
- Shift-Up / Shift-Down: prefix-based history search.
- Ctrl+X Ctrl+E: open buffer in ``$EDITOR``.
- Ctrl+C: ``agent.halt()`` while active; clear input buffer when
  idle. Never exits.
- Ctrl+D / ``/quit``: exit the REPL.
- Ctrl+_ / Esc-z: undo.
- Ctrl+Z: suspend.

The pending list is the REPL's local view of user texts not yet
drained by the runtime. ``GatedDeque`` doesn't support tag-based
peek / pop, so the REPL maintains this list itself and the
keybindings update it.
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


def build_key_bindings(agent: Agent, pending: list[str]) -> KeyBindings:
    """Build the REPL keybindings bound to ``agent`` and ``pending``.

    Args:
      agent: Agent these key handlers will mutate.
      pending: REPL-local list of texts typed while the agent was
          busy. Submitted-when-active inputs append here; Up pops
          the tail.

    Returns:
      kb: Configured ``KeyBindings``.

    """
    kb = KeyBindings()
    kb.add("enter", filter=~is_done)(functools.partial(_kb_submit, agent, pending))
    kb.add("escape", "enter")(_kb_newline)
    kb.add("up")(functools.partial(_kb_up, pending))
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
    pending: list[str],
    event: KeyPressEvent,
) -> None:
    """Submit when idle; queue input into the pending list while active.

    Slash commands (``/...``) always route through the pump regardless
    of busy state so e.g. ``/model`` prints its read-mode response
    immediately instead of being pushed as a ``UserMessage`` the model
    would treat as a fresh prompt. The pump's dispatcher decides
    whether to print directly (reads) or queue an event (writes).

    Plain text submissions push a ``UserMessage`` to the runtime inbox
    when busy (the runtime preempts and stubs unfinished tools) and
    record the text in ``pending`` so the dim preview and Up-arrow
    edit-back keep working.
    """
    buf = event.current_buffer
    text = buf.text
    stripped = text.strip()
    if stripped.startswith("/"):
        buf.validate_and_handle()
        return
    if agent.work is None and not agent.runtime.cohort:
        buf.validate_and_handle()
        return
    if not stripped:
        return
    agent.runtime.inbox.push_back(UserMessage(text=text))
    pending.append(text)
    buf.append_to_history()
    buf.reset()


def _kb_newline(event: KeyPressEvent) -> None:
    """Insert a literal newline into the current buffer (Alt+Enter)."""
    event.current_buffer.insert_text("\n")


def _kb_up(pending: list[str], event: KeyPressEvent) -> None:
    """Lift the most recent pending text into the buffer; else history-back."""
    if pending:
        text = pending.pop()
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
