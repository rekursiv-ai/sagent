"""v3 prompt-toolkit keybindings.

Translates key events into agent method calls (sync) or buffered text
that the input pump consumes on the next ``next_line()`` call.

Bindings:

- Enter: queue inline input while active; submit when idle.
- Tab: queue an end-of-turn follow-up while active; submit when idle.
- Alt+Enter: insert a newline (multi-line composition).
- Shift+Left: lift the queued user message back into the buffer.
- Up: history-backward.
- Shift-Up / Shift-Down: prefix-based history search.
- Ctrl+X Ctrl+E: open buffer in ``$EDITOR``.
- Ctrl+C: ``agent.abort()`` while active; exit when idle.
- Ctrl+_ / Esc-z: undo.
- Ctrl+Z: suspend.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import functools

from prompt_toolkit.filters import is_done
from prompt_toolkit.key_binding import KeyBindings

from sagent.custom_types import TextDescriptor, TextMessage
from sagent.lib.descriptors import QUEUED_USER_MESSAGE


if TYPE_CHECKING:
    from prompt_toolkit.key_binding import KeyPressEvent

    from sagent.agent.agent import Agent


def build_key_bindings(agent: Agent) -> KeyBindings:
    """Build the v3 REPL keybindings bound to ``agent``.

    Args:
      agent: Agent these key handlers will mutate.

    Returns:
      kb: Configured ``KeyBindings``.

    """
    kb = KeyBindings()
    kb.add("enter", filter=~is_done)(functools.partial(_kb_submit, agent))
    kb.add("tab", filter=~is_done)(functools.partial(_kb_queue, agent))
    kb.add("escape", "enter")(_kb_newline)
    kb.add("s-left")(functools.partial(_kb_edit_latest_queued, agent))
    kb.add("up")(_kb_history_back)
    kb.add("s-up")(_kb_history_prefix_back)
    kb.add("s-down")(_kb_history_prefix_fwd)
    kb.add("c-x", "c-e")(_kb_open_editor)
    kb.add("c-c")(functools.partial(_kb_ctrl_c, agent))
    kb.add("c-z")(_kb_suspend)
    kb.add("c-_")(_kb_undo)
    kb.add("escape", "z")(_kb_undo)
    return kb


def _kb_submit(agent: Agent, event: KeyPressEvent) -> None:
    """Submit when idle; queue inline input while active."""
    if agent.work is None:
        event.current_buffer.validate_and_handle()
        return
    _queue_buffer(agent, event, descriptor="text/x-user-message")


def _kb_queue(agent: Agent, event: KeyPressEvent) -> None:
    """Queue an end-of-turn follow-up while active; submit when idle."""
    if agent.work is None:
        event.current_buffer.validate_and_handle()
        return
    _queue_buffer(agent, event, descriptor=QUEUED_USER_MESSAGE)


def _queue_buffer(
    agent: Agent, event: KeyPressEvent, *, descriptor: TextDescriptor
) -> None:
    """Move non-empty active prompt text into the inbox."""
    buf = event.current_buffer
    text = buf.text
    if not text.strip():
        return
    _ = agent.inbox.put(TextMessage(text, descriptor))
    buf.append_to_history()
    buf.reset()


def _kb_newline(event: KeyPressEvent) -> None:
    event.current_buffer.insert_text("\n")


def _kb_edit_latest_queued(agent: Agent, event: KeyPressEvent) -> None:
    """Pop the newest queued user message into an empty buffer."""
    buf = event.current_buffer
    if buf.text.strip():
        return
    tail = agent.pop_latest_queued_user_message()
    if tail is None:
        return
    buf.text = str(tail.content)
    buf.cursor_position = len(buf.text)


def _kb_history_back(event: KeyPressEvent) -> None:
    """Navigate prompt history backward."""
    event.current_buffer.history_backward(count=1)


def _kb_history_prefix_back(event: KeyPressEvent) -> None:
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
    event.current_buffer.open_in_editor()


def _kb_ctrl_c(agent: Agent, event: KeyPressEvent) -> None:
    """Active: ``agent.abort()``. Idle: exit."""
    if agent.work is not None:
        agent.abort()
        return
    event.app.exit(exception=KeyboardInterrupt())


def _kb_suspend(event: KeyPressEvent) -> None:
    event.app.suspend_to_background()


def _kb_undo(event: KeyPressEvent) -> None:
    event.current_buffer.undo()
