"""Key bindings for the REPL prompt."""

from __future__ import annotations

from typing import TYPE_CHECKING

import functools

from prompt_toolkit.filters import is_done
from prompt_toolkit.key_binding import KeyBindings

from sagent.custom_types import TextMessage


if TYPE_CHECKING:
    from prompt_toolkit.key_binding import KeyPressEvent

    from sagent.agent import Agent


def build_key_bindings(agent: Agent | None = None) -> KeyBindings:
    r"""Custom key bindings for the REPL.

    - Enter submits when idle; queues into ``agent.inbox`` during a request.
    - Alt+Enter inserts a newline.
    - Up arrow: if inbox is non-empty and buffer is empty, lifts the
      tail entry back into the buffer. Otherwise history-backward.
    - Shift+Up/Down for prefix-based history search.
    - Ctrl+X Ctrl+E opens the buffer in ``$EDITOR`` - gated on idle.
    - Ctrl+C: cancels the active request, or exits when idle.

    Args:
      agent: Optional running agent for inbox integration.

    Returns:
      kb: ``KeyBindings`` instance ready to hand to ``PromptSession``.

    """
    kb = KeyBindings()
    kb.add("enter", filter=~is_done)(functools.partial(_kb_submit, agent))
    kb.add("escape", "enter")(_kb_newline)
    kb.add("up")(functools.partial(_kb_up, agent))
    kb.add("s-up")(_kb_history_prefix_back)
    kb.add("c-x", "c-e")(_kb_open_editor)
    kb.add("c-c")(functools.partial(_kb_ctrl_c, agent))
    kb.add("s-down")(_kb_history_prefix_fwd)
    kb.add("c-z")(_kb_suspend)
    kb.add("c-_")(_kb_undo)
    kb.add("escape", "z")(_kb_undo)
    return kb


def _kb_submit(agent: Agent | None, event: KeyPressEvent) -> None:
    """Queue input into agent inbox when active, otherwise submit."""
    buf = event.current_buffer
    if agent is not None and agent.active:
        text = buf.text.strip()
        if text.lower() in ("quit", "exit"):
            buf.validate_and_handle()
            return
        if text == "/clear" or text.startswith("/clear "):
            agent.inbox.put_left(TextMessage(text[6:].strip(), "text/x-clear-request"))
            buf.append_to_history()
            buf.reset()
            return
        if text:
            agent.inbox.put(TextMessage(text, "text/x-user-message"))
            buf.append_to_history()
            buf.reset()
        return
    buf.validate_and_handle()


def _kb_newline(event: KeyPressEvent) -> None:
    event.current_buffer.insert_text("\n")


def _kb_up(agent: Agent | None, event: KeyPressEvent) -> None:
    """Pop inbox tail into the buffer, or navigate history."""
    buf = event.current_buffer
    if agent is not None and agent.inbox and not buf.text.strip():
        tail = agent.inbox.pop_tail()
        if tail is not None:
            if tail.descriptor == "text/x-clear-request":
                reason = str(tail.content)
                buf.text = f"/clear {reason}" if reason else "/clear"
            else:
                buf.text = str(tail.content)
            buf.cursor_position = len(buf.text)
            return
    buf.history_backward(count=1)


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


def _kb_open_editor(event: KeyPressEvent) -> None:
    event.current_buffer.open_in_editor()


def _kb_ctrl_c(agent: Agent | None, event: KeyPressEvent) -> None:
    """Cancel active request or exit when idle."""
    if agent is not None and agent.active:
        agent.tool_state.abort_event.set()
        t = agent.inflight
        if t is not None and not t.done():
            t.cancel()
        return
    event.app.exit(exception=KeyboardInterrupt())


def _kb_suspend(event: KeyPressEvent) -> None:
    event.app.suspend_to_background()


def _kb_undo(event: KeyPressEvent) -> None:
    event.current_buffer.undo()


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
