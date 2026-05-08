"""v2 prompt-toolkit keybindings.

Adapted from v1's ``repl/keybindings.py`` with the v1 Agent surface
swapped for v2's deque + handler model:

- ``agent.active`` -> ``bool(agent.tasks)`` (any spawned task in flight).
- ``agent.tool_state.abort_event.set() + agent.inflight.cancel()`` ->
  put_left ``text/x-abort`` (handled by :class:`AbortHandler`).

Bindings:

- Enter: when active, queue the text into ``agent.inbox`` as a user
  message; when idle, submit (returns from ``prompt_async``).
- Alt+Enter: insert a newline (multi-line composition).
- Up: when buffer empty and inbox non-empty, lift the tail entry back
  into the buffer; otherwise history-backward.
- Shift-Up / Shift-Down: prefix-based history search.
- Ctrl+X Ctrl+E: open buffer in ``$EDITOR``.
- Ctrl+C: post abort while active; exit when idle.
- Ctrl+_ / Esc-z: undo.
- Ctrl+Z: suspend.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import functools

from prompt_toolkit.filters import is_done
from prompt_toolkit.key_binding import KeyBindings

from sagent.custom_types import TextMessage


if TYPE_CHECKING:
    from prompt_toolkit.key_binding import KeyPressEvent

    from sagent.agent.agent import Agent


_QUIT_WORDS = ("/quit",)


def build_key_bindings(agent: Agent) -> KeyBindings:
    """Build the v2 REPL keybindings bound to ``agent``."""
    kb = KeyBindings()
    kb.add("enter", filter=~is_done)(functools.partial(_kb_submit, agent))
    kb.add("escape", "enter")(_kb_newline)
    kb.add("up")(functools.partial(_kb_up, agent))
    kb.add("s-up")(_kb_history_prefix_back)
    kb.add("s-down")(_kb_history_prefix_fwd)
    kb.add("c-x", "c-e")(_kb_open_editor)
    kb.add("c-c")(functools.partial(_kb_ctrl_c, agent))
    kb.add("c-z")(_kb_suspend)
    kb.add("c-_")(_kb_undo)
    kb.add("escape", "z")(_kb_undo)
    return kb


def _kb_submit(agent: Agent, event: KeyPressEvent) -> None:
    """Queue input into ``agent.inbox`` while active, otherwise submit."""
    buf = event.current_buffer
    if bool(agent.tasks):
        text = buf.text.strip()
        if text.lower() in _QUIT_WORDS:
            buf.validate_and_handle()
            return
        if text == "/clear" or text.startswith("/clear "):
            # Urgent: preempt anything queued / in-flight (parity w/ v1).
            agent.inbox.put_left(
                TextMessage(text[6:].strip(), "text/x-clear-request"),
            )
            buf.append_to_history()
            buf.reset()
            return
        if text == "/compact" or text.startswith("/compact "):
            agent.inbox.put(
                TextMessage(text[8:].strip(), "text/x-compact-request"),
            )
            buf.append_to_history()
            buf.reset()
            return
        if text == "/uncompact" or text.startswith("/uncompact "):
            agent.inbox.put(
                TextMessage(text[10:].strip(), "text/x-uncompact-request"),
            )
            buf.append_to_history()
            buf.reset()
            return
        if text == "/model" or text.startswith("/model "):
            agent.inbox.put(
                TextMessage(text[6:].strip(), "text/x-model-switch-request"),
            )
            buf.append_to_history()
            buf.reset()
            return
        if text == "/provider" or text.startswith("/provider "):
            rest = text[9:].strip()
            args_str = f"--provider {rest}" if rest else ""
            agent.inbox.put(
                TextMessage(args_str, "text/x-model-switch-request"),
            )
            buf.append_to_history()
            buf.reset()
            return
        if text.startswith("/abort"):
            agent.inbox.put_left(TextMessage("", "text/x-abort"))
            buf.append_to_history()
            buf.reset()
            return
        if text.startswith("/"):
            cmd = text.split(maxsplit=1)[0]
            agent.inbox.put(
                TextMessage(
                    f"unknown command: {cmd}. Supported: "
                    "/clear /compact /uncompact /model /provider "
                    "/abort /quit",
                    "text/x-error",
                ),
            )
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


def _kb_up(agent: Agent, event: KeyPressEvent) -> None:
    """Pop inbox tail into the buffer, or navigate history."""
    buf = event.current_buffer
    if len(agent.inbox) > 0 and not buf.text.strip():
        tail = agent.inbox.pop_tail()
        if tail is not None:
            content = str(tail.content)
            if tail.descriptor == "text/x-clear-request":
                buf.text = f"/clear {content}".rstrip()
            elif tail.descriptor == "text/x-compact-request":
                buf.text = f"/compact {content}".rstrip()
            elif tail.descriptor == "text/x-uncompact-request":
                buf.text = f"/uncompact {content}".rstrip()
            elif tail.descriptor == "text/x-model-switch-request":
                buf.text = f"/model {content}".rstrip()
            else:
                buf.text = content
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
    """Post abort while active; exit when idle."""
    if bool(agent.tasks):
        agent.inbox.put_left(TextMessage("", "text/x-abort"))
        return
    event.app.exit(exception=KeyboardInterrupt())


def _kb_suspend(event: KeyPressEvent) -> None:
    event.app.suspend_to_background()


def _kb_undo(event: KeyPressEvent) -> None:
    event.current_buffer.undo()
