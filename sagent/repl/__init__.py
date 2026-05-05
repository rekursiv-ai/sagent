r"""Rich interactive REPL for the agent.

UX
~~

Layout
    A persistent ``> `` prompt sits at the bottom, with full
    prompt_toolkit editing (history, arrow keys, paste, Alt+Enter
    for newline). A bottom toolbar shows spinner + elapsed +
    tokens while a request is running, or the last request's
    cost/tokens summary when idle. Dim gray on terminal-native
    background - no bright inverse bar.

Submitting
    You type, hit Enter. The input clears from the live prompt
    and reappears in scrollback as a full-width dark-gray bar -
    the message highlighted edge-to-edge, creating a visual
    separator. The model's response streams below. On response
    end a cost line follows.

Typing during a request
    The prompt stays fully editable. Each Enter pushes the
    buffer into the inbox queue and resets the buffer so you
    can keep typing. A dim one-line preview of the most recently
    queued entry appears above the ``> `` line. When the running
    request finishes, all queued entries are squashed with ``\\n\\n``
    and sent as one message.

Editing what you queued
    Up arrow with an empty buffer lifts the *most recently
    queued* entry back into the buffer and removes it from the
    queue. Edit, hit Enter to re-queue. With non-empty buffer
    or empty queue, up arrow does normal history recall.

Request handoff
    When the running request finishes, anything queued ships
    automatically - no extra Enter. Any in-progress typing you
    hadn't Enter'd yet is combined with the queued text in that
    automatic send. The combined message appears as its own
    dark-gray bar and the next request starts immediately.

Interrupts
    Ctrl+C during a request cancels that request; the prompt stays
    alive. Ctrl+C at the idle prompt exits the REPL. ``quit`` or
    ``exit`` also exits cleanly (waiting for any in-flight request
    to finish first).

Edit-tool diffs
    When the agent calls the ``Edit`` tool, the REPL renders the
    old/new diff inline: tight gutter (``  617 - ``, 9-char wide),
    dim-gray line numbers visible but not glaring on the
    red/green bg, syntax colors preserved through word-level
    highlights.

Usage::

    from sagent.repl import run_repl

    await run_repl(agent, name="ant")
"""

from __future__ import annotations

from prompt_toolkit.input.ansi_escape_sequences import ANSI_SEQUENCES
from prompt_toolkit.keys import Keys

from sagent.repl.repl import run_repl


# Terminals with focus reporting send \x1b[I / \x1b[O on focus in/out.
# prompt_toolkit 3.0.52 doesn't recognise these; map them to Ignore so
# the Vt100 parser doesn't stall on the incomplete escape sequence.
ANSI_SEQUENCES.setdefault("\x1b[I", Keys.Ignore)
ANSI_SEQUENCES.setdefault("\x1b[O", Keys.Ignore)

__all__ = ["run_repl"]
