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

from sagent.repl.input_queues import (
    InputQueues,
    Lane,
    QueuedInputBlock,
)
from sagent.types.runtime import UserMessage


if TYPE_CHECKING:
    from prompt_toolkit.buffer import Buffer
    from prompt_toolkit.key_binding import KeyPressEvent

    from sagent.agent.agent import Agent


@dataclass(slots=True, kw_only=True)
class NavState:
    """Up/Down navigation cursor + snapshot.

    ``cursor == 0`` means "not navigating": the buffer holds the user's
    typing and the REPL queue preview shows urgent/deferred blocks.
    ``cursor > 0`` means navigation is in progress; ``snapshot`` holds
    the queued blocks plus the original buffer text captured when Up was
    pressed. Final Down restores the snapshot; Enter / Tab at
    ``cursor > 0`` commits the buffer back into the queues and restores
    the original buffer text.
    """

    cursor: int = 0
    snapshot_queue: tuple[QueuedInputBlock, ...] = field(default_factory=tuple)
    snapshot_urgent_count: int = 0
    snapshot_input: str = ""

    def begin(
        self,
        queued: list[QueuedInputBlock],
        buffer_text: str,
        *,
        urgent_count: int = 0,
    ) -> None:
        """Capture a snapshot at the start of navigation."""
        self.snapshot_queue = tuple(queued)
        self.snapshot_urgent_count = urgent_count
        self.snapshot_input = buffer_text
        self.cursor = 1

    def end(self) -> None:
        """Clear the snapshot when navigation completes."""
        self.cursor = 0
        self.snapshot_queue = ()
        self.snapshot_urgent_count = 0
        self.snapshot_input = ""


def build_key_bindings(
    agent: Agent, queues: InputQueues, nav: NavState | None = None
) -> KeyBindings:
    """Build the REPL keybindings bound to ``agent``, ``queues``, and ``nav``.

    Args:
      agent: Agent these key handlers will mutate.
      queues: REPL-local urgent/deferred input queues.
      nav: Up/Down navigation state. Created fresh if omitted (for
          callers that don't share state across binding builds).

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


def _commit_queued_and_restore(
    queues: InputQueues, nav: NavState, buf_text: str, *, lane: Lane = "deferred"
) -> str:
    """Commit ``buf_text`` to queue + restore snapshot; return restored buffer.

    Shared helper for Enter and Tab at ``cursor > 0``.

    Cursor / edit_mode / lane truth table::

        cursor  snapshot_queue  derived edit_mode  case
        1       non-empty       True               edit-in-place (replace head)
        1       empty           False              extension (Case 2 fallback)
        >1      *               False              extension (scrolled past)

    Lane mapping::

        caller  lane         result
        Enter   "urgent"     committed block dispatches at next chat-safe
                             boundary
        Tab     "deferred"   navigation-from-tab still defers

    The snapshot is cleared and the snapshot-input is returned so the
    caller can restore the buffer to the user's pre-navigation typing.
    """
    queues.replace_from_navigation(
        nav.snapshot_queue,
        buf_text,
        edit_mode=nav.cursor == 1 and bool(nav.snapshot_queue),
        urgent_count=nav.snapshot_urgent_count,
        lane=lane,
    )
    restored = nav.snapshot_input
    nav.end()
    return restored


def _restore_navigation_snapshot(
    queues: InputQueues, nav: NavState, buf: Buffer
) -> None:
    """Restore queue + buffer from the nav snapshot; clear nav state.

    Shared by the whitespace-Enter / whitespace-Tab paths: the user
    cleared their edit while navigating, so treat it as a "cancel my
    edit" gesture -- the same effect as a final Down at ``cursor == 1``.
    Without this, the snapshot would strand: cursor stays at 1 forever
    and the lifted queue blocks disappear.
    """
    queues.restore_from_snapshot(
        nav.snapshot_queue, urgent_count=nav.snapshot_urgent_count
    )
    snapshot_input = nav.snapshot_input
    nav.end()
    buf.text = snapshot_input
    buf.cursor_position = len(snapshot_input)


def _kb_submit(
    agent: Agent,
    queues: InputQueues,
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
        # Whitespace-only Enter: during navigation, "discard my edit" --
        # restore the snapshot (queue + buffer text) so the lifted blocks
        # don't strand. Outside navigation, just clear the stale spaces.
        if nav.cursor > 0:
            _restore_navigation_snapshot(queues, nav, buf)
            return
        buf.reset()
        return
    if nav.cursor > 0:
        restored = _commit_queued_and_restore(queues, nav, text, lane="urgent")
        # No ``buf.append_to_history()`` / ``buf.reset()`` here -- the
        # nav-Enter contract restores the user's pre-navigation typing
        # to the buffer. The committed text was lifted from a queued
        # block (already in history's queue lane), not freshly typed;
        # appending it to prompt-toolkit history would double-record
        # navigation echoes. Resetting would discard the restored
        # snapshot.
        buf.text = restored
        buf.cursor_position = len(buf.text)
        # An already-idle runtime won't fire another ``AgentIdle`` to
        # drain the freshly-staged urgent block, so push it now.
        if agent.runtime.is_idle:
            queues.commit_urgent(agent)
        return
    if (
        agent.runtime.model_call is not None
        and not agent.runtime.cohort
        and not agent.runtime.inbox.gate_armed
    ):
        # prompt-toolkit does not surface pasted attachments today;
        # pass the empty tuple explicitly so a future input source that
        # threads them through cannot silently drop them here.
        queues.stage_urgent(text, attachments=())
    else:
        agent.runtime.inbox.push_back(UserMessage(text=text))
    buf.append_to_history()
    buf.reset()


def _kb_defer(
    agent: Agent,
    queues: InputQueues,
    nav: NavState,
    event: KeyPressEvent,
) -> None:
    """Tab handler: stage buffer for deferred dispatch.

    At ``cursor == 0`` (no navigation): the text is appended to the
    deferred queue and the buffer is cleared. ``install_input_queue_committer``
    in :mod:`repl.run_repl` commits deferred input as a single
    ``UserDeferredMessage`` on ``AgentIdle``; an already-armed user gate is
    released immediately because no future ``AgentIdle`` will arrive.

    At ``cursor > 0`` (navigation active): same commit path as Enter --
    queue gets ``snapshot_queue + [buffer]`` and the buffer is restored
    to ``snapshot_input`` so the user's pre-navigation typing isn't
    lost.
    """
    buf = event.current_buffer
    text = buf.text
    if not text.strip():
        # Whitespace-only Tab: during navigation, treat as "discard my edit"
        # and restore the snapshot. Outside navigation, no-op (Tab on an
        # empty buffer has never staged anything).
        if nav.cursor > 0:
            _restore_navigation_snapshot(queues, nav, buf)
        return
    if nav.cursor > 0:
        restored = _commit_queued_and_restore(queues, nav, text)
        buf.text = restored
        buf.cursor_position = len(buf.text)
        return
    queues.stage_deferred(text)
    # An already-idle runtime won't fire another ``AgentIdle`` to drain
    # the freshly-staged block, so push it now. An armed user gate
    # (post-Halt) also needs an immediate push because the gate will
    # release on the deferred message itself.
    if agent.runtime.inbox.gate_armed or agent.runtime.is_idle:
        queues.commit_deferred_on_idle(agent)
    buf.append_to_history()
    buf.reset()


def _kb_down(
    queues: InputQueues,
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
        queues.restore_from_snapshot(
            nav.snapshot_queue,
            urgent_count=nav.snapshot_urgent_count,
        )
        buf.text = snapshot_input
        buf.cursor_position = len(buf.text)
        nav.end()
        return
    nav.cursor -= 1
    history_strings = _history_strings(buf)
    if case_1 and nav.cursor == 1:
        # Back to "queue dequeued" position.
        buf.text = "\n\n".join(block.text for block in nav.snapshot_queue)
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
    queues: InputQueues,
    nav: NavState,
    event: KeyPressEvent,
) -> None:
    """Up handler. See :mod:`repl.input_pane` for the full contract.

    First Up: capture ``(queued_blocks, buffer_text)`` snapshot
    and either (case 1) lift the queue into the buffer, or (case 2)
    pull ``history[-1]`` into the buffer. Subsequent Ups walk older
    history. No-op when there's neither a queue nor history available.
    """
    buf = event.current_buffer
    history_strings = _history_strings(buf)
    if nav.cursor == 0:
        restore_blocks = queues.restore_blocks()
        if restore_blocks:
            # Case 1: lift queued input.
            nav.begin(
                list(queues.snapshot_blocks()),
                buf.text,
                urgent_count=len(queues.urgent),
            )
            buf.text = "\n\n".join(restore_blocks)
            queues.clear()
        elif history_strings:
            # Case 2: walk history[-1] in.
            nav.begin([], buf.text)
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
    """Open the current buffer in ``$EDITOR`` (Ctrl+X Ctrl+E).

    Blocks the prompt-toolkit event loop until the editor exits --
    prompt-toolkit's ``open_in_editor`` shells out synchronously. The
    REPL renderer and the underlying agent loop both pause for the
    editor session; this is the documented prompt-toolkit behavior, not
    a defect to work around here.
    """
    event.current_buffer.open_in_editor()


def _kb_ctrl_c(agent: Agent, event: KeyPressEvent) -> None:
    r"""Abandon the line being composed; never exit the REPL.

    One rule, matching every Unix line editor (bash, readline, ipython):
    Ctrl+C abandons the line you are composing and gives you a fresh
    prompt. The abandoned text is recorded in history (Up-arrow recalls
    it) but is never carried into the next turn -- typing a new line must
    not re-inject the old one.

    Two facts stay orthogonal to that rule:

    - When the agent is busy (``agent.work`` -- a model call or
      compaction -- or a non-empty ``runtime.cohort``), Ctrl+C also halts
      the running turn. That is the same "abandon the current activity"
      intent applied to the agent rather than the line.
    - Queued urgent/deferred blocks were *deliberately submitted* with
      Enter/Tab; they are not the line being composed, so Ctrl+C leaves
      them untouched -- this handler never reads or mutates the queues.

    To exit the REPL use Ctrl+D or ``/quit``.

    Args:
      agent: Agent whose running turn is halted when busy.
      event: Key event carrying the buffer to abandon.

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
