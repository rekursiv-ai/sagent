r"""REPL queue and deferred panes: at most one message each.

The spec (``docs/private/input_ux.md``) mandates that the queue pane and
the deferred pane each hold AT MOST ONE message. Staging into a
populated pane coalesces by append: the new text joins the existing
with ``\n\n``, attachments concatenate, FIFO preserved. So each pane is
modelled as a single optional :class:`QueuedInputBlock`, never a list.

Dispatch-vs-stage is decided by the caller (a pure function of agent
busy state); this module only owns the staged content and how it
renders.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from sagent.types.runtime import (
    BytesMessage,
    UserDeferredMessage,
    UserMessage,
)


if TYPE_CHECKING:
    from sagent.agent.agent import Agent


@dataclass(frozen=True, slots=True, kw_only=True)
class QueuedInputBlock:
    """One staged user-input message (the entire content of one pane)."""

    text: str
    attachments: tuple[BytesMessage, ...] = ()


@dataclass(slots=True, kw_only=True)
class InputQueues:
    """The queue pane and deferred pane, each at most one message.

    ``queue`` is Enter-while-busy content that dispatches at the next
    chat-safe boundary. ``deferred`` is Tab-while-busy content that
    waits for the round chain to go idle. Each is a single coalesced
    message or ``None`` when its pane is empty.
    """

    queue: QueuedInputBlock | None = None
    deferred: QueuedInputBlock | None = None

    def has_any(self) -> bool:
        """Return whether either pane holds a message."""
        return self.queue is not None or self.deferred is not None

    def stage_queue(
        self, text: str, attachments: tuple[BytesMessage, ...] = ()
    ) -> None:
        """Coalesce ``text`` into the queue pane (append after existing)."""
        self.queue = _coalesce(self.queue, text, attachments)

    def stage_deferred(
        self, text: str, attachments: tuple[BytesMessage, ...] = ()
    ) -> None:
        """Coalesce ``text`` into the deferred pane (append after existing)."""
        self.deferred = _coalesce(self.deferred, text, attachments)

    def clear(self) -> None:
        """Empty both panes."""
        self.queue = None
        self.deferred = None

    def render_lines(self) -> list[str]:
        """Return pane lines top-to-bottom: deferred above queue.

        The deferred message carries a ``[deferred]`` prefix; the queue
        message carries none. Empty panes contribute nothing. Order
        matches the spec's vertical layout (deferred pane above queue
        pane, both above the input pane).
        """
        lines: list[str] = []
        if self.deferred is not None:
            lines.append(f"[deferred] {self.deferred.text}")
        if self.queue is not None:
            lines.append(self.queue.text)
        return lines

    def pop_queue_message(self) -> UserMessage | None:
        """Remove the queue pane's message and return it as a ``UserMessage``."""
        if self.queue is None:
            return None
        message = UserMessage(text=self.queue.text, attachments=self.queue.attachments)
        self.queue = None
        return message

    def commit_queue(self, agent: Agent) -> bool:
        """Dispatch the queue pane's message to the inbox, if present."""
        message = self.pop_queue_message()
        if message is None:
            return False
        agent.runtime.inbox.push_back(message)
        return True

    def commit_deferred_on_idle(self, agent: Agent) -> bool:
        """Dispatch the deferred pane's message to the inbox, if present."""
        if self.deferred is None:
            return False
        agent.runtime.inbox.push_back(
            UserDeferredMessage(
                text=self.deferred.text,
                attachments=self.deferred.attachments,
            )
        )
        self.deferred = None
        return True

    def peek_tail_preview(self) -> str:
        """Return the most-committed pane's text for discard messages.

        Read-only. The queue pane outranks the deferred pane
        (Enter-staged is nearer dispatch than Tab-staged).
        """
        if self.queue is not None:
            return self.queue.text
        if self.deferred is not None:
            return self.deferred.text
        return ""


def _coalesce(
    existing: QueuedInputBlock | None,
    text: str,
    attachments: tuple[BytesMessage, ...],
) -> QueuedInputBlock:
    r"""Append ``text`` after ``existing``; join with ``\n\n``."""
    if existing is None:
        return QueuedInputBlock(text=text, attachments=attachments)
    return QueuedInputBlock(
        text=f"{existing.text}\n\n{text}",
        attachments=existing.attachments + attachments,
    )
