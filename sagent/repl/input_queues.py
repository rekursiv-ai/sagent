"""REPL-local urgent/deferred input queues."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from sagent.types.runtime import (
    BytesMessage,
    UserDeferredMessage,
    UserMessage,
)


if TYPE_CHECKING:
    from collections.abc import Sequence

    from sagent.agent.agent import Agent


type Lane = Literal["urgent", "deferred"]


@dataclass(frozen=True, slots=True, kw_only=True)
class QueuedInputBlock:
    """One editable queued user-input block."""

    text: str
    attachments: tuple[BytesMessage, ...] = ()


@dataclass(slots=True, kw_only=True)
class InputQueues:
    """User-editable REPL input queues.

    ``urgent`` is Enter-created input that should run at the next chat-safe
    boundary. ``deferred`` is Tab-created input that waits for ``AgentIdle``.
    """

    urgent: list[QueuedInputBlock] = field(default_factory=list)
    deferred: list[QueuedInputBlock] = field(default_factory=list)

    def has_any(self) -> bool:
        """Return whether either queue has staged text."""
        return bool(self.urgent or self.deferred)

    def stage_urgent(
        self, text: str, attachments: tuple[BytesMessage, ...] = ()
    ) -> None:
        """Append an urgent block."""
        self.urgent.append(QueuedInputBlock(text=text, attachments=attachments))

    def stage_deferred(
        self, text: str, attachments: tuple[BytesMessage, ...] = ()
    ) -> None:
        """Append a deferred block."""
        self.deferred.append(QueuedInputBlock(text=text, attachments=attachments))

    def clear(self) -> None:
        """Discard all staged text."""
        self.urgent.clear()
        self.deferred.clear()

    def render_blocks(self) -> list[str]:
        """Return labelled queue blocks for the input pane.

        Urgent blocks render bare: they dispatch at the next chat-safe
        boundary so the user does not need a label to know what will
        happen. Deferred blocks keep their ``deferred:`` prefix because
        they wait for an explicit cue and the label is the only
        in-pane signal that the text won't auto-send.
        """
        return [
            *(block.text for block in self.urgent),
            *(f"deferred: {block.text}" for block in self.deferred),
        ]

    def restore_blocks(self) -> list[str]:
        """Return editable block text in Up-arrow priority order."""
        return [
            *(block.text for block in self.urgent),
            *(block.text for block in self.deferred),
        ]

    def snapshot_blocks(self) -> tuple[QueuedInputBlock, ...]:
        """Return editable blocks in Up-arrow priority order."""
        return (*self.urgent, *self.deferred)

    def replace_from_navigation(
        self,
        nav_queue: Sequence[QueuedInputBlock],
        text: str,
        *,
        edit_mode: bool,
        urgent_count: int = 0,
        lane: Lane = "deferred",
    ) -> None:
        """Replace queue contents after Up/Down navigation commit.

        Edit mode replaces only the head block the user lifted (the
        first urgent block when urgent exists; otherwise the first
        deferred block). The remaining snapshot tail is restored
        verbatim so blocks the user did not touch survive the commit.

        Commit-lane truth table::

            edit_mode  urgent_count  lane       -> committed lane
            False      *             "urgent"      urgent
            False      *             "deferred"    deferred
            True       0             "urgent"      urgent
            True       0             "deferred"    deferred (tail)
            True       >0            *             urgent  (head was urgent;
                                                            stays urgent regardless of lane)

        ``lane`` is the caller's commit intent: Enter passes
        ``"urgent"`` so the committed block dispatches at the next
        chat-safe boundary; Tab keeps the default ``"deferred"`` so
        navigation-from-tab still defers.
        """
        self.clear()
        if edit_mode and nav_queue:
            attachments = nav_queue[0].attachments
            head_is_urgent = urgent_count > 0 or lane == "urgent"
            if head_is_urgent:
                self.stage_urgent(text, attachments)
                tail_urgent = nav_queue[1:urgent_count] if urgent_count else ()
                tail_deferred = nav_queue[max(urgent_count, 1) :]
            else:
                self.stage_deferred(text, attachments)
                tail_urgent = ()
                tail_deferred = nav_queue[1:]
            self.urgent.extend(tail_urgent)
            self.deferred.extend(tail_deferred)
            return
        self.restore_from_snapshot(nav_queue, urgent_count=urgent_count)
        # Non-edit commits (Up-navigated to a history slot, then Enter /
        # Tab) carry the *first snapshot block*'s attachments forward --
        # the snapshot block is the user's intent for "what was queued
        # when they started navigating", and discarding its attachments
        # silently loses image/PDF payloads the user staged before
        # navigation began.
        attachments = nav_queue[0].attachments if nav_queue else ()
        if lane == "urgent":
            self.stage_urgent(text, attachments)
        else:
            self.stage_deferred(text, attachments)

    def restore_from_snapshot(
        self, nav_queue: Sequence[QueuedInputBlock], *, urgent_count: int = 0
    ) -> None:
        """Restore queue contents from a navigation snapshot."""
        self.clear()
        for block in nav_queue[:urgent_count]:
            self.urgent.append(block)
        for block in nav_queue[urgent_count:]:
            self.deferred.append(block)

    def pop_urgent_message(self) -> UserMessage | None:
        """Remove urgent blocks and return them as one ``UserMessage``."""
        if not self.urgent:
            return None
        message = UserMessage(
            text=_join_text(self.urgent),
            attachments=_join_attachments(self.urgent),
        )
        self.urgent.clear()
        return message

    def commit_urgent(self, agent: Agent) -> bool:
        """Commit urgent blocks as a ``UserMessage`` if present."""
        message = self.pop_urgent_message()
        if message is None:
            return False
        agent.runtime.inbox.push_back(message)
        return True

    def commit_deferred_on_idle(self, agent: Agent) -> bool:
        """Commit deferred blocks when the current round is naturally idle."""
        if not self.deferred:
            return False
        agent.runtime.inbox.push_back(
            UserDeferredMessage(
                text=_join_text(self.deferred),
                attachments=_join_attachments(self.deferred),
            )
        )
        self.deferred.clear()
        return True

    def peek_tail_preview(self) -> str:
        """Return the most important staged block for discard messages.

        Read-only: leaves both lanes intact. Urgent has priority over
        deferred (Enter-staged is "more committed" than Tab-staged), and
        within a lane the most recently appended block is returned --
        the user-visible tail of the queue preview.
        """
        if self.urgent:
            return self.urgent[-1].text
        if self.deferred:
            return self.deferred[-1].text
        return ""


def _join_text(blocks: list[QueuedInputBlock]) -> str:
    return "\n\n".join(block.text for block in blocks)


def _join_attachments(blocks: list[QueuedInputBlock]) -> tuple[BytesMessage, ...]:
    return tuple(attachment for block in blocks for attachment in block.attachments)
