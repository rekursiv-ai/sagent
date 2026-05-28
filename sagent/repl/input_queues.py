"""REPL-local urgent/deferred input queues."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from sagent.types.runtime import UserMessage, UserQueuedMessage


if TYPE_CHECKING:
    from collections.abc import Sequence

    from sagent.agent.agent import Agent


@dataclass(slots=True, kw_only=True)
class InputQueues:
    """User-editable REPL input queues.

    ``urgent`` is Enter-created input that should run at the next chat-safe
    boundary. ``deferred`` is Tab-created input that waits for ``ModelIdle``.
    """

    urgent: list[str] = field(default_factory=list)
    deferred: list[str] = field(default_factory=list)

    def has_any(self) -> bool:
        """Return whether either queue has staged text."""
        return bool(self.urgent or self.deferred)

    def stage_urgent(self, text: str) -> None:
        """Append an urgent block."""
        self.urgent.append(text)

    def stage_deferred(self, text: str) -> None:
        """Append a deferred block."""
        self.deferred.append(text)

    def clear(self) -> None:
        """Discard all staged text."""
        self.urgent.clear()
        self.deferred.clear()

    def render_blocks(self) -> list[str]:
        """Return labelled queue blocks for the input pane."""
        return [
            *(f"urgent: {text}" for text in self.urgent),
            *(f"deferred: {text}" for text in self.deferred),
        ]

    def restore_blocks(self) -> list[str]:
        """Return editable blocks in Up-arrow priority order."""
        return [*self.urgent, *self.deferred]

    def replace_from_navigation(
        self, nav_queue: Sequence[str], text: str, *, edit_mode: bool
    ) -> None:
        """Replace queue contents after Up/Down navigation commit."""
        self.clear()
        if edit_mode and nav_queue:
            self.stage_deferred(text)
            return
        for block in nav_queue:
            self.stage_deferred(block)
        self.stage_deferred(text)

    def restore_from_snapshot(self, nav_queue: Sequence[str]) -> None:
        """Restore queue contents from a navigation snapshot."""
        self.clear()
        for block in nav_queue:
            self.stage_deferred(block)

    def commit_urgent(self, agent: Agent) -> bool:
        """Commit urgent blocks as a ``UserMessage`` if present."""
        if not self.urgent:
            return False
        agent.runtime.inbox.push_back(UserMessage(text="\n\n".join(self.urgent)))
        self.urgent.clear()
        return True

    def commit_deferred_on_idle(self, agent: Agent) -> bool:
        """Commit deferred blocks when the current round is naturally idle."""
        if not self.deferred:
            return False
        agent.runtime.inbox.push_back(
            UserQueuedMessage(text="\n\n".join(self.deferred))
        )
        self.deferred.clear()
        return True

    def pop_tail_preview(self) -> str:
        """Return the last staged block for discard messages."""
        if self.deferred:
            return self.deferred[-1]
        if self.urgent:
            return self.urgent[-1]
        return ""
