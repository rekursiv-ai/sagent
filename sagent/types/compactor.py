"""Compaction Protocols.

``Compactor`` is the conversation-compaction strategy contract.
``CompactRestorable`` is the optional opt-in for tools that want to
re-inject state into history after compaction.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

from sagent.types.history import HistoryEntry
from sagent.types.model import Model
from sagent.types.tools import Tool


if TYPE_CHECKING:
    # ``ToolState`` lives in ``tools/core.py``; forward-referencing it on
    # ``CompactRestorable.post_compact_restore`` keeps the tools layer
    # out of the types tree.
    from sagent.tools.core import ToolState


__all__ = [
    "CompactRestorable",
    "Compactor",
]


@runtime_checkable
class Compactor(Protocol):
    """Conversation compaction strategy.

    The Agent layer's ``_AgentCompactor`` wrapper bridges this rich
    interface to the runtime's lean ``compact(history, model, args)
    -> list[HistoryEntry]`` form, threading ``transcript_path``,
    ``direction``, ``keep_recent``, ``custom_instructions``, and
    ``summary_pointers`` from agent state.
    """

    async def should_compact(
        self,
        input_tokens: int,
        max_request_tokens: int,
        max_response_tokens: int = 0,
    ) -> bool:
        """Decide whether the conversation should be compacted now.

        Args:
          input_tokens: Estimated current input token count.
          max_request_tokens: Budget cap for input tokens.
          max_response_tokens: Reserved output tokens (subtracted from cap).

        Returns:
          should: True when compaction should run before the next call.

        """
        ...

    async def compact(
        self,
        history: list[HistoryEntry],
        model: Model,
        transcript_path: Path | None = None,
        direction: Literal["from", "up_to"] = "from",
        keep_recent: int | None = None,
        custom_instructions: str | None = None,
        summary_pointers: list[tuple[str, str]] | None = None,
    ) -> list[HistoryEntry]:
        """Summarize the conversation into a compact entry list.

        Args:
          history: Full conversation history.
          model: Model used to write the summary.
          transcript_path: Optional path of the pre-compact transcript on
              disk (for ``Recompact``).
          direction: ``"from"`` keeps tail, ``"up_to"`` keeps head.
          keep_recent: Number of recent entries preserved verbatim.
          custom_instructions: Extra instructions for the summarizer.
          summary_pointers: ``(path, topic)`` pairs to surface in the summary.

        Returns:
          summary: Compacted history.

        """
        ...

    def maintain(
        self,
        history: list[HistoryEntry],
        tools: dict[str, Tool],
        *,
        last_response_time: float = 0.0,
        gap_sec: float = 3600.0,
        keep_recent: int = 5,
    ) -> None:
        """Apply between-request context maintenance (microcompaction).

        Args:
          history: Conversation history; mutated in place.
          tools: Tool registry; consulted for tool-specific trimming rules.
          last_response_time: Wall-clock seconds of the last response;
              used to gate microcompaction on cache-cold likelihood.
              ``0.0`` means "treat as cold (always microcompact)".
          gap_sec: Microcompaction is skipped when ``time.time() -
              last_response_time <= gap_sec`` so cache-warm requests
              aren't disturbed.
          keep_recent: Number of recent clearable results preserved.

        """


@runtime_checkable
class CompactRestorable(Protocol):
    """Optional protocol for tools that restore state after compaction.

    Tools opt in by implementing ``post_compact_restore``; the
    compactor wrapper invokes it on every tool that satisfies the
    protocol.
    """

    async def post_compact_restore(
        self,
        history: list[HistoryEntry],
        tool_state: ToolState,
        *,
        budget_chars: int = 100_000,
    ) -> None:
        """Re-inject tool-specific context into history after compaction.

        Best-effort; failure is logged and swallowed by the caller.

        Args:
          history: Post-compaction history; mutated in place.
          tool_state: Active tool state for tool-specific lookups.
          budget_chars: Character budget the hook should respect.

        """
