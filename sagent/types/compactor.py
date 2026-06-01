"""Compaction Protocols.

``Compactor`` is the conversation-compaction strategy contract.
``CompactRestorable`` is the optional opt-in for tools that want to
re-inject state into history after compaction.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from sagent.types.model import Model
from sagent.types.runtime import ModelContextEvent
from sagent.types.tape import ContextSplice, TapeRecord, TapeRef


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

    Compactors produce :class:`ContextSplice` events instead of
    replacement history. The agent's wrapper appends the override to
    the runtime tape; the resolver renders it like any other override.

    The Agent layer's ``_AgentCompactor`` bridges this rich interface
    to the runtime's lean ``compact(tape, context, model, args)``
    form, threading ``custom_instructions`` from agent state.
    """

    def should_compact(
        self,
        current_tokens: int,
        max_request_tokens: int,
        system_tokens: int = 0,
    ) -> bool:
        """Decide whether the conversation should be compacted now.

        Args:
          current_tokens: Current context size -- the provider's exact last
              request total plus an estimate of entries appended since.
          max_request_tokens: Budget cap for input tokens.
          system_tokens: Estimated system-prompt tokens -- incompressible
              overhead excluded from both window and request.

        Returns:
          should: True when compaction should run before the next call.

        """
        ...

    async def compact(
        self,
        tape: Sequence[TapeRecord],
        context: Sequence[ModelContextEvent],
        model: Model,
        mint_ref: Callable[[], TapeRef],
        custom_instructions: str | None = None,
    ) -> ContextSplice:
        """Produce a barrier override that replaces older context.

        Args:
          tape: Append-only session tape.
          context: Resolved provider-facing context (the resolver's
              ``messages`` view of ``tape``).
          model: Model used to write the summary.
          mint_ref: Factory returning fresh ``TapeRef`` values for the
              compactor's produced records.
          custom_instructions: Extra instructions for the summarizer.

        Returns:
          override: Barrier ``ContextSplice`` with the summary payload.

        """
        ...


@runtime_checkable
class CompactRestorable(Protocol):
    """Optional protocol for tools that restore state after compaction.

    Tools opt in by implementing ``post_compact_restore``; the
    compactor wrapper invokes it on every tool that satisfies the
    protocol against the payload-under-construction (a plain mutable
    ``list[ModelContextEvent]``) before the override is frozen and appended
    to the tape.
    """

    async def post_compact_restore(
        self,
        history: list[ModelContextEvent],
        tool_state: ToolState,
        *,
        budget_chars: int = 100_000,
    ) -> None:
        """Re-inject tool-specific context into the override payload.

        Best-effort; failure is logged and swallowed by the caller.

        Args:
          history: Payload-under-construction; mutated in place before
              the override is frozen.
          tool_state: Active tool state for tool-specific lookups.
          budget_chars: Character budget the hook should respect.

        """
