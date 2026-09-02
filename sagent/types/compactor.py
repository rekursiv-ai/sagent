"""Compaction Protocols.

``Compactor`` is the conversation-compaction strategy contract.
``ReattachPolicy`` sizes what it pulls back in afterwards.
``CompactRestorable`` is the optional opt-in for tools that want to
re-inject state into history after compaction.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from sagent.types.model import AgentSettings, Model
from sagent.types.runtime import ModelContextEvent
from sagent.types.tape import ContextSplice, TapeRecord, TapeRef


if TYPE_CHECKING:
    # ``ToolState`` lives in ``agent/state.py``; forward-referencing it on
    # ``CompactRestorable.post_compact_restore`` keeps the agent layer
    # out of the types tree.
    from sagent.agent.state import ToolState


__all__ = [
    "CompactRestorable",
    "Compactor",
    "ReattachPolicy",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class ReattachPolicy:
    """How much recently-read file content survives a compaction."""

    count: int = 5
    """Number of recently-read files to re-attach."""

    max_tokens: int = 0
    """Per-file token cap; ``0`` disables the per-file limit."""

    budget_tokens: int = 0
    """Total token budget across every re-attached file; ``0`` disables."""

    def __post_init__(self) -> None:
        if self.count < 0:
            raise ValueError(f"count must be >= 0, got {self.count}")
        if self.max_tokens < 0:
            raise ValueError(f"max_tokens must be >= 0, got {self.max_tokens}")
        if self.budget_tokens < 0:
            raise ValueError(f"budget_tokens must be >= 0, got {self.budget_tokens}")

    @classmethod
    def from_settings(cls, settings: AgentSettings) -> ReattachPolicy:
        """Derive proportional defaults from an agent's input window.

        Args:
          settings: The agent's chosen caps.

        Returns:
          policy: Re-attach caps proportional to ``max_request_tokens``.

        """
        window = settings.max_request_tokens
        return cls(
            count=5,
            max_tokens=max(window // 40, 2_000),
            budget_tokens=max(window // 4, 10_000),
        )


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

    @property
    def reattach(self) -> ReattachPolicy:
        """How much file content this strategy pulls back after compacting."""
        ...

    def largest_context(self, settings: AgentSettings) -> int:
        """Return the largest input this strategy lets a request reach.

        The compaction heuristic as a token count. :meth:`should_compact`
        is the same fact as a bool, so an implementation MUST derive one
        from the other -- four call sites read this heuristic (the
        proactive turn gate, the model-swap gate, the scrunch target, and
        the post-compact enrich budget) and each re-derived its own until
        they disagreed by 137k tokens on a 1M window, leaving a band where
        every turn passed and any model swap compacted.

        Compaction is a heuristic, not a property of the model, so this
        lives on the strategy rather than on ``AgentSettings``: overriding
        the policy has to move the scalar and the predicate together, or a
        custom compactor's scrunch target silently disagrees with its own
        gate.

        Args:
          settings: The agent's chosen caps. An implementation is expected to
              honour ``max_response_tokens`` -- vendors bill input and
              output against one window (Anthropic rejects with ``input
              length and max_tokens exceed context limit: A + B > W``), so
              a strategy reserving only a FRACTION of the window for the
              reply fires past the point the provider accepts.

        Returns:
          largest: Input tokens a request may occupy, floored at ``0``.

        """
        ...

    def should_compact(
        self,
        current_tokens: int,
        largest_context: int,
        system_tokens: int = 0,
    ) -> bool:
        """Decide whether the conversation should be compacted now.

        Args:
          current_tokens: Current context size -- the provider's exact last
              request total plus an estimate of entries appended since.
          largest_context: Result of :meth:`largest_context` -- the window
              already net of every reservation the strategy makes. Passed
              rather than recomputed so a caller sizing something else
              against the same number (scrunch) cannot drift from the gate.
          system_tokens: Estimated system-prompt tokens -- incompressible
              overhead excluded from both the limit and the request.

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
        budget_tokens: int,
        estimate_tokens: Callable[[str], int],
    ) -> None:
        """Re-inject tool-specific context into the override payload.

        Best-effort; failure is logged and swallowed by the caller.

        Args:
          history: Payload-under-construction; mutated in place before
              the override is frozen.
          tool_state: Active tool state for tool-specific lookups.
          budget_tokens: Tokens the hook may add. ``0`` means unbounded.
          estimate_tokens: The model's own tokenizer -- the only thing
              that measures ``budget_tokens`` in the same unit the
              compactor computed it in.

        """
