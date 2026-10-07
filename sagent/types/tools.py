"""Tool contract.

The Protocol every concrete tool implementation must satisfy, plus
``ToolResultPolicy`` -- how much of what a tool returns stays in
history. The runtime needs only ``name`` + ``run``; the wrapper / REPL
layer consumes the rest (``tool_id``, ``description``,
``directive_schema``, ``summary``, ``prompt``).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Protocol, runtime_checkable


if TYPE_CHECKING:
    from collections.abc import Mapping

    from sagent.lib.codec import PlainTree
    from sagent.types.runtime import ToolResult
    from sagent.types.settings import AgentSettings


__all__ = [
    "DEFAULT_MAX_RESULT_CHARS",
    "MAX_RESULT_TOKENS",
    "MAX_ROUND_RESULT_CHARS",
    "ResultBounded",
    "Tool",
    "ToolResultClearable",
    "ToolResultPolicy",
]


# Claude Code's limits (``constants/toolLimits.ts``, ``FileReadTool/limits.ts``).
# Fixed, not window-derived: a fraction of a 1M window let one Grep result
# carry 113k tokens, and a 450k-character line, into session ``ca1c4eb5``.
DEFAULT_MAX_RESULT_CHARS: Final = 50_000
"""Characters one result may hold before it is persisted; caps every tool's own."""

MAX_ROUND_RESULT_CHARS: Final = 200_000
"""Characters one round of parallel tool results may hold in total."""

MAX_RESULT_TOKENS: Final = 25_000
"""Tokens a self-bounding tool (Read, Grep, Glob, Bash) emits before paging."""


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolResultPolicy:
    """Host overrides for stored and provider-visible tool results.

    ``persist_tokens`` bounds each stored result independently.
    ``message_budget_tokens`` bounds their aggregate request materialization.
    """

    persist_tokens: int = 0
    """Host per-result token cap; ``0`` leaves only the fixed limits."""

    message_budget_tokens: int = 0
    """Aggregate live tool-result budget for one request; ``0`` disables."""

    def __post_init__(self) -> None:
        """Validate policy parameters."""
        if self.persist_tokens < 0:
            raise ValueError(f"persist_tokens must be >= 0, got {self.persist_tokens}")
        if self.message_budget_tokens < 0:
            raise ValueError(
                f"message_budget_tokens must be >= 0, got {self.message_budget_tokens}",
            )

    @classmethod
    def from_settings(cls, settings: AgentSettings) -> ToolResultPolicy:
        """Derive the aggregate budget from an agent's input window.

        No per-result cap is derived: :data:`DEFAULT_MAX_RESULT_CHARS`, each
        tool's ``max_result_chars``, and the room left in the window bound a
        single result instead.

        Args:
          settings: The agent's chosen caps.

        Returns:
          policy: Aggregate budget proportional to ``max_request_tokens``.

        Raises:
          ValueError: If ``settings`` has not resolved its input window.

        """
        window = settings.max_request_tokens
        if window is None:
            raise ValueError(
                "ToolResultPolicy.from_settings requires resolved AgentSettings",
            )
        return cls(message_budget_tokens=window // 2)


@runtime_checkable
class ResultBounded(Protocol):
    """A tool that declares how many characters one of its results may hold.

    Optional, so the many host and test tools that omit it keep satisfying
    :class:`Tool`; those get :data:`DEFAULT_MAX_RESULT_CHARS`.
    """

    @property
    def max_result_chars(self) -> int:
        """Characters one result may hold before it is persisted to disk.

        Clamped to :data:`DEFAULT_MAX_RESULT_CHARS`. ``0`` exempts a tool that
        bounds itself and whose persisted output would be read back through
        itself (Read): persisting it would only loop.
        """
        ...


class ToolResultClearable(Protocol):
    """What a provider reads to decide whose RESULTS its server may drop.

    Narrower than ``Tool`` on purpose: deciding which tool results the server
    may clear needs a name and that one flag, not the ten members the agent
    runtime consumes.
    """

    @property
    def name(self) -> str:
        """Human-readable tool name, e.g. ``"Bash"``."""
        ...

    @property
    def clearable_results(self) -> bool:
        """Whether server-side context management may drop this tool's results."""
        ...


@runtime_checkable
class Tool(Protocol):
    """Tool interface for the wrapper layer.

    The runtime sees only ``name`` + ``run``; the wrapper layer
    consumes the rest (``tool_id``, ``description``,
    ``directive_schema``, ``summary``, ``prompt``).

    Note: ``@runtime_checkable`` enables ``isinstance(obj, Tool)`` for
    duck-typed registration, but Python's protocol-isinstance only
    verifies attribute *presence* -- it does not validate signatures
    or that ``run`` is actually an ``async def``. Concrete tools that
    pass ``isinstance`` may still misbehave at call time (e.g. a
    synchronous ``run`` raises ``TypeError`` when awaited). Treat the
    check as a registration smoke test, not a correctness guarantee;
    static type checking (``basedpyright``) catches the deeper shape
    mismatches.
    """

    # Read-only, and deliberately NOT ``ClassVar``: a tool's identity is
    # never written by a consumer, so demanding a settable attribute
    # excludes a frozen dataclass and a computed ``property``, while
    # demanding a ``ClassVar`` excludes the many tools that assign theirs
    # per instance. A read-only property admits all three.
    @property
    def name(self) -> str:
        """Human-readable tool name, e.g. ``"Bash"``."""
        ...

    @property
    def tool_id(self) -> str:
        """MIME-style identifier, e.g. ``"application/x-tool-bash"``."""
        ...

    @property
    def description(self) -> str:
        """Human/model-facing description rendered into the tool schema."""
        ...

    @property
    def directive_schema(self) -> Mapping[str, PlainTree]:
        """Frozen JSON Schema for the tool's directive."""
        ...

    @property
    def clearable_results(self) -> bool:
        """Whether server-side context management may drop this tool's results."""
        ...

    def summary(self, args: Mapping[str, object]) -> str:
        """Build a short label for a pending invocation.

        Args:
          args: Directive arguments to be passed to ``run``.

        Returns:
          label: Pre-execution label for renderers.

        """
        ...

    def prompt(self) -> str | None:
        """Per-request system-prompt contribution for this tool.

        Returns:
          contribution: Prompt fragment, ``""`` for no contribution this
              round, or ``None`` to signal "no change since last call"
              so per-section caches can stay byte-identical.

        """
        ...

    def serialize_key(self, args: Mapping[str, object]) -> str | None:
        """Return a serialization key, or ``None`` for unrestricted parallelism.

        Calls in one cohort sharing a non-``None`` key run sequentially
        in submission order; this is how same-file Read/Edit/Write avoid
        racing each other (they return the resolved file path). Tools
        with no shared resource return ``None``.

        Args:
          args: Directive arguments to be passed to ``run``.

        Returns:
          key: Stable key shared by calls that must serialize, or
              ``None`` to run fully in parallel.

        """
        ...

    async def run(self, args: Mapping[str, object]) -> ToolResult:
        """Execute the tool with parsed args.

        Must not raise; populate ``ToolResult(is_error=True)`` on
        failure.

        Args:
          args: Directive arguments parsed from the model output.

        Returns:
          result: Completed tool result.

        """
        ...
