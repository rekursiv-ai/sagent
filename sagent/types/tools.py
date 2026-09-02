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
from typing import Protocol, runtime_checkable

from sagent.lib.custom_json import JSON
from sagent.types.model import AgentSettings
from sagent.types.runtime import ToolResult


__all__ = [
    "Tool",
    "ToolResultPolicy",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolResultPolicy:
    """When a tool result is off-loaded to disk instead of kept in history.

    Both thresholds are read against a running per-request total, so a
    result's fate depends on what ran before it in the same turn.
    """

    persist_tokens: int = 0
    """Per-result token threshold for disk off-loading; ``0`` disables."""

    message_budget_tokens: int = 0
    """Aggregate live tool-result budget for one request; ``0`` disables."""

    def __post_init__(self) -> None:
        if self.persist_tokens < 0:
            raise ValueError(f"persist_tokens must be >= 0, got {self.persist_tokens}")
        if self.message_budget_tokens < 0:
            raise ValueError(
                f"message_budget_tokens must be >= 0, got {self.message_budget_tokens}"
            )

    @classmethod
    def from_settings(cls, settings: AgentSettings) -> ToolResultPolicy:
        """Derive proportional defaults from an agent's input window.

        Args:
          settings: The agent's chosen caps.

        Returns:
          policy: Off-load thresholds proportional to ``max_request_tokens``.

        """
        window = settings.max_request_tokens
        return cls(
            persist_tokens=window // 4,
            message_budget_tokens=window // 2,
        )


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
    def directive_schema(self) -> JSON:
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
