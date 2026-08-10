"""Tool contract.

The Protocol every concrete tool implementation must satisfy. The
runtime needs only ``name`` + ``run``; the wrapper / REPL layer
consumes the rest (``tool_id``, ``description``, ``directive_schema``,
``summary``, ``prompt``).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

from sagent.lib.custom_json import JSON
from sagent.types.runtime import ToolResult


__all__ = [
    "Tool",
]


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
