"""Tool contract.

The Protocol every concrete tool implementation must satisfy. The
runtime needs only ``name`` + ``run``; the wrapper / REPL layer
consumes the rest (``tool_id``, ``description``, ``directive_schema``,
``summary``, ``summary_result``, ``prompt``).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

from sagent.lib.json import JSON
from sagent.types.history import ToolResult


__all__ = [
    "Tool",
]


@runtime_checkable
class Tool(Protocol):
    """Tool interface for the wrapper layer.

    The runtime sees only ``name`` + ``run``; the wrapper layer
    consumes the rest (``tool_id``, ``description``,
    ``directive_schema``, ``summary``, ``summary_result``, ``prompt``).
    """

    name: str
    """Human-readable tool name, e.g. ``"Bash"``."""

    tool_id: str
    """MIME-style identifier, e.g. ``"application/x-tool-bash"``."""

    description: str
    """Human/model-facing description rendered into the tool schema."""

    directive_schema: JSON
    """Frozen JSON Schema for the tool's directive."""

    clearable_results: bool
    """Whether server-side context management may drop this tool's results."""

    def summary(self, args: Mapping[str, object]) -> str:
        """Build a short label for a pending invocation.

        Args:
          args: Directive arguments to be passed to ``run``.

        Returns:
          label: Pre-execution label for renderers.

        """
        ...

    def summary_result(self, result: ToolResult) -> str | None:
        """Build a short receipt for a completed invocation.

        Args:
          result: The completed ``ToolResult``.

        Returns:
          receipt: Short receipt line, or ``None`` to suppress it.

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
