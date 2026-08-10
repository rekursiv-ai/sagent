"""Write tool: create or overwrite files."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import re

from sagent.agent.state import get_tool_state
from sagent.lib.atomic_file import atomic_write_bytes
from sagent.lib.custom_json import json_freeze
from sagent.tools.core import (
    file_lock_key,
    load_tool_description,
    locked_file_write,
    resolve_tool_path,
)
from sagent.tools.display import Toggle, Wrap
from sagent.tools.tool_spec import CLI_SETTABLE
from sagent.types.runtime import ToolResult


# Matches the ``Wrote N bytes to PATH`` confirmation produced by ``_run``.
_WRITE_OK_RE = re.compile(r"^Wrote (\d+) bytes to ")


@dataclass(frozen=True, slots=True, kw_only=True)
class Write:
    """Create or overwrite files."""

    name = "Write"
    tool_id = "application/x-tool-write"
    clearable_results = False
    description = load_tool_description("Write")
    directive_schema = json_freeze(
        {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Path to the file."},
                "content": {"type": "string", "description": "File content to write."},
            },
            "required": ["file_path", "content"],
        }
    )

    output: Annotated[Toggle, CLI_SETTABLE] = "off"
    """Whether the result body renders in the pane."""

    output_head_rows: Annotated[int, CLI_SETTABLE] = 2
    """Leading body rows kept."""

    output_tail_rows: Annotated[int, CLI_SETTABLE] = 2
    """Trailing body rows kept, after a ``⋯ N lines ⋯`` marker."""

    output_max_width: Annotated[int, CLI_SETTABLE] = 0
    """Cell width cap; ``0`` uses the pane width."""

    output_wrap: Annotated[Wrap, CLI_SETTABLE] = "wrap"
    """``wrap`` continues an over-wide line, ``chop`` marks the cut."""

    def serialize_key(self, args: Mapping[str, object]) -> str | None:
        """Serialize same-file Read/Edit/Write within a cohort.

        Args:
          args: Directive carrying ``file_path``.

        Returns:
          key: Canonical path, or ``None`` when no path was supplied.

        """
        path = resolve_tool_path(str(args.get("file_path", "")))
        return file_lock_key(path) if path else None

    def summary(self, args: Mapping[str, object]) -> str:
        """Return a short label for this tool invocation.

        Args:
          args: Directive arguments destined for ``run``.

        Returns:
          label: ``Write <basename>`` line shown before invocation.

        """
        file_path = str(args.get("file_path", ""))
        fname = Path(file_path).name if file_path else "?"
        return f"Write {fname}"

    def prompt(self) -> str:
        """Return supplemental prompt text for this tool.

        Returns:
          contribution: Empty string (no per-request prompt fragment).

        """
        return ""

    async def run(self, args: Mapping[str, object]) -> ToolResult:
        """Write content to a file.

        Args:
          args: Directive with ``file_path`` and ``content`` keys.

        Returns:
          result: ``ToolResult`` carrying the byte-count confirmation or
              the failure reason.

        """
        file_path = resolve_tool_path(str(args.get("file_path", "")))
        content = str(args.get("content", ""))
        # Shared registry with Edit: same path → same lock → a concurrent
        # Edit and Write on the same file serialize against each other.
        return await locked_file_write(file_path, lambda: self._run(file_path, content))

    def _run(self, file_path: str, content: str) -> ToolResult:
        """Run the write synchronously with stale-file and mode-preservation checks."""
        p = Path(file_path)
        if p.is_dir():
            return ToolResult(
                call_id="",
                content=f"{file_path} is a directory, not a file.",
                is_error=True,
            )
        state = get_tool_state()
        file_mode: int | None = None
        if p.exists():
            error = state.enforce_read(file_path)
            if error:
                return ToolResult(call_id="", content=error, is_error=True)
            if state.check_stale(file_path):
                return ToolResult(
                    call_id="",
                    content=(
                        "File has been modified since read, either by the user, a"
                        " linter, or another agent. Read it again before"
                        " attempting to write it."
                    ),
                    is_error=True,
                )
            # Preserve the existing file's mode - atomic rename creates
            # a fresh inode that would otherwise pick up umask defaults
            # and silently flip e.g. ``0o600`` → ``0o644``.
            file_mode = p.stat().st_mode & 0o777
        data = content.encode("utf-8")
        atomic_write_bytes(p, data, file_mode=file_mode)
        state.mark_written(file_path)
        return ToolResult(
            call_id="",
            content=f"Wrote {len(data)} bytes to {file_path}",
        )
