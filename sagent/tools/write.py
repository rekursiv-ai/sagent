"""Write tool: create or overwrite files."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import asyncio
import re

from sagent.lib.atomic_file import atomic_write_bytes
from sagent.lib.json import JSON, json_freeze
from sagent.tools.core import (
    get_file_write_lock,
    get_tool_state,
    load_tool_description,
    resolve_tool_path,
)
from sagent.types.history import ToolResult


# Matches the ``Wrote N bytes to PATH`` confirmation produced by ``_run``.
_WRITE_OK_RE = re.compile(r"^Wrote (\d+) bytes to ")


class Write:
    """Create or overwrite files."""

    name: str = "Write"
    tool_id: str = "application/x-tool-write"
    description: str = load_tool_description("Write")
    supports_microcompaction: bool = True
    emit_tool_summary: bool = False
    directive_schema: JSON = json_freeze(
        {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Path to the file."},
                "content": {"type": "string", "description": "File content to write."},
            },
            "required": ["file_path", "content"],
        }
    )

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

    def summary_result(self, result: ToolResult) -> str | None:
        """One-line receipt: confirmation count from the success message.

        Args:
          result: Completed ``ToolResult`` from ``run``.

        Returns:
          receipt: ``wrote N bytes`` line, or ``None`` when suppressed.

        """
        if not self.emit_tool_summary or result.is_error:
            return None
        match = _WRITE_OK_RE.match(result.content.strip())
        if match:
            return f"wrote {match.group(1)} bytes"
        return None

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
        async with get_file_write_lock(file_path):
            return await asyncio.to_thread(self._run, file_path, content)

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
