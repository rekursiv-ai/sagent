"""Write tool: create or overwrite files."""

from __future__ import annotations

from pathlib import Path

import re

from sagent.custom_types import Message, TextMessage
from sagent.lib.atomic_file import atomic_write_bytes
from sagent.lib.json import JSON, json_freeze
from sagent.lib.message import get_directive
from sagent.tools.core import (
    get_file_write_lock,
    get_tool_state,
    load_tool_description,
    resolve_tool_path,
    run_sync,
)


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

    def summary(self, msg: Message) -> str:
        """Return a short label for this tool invocation.

        Args:
          msg: Incoming tool-use message.

        Returns:
          label: Human-readable summary with filename.

        """
        directive = get_directive(msg)
        file_path = str(directive.get("file_path", ""))
        fname = Path(file_path).name if file_path else "?"
        return f"Write {fname}"

    def summary_result(self, result: Message) -> str | None:
        """One-line receipt: confirmation count from the success message."""
        if not self.emit_tool_summary:
            return None
        if result.descriptor != "text/plain":
            return None
        text = str(result.content).strip()
        # ``_run`` returns "Wrote N bytes to PATH"; surface the byte count.
        match = _WRITE_OK_RE.match(text)
        if match:
            return f"wrote {match.group(1)} bytes"
        return text or None

    def prompt(self) -> str:
        """Return supplemental prompt text for this tool.

        Returns:
          prompt: Always empty.

        """
        return ""

    async def run(self, msg: Message) -> Message:
        """Write content to a file.

        Args:
          msg: Incoming tool-use message containing the directive.

        Returns:
          result: Tool result Message confirming the write.

        """
        directive = get_directive(msg)
        file_path = resolve_tool_path(str(directive.get("file_path", "")))
        content = str(directive.get("content", ""))
        # Shared registry with Edit: same path → same lock → a concurrent
        # Edit and Write on the same file serialize against each other.
        async with get_file_write_lock(file_path):
            return await run_sync(
                self._run, parent_id=msg.id, file_path=file_path, content=content
            )

    def _run(self, *, file_path: str, content: str) -> str | Message:
        p = Path(file_path)
        if p.is_dir():
            return TextMessage(
                f"{file_path} is a directory, not a file.", "text/x-error"
            )
        state = get_tool_state()
        file_mode: int | None = None
        if p.exists():
            error = state.enforce_read(file_path)
            if error:
                return TextMessage(error, "text/x-error")
            if state.check_stale(file_path):
                return TextMessage(
                    (
                        "File has been modified since read, either by the user, a"
                        " linter, or another agent. Read it again before"
                        " attempting to write it."
                    ),
                    "text/x-error",
                )
            # Preserve the existing file's mode - atomic rename creates
            # a fresh inode that would otherwise pick up umask defaults
            # and silently flip e.g. ``0o600`` → ``0o644``.
            file_mode = p.stat().st_mode & 0o777
        data = content.encode("utf-8")
        atomic_write_bytes(p, data, file_mode=file_mode)
        state.mark_written(file_path)
        return f"Wrote {len(data)} bytes to {file_path}"
