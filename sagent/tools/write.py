"""Write: create a file, or replace one the model has read.

Typical usage example:

  result = await Write().call(context, {"file_path": "notes.md", "content": "hi"})
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.resources import files
from typing import TYPE_CHECKING, Annotated

import asyncio
import os
import uuid

from sagent.session.types import FileWriteResult, UncategorizedToolResult
from sagent.tools.display import ToolDisplay
from sagent.tools.types import parse, schema_of


if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from treekle import PlainTree

    from sagent.tools.types import ToolContext


__all__ = [
    "Write",
    "write_file",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class Write:
    """Create a file, or replace one the model has read."""

    @dataclass(frozen=True, slots=True, kw_only=True)
    class Arguments:
        """What the model passes."""

        file_path: Annotated[str, "Path to the file."]
        content: Annotated[str, "File content to write."]

    name = "Write"
    description = (files(__package__) / "assets" / "write.md").read_text(
        encoding="utf-8",
    )
    schema = schema_of(Arguments)

    display: ToolDisplay = field(default=ToolDisplay())

    def summary(self, arguments: Mapping[str, PlainTree]) -> str:
        """Return ``Write <path>``."""
        return f"Write {arguments.get('file_path', '')}"

    def serialize_key(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> str | None:
        """Return the file, so calls on one file run in order."""
        return str(context.path(str(arguments.get("file_path", ""))).resolve())

    async def call(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> FileWriteResult | UncategorizedToolResult:
        """Write ``content`` to ``file_path``, keeping an existing file's mode.

        An existing file must have been read first, and not changed since, so
        the model never overwrites what it has not seen.

        Args:
          context: The calling agent's state.
          arguments: ``file_path``, relative to ``context.cwd``, and its new
            ``content``.

        Returns:
          result: What was written, or why not.

        """
        a = parse(arguments, Write.Arguments)
        return await asyncio.to_thread(
            _write,
            context,
            context.path(a.file_path),
            a.content,
        )


def write_file(path: Path, data: bytes) -> None:
    """Replace ``path`` with ``data`` at once, keeping its mode.

    A reader sees the old file or the new one, never part of either. The
    parent directories are made as needed, and a symlink is written through.

    Args:
      path: The file to write.
      data: Its new bytes.

    """
    # A dangling link too: replacing it would turn the link into a plain file.
    if path.is_symlink():
        path = path.resolve()
    # A rename puts a new inode in place, which would otherwise take the
    # umask's mode and turn a ``0o600`` file into ``0o644``.
    mode = path.stat().st_mode & 0o777 if path.exists() else None
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
        try:
            if mode is not None:
                os.fchmod(fd, mode)
            view = memoryview(data)
            while view:
                view = view[os.write(fd, view) :]
            # Synced before the rename, so a crash cannot leave an empty file.
            os.fsync(fd)
        finally:
            os.close(fd)
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _write(
    context: ToolContext,
    path: Path,
    content: str,
) -> FileWriteResult | UncategorizedToolResult:
    """Write ``content`` to ``path`` if the model has seen what it replaces."""
    if path.is_dir():
        return _error(f"{path} is a directory, not a file.")
    if path.exists() and path not in context.files:
        return _error(f"{path} exists and has not been read. Read it first.")
    if context.files.changed(path):
        return _error(
            f"{path} changed since it was read, by the user, a linter, or"
            " another agent. Read it again before writing it.",
        )
    write_file(path, content.encode())
    context.files.saw(path)
    return FileWriteResult(call_id="", path=str(path), content=content)


def _error(message: str) -> UncategorizedToolResult:
    return UncategorizedToolResult(call_id="", content=message, is_error=True)
