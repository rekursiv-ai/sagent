"""List tool: directory listing (replacement for ``ls``)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import time

from sagent.custom_types import Message, TextMessage
from sagent.lib.json import JSON, bool_val, int_val, json_freeze
from sagent.lib.message import get_directive
from sagent.tools.core import (
    get_tool_state,
    load_tool_description,
    run_sync,
)
from sagent.tools.lib.bash import Node, unwrap_cd_prefix


_NUDGE = "ls via Bash is a bad UX. Use the List tool."
_NUDGE_GLOB = "ls glob via Bash is a bad UX. Use the Glob tool."


class List:
    """List directory contents."""

    name: str = "List"
    tool_id: str = "application/x-tool-list"
    description: str = load_tool_description("List")
    supports_microcompaction: bool = True
    directive_schema: JSON = json_freeze(
        {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory to list (absolute or cwd-relative).",
                },
                "show_hidden": {
                    "type": "boolean",
                    "description": "Include dotfiles. Default false.",
                },
                "long": {
                    "type": "boolean",
                    "description": (
                        "Include size and mtime per entry (like ``ls -l``). Default false."
                    ),
                },
                "max_results": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Maximum number of entries. Default 500. Must be ≥ 1.",
                },
            },
            "required": ["path"],
        }
    )

    def summary(self, msg: Message) -> str:
        """Return a short label for this tool invocation.

        Args:
          msg: Incoming tool-use message.

        Returns:
          label: Human-readable summary with directory path.

        """
        directive = get_directive(msg)
        path = str(directive.get("path", "")) or "."
        return f"List {path}"

    def prompt(self) -> str:
        """Return supplemental prompt text for this tool.

        Returns:
          prompt: Always empty.

        """
        return ""

    async def run(self, msg: Message) -> Message:
        """List entries in a directory.

        Args:
          msg: Incoming tool-use message containing the directive.

        Returns:
          result: Tool result Message with directory listing.

        """
        directive = get_directive(msg)
        path = str(directive.get("path", ".") or ".")
        show_hidden = bool_val(directive.get("show_hidden"), False)
        long = bool_val(directive.get("long"), False)
        max_results = int_val(directive.get("max_results"), 500)
        return await run_sync(
            self._run,
            parent_id=msg.id,
            path=path,
            show_hidden=show_hidden,
            long=long,
            max_results=max_results,
        )

    def _run(
        self,
        *,
        path: str = ".",
        show_hidden: bool = False,
        long: bool = False,
        max_results: int = 500,
    ) -> str | Message:
        if not Path(path).is_absolute():
            path = str(Path(get_tool_state().bash_cwd) / path)
        p = Path(path)
        if not p.exists():
            return TextMessage(f"Not found: {path}", "text/x-error")
        if not p.is_dir():
            return TextMessage(f"Not a directory: {path}", "text/x-error")
        try:
            entries = sorted(p.iterdir(), key=lambda e: e.name)
        except OSError as err:
            return TextMessage(f"Error reading {path}: {err}", "text/x-error")
        if not show_hidden:
            entries = [e for e in entries if not e.name.startswith(".")]
        total = len(entries)
        entries = entries[:max_results]
        lines: list[str] = []
        for e in entries:
            name = e.name + ("/" if e.is_dir() else "")
            if long:
                try:
                    st = e.stat()
                    size = st.st_size
                    mtime = time.strftime("%Y-%m-%d %H:%M", time.localtime(st.st_mtime))
                except OSError:
                    size, mtime = 0, "?"
                lines.append(f"{size:>10}  {mtime}  {name}")
            else:
                lines.append(name)
        out = "\n".join(lines) or "(empty directory)"
        if total > max_results:
            out += f"\n... ({total - max_results} more)"
        return out

    def bash_match(self, trees: Sequence[Node]) -> str | None:
        """Emit a hint if the command is a bare ``ls [DIR]``.

        Accepts ``cd PATH && CMD`` compounds via ``unwrap_cd_prefix``.
        Also handles ``ls -l``/``ls -la`` by suggesting ``long=True``
        (and ``show_hidden=True`` for ``-a``).

        Args:
          trees: Parsed shell AST nodes.

        Returns:
          hint: Suggested List invocation, or ``None`` if no match.

        """
        unwrapped = unwrap_cd_prefix(trees)
        if unwrapped is None:
            return None
        cwd, cmd = unwrapped
        if cmd.exe != "ls" or cmd.env_prefix:
            return None
        return _match_ls(cwd, cmd.args)


def _match_ls(cwd: str | None, args: tuple[str, ...]) -> str | None:
    """Validate ``ls`` args and return a fixed hint string.

    Flags supported: ``-l`` (long), ``-a`` (show hidden), ``-la``/``-al``.
    Any other flag bails. More than one positional bails. A positional
    containing glob metacharacters routes to Glob rather than List.
    """
    del cwd  # Hint is a fixed string; path resolution is the LLM's job.
    positional: list[str] = []
    for a in args:
        if a == "--":
            continue
        if a.startswith("-") and a != "-":
            if a.startswith("--"):
                return None
            for c in a[1:]:
                if c not in {"l", "a"}:
                    return None
            continue
        positional.append(a)
    if len(positional) > 1:
        return None
    raw = positional[0] if positional else None
    # ``ls DIR/*.py`` is a content-glob, not a directory listing.
    if raw is not None and any(c in raw for c in "*?["):
        return _NUDGE_GLOB
    return _NUDGE
