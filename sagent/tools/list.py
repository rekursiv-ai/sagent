"""List tool: directory listing (replacement for ``ls``)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Final

import asyncio
import time

from sagent.agent.state import get_tool_state
from sagent.lib.custom_json import bool_val, int_val, json_freeze
from sagent.tools.core import load_tool_description
from sagent.tools.display import Toggle, Wrap
from sagent.tools.lib.bash import (
    Node,
    render_command,
    replaceable,
    walk_commands,
)
from sagent.tools.lib.path_sort import (
    SORT_VALUES,
    safe_mtime,
    safe_size,
    sort_paths,
)
from sagent.tools.tool_spec import CLI_SETTABLE
from sagent.types.runtime import ToolResult


_NUDGE_PREFIX: Final = "ls via Bash is a bad UX. Use the List tool"

_LS_EXES: frozenset[str] = frozenset({"ls"})

_DEFAULT_SORT: Final = "name"


@dataclass(frozen=True, slots=True, kw_only=True)
class List:
    """List directory contents."""

    name = "List"
    tool_id = "application/x-tool-list"
    clearable_results = True
    description = load_tool_description("List")
    directive_schema = json_freeze(
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
                        "Include size and mtime per entry (like ``ls -l``)."
                        " Default false."
                    ),
                },
                "sort": {
                    "type": "string",
                    "enum": list(SORT_VALUES),
                    "description": (
                        "Result ordering. Default 'name' (alphabetical)."
                        " 'mtime_desc' = newest first; 'size_desc' = largest first."
                    ),
                },
                "max_results": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Maximum number of entries. Default 500.",
                },
            },
            "required": ["path"],
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

    def summary(self, args: Mapping[str, object]) -> str:
        """Return a short label for this directory listing.

        Args:
          args: Directive carrying ``path``.

        Returns:
          label: ``List <path>`` line shown before invocation.

        """
        path = str(args.get("path", "")) or "."
        return f"List {path}"

    def prompt(self) -> str:
        """Return no supplemental system-prompt text for List.

        Returns:
          contribution: Empty string.

        """
        return ""

    def serialize_key(self, args: Mapping[str, object]) -> str | None:
        """Run in parallel: read-only listing needs no serialization."""
        del args
        return None

    async def run(self, args: Mapping[str, object]) -> ToolResult:
        """List directory entries with optional sort and long-format details.

        Args:
          args: Directive with ``path`` and optional ``show_hidden`` /
              ``long`` / ``sort`` / ``max_results``.

        Returns:
          result: Entry listing (one per line), or an error when the
              target is missing or not a directory.

        """
        path = str(args.get("path", ".") or ".")
        show_hidden = bool_val(args.get("show_hidden"), False)
        long = bool_val(args.get("long"), False)
        sort = str(args.get("sort", _DEFAULT_SORT) or _DEFAULT_SORT)
        max_results = int_val(args.get("max_results"), 500)
        return await asyncio.to_thread(
            self._run, path, show_hidden, long, sort, max_results
        )

    def _run(
        self,
        path: str,
        show_hidden: bool,
        long: bool,
        sort: str,
        max_results: int,
    ) -> ToolResult:
        """Run the directory listing synchronously and return the result."""
        if sort not in SORT_VALUES:
            return ToolResult(
                call_id="",
                content=f"unknown sort: {sort!r} (expected one of {list(SORT_VALUES)})",
                is_error=True,
            )
        if max_results < 1:
            return ToolResult(
                call_id="",
                content=f"max_results must be >= 1; got {max_results}.",
                is_error=True,
            )
        if not Path(path).is_absolute():
            path = str(Path(get_tool_state().bash_cwd) / path)
        p = Path(path)
        if not p.exists():
            return ToolResult(call_id="", content=f"Not found: {path}", is_error=True)
        if not p.is_dir():
            return ToolResult(
                call_id="", content=f"Not a directory: {path}", is_error=True
            )
        try:
            entries = list(p.iterdir())
        except OSError as err:
            return ToolResult(
                call_id="", content=f"Error reading {path}: {err}", is_error=True
            )
        if not show_hidden:
            entries = [e for e in entries if not e.name.startswith(".")]
        sort_paths(entries, sort)
        total = len(entries)
        entries = entries[:max_results]
        lines: list[str] = []
        for e in entries:
            name = e.name + ("/" if e.is_dir() else "")
            if long:
                size = safe_size(e)
                mtime_raw = safe_mtime(e)
                mtime = (
                    time.strftime("%Y-%m-%d %H:%M", time.localtime(mtime_raw))
                    if mtime_raw
                    else "?"
                )
                lines.append(f"{size:>10}  {mtime}  {name}")
            else:
                lines.append(name)
        out = "\n".join(lines) or "(empty directory)"
        if total > max_results:
            out += f"\n... ({total - max_results} more)"
        return ToolResult(call_id="", content=out)

    def bash_match(self, trees: Sequence[Node]) -> str | None:
        """Emit a hint when any command is an ``ls`` invocation.

        Policy only: :func:`walk_commands` supplies every simple command
        with its context, so a leading ``cd``, an enclosing loop, and a
        trailing ``| head`` all reach this matcher without it re-deriving
        AST shape.

        Args:
          trees: Parsed bashlex command trees from the active Bash call.

        Returns:
          hint: Nudge string redirecting to the List tool, or ``None``.

        """
        for inv in walk_commands(trees):
            if not replaceable(inv, exes=_LS_EXES):
                continue
            if _ls_has_glob_positional(inv.args):
                return "ls glob via Bash is a bad UX. Use the Glob tool."
            sink = inv.piped_into
            fields = _ls_fields(
                inv.args,
                max_results=_parse_line_count(sink.args) if sink else None,
                flip_sort=bool(sink) and sink.exe == "tail",
            )
            call = f" Try: List path={_ls_path(inv.args)!r}"
            if fields:
                call += f" {fields}"
            return f"{_NUDGE_PREFIX}. Replaces: `{render_command(inv)}`.{call}"
        return None


@dataclass(frozen=True, slots=True, kw_only=True)
class _LsParse:
    """Parsed ``ls`` flag state used to build a List-tool nudge."""

    long: bool
    """``-l`` flag was present."""

    show_hidden: bool
    """``-a`` / ``-A`` flag was present."""

    sort: str | None
    """Sort key derived from ``-t`` / ``-S`` / ``-r``, or ``None``."""


def _ls_path(args: tuple[str, ...]) -> str:
    """Directory operand of an ``ls`` invocation; ``"."`` when omitted."""
    return next((a for a in args if not a.startswith("-") and a != "--"), ".")


def _ls_fields(
    args: tuple[str, ...], *, max_results: int | None, flip_sort: bool
) -> str:
    """Render List keywords for ``ls`` args, or ``""`` when untranslatable.

    ``tail`` flips the sort: the last N of an ascending listing is the
    first N of a descending one, which is what ``sort`` plus
    ``max_results`` already say.
    """
    parsed = _parse_ls(args)
    if parsed is None:
        return ""
    sort = _flip_sort(parsed.sort) if flip_sort else parsed.sort
    pieces: list[str] = []
    if sort is not None:
        pieces.append(f"sort={sort!r}")
    if parsed.long:
        pieces.append("long=true")
    if parsed.show_hidden:
        pieces.append("show_hidden=true")
    if max_results is not None:
        pieces.append(f"max_results={max_results}")
    return " ".join(pieces)


def _flip_sort(sort: str | None) -> str:
    """Reverse a sort-direction key. ``None`` means default name asc."""
    if sort is None:
        return "name_desc"
    if sort.endswith("_desc"):
        return sort.removesuffix("_desc")
    return f"{sort}_desc"


def _ls_has_glob_positional(args: tuple[str, ...]) -> bool:
    """True iff any non-flag argument contains a glob metacharacter."""
    for a in args:
        if a == "--" or (a.startswith("-") and a != "-"):
            continue
        if any(c in a for c in "*?["):
            return True
    return False


def _parse_ls(args: tuple[str, ...]) -> _LsParse | None:
    """Parse ``ls`` flags. Returns None on unsupported flag or arg shape."""
    long = False
    show_hidden = False
    sort_t = False
    sort_s = False
    reverse = False
    positional: list[str] = []
    for a in args:
        if a == "--":
            continue
        if a.startswith("-") and a != "-":
            if a.startswith("--"):
                return None
            for c in a[1:]:
                if c == "l":
                    long = True
                elif c in {"a", "A"}:
                    show_hidden = True
                elif c == "t":
                    sort_t = True
                elif c == "S":
                    sort_s = True
                elif c == "r":
                    reverse = True
                else:
                    return None
            continue
        positional.append(a)
    if len(positional) > 1:
        return None
    if sort_t and sort_s:
        return None
    sort: str | None
    if sort_t:
        sort = "mtime" if reverse else "mtime_desc"
    elif sort_s:
        sort = "size" if reverse else "size_desc"
    elif reverse:
        sort = "name_desc"
    else:
        sort = None
    return _LsParse(long=long, show_hidden=show_hidden, sort=sort)


def _parse_line_count(args: tuple[str, ...]) -> int | None:
    """Extract line count from ``head``/``tail`` args."""
    if not args:
        return 10
    count: int | None = None
    i = 0
    while i < len(args):
        a = args[i]
        if a == "-n":
            if i + 1 >= len(args) or not args[i + 1].isdigit():
                return None
            count = int(args[i + 1])
            i += 2
            continue
        if a.startswith("-n"):
            if not a[2:].isdigit():
                return None
            count = int(a[2:])
            i += 1
            continue
        if len(a) > 1 and a[0] == "-" and a[1:].isdigit():
            count = int(a[1:])
            i += 1
            continue
        return None
    if count is None or count < 1:
        return None
    return count
