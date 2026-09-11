"""List tool: directory listing (replacement for ``ls``)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Final

import asyncio
import time

from sagent.agent.state import get_tool_state
from sagent.lib.custom_json import BoolCodec, IntCodec, json_freeze
from sagent.tools.core import load_tool_description
from sagent.tools.display import Toggle, Wrap
from sagent.tools.lib.bash import (
    Node,
    bounding_sink,
    cwd_is_known,
    operands,
    parse_line_count,
    render_command,
    replaceable,
    resolve_cwd_path,
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
                "keep_last": {
                    "type": "integer",
                    "minimum": 1,
                    "description": (
                        "Keep only the trailing N entries, in listing order"
                        " (like ``| tail -n N``). Takes precedence over"
                        " max_results."
                    ),
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
              ``long`` / ``sort`` / ``max_results`` / ``keep_last``.

        Returns:
          result: Entry listing (one per line), or an error when the
              target is missing or not a directory.

        """
        path = str(args.get("path", ".") or ".")
        show_hidden = BoolCodec.coerce(args.get("show_hidden"), False)
        long = BoolCodec.coerce(args.get("long"), False)
        sort = str(args.get("sort", _DEFAULT_SORT) or _DEFAULT_SORT)
        max_results = IntCodec.coerce(args.get("max_results"), 500)
        # ``0`` is not "disabled": the schema floor is 1, so a supplied
        # zero is a malformed directive. Distinguish absent from zero.
        keep_last = IntCodec.coerce(args.get("keep_last"), 0)
        if args.get("keep_last") is not None and keep_last < 1:
            return ToolResult(
                call_id="",
                content=f"keep_last must be >= 1; got {keep_last}.",
                is_error=True,
            )
        return await asyncio.to_thread(
            self._run,
            path,
            show_hidden=show_hidden,
            long=long,
            sort=sort,
            max_results=max_results,
            keep_last=keep_last,
        )

    def _run(
        self,
        path: str,
        *,
        show_hidden: bool,
        long: bool,
        sort: str,
        max_results: int,
        keep_last: int = 0,
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
        # ``keep_last`` slices the TAIL without reordering, which is what
        # ``| tail -n N`` does. Expressing it as a flipped sort returned
        # the same entries reversed.
        entries = entries[-keep_last:] if keep_last > 0 else entries[:max_results]
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
        if total > len(entries):
            out += f"\n... ({total - len(entries)} more)"
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
            if _ls_has_glob_positional(operands("ls", inv.args)):
                return "ls glob via Bash is a bad UX. Use the Glob tool."
            # Anywhere downstream, not just adjacent: ``ls | cat | head -5``
            # is the same five rows as ``ls | head -5``, and reading only
            # the neighbour dropped the bound on the first spelling.
            sink = bounding_sink(inv)
            count = parse_line_count(sink.args) if sink else None
            call = _ls_call(
                inv.args,
                cwd=inv.cwd,
                max_results=count,
                # ``tail`` keeps the LAST N in listing order. Rendering it
                # as a reversed sort returned those same entries backwards.
                keep_last=sink is not None and sink.exe == "tail",
            )
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

    path: str
    """The single directory operand; ``"."`` when omitted."""


def _ls_call(
    args: tuple[str, ...], *, cwd: str, max_results: int | None, keep_last: bool
) -> str:
    """Render a concrete List call for ``ls`` args, or ``""``.

    One parse decides everything: ``_parse_ls`` already rejects several
    roots and unknown flags, so deriving the path separately handed back
    a call that same parse had refused -- ``ls a b`` suggested only
    ``a``, and ``ls -I '*.pyc' /src`` named the ignore-glob as the
    directory.

    ``tail`` becomes ``keep_last``, not a flipped sort. Measured in a
    directory of ``f1``..``f5``: ``ls | tail -n 3`` prints ``f3 f4 f5``,
    while ``sort='name_desc' max_results=3`` returns ``f5 f4 f3`` -- the
    same entries in the opposite order.
    """
    parsed = _parse_ls(args)
    if parsed is None:
        return ""
    # List resolves a relative path against the AGENT's cwd, not the
    # shell's, so a ``cd`` prefix dropped here lists a different tree.
    if not cwd_is_known(cwd):
        # The ``cd`` destination is not in the command text.
        return ""
    target = resolve_cwd_path(cwd, parsed.path) or "."
    pieces = [f"path={target!r}", *_ls_fields(parsed)]
    if max_results is not None:
        pieces.append(
            f"keep_last={max_results}" if keep_last else f"max_results={max_results}"
        )
    return f" Try: List {' '.join(pieces)}"


def _ls_fields(parsed: _LsParse) -> list[str]:
    """Render the non-path List keywords implied by parsed ``ls`` flags."""
    pieces: list[str] = []
    if parsed.sort is not None:
        pieces.append(f"sort={parsed.sort!r}")
    if parsed.long:
        pieces.append("long=true")
    if parsed.show_hidden:
        pieces.append("show_hidden=true")
    return pieces


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
    return _LsParse(
        long=long,
        show_hidden=show_hidden,
        sort=sort,
        path=positional[0] if positional else ".",
    )
