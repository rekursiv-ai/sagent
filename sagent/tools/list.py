"""List: a directory's entries, like ``ls``.

Typical usage example:

  result = await List().call(context, {"path": "src", "sort": "mtime_desc"})
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.resources import files
from typing import TYPE_CHECKING, Annotated, Literal

import asyncio
import time

from sagent.session.types import UncategorizedToolResult
from sagent.tools.command_analyzer import (
    Invocation,
    bounding_sink,
    line_count,
    operands,
    replaceable,
    resolve,
)
from sagent.tools.display import ToolDisplay
from sagent.tools.types import Arg, parse, schema_of


if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from treekle import PlainTree

    from sagent.tools.types import ToolContext


__all__ = [
    "List",
    "Sort",
    "ls_row",
    "sort_paths",
]


type Sort = Literal[
    "name",
    "name_desc",
    "mtime",
    "mtime_desc",
    "size",
    "size_desc",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class List:
    """List a directory's entries."""

    @dataclass(frozen=True, slots=True, kw_only=True)
    class Arguments:
        """What the model passes."""

        path: Annotated[str, "Directory to list (absolute or cwd-relative)."]
        show_hidden: Annotated[bool, "Include dotfiles. Default false."] = False
        long: Annotated[
            bool,
            "Include size and mtime per entry, like ``ls -l``.",
        ] = False
        sort: Annotated[
            Sort,
            "Result ordering. Default 'name'. 'mtime_desc' is newest first;"
            " 'size_desc' largest first.",
        ] = "name"
        max_results: Annotated[
            int,
            "Most entries. Default 500.",
            Arg(minimum=1),
        ] = 500
        keep_last: Annotated[
            int | None,
            "Keep only the last N entries, in listing order, like"
            " ``| tail -n N``. Overrides max_results.",
            Arg(minimum=1),
        ] = None

    name = "List"
    description = (files(__package__) / "assets" / "list.md").read_text(
        encoding="utf-8",
    )
    schema = schema_of(Arguments)

    display: ToolDisplay = field(default=ToolDisplay())

    def summary(self, arguments: Mapping[str, PlainTree]) -> str:
        """Return ``List <path>``."""
        return f"List {arguments.get('path') or '.'}"

    def serialize_key(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> str | None:
        """Return ``None``: listing changes nothing."""
        del context, arguments
        return None

    def from_shell(
        self,
        invocation: Invocation,
    ) -> Mapping[str, PlainTree] | str | None:
        """Return the List call for ``ls`` of one directory.

        Args:
          invocation: One simple command, with its pipeline.

        Returns:
          arguments: The List call showing the same entries; a message for an
            ``ls`` no call matches, such as one with a glob; or ``None`` when
            the command is not ``ls``.

        """
        if not replaceable(invocation, exes=frozenset({"ls"}), bare=True):
            return None
        refusal = "List one directory per List call; for a pattern, use Glob."
        # Past ``--`` every word is a name, however it starts.
        args = invocation.args
        options = args[: args.index("--")] if "--" in args else args
        flags = "".join(a[1:] for a in options if a.startswith("-") and a != "-")
        paths = operands(args)
        # ``-1`` and ``-h`` change only how rows look.
        if (
            set(flags) - set("laAtSr1h")
            or ("t" in flags and "S" in flags)
            or len(paths) > 1
        ):
            return refusal
        target = resolve(invocation.cwd, paths[0] if paths else ".")
        if target is None or any(c in target for c in "*?[$`"):
            return refusal
        key = "mtime" if "t" in flags else "size" if "S" in flags else "name"
        # ``ls -t`` lists newest first; ``-r`` reverses whichever order.
        newest_first = key != "name"
        sort = key if ("r" in flags) == newest_first else f"{key}_desc"
        arguments: dict[str, PlainTree] = {"path": target or "."}
        if sort != "name":
            arguments["sort"] = sort
        if "l" in flags:
            arguments["long"] = True
        if set(flags) & set("aA"):
            arguments["show_hidden"] = True
        sink = bounding_sink(invocation)
        if sink is not None:
            count = line_count(sink.args)
            if count is None:
                return refusal
            arguments["keep_last" if sink.exe == "tail" else "max_results"] = count
        return arguments

    async def call(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> UncategorizedToolResult:
        """List a directory's entries, directories marked with ``/``.

        Args:
          context: The calling agent's state.
          arguments: What the model passed, as ``List.Arguments``.

        Returns:
          result: One entry per line, or why there are none.

        """
        return await asyncio.to_thread(_list, context, parse(arguments, List.Arguments))


def sort_paths(paths: list[Path], sort: Sort) -> None:
    """Sort ``paths`` in place.

    Args:
      paths: The paths to sort.
      sort: By ``name`` (as GNU ``ls``), ``mtime``, or ``size``; ``_desc``
        reverses.

    """
    key = sort.removesuffix("_desc")
    if key == "name":
        paths.sort(key=lambda p: (p.name.casefold(), p.name))
    else:
        paths.sort(key=lambda p: _stat(p, key))
    if sort.endswith("_desc"):
        paths.reverse()


def ls_row(path: Path, *, name: str) -> str:
    """Return ``path`` as an ``ls -l`` row: size, mtime, then ``name``.

    Args:
      path: The file or directory.
      name: How the row names it.

    Returns:
      row: ``<size>  <YYYY-mm-dd HH:MM>  <name>``; ``?`` for an unreadable time.

    """
    mtime = _stat(path, "mtime")
    when = time.strftime("%Y-%m-%d %H:%M", time.localtime(mtime)) if mtime else "?"
    return f"{_stat(path, 'size'):>10}  {when}  {name}"


def _list(context: ToolContext, a: List.Arguments) -> UncategorizedToolResult:
    """List the directory ``a`` names."""
    directory = context.path(a.path or ".")
    if not directory.exists():
        return _error(f"Not found: {directory}")
    if not directory.is_dir():
        return _error(f"Not a directory: {directory}")
    try:
        entries = list(directory.iterdir())
    except OSError as e:
        return _error(f"Error reading {directory}: {e}")
    if not a.show_hidden:
        entries = [e for e in entries if not e.name.startswith(".")]
    sort_paths(entries, a.sort)
    total = len(entries)
    # A tail keeps listing order, as ``| tail`` does; a reversed sort would
    # return the same entries backwards.
    entries = (
        entries[: a.max_results] if a.keep_last is None else entries[-a.keep_last :]
    )
    names = [e.name + ("/" if e.is_dir() else "") for e in entries]
    rows = (
        [ls_row(e, name=n) for e, n in zip(entries, names, strict=True)]
        if a.long
        else names
    )
    out = "\n".join(rows) or "(empty directory)"
    if total > len(entries):
        out += f"\n... ({total - len(entries):,} more)"
    return UncategorizedToolResult(call_id="", content=out)


def _stat(path: Path, key: str) -> float:
    """Return ``path``'s mtime or size, or ``0`` for a broken link."""
    try:
        stat = path.stat()
    except OSError:
        return 0
    return stat.st_mtime if key == "mtime" else stat.st_size


def _error(message: str) -> UncategorizedToolResult:
    return UncategorizedToolResult(call_id="", content=message, is_error=True)
