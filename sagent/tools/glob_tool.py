"""Glob: the files under a directory whose paths match a pattern.

Unlike List, Glob walks the whole tree, skips what ``.gitignore`` ignores,
and returns absolute paths of files only.

Typical usage example:

  result = await settled(Glob().call(context, {"pattern": "**/*.py", "sort": "mtime_desc"}))
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import partial
from importlib.resources import files
from typing import TYPE_CHECKING, Annotated

import asyncio

from sagent.journal.types import GlobResult
from sagent.lib.files.glob import GlobError, glob, in_path_order, streamed_paths
from sagent.tools.command_analyzer import (
    Invocation,
    bounding_sink,
    line_count,
    replaceable,
    resolve,
)
from sagent.tools.display import ToolDisplay
from sagent.tools.list import Sort, ls_row, sort_paths
from sagent.tools.types import (
    Arg,
    Stream,
    ThreadWriter,
    ToolOutput,
    ToolStreamingResult,
    json_schema,
    parse,
    whole_output,
    write_fitting,
)


if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from treekle import PlainTree

    from sagent.tools.types import ToolContext


__all__ = [
    "Glob",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class Glob:
    """Find files whose paths match a pattern."""

    @dataclass(frozen=True, slots=True, kw_only=True)
    class Arguments:
        """What the model passes."""

        pattern: Annotated[str, "The glob pattern to match files against."]
        path: Annotated[str, "Directory to search in. Defaults to the cwd."] = "."
        sort: Annotated[
            Sort,
            "Result ordering. Default 'none': as found, each path as soon as found."
            " 'name' is by path; 'mtime_desc' is newest first.",
        ] = "none"
        long: Annotated[
            bool,
            "Prefix each match with size and mtime, like ``ls -l``.",
        ] = False
        max_results: Annotated[
            int,
            "Most matches. Omit or 0 to return as many as fit.",
            Arg(minimum=0),
        ] = 0
        offset: Annotated[
            int,
            "Skip the first N matches, to page through many.",
            Arg(minimum=0),
        ] = 0

    name = "Glob"
    description = (files(__package__) / "assets" / "glob.md").read_text(
        encoding="utf-8",
    )
    schema = json_schema(Arguments)

    display: ToolDisplay = field(default=ToolDisplay())

    def summary(self, arguments: Mapping[str, PlainTree]) -> str:
        """Return ``Glob <pattern>``, with ``in <path>`` when not the cwd."""
        path = arguments.get("path")
        where = f" in {path}" if path not in {None, "", "."} else ""
        return f"Glob {arguments.get('pattern', '')}{where}"

    def serialize_key(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> str | None:
        """Return ``None``: matching changes nothing."""
        del context, arguments
        return None

    def from_shell(
        self,
        invocation: Invocation,
    ) -> Mapping[str, PlainTree] | str | None:
        """Return the Glob call for ``find ROOT [-type f] -name PATTERN``.

        Glob skips what ``.gitignore`` ignores, where ``find`` does not, which
        is what a model listing source files wants. Any other predicate --
        ``-maxdepth``, ``-type d``, ``-delete`` -- asks what Glob cannot say,
        so the command stays a shell command, as Edit's regex ``sed`` does.

        Args:
          invocation: One simple command, with its pipeline.

        Returns:
          arguments: The Glob call for the same names, or ``None``.

        """
        # Bash fills in ``$x`` before find sees it; Glob would match it as is.
        # ``replaceable`` refuses a pipe into ``xargs``, which acts on each match.
        if invocation.expands or not replaceable(
            invocation,
            exes=frozenset({"find"}),
            bare=True,
        ):
            return None
        root, name = ".", ""
        args = list(invocation.args)
        while args:
            arg = args.pop(0)
            # Two names must both match, which one Glob pattern cannot say.
            if arg == "-name" and args and not name:
                name = args.pop(0)
            elif arg == "-type" and args[:1] == ["f"]:
                # Glob matches only files, so ``-type f`` says nothing more.
                args.pop(0)
            elif arg.startswith("-") or root != ".":
                return None
            else:
                root = arg
        target = resolve(invocation.cwd, root)
        if not name or target is None or "/" in name:
            return None
        arguments: dict[str, PlainTree] = {"pattern": f"**/{name}"}
        if target:
            arguments["path"] = target
        sink = bounding_sink(invocation)
        if sink is not None:
            count = line_count(sink.args)
            if count is None or sink.exe == "tail":
                return None
            arguments["max_results"] = count
        return arguments

    def call(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> ToolStreamingResult[GlobResult, str, str]:
        """Find files under ``path`` matching ``pattern``.

        An absolute ``pattern`` ignores ``path``.

        Args:
          context: The calling agent's state.
          arguments: What the model passed, as ``Glob.Arguments``.

        Returns:
          streaming_result: One absolute path per line, each its own item, with a
            note to resume when cut.

        """
        stdout = Stream[str]()
        return whole_output(self._run(context, arguments, stdout), stdout=stdout)

    async def _run(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
        stdout: Stream[str],
    ) -> ToolOutput[GlobResult, str, str]:
        a = parse(arguments, Glob.Arguments)
        result = GlobResult(pattern=a.pattern, path=a.path)
        root = context.path(a.path or ".")
        try:
            if a.sort != "none":
                page = await asyncio.to_thread(_sorted_page, context, a, root)
                return ToolOutput(stdout=tuple(page), result=result)
            found, whole = await asyncio.to_thread(
                _stream,
                context,
                a,
                root,
                ThreadWriter(stdout, asyncio.get_running_loop()),
            )
        except GlobError as e:
            return ToolOutput(stderr=(str(e),), result=replace(result, exit_code=1))
        if not found:
            return ToolOutput(stdout=("(no matches)",), result=result)
        # Every path written, as found, is the answer when nothing is left to page.
        if (
            whole
            and not a.offset
            and (not a.max_results or len(found) <= a.max_results)
        ):
            return ToolOutput(result=result)
        # A page is of one order every walk shares, so ``offset`` resumes it.
        page = await asyncio.to_thread(_page, context, a, in_path_order(found))
        return ToolOutput(result=result, shown="".join(page))


def _sorted_page(context: ToolContext, a: Glob.Arguments, root: Path) -> list[str]:
    """Return the page of matches ``a`` asks for, sorted as it says."""
    found = glob(root, a.pattern)
    # ``glob`` sorts by path, which is ``name`` here.
    if a.sort == "name_desc":
        found.reverse()
    elif a.sort != "name":
        sort_paths(found, a.sort)
    return _page(context, a, found)


def _stream(
    context: ToolContext,
    a: Glob.Arguments,
    root: Path,
    writer: ThreadWriter[str],
) -> tuple[list[Path], bool]:
    """Write each match as found while it fits; return them all, and whether all fit."""
    return write_fitting(
        context,
        streamed_paths(root, a.pattern),
        partial(_row, a=a),
        writer,
    )


def _page(context: ToolContext, a: Glob.Arguments, matches: list[Path]) -> list[str]:
    """Return the window of ``matches`` ``a`` asks for that fits, a unit a path, and the note."""
    end = a.offset + a.max_results if a.max_results else None
    window = matches[a.offset : end]
    if not window:
        # Past the last match, saying none matched would be false.
        return ["(no matches in this window)" if matches else "(no matches)"]
    total = len(matches)
    # Each row after the first brings the newline before it, so the text ends on
    # none; room is left for the longest note the cut could need.
    units = (_row(p, a) if i == 0 else f"\n{_row(p, a)}" for i, p in enumerate(window))
    page = list(context.fitting(units, reserve=len(_more(total, offset=total))))
    if a.offset + len(page) < total:
        page.append(_more(total - a.offset - len(page), offset=a.offset + len(page)))
    return page


def _row(path: Path, a: Glob.Arguments) -> str:
    """Return ``path`` as a row: absolute, with its size and time under ``long``."""
    # ``absolute``, not ``resolve``: a link is listed under its own path.
    name = str(path.absolute())
    return ls_row(path, name=name) if a.long else name


def _more(entries: int, *, offset: int) -> str:
    """Return the note saying ``entries`` were cut, and where to resume."""
    return f"\n... ({entries:,} more; pass offset={offset} to continue)"
