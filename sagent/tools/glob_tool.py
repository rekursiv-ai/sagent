"""Glob: the files under a directory whose paths match a pattern.

Unlike List, Glob walks the whole tree, skips what ``.gitignore`` ignores,
and returns absolute paths of files only.

Typical usage example:

  result = await Glob().call(context, {"pattern": "**/*.py", "sort": "mtime_desc"})
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.resources import files
from typing import TYPE_CHECKING, Annotated

import asyncio

from sagent.lib.files.glob import GlobError, glob
from sagent.session.types import UncategorizedToolResult
from sagent.tools.command_analyzer import (
    Invocation,
    bounding_sink,
    line_count,
    replaceable,
    resolve,
)
from sagent.tools.display import ToolDisplay
from sagent.tools.list import Sort, ls_row, sort_paths
from sagent.tools.types import Arg, parse, schema_of


if TYPE_CHECKING:
    from collections.abc import Mapping

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
            "Result ordering. Default 'name'; 'mtime_desc' is newest first.",
        ] = "name"
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
    schema = schema_of(Arguments)

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

    async def call(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> UncategorizedToolResult:
        """Find files under ``path`` matching ``pattern``.

        An absolute ``pattern`` ignores ``path``.

        Args:
          context: The calling agent's state.
          arguments: What the model passed, as ``Glob.Arguments``.

        Returns:
          result: One absolute path per line, with a note to resume when cut.

        """
        return await asyncio.to_thread(_glob, context, parse(arguments, Glob.Arguments))


def _glob(context: ToolContext, a: Glob.Arguments) -> UncategorizedToolResult:
    """Find the files ``a`` asks for."""
    try:
        matches = glob(context.path(a.path or "."), a.pattern)
    except GlobError as e:
        return UncategorizedToolResult(call_id="", content=str(e), is_error=True)
    if not matches:
        return UncategorizedToolResult(call_id="", content="(no matches)")
    sort_paths(matches, a.sort)
    end = a.offset + a.max_results if a.max_results else None
    # ``absolute``, not ``resolve``: a link is listed under its own path.
    rows = (
        (ls_row(p, name=str(p.absolute())) if a.long else str(p.absolute())) + "\n"
        for p in matches[a.offset : end]
    )
    note = f"\n... ({len(matches):,} more; pass offset={len(matches)} to continue)"
    body, kept = context.fit(rows, reserve=len(note))
    body = body.rstrip("\n") or "(no matches in this window)"
    if a.offset + kept < len(matches):
        body += (
            f"\n... ({len(matches) - a.offset - kept:,} more;"
            f" pass offset={a.offset + kept} to continue)"
        )
    return UncategorizedToolResult(call_id="", content=body)
