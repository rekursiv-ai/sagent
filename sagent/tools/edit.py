"""Edit: replace exact text in a file.

An unread file may be edited: the exact ``old_string`` is the check that the
model knows what it is replacing. A file read and then changed may not.

Typical usage example:

  streaming_result = Edit().call(context, {"file_path": "a.py", "old_string": "x = 1", "new_string": "x = 2"})
  result = await settled(streaming_result)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.resources import files
from typing import TYPE_CHECKING, Annotated, cast

import asyncio
import re

from sagent.providers.login import file_lock
from sagent.session.types import EditResult, Splice
from sagent.tools.command_analyzer import (
    Invocation,
    operands,
    replaceable,
    resolve,
)
from sagent.tools.display import DisplaySpec, ToolDisplay
from sagent.tools.types import (
    ToolOutput,
    ToolStreamingResult,
    json_schema,
    parse,
    whole_output,
)
from sagent.tools.write import file_lock_path, write_file


if TYPE_CHECKING:
    from collections.abc import Mapping

    from treekle import PlainTree

    from sagent.tools.types import ToolContext


__all__ = [
    "Edit",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class Edit:
    """Replace exact text in a file."""

    @dataclass(frozen=True, slots=True, kw_only=True)
    class Arguments:
        """What the model passes."""

        file_path: Annotated[str, "Path to the file."]
        old_string: Annotated[str, "Text to find."]
        new_string: Annotated[str, "Replacement text."]
        replace_all: Annotated[bool, "Replace all occurrences."] = False

    name = "Edit"
    description = (files(__package__) / "assets" / "edit.md").read_text(
        encoding="utf-8",
    )
    schema = json_schema(Arguments)

    display: ToolDisplay = field(default=ToolDisplay(output=DisplaySpec()))

    def summary(self, arguments: Mapping[str, PlainTree]) -> str:
        """Return ``Edit <path>``."""
        return f"Edit {arguments.get('file_path', '')}"

    def serialize_key(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> str | None:
        """Return the file, so calls on one file run in order."""
        return str(context.path(str(arguments.get("file_path", ""))).resolve())

    def from_shell(
        self,
        invocation: Invocation,
    ) -> Mapping[str, PlainTree] | str | None:
        """Return the Edit call for ``sed -i 's/OLD/NEW/[g]' FILE``.

        Only literal text: ``sed`` matches a regular expression and Edit exact
        text, so a script using regex syntax stays a shell command.

        Args:
          invocation: One simple command, with its pipeline.

        Returns:
          arguments: The Edit call making the same change, or ``None``.

        """
        # Bash fills in ``$x`` before sed sees it; Edit would write it as is.
        if invocation.expands or not replaceable(invocation, exes=frozenset({"sed"})):
            return None
        args = invocation.args
        # Any flag but the one that edits in place, ``-e`` and ``-f`` among them,
        # refuses the call, so no flag's value is ever taken for an operand.
        flags = [a for a in args if a.startswith("-")]
        given = operands(args)
        if flags not in (["-i"], ["--in-place"]) or len(given) != 2:
            return None
        script = re.fullmatch(r"s/([^/]*)/([^/]*)/(g?)", given[0])
        if script is None:
            return None
        files = given[1:]
        # Every group takes part in a match, so none is ``None``.
        old, new, every = cast("tuple[str, str, str]", script.groups())
        # Regex syntax in OLD, or a back-reference in NEW, is not literal.
        if not old or re.search(r"[.*\[\]^$\\]", old) or re.search(r"[&\\]", new):
            return None
        target = resolve(invocation.cwd, files[0])
        if target is None or any(c in target for c in "*?[$`"):
            return None
        return {
            "file_path": target,
            "old_string": old,
            "new_string": new,
            "replace_all": bool(every),
        }

    def call(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> ToolStreamingResult[EditResult, Splice | str, str]:
        """Replace ``old_string`` with ``new_string`` in ``file_path``.

        ``old_string`` must occur once, unless ``replace_all``.

        Args:
          context: The calling agent's state.
          arguments: What the model passed, as ``Edit.Arguments``.

        Returns:
          streaming_result: The line the model reads, then each replacement made; or why
            none was.

        """
        return whole_output(self._run(context, arguments))

    async def _run(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> ToolOutput[EditResult, Splice | str, str]:
        a = parse(arguments, Edit.Arguments)
        async with file_lock(file_lock_path(context.path(a.file_path))):
            return await asyncio.to_thread(_edit, context, a)


def _edit(
    context: ToolContext,
    a: Edit.Arguments,
) -> ToolOutput[EditResult, Splice | str, str]:
    """Make the replacement ``a`` asks for, if the file is as the model last saw it."""
    path = context.path(a.file_path)
    # ``str.replace("", ...)`` inserts between every character.
    if not a.old_string:
        return ToolOutput(
            stderr=("old_string is empty.",),
            result=EditResult(exit_code=1),
        )
    if not path.is_file():
        return ToolOutput(
            stderr=(f"{path} is not a file.",),
            result=EditResult(exit_code=1),
        )
    if context.files.changed(path):
        return ToolOutput(
            stderr=(
                f"{path} changed since it was read, by the user, a linter, or another agent. Read it again before editing it.",
            ),
            result=EditResult(exit_code=1),
        )
    try:
        raw = path.read_bytes().decode()
    except UnicodeDecodeError:
        return ToolOutput(
            stderr=(f"{path} is not UTF-8. Use Bash with an explicit decoder.",),
            result=EditResult(exit_code=1),
        )
    # Read shows a CRLF file with ``\n``, so the strings the model quotes have
    # ``\n``; the file keeps its ``\r\n``. A file of mixed endings is matched as is.
    crlf = "\r\n" in raw and raw.count("\r\n") == raw.count("\n")
    text, old, new = raw, a.old_string, a.new_string
    if crlf:
        text, old, new = (s.replace("\r\n", "\n") for s in (raw, old, new))
    count = text.count(old)
    if count == 0:
        return ToolOutput(
            stderr=("old_string not found in file.",),
            result=EditResult(exit_code=1),
        )
    if count > 1 and not a.replace_all:
        return ToolOutput(
            stderr=(
                f"old_string found {count} times. Set replace_all, or give more context to make it unique.",
            ),
            result=EditResult(exit_code=1),
        )
    # Every occurrence: without ``replace_all`` there is exactly one.
    edited = text.replace(old, new)
    write_file(path, (edited.replace("\n", "\r\n") if crlf else edited).encode())
    context.files.saw(path)
    return ToolOutput(
        stdout=(
            f"Edited {path}.",
            *(
                Splice(before=old, after=new, start=start, count=old.count("\n") + 1)
                for start in _lines_at(text, old)
            ),
        ),
        result=EditResult(path=str(path)),
    )


def _lines_at(text: str, needle: str) -> list[int]:
    """Return the one-based line of each ``needle`` in ``text``."""
    lines: list[int] = []
    line, scanned, at = 1, 0, text.find(needle)
    while at >= 0:
        line += text.count("\n", scanned, at)
        lines.append(line)
        scanned = at
        at = text.find(needle, at + len(needle))
    return lines
