"""Grep: search file contents with ripgrep.

Matches come back one per line in every output mode, so ``offset``,
``keep_first``, and ``keep_last`` page them alike. A long line is cut around
its first match, which is the part a search exists to show.

Typical usage example:

  result = await Grep().call(context, {"pattern": "def main", "-C": 2})
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.resources import files
from typing import TYPE_CHECKING, Annotated

import asyncio
import re

from treekle import PlainTree

from sagent.lib.files.grep import GrepError, OutputMode, Query, Row, grep_rows
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

    from sagent.tools.types import ToolContext


__all__ = [
    "Grep",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class Grep:
    """Search file contents with ripgrep."""

    @dataclass(frozen=True, slots=True, kw_only=True)
    class Arguments:
        """What the model passes; ripgrep's own spellings, such as ``-B``, as keys."""

        pattern: Annotated[
            str,
            "The regular expression to search file contents for.",
        ]
        path: Annotated[
            str,
            "File or directory to search (rg PATH). Defaults to the cwd.",
        ] = "."
        glob: Annotated[
            str,
            'Files to search, e.g. "*.js" or "*.{ts,tsx}" (rg --glob).',
        ] = ""
        file_type: Annotated[
            str,
            "File type to search (rg --type): js, py, rust, go, ...",
            Arg(key="type"),
        ] = ""
        exclude: Annotated[
            str,
            'Files to skip, e.g. "*.test.py" (rg --glob "!PAT").',
        ] = ""
        output_mode: Annotated[
            OutputMode,
            'Defaults to "files_with_matches".',
        ] = "files_with_matches"
        before: Annotated[
            int,
            'Lines before each match (rg -B); "content" mode only.',
            Arg(minimum=0, key="-B"),
        ] = 0
        after: Annotated[
            int,
            'Lines after each match (rg -A); "content" mode only.',
            Arg(minimum=0, key="-A"),
        ] = 0
        around: Annotated[int, "Alias for context.", Arg(minimum=0, key="-C")] = 0
        context: Annotated[
            int,
            'Lines around each match (rg -C); "content" mode only.',
            Arg(minimum=0),
        ] = 0
        ignore_case: Annotated[bool, "Ignore case (rg -i).", Arg(key="-i")] = False
        line_numbers: Annotated[
            bool,
            "Show line numbers (rg -n). Defaults to true.",
            Arg(key="-n"),
        ] = True
        keep_first: Annotated[
            int,
            "Keep the first N entries. Omit or 0 to keep as many as fit.",
            Arg(minimum=0),
        ] = 0
        keep_last: Annotated[
            int,
            "Keep the last N entries instead; overrides keep_first and offset.",
            Arg(minimum=0),
        ] = 0
        offset: Annotated[
            int,
            "Skip the first N entries before keep_first.",
            Arg(minimum=0),
        ] = 0
        multiline: Annotated[
            bool,
            "Let . match newlines and patterns span lines (rg -U --multiline-dotall).",
        ] = False
        pcre: Annotated[
            bool,
            "Use PCRE2 (rg -P): lookaround, backrefs. Default Rust regex.",
        ] = False

    name = "Grep"
    description = (files(__package__) / "assets" / "grep.md").read_text(
        encoding="utf-8",
    )
    schema = schema_of(Arguments)

    display: ToolDisplay = field(default=ToolDisplay())

    max_line_chars: int = 500
    """Longest line shown, cut around its first match."""

    max_result_chars: int = 20_000
    """How much text one result may hold, if below the context's."""

    def summary(self, arguments: Mapping[str, PlainTree]) -> str:
        """Return ``Grep '<pattern>'``, with ``in <path>`` when not the cwd."""
        path = arguments.get("path")
        where = f" in {path}" if path not in {None, "", "."} else ""
        return f"Grep {str(arguments.get('pattern', ''))!r}{where}"

    def serialize_key(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> str | None:
        """Return ``None``: searching changes nothing."""
        del context, arguments
        return None

    def from_shell(
        self,
        invocation: Invocation,
    ) -> Mapping[str, PlainTree] | str | None:
        """Return the Grep call for ``grep`` or ``rg`` over files.

        Three shapes search files: ``grep p f``, ``cat f | grep p``, and
        ``find ROOT -name G | xargs grep p``.

        Args:
          invocation: One simple command, with its pipeline.

        Returns:
          arguments: The Grep call for the same search; a message for flags no
            call matches; or ``None`` when the command searches no file, as
            ``git log | grep fix`` does, or needs the shell, as ``grep -q``.

        """
        searches = frozenset({"grep", "rg"})
        refusal = "Search with Grep: pattern, path, glob, -i, -C, output_mode."
        source = invocation.piped_from
        # Bash fills in ``$x`` before the search runs; Grep would take it as written.
        if invocation.expands or (source is not None and source.expands):
            return None
        if invocation.exe == "xargs":
            search = _xargs_search(invocation, searches=searches)
            if search is None:
                return None
            exe, args, root, name = search
            path = resolve(invocation.cwd, root)
            extra: dict[str, PlainTree] = {"glob": name} if name else {}
        else:
            deny = frozenset({"-v", "--invert-match", "-q", "--quiet"})
            common = frozenset({"-A", "-B", "-C", "-m", "-e", "-f", "--color"})
            takes_value = common | (
                {"-d", "-D", "--label"} if invocation.exe == "grep" else {"-g", "-t"}
            )
            if not replaceable(
                invocation,
                exes=searches,
                deny=deny,
                takes_value=takes_value,
                fed=True,
                counted=True,
            ):
                return None
            exe, args, path = invocation.exe, invocation.args, ""
            extra = dict[str, PlainTree]()
            if source is not None:
                # ``cat f | grep p``; a window, as ``head -n 20 f``, searches
                # part of the file, which Grep cannot.
                if source.exe not in {"cat", "head", "tail", "sed"}:
                    # ``git log | grep fix`` searches a stream, not a file.
                    return None
                files = operands(source.args)
                if source.exe != "cat" or source.args != files or len(files) != 1:
                    return refusal
                path = resolve(invocation.cwd, files[0])
        parsed = _grep_arguments(args)
        if parsed is None or path is None:
            return refusal
        arguments, patterns = parsed
        named = arguments.pop("path", None)
        fed = source is not None
        # A file named beside the fed or listed ones is searched as well.
        if fed and named is not None:
            return refusal
        # Without a file or ``-r``, grep reads its input, as both do for ``-``.
        if named == "-" or (
            exe == "grep" and not fed and named is None and not _flagged(args, "rR")
        ):
            return None
        if (
            exe == "grep"
            and not _flagged(args, "EP")
            and any(re.search(r"[+?|(){}]", p) for p in patterns)
        ):
            # A basic regular expression reads these as themselves, Grep as syntax.
            return None
        target = path if fed else resolve(invocation.cwd, str(named or ""))
        if target is None or any(c in target for c in "*?[$`"):
            return refusal
        if target:
            arguments["path"] = target
        arguments.update(extra)
        for sink in invocation.downstream():
            if sink.exe == "wc":
                arguments["output_mode"] = "count"
        sink = bounding_sink(invocation)
        if sink is not None:
            count = line_count(sink.args)
            if count is None:
                return refusal
            arguments["keep_last" if sink.exe == "tail" else "keep_first"] = count
        return arguments

    async def call(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> UncategorizedToolResult:
        """Search the files under ``path`` for ``pattern``.

        Args:
          context: The calling agent's state.
          arguments: What the model passed, as ``Grep.Arguments``.

        Returns:
          result: One entry per line, with a note to resume when cut.

        """
        a = parse(arguments, Grep.Arguments)
        around = max(a.around, a.context)
        query = Query(
            pattern=a.pattern,
            glob=a.glob,
            exclude=a.exclude,
            file_type=a.file_type,
            output_mode=a.output_mode,
            context_before=max(a.before, around),
            context_after=max(a.after, around),
            case_insensitive=a.ignore_case,
            line_numbers=a.line_numbers,
            multiline=a.multiline,
            pcre=a.pcre,
        )
        try:
            rows = await asyncio.to_thread(
                grep_rows,
                context.path(a.path or "."),
                query,
            )
        except GrepError as e:
            return UncategorizedToolResult(call_id="", content=str(e), is_error=True)
        # ``keep_last`` starts the window at the tail; the resume note names it.
        start = max(0, len(rows) - a.keep_last) if a.keep_last else a.offset
        end = start + a.keep_first if a.keep_first and not a.keep_last else None
        window = [self._clip(row) + "\n" for row in rows[start:end]]
        if not window:
            # Past the last entry, saying none matched would be false.
            content = "(no matches in this window)" if rows else "(no matches)"
            return UncategorizedToolResult(call_id="", content=content)
        note = f"\n... ({len(window):,} more entries; pass offset={start + len(window)} to continue)"
        body, kept = context.fit(window, reserve=len(note), cap=self.max_result_chars)
        body = body.rstrip("\n")
        if kept < len(window):
            body += (
                f"\n... ({len(window) - kept:,} more entries;"
                f" pass offset={start + kept} to continue)"
            )
        return UncategorizedToolResult(call_id="", content=body)

    # The ``path:line:`` prefix is kept whole: it is how the row is found.
    def _clip(self, row: Row) -> str:
        """Cut a row to ``max_line_chars`` around its first match."""
        cut = row.prefix_end
        text = row.text[cut:]
        if len(text) <= self.max_line_chars:
            return row.text
        center = max(0, row.column - cut)
        start = max(
            0,
            min(center - self.max_line_chars // 2, len(text) - self.max_line_chars),
        )
        end = start + self.max_line_chars
        head = f"[{start:,} chars] " if start else ""
        tail = f" [{len(text) - end:,} chars]" if end < len(text) else ""
        return f"{row.text[:cut]}{head}{text[start:end]}{tail}"


# The pattern is the first operand unless ``-e`` names it; one more operand is the path,
# returned under ``path`` relative to the shell.
# Returns ``None`` when an argument has no Grep counterpart.
def _grep_arguments(
    args: tuple[str, ...],
) -> tuple[dict[str, PlainTree], list[str]] | None:
    """Return ``grep`` or ``rg`` arguments as Grep's, and the patterns they join."""
    arguments: dict[str, PlainTree] = {"output_mode": "content"}
    operand: list[str] = []
    patterns: list[str] = []
    i = 0
    while i < len(args):
        arg = args[i]
        value = args[i + 1] if i + 1 < len(args) else None
        if arg == "-" or not arg.startswith("-"):
            operand.append(arg)
        elif arg in {"-e", "--regexp"} and value is not None:
            patterns.append(value)
            i += 1
        elif arg.partition("=")[0] in {"--include", "--exclude"}:
            name, eq, given = arg.partition("=")
            if not eq:
                if value is None:
                    return None
                given, i = value, i + 1
            arguments["glob" if name == "--include" else "exclude"] = given
        elif arg in {"-A", "-B", "-C"} and value is not None and value.isdigit():
            arguments[arg] = int(value)
            i += 1
        elif arg.startswith("--") or set(arg[1:]) - set("rRlcniEPhHs"):
            return None
        else:
            for letter in arg[1:]:
                if letter == "l":
                    arguments["output_mode"] = "files_with_matches"
                elif letter == "c":
                    arguments["output_mode"] = "count"
                elif letter == "i":
                    arguments["-i"] = True
                elif letter == "P":
                    arguments["pcre"] = True
        i += 1
    if not patterns:
        patterns, operand = operand[:1], operand[1:]
    if not patterns or len(operand) > 1:
        return None
    arguments["pattern"] = "|".join(patterns)
    if operand:
        arguments["path"] = operand[0]
    return arguments, patterns


def _flagged(args: tuple[str, ...], letters: str) -> bool:
    """Return whether a short-flag cluster in ``args`` holds one of ``letters``."""
    return any(
        a.startswith("-") and not a.startswith("--") and set(a[1:]) & set(letters)
        for a in args
    )


# ``find ROOT [-type f] [-name G] [-print0] | xargs [-0] [-r] grep ...``: the
# ``find`` only lists what to search. ``-iname`` and ``-type d`` change what is
# searched in ways Grep's ``glob`` and ``path`` cannot.
def _xargs_search(
    invocation: Invocation,
    *,
    searches: frozenset[str],
) -> tuple[str, tuple[str, ...], str, str] | None:
    """Return the search, its arguments, root, and name glob of ``find | xargs grep``."""
    source = invocation.piped_from
    if source is None or source.exe != "find" or source.piped_from is not None:
        return None
    # As for a bare search: what follows may only cut or count, so that a
    # ``| xargs rm`` after it is never dropped with the shell command.
    if not replaceable(
        invocation,
        exes=frozenset({"xargs"}),
        bare=True,
        fed=True,
        counted=True,
    ):
        return None
    args = list(invocation.args)
    while args and args[0] in {"-0", "--null", "-r", "--no-run-if-empty"}:
        args.pop(0)
    if not args or args[0] not in searches:
        return None
    root, name = ".", ""
    rest = list(source.args)
    while rest:
        arg = rest.pop(0)
        if arg in {"-print", "-print0"}:
            continue
        # Two names must both match, which one glob cannot say.
        if arg == "-name" and rest and not name:
            name = rest.pop(0)
        elif arg == "-type" and rest[:1] == ["f"]:
            rest.pop(0)
        elif arg.startswith("-") or root != ".":
            return None
        else:
            root = arg
    return args[0], tuple(args[1:]), root, name
