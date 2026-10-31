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
from sagent.tools.types import Arg, json_schema, parse


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
            tuple[str, ...] | str,
            "File or directory to search, or several in order (rg PATH...)."
            " Defaults to the cwd.",
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
            'Defaults to "files_with_matches"; "files_without_match" is rg'
            " --files-without-match.",
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
        invert: Annotated[
            bool,
            "Select lines that do not match (rg -v).",
            Arg(key="-v"),
        ] = False
        word: Annotated[bool, "Match whole words only (rg -w).", Arg(key="-w")] = False
        whole_line: Annotated[
            bool,
            "Match whole lines only (rg -x).",
            Arg(key="-x"),
        ] = False
        fixed: Annotated[
            bool,
            "Read pattern as literal text, not a regex (rg -F).",
            Arg(key="-F"),
        ] = False
        only_matching: Annotated[
            bool,
            'One entry per match, holding just the match (rg -o); "count" counts'
            " matches.",
            Arg(key="-o"),
        ] = False
        max_count: Annotated[
            int,
            "Stop each file after N selected lines (rg -m). Omit or 0 for no cap.",
            Arg(minimum=0, key="-m"),
        ] = 0
        text: Annotated[
            bool,
            "Search binary files as text (rg -a).",
            Arg(key="-a"),
        ] = False
        search_ignored: Annotated[
            bool,
            "Also search files .gitignore/.ignore name (rg --no-ignore).",
            Arg(key="no_ignore"),
        ] = False
        follow: Annotated[bool, "Follow symlinks (rg -L)."] = False
        max_depth: Annotated[
            int,
            "Descend at most N directory levels (rg --max-depth). Omit or 0: no cap.",
            Arg(minimum=0),
        ] = 0

    name = "Grep"
    description = (files(__package__) / "assets" / "grep.md").read_text(
        encoding="utf-8",
    )
    schema = json_schema(Arguments)

    display: ToolDisplay = field(default=ToolDisplay())

    max_line_chars: int = 500
    """Longest line shown, cut around its first match."""

    max_result_chars: int = 20_000
    """How much text one result may hold, if below the context's."""

    def summary(self, arguments: Mapping[str, PlainTree]) -> str:
        """Return ``Grep '<pattern>'``, with ``in <path>`` when not the cwd."""
        path = arguments.get("path")
        if isinstance(path, list):
            path = ", ".join(map(str, path))
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
            deny = frozenset({"-q", "--quiet"})
            common = frozenset(
                {"-A", "-B", "-C", "-m", "--max-count", "-e", "-f", "--color"},
            )
            takes_value = common | (
                {"-d", "-D", "--label"}
                if invocation.exe == "grep"
                else {"-g", "--glob", "-t", "-d", "--max-depth"}
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
        parsed = _grep_arguments(args, exe=exe)
        if parsed is None or path is None:
            return refusal
        arguments, patterns, named = parsed
        fed = source is not None
        # A file named beside the fed or listed ones is searched as well.
        if fed and named:
            return refusal
        # Without a file or ``-r``, grep reads its input, as both do for ``-``.
        if "-" in named or (
            exe == "grep" and not fed and not named and not _flagged(args, "rR")
        ):
            return None
        if _differs(arguments, patterns=patterns):
            return None
        if (
            exe == "grep"
            and "-F" not in arguments
            and not _flagged(args, "EP")
            and any(re.search(r"[+?|(){}]", p) for p in patterns)
        ):
            # A basic regular expression reads these as themselves, Grep as syntax.
            return None
        resolved = (
            [path] if fed else [resolve(invocation.cwd, n) for n in named or [""]]
        )
        targets = [t for t in resolved if t is not None]
        if len(targets) < len(resolved) or any(
            c in t for t in targets for c in "*?[$`"
        ):
            return refusal
        if len(targets) > 1:
            arguments["path"] = targets
        elif targets[0]:
            arguments["path"] = targets[0]
        arguments.update(extra)
        # ``wc -l`` counts printed lines, which Grep's rows are; no count stands
        # in for them, as a binary file's matches print as one note.
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
        named = (a.path,) if isinstance(a.path, str) else a.path
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
            text=a.text,
            invert=a.invert,
            word=a.word,
            whole_line=a.whole_line,
            fixed=a.fixed,
            only_matching=a.only_matching,
            max_count=a.max_count,
            search_ignored=a.search_ignored,
            follow=a.follow,
            max_depth=a.max_depth,
        )
        try:
            rows = await asyncio.to_thread(
                grep_rows,
                [context.path(p or ".") for p in named],
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


# Flag spellings and their Grep argument, per command; a ``None`` value drops the
# argument, as ``rg -s`` undoes ``-i``. Each is a flag the command shares no other
# meaning for: ``-L`` follows links in ``rg`` but lists files in ``grep``, and
# ``rg -r``/``-E``/``-h`` replace, set an encoding, and print help.
def _flags(exe: str) -> dict[str, tuple[str, PlainTree] | tuple[()]]:
    """Return each value-less flag ``exe`` takes that Grep can say."""
    shared: dict[str, tuple[str, PlainTree] | tuple[()]] = {
        "-l": ("output_mode", "files_with_matches"),
        "--files-with-matches": ("output_mode", "files_with_matches"),
        "--files-without-match": ("output_mode", "files_without_match"),
        "-c": ("output_mode", "count"),
        "--count": ("output_mode", "count"),
        "-i": ("-i", True),
        "--ignore-case": ("-i", True),
        "-P": ("pcre", True),
        "-v": ("-v", True),
        "--invert-match": ("-v", True),
        "-w": ("-w", True),
        "--word-regexp": ("-w", True),
        "-x": ("-x", True),
        "--line-regexp": ("-x", True),
        "-F": ("-F", True),
        "--fixed-strings": ("-F", True),
        "-o": ("-o", True),
        "--only-matching": ("-o", True),
        "-a": ("-a", True),
        "--text": ("-a", True),
        "-n": (),
        "-H": (),
    }
    if exe == "grep":
        return {
            **shared,
            "-r": (),
            "-R": ("follow", True),
            "--dereference-recursive": ("follow", True),
            "-L": ("output_mode", "files_without_match"),
            "-E": (),
            "-h": (),
            "-s": (),
        }
    return {
        **shared,
        "-L": ("follow", True),
        "--follow": ("follow", True),
        "-u": ("no_ignore", True),
        "--no-ignore": ("no_ignore", True),
        "-N": ("-n", False),
        "-s": ("-i", None),
        "-.": (),
    }


# The pattern is the first operand unless ``-e`` names it; the other operands are
# the paths, relative to the shell.
# Returns ``None`` when an argument has no Grep counterpart.
def _grep_arguments(
    args: tuple[str, ...],
    *,
    exe: str,
) -> tuple[dict[str, PlainTree], list[str], list[str]] | None:
    """Return ``grep`` or ``rg`` arguments as Grep's, the patterns, and the paths."""
    flags = _flags(exe)
    arguments: dict[str, PlainTree] = {"output_mode": "content"}
    operand: list[str] = []
    patterns: list[str] = []
    globs: list[str] = []
    i = 0
    while i < len(args):
        arg = args[i]
        name, eq, given = arg.partition("=")
        valued = _valued(name, exe=exe)
        if arg == "-" or not arg.startswith("-"):
            operand.append(arg)
        elif valued:
            if not eq:
                if i + 1 == len(args):
                    return None
                given, i = args[i + 1], i + 1
            if valued == "pattern":
                patterns.append(given)
            elif valued == "glob":
                globs.append(given)
            elif valued == "--include":
                arguments["glob"] = given
            elif valued == "--exclude":
                arguments["exclude"] = given
            elif given.isdigit():
                arguments[valued] = int(given)
            else:
                return None
        elif arg.startswith("--"):
            if arg not in flags:
                return None
            _apply(arguments, flags[arg])
        else:
            # ``rg -uuu`` also searches binary files, which ``-a`` does otherwise.
            if exe == "rg" and arg.count("u") > 2:
                return None
            for letter in arg[1:]:
                if f"-{letter}" not in flags:
                    return None
                _apply(arguments, flags[f"-{letter}"])
        i += 1
    if not _place_globs(arguments, globs):
        return None
    if not patterns:
        patterns, operand = operand[:1], operand[1:]
    if not patterns:
        return None
    arguments["pattern"] = "|".join(patterns)
    return arguments, patterns, operand


def _valued(name: str, *, exe: str) -> str:
    """Return the Grep argument flag ``name`` sets from its value; ``""`` if none."""
    if name in {"-e", "--regexp"}:
        return "pattern"
    if name in {"-A", "-B", "-C"}:
        return name
    if name in {"-m", "--max-count"}:
        return "-m"
    if exe == "grep" and name in {"--include", "--exclude"}:
        return name
    if exe == "rg" and name in {"-g", "--glob"}:
        return "glob"
    if exe == "rg" and name in {"-d", "--max-depth"}:
        return "max_depth"
    return ""


def _apply(
    arguments: dict[str, PlainTree],
    flag: tuple[str, PlainTree] | tuple[()],
) -> None:
    """Set the Grep argument ``flag`` names; a ``None`` value removes it."""
    if not flag:
        return
    key, value = flag
    if value is None:
        arguments.pop(key, None)
    else:
        arguments[key] = value


def _place_globs(arguments: dict[str, PlainTree], globs: list[str]) -> bool:
    """Set ``rg -g`` globs as Grep's; ``False`` if more than one per side."""
    keep = [g for g in globs if not g.startswith("!")]
    skip = [g.removeprefix("!") for g in globs if g.startswith("!")]
    if len(keep) > 1 or len(skip) > 1:
        return False
    if keep:
        arguments["glob"] = keep[0]
    if skip:
        arguments["exclude"] = skip[0]
    return True


# Grep answers as ripgrep does, so a ``grep`` whose output differs only in
# GNU detail -- ``-c`` zero rows, ``-o`` context, ``-m`` context markers -- is
# still Grep's; these differ in what is searched for.
def _differs(arguments: Mapping[str, PlainTree], *, patterns: list[str]) -> bool:
    """Return whether Grep cannot say what ``arguments`` search for."""
    return (
        # ``0`` is no cap to Grep and no output to ``-m``/``--max-depth``.
        arguments.get("-m") == 0
        or arguments.get("max_depth") == 0
        # Text joined by ``|`` would be read as one literal.
        or (arguments.get("-F") is True and len(patterns) > 1)
        # Grep refuses ``-o`` with ``-v``, which ripgrep answers inconsistently.
        or (arguments.get("-o") is True and arguments.get("-v") is True)
    )


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
