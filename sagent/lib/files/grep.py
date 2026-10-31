"""Search file contents as ripgrep does: ``rg`` when installed, else in Python.

Both backends take one :class:`Query` and return the same lines -- one entry per
match, per context line, per ``--`` group separator, per counted file, or per
listed file -- each as a :class:`Row` locating its first match. A directory is
walked in path order with ``.git``/``.svn``/``.hg`` skipped, ``.gitignore``d
trees skipped unless ``search_ignored``, symlinks followed only under
``follow``, hidden entries searched, and binary (NUL-bearing) or unreadable
files skipped. A file named outright is searched whatever its name, type or
ignore status, as ripgrep searches it, and one that is missing or unreadable is
an error. The Python backend walks with ``rignore``, which binds the Rust
``ignore`` library ripgrep itself is built on.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from stat import S_ISREG
from typing import TYPE_CHECKING, Final, Literal

import codecs
import dataclasses
import itertools
import os
import re
import shutil
import subprocess

import rignore


if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence


OutputMode = Literal["content", "files_with_matches", "files_without_match", "count"]

TYPE_GLOBS: Final[dict[str, tuple[str, ...]]] = {
    "py": ("*.py",),
    "js": ("*.js", "*.jsx", "*.mjs"),
    "ts": ("*.ts", "*.tsx"),
    "rust": ("*.rs",),
    "go": ("*.go",),
    "java": ("*.java",),
    "c": ("*.c", "*.h"),
    "cpp": ("*.cpp", "*.hpp", "*.cc", "*.cxx"),
    "md": ("*.md",),
    "yaml": ("*.yaml", "*.yml"),
    "json": ("*.json",),
    "toml": ("*.toml",),
    "html": ("*.html", "*.htm"),
    "css": ("*.css",),
    "sh": ("*.sh", "*.bash"),
}
"""The ``file_type`` names both backends accept, and the files each selects."""

_NEWLINE_HINT: Final = (
    "pattern references a newline but multiline is off. "
    'Pass multiline=true to match across lines (literal "\\n" '
    "or `.` spanning newlines)."
)

# Field separators ripgrep is told to print; no path or text line holds them.
_MATCH: Final = b"\x01"
_CONTEXT: Final = b"\x02"

# What ripgrep's own engine says when only PCRE2 can run the pattern.
_NEEDS_PCRE2: Final = ("look-around", "backreferences are not supported")

# Linux caps one argv near 2 MiB and each string at 128 KiB; a few thousand
# named files stay well under both when split here.
ARGV_CHARS: Final = 100_000
"""Most characters of named paths one ``rg`` run takes."""


class GrepError(Exception):
    """A query no backend can answer; the message is written for the caller."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Query:
    """One search, in ripgrep's terms.

    Attributes:
      pattern: Regex to find. Ripgrep's syntax, falling back to PCRE2 when it
        needs lookaround or backreferences (``rg --engine auto``).
      glob: Only walked files matching this glob (``rg --glob``); brace sets
        allowed.
      exclude: Skip walked files matching this glob (``rg --glob !PAT``).
      file_type: A :data:`TYPE_GLOBS` name (``rg --type``).
      output_mode: Matching lines, files with or without a match, or per-file
        counts (of matches under ``only_matching``, else of lines).
      context_before: Lines shown before each match (``rg -B``).
      context_after: Lines shown after each match (``rg -A``).
      case_insensitive: ``rg -i``.
      line_numbers: ``rg -n``.
      multiline: Let a match span lines (``rg -U --multiline-dotall``).
      pcre: PCRE2 syntax only (``rg -P``); needs ``rg``.
      text: Search binary files as text (``rg --text``).
      invert: Select the lines that do not match (``rg -v``).
      word: Match only where no word character borders the match (``rg -w``).
      whole_line: Match only whole lines (``rg -x``).
      fixed: Read ``pattern`` as text, not a regex (``rg -F``).
      only_matching: One row per match, holding just the match (``rg -o``).
      max_count: Stop each file after this many selected lines; 0 for no cap
        (``rg -m``).
      search_ignored: Also walk what ignore files name (``rg --no-ignore``).
      follow: Follow symlinks in a walk (``rg -L``).
      max_depth: Walk at most this many levels below a named directory; 0 for
        no cap (``rg --max-depth``).

    """

    pattern: str
    glob: str = ""
    exclude: str = ""
    file_type: str = ""
    output_mode: OutputMode = "files_with_matches"
    context_before: int = 0
    context_after: int = 0
    case_insensitive: bool = False
    line_numbers: bool = True
    multiline: bool = False
    pcre: bool = False
    text: bool = False
    invert: bool = False
    word: bool = False
    whole_line: bool = False
    fixed: bool = False
    only_matching: bool = False
    max_count: int = 0
    search_ignored: bool = False
    follow: bool = False
    max_depth: int = 0

    def __post_init__(self) -> None:
        """Refuse a negative count, which each backend would reject differently."""
        for name, value in (
            ("context_before", self.context_before),
            ("context_after", self.context_after),
            ("max_count", self.max_count),
            ("max_depth", self.max_depth),
        ):
            if value < 0:
                raise ValueError(f"{name} must be at least 0, got {value}.")


@dataclass(frozen=True, slots=True, kw_only=True)
class Row:
    """One output line and where its first match begins.

    Attributes:
      text: The line as ``rg --no-heading`` prints it, ``name:N:`` prefix included.
      prefix_end: Character offset in ``text`` where the line's own text begins,
        after its ``name:N:``/``name-N-`` prefix; 0 for a row that is not a line.
      column: Character offset in ``text`` where the line's first match begins;
        ``-1`` when no match begins there: context, ``--``, a count, a file
        name, a binary-file note, an inverted line, or a line a multiline match
        continues onto.

    """

    text: str
    prefix_end: int = 0
    column: int = -1


def grep(
    paths: Path | Sequence[Path],
    query: Query,
    /,
    *,
    timeout_sec: float = 30,
) -> list[str]:
    """Run ``query`` over ``paths`` and return the text of :func:`grep_rows`."""
    return [row.text for row in grep_rows(paths, query, timeout_sec=timeout_sec)]


def grep_rows(
    paths: Path | Sequence[Path],
    query: Query,
    /,
    *,
    timeout_sec: float = 30,
) -> list[Row]:
    """Run ``query`` over ``paths`` and return its output rows.

    Args:
      paths: A file or directory, or several, searched in the order given.
      query: What to find and how to report it.
      timeout_sec: Cap on each ``rg`` run.

    Returns:
      rows: The rendered output, one entry per line; empty when nothing matched.

    Raises:
      GrepError: A path is missing or unreadable, ``file_type`` is unknown, the pattern is
        invalid, two options cannot combine, or the search could not finish.

    """
    targets: list[Path] = [paths] if isinstance(paths, Path) else list(paths)
    # Checked once, before dispatch, so both backends agree: ``rg`` exits 2 on
    # either while a walk finds nothing and would report "no matches".
    if query.file_type and query.file_type not in TYPE_GLOBS:
        raise GrepError(
            f"unknown type: {query.file_type!r} (expected one of {sorted(TYPE_GLOBS)})",
        )
    if clash := _clash(query):
        raise GrepError(clash)
    # Here, not from ``rg``'s stderr: ``--no-messages`` silences it, and a walk
    # skips what it cannot read, but a path the caller named is never skipped.
    for path in targets:
        if not path.exists():
            raise GrepError(f"no such file or directory: {path}")
        # A directory needs ``x`` as well to be walked.
        if not os.access(path, os.R_OK | (os.X_OK if path.is_dir() else 0)):
            raise GrepError(f"permission denied: {path}")
    if not targets:
        return []
    rg = shutil.which("rg")
    # However small the search, never in-process when ``rg`` is installed: Python
    # ``re`` backtracks without bound while holding the GIL, so one pattern froze
    # the calling process for good. Ripgrep's engine is linear, and its run is
    # bounded by ``timeout_sec``.
    if rg is not None:
        return _grep_rg(rg, targets, query, timeout_sec=timeout_sec)
    return _grep_python(targets, query)


def rg_command(rg: str, paths: Sequence[Path], query: Query) -> list[str]:
    """Return the ``rg`` argv answering ``query`` over ``paths``.

    Args:
      rg: The ripgrep executable.
      paths: Files and directories to search.
      query: What to find and how to report it.

    Returns:
      argv: The command, ``rg`` first.

    """
    cmd = [
        rg,
        "--no-heading",
        # Even for one file: the Python backend always names it, and a caller
        # paging the lines cannot tell which file a bare line came from.
        "--with-filename",
        "--hidden",
        # Ripgrep reads a NUL in a named file as a line break, except in a memory
        # map -- made when at most ten paths, all files, are named, and never on
        # macOS -- so one query over one file would answer two ways.
        "--no-mmap",
        # A walked file rg cannot read -- unreadable, a broken or looping link --
        # is skipped, as the Python backend skips it; unsilenced, its message
        # and exit 2 would fail the whole search.
        "--no-messages",
        # No ``--sort path``: it makes ripgrep search one file at a time (8x
        # slower on this repo). Its parallel output is grouped per file and
        # sorted here instead, and control-byte field separators make each
        # row's file name and kind unambiguous for that.
        f"--field-match-separator={_MATCH.decode()}",
        f"--field-context-separator={_CONTEXT.decode()}",
        # No ``--max-columns``: it hides a match far right on a long line (a
        # needle in a minified bundle) that the Python backend returns whole.
    ]
    for glob in _globs(query):
        cmd.extend(["--glob", glob])
    if query.line_numbers:
        cmd.append("-n")
    if query.case_insensitive:
        cmd.append("-i")
    if query.multiline:
        cmd.extend(["-U", "--multiline-dotall"])
    if query.output_mode == "files_with_matches":
        cmd.append("-l")
    elif query.output_mode == "files_without_match":
        cmd.append("--files-without-match")
    elif query.output_mode == "count":
        cmd.append("-c")
    elif _columned(query):
        cmd.append("--column")
    if query.context_before > 0:
        cmd.extend(["-B", str(query.context_before)])
    if query.context_after > 0:
        cmd.extend(["-A", str(query.context_after)])
    if query.file_type:
        cmd.extend(["--type-clear", query.file_type])
        for pattern in TYPE_GLOBS[query.file_type]:
            cmd.extend(["--type-add", f"{query.file_type}:{pattern}"])
        cmd.extend(["--type", query.file_type])
    if query.text:
        cmd.append("--text")
    if query.pcre:
        cmd.append("-P")
    for on, flag in (
        (query.invert, "-v"),
        (query.word, "-w"),
        (query.whole_line, "-x"),
        (query.fixed, "-F"),
        (query.only_matching, "-o"),
    ):
        if on:
            cmd.append(flag)
    if query.max_count:
        cmd.extend(["--max-count", str(query.max_count)])
    if query.search_ignored:
        cmd.append("--no-ignore")
    if query.follow:
        cmd.append("--follow")
    if query.max_depth:
        cmd.extend(["--max-depth", str(query.max_depth)])
    cmd.extend(["--", query.pattern, *map(str, paths)])
    return cmd


def _grep_rg(
    rg: str,
    paths: Sequence[Path],
    query: Query,
    *,
    timeout_sec: float,
) -> list[Row]:
    """Answer ``query`` with ripgrep, one run per argv-sized batch of paths."""
    batches = list(_batches(paths))
    run = partial(_run_rg, query=query, timeout_sec=timeout_sec)
    try:
        first = run(rg_command(rg, batches[0], query))
    except GrepError as error:
        # Not ``--engine auto``: it also sends a refused ``\n`` to PCRE2,
        # which then silently matches nothing on a line.
        if query.pcre or not any(why in str(error) for why in _NEEDS_PCRE2):
            raise
        query = dataclasses.replace(query, pcre=True)
        first = run(rg_command(rg, batches[0], query))
    # Each run is its own process, so further batches overlap freely.
    commands = [rg_command(rg, batch, query) for batch in batches[1:]]
    with ThreadPoolExecutor(max_workers=min(len(batches), os.cpu_count() or 1)) as pool:
        rest = pool.map(run, commands)
        blocks = [*first, *(block for found in rest for block in found)]
    order = {str(path): i for i, path in enumerate(paths)}
    blocks.sort(key=lambda block: _sort_key(block[0], paths, order))
    rows: list[Row] = []
    for _, block in blocks:
        # Ripgrep separates context groups of different files with ``--``.
        if rows and _has_context(query):
            rows.append(Row(text="--"))
        rows.extend(block)
    return rows


# ``-l`` prints one name per line. Content rows read ``name M n M [col M] text``
# with ``M`` the ``_MATCH``/``_CONTEXT`` byte, which no path holds, so the first
# one ends the name. ``-c`` rows split at their last ``:``; a binary-file note
# splits at its fixed suffix, so either preserves colons in the file name.
# ``--`` lines separate context groups and are re-added after sorting.
def _file_blocks(out: bytes, *, query: Query) -> list[tuple[str, list[Row]]]:
    """Group ripgrep's output into ``(file, rows)`` blocks."""
    lines = out.removesuffix(b"\n").split(b"\n") if out else []
    if query.output_mode in {"files_with_matches", "files_without_match"}:
        names = (line.decode(errors="replace") for line in lines)
        return [(name, [Row(text=name)]) for name in names]
    blocks: list[tuple[str, list[Row]]] = []
    gap = False
    previous = ("", 0)
    for line in lines:
        if line == b"--":
            gap = True
            continue
        cut = min(
            (i for i in (line.find(_MATCH), line.find(_CONTEXT)) if i >= 0),
            default=-1,
        )
        if cut < 0:
            name, text = _plain_row(line.decode(errors="replace"))
            row = Row(text=text)
        else:
            name, number, row = _content_row(line, cut=cut, query=query)
            matched = row.column >= 0
            # ``-U`` prints a run of matched lines as one block that repeats its
            # first column on every line, so only the run's first line is located.
            if query.multiline and previous == (name, number - 1):
                row = Row(text=row.text, prefix_end=row.prefix_end)
            previous = (name, number) if matched else ("", 0)
        if not blocks or blocks[-1][0] != name:
            blocks.append((name, []))
        elif gap:
            # A gap between groups of one file stays; between files it is
            # re-added after sorting.
            blocks[-1][1].append(Row(text="--"))
        gap = False
        blocks[-1][1].append(row)
    return blocks


# ``--column`` implies ``-n`` and counts 1-based bytes of the printed UTF-8 line.
def _content_row(
    line: bytes,
    *,
    cut: int,
    query: Query,
) -> tuple[str, int, Row]:
    """Split a content row into its file, line number, and :class:`Row`."""
    marker = line[cut : cut + 1]
    number, _, rest = line[cut + 1 :].partition(marker)
    name = line[:cut].decode(errors="replace")
    shown = "-" if marker == _CONTEXT else ":"
    prefix = f"{name}{shown}{number.decode()}{shown}" if query.line_numbers else ""
    prefix = prefix or f"{name}{shown}"
    column = -1
    if marker == _MATCH and _columned(query):
        byte_column, _, rest = rest.partition(marker)
        # Under ``-o`` the column indexes the source line, but the row holds
        # only the match, which therefore begins the row's text.
        column = len(prefix) + (
            0
            if query.only_matching
            else len(rest[: int(byte_column) - 1].decode(errors="replace"))
        )
    text = prefix + rest.decode(errors="replace")
    return name, int(number), Row(text=text, prefix_end=len(prefix), column=column)


def _plain_row(row: str) -> tuple[str, str]:
    """Split a ``name:count`` or ``name: binary file matches ...`` row."""
    note = row.find(": binary file matches (")
    if note >= 0:
        return row[:note], row
    return row.rpartition(":")[0], row


# Named paths keep the caller's order, as ``rg`` keeps argv order; files a walk
# found sort by path components within the directory named, as ``--sort path``.
def _sort_key(
    name: str,
    paths: Sequence[Path],
    order: dict[str, int],
) -> tuple[int, tuple[str, ...]]:
    """Return where ``name`` falls among ``paths`` and then within its walk."""
    if name in order:
        return order[name], ()
    for i, path in enumerate(paths):
        prefix = str(path).rstrip("/") + "/"
        if name.startswith(prefix):
            return i, tuple(name[len(prefix) :].split("/"))
    return len(paths), (name,)


def _run_rg(
    cmd: list[str],
    *,
    query: Query,
    timeout_sec: float,
) -> list[tuple[str, list[Row]]]:
    """Run one ``rg`` argv and return its output, grouped by file."""
    try:
        done = subprocess.run(  # noqa: S603 -- Fixed ripgrep argv, no shell.
            cmd,
            capture_output=True,
            timeout=timeout_sec,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise GrepError(
            f"ripgrep timed out after {timeout_sec:g}s; narrow the path or pattern.",
        ) from error
    # Exit 2 with nothing said is files ``--no-messages`` skipped; what was found
    # stands.
    err = done.stderr.decode(errors="replace").strip()
    if done.returncode >= 2 and err:
        if 'the literal "\\n" is not allowed' in err:
            err = _NEWLINE_HINT
        raise GrepError(f"ripgrep error (exit {done.returncode}): {err}")
    # Bytes, split on ``\n`` alone: a ``\r`` inside a matched line is part of
    # it, and text-mode decoding would turn it into a line break.
    return _file_blocks(done.stdout, query=query)


def _batches(paths: Sequence[Path]) -> Iterator[list[Path]]:
    """Split ``paths`` into runs whose names, space-joined, fit ``ARGV_CHARS``."""
    start = 0
    size = -1
    for index, path in enumerate(paths):
        width = len(str(path)) + 1
        if index > start and size + width > ARGV_CHARS:
            yield list(paths[start:index])
            start, size = index, -1
        size += width
    yield list(paths[start:])


# Later overrides win in ripgrep's ignore library, so the exclude must come last.
def _globs(query: Query) -> list[str]:
    """Return the ``--glob`` overrides both backends apply."""
    return [
        "!.git",
        "!.svn",
        "!.hg",
        *([query.glob] if query.glob else []),
        *([f"!{query.exclude}"] if query.exclude else []),
    ]


def _compile(query: Query) -> re.Pattern[str]:
    """Compile ``query`` for Python ``re``, refusing what only ripgrep can run."""
    if query.pcre:
        # Python ``re`` is a different language; answering under PCRE2's name
        # returns results the caller cannot tell were computed otherwise.
        raise GrepError(
            "pcre=true requires ripgrep, which is not installed;"
            " the Python fallback cannot provide PCRE2 semantics.",
        )
    if not query.multiline and (
        "\n" in query.pattern if query.fixed else _references_newline(query.pattern)
    ):
        raise GrepError(_NEWLINE_HINT)
    pattern = re.escape(query.pattern) if query.fixed else query.pattern
    # Ripgrep 14's ``-w`` asks only that no word character border the match.
    if query.word:
        pattern = rf"(?<!\w)(?:{pattern})(?!\w)"
    if query.whole_line:
        pattern = f"^(?:{pattern})$"
    # Ripgrep always matches ``^``/``$`` at line boundaries.
    flags = re.MULTILINE
    if query.multiline:
        flags |= re.DOTALL
    if query.case_insensitive:
        flags |= re.IGNORECASE
    try:
        return re.compile(pattern, flags)
    except re.error as error:
        raise GrepError(
            f"ripgrep error (Python fallback): invalid regex pattern: {error}",
        ) from error


def _references_newline(pattern: str) -> bool:
    """Detect newline escapes outside classes without confusing escaped brackets."""
    class_start = -1
    index = 0
    while index < len(pattern):
        char = pattern[index]
        if char == "\\":
            if class_start < 0 and pattern[index + 1 : index + 2] == "n":
                return True
            index += 2
            continue
        if char == "[" and class_start < 0:
            class_start = index + 1 + (pattern[index + 1 : index + 2] == "^")
        elif char == "]" and index > class_start:
            class_start = -1
        elif char == "\n" and class_start < 0:
            return True
        index += 1
    return False


def _grep_python(paths: Sequence[Path], query: Query) -> list[Row]:
    """Answer ``query`` in Python, matching ripgrep's output line for line."""
    pattern = _compile(query)
    lines: list[Row] = []
    for path in paths:
        for name, file, walked in _files(path, query):
            rows = _search_file(pattern, file, name=name, walked=walked, query=query)
            if lines and rows and _has_context(query):
                lines.append(Row(text="--"))
            lines.extend(rows)
    return lines


# Each comes with the name ripgrep prints for it and whether a walk found it.
def _files(named: Path, query: Query) -> list[tuple[str, Path, bool]]:
    """Return ``named`` itself, or each file a walk of it keeps, in path order."""
    if named.is_dir():
        return _sorted((name, path, True) for name, path in _walked(named, query))
    return [(str(named), named, False)]


def _walked(named: Path, query: Query) -> Iterator[tuple[str, Path]]:
    """Yield each regular file a walk of directory ``named`` keeps."""
    # ``rg .`` prints ``./a``: a walked file is named under the argument as
    # given, which a ``Path`` join would normalize away.
    prefix = str(named) + "/"
    start = len(str(named)) + 1
    ignores = not query.search_ignored
    # ``ignore_hidden=False`` is ``--hidden``.
    walker = rignore.walk(
        named,
        ignore_hidden=False,
        read_ignore_files=ignores,
        read_parents_ignores=ignores,
        read_git_ignore=ignores,
        read_global_git_ignore=ignores,
        read_git_exclude=ignores,
        overrides=_globs(query),
        max_depth=query.max_depth or None,
        follow_links=query.follow,
    )
    for file in _entries(iter(walker)):
        # Unless following, a walk lists a symlink without following it; only a
        # regular file is searched, and a broken link is none.
        try:
            status = file.stat() if query.follow else file.lstat()
        except OSError:
            continue
        if not S_ISREG(status.st_mode):
            continue
        if query.file_type and not any(
            file.match(g) for g in TYPE_GLOBS[query.file_type]
        ):
            continue
        yield prefix + str(file)[start:], file


# A followed link that is broken or loops raises from ``next`` and the walk goes
# on past it, as ripgrep skips it; a ``for`` loop would end the walk there.
def _entries(walker: Iterator[Path]) -> Iterator[Path]:
    """Yield each path ``walker`` lists, skipping the entries it fails on."""
    while True:
        try:
            yield next(walker)
        except StopIteration:
            return
        except OSError:
            continue


# Splitting the name per component sorts as ``--sort path`` does.
def _sorted(found: Iterable[tuple[str, Path, bool]]) -> list[tuple[str, Path, bool]]:
    return sorted(found, key=lambda entry: entry[0].split("/"))


def _search_file(
    pattern: re.Pattern[str],
    file: Path,
    *,
    name: str,
    walked: bool,
    query: Query,
) -> list[Row]:
    """Render one file's output lines, or none when it is skipped."""
    try:
        raw = file.read_bytes()
    except OSError:
        return []
    text = _decode(raw)
    if "\0" not in text or query.text:
        return _file_lines(pattern, text, name=name, query=query)
    if walked:
        return []
    # A binary file named outright is searched as ripgrep reads one, each NUL a
    # line break -- save in a multiline search, which holds the whole file:
    # there each NUL stays, swapped here for a character ``.`` matches and no
    # literal does. Its lines are never printed -- one note stands for all.
    found = _file_lines(
        pattern,
        text.replace("\0", "\ue000" if query.multiline else "\n"),
        name=name,
        query=query,
    )
    if found and query.output_mode == "content":
        offset = raw.index(b"\0")
        note = f'{name}: binary file matches (found "\\0" byte around offset {offset})'
        return [Row(text=note)]
    return found


def _decode(raw: bytes) -> str:
    """Decode as ripgrep does: honour a UTF-8/UTF-16 BOM, else lossy UTF-8."""
    if raw.startswith(codecs.BOM_UTF8):
        return raw[len(codecs.BOM_UTF8) :].decode(errors="replace")
    if raw.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        # pragma: no mutate start -- codec names are case-insensitive.
        return raw.decode("utf-16", errors="replace")
        # pragma: no mutate end
    return raw.decode(errors="replace")


def _lines(text: str) -> list[str]:
    """Split on newline only; a final newline ends the last line, not a new one."""
    split = text.split("\n")
    if split[-1] == "":
        split.pop()
    return split


# One search over the whole text, not one per line: a file without a match costs
# a single scan. A whole-text hit only nominates its line; the line is then
# searched by itself, since only that decides how ``^``/``$``/``\s``/lookaround
# see its ends. A nominee that fails resumes the scan one character on, so a
# cross-line false hit cannot hide a real match later on the same line.
def _line_hits(pattern: re.Pattern[str], text: str) -> dict[int, int]:
    """Map each 0-based line ``pattern`` matches, searched alone, to its column."""
    hits: dict[int, int] = {}
    line_no = 0
    counted = 0
    pos = 0
    while pos <= len(text) and (match := pattern.search(text, pos)) is not None:
        start = text.rfind("\n", 0, match.start()) + 1
        # Past the final newline, or in an empty file, there is no line.
        if start == len(text) and (not text or text.endswith("\n")):
            break
        end = text.find("\n", start)
        end = len(text) if end < 0 else end
        line_no += text.count("\n", counted, start)
        counted = start
        if found := pattern.search(text[start:end]):
            hits[line_no] = found.start()
            pos = end + 1
        else:
            pos = match.start() + 1
    return hits


def _file_lines(
    pattern: re.Pattern[str],
    text: str,
    *,
    name: str,
    query: Query,
) -> list[Row]:
    """Render one file's output lines."""
    split = _lines(text)
    if query.multiline:
        count, chosen = _multiline_hits(pattern, text)
        shown = chosen
    else:
        hits = _line_hits(pattern, text)
        # Selected lines, each with its match column; a line in trailing
        # context that is selected shows as one, though ``max_count`` was met.
        shown = (
            {i: -1 for i in range(len(split)) if i not in hits}
            if query.invert
            else hits
        )
        chosen = dict(itertools.islice(shown.items(), query.max_count or None))
        count = len(chosen)
    if query.only_matching:
        count = sum(len(_matches(pattern, split[i])) for i in chosen)
    if query.output_mode == "files_without_match":
        return [] if count else [Row(text=name)]
    if not count:
        return []
    if query.output_mode == "files_with_matches":
        return [Row(text=name)]
    if query.output_mode == "count":
        return [Row(text=f"{name}:{count}")]
    if not _has_context(query):
        return [
            row
            for i in chosen
            for row in _selected(
                pattern,
                name=name,
                index=i,
                split=split,
                shown=shown,
                query=query,
            )
        ]
    out: list[Row] = []
    for start, end in _context_groups(list(chosen), len(split), query=query):
        if out:
            out.append(Row(text="--"))
        for j in range(start, end):
            out.extend(
                _selected(
                    pattern,
                    name=name,
                    index=j,
                    split=split,
                    shown=shown,
                    query=query,
                )
                if j in shown
                else [_row(name, j, split[j], query=query, separator="-")],
            )
    return out


def _selected(
    pattern: re.Pattern[str],
    *,
    name: str,
    index: int,
    split: list[str],
    shown: dict[int, int],
    query: Query,
) -> list[Row]:
    """Render selected line ``index``: whole, or one row per match under ``-o``."""
    line = split[index]
    if not query.only_matching:
        return [_row(name, index, line, query=query, column=shown[index])]
    return [
        _row(name, index, match.group(), query=query, column=0)
        for match in _matches(pattern, line)
    ]


# Ripgrep's engine never reports an empty match where the previous match ended,
# which Python's ``finditer`` does: ``o*`` over ``foo`` is ``""``, ``"oo"`` to
# ripgrep, and ``""``, ``"oo"``, ``""`` to Python.
def _matches(pattern: re.Pattern[str], line: str) -> list[re.Match[str]]:
    """Return each match in ``line``, in the order ripgrep reports them."""
    found: list[re.Match[str]] = []
    pos = 0
    last = -1
    while pos <= len(line) and (match := pattern.search(line, pos)) is not None:
        empty = match.start() == match.end()
        if not (empty and match.start() == last):
            found.append(match)
            last = match.end()
        pos = match.end() + empty
    return found


# Ripgrep searches a pattern that can match a newline across lines and ``-c``
# counts its matches; any other pattern it searches line by line and ``-c``
# counts matching lines (``searcher/glue.rs``, ``printer/summary.rs``). A match
# holding a newline is the evidence used here, so ``x|\n`` over text with no
# newline match counts lines where ripgrep counts matches. An empty match
# after a final newline sits on no line.
def _multiline_hits(
    pattern: re.Pattern[str],
    text: str,
) -> tuple[int, dict[int, int]]:
    """Return the ``-c`` count and each touched 0-based line's match column."""
    matches = [
        match
        for match in pattern.finditer(text)
        if match.start() != len(text) or (text and not text.endswith("\n"))
    ]
    lines: dict[int, int] = {}
    for match in matches:
        start = text[: match.start()].count("\n")
        lines.setdefault(start, match.start() - text.rfind("\n", 0, match.start()) - 1)
        # A match ending in ``\n`` ends on that newline's line, not the next.
        for line in range(
            start + 1,
            start + match.group().removesuffix("\n").count("\n") + 1,
        ):
            lines.setdefault(line, -1)
    across = any("\n" in match.group() for match in matches)
    # Ripgrep locates only the first line of a run of matched lines.
    located = {i: -1 if i - 1 in lines else column for i, column in lines.items()}
    return (len(matches) if across else len(lines)), dict(sorted(located.items()))


def _clash(query: Query) -> str:
    """Name a combination ripgrep answers inconsistently; ``""`` when none."""
    if query.multiline:
        # ``rg -U -v`` prints inverted lines yet counts none; ``-U -m 1`` prints
        # a match run past the cap; ``-U -o`` repeats a column per line.
        for on, name in (
            (query.invert, "invert"),
            (query.only_matching, "only_matching"),
            (query.max_count > 0, "max_count"),
        ):
            if on:
                return f"{name} cannot combine with multiline."
    # ``rg -o -v`` prints whole lines but counts zero; ``grep -o -v`` prints none.
    if query.invert and query.only_matching:
        return "only_matching cannot combine with invert."
    return ""


def _columned(query: Query) -> bool:
    """Return whether content rows carry ``rg --column``'s match column."""
    # ``rg -v --column`` puts a column on context lines and none on selected ones.
    return query.output_mode == "content" and not query.invert


def _has_context(query: Query) -> bool:
    """Return whether content output carries context groups."""
    return query.output_mode == "content" and (
        query.context_before > 0 or query.context_after > 0
    )


# One group per match repeated the shared lines, so a caller counting
# occurrences in the output counted them twice; ripgrep merges instead.
def _context_groups(
    hits: list[int],
    total: int,
    *,
    query: Query,
) -> list[tuple[int, int]]:
    """Merge each match's context window with its overlapping neighbours."""
    groups: list[tuple[int, int]] = []
    for i in hits:
        start = max(0, i - query.context_before)
        end = min(total, i + query.context_after + 1)
        if groups and start <= groups[-1][1]:
            groups[-1] = (groups[-1][0], max(groups[-1][1], end))
            continue
        groups.append((start, end))
    return groups


def _row(
    name: str,
    index: int,
    line: str,
    *,
    query: Query,
    separator: str = ":",
    column: int = -1,
) -> Row:
    """Render one line as ``rg --no-heading``: ``:`` for a match, ``-`` for context."""
    prefix = (
        f"{name}{separator}{f'{index + 1}{separator}' if query.line_numbers else ''}"
    )
    return Row(
        text=prefix + line,
        prefix_end=len(prefix),
        column=len(prefix) + column if column >= 0 else -1,
    )
