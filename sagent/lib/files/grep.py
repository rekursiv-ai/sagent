"""Search file contents as ripgrep does: ``rg`` when installed, else in Python.

Both backends take one :class:`Query` and return the same lines -- one entry per
match, per context line, per ``--`` group separator, per counted file, or per
matching file. A directory is walked in path order with ``.git``/``.svn``/
``.hg`` and ``.gitignore``d trees skipped, hidden entries searched, symlinks
not followed, and binary (NUL-bearing) files skipped. A file named outright is
searched whatever its name, type or ignore status, as ripgrep searches it. The
Python backend walks with ``rignore``, which binds the Rust ``ignore`` library
ripgrep itself is built on.
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
import os
import re
import shutil
import subprocess
import time

import rignore


if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence


OutputMode = Literal["content", "files_with_matches", "count"]

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
_MATCH: Final = "\x01"
_CONTEXT: Final = "\x02"

# What ripgrep's own engine says when only PCRE2 can run the pattern.
_NEEDS_PCRE2: Final = ("look-around", "backreferences are not supported")

# Linux caps one argv near 2 MiB and each string at 128 KiB; a few thousand
# named files stay well under both when split here.
ARGV_CHARS: Final = 100_000
"""Most characters of named paths one ``rg`` run takes."""

# Starting ``rg`` and walking costs ~5 ms. Python's ``re`` runs ~0.02 to ~50 ns a
# character depending on the pattern, so the choice is measured, not guessed:
# list up to ``SMALL_FILES`` files (a bigger tree goes to ``rg`` after ~2 ms),
# time the compiled pattern over ``_SAMPLE`` characters of their text, and search
# in-process only when the projected cost beats ``_RG_MS``.
#
# Raising ``SMALL_FILES`` is not the free win it looks like: the walk that
# counts the files is discarded when ``rg`` then runs, so every larger tree
# pays it twice, and ``_projected_ms`` already vetoes a tree past
# ``_RG_MS / _PER_FILE_MS`` files -- so a higher cap buys no in-process search.
SMALL_FILES: Final = 128
"""Most files searched in-process when ``rg`` is installed; ``-1`` always uses it."""
_SAMPLE: Final = 16 << 10
_RG_MS: Final = 4.0
_PER_FILE_MS: Final = 0.02

_Found = list[list[tuple[str, Path, bool]]]
"""Per target, each file to search: its printed name, path, and whether walked."""


# Python ``re`` and ripgrep read most syntax alike but not all: ``&&`` in a class,
# ``\A``/``\Z``, ``(?>``, ``{,n}``, ``\N{}``, ``\p{}``, ``\x{}``, POSIX classes. With
# ``rg`` installed the answer must be ripgrep's, so only a pattern made of the
# shared core -- literals, ``.``, classes without these, anchors, ``\w\d\s\b``,
# groups, alternation, ``?*+{m,n}`` -- is searched in-process.
_SHARED: Final = re.compile(
    r"""(?x)
    (?: [^\\\[\]{}()&]                       # a plain character or metachar
      | \\[\\.^$|?*+()\[\]{}/\-dDwWsSbBnrt]   # an escape both read the same
      | \{\d+(?:,\d*)?\}                      # a bounded repeat with a minimum
      | \(\?[:imsx]*:? | \( | \)              # plain and non-capturing groups
      | \[\^?\]?(?:[^\]\\\[&]|\\[\\\]\-dDwWsSnrt^])*\]  # a class with no set ops
    )*""",
)


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
      output_mode: Matching lines, matching files, or per-file counts.
      context_before: Lines shown before each match (``rg -B``).
      context_after: Lines shown after each match (``rg -A``).
      case_insensitive: ``rg -i``.
      line_numbers: ``rg -n``.
      multiline: Let a match span lines (``rg -U --multiline-dotall``).
      pcre: PCRE2 syntax only (``rg -P``); needs ``rg``.
      text: Search binary files as text (``rg --text``).

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


def grep(
    paths: Path | Sequence[Path],
    query: Query,
    /,
    *,
    timeout_sec: float = 30,
) -> list[str]:
    """Run ``query`` over ``paths`` and return its output lines.

    Args:
      paths: A file or directory, or several, searched in the order given.
      query: What to find and how to report it.
      timeout_sec: Cap on each ``rg`` run.

    Returns:
      lines: The rendered output, one entry per line; empty when nothing matched.

    Raises:
      GrepError: A path is missing, ``file_type`` is unknown, the pattern is
        invalid, or the search could not finish.

    """
    targets: list[Path] = [paths] if isinstance(paths, Path) else list(paths)
    # Checked once, before dispatch, so both backends agree: ``rg`` exits 2 on
    # either while a walk finds nothing and would report "no matches".
    if query.file_type and query.file_type not in TYPE_GLOBS:
        raise GrepError(
            f"unknown type: {query.file_type!r} (expected one of {sorted(TYPE_GLOBS)})",
        )
    if not targets:
        return []
    rg = shutil.which("rg")
    found = None if rg is None else _small(targets, query)
    if rg is not None and found is None:
        return _grep_rg(rg, targets, query, timeout_sec=timeout_sec)
    for path in targets:
        if not path.exists():
            raise GrepError(f"no such file or directory: {path}")
    return _grep_python(targets, query, found=found)


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
        # No ``--sort path``: it makes ripgrep search one file at a time (8x
        # slower on this repo). Its parallel output is grouped per file and
        # sorted here instead, and control-byte field separators make each
        # row's file name and kind unambiguous for that.
        f"--field-match-separator={_MATCH}",
        f"--field-context-separator={_CONTEXT}",
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
    elif query.output_mode == "count":
        cmd.append("-c")
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
    cmd.extend(["--", query.pattern, *map(str, paths)])
    return cmd


def _grep_rg(
    rg: str,
    paths: Sequence[Path],
    query: Query,
    *,
    timeout_sec: float,
) -> list[str]:
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
    lines: list[str] = []
    for _, rows in blocks:
        # Ripgrep separates context groups of different files with ``--``.
        if lines and _has_context(query):
            lines.append("--")
        lines.extend(rows)
    return lines


# ``-l`` prints one name per line. Content rows read ``name M [n M] text`` with
# ``M`` the ``_MATCH``/``_CONTEXT`` byte, which no path holds, so the first one
# ends the name. ``-c`` rows split at their last ``:``; a binary-file note
# splits at its fixed suffix, so either preserves colons in the file name.
# ``--`` lines separate context groups and are re-added after sorting.
def _file_blocks(
    out: str,
    *,
    names_only: bool,
    numbered: bool,
) -> list[tuple[str, list[str]]]:
    """Group ripgrep's output into ``(file, rendered rows)`` blocks."""
    rows = _lines(out)
    if names_only:
        return [(name, [name]) for name in rows]
    blocks: list[tuple[str, list[str]]] = []
    gap = False
    for row in rows:
        if row == "--":
            gap = True
            continue
        cut = min(
            (i for i in (row.find(_MATCH), row.find(_CONTEXT)) if i >= 0),
            default=-1,
        )
        if cut < 0:
            name, rendered = _plain_row(row)
        else:
            name = row[:cut]
            marker = row[cut]
            shown = ":" if marker == _MATCH else "-"
            rest = row[cut + 1 :]
            if numbered:
                number, _, rest = rest.partition(marker)
                rest = f"{number}{shown}{rest}"
            rendered = f"{name}{shown}{rest}"
        if not blocks or blocks[-1][0] != name:
            blocks.append((name, []))
        elif gap:
            # A gap between groups of one file stays; between files it is
            # re-added after sorting.
            blocks[-1][1].append("--")
        gap = False
        blocks[-1][1].append(rendered)
    return blocks


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
) -> list[tuple[str, list[str]]]:
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
    if done.returncode >= 2:
        err = done.stderr.decode(errors="replace").strip() or "unknown"
        if 'the literal "\\n" is not allowed' in err:
            err = _NEWLINE_HINT
        # A named path rg cannot open: the same error the Python backend gives.
        missing = re.search(
            r"^rg: (.*): No such file or directory \(os error 2\)$",
            err,
            re.MULTILINE,
        )
        if missing:
            raise GrepError(f"no such file or directory: {missing[1]}")
        raise GrepError(f"ripgrep error (exit {done.returncode}): {err}")
    # Bytes, split on ``\n`` alone: a ``\r`` inside a matched line is part of
    # it, and text-mode decoding would turn it into a line break.
    return _file_blocks(
        done.stdout.decode(errors="replace"),
        names_only=query.output_mode == "files_with_matches",
        numbered=query.line_numbers,
    )


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
    if not query.multiline and _references_newline(query.pattern):
        raise GrepError(_NEWLINE_HINT)
    # Ripgrep always matches ``^``/``$`` at line boundaries.
    flags = re.MULTILINE
    if query.multiline:
        flags |= re.DOTALL
    if query.case_insensitive:
        flags |= re.IGNORECASE
    try:
        return re.compile(query.pattern, flags)
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


def _grep_python(
    paths: Sequence[Path],
    query: Query,
    *,
    found: _Found | None = None,
) -> list[str]:
    """Answer ``query`` in Python, matching ripgrep's output line for line."""
    pattern = _compile(query)
    lines: list[str] = []
    listed = found if found is not None else (list(_files(p, query)) for p in paths)
    for files in listed:
        for name, file, walked in files:
            rows = _search_file(pattern, file, name=name, walked=walked, query=query)
            if lines and rows and _has_context(query):
                lines.append("--")
            lines.extend(rows)
    return lines


# Each comes with the name ripgrep prints for it and whether a walk found it.
def _files(named: Path, query: Query) -> list[tuple[str, Path, bool]]:
    """Return ``named`` itself, or each file a walk of it keeps, in path order."""
    if named.is_dir():
        return _sorted((name, path, True) for name, path, _ in _walked(named, query))
    return [(str(named), named, False)]


def _walked(named: Path, query: Query) -> Iterator[tuple[str, Path, int]]:
    """Yield each regular file a walk of directory ``named`` keeps, with its size."""
    # ``rg .`` prints ``./a``: a walked file is named under the argument as
    # given, which a ``Path`` join would normalize away.
    prefix = str(named) + "/"
    start = len(str(named)) + 1
    # ``ignore_hidden=False`` is ``--hidden``.
    for file in rignore.walk(named, ignore_hidden=False, overrides=_globs(query)):
        # A walk lists a symlink without following it; only a regular file is
        # searched. The size rides along from this same call so that ``_small``
        # need not stat every walked file a second time.
        try:
            status = file.lstat()
        except OSError:
            continue
        if not S_ISREG(status.st_mode):
            continue
        if query.file_type and not any(
            file.match(g) for g in TYPE_GLOBS[query.file_type]
        ):
            continue
        yield prefix + str(file)[start:], file, status.st_size


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
) -> list[str]:
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
    # A binary file named outright is searched with each NUL swapped for a byte
    # that ``.`` matches and no literal does, as ripgrep swaps it; its lines are
    # never printed -- one note stands for all of them.
    found = _file_lines(pattern, text.replace("\0", "\ue000"), name=name, query=query)
    if found and query.output_mode == "content":
        offset = raw.index(b"\0")
        return [
            f'{name}: binary file matches (found "\\0" byte around offset {offset})',
        ]
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
def _line_hits(pattern: re.Pattern[str], text: str) -> list[int]:
    """Return the 0-based lines ``pattern`` matches when searched line by line."""
    hits: list[int] = []
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
        if pattern.search(text[start:end]):
            hits.append(line_no)
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
) -> list[str]:
    """Render one file's output lines."""
    if query.multiline:
        count, hits = _multiline_hits(pattern, text)
    else:
        hits = _line_hits(pattern, text)
        count = len(hits)
    if not count:
        return []
    split = _lines(text)
    if query.output_mode == "files_with_matches":
        return [name]
    if query.output_mode == "count":
        return [f"{name}:{count}"]
    if not _has_context(query):
        return [_row(name, i, split[i], query=query) for i in hits]
    shown = set(hits)
    out: list[str] = []
    for start, end in _context_groups(hits, len(split), query=query):
        if out:
            out.append("--")
        out.extend(
            _row(name, j, split[j], query=query, separator=":" if j in shown else "-")
            for j in range(start, end)
        )
    return out


# Ripgrep searches a pattern that can match a newline across lines and ``-c``
# counts its matches; any other pattern it searches line by line and ``-c``
# counts matching lines (``searcher/glue.rs``, ``printer/summary.rs``). A match
# holding a newline is the evidence used here, so ``x|\n`` over text with no
# newline match counts lines where ripgrep counts matches. An empty match
# after a final newline sits on no line.
def _multiline_hits(pattern: re.Pattern[str], text: str) -> tuple[int, list[int]]:
    """Return the ``-c`` count and the 0-based lines the matches touch."""
    matches = [
        match
        for match in pattern.finditer(text)
        if match.start() != len(text) or (text and not text.endswith("\n"))
    ]
    lines: set[int] = set()
    for match in matches:
        start = text[: match.start()].count("\n")
        # A match ending in ``\n`` ends on that newline's line, not the next.
        lines.update(
            range(start, start + match.group().removesuffix("\n").count("\n") + 1),
        )
    across = any("\n" in match.group() for match in matches)
    return (len(matches) if across else len(lines)), sorted(lines)


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
) -> str:
    """Render one line as ``rg --no-heading``: ``:`` for a match, ``-`` for context."""
    number = f"{index + 1}{separator}" if query.line_numbers else ""
    return f"{name}{separator}{number}{line}"


def _small(targets: Sequence[Path], query: Query) -> _Found | None:
    """Return the files to search when doing it in-process beats starting ``rg``."""
    # ``grep`` returns before this on no targets, so one target already exceeds a
    # negative ``SMALL_FILES``; a separate ``< 0`` guard would never decide a run.
    if query.pcre or len(targets) > SMALL_FILES:
        return None
    if not _SHARED.fullmatch(query.pattern):
        return None
    try:
        pattern = _compile(query)
    except GrepError:
        return None  # Ripgrep's dialect, not Python's: ripgrep judges it.
    found: _Found = []
    files: list[Path] = []
    # pragma: no mutate start -- a one-byte seed reaches the dispatch only at
    # the exact ``_SAMPLE`` boundary, and then only through a timing estimate
    # no test can pin to a single byte.
    size = 0
    # pragma: no mutate end
    for target in targets:
        walked = target.is_dir()
        sized: Iterator[tuple[str, Path, int]]
        if walked:
            sized = _walked(target, query)
        else:
            # A missing named file is dropped rather than raised on, so that
            # ``grep``'s own existence check names it as the fault instead of
            # an ``OSError`` escaping from the size probe.
            try:
                sized = iter([(str(target), target, target.stat().st_size)])
            except OSError:
                sized = iter(())
        kept: list[tuple[str, Path, bool]] = []
        for name, path, grown in sized:
            size += grown
            files.append(path)
            if len(files) > SMALL_FILES:
                return None
            kept.append((name, path, walked))
        # A named target contributes a single entry, so this only ever orders
        # the files a walk found -- which is what ``rg --sort path`` gives.
        found.append(_sorted(kept))
    return found if _projected_ms(pattern, files, size) < _RG_MS else None


def _projected_ms(pattern: re.Pattern[str], files: Sequence[Path], size: int) -> float:
    """Project the in-process cost of ``files`` from ``pattern``'s speed on them."""
    if size <= _SAMPLE:
        return len(files) * _PER_FILE_MS
    sample = ""
    for file in files:
        try:
            with file.open("rb") as handle:
                sample += _decode(handle.read(_SAMPLE - len(sample)))
        except OSError:
            continue
        if len(sample) >= _SAMPLE:
            break
    # A NUL ends no line here; a pattern never matching keeps the scan full.
    probe = sample + "\0"
    start = time.perf_counter()
    for _ in pattern.finditer(probe):
        pass
    scan_ms = (time.perf_counter() - start) * 1000 * size / max(1, len(probe))
    return scan_ms + len(files) * _PER_FILE_MS
