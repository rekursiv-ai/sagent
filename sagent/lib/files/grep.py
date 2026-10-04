"""Search file contents as ripgrep does: ``rg`` when installed, else in Python.

Both backends take one :class:`Query` and return the same lines -- one entry per
match, per context line, per ``--`` group separator, per counted file, or per
matching file -- in the same order: files sorted by path, ``.git``/``.svn``/
``.hg`` and ``.gitignore``d trees skipped, hidden entries searched. The Python
backend walks with ``rignore``, the ``ignore`` create ripgrep itself is built on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal

import re
import shutil
import subprocess

import rignore


if TYPE_CHECKING:
    from pathlib import Path


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


class GrepError(Exception):
    """A query no backend can answer; the message is written for the caller."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Query:
    """One search, in ripgrep's terms.

    Attributes:
      pattern: Regex to find.
      glob: Only files matching this glob (``rg --glob``); brace sets allowed.
      exclude: Skip files matching this glob (``rg --glob !PAT``).
      file_type: A :data:`TYPE_GLOBS` name (``rg --type``).
      output_mode: Matching lines, matching files, or per-file counts.
      context_before: Lines shown before each match (``rg -B``).
      context_after: Lines shown after each match (``rg -A``).
      case_insensitive: ``rg -i``.
      line_numbers: ``rg -n``.
      multiline: Let a match span lines (``rg -U --multiline-dotall``).
      pcre: PCRE2 syntax (``rg -P``); needs ``rg``.

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


def grep(path: Path, query: Query, /, *, timeout_sec: float = 30) -> list[str]:
    """Run ``query`` under ``path`` and return its output lines.

    Args:
      path: File or directory to search.
      query: What to find and how to report it.
      timeout_sec: Cap on an ``rg`` run.

    Returns:
      lines: The rendered output, one entry per line; empty when nothing matched.

    Raises:
      GrepError: ``path`` is missing, ``file_type`` is unknown, the pattern is
        invalid, or the search could not finish.

    """
    # Checked once, before dispatch, so both backends agree: ``rg`` exits 2 on
    # either while a walk finds nothing and would report "no matches".
    if query.file_type and query.file_type not in TYPE_GLOBS:
        raise GrepError(
            f"unknown type: {query.file_type!r} (expected one of {sorted(TYPE_GLOBS)})",
        )
    if not path.exists():
        raise GrepError(f"no such file or directory: {path}")
    rg = shutil.which("rg")
    if rg:
        return _grep_rg(rg, path, query, timeout_sec=timeout_sec)
    return _grep_python(path, query)


def rg_command(rg: str, path: Path, query: Query) -> list[str]:
    """Return the ``rg`` argv answering ``query`` under ``path``.

    Args:
      rg: The ripgrep executable.
      path: File or directory to search.
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
        # Deterministic order: ripgrep's parallel walk emits files as workers
        # finish, so a caller paging the lines would see a different slice
        # each run, and disagree with the sorted Python backend.
        "--sort",
        "path",
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
        cmd.extend(["--type", query.file_type])
    if query.pcre:
        cmd.append("-P")
    cmd.extend(["--", query.pattern, str(path)])
    return cmd


def _grep_rg(rg: str, path: Path, query: Query, *, timeout_sec: float) -> list[str]:
    """Answer ``query`` with ripgrep."""
    try:
        done = subprocess.run(  # noqa: S603 -- Fixed ripgrep argv, no shell.
            rg_command(rg, path, query),
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise GrepError(
            f"ripgrep timed out after {timeout_sec:g}s; narrow the path or pattern.",
        ) from error
    if done.returncode >= 2:
        err = done.stderr.strip() or "unknown"
        if not query.multiline and 'the literal "\\n" is not allowed' in err:
            err = _NEWLINE_HINT
        raise GrepError(f"ripgrep error (exit {done.returncode}): {err}")
    return done.stdout.splitlines()


# Later overrides win in the ignore create, so the exclude must come last.
def _globs(query: Query) -> list[str]:
    """Return the ``--glob`` overrides both backends apply."""
    return [
        "!.git",
        "!.svn",
        "!.hg",
        *([query.glob] if query.glob else []),
        *([f"!{query.exclude}"] if query.exclude else []),
    ]


def _grep_python(path: Path, query: Query) -> list[str]:
    """Answer ``query`` in Python, matching ripgrep's output line for line."""
    if query.pcre:
        # Python ``re`` is a different language; answering under PCRE2's name
        # returns results the caller cannot tell were computed otherwise.
        raise GrepError(
            "pcre=true requires ripgrep, which is not installed;"
            " the Python fallback cannot provide PCRE2 semantics.",
        )
    if not query.multiline and r"\n" in query.pattern:
        raise GrepError(_NEWLINE_HINT)
    flags = (re.DOTALL if query.multiline else 0) | (
        re.IGNORECASE if query.case_insensitive else 0
    )
    try:
        pattern = re.compile(query.pattern, flags)
    except re.error as error:
        raise GrepError(
            f"ripgrep error (Python fallback): invalid regex pattern: {error}",
        ) from error
    lines: list[str] = []
    # ``ignore_hidden=False`` is ``--hidden``; sorting ``Path``s compares per
    # component, as ``--sort path`` does.
    walked = rignore.walk(path, ignore_hidden=False, overrides=_globs(query))
    for file in sorted(walked):
        if not file.is_file() or (
            query.file_type
            and not any(file.match(g) for g in TYPE_GLOBS[query.file_type])
        ):
            continue
        try:
            text = file.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        lines.extend(_file_lines(pattern, text, name=str(file), query=query))
    return lines


def _file_lines(
    pattern: re.Pattern[str],
    text: str,
    *,
    name: str,
    query: Query,
) -> list[str]:
    """Render one file's output lines."""
    split = text.splitlines()
    rows = (
        [(text[: m.start()].count("\n"), m.group()) for m in pattern.finditer(text)]
        if query.multiline
        else [(i, line) for i, line in enumerate(split) if pattern.search(line)]
    )
    if not rows:
        return []
    if query.output_mode == "files_with_matches":
        return [name]
    if query.output_mode == "count":
        return [f"{name}:{len(rows)}"]
    if query.multiline or (query.context_before <= 0 and query.context_after <= 0):
        return [_row(name, i, line, query=query) for i, line in rows]
    hits = {i for i, _ in rows}
    out: list[str] = []
    for start, end in _context_groups(sorted(hits), len(split), query=query):
        if out:
            out.append("--")
        out.extend(
            _row(name, j, split[j], query=query, separator=":" if j in hits else "-")
            for j in range(start, end)
        )
    return out


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
