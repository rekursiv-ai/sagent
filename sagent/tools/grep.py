"""Grep tool: ripgrep-first content search with Python fallback."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import cast

import logging
import os
import re
import shutil
import subprocess
import sys

from sagent.custom_types import Message, TextMessage
from sagent.lib.json import JSON, bool_val, int_val, json_freeze
from sagent.lib.message import get_directive
from sagent.tools.core import (
    get_tool_state,
    load_tool_description,
    run_sync,
)
from sagent.tools.lib.bash import (
    Node,
    match_pipeline,
    unwrap_cd_prefix,
)


logger = logging.getLogger(__name__)


# File type extensions for grep --type filter.
_TYPE_GLOBS: dict[str, list[str]] = {
    "py": ["*.py"],
    "js": ["*.js", "*.jsx", "*.mjs"],
    "ts": ["*.ts", "*.tsx"],
    "rust": ["*.rs"],
    "go": ["*.go"],
    "java": ["*.java"],
    "c": ["*.c", "*.h"],
    "cpp": ["*.cpp", "*.hpp", "*.cc", "*.cxx"],
    "md": ["*.md"],
    "yaml": ["*.yaml", "*.yml"],
    "json": ["*.json"],
    "toml": ["*.toml"],
    "html": ["*.html", "*.htm"],
    "css": ["*.css"],
    "sh": ["*.sh", "*.bash"],
}

# Try ripgrep first, fall back to Python.
_RG_PATH = shutil.which("rg")


# Short grep flags whose semantics we know how to express via the
# Grep tool's schema. Bundled forms (``-rln``) are split char-by-char
# and each character checked. Anything outside these sets bails.
_GREP_TRANSLATABLE_FLAGS: frozenset[str] = frozenset(
    {
        "-r",
        "-R",  # recursive (Grep tool is recursive by default)
        "-l",  # → output_mode="files_with_matches"
        "-c",  # → output_mode="count"
        "-n",  # → -n
        "-i",  # → -i
        "-E",  # extended regex (ripgrep's default is close enough)
        "-P",  # PCRE2 → Grep's pcre=True
        # Output-format flags we translate lossily: Grep tool always
        # shows filenames in content mode, so ``-h`` (no filenames)
        # loses info but the nudge is still useful. ``-H`` and ``-s``
        # are (near-)no-ops for our output shape.
        "-h",
        "-H",
        "-s",
    }
)

# Short flags that consume the next token as their value.
_GREP_VALUE_FLAGS: frozenset[str] = frozenset({"-B", "-A", "-C"})

# Long flags we translate. ``--include`` → ``glob``; ``--exclude`` →
# ``exclude``. Both forms (``--flag=VAL`` and ``--flag VAL``) are
# accepted.
_GREP_LONG_VALUE_FLAGS: frozenset[str] = frozenset({"--include", "--exclude"})
_NUDGE = "grep via Bash is a bad UX. Use the Grep tool."


class Grep:
    """Search file contents with regex patterns."""

    name: str = "Grep"
    tool_id: str = "application/x-tool-grep"
    description: str = load_tool_description("Grep")
    supports_microcompaction: bool = True
    emit_tool_summary: bool = False
    directive_schema: JSON = json_freeze(
        {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The regular expression pattern to search for in file contents",
                },
                "path": {
                    "type": "string",
                    "description": "File or directory to search in (rg PATH). Defaults to current working directory.",
                },
                "glob": {
                    "type": "string",
                    "description": 'Glob pattern to filter files (e.g. "*.js", "*.{ts,tsx}") - maps to rg --glob',
                },
                "type": {
                    "type": "string",
                    "description": "File type to search (rg --type). Common types: js, py, rust, go, java, etc.",
                },
                "output_mode": {
                    "type": "string",
                    "enum": ["content", "files_with_matches", "count"],
                    "description": 'Output mode. Defaults to "files_with_matches".',
                },
                "-B": {
                    "type": "number",
                    "minimum": 0,
                    "description": 'Number of lines to show before each match (rg -B). Requires output_mode: "content". Must be ≥ 0.',
                },
                "-A": {
                    "type": "number",
                    "minimum": 0,
                    "description": 'Number of lines to show after each match (rg -A). Requires output_mode: "content". Must be ≥ 0.',
                },
                "-C": {
                    "type": "number",
                    "minimum": 0,
                    "description": "Alias for context. Must be ≥ 0.",
                },
                "context": {
                    "type": "number",
                    "minimum": 0,
                    "description": 'Number of lines to show before and after each match (rg -C). Requires output_mode: "content". Must be ≥ 0.',
                },
                "-i": {
                    "type": "boolean",
                    "description": "Case insensitive search (rg -i)",
                },
                "-n": {
                    "type": "boolean",
                    "description": "Show line numbers in output (rg -n). Defaults to true.",
                },
                "keep_first": {
                    "type": "number",
                    "minimum": 0,
                    "description": "Keep only the first N lines/entries. Defaults to 250. Pass 0 for unlimited. Ignored when keep_last is set.",
                },
                "keep_last": {
                    "type": "number",
                    "minimum": 0,
                    "description": "Keep only the last N lines/entries. Defaults to 0 (disabled). When set, takes precedence over keep_first and offset.",
                },
                "offset": {
                    "type": "number",
                    "minimum": 0,
                    "description": "Skip first N lines/entries before applying keep_first. Defaults to 0. Ignored when keep_last is set.",
                },
                "multiline": {
                    "type": "boolean",
                    "description": "Enable multiline mode where . matches newlines and patterns can span lines (rg -U --multiline-dotall). Default: false.",
                },
                "pcre": {
                    "type": "boolean",
                    "description": (
                        "Enable PCRE2 regex (rg -P): lookaround, backrefs,"
                        " etc. Default: false (Rust regex)."
                    ),
                },
                "exclude": {
                    "type": "string",
                    "description": (
                        'Glob pattern of files to EXCLUDE (rg --glob "!PAT").'
                        ' E.g. "*.test.py".'
                    ),
                },
            },
            "required": ["pattern"],
        }
    )

    def summary(self, msg: Message) -> str:
        """Return a short label for this tool invocation.

        Args:
          msg: Incoming tool-use message.

        Returns:
          label: Human-readable summary with pattern and path.

        """
        directive = get_directive(msg)
        pattern = str(directive.get("pattern", ""))
        path = str(directive.get("path", "")) or "."
        if len(pattern) > 40:
            pattern = pattern[:37] + "..."
        suffix = f" in {path}" if path != "." else ""
        return f"Grep {pattern!r}{suffix}"

    def summary_result(self, result: Message) -> str | None:
        """One-line receipt: hit count."""
        if not self.emit_tool_summary:
            return None
        if result.descriptor != "text/plain":
            return None
        text = str(result.content).strip()
        if not text or text.startswith("(no matches"):
            return "no matches"
        # Output is one match per line; line count == match count for
        # files_with_matches and content modes both.
        return f"{text.count(chr(10)) + 1} hits"

    def prompt(self) -> str:
        """Return supplemental prompt text for this tool.

        Returns:
          prompt: Always empty.

        """
        return ""

    async def run(self, msg: Message) -> Message:
        """Search for a regex pattern in files and return matches.

        Args:
          msg: Incoming tool-use message containing the directive.

        Returns:
          result: Tool result Message with matching lines or filenames.

        """
        directive = get_directive(msg)
        # Extract known params explicitly; everything else flows through
        # as **kwargs (-B/-A/-C/-i/glob/type/pcre/exclude/context).
        known = {
            "pattern",
            "path",
            "output_mode",
            "keep_first",
            "keep_last",
            "offset",
            "multiline",
        }
        kwargs: dict[str, object] = {
            k: v for k, v in directive.items() if k not in known
        }
        return await run_sync(
            self._run,
            parent_id=msg.id,
            pattern=str(directive.get("pattern", "")),
            path=str(directive.get("path", ".")),
            output_mode=str(directive.get("output_mode", "files_with_matches")),
            keep_first=int_val(directive.get("keep_first"), 250),
            keep_last=int_val(directive.get("keep_last"), 0),
            offset=int_val(directive.get("offset"), 0),
            multiline=bool_val(directive.get("multiline"), False),
            **kwargs,
        )

    def _run(
        self,
        *,
        pattern: str = "",
        path: str = ".",
        output_mode: str = "files_with_matches",
        keep_first: int = 250,
        keep_last: int = 0,
        offset: int = 0,
        multiline: bool = False,
        **kwargs: object,  # Non-identifier params: -B, -A, -C, -i, glob, type
    ) -> str | TextMessage:
        glob_filter = _kw_str(kwargs, "glob", "glob_filter")
        file_type = _kw_str(kwargs, "type", "file_type")
        exclude = _kw_str(kwargs, "exclude")
        pcre = _kw_bool(kwargs, "pcre")
        context_before = _kw_int(kwargs, "-B", "context_before")
        context_after = _kw_int(kwargs, "-A", "context_after")
        context_symmetric = _kw_int(kwargs, "-C", "context")
        case_insensitive = _kw_bool(kwargs, "-i", "case_insensitive")
        show_line_numbers = _kw_bool(kwargs, "-n", "show_line_numbers", default=True)
        if not Path(path).is_absolute():
            path = str(Path(get_tool_state().bash_cwd) / path)
        if context_symmetric > 0:
            context_before = max(context_before, context_symmetric)
            context_after = max(context_after, context_symmetric)
        if _RG_PATH:
            return _grep_rg(
                pattern=pattern,
                path=path,
                glob_filter=glob_filter,
                file_type=file_type,
                exclude=exclude,
                pcre=pcre,
                output_mode=output_mode,
                keep_first=keep_first,
                keep_last=keep_last,
                context_before=context_before,
                context_after=context_after,
                case_insensitive=case_insensitive,
                show_line_numbers=show_line_numbers,
                multiline=multiline,
                offset=offset,
            )
        return _grep_python(
            pattern=pattern,
            path=path,
            glob_filter=glob_filter,
            file_type=file_type,
            exclude=exclude,
            output_mode=output_mode,
            keep_first=keep_first,
            keep_last=keep_last,
            context_before=context_before,
            context_after=context_after,
            case_insensitive=case_insensitive,
            show_line_numbers=show_line_numbers,
            multiline=multiline,
            offset=offset,
        )

    def bash_match(self, trees: Sequence[Node]) -> str | None:
        """Emit a tool-use nudge for a replaceable grep shape.

        Handles four shapes:

        - ``grep PATTERN [PATH]`` (simple)
        - ``cd X && grep PATTERN [PATH]``
        - ``find X … | xargs grep PATTERN`` (pipeline)
        - ``cat FILE | grep PATTERN`` (pipeline)

        Args:
          trees: Parsed shell AST nodes.

        Returns:
          nudge: Suggested Grep invocation, or ``None`` if no match.

        """
        single = _match_single_grep(trees)
        if single is not None:
            return single
        pipeline = _match_pipeline_grep(trees)
        if pipeline is not None:
            return pipeline
        return None


# Kwargs accessors for schema keys like ``-B`` / ``-A`` that aren't
# valid Python identifiers. These flow through ``**kwargs`` with
# ``object`` value type; the helpers coerce and supply defaults.


def _kw_str(
    kwargs: dict[str, object], key: str, *fallbacks: str, default: str = ""
) -> str:
    for k in (key, *fallbacks):
        v = kwargs.get(k)
        if v is not None:
            return str(v)
    return default


def _kw_int(
    kwargs: dict[str, object], key: str, *fallbacks: str, default: int = 0
) -> int:
    for k in (key, *fallbacks):
        v = kwargs.get(k)
        if v is not None:
            return int(cast(int | str, v))
    return default


def _kw_bool(
    kwargs: dict[str, object], key: str, *fallbacks: str, default: bool = False
) -> bool:
    for k in (key, *fallbacks):
        v = kwargs.get(k)
        if v is not None:
            return bool_val(v, default)
    return default


def _match_single_grep(trees: Sequence[Node]) -> str | None:
    """Match ``grep …`` (optionally prefixed by ``cd X &&``)."""
    unwrapped = unwrap_cd_prefix(trees)
    if unwrapped is None:
        return None
    _, cmd = unwrapped
    if cmd.exe != "grep" or cmd.env_prefix:
        return None
    if not _parse_grep_args(cmd.args, positional_path=True):
        return None
    return _NUDGE


def _match_pipeline_grep(trees: Sequence[Node]) -> str | None:
    """Match ``find X … | xargs grep …``, ``cat FILE | grep …``, or
    ``grep … | {head,tail,less,more,cat}``.
    """
    pair = match_pipeline(trees)
    if pair is None:
        return None
    first, second = pair

    # Shape 1: find … | xargs grep …
    if first.exe == "find" and second.exe == "xargs":
        if not _parse_find_for_grep(first.args):
            return None
        grep_args = _strip_xargs_prefix(second.args)
        if grep_args is None:
            return None
        if not _parse_grep_args(grep_args, positional_path=False):
            return None
        return _NUDGE

    # Shape 2: cat FILE | grep PATTERN
    if first.exe == "cat" and second.exe == "grep":
        if len(first.args) != 1 or first.args[0].startswith("-"):
            return None
        if not _parse_grep_args(second.args, positional_path=False):
            return None
        return _NUDGE

    # Shape 3: grep … | DISPLAY_SHAPER -- user is just truncating or
    # paginating Grep's output; Grep tool handles both natively.
    if first.exe == "grep" and second.exe in _DISPLAY_SHAPERS:
        if not _parse_grep_args(first.args, positional_path=True):
            return None
        return _NUDGE

    # Shape 4: grep … | wc -l -- line count equals Grep output_mode="count".
    if (
        first.exe == "grep"
        and second.exe == "wc"
        and second.args == ("-l",)
        and _parse_grep_args(first.args, positional_path=True)
    ):
        return _NUDGE

    return None


# Post-processors that only truncate/paginate grep's output. Piping
# grep into one of these is equivalent to calling Grep directly (it
# has its own truncation). ``wc -l`` is handled as Shape 4 above
# (not here, since other wc flags have different semantics).
# Excluded: ``sort``/``uniq``/``awk``/``sed`` (actual transforms).
_DISPLAY_SHAPERS: frozenset[str] = frozenset({"head", "tail", "less", "more", "cat"})


def _parse_grep_args(args: tuple[str, ...], *, positional_path: bool) -> bool:
    """Validate that ``args`` is a grep shape we understand.

    ``positional_path=True`` accepts 1 or 2 positionals (pattern,
    optional path). ``positional_path=False`` accepts exactly 1
    (pattern only) - used for pipeline shapes where the file list
    comes from stdin.

    Returns True for supported shapes. Extracted values are discarded:
    the nudge is the fixed string ``_NUDGE`` and the LLM re-derives its
    own Grep args from the original bash command.
    """
    positional: list[str] = []
    i = 0
    while i < len(args):
        a = args[i]
        if a == "-" or not a.startswith("-"):
            positional.append(a)
            i += 1
            continue
        if a.startswith("--"):
            name, eq, _ = a.partition("=")
            if name not in _GREP_LONG_VALUE_FLAGS:
                return False
            if eq:
                i += 1
            else:
                if i + 1 >= len(args):
                    return False
                i += 2
            continue
        if a in _GREP_VALUE_FLAGS:
            if i + 1 >= len(args):
                return False
            i += 2
            continue
        for c in a[1:]:
            if f"-{c}" not in _GREP_TRANSLATABLE_FLAGS:
                return False
        i += 1
    if positional_path:
        return len(positional) in (1, 2)
    return len(positional) == 1


# Simple xargs flags we know how to ignore (data-plumbing only, no
# effect on what ``grep`` sees beyond NUL-separated stdin).
_XARGS_PLUMBING_FLAGS: frozenset[str] = frozenset(
    {"-0", "--null", "-r", "--no-run-if-empty"}
)


def _strip_xargs_prefix(args: tuple[str, ...]) -> tuple[str, ...] | None:
    """Return the ``grep …`` tail of an ``xargs [-0|-r …] grep …`` call.

    Bails on any xargs option outside our plumbing allowlist (``-I``,
    ``-n``, ``-P``, etc. change how grep is invoked per-file, which
    doesn't round-trip to a single Grep tool call).
    """
    i = 0
    while i < len(args):
        a = args[i]
        if a in _XARGS_PLUMBING_FLAGS:
            i += 1
            continue
        if a.startswith("-"):
            return None
        if a != "grep":
            return None
        return args[i + 1 :]
    return None


def _parse_find_for_grep(args: tuple[str, ...]) -> bool:
    """Validate that ``args`` is a ``find`` shape we understand.

    Accepts ``find [PATH] [-type f|d] [-name|-iname GLOB] [-print|-print0]``.
    Whitelist-only: any predicate outside the branches below bails.
    Extracted values are discarded - the nudge is a fixed string.
    """
    seen_path = False
    i = 0
    while i < len(args):
        a = args[i]
        if a in ("-print", "-print0"):
            i += 1
            continue
        if a in ("-name", "-iname"):
            if i + 1 >= len(args):
                return False
            i += 2
            continue
        if a == "-type":
            if i + 1 >= len(args) or args[i + 1] not in {"f", "d"}:
                return False
            i += 2
            continue
        if a.startswith("-"):
            return False
        if seen_path:
            return False
        seen_path = True
        i += 1
    return True


def _grep_rg(
    *,
    pattern: str,
    path: str,
    glob_filter: str,
    file_type: str,
    exclude: str,
    pcre: bool,
    output_mode: str,
    keep_first: int,
    keep_last: int,
    context_before: int,
    context_after: int,
    case_insensitive: bool,
    show_line_numbers: bool,
    multiline: bool,
    offset: int,
) -> str | TextMessage:
    """Grep using ripgrep."""
    if offset > 0 and (context_before > 0 or context_after > 0):
        logger.warning("grep: offset ignored when context lines are requested")
        offset = 0
    cmd = _build_rg_cmd(
        pattern=pattern,
        path=path,
        glob_filter=glob_filter,
        file_type=file_type,
        exclude=exclude,
        pcre=pcre,
        output_mode=output_mode,
        context_before=context_before,
        context_after=context_after,
        case_insensitive=case_insensitive,
        show_line_numbers=show_line_numbers,
        multiline=multiline,
    )
    result = subprocess.run(  # noqa: S603 -- trusted fixed argv, not user input
        cmd,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if result.returncode >= 2:
        err = result.stderr.strip() or "unknown"
        return TextMessage(
            f"ripgrep error (exit {result.returncode}): {err}",
            "text/x-error",
        )
    lines = result.stdout.strip().split("\n")
    if keep_last > 0:
        lines = lines[-keep_last:]
    else:
        if offset > 0:
            lines = lines[offset:]
        if keep_first > 0:
            lines = lines[:keep_first]
    return "\n".join(lines) if lines and lines[0] else "(no matches)"


def _build_rg_cmd(
    *,
    pattern: str,
    path: str,
    glob_filter: str,
    file_type: str,
    exclude: str,
    pcre: bool,
    output_mode: str,
    context_before: int,
    context_after: int,
    case_insensitive: bool,
    show_line_numbers: bool,
    multiline: bool,
) -> list[str]:
    """Build the ripgrep argv."""
    assert _RG_PATH is not None
    cmd = [
        _RG_PATH,
        "--no-heading",
        "--hidden",
        "--max-columns",
        "500",
        "--glob",
        "!.git",
        "--glob",
        "!.svn",
        "--glob",
        "!.hg",
    ]
    if show_line_numbers:
        cmd.append("-n")
    if case_insensitive:
        cmd.append("-i")
    if multiline:
        cmd.extend(["-U", "--multiline-dotall"])
    if output_mode == "files_with_matches":
        cmd.append("-l")
    elif output_mode == "count":
        cmd.append("-c")
    if context_before > 0:
        cmd.extend(["-B", str(context_before)])
    if context_after > 0:
        cmd.extend(["-A", str(context_after)])
    if glob_filter:
        cmd.extend(["--glob", glob_filter])
    if exclude:
        cmd.extend(["--glob", f"!{exclude}"])
    if file_type:
        cmd.extend(["--type", file_type])
    if pcre:
        cmd.append("-P")
    cmd.extend(["--", pattern, path])
    return cmd


class _GrepState:
    """Accumulator for Python-fallback grep results."""

    __slots__ = (
        "context_after",
        "context_before",
        "file_counts",
        "matches",
        "max_results",
        "offset",
        "output_mode",
        "skipped",
    )

    def __init__(
        self,
        *,
        output_mode: str,
        offset: int,
        max_results: int,
        context_before: int,
        context_after: int,
    ) -> None:
        self.output_mode = output_mode
        self.offset = offset
        self.max_results = max_results
        self.context_before = context_before
        self.context_after = context_after
        self.matches: list[str] = []
        self.file_counts: dict[str, int] = {}
        self.skipped = 0

    @property
    def full(self) -> bool:
        return len(self.matches) >= self.max_results

    def _skip(self) -> bool:
        """Return True if this result should be skipped (offset accounting)."""
        if self.skipped < self.offset:
            self.skipped += 1
            return True
        return False

    def process_multiline(self, pat: re.Pattern[str], text: str, filepath: str) -> None:
        """Accumulate matches for one file in multiline mode."""
        found = list(pat.finditer(text))
        if not found:
            return
        if self.output_mode == "files_with_matches":
            if not self._skip():
                self.matches.append(filepath)
            return
        if self.output_mode == "count":
            self.file_counts[filepath] = len(found)
            return
        for m in found:
            if self._skip():
                continue
            line_num = text[: m.start()].count("\n") + 1
            matched_text = m.group()
            if len(matched_text) > 200:
                matched_text = matched_text[:200] + "..."
            self.matches.append(f"{filepath}:{line_num}:{matched_text}")
            if self.full:
                break

    def process_lines(
        self,
        pat: re.Pattern[str],
        lines: list[str],
        filepath: str,
    ) -> None:
        """Accumulate matches for one file in line-by-line mode."""
        file_match_count = 0
        for i, line in enumerate(lines):
            if not pat.search(line):
                continue
            file_match_count += 1
            if self.output_mode == "files_with_matches":
                if not self._skip():
                    self.matches.append(filepath)
                break
            if self.output_mode == "count":
                continue
            if self._skip():
                continue
            self._emit_line(lines, filepath, i, line)
            if self.full:
                break
        if self.output_mode == "count" and file_match_count > 0:
            self.file_counts[filepath] = file_match_count

    def _emit_line(
        self,
        all_lines: list[str],
        filepath: str,
        i: int,
        line: str,
    ) -> None:
        """Append one match (with optional context) to ``self.matches``."""
        if self.context_before > 0 or self.context_after > 0:
            start = max(0, i - self.context_before)
            end = min(len(all_lines), i + self.context_after + 1)
            for j in range(start, end):
                prefix = ">" if j == i else " "
                self.matches.append(f"{filepath}:{j + 1}:{prefix} {all_lines[j]}")
            self.matches.append("--")
        else:
            self.matches.append(f"{filepath}:{i + 1}:{line}")

    def format(self, *, keep_last: int) -> str:
        """Return final output string."""
        if self.output_mode == "count":
            items = list(self.file_counts.items())
            if keep_last > 0:
                items = items[-keep_last:]
            return "\n".join(f"{p}:{c}" for p, c in items) or "(no matches)"
        matches = self.matches
        if keep_last > 0:
            matches = matches[-keep_last:]
        return "\n".join(matches) or "(no matches)"


def _grep_python(
    *,
    pattern: str,
    path: str,
    glob_filter: str,
    file_type: str,
    exclude: str,
    output_mode: str,
    keep_first: int,
    keep_last: int,
    context_before: int,
    context_after: int,
    case_insensitive: bool,
    show_line_numbers: bool,
    multiline: bool,
    offset: int,
) -> str:
    """Grep using Python regex (fallback)."""
    del show_line_numbers
    flags = 0
    if multiline:
        flags |= re.DOTALL
    if case_insensitive:
        flags |= re.IGNORECASE
    pat = re.compile(pattern, flags)
    max_results = sys.maxsize if keep_last > 0 or keep_first <= 0 else keep_first
    state = _GrepState(
        output_mode=output_mode,
        offset=offset,
        max_results=max_results,
        context_before=context_before,
        context_after=context_after,
    )
    for f in _collect_files(Path(path), glob_filter, file_type, exclude):
        if not f.is_file():
            continue
        try:
            text = f.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        filepath = str(f)
        if multiline:
            state.process_multiline(pat, text, filepath)
        else:
            state.process_lines(pat, text.splitlines(), filepath)
        if state.full:
            break
    return state.format(keep_last=keep_last)


def _collect_files(
    root: Path,
    glob_filter: str,
    file_type: str,
    exclude: str,
) -> list[Path]:
    """Walk *root* and return files matching the glob/type/exclude filters."""
    if root.is_file():
        return [root]
    globs = _TYPE_GLOBS.get(file_type, []) if file_type else []
    if glob_filter:
        globs = [glob_filter]
    if not globs:
        globs = ["*"]
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = Path(dirpath).relative_to(root)
        if exclude:
            dirnames[:] = [d for d in dirnames if not (rel_dir / d).match(exclude)]
        dirnames.sort()
        for fname in sorted(filenames):
            fpath = Path(dirpath) / fname
            rel = fpath.relative_to(root)
            if exclude and rel.match(exclude):
                continue
            if any(rel.match(g) for g in globs):
                files.append(fpath)
    return files
