"""Grep tool: ripgrep-first content search with Python fallback."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Final

import logging
import os
import re
import shutil
import subprocess

from sagent.agent.state import current_agent_var, get_tool_state
from sagent.lib.custom_json import bool_val, int_val, json_freeze
from sagent.tools.core import load_tool_description, run_sync
from sagent.tools.display import Toggle, Wrap
from sagent.tools.lib.bash import (
    Invocation,
    Node,
    walk_commands,
)
from sagent.tools.tool_spec import CLI_SETTABLE
from sagent.types.runtime import ToolResult


logger = logging.getLogger(__name__)

# File type extensions for grep --type filter.
_TYPE_GLOBS: Final[dict[str, list[str]]] = {
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

# Bash executables we redirect to the Grep tool. ``rg`` shares grep's
# basic shape (``rg PATTERN [PATH]`` with the same -i/-n/-l/-c/-A/-B/-C
# flags), so the same parsers apply; ripgrep-only flags like ``-U`` or
# ``-t`` fall through and the nudge bails on those shapes.
_GREP_EXES: frozenset[str] = frozenset({"grep", "rg"})
_NUDGE: Final = "grep/rg via Bash is a bad UX. Use the Grep tool."

# Entry cap when no agent is in context (standalone use, tests).
_FALLBACK_KEEP_FIRST: Final = 1_000

# Characters a rendered match line costs, used to turn the agent's
# character budget into an entry count. Set above a typical
# ``path:line:text`` row so the derived cap errs small.
_ASSUMED_CHARS_PER_MATCH: Final = 150


def _default_keep_first() -> int:
    """Entry cap for an unpaginated Grep, derived from the active budget.

    A result over ``max_result_chars`` is off-loaded to disk or elided,
    and either outcome returns less than a bounded reply would -- so an
    "unlimited" default silently lost matches on a wide pattern. The
    derived cap keeps one reply whole; ``offset`` reaches the rest.

    Returns:
      limit: Maximum entries returned by default; the fallback constant
          when no agent is in context.

    """
    agent = current_agent_var.get(None)
    ceiling = agent.max_result_chars if agent is not None else 0
    if ceiling <= 0:
        return _FALLBACK_KEEP_FIRST
    return max(_FALLBACK_KEEP_FIRST, ceiling // _ASSUMED_CHARS_PER_MATCH)


# Mirrors the ``output_mode`` enum advertised in ``directive_schema``.
# Validated at runtime so an unknown value errors instead of silently
# behaving like ``files_with_matches``.
_OUTPUT_MODES: frozenset[str] = frozenset({"content", "files_with_matches", "count"})


@dataclass(frozen=True, slots=True, kw_only=True)
class Grep:
    """Search file contents with regex patterns."""

    name = "Grep"
    tool_id = "application/x-tool-grep"
    clearable_results = True
    description = load_tool_description("Grep")
    directive_schema = json_freeze(
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
                    "type": "integer",
                    "minimum": 0,
                    "description": 'Number of lines to show before each match (rg -B). Requires output_mode: "content". Must be ≥ 0.',
                },
                "-A": {
                    "type": "integer",
                    "minimum": 0,
                    "description": 'Number of lines to show after each match (rg -A). Requires output_mode: "content". Must be ≥ 0.',
                },
                "-C": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Alias for context. Must be ≥ 0.",
                },
                "context": {
                    "type": "integer",
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
                    "type": "integer",
                    "minimum": 0,
                    "description": "Keep only the first N lines/entries. Omit for a budget-derived default; 0 means unlimited. Ignored when keep_last is set.",
                },
                "keep_last": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Keep only the last N lines/entries. Defaults to 0 (disabled). When set, takes precedence over keep_first and offset.",
                },
                "offset": {
                    "type": "integer",
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

    output: Annotated[Toggle, CLI_SETTABLE] = "off"
    """Whether the result body renders in the pane."""

    output_head_rows: Annotated[int, CLI_SETTABLE] = 2
    """Leading body rows kept."""

    output_tail_rows: Annotated[int, CLI_SETTABLE] = 2
    """Trailing body rows kept, after a ``⋯ N lines ⋯`` marker."""

    output_max_width: Annotated[int, CLI_SETTABLE] = 0
    """Cell width cap; ``0`` uses the pane width."""

    output_wrap: Annotated[Wrap, CLI_SETTABLE] = "wrap"
    """``wrap`` continues an over-wide line, ``chop`` marks the cut."""

    def summary(self, args: Mapping[str, object]) -> str:
        """Return a short label for this tool invocation.

        Args:
          args: Parsed tool directive mapping.

        Returns:
          label: Compact one-line label for renderer display.

        """
        pattern = str(args.get("pattern", ""))
        path = str(args.get("path", "")) or "."
        suffix = f" in {path}" if path != "." else ""
        return f"Grep {pattern!r}{suffix}"

    def prompt(self) -> str:
        """Return supplemental prompt text for this tool.

        Returns:
          text: Supplemental prompt text; empty for Grep.

        """
        return ""

    def serialize_key(self, args: Mapping[str, object]) -> str | None:
        """Run in parallel: read-only search needs no serialization."""
        del args
        return None

    async def run(self, args: Mapping[str, object]) -> ToolResult:
        """Search for a regex pattern in files and return matches.

        Args:
          args: Parsed tool directive mapping.

        Returns:
          result: ``ToolResult`` carrying matches (or counts/filenames)
            per ``output_mode``.

        """
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
        kwargs: dict[str, object] = {k: v for k, v in args.items() if k not in known}
        keep_first = int_val(args.get("keep_first"), _default_keep_first())
        keep_last = int_val(args.get("keep_last"), 0)
        offset = int_val(args.get("offset"), 0)
        context_before = _kw_int(kwargs, "-B", "context_before")
        context_after = _kw_int(kwargs, "-A", "context_after")
        context_symmetric = _kw_int(kwargs, "-C", "context")
        # Schema declares all pagination/context knobs as ``minimum: 0``
        # integers but ``int_val`` accepts negatives, which then index
        # from the end of the result list (``lines[-N:]`` returns the
        # tail instead of failing). Enforce the schema floor here so a
        # malformed directive surfaces as a tool error rather than
        # mystery output.
        bounds_err = _check_nonnegative(
            ("keep_first", keep_first, args.get("keep_first")),
            ("keep_last", keep_last, args.get("keep_last")),
            ("offset", offset, args.get("offset")),
            ("-B", context_before, args.get("-B")),
            ("-A", context_after, args.get("-A")),
            ("-C", context_symmetric, args.get("-C") or args.get("context")),
        )
        if bounds_err is not None:
            return bounds_err
        return await run_sync(
            self._run,
            pattern=str(args.get("pattern", "")),
            path=str(args.get("path", ".")),
            output_mode=str(args.get("output_mode", "files_with_matches")),
            keep_first=keep_first,
            keep_last=keep_last,
            offset=offset,
            multiline=bool_val(args.get("multiline"), False),
            **kwargs,
        )

    def _run(
        self,
        *,
        pattern: str = "",
        path: str = ".",
        output_mode: str = "files_with_matches",
        keep_first: int = 0,
        keep_last: int = 0,
        offset: int = 0,
        multiline: bool = False,
        **kwargs: object,  # Non-identifier params: -B, -A, -C, -i, glob, type
    ) -> str | ToolResult:
        """Dispatch the grep search to ripgrep or the Python fallback."""
        if output_mode not in _OUTPUT_MODES:
            return ToolResult(
                call_id="",
                content=(
                    f"unknown output_mode: {output_mode!r}"
                    f" (expected one of {sorted(_OUTPUT_MODES)})"
                ),
                is_error=True,
            )
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

        Policy only: :func:`walk_commands` supplies every simple command
        with its context, so a leading ``cd``, an enclosing loop, and a
        trailing ``| head`` all reach this matcher without it re-deriving
        AST shape -- the omission that left ``cd X && grep p f | head``
        silent while ``grep p f | head`` nudged.

        Args:
          trees: Parsed bash command-trees from the Bash directive.

        Returns:
          nudge: Suggestion text when the shape is replaceable, else ``None``.

        """
        for inv in walk_commands(trees):
            if inv.env_prefix or inv.captures_stdout:
                continue
            if _searches_files(inv):
                return _NUDGE
        return None


# Kwargs accessors for schema keys like ``-B`` / ``-A`` that aren't
# valid Python identifiers. These flow through ``**kwargs`` with
# ``object`` value type; the helpers coerce and supply defaults.


def _kw_str(
    kwargs: dict[str, object], key: str, *fallbacks: str, default: str = ""
) -> str:
    """Coerce the first non-None kwargs entry among aliases to a string."""
    for k in (key, *fallbacks):
        v = kwargs.get(k)
        if v is not None:
            return str(v)
    return default


def _kw_int(
    kwargs: dict[str, object], key: str, *fallbacks: str, default: int = 0
) -> int:
    """Coerce the first non-None kwargs entry among aliases to an int.

    Unparseable values fall back to ``default`` rather than raising:
    ``Tool.run`` must not raise, and the schema gate already rejects
    non-integers on the production path. This keeps a direct ``_run``
    caller (tests, internal reuse) from escaping the tool envelope --
    the same defense-in-depth ``Read._check_minimum`` provides.
    """
    for k in (key, *fallbacks):
        v = kwargs.get(k)
        if v is None:
            continue
        if isinstance(v, bool):
            return default
        if isinstance(v, (int, float)):
            return int(v)
        if isinstance(v, str):
            try:
                return int(v)
            except ValueError:
                return default
        return default
    return default


def _kw_bool(
    kwargs: dict[str, object], key: str, *fallbacks: str, default: bool = False
) -> bool:
    """Coerce the first non-None kwargs entry among aliases to a bool."""
    for k in (key, *fallbacks):
        v = kwargs.get(k)
        if v is not None:
            return bool_val(v, default)
    return default


def _shaping_sink(sink: Invocation) -> bool:
    """Whether the sink only truncates or counts what grep already found.

    ``wc -l`` is Grep's ``output_mode="count"``; ``wc -c`` counts BYTES,
    which Grep cannot express, so that shape stays with Bash.
    """
    if sink.exe in _DISPLAY_SHAPERS:
        return True
    return sink.exe == "wc" and sink.args == ("-l",)


def _searches_files(inv: Invocation) -> bool:
    """Whether this invocation is a search the Grep tool replaces.

    Every question is answered from ``inv``'s own pipeline links. Asking
    it of the whole command line instead let an unrelated statement --
    ``cat -n a.py; grep -n p f`` -- decide the verdict.
    """
    if any(d.captures_stdout for d in inv.downstream()):
        # Something downstream writes a file, so the pipeline's product
        # is the file, not the matches.
        return False
    if inv.exe in _GREP_EXES:
        # A ``cat -n f | grep p`` source is adding line numbers, not
        # simply feeding the file; that is not the shape Grep replaces.
        source = inv.piped_from
        if (
            source is not None
            and source.exe == "cat"
            and any(a.startswith("-") for a in source.args)
        ):
            return False
        # A pipeline sink that only truncates or counts is what Grep's
        # own paging and ``output_mode="count"`` already do.
        if inv.piped_into is not None and not _shaping_sink(inv.piped_into):
            return False
        return _parse_grep_args(inv.args, positional_path=True)
    if inv.exe == "xargs":
        # ``find … | xargs grep …``: the search is the xargs payload,
        # and the find half only enumerates what to search.
        grep_args = _strip_xargs_prefix(inv.args)
        if grep_args is None:
            return False
        source = inv.piped_from
        if source is None or source.exe != "find":
            return False
        return _parse_find_for_grep(source.args) and _parse_grep_args(
            grep_args, positional_path=False
        )
    return False


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
    """Return the search-command tail of ``xargs [-0|-r …] {grep,rg} …``.

    Bails on any xargs option outside our plumbing allowlist (``-I``,
    ``-n``, ``-P``, etc. change how the search is invoked per-file,
    which doesn't round-trip to a single Grep tool call).
    """
    i = 0
    while i < len(args):
        a = args[i]
        if a in _XARGS_PLUMBING_FLAGS:
            i += 1
            continue
        if a.startswith("-"):
            return None
        if a not in _GREP_EXES:
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


def _paginate(text: str, *, keep_first: int, keep_last: int, offset: int) -> str:
    """Apply the pagination knobs to already-rendered output.

    The single place either backend slices. Both produce one entry per
    line for every ``output_mode``, so slicing here means ``offset`` and
    ``keep_first`` mean the same thing in ripgrep and in the fallback --
    and in ``content``, ``count``, and ``files_with_matches`` alike.
    Slicing inside the accumulator instead let the unit differ per mode
    (content lines vs. per-file counts vs. context separators).

    Args:
      text: Rendered output, one entry per line.
      keep_first: Keep only the leading N entries; ``0`` is unlimited.
      keep_last: Keep only the trailing N entries; takes precedence.
      offset: Skip N leading entries before ``keep_first``.

    Returns:
      paginated: The selected entries, or ``(no matches)`` when empty.

    """
    if not text or text == "(no matches)":
        return text or "(no matches)"
    lines = text.split("\n")
    if keep_last > 0:
        lines = lines[-keep_last:]
    else:
        if offset > 0:
            lines = lines[offset:]
        if keep_first > 0:
            lines = lines[:keep_first]
    return "\n".join(lines) if lines and lines[0] else "(no matches)"


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
) -> str | ToolResult:
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
        if not multiline and 'the literal "\\n" is not allowed' in err:
            err = (
                "pattern references a newline but multiline is off. "
                'Pass multiline=true to match across lines (literal "\\n" '
                "or `.` spanning newlines)."
            )
        return ToolResult(
            call_id="",
            content=f"ripgrep error (exit {result.returncode}): {err}",
            is_error=True,
        )
    return _paginate(
        result.stdout.strip(),
        keep_first=keep_first,
        keep_last=keep_last,
        offset=offset,
    )


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
        # Deterministic order. ripgrep's default parallel walk emits files
        # in whatever order the workers finish, so ``offset`` selects a
        # different slice run to run -- and the Python fallback, which
        # walks ``sorted()``, disagrees with it on every query.
        "--sort",
        "path",
        # No column cap. ``--max-columns`` replaces a long line with a
        # placeholder, and ``--max-columns-preview`` only restores its
        # LEADING columns -- a match further right (a needle in a minified
        # bundle) stays invisible, while the Python fallback returns the
        # line whole. Total size is bounded downstream by the tool-result
        # cap, which says what it dropped.
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
        "output_mode",
        "show_line_numbers",
    )

    def __init__(
        self,
        *,
        output_mode: str,
        context_before: int,
        context_after: int,
        show_line_numbers: bool,
    ) -> None:
        self.output_mode = output_mode
        self.context_before = context_before
        self.context_after = context_after
        self.show_line_numbers = show_line_numbers
        self.matches: list[str] = []
        self.file_counts: dict[str, int] = {}

    def process_multiline(self, pat: re.Pattern[str], text: str, filepath: str) -> None:
        """Accumulate matches for one file in multiline mode.

        Args:
          pat: Compiled regex applied to the full file body.
          text: File contents (entire text used for cross-line matches).
          filepath: Path string used in result lines.

        """
        found = list(pat.finditer(text))
        if not found:
            return
        if self.output_mode == "files_with_matches":
            self.matches.append(filepath)
            return
        if self.output_mode == "count":
            self.file_counts[filepath] = len(found)
            return
        for m in found:
            line_num = text[: m.start()].count("\n") + 1
            matched_text = m.group()
            if self.show_line_numbers:
                self.matches.append(f"{filepath}:{line_num}:{matched_text}")
            else:
                self.matches.append(f"{filepath}:{matched_text}")

    def process_lines(
        self,
        pat: re.Pattern[str],
        lines: list[str],
        filepath: str,
    ) -> None:
        """Accumulate matches for one file in line-by-line mode.

        Args:
          pat: Compiled regex applied per-line.
          lines: Pre-split file contents (one entry per line).
          filepath: Path string used in result lines.

        """
        file_match_count = 0
        for i, line in enumerate(lines):
            if not pat.search(line):
                continue
            file_match_count += 1
            if self.output_mode == "files_with_matches":
                self.matches.append(filepath)
                break
            if self.output_mode == "count":
                continue
            self._emit_line(lines, filepath, i, line)
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
                self._append_content_line(filepath, j, all_lines[j])
            self.matches.append("--")
        else:
            self._append_content_line(filepath, i, line)

    def _append_content_line(self, filepath: str, i: int, line: str) -> None:
        if self.show_line_numbers:
            self.matches.append(f"{filepath}:{i + 1}:{line}")
        else:
            self.matches.append(f"{filepath}:{line}")

    def format(self) -> str:
        """Render accumulated results, one entry per line.

        Pagination is applied afterwards by :func:`_paginate`, uniformly
        with the ripgrep path.

        Returns:
          text: Newline-joined output, or ``(no matches)`` when empty.

        """
        if self.output_mode == "count":
            counts = self.file_counts.items()
            return "\n".join(f"{p}:{c}" for p, c in counts) or "(no matches)"
        return "\n".join(self.matches) or "(no matches)"


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
) -> str | ToolResult:
    """Grep using Python regex (fallback)."""
    if offset > 0 and (context_before > 0 or context_after > 0):
        logger.warning("grep: offset ignored when context lines are requested")
        offset = 0
    if not multiline and r"\n" in pattern:
        return ToolResult(
            call_id="",
            content=(
                "pattern references a newline but multiline is off. "
                'Pass multiline=true to match across lines (literal "\\n" '
                "or `.` spanning newlines)."
            ),
            is_error=True,
        )
    flags = 0
    if multiline:
        flags |= re.DOTALL
    if case_insensitive:
        flags |= re.IGNORECASE
    try:
        pat = re.compile(pattern, flags)
    except re.error as exc:
        return ToolResult(
            call_id="",
            content=f"ripgrep error (Python fallback): invalid regex pattern: {exc}",
            is_error=True,
        )
    state = _GrepState(
        output_mode=output_mode,
        context_before=context_before,
        context_after=context_after,
        show_line_numbers=show_line_numbers,
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
    return _paginate(
        state.format(), keep_first=keep_first, keep_last=keep_last, offset=offset
    )


def _collect_files(
    root: Path,
    glob_filter: str,
    file_type: str,
    exclude: str,
) -> list[Path]:
    """Walk *root* and return files matching the glob/type/exclude filters."""
    globs = _TYPE_GLOBS.get(file_type, []) if file_type else []
    if glob_filter:
        globs = [glob_filter]
    if not globs:
        globs = ["*"]
    if root.is_file():
        return [root] if _path_matches(root.name, globs, exclude) else []
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = Path(dirpath).relative_to(root)
        dirnames[:] = [
            dirname
            for dirname in dirnames
            if dirname not in {".git", ".svn", ".hg"}
            and not (exclude and (rel_dir / dirname).match(exclude))
        ]
        dirnames.sort()
        for fname in sorted(filenames):
            fpath = Path(dirpath) / fname
            rel = fpath.relative_to(root)
            if _path_matches(str(rel), globs, exclude):
                files.append(fpath)
    return files


def _path_matches(path: str, globs: Sequence[str], exclude: str) -> bool:
    rel = Path(path)
    if exclude and rel.match(exclude):
        return False
    return any(rel.match(glob) for glob in globs)


def _check_nonnegative(
    *fields: tuple[str, int, object],
) -> ToolResult | None:
    """Reject schema-violating negative knobs at the tool entrypoint.

    Each tuple is ``(name, coerced, raw)``: when the caller supplied
    ``raw`` (anything but ``None``) but the coerced int is negative,
    surface a tool error instead of letting it index from the end of
    a result slice downstream.
    """
    for name, coerced, raw in fields:
        if raw is None:
            continue
        if coerced < 0:
            return ToolResult(
                call_id="",
                content=f"'{name}' must be ≥ 0, got {coerced}.",
                is_error=True,
            )
    return None
