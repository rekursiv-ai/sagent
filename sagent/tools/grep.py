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

from sagent.agent.state import approx_tokens, get_tool_state
from sagent.lib.custom_json import BoolCodec, IntCodec, json_freeze
from sagent.tools.core import (
    bound_by_tokens,
    load_tool_description,
    result_token_budget,
    run_sync,
)
from sagent.tools.display import Toggle, Wrap
from sagent.tools.lib.bash import (
    Invocation,
    Node,
    bounding_sink,
    cwd_is_known,
    operands,
    parse_line_count,
    render_command,
    replaceable,
    resolve_cwd_path,
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
        "-l",  # -> output_mode="files_with_matches"
        "-c",  # -> output_mode="count"
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

# ``grep`` reads by default and has no write mode, so nothing here makes
# Bash necessary. Notably ABSENT: ``-c``, which counts matching lines and
# is ``output_mode="count"``; a denylist shared with ``head`` (where
# ``-c`` means bytes) silently dropped every ``grep -c``.
#
# ``-v`` inverts the match, and ``directive_schema`` has no property for
# that -- ``exclude`` is a PATH glob (it becomes ``rg --glob !PAT``).
# Measured on a file containing ``ERROR DEBUG b``: the pipeline
# ``grep ERROR | grep -v DEBUG`` drops that line while
# ``exclude='DEBUG'`` returns it.
#
# ``-q`` prints NOTHING and is read for its exit status -- the whole
# point of ``grep -q x f && action``. A tool call returns matches, not a
# status a later shell command can branch on, so Bash is necessary.
_GREP_DENY: frozenset[str] = frozenset({"-v", "--invert-match", "-q", "--quiet"})

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
        keep_first = IntCodec.coerce(args.get("keep_first"), 0)
        keep_last = IntCodec.coerce(args.get("keep_last"), 0)
        offset = IntCodec.coerce(args.get("offset"), 0)
        context_before = _kw_int(kwargs, "-B", "context_before")
        context_after = _kw_int(kwargs, "-A", "context_after")
        context_symmetric = _kw_int(kwargs, "-C", "context")
        # Schema declares all pagination/context knobs as ``minimum: 0``
        # integers but ``IntCodec.coerce`` accepts negatives, which then index
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
            multiline=BoolCodec.coerce(args.get("multiline"), False),
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
        # Checked before dispatch so both backends agree. ``rg`` exits 2
        # on an unknown type; the fallback's ``.get`` miss fell through to
        # ``["*"]`` and searched the whole tree instead.
        if file_type and file_type not in _TYPE_GLOBS:
            return ToolResult(
                call_id="",
                content=(
                    f"unknown type: {file_type!r}"
                    f" (expected one of {sorted(_TYPE_GLOBS)})"
                ),
                is_error=True,
            )
        # Checked once, before dispatch: rg exits 2 on a missing path
        # while the fallback walked it and reported ``(no matches)`` --
        # "found nothing" and "no such path" are different answers.
        if not Path(path).exists():
            return ToolResult(
                call_id="",
                content=f"no such file or directory: {path}",
                is_error=True,
            )
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

    def bash_match(self, trees: Sequence[Node]) -> str | None:
        """Emit a tool-use nudge for a replaceable grep shape.

        Detection is :func:`replaceable`; this decides only which
        executables Grep claims. ``find … | xargs grep`` is claimed too:
        the search is the xargs payload and the find half only
        enumerates what to search.

        Args:
          trees: Parsed bash command-trees from the Bash directive.

        Returns:
          nudge: Suggestion text when the shape is replaceable, else ``None``.

        """
        for inv in walk_commands(trees):
            if replaceable(inv, exes=_GREP_EXES, deny=_GREP_DENY) or (
                inv.exe == "xargs" and _xargs_searches_files(inv)
            ):
                return _nudge_for(inv)
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
            return BoolCodec.coerce(v, default)
    return default


def _nudge_for(inv: Invocation) -> str:
    """Render the nudge, with concrete Grep arguments when translatable.

    Translation runs AFTER detection and may fail freely: an untranslated
    flag costs the caller a worked example, not the nudge. Gating
    detection on it instead made every flag outside
    :data:`_GREP_TRANSLATABLE_FLAGS` -- most of grep's ~80 -- silent.

    The translated args come from the SEARCH, which is not always
    ``inv``: an ``xargs`` payload carries the search behind the wrapper,
    and a stdin-fed search's path sits on the producer feeding it.
    Translating ``inv.args`` blindly rendered ``xargs grep pat`` as
    ``pattern='grep' path='pat'``.
    """
    args = _search_args(inv)
    fields = _translate_grep_args(args, cwd=inv.cwd) if args is not None else ""
    if fields:
        fields += _sink_fields(inv)
    call = f" Try: Grep {fields}" if fields else ""
    return f"{_NUDGE} Replaces: `{render_command(inv)}`.{call}"


def _search_args(inv: Invocation) -> tuple[str, ...] | None:
    """Argv of the SEARCH itself, with its path operand resolved.

    Returns ``None`` when the operand cannot be recovered, so the caller
    drops the worked example rather than inventing one.
    """
    if inv.exe == "xargs":
        payload = _strip_xargs_prefix(inv.args)
        source = inv.piped_from
        if payload is None or source is None:
            return None
        # ``find``'s own vocabulary: its operands are the ROOTS, and a
        # predicate's value (``-name '*.py'``) is not one of them.
        roots = operands("find", source.args)
        if len(roots) > 1:
            return None
        # ``find -name '*.py'`` omits the root, which defaults to the
        # cwd. Reading the first non-flag token as the root instead named
        # the GLOB as the path.
        root = roots[0] if roots else "."
        name = next(
            (source.args[i + 1] for i, a in enumerate(source.args) if a == "-name"),
            "",
        )
        # ``-name`` is optional: ``find /src -type f`` names a root and
        # nothing else, which is a Grep call with no ``glob``.
        return (*payload, root, *(("--include", name) if name else ()))
    if inv.piped_from is None:
        return inv.args
    # ``cat f | grep p``: the pattern is here, the path one hop upstream.
    # Ask through ``operands`` so a producer's flag VALUE is not counted
    # as a second path -- ``head -n 20 f`` spends its ``20`` on the flag.
    source = inv.piped_from
    paths = [a for a in operands(source.exe, source.args) if not _is_sed_script(a)]
    if len(paths) != 1:
        return None
    return (*inv.args, paths[0])


def _is_sed_script(arg: str) -> bool:
    """Whether ``arg`` is a bare ``sed`` script rather than a path.

    ``sed -n '1,50p' f`` puts the script in operand position, so a
    producer's path cannot be recovered by counting operands alone.
    """
    return bool(_SED_SCRIPT.fullmatch(arg))


# A ``sed`` address plus command (``5p``, ``1,50p``, ``10,$p``) or a
# substitution -- the spellings that appear where a path would.
_SED_SCRIPT: Final = re.compile(r"\d+(,(\d+|\$))?[a-z]|s/.*/.*/[a-z]*")


def _sink_fields(inv: Invocation) -> str:
    """Render the downstream stages ``replaceable`` folded into this call.

    ``_sink_blocks`` accepts ``| wc -l`` precisely because Grep expresses
    it as ``output_mode="count"``; omitting it advertises a different
    search than the one replaced.

    Notably ABSENT: ``| grep -v X``. Measured on a file whose line reads
    ``ERROR DEBUG b``: the pipeline drops that line, while
    ``exclude='DEBUG'`` returns it -- ``exclude`` is a PATH glob passed
    to ``rg --glob !PAT``, not a line filter.
    """
    fields = "".join(
        ' output_mode="count"'
        for sink in inv.downstream()
        if sink.exe == "wc" and sink.args == ("-l",)
    )
    # ``| head -5`` bounds the search, and Grep says that directly. A
    # suggestion without it returns the whole match set.
    sink = bounding_sink(inv)
    count = parse_line_count(sink.args) if sink is not None else None
    if count is not None:
        fields += (
            f" keep_last={count}"
            if sink and sink.exe == "tail"
            else (f" keep_first={count}")
        )
    return fields


def _translate_grep_args(args: tuple[str, ...], *, cwd: str = "") -> str:
    """Render ``args`` as Grep tool keywords, or ``""`` when unsupported.

    ``cwd`` is the enclosing ``cd`` prefix. Grep resolves a relative
    ``path`` against the AGENT's cwd, not the shell's, so dropping it
    searches a different tree than the command being replaced.
    """
    fields: list[str] = []
    positional: list[str] = []
    patterns: list[str] = []
    i = 0
    while i < len(args):
        a = args[i]
        if a == "-" or not a.startswith("-"):
            positional.append(a)
            i += 1
            continue
        if a in ("-e", "--regexp"):
            # ``-e`` NAMES the pattern -- the only spelling that works
            # for a pattern beginning with a dash. Skipping it as an
            # ordinary flag value left the search with no pattern at all.
            if i + 1 >= len(args):
                return ""
            patterns.append(args[i + 1])
            i += 2
            continue
        if a.startswith("--"):
            name, eq, value = a.partition("=")
            if name not in _GREP_LONG_VALUE_FLAGS:
                return ""
            if not eq:
                if i + 1 >= len(args):
                    return ""
                value = args[i + 1]
            fields.append(
                f"glob={value!r}" if name == "--include" else f"exclude={value!r}"
            )
            i += 1 if eq else 2
            continue
        if a in _GREP_VALUE_FLAGS:
            if i + 1 >= len(args):
                return ""
            # Bare, not quoted: the schema property IS ``-B``, and the
            # sibling booleans already render ``-i=true``. ``"-B"=3``
            # parses as nothing a caller can paste.
            fields.append(f"{a}={args[i + 1]}")
            i += 2
            continue
        for c in a[1:]:
            if f"-{c}" not in _GREP_TRANSLATABLE_FLAGS:
                return ""
            if c == "l":
                fields.append('output_mode="files_with_matches"')
            elif c == "c":
                fields.append('output_mode="count"')
            elif c == "i":
                fields.append("-i=true")
            elif c == "P":
                fields.append("pcre=true")
        i += 1
    # With ``-e`` the pattern is already named, so every positional is a
    # path; without it the first positional is the pattern.
    if patterns:
        pattern, paths = "|".join(patterns), positional
    elif positional:
        pattern, paths = positional[0], positional[1:]
    else:
        return ""
    if len(paths) > 1:
        return ""
    head = f"pattern={pattern!r}"
    target = resolve_cwd_path(cwd, paths[0] if paths else None)
    if paths and not cwd_is_known(cwd):
        # ``cd``/``cd -`` moved somewhere the text does not name, so the
        # operand cannot be resolved and a pattern-only call would search
        # a different tree than the command it claims to replace.
        return ""
    if target:
        head += f" path={target!r}"
    return " ".join([head, *fields])


def _xargs_searches_files(inv: Invocation) -> bool:
    """Whether ``find … | xargs grep …`` is one Grep call.

    ``xargs`` runs an arbitrary command, so this shape is recognised by
    its payload rather than by :func:`replaceable`: the operand lives on
    the ``find`` half, which only enumerates what to search.
    """
    if any(d.captures_stdout for d in inv.downstream()):
        return False
    grep_args = _strip_xargs_prefix(inv.args)
    if grep_args is None:
        return False
    source = inv.piped_from
    if source is None or source.exe != "find":
        return False
    return (
        _parse_find_for_grep(source.args)
        and len([a for a in grep_args if not a.startswith("-")]) == 1
    )


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
        if a == "-name":
            if i + 1 >= len(args):
                return False
            i += 2
            continue
        if a == "-iname":
            # Case-INsensitive. Grep's ``glob`` is case-sensitive, so
            # rendering the same pattern asks a different question.
            return False
        if a == "-type":
            # Only ``f``. ``-type d`` feeds xargs DIRECTORIES, so the
            # search recurses each one -- a shape Grep's single ``path``
            # cannot express, and the nudge named the pattern as a path.
            if i + 1 >= len(args) or args[i + 1] != "f":
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

    ``offset`` applies with context lines too. Both backends used to zero
    it and prepend an "offset ignored" notice, which left the reply both
    ignoring the knob AND telling the reader to pass it -- a resume note
    that cannot be followed. Slicing rendered lines is agnostic to how
    they were produced, so context rows page like any others; a group
    separator inside the window is simply one more entry.

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
    # Where the shown window BEGINS in the full match set. Every resume
    # note is phrased from this, never from the caller's ``offset``:
    # ``keep_last`` slices a tail and leaves ``offset`` at 0, so a note
    # built from ``offset`` named a position at the HEAD while the body
    # showed the TAIL -- following it re-fetched rows already shown and
    # never reached the omitted ones.
    if keep_last > 0:
        start = max(0, len(lines) - keep_last)
        lines = lines[start:]
    else:
        start = offset
        if offset > 0:
            lines = lines[offset:]
        if keep_first > 0:
            lines = lines[:keep_first]
    if not lines or not lines[0]:
        return "(no matches)"
    # The token bound is the backstop for an unpaginated search: the
    # caller's knobs are a window, and neither says how wide a match is.
    # The resume note is part of the result, so its own cost comes out of
    # the budget -- appending it afterwards put the reply back over.
    budget = result_token_budget()
    # Reserve the note by rendering the REAL one for the worst case (all
    # entries withheld), not a second hand-written copy: a stand-in with
    # placeholder values drifts from what actually ships.
    reserved = _resume_note(withheld=len(lines), resume=start + len(lines))
    body, kept = bound_by_tokens(
        (f"{line}\n" for line in lines),
        budget=max(1, budget - approx_tokens(reserved)),
    )
    body = body.rstrip("\n")
    if kept < len(lines):
        body += _resume_note(withheld=len(lines) - kept, resume=start + kept)
    return body


def _resume_note(*, withheld: int, resume: int) -> str:
    """Render the continuation note for entries this reply did not show."""
    return f"\n... ({withheld} more entries; pass offset={resume} to continue)"


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
    try:
        result = subprocess.run(  # noqa: S603 -- trusted fixed argv, not user input
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except subprocess.TimeoutExpired:
        # ``Tool.run`` returns failures; a raised exception reaches the
        # agent loop as a crash rather than a tool error.
        return ToolResult(
            call_id="",
            content="ripgrep timed out after 30s; narrow the path or pattern.",
            is_error=True,
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
        hits = [i for i, line in enumerate(lines) if pat.search(line)]
        if not hits:
            return
        if self.output_mode == "files_with_matches":
            self.matches.append(filepath)
            return
        if self.output_mode == "count":
            self.file_counts[filepath] = len(hits)
            return
        if self.context_before <= 0 and self.context_after <= 0:
            for i in hits:
                self._append_content_line(filepath, i, lines[i])
            return
        for start, end in self._context_groups(hits, len(lines)):
            for j in range(start, end):
                self._append_content_line(filepath, j, lines[j])
            self.matches.append("--")

    def _context_groups(self, hits: list[int], total: int) -> list[tuple[int, int]]:
        """Merge each match's context window with its overlapping neighbours.

        One group per match repeated the shared lines: two matches a line
        apart printed the overlap twice, so a caller counting occurrences
        in the output counted them twice. ripgrep merges instead.
        """
        groups: list[tuple[int, int]] = []
        for i in hits:
            start = max(0, i - self.context_before)
            end = min(total, i + self.context_after + 1)
            if groups and start <= groups[-1][1]:
                groups[-1] = (groups[-1][0], max(groups[-1][1], end))
                continue
            groups.append((start, end))
        return groups

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
    pcre: bool = False,
) -> str | ToolResult:
    """Grep using Python regex (fallback)."""
    if pcre:
        # The schema promises PCRE2 (lookaround, backrefs). Python ``re``
        # is a different language, and silently substituting it returns
        # results under a name that does not describe them.
        return ToolResult(
            call_id="",
            content=(
                "pcre=true requires ripgrep, which is not installed;"
                " the Python fallback cannot provide PCRE2 semantics."
            ),
            is_error=True,
        )
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
        state.format(),
        keep_first=keep_first,
        keep_last=keep_last,
        offset=offset,
    )


def _collect_files(
    root: Path,
    glob_filter: str,
    file_type: str,
    exclude: str,
) -> list[Path]:
    """Walk *root* and return files matching the glob/type/exclude filters."""
    globs = _TYPE_GLOBS[file_type] if file_type else []
    if glob_filter:
        globs = _expand_braces(glob_filter)
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


def _expand_braces(glob: str) -> list[str]:
    """Expand one ``{a,b}`` alternation into separate globs.

    ``Path.match`` has no brace syntax, so the ``"*.{ts,tsx}"`` the schema
    advertises matched NOTHING in this backend while ripgrep matched both
    extensions. One group is enough for the documented shape; a glob with
    several is passed through and simply matches literally, as before.
    """
    before, brace, rest = glob.partition("{")
    body, close, after = rest.partition("}")
    if not brace or not close or "{" in after:
        return [glob]
    return [f"{before}{alt}{after}" for alt in body.split(",")]


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
