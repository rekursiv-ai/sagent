"""Glob tool: fast path-pattern matching."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Final

import time

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
    FIND_DENY_FLAGS,
    Node,
    bounding_sink,
    parse_line_count,
    render_command,
    replaceable,
    resolve_cwd_path,
    walk_commands,
)
from sagent.tools.lib.path_sort import (
    SORT_VALUES,
    safe_mtime,
    safe_size,
    sort_paths,
)
from sagent.tools.tool_spec import CLI_SETTABLE
from sagent.types.runtime import ToolResult


_NUDGE: Final = "find via Bash is a bad UX. Use the Glob tool."

_FIND_EXES: frozenset[str] = frozenset({"find"})

_DEFAULT_SORT: Final = "name"


@dataclass(frozen=True, slots=True, kw_only=True)
class Glob:
    """Match file paths against glob patterns.

    Differences vs the List tool (when both could apply to "what's in
    DIR?"):
      * Glob returns full resolved paths; List returns basenames.
      * Glob does not append ``/`` to directories; List does.
      * Glob's pattern controls dotfile inclusion (``*`` excludes,
        ``.*`` matches only). List has an explicit ``show_hidden``
        toggle that returns visible + hidden in a single call -- Glob
        requires two calls and a merge.
      * Glob returns ``(no matches)`` for missing path, non-directory
        path, and empty directory alike. List distinguishes the three
        (``Not found:``, ``Not a directory:``, ``(empty directory)``).
      * Glob silently returns no matches when given a non-directory
        ``path``; List errors.

    Use List for "show me this directory" inspection. Use Glob for
    pattern matching across a tree (especially recursive ``**/*.py``).
    """

    name = "Glob"
    tool_id = "application/x-tool-glob"
    clearable_results = True
    description = load_tool_description("Glob")
    directive_schema = json_freeze(
        {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The glob pattern to match files against",
                },
                "path": {
                    "type": "string",
                    "description": (
                        "Directory to search in. Defaults to current working directory."
                    ),
                },
                "sort": {
                    "type": "string",
                    "enum": list(SORT_VALUES),
                    "description": (
                        "Result ordering. Default 'name' (alphabetical,"
                        " mirrors ``ls``). Use 'mtime_desc' for newest first."
                    ),
                },
                "long": {
                    "type": "boolean",
                    "description": (
                        "Prefix each result with size and mtime columns"
                        " (like ``ls -l``). Default false."
                    ),
                },
                "max_results": {
                    "type": "integer",
                    "minimum": 0,
                    "description": (
                        "Maximum number of matches to return. Omit for a"
                        " budget-derived default; ``0`` means unlimited."
                    ),
                },
                "offset": {
                    "type": "integer",
                    "minimum": 0,
                    "description": (
                        "Skip the first N matches. Pair with ``max_results``"
                        " to page through a match set too large for one"
                        " reply."
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
          args: Directive with ``pattern`` and optional ``path``.

        Returns:
          label: ``Glob <pattern>[ in <path>]`` line shown before invocation.

        """
        pattern = str(args.get("pattern", ""))
        path = str(args.get("path", "")) or "."
        suffix = f" in {path}" if path != "." else ""
        return f"Glob {pattern}{suffix}"

    def prompt(self) -> str:
        """Return supplemental prompt text for this tool.

        Returns:
          contribution: Empty string.

        """
        return ""

    def serialize_key(self, args: Mapping[str, object]) -> str | None:
        """Run in parallel: read-only listing needs no serialization."""
        del args
        return None

    async def run(self, args: Mapping[str, object]) -> ToolResult:
        """Match files against a glob pattern and return paths.

        Args:
          args: Directive with ``pattern`` and optional ``path`` / ``sort``
              / ``long`` / ``max_results`` / ``offset``.

        Returns:
          result: One match per line (resolved paths), or ``(no matches)``.

        """
        return await run_sync(
            self._run,
            pattern=str(args.get("pattern", "")),
            path=str(args.get("path", ".") or "."),
            sort=str(args.get("sort", _DEFAULT_SORT) or _DEFAULT_SORT),
            long=BoolCodec.coerce(args.get("long"), False),
            max_results=IntCodec.coerce(args.get("max_results"), 0),
            offset=IntCodec.coerce(args.get("offset"), 0),
        )

    def _run(
        self,
        *,
        pattern: str,
        path: str = ".",
        sort: str = _DEFAULT_SORT,
        long: bool = False,
        max_results: int = 0,
        offset: int = 0,
    ) -> str | ToolResult:
        """Run the glob synchronously and return formatted matches."""
        if sort not in SORT_VALUES:
            return ToolResult(
                call_id="",
                content=f"unknown sort: {sort!r} (expected one of {list(SORT_VALUES)})",
                is_error=True,
            )
        if max_results < 0:
            return ToolResult(
                call_id="",
                content=f"max_results must be >= 0; got {max_results}.",
                is_error=True,
            )
        if offset < 0:
            return ToolResult(
                call_id="",
                content=f"offset must be >= 0; got {offset}.",
                is_error=True,
            )
        # Python's Path.glob requires a relative pattern. If the
        # caller passes an absolute pattern (e.g. ``/abs/dir/*.py``),
        # split it at the first component containing a glob char.
        # Everything before becomes the root; everything after is
        # the relative pattern. Matches what shell globs expect.
        pat_path = Path(pattern)
        if pat_path.is_absolute():
            parts = pat_path.parts
            split_at = next(
                (i for i, part in enumerate(parts) if any(c in part for c in "*?[")),
                len(parts),
            )
            root = Path(*parts[:split_at]) if split_at > 0 else Path("/")
            rel = str(Path(*parts[split_at:])) if split_at < len(parts) else ""
            matches = list(root.glob(rel)) if rel else ([root] if root.exists() else [])
        else:
            if not Path(path).is_absolute():
                path = str(Path(get_tool_state().bash_cwd) / path)
            root = Path(path)
            matches = list(root.glob(pattern))

        matches = _honor_dotfile_rule(matches, pattern=pattern, root=root)
        sort_paths(matches, sort)
        if not matches:
            return "(no matches)"
        total = len(matches)
        window = matches[offset:]
        # ``max_results=0`` means "no caller-supplied window"; the token
        # bound below is what actually stops the reply.
        shown = window[:max_results] if max_results > 0 else window
        render = _long_line if long else _plain_line
        note = f"\n... ({total} more; pass offset={total} to continue)"
        result, kept = bound_by_tokens(
            (render(m) for m in shown),
            budget=max(1, result_token_budget() - approx_tokens(note)),
        )
        result = result.rstrip("\n") or "(no matches in this window)"
        remaining = total - offset - kept
        if remaining > 0:
            resume = offset + kept
            result += f"\n... ({remaining} more; pass offset={resume} to continue)"
        return result

    def bash_match(self, trees: Sequence[Node]) -> str | None:
        """Emit a tool-use nudge for ``find`` used to enumerate paths.

        Detection is :func:`replaceable`; this decides only that Glob
        claims ``find`` and that acting predicates make Bash necessary.
        ``find … | xargs grep`` is skipped because Grep claims it: the
        find half only enumerates what to search. Directory listing
        (``ls``) belongs to the List tool.

        Args:
          trees: Parsed bashlex command trees from the active Bash call.

        Returns:
          hint: Nudge string redirecting to the Glob tool, or ``None``.

        """
        for inv in walk_commands(trees):
            # The classifier's denylist verbatim: these predicates ACT on
            # what they match, so the command's product is the write or the
            # exec, not the path list. A second copy diverged on -fprint0.
            if not replaceable(inv, exes=_FIND_EXES, deny=FIND_DENY_FLAGS):
                continue
            if inv.piped_into is not None and inv.piped_into.exe == "xargs":
                continue
            sink = bounding_sink(inv)
            call = _glob_call(
                inv.args,
                cwd=inv.cwd,
                # ``| head -50`` bounds the listing, which Glob says
                # directly as ``max_results``.
                max_results=parse_line_count(sink.args) if sink else None,
            )
            return f"{_NUDGE} Replaces: `{render_command(inv)}`.{call}"
        return None


def _honor_dotfile_rule(matches: list[Path], *, pattern: str, root: Path) -> list[Path]:
    """Drop hidden matches unless the pattern's own segment asks for them.

    The shell rule this tool advertises: ``*`` does not match a leading
    dot, ``.*`` matches only those. ``Path.glob`` implements neither, so
    an unfiltered ``*`` handed back ``.env`` and every ``.git`` entry to a
    caller who asked for visible files -- and List's ``show_hidden``
    toggle exists precisely because Glob was supposed to answer this
    through the pattern instead.

    Matched PER SEGMENT against the pattern's corresponding segment, since
    ``.config/*.json`` names a hidden directory explicitly and its
    contents are then not hidden by the caller's reckoning.

    Args:
      matches: Paths ``Path.glob`` returned.
      pattern: The caller's glob pattern.
      root: Directory the pattern was resolved against.

    Returns:
      kept: Matches whose hidden segments were each asked for.

    """
    segments = Path(pattern).parts
    if not any(part.startswith(".") for part in segments):
        wants_hidden = ()
    else:
        wants_hidden = tuple(part.startswith(".") for part in segments)
    kept: list[Path] = []
    for match in matches:
        try:
            relative = match.relative_to(root).parts
        except ValueError:
            kept.append(match)
            continue
        hidden = [i for i, part in enumerate(relative) if part.startswith(".")]
        # ``**`` spans any depth, so a positional pattern segment does not
        # line up; require the pattern to name a dot somewhere instead.
        recursive = "**" in segments
        if all(
            (
                wants_hidden[i]
                if i < len(wants_hidden) and not recursive
                else bool(wants_hidden)
            )
            for i in hidden
        ):
            kept.append(match)
    return kept


def _plain_line(p: Path) -> str:
    """Render one resolved path, newline-terminated for the token bound."""
    return f"{p.resolve()}\n"


def _long_line(p: Path) -> str:
    """Render one ``ls -l``-style row: ``<size>  <mtime>  <path>``."""
    size = safe_size(p)
    mtime_raw = safe_mtime(p)
    mtime = (
        time.strftime("%Y-%m-%d %H:%M", time.localtime(mtime_raw)) if mtime_raw else "?"
    )
    return f"{size:>10}  {mtime}  {p.resolve()}\n"


def _glob_call(
    args: tuple[str, ...],
    *,
    cwd: str = "",
    max_results: int | None = None,
) -> str:
    """Render a concrete Glob call, or ``""`` when a predicate is untranslatable.

    Runs after detection, so an unsupported predicate (``-newer``,
    ``-maxdepth``) costs the caller a worked example rather than the
    nudge itself -- gating detection on this parse is what made most of
    ``find``'s ~80 predicates silent.

    ``cwd`` is the enclosing ``cd`` prefix. Glob resolves a relative
    ``path`` against the AGENT's cwd, not the shell's, so dropping it
    searches a different tree than the command being replaced.
    """
    path = ""
    pattern = ""
    i = 0
    while i < len(args):
        a = args[i]
        if a == "-name":
            if i + 1 >= len(args):
                return ""
            pattern = args[i + 1]
            i += 2
            continue
        if a == "-iname":
            # ``Path.glob`` is case-sensitive on Linux, so rendering the
            # same pattern under Glob asks a different question.
            return ""
        if a == "-type":
            # ``-type f`` is lossless: Glob's pattern already matches
            # files. ``-type d`` restricts to directories, which Glob
            # cannot express -- dropping it silently widens the result.
            if i + 1 >= len(args) or args[i + 1] != "f":
                return ""
            i += 2
            continue
        if a.startswith("-"):
            return ""
        if path:
            # Multiple bare paths -- Glob takes one root.
            return ""
        path = a
        i += 1
    if not pattern:
        return ""
    target = resolve_cwd_path(cwd, path)
    root = f" path={target!r}" if target else ""
    cap = f" max_results={max_results}" if max_results is not None else ""
    return f" Try: Glob pattern='**/{pattern}'{root}{cap}"
