"""Glob tool: fast path-pattern matching."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Annotated, Final

import time

from sagent.agent.state import current_agent_var, get_tool_state
from sagent.lib.custom_json import JSON, bool_val, int_val, json_freeze
from sagent.tools.core import load_tool_description, run_sync
from sagent.tools.display import Toggle, Wrap
from sagent.tools.lib.bash import Node, walk_commands
from sagent.tools.lib.path_sort import (
    SORT_VALUES,
    safe_mtime,
    safe_size,
    sort_paths,
)
from sagent.tools.tool_spec import CLI_SETTABLE
from sagent.types.runtime import ToolResult


_NUDGE: Final = "find via Bash is a bad UX. Use the Glob tool."

# Match cap when no agent is in context (standalone use, tests).
_FALLBACK_MAX_RESULTS: Final = 1_000

# Characters a rendered match line costs, used to turn the agent's
# character budget into a match count. Set above a typical absolute path
# so the derived cap errs small.
_ASSUMED_CHARS_PER_MATCH: Final = 120


def _default_max_results() -> int:
    """Match cap for a windowless Glob, derived from the active budget.

    A result over ``max_result_chars`` is off-loaded or elided, so an
    "unlimited" default silently returned less than a bounded one on a
    wide pattern. Deriving the bound keeps one reply whole and pairs it
    with ``offset`` so the remainder stays reachable.

    Returns:
      limit: Maximum matches returned by default; the fallback constant
          when no agent is in context.

    """
    agent = current_agent_var.get(None)
    ceiling = agent.max_result_chars if agent is not None else 0
    if ceiling <= 0:
        return _FALLBACK_MAX_RESULTS
    return max(_FALLBACK_MAX_RESULTS, ceiling // _ASSUMED_CHARS_PER_MATCH)


_DEFAULT_SORT: Final = "name"


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

    name: str = "Glob"
    tool_id: str = "application/x-tool-glob"
    clearable_results: bool = True
    description: str = load_tool_description("Glob")
    directive_schema: JSON = json_freeze(
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
            long=bool_val(args.get("long"), False),
            max_results=int_val(args.get("max_results"), _default_max_results()),
            offset=int_val(args.get("offset"), 0),
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

        sort_paths(matches, sort)
        if not matches:
            return "(no matches)"
        total = len(matches)
        window = matches[offset:]
        # ``max_results=0`` means unlimited; the default comes from the
        # active budget so one reply stays under the size at which the
        # result would be off-loaded or elided.
        shown = window[:max_results] if max_results > 0 else window
        if long:
            lines = [_long_line(m) for m in shown]
        else:
            lines = [str(m.resolve()) for m in shown]
        result = "\n".join(lines) or "(no matches in this window)"
        remaining = total - offset - len(shown)
        if remaining > 0:
            resume = offset + len(shown)
            result += f"\n... ({remaining} more; pass offset={resume} to continue)"
        return result

    def bash_match(self, trees: Sequence[Node]) -> str | None:
        """Emit a tool-use nudge for ``find … -name GLOB``.

        Policy only: :func:`walk_commands` supplies every simple command
        with its context, so a leading ``cd``, an enclosing loop, and a
        sequence all reach this matcher without it re-deriving AST shape.
        Bails on ``find`` predicates Glob can't express
        (time/size/perm/exec/depth). Directory listing (``ls``) is
        handled by the List tool, not Glob.

        Args:
          trees: Parsed bashlex command trees from the active Bash call.

        Returns:
          hint: Nudge string redirecting to the Glob tool, or ``None``.

        """
        for inv in walk_commands(trees):
            if inv.exe != "find" or inv.env_prefix or inv.captures_stdout:
                continue
            # ``find … | xargs grep`` is a SEARCH, which Grep owns; the
            # find half only enumerates what to search.
            if inv.piped_into:
                continue
            hint = _match_find(inv.cwd or None, inv.args)
            if hint is not None:
                return hint
        return None


def _long_line(p: Path) -> str:
    """Render one ``ls -l``-style row: ``<size>  <mtime>  <path>``."""
    size = safe_size(p)
    mtime_raw = safe_mtime(p)
    mtime = (
        time.strftime("%Y-%m-%d %H:%M", time.localtime(mtime_raw)) if mtime_raw else "?"
    )
    return f"{size:>10}  {mtime}  {p.resolve()}"


def _match_find(cwd: str | None, args: tuple[str, ...]) -> str | None:
    """Match a ``find [PATH] [-type f|d] -name GLOB`` for a Glob nudge."""
    # Shape: ``find [PATH] [-type f|d] -name GLOB``. PATH is the first
    # non-flag arg (or "." if omitted). ``-type`` is accepted but not
    # translated. Whitelist-only parsing: any predicate outside the
    # branches below bails.
    del cwd  # Hint is a fixed string; path resolution is the LLM's job.
    seen_path = False
    seen_name = False
    i = 0
    while i < len(args):
        a = args[i]
        if a in {"-name", "-iname"}:
            if i + 1 >= len(args):
                return None
            seen_name = True
            i += 2
            continue
        if a == "-type":
            if i + 1 >= len(args) or args[i + 1] not in {"f", "d"}:
                return None
            i += 2
            continue
        if a.startswith("-"):
            return None
        if seen_path:
            # Multiple bare paths - ambiguous for Glob.
            return None
        seen_path = True
        i += 1
    if not seen_name:
        return None
    return _NUDGE
