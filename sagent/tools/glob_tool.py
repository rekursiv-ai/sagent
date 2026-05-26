"""Glob tool: fast path-pattern matching."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import time

from sagent.lib.json import JSON, bool_val, int_val, json_freeze
from sagent.tools.core import (
    get_tool_state,
    load_tool_description,
    run_sync,
)
from sagent.tools.lib.bash import Node, unwrap_cd_prefix
from sagent.tools.lib.path_sort import (
    SORT_VALUES,
    safe_mtime,
    safe_size,
    sort_paths,
)
from sagent.types.history import ToolResult


_NUDGE = "find via Bash is a bad UX. Use the Glob tool."

_DEFAULT_SORT = "name"


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
    emit_tool_summary: bool = False
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
                    "minimum": 1,
                    "description": (
                        "Maximum number of results to return (default 200)."
                        " Must be ≥ 1."
                    ),
                },
            },
            "required": ["pattern"],
        }
    )

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

    def summary_result(self, result: ToolResult) -> str | None:
        """One-line receipt: number of matches.

        Args:
          result: Completed ``ToolResult`` from ``run``.

        Returns:
          receipt: ``N matches`` / ``no matches``, or ``None`` when suppressed.

        """
        if not self.emit_tool_summary or result.is_error:
            return None
        text = result.content.strip()
        if not text or text.startswith("(no matches"):
            return "no matches"
        return f"{text.count(chr(10)) + 1} matches"

    def prompt(self) -> str:
        """Return supplemental prompt text for this tool.

        Returns:
          contribution: Empty string.

        """
        return ""

    async def run(self, args: Mapping[str, object]) -> ToolResult:
        """Match files against a glob pattern and return paths.

        Args:
          args: Directive with ``pattern`` and optional ``path`` / ``sort``
              / ``long`` / ``max_results``.

        Returns:
          result: One match per line (resolved paths), or ``(no matches)``.

        """
        return await run_sync(
            self._run,
            pattern=str(args.get("pattern", "")),
            path=str(args.get("path", ".") or "."),
            sort=str(args.get("sort", _DEFAULT_SORT) or _DEFAULT_SORT),
            long=bool_val(args.get("long"), False),
            max_results=int_val(args.get("max_results"), 200),
        )

    def _run(
        self,
        *,
        pattern: str,
        path: str = ".",
        sort: str = _DEFAULT_SORT,
        long: bool = False,
        max_results: int = 200,
    ) -> str | ToolResult:
        """Run the glob synchronously and return formatted matches."""
        if sort not in SORT_VALUES:
            return ToolResult(
                call_id="",
                content=f"unknown sort: {sort!r} (expected one of {list(SORT_VALUES)})",
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
        truncated = matches[:max_results]
        if long:
            lines = [_long_line(m) for m in truncated]
        else:
            lines = [str(m.resolve()) for m in truncated]
        result = "\n".join(lines)
        if len(matches) > max_results:
            result += f"\n... ({len(matches) - max_results} more)"
        return result

    def bash_match(self, trees: Sequence[Node]) -> str | None:
        """Emit a tool-use nudge for ``find … -name GLOB``.

        Accepts ``cd PATH && CMD`` compounds via ``unwrap_cd_prefix``.
        Bails on ``find`` predicates Glob can't express
        (time/size/perm/exec/depth). Directory listing (``ls``) is
        handled by the List tool, not Glob.

        Args:
          trees: Parsed bashlex command trees from the active Bash call.

        Returns:
          hint: Nudge string redirecting to the Glob tool, or ``None``.

        """
        unwrapped = unwrap_cd_prefix(trees)
        if unwrapped is None:
            return None
        cwd, cmd = unwrapped
        if cmd.exe != "find" or cmd.env_prefix:
            return None
        return _match_find(cwd, cmd.args)


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
