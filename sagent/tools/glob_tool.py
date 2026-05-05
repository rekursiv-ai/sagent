"""Glob tool: fast path-pattern matching."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from sagent.custom_types import Message
from sagent.lib.json import JSON, int_val, json_freeze
from sagent.lib.message import get_directive
from sagent.tools.core import (
    get_tool_state,
    load_tool_description,
    run_sync,
)
from sagent.tools.lib.bash import Node, unwrap_cd_prefix


_NUDGE = "find via Bash is a bad UX. Use the Glob tool."


class Glob:
    """Match file paths against glob patterns."""

    name: str = "Glob"
    tool_id: str = "application/x-tool-glob"
    description: str = load_tool_description("Glob")
    supports_microcompaction: bool = True
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
                    "description": "Directory to search in. Defaults to current working directory.",
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
        suffix = f" in {path}" if path != "." else ""
        return f"Glob {pattern}{suffix}"

    def prompt(self) -> str:
        """Return supplemental prompt text for this tool.

        Returns:
          prompt: Always empty.

        """
        return ""

    async def run(self, msg: Message) -> Message:
        """Match files against a glob pattern and return paths.

        Args:
          msg: Incoming tool-use message containing the directive.

        Returns:
          result: Tool result Message with matching file paths.

        """
        directive = get_directive(msg)
        pattern = str(directive.get("pattern", ""))
        path = str(directive.get("path", ".") or ".")
        max_results = int_val(directive.get("max_results"), 200)
        return await run_sync(
            self._run,
            parent_id=msg.id,
            pattern=pattern,
            path=path,
            max_results=max_results,
        )

    def _run(self, *, pattern: str, path: str = ".", max_results: int = 200) -> str:
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

        matches.sort(key=_safe_mtime, reverse=True)
        if not matches:
            return "(no matches)"
        result = "\n".join(str(m.resolve()) for m in matches[:max_results])
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
          trees: Parsed shell AST nodes.

        Returns:
          nudge: Suggested Glob invocation, or ``None`` if no match.

        """
        unwrapped = unwrap_cd_prefix(trees)
        if unwrapped is None:
            return None
        cwd, cmd = unwrapped
        if cmd.exe != "find" or cmd.env_prefix:
            return None
        return _match_find(cwd, cmd.args)


def _safe_mtime(p: Path) -> float:
    try:
        return p.stat().st_mtime
    except OSError:
        return 0.0


def _match_find(cwd: str | None, args: tuple[str, ...]) -> str | None:
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
