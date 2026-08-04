"""Wiki tool: read/list operations against an LLM wiki.

Backed by the ``llm-wiki`` spec (``docs/llm-wiki/SPEC.md``) - the
five orchestration skills (``wiki-init``, ``wiki-ingest``,
``wiki-query``, ``wiki-update``, ``wiki-lint``) drive the workflow
via Read/Write/Edit; this tool provides the structural primitives:

- locate the wiki root (by ``SCHEMA.md``, walking up from cwd)
- list pages / read index / read a page by slug
- a deterministic ``lint`` check: broken ``[[slug]]`` wikilinks,
  missing frontmatter (``title``, ``tags``, ``sources``, ``updated``)

Not included: ingest/update/query - those are LLM-driven via skills.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Final

import logging
import re

from sagent.agent.state import get_tool_state
from sagent.lib.custom_json import JSON, json_freeze
from sagent.tools.core import load_tool_description, run_sync
from sagent.tools.prompt_text import escape_prompt_text
from sagent.types.runtime import ToolResult


logger = logging.getLogger(__name__)

_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_WIKILINK_RE = re.compile(r"\[\[([a-z0-9][a-z0-9-]*)\]\]")
_FRONTMATTER_RE = re.compile(
    r"\A---\s*\n(?P<body>.*?)\n---\s*\n",
    re.DOTALL,
)
_REQUIRED_FRONTMATTER: Final = ("title", "tags", "sources", "updated")


def find_root(start: str | Path) -> Path | None:
    """Walk from ``start`` upward, looking for ``SCHEMA.md``.

    Args:
      start: Path (file or directory) at which to begin the upward search.

    Returns:
      root: Directory containing ``SCHEMA.md`` (or ``wiki/SCHEMA.md``),
        or ``None`` if no wiki is found before reaching the filesystem root.

    """
    p = Path(start).resolve()
    candidates = [p, *p.parents]
    for d in candidates:
        if (d / "SCHEMA.md").is_file():
            return d
        wiki_dir = d / "wiki"
        if (wiki_dir / "SCHEMA.md").is_file():
            return wiki_dir
    return None


def _pages_dir(root: Path) -> Path:
    """Return the ``pages/`` directory under a wiki root."""
    return root / "pages"


def list_pages(root: Path) -> list[str]:
    """Return all page slugs under ``<root>/pages/`` (flat).

    Args:
      root: Wiki root directory (as returned by ``find_root``).

    Returns:
      slugs: Sorted list of page slugs (filenames without the ``.md`` suffix).

    """
    pd = _pages_dir(root)
    if not pd.exists():
        return []
    return sorted(p.stem for p in pd.glob("*.md"))


def valid_slug(slug: str) -> bool:
    """Return whether ``slug`` matches the wiki's slug grammar.

    Args:
      slug: Candidate page slug.

    Returns:
      is_valid: True iff ``slug`` is lowercase alphanumeric with hyphens.

    """
    return _SLUG_RE.fullmatch(slug) is not None


def read_page(root: Path, slug: str) -> str | None:
    """Read a page's markdown body by slug.

    Args:
      root: Wiki root directory.
      slug: Page slug (validated against the wiki's slug grammar).

    Returns:
      text: Page markdown contents, or ``None`` if the slug is invalid,
        the file is missing, or it can't be decoded as UTF-8.

    """
    if not valid_slug(slug):
        return None
    fp = _pages_dir(root) / f"{slug}.md"
    if not fp.exists():
        return None
    try:
        return fp.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def _iter_page_files(root: Path) -> Iterable[Path]:
    """Yield page files under ``<root>/pages/`` in sorted order."""
    pd = _pages_dir(root)
    if pd.exists():
        yield from sorted(pd.glob("*.md"))


def _parse_frontmatter_keys(text: str) -> set[str]:
    """Extract the set of top-level keys from a page's YAML frontmatter."""
    m = _FRONTMATTER_RE.match(text)
    if m is None:
        return set()
    return {
        line.split(":", 1)[0].strip()
        for line in m.group("body").splitlines()
        if line.strip() and not line.lstrip().startswith("#") and ":" in line
    }


def lint(root: Path) -> dict[str, list[str]]:
    """Return deterministic lint errors for the wiki.

    Two categories:
    - ``broken_links``: wikilink targets with no corresponding page
    - ``missing_frontmatter``: pages missing required frontmatter keys

    Args:
      root: Wiki root directory.

    Returns:
      errors: Mapping with ``broken_links`` and ``missing_frontmatter``
        keys, each listing human-readable error lines.

    """
    errors: dict[str, list[str]] = {
        "broken_links": [],
        "missing_frontmatter": [],
    }
    known = set(list_pages(root))
    for fp in _iter_page_files(root):
        try:
            text = fp.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        keys = _parse_frontmatter_keys(text)
        missing = [k for k in _REQUIRED_FRONTMATTER if k not in keys]
        if missing:
            errors["missing_frontmatter"].append(
                f"{fp.name}: missing {', '.join(missing)}",
            )
        for m in _WIKILINK_RE.finditer(text):
            target = m.group(1)
            if target not in known:
                errors["broken_links"].append(
                    f"{fp.name}: [[{target}]] -> no such page",
                )
    return errors


_OPERATIONS: Final = ("locate", "list", "read_page", "read_index", "lint")


class Wiki:
    """Tool: structural primitives for an LLM wiki."""

    name: str = "Wiki"
    tool_id: str = "application/x-tool-wiki"
    clearable_results: bool = False
    description: str = load_tool_description("Wiki")
    directive_schema: JSON = json_freeze(
        {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": list(_OPERATIONS),
                    "description": "Which operation to run.",
                },
                "slug": {
                    "type": "string",
                    "description": "Page slug (required for 'read_page').",
                },
                "cwd": {
                    "type": "string",
                    "description": "Optional cwd override for locating the wiki.",
                },
            },
            "required": ["operation"],
        }
    )

    def summary(self, args: Mapping[str, object]) -> str:
        """Return a short label for this Wiki operation.

        Args:
          args: Directive with ``operation`` and optional ``slug``.

        Returns:
          label: ``Wiki <op>[:<slug>]`` line shown before invocation.

        """
        operation = str(args.get("operation", ""))
        slug = str(args.get("slug", ""))
        suffix = f":{slug}" if slug else ""
        return f"Wiki {operation}{suffix}"

    def summary_result(self, result: ToolResult) -> str | None:
        """Suppress the per-call receipt for Wiki.

        Args:
          result: Completed ``ToolResult`` (ignored).

        Returns:
          receipt: Always ``None`` (no receipt line).

        """
        del result
        return None

    def prompt(self) -> str:
        """Return no supplemental system-prompt text for Wiki.

        Returns:
          contribution: Empty string.

        """
        return ""

    def serialize_key(self, args: Mapping[str, object]) -> str | None:
        """Run in parallel: wiki access has no shared in-process resource."""
        del args
        return None

    async def run(self, args: Mapping[str, object]) -> ToolResult:
        """Dispatch a Wiki operation (locate, list, read_page, read_index, lint).

        Args:
          args: Directive with ``operation`` and optional ``slug`` / ``cwd``.

        Returns:
          result: Operation output, or an error when the operation is
              unknown or no wiki is found.

        """
        operation = str(args.get("operation", ""))
        slug = str(args.get("slug", ""))
        cwd = str(args.get("cwd", ""))
        return await run_sync(
            self._run,
            operation=operation,
            slug=slug,
            cwd=cwd,
        )

    def _run(
        self, *, operation: str, slug: str = "", cwd: str = ""
    ) -> str | ToolResult:
        """Locate the wiki root and route to the operation handler."""
        if operation not in _OPERATIONS:
            return ToolResult(
                call_id="",
                content=(
                    f"Unknown operation {operation!r}. Valid: {', '.join(_OPERATIONS)}"
                ),
                is_error=True,
            )
        start = cwd or get_tool_state().bash_cwd
        root = find_root(start)
        if root is None:
            return ToolResult(
                call_id="",
                content=(
                    "No wiki found (walked up from "
                    f"{start} looking for SCHEMA.md or wiki/SCHEMA.md)."
                ),
                is_error=True,
            )
        if operation == "locate":
            return str(root)
        if operation == "list":
            return "\n".join(list_pages(root)) or "(no pages)"
        if operation == "read_page":
            return _read_page_op(root, slug)
        if operation == "read_index":
            return _read_index_op(root)
        return _lint_op(root)


def _read_page_op(root: Path, slug: str) -> str | ToolResult:
    """Handle the ``read_page`` operation: read a page by slug."""
    if not slug:
        return ToolResult(
            call_id="",
            content=(
                "ToolInputError: Wiki failed: operation='read_page' requires"
                " `slug`.\n\nWiki requires: `slug`.\n\nThis tool call was not"
                " executed because its JSON directive was missing or"
                " misstated required fields. Do not repeat the same empty or"
                " incomplete call. Either retry this tool with the required"
                " fields, choose a different tool that fits the task, or"
                " explain why the required value is unavailable."
            ),
            is_error=True,
        )
    if not valid_slug(slug):
        return ToolResult(
            call_id="", content=f"Invalid page slug: {slug!r}", is_error=True
        )
    content = read_page(root, slug)
    if content is None:
        return ToolResult(call_id="", content=f"No such page: {slug}", is_error=True)
    return escape_prompt_text(content)


def _read_index_op(root: Path) -> str | ToolResult:
    """Handle the ``read_index`` operation: read ``index.md`` if present."""
    idx = root / "index.md"
    if not idx.exists():
        return "(no index.md)"
    try:
        return escape_prompt_text(idx.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError) as e:
        return ToolResult(call_id="", content=f"Read error: {e}", is_error=True)


def _lint_op(root: Path) -> str:
    """Handle the ``lint`` operation and render a human-readable report."""
    result = lint(root)
    broken = result["broken_links"]
    missing = result["missing_frontmatter"]
    if not broken and not missing:
        return "Lint clean."
    parts: list[str] = []
    if broken:
        parts.append("Broken wikilinks:")
        parts.extend(f"  {e}" for e in broken)
    if missing:
        parts.append("Missing frontmatter:")
        parts.extend(f"  {e}" for e in missing)
    return "\n".join(parts)
