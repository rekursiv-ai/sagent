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

from collections.abc import Iterable
from pathlib import Path

import logging
import re

from sagent.custom_types import Message, TextMessage
from sagent.lib.json import JSON, json_freeze
from sagent.lib.message import get_directive
from sagent.tools.core import (
    get_tool_state,
    load_tool_description,
    run_sync,
)
from sagent.tools.input_errors import tool_input_error_text


logger = logging.getLogger(__name__)


_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_WIKILINK_RE = re.compile(r"\[\[([a-z0-9][a-z0-9-]*)\]\]")
_FRONTMATTER_RE = re.compile(
    r"\A---\s*\n(?P<body>.*?)\n---\s*\n",
    re.DOTALL,
)
_REQUIRED_FRONTMATTER = ("title", "tags", "sources", "updated")


def find_root(start: str | Path) -> Path | None:
    """Walk from ``start`` upward, looking for ``SCHEMA.md``.

    Args:
      start: Directory to begin the upward search from.

    Returns:
      root: Directory containing ``SCHEMA.md``, or None.

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
    return root / "pages"


def list_pages(root: Path) -> list[str]:
    """Return all page slugs under ``<root>/pages/`` (flat).

    Args:
      root: Wiki root directory.

    Returns:
      slugs: Sorted list of page slugs.

    """
    pd = _pages_dir(root)
    if not pd.exists():
        return []
    return sorted(p.stem for p in pd.glob("*.md"))


def valid_slug(slug: str) -> bool:
    return _SLUG_RE.fullmatch(slug) is not None


def read_page(root: Path, slug: str) -> str | None:
    """Read a page's markdown body by slug.

    Args:
      root: Wiki root directory.
      slug: Page slug (filename stem).

    Returns:
      content: Page markdown text, or None if not found.

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
    pd = _pages_dir(root)
    if pd.exists():
        yield from sorted(pd.glob("*.md"))


def _parse_frontmatter_keys(text: str) -> set[str]:
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
      errors: Dict mapping category name to list of error strings.

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


_OPERATIONS = ("locate", "list", "read_page", "read_index", "lint")


class Wiki:
    """Tool: structural primitives for an LLM wiki."""

    name: str = "Wiki"
    tool_id: str = "application/x-tool-wiki"
    description: str = load_tool_description("Wiki")
    supports_microcompaction: bool = False
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

    def summary(self, msg: Message) -> str:
        """Return a short label for this wiki operation.

        Args:
          msg: Tool call message.

        Returns:
          label: "Wiki <operation>:<slug>".

        """
        directive = get_directive(msg)
        operation = str(directive.get("operation", ""))
        slug = str(directive.get("slug", ""))
        suffix = f":{slug}" if slug else ""
        return f"Wiki {operation}{suffix}"

    def summary_result(self, result: Message) -> str | None:
        del result
        return None

    def prompt(self) -> str:
        """Return per-request system prompt text.

        Returns:
          prompt: Always empty for this tool.

        """
        return ""

    async def run(self, msg: Message) -> Message:
        """Dispatch the requested wiki operation.

        Args:
          msg: Tool call message with ``operation`` and optional ``slug``/``cwd``.

        Returns:
          result: Operation result or error message.

        """
        directive = get_directive(msg)
        operation = str(directive.get("operation", ""))
        slug = str(directive.get("slug", ""))
        cwd = str(directive.get("cwd", ""))
        return await run_sync(
            self._run,
            parent_id=msg.id,
            operation=operation,
            slug=slug,
            cwd=cwd,
        )

    def _run(self, *, operation: str, slug: str = "", cwd: str = "") -> str | Message:
        if operation not in _OPERATIONS:
            return TextMessage(
                f"Unknown operation {operation!r}. Valid: {', '.join(_OPERATIONS)}",
                "text/x-error",
            )
        start = cwd or get_tool_state().bash_cwd
        root = find_root(start)
        if root is None:
            return TextMessage(
                (
                    "No wiki found (walked up from "
                    f"{start} looking for SCHEMA.md or wiki/SCHEMA.md)."
                ),
                "text/x-error",
            )
        dispatch = {
            "locate": lambda: str(root),
            "list": lambda: "\n".join(list_pages(root)) or "(no pages)",
            "read_page": lambda: _read_page_op(root, slug),
            "read_index": lambda: _read_index_op(root),
            "lint": lambda: _lint_op(root),
        }
        return dispatch[operation]()


def _read_page_op(root: Path, slug: str) -> str | Message:
    if not slug:
        return TextMessage(
            tool_input_error_text(
                "Wiki",
                "operation='read_page' requires `slug`.",
                required=("slug",),
            ),
            "text/x-error",
        )
    if not valid_slug(slug):
        return TextMessage(f"Invalid page slug: {slug!r}", "text/x-error")
    content = read_page(root, slug)
    if content is None:
        return TextMessage(f"No such page: {slug}", "text/x-error")
    return content


def _read_index_op(root: Path) -> str | Message:
    idx = root / "index.md"
    if not idx.exists():
        return "(no index.md)"
    try:
        return idx.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        return TextMessage(f"Read error: {e}", "text/x-error")


def _lint_op(root: Path) -> str:
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
