"""WebSearch tool: query a pluggable search backend."""

from __future__ import annotations

from typing import cast, get_args

import asyncio

from sagent.custom_types import Message, TextMessage
from sagent.lib.json import JSON, JSONValue, json_freeze
from sagent.lib.message import get_directive
from sagent.lib.web import DEFAULT_SEARCH_BACKEND, SearchBackends, search
from sagent.tools.core import (
    TOOL_RESULT_MAX_CHARS,
    load_tool_description,
    truncate,
)


class WebSearch:
    """Search the web using a configurable backend."""

    name: str = "WebSearch"
    tool_id: str = "application/x-tool-websearch"
    supports_microcompaction: bool = True

    @property
    def description(self) -> str:
        """Return the tool description, re-evaluating ``{{NOW}}`` each access.

        A long-running process that spans a month boundary must not freeze
        the substitution at import time.

        Returns:
          description: Rendered tool description with current timestamp.

        """
        return load_tool_description("WebSearch")

    directive_schema: JSON = json_freeze(
        {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query string."},
                "allowed_domains": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Only include results from these domains.",
                },
                "blocked_domains": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Exclude results from these domains.",
                },
                "backend": {
                    "type": "string",
                    "enum": get_args(SearchBackends),
                    "description": f'Search backend (default: "{DEFAULT_SEARCH_BACKEND}").',
                },
            },
            "required": ["query"],
        }
    )

    def summary(self, msg: Message) -> str:
        """Return a short display label for this invocation.

        Args:
          msg: Directive message.

        Returns:
          label: Human-readable summary string.

        """
        directive = get_directive(msg)
        query = str(directive.get("query", ""))
        if len(query) > 50:
            query = query[:47] + "..."
        return f"WebSearch {query!r}"

    def summary_result(self, result: Message) -> str | None:
        del result
        return None

    def prompt(self) -> str:
        """Return supplemental system-prompt text.

        Returns:
          prompt: Empty string; this tool adds no prompt.

        """
        return ""

    async def run(self, msg: Message) -> Message:
        """Execute the web search and return formatted results.

        Args:
          msg: Directive message containing query and optional filters.

        Returns:
          result: Plain-text search results or an error message.

        """
        directive = get_directive(msg)
        query = str(directive.get("query", ""))
        backend: SearchBackends = DEFAULT_SEARCH_BACKEND
        backend_val = directive.get("backend")
        if isinstance(backend_val, str) and backend_val in get_args(SearchBackends):
            backend = cast(SearchBackends, backend_val)
        elif backend_val is not None:
            return TextMessage(
                f"Invalid backend {backend_val!r}. Valid: {', '.join(get_args(SearchBackends))}.",
                "text/x-error",
            )
        q = _build_query(
            query,
            directive.get("allowed_domains"),
            directive.get("blocked_domains"),
        )
        try:
            results = await asyncio.to_thread(search, q, backend=backend)
        except (RuntimeError, ValueError) as err:
            return TextMessage(str(err), "text/x-error")
        if not results:
            text = "(no results)"
        else:
            text = "\n\n".join(
                f"[{r.title}]({r.url})\n{r.snippet}" for r in results[:10]
            )
        return TextMessage(truncate(text, TOOL_RESULT_MAX_CHARS), "text/plain")


def _build_query(
    query: str,
    allowed_domains: JSONValue | None,
    blocked_domains: JSONValue | None,
) -> str:
    """Return *query* with site:/-site: filters appended."""
    allowed = (
        list(allowed_domains) if isinstance(allowed_domains, (list, tuple)) else []
    )
    blocked = (
        list(blocked_domains) if isinstance(blocked_domains, (list, tuple)) else []
    )
    for domain in allowed:
        if isinstance(domain, str) and domain:
            query += f" site:{domain.strip()}"
    for domain in blocked:
        if isinstance(domain, str) and domain:
            query += f" -site:{domain.strip()}"
    return query
