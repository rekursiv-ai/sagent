"""WebSearch tool: query a pluggable search backend."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast, get_args

import asyncio

from sagent.lib.json import JSON, JSONValue, json_freeze
from sagent.lib.web import DEFAULT_SEARCH_BACKEND, SearchBackends, search
from sagent.lib.web.search import CaptchaError, SearchError
from sagent.tools.core import (
    TOOL_RESULT_MAX_CHARS,
    load_tool_description,
    truncate,
)
from sagent.types.runtime import ToolResult


class WebSearch:
    """Search the web using a configurable backend."""

    name: str = "WebSearch"
    tool_id: str = "application/x-tool-websearch"
    clearable_results: bool = True

    @property
    def description(self) -> str:
        """Return the tool description, re-evaluating ``{{NOW}}`` each access.

        A long-running process that spans a month boundary must not freeze
        the substitution at import time.
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
                    "description": (
                        f'Search backend (default: "{DEFAULT_SEARCH_BACKEND}").'
                    ),
                },
            },
            "required": ["query"],
        }
    )

    def summary(self, args: Mapping[str, object]) -> str:
        """Return a short display label for this invocation.

        Args:
          args: Directive carrying the ``query`` string.

        Returns:
          label: ``WebSearch '<query>'`` line shown before invocation.

        """
        query = str(args.get("query", ""))
        if len(query) > 50:
            query = query[:47] + "..."
        return f"WebSearch {query!r}"

    def summary_result(self, result: ToolResult) -> str | None:
        """Suppress the per-call receipt for WebSearch.

        Args:
          result: Completed ``ToolResult`` (ignored).

        Returns:
          receipt: Always ``None`` (no receipt line).

        """
        del result
        return None

    def prompt(self) -> str:
        """Return no supplemental system-prompt text for WebSearch.

        Returns:
          contribution: Empty string.

        """
        return ""

    def serialize_key(self, args: Mapping[str, object]) -> str | None:
        """Run in parallel: independent network fetch, no serialization."""
        del args
        return None

    async def run(self, args: Mapping[str, object]) -> ToolResult:
        """Execute the web search and return formatted results.

        Args:
          args: Directive with ``query``, optional ``allowed_domains`` /
              ``blocked_domains`` / ``backend``.

        Returns:
          result: Top results rendered as ``[title](url)`` over a
              snippet line, or an error when the backend rejects the query.

        """
        query = str(args.get("query", ""))
        backend: SearchBackends = DEFAULT_SEARCH_BACKEND
        backend_val = args.get("backend")
        if isinstance(backend_val, str) and backend_val in get_args(SearchBackends):
            backend = cast(SearchBackends, backend_val)
        elif backend_val is not None:
            return ToolResult(
                call_id="",
                content=(
                    f"Invalid backend {backend_val!r}."
                    f" Valid: {', '.join(get_args(SearchBackends))}."
                ),
                is_error=True,
            )
        q = _build_query(
            query,
            args.get("allowed_domains"),
            args.get("blocked_domains"),
        )
        try:
            results = await asyncio.to_thread(search, q, backend=backend)
        except (CaptchaError, RuntimeError, SearchError, ValueError) as err:
            return ToolResult(call_id="", content=str(err), is_error=True)
        if not results:
            text = "(no results)"
        else:
            text = "\n\n".join(
                f"[{r.title}]({r.url})\n{r.snippet}" for r in results[:10]
            )
        return ToolResult(call_id="", content=truncate(text, TOOL_RESULT_MAX_CHARS))


def _build_query(
    query: str,
    allowed_domains: object,
    blocked_domains: object,
) -> str:
    """Return *query* with site:/-site: filters appended."""
    allowed = (
        list(cast(list[JSONValue], allowed_domains))
        if isinstance(allowed_domains, (list, tuple))
        else []
    )
    blocked = (
        list(cast(list[JSONValue], blocked_domains))
        if isinstance(blocked_domains, (list, tuple))
        else []
    )
    for domain in allowed:
        if isinstance(domain, str) and domain:
            query += f" site:{domain.strip()}"
    for domain in blocked:
        if isinstance(domain, str) and domain:
            query += f" -site:{domain.strip()}"
    return query
