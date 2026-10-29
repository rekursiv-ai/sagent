"""WebSearch: search the web, and return each hit's title, URL, and snippet.

Typical usage example:

  result = await WebSearch(session=FetchSession()).call(context, {"query": "python 3.14 release notes"})
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.resources import files
from typing import TYPE_CHECKING, Annotated

import asyncio
import re
import time

from wesearch.fetch import FetchSession, PolicyParams, Transport
from wesearch.mcp.types import SearchParamsSchema
from wesearch.search.render import format_result
from wesearch.search.search import search
from wesearch.search.types import SearchBackends, SearxngCategory
from wesearch.types.errors import BotDetectionError

from sagent.session.types import (
    UncategorizedToolResult,
    WebSearchResult,
    WebSearchResults,
)
from sagent.tools.display import ToolDisplay
from sagent.tools.types import parse, schema_of


if TYPE_CHECKING:
    from collections.abc import Mapping

    from treekle import PlainTree

    from sagent.tools.types import ToolContext


__all__ = [
    "WebSearch",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class WebSearch:
    """Search the web."""

    @dataclass(frozen=True, slots=True, kw_only=True)
    class Arguments:
        """What the model passes."""

        query: Annotated[str, SearchParamsSchema.query.description]
        backend: Annotated[
            SearchBackends | None,
            SearchParamsSchema.backend.description,
        ] = None
        categories: Annotated[
            SearxngCategory,
            SearchParamsSchema.categories.description,
        ] = "general"
        transport: Annotated[
            Transport | None,
            SearchParamsSchema.transport.description,
        ] = None
        allowed_domains: Annotated[
            tuple[str, ...],
            "Only include results from these domains.",
        ] = ()
        blocked_domains: Annotated[
            tuple[str, ...],
            "Exclude results from these domains.",
        ] = ()

    name = "WebSearch"
    description = (files(__package__) / "assets" / "websearch.md").read_text(
        encoding="utf-8",
    )
    schema = schema_of(Arguments)

    display: ToolDisplay = field(default=ToolDisplay())

    session: FetchSession = field(repr=False)
    """Connections, browser, and egress every search reuses; whoever made it closes it."""

    policy: PolicyParams = field(default_factory=PolicyParams)
    """How to fetch result pages when a call does not say."""

    def summary(self, arguments: Mapping[str, PlainTree]) -> str:
        """Return ``WebSearch '<query>'``."""
        return f"WebSearch {str(arguments.get('query', ''))!r}"

    def serialize_key(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> str | None:
        """Return ``None``: searches are independent."""
        del context, arguments
        return None

    async def call(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> WebSearchResults | UncategorizedToolResult:
        """Search for ``query``, scoped to the domains given.

        Args:
          context: The calling agent's state.
          arguments: What the model passed, as ``WebSearch.Arguments``.

        Returns:
          result: The hits that fit in one result, or why there are none.

        """
        a = parse(arguments, WebSearch.Arguments)
        # Only a bare hostname is spliced in, so a filter cannot carry query
        # syntax that widens the scope (``x.com -site:trusted.com``).
        hostname = re.compile(r"(?:\*\.)?[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+(?::\d+)?")
        for domain in (*a.allowed_domains, *a.blocked_domains):
            if not hostname.fullmatch(domain.strip()):
                return _error(f"Not a hostname: {domain!r}.")
        scoped = " ".join(
            (
                a.query,
                *(f"site:{d.strip()}" for d in a.allowed_domains),
                *(f"-site:{d.strip()}" for d in a.blocked_domains),
            ),
        )
        started = time.monotonic()
        try:
            hits = await asyncio.to_thread(
                search,
                scoped,
                backend=a.backend,
                categories=a.categories,
                transport=a.transport or self.policy.transport,
                session=self.session,
            )
        except BotDetectionError as e:
            # The instance may say more than its class does, such as how long
            # a cooldown lasts on this IP.
            reason = str(e)
            return _error(f"{reason} {e.guidance}" if reason != e.guidance else reason)
        except (RuntimeError, ValueError) as e:
            return _error(str(e))
        if not hits:
            return UncategorizedToolResult(call_id="", content="(no results)")
        results = tuple(
            WebSearchResult(
                url=hit.url,
                title=hit.title,
                # The snippet plus the fields its category adds: a paper's
                # DOI, a video's duration, a package's version.
                snippet=format_result(hit).partition("\n")[2],
            )
            for hit in hits
        )
        _, kept = context.fit(f"{r.title}\n{r.url}\n{r.snippet}\n\n" for r in results)
        return WebSearchResults(
            call_id="",
            query=a.query,
            duration_sec=time.monotonic() - started,
            content=results[:kept],
        )


def _error(message: str) -> UncategorizedToolResult:
    return UncategorizedToolResult(call_id="", content=message, is_error=True)
