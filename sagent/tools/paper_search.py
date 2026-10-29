"""PaperSearch: text search across scholarly backends.

The tool owns its arguments, a result cache, and rendering; ``wesearch``
owns every backend, rank fusion, and per-source rate limits.

Typical usage example:

  result = await PaperSearch(session=session).call(context, {"query": "diffusion models"})
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.resources import files
from typing import TYPE_CHECKING, Annotated

import asyncio

from wesearch.fetch import FetchSession
from wesearch.paper.errors import PaperError
from wesearch.paper.render import format_record, truncation_notice
from wesearch.paper.search import Source, search
from wesearch.types.errors import BotDetectionError

import cachetools

from sagent.session.types import UncategorizedToolResult
from sagent.tools.display import ToolDisplay
from sagent.tools.paper_common import error
from sagent.tools.types import Arg, parse, schema_of


if TYPE_CHECKING:
    from collections.abc import Mapping

    from treekle import PlainTree

    from sagent.tools.types import ToolContext


__all__ = [
    "PaperSearch",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class PaperSearch:
    """Search the scholarly literature by text."""

    @dataclass(frozen=True, slots=True, kw_only=True)
    class Arguments:
        """What the model passes."""

        query: Annotated[
            str,
            "Free-form text. Every backend matches title/abstract text, NOT author"
            " names -- an author surname in the query can yield zero hits. To"
            " search by author, use the PaperAuthor tool.",
        ]
        source: Annotated[
            Source,
            "'fused' (default): reciprocal-rank-fuse Semantic Scholar + OpenAlex,"
            " resilient to either being down. 's2': Semantic Scholar only (precise"
            " ranking, but rate-limited and no author-name matching). 'openalex':"
            " OpenAlex only (no key, broad coverage, author search). 'searxng':"
            " self-hosted SearXNG science metasearch (adds PubMed, Crossref, arXiv"
            " breadth; no citation graph, best-effort year/OA filtering).",
        ] = "fused"
        limit: Annotated[
            int | None,
            "Max hits. Omit to let the backend decide its default page. In fused"
            " mode, applies to the merged set.",
            Arg(minimum=1),
        ] = None
        year_from: Annotated[
            int | None,
            "Inclusive lower bound on publication year.",
            Arg(minimum=0),
        ] = None
        year_to: Annotated[
            int | None,
            "Inclusive upper bound on publication year.",
            Arg(minimum=0),
        ] = None
        open_access_only: Annotated[
            bool,
            "Restrict to papers with a known open-access PDF.",
        ] = False
        abstract_chars: Annotated[
            int | None,
            "Truncate every abstract to this many characters. Omit for full abstracts.",
            Arg(minimum=1),
        ] = None

    name = "PaperSearch"
    description = (files(__package__) / "assets" / "papersearch.md").read_text(
        encoding="utf-8",
    )
    schema = schema_of(Arguments)

    display: ToolDisplay = field(default=ToolDisplay())

    session: FetchSession = field(repr=False)
    """Connections, browser, and egress every search reuses; whoever made it closes it."""

    # Results are stable on an agent's timescale; the cache spares the shared
    # rate gates a repeated query.
    cache: cachetools.LRUCache[tuple[object, ...], str] = field(
        default_factory=lambda: cachetools.LRUCache[tuple[object, ...], str](
            maxsize=1024,
        ),
        repr=False,
    )
    """Rendered results, by query and filters."""

    def summary(self, arguments: Mapping[str, PlainTree]) -> str:
        """Return ``PaperSearch '<query>'``, and the source when not fused."""
        query = str(arguments.get("query", "")).strip()
        source = str(arguments.get("source") or "fused")
        label = f"PaperSearch {query!r}" if query else "PaperSearch"
        return label if source == "fused" else f"{label} ({source})"

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
    ) -> UncategorizedToolResult:
        """Search for ``query`` with the filters given.

        Args:
          context: The calling agent's state.
          arguments: What the model passed, as ``PaperSearch.Arguments``.

        Returns:
          result: One paper per line, or why there are none.

        Raises:
          ValueError: The query is empty, or the years are inverted.

        """
        del context
        a = parse(arguments, PaperSearch.Arguments)
        query = a.query.strip()
        if not query:
            raise ValueError("'query' is required.")
        if (
            a.year_from is not None
            and a.year_to is not None
            and a.year_from > a.year_to
        ):
            raise ValueError(
                f"'year_from' ({a.year_from}) must not exceed 'year_to' ({a.year_to}).",
            )
        key = (
            a.source,
            query,
            a.limit,
            a.year_from,
            a.year_to,
            a.open_access_only,
            a.abstract_chars,
        )
        cached = self.cache.get(key)
        if cached is not None:
            return UncategorizedToolResult(call_id="", content=cached)
        try:
            found = await asyncio.to_thread(
                search,
                query,
                source=a.source,
                limit=a.limit,
                year_from=a.year_from,
                year_to=a.year_to,
                open_access_only=a.open_access_only,
                session=self.session,
            )
        except BotDetectionError as e:
            return error(
                f"{_named(a.source)} returned a CAPTCHA: {e}. The scraping IP is"
                " rate-limited; retry later, change egress IP, or use another source.",
            )
        except PaperError as e:
            return error(str(e))
        shown = found.records if a.limit is None else found.records[: a.limit]
        text = (
            "\n".join(format_record(r, abstract_chars=a.abstract_chars) for r in shown)
            + truncation_notice(len(shown), found.total)
            if shown
            else "(no results)" + _empty_hint(query)
        )
        if found.complete:
            self.cache[key] = text
        return UncategorizedToolResult(call_id="", content=text)


# A source with no name here is named by its words: ``some_source`` is ``Some Source``.
def _named(source: str) -> str:
    """Return the name a user knows ``source`` by."""
    match source:
        case "s2":
            return "Semantic Scholar"
        case "openalex":
            return "OpenAlex"
        case "searxng":
            return "SearXNG"
        case "fused":
            return "Semantic Scholar or OpenAlex"
        case _:
            return source.replace("_", " ").title()


# Every backend ANDs the terms against title and abstract, so one rare term
# zeroes a query; and none matches author names.
def _empty_hint(query: str) -> str:
    """Return the advice given with an empty result."""
    hint = (
        "\nNote: every query term must appear in a paper's title/abstract (terms are"
        " AND-ed). A zero result usually means one specific term has no match --"
        " drop the most specific term and retry to broaden."
        if len(query.split()) > 1
        else ""
    )
    return hint + (
        "\nNote: paper search matches title/abstract text, not author names (true"
        " for every source). If your query was an author name, use the PaperAuthor"
        " tool instead."
    )
