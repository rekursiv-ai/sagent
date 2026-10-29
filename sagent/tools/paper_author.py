"""PaperAuthor: find authors, and list an author's papers.

Typical usage example:

  result = await PaperAuthor(session=session).call(context, {"query": "Yoshua Bengio"})
  result = await PaperAuthor(session=session).call(
      context, {"ids": "1741101", "operation": "papers", "year_from": 2022},
  )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.resources import files
from typing import TYPE_CHECKING, Annotated, Literal

import asyncio

from wesearch.fetch import FetchSession
from wesearch.paper.authors import author_metadata, author_papers, search_authors
from wesearch.paper.errors import PaperError
from wesearch.paper.render import (
    format_author_block,
    format_author_line,
    format_record,
    truncation_notice,
)

import cachetools

from sagent.session.types import UncategorizedToolResult
from sagent.tools.display import ToolDisplay
from sagent.tools.paper_common import error, paper_ids, summary_ids
from sagent.tools.types import Arg, parse, schema_of


if TYPE_CHECKING:
    from collections.abc import Mapping

    from treekle import PlainTree

    from sagent.tools.types import ToolContext


__all__ = [
    "PaperAuthor",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class PaperAuthor:
    """Search authors by name, look them up, or list one's papers."""

    @dataclass(frozen=True, slots=True, kw_only=True)
    class Arguments:
        """What the model passes."""

        query: Annotated[
            str,
            "Author name to search for (free-form). Mutually exclusive with 'ids'.",
        ] = ""
        ids: Annotated[
            tuple[str, ...] | str,
            "Semantic Scholar author id(s) (opaque integer strings, e.g."
            " '1741101'): a single id as a bare string, or several as an array."
            " Mutually exclusive with 'query'. For author metadata, pass every id"
            " at once: they are resolved in ONE batched request (up to 500), far"
            " more efficient against the 1 request/second rate limit than one call"
            " per id; results come back in input order. 'operation=papers' lists a"
            " single author's works, so pass exactly one id with it.",
        ] = ()
        operation: Annotated[
            Literal["papers"] | None,
            "Optional. With exactly one id: 'papers' lists that author's"
            " publications. Omit for author metadata.",
        ] = None
        limit: Annotated[
            int | None,
            "Max results. Omit to let Semantic Scholar decide (one default page).",
            Arg(minimum=1),
        ] = None
        year_from: Annotated[
            int | None,
            "Papers only: inclusive lower bound on publication year.",
            Arg(minimum=0),
        ] = None
        year_to: Annotated[
            int | None,
            "Papers only: inclusive upper bound on publication year.",
            Arg(minimum=0),
        ] = None
        abstract_chars: Annotated[
            int | None,
            "Papers only: truncate every abstract to this many characters. Omit for"
            " full abstracts.",
            Arg(minimum=1),
        ] = None

    name = "PaperAuthor"
    description = (files(__package__) / "assets" / "paperauthor.md").read_text(
        encoding="utf-8",
    )
    schema = schema_of(Arguments)

    display: ToolDisplay = field(default=ToolDisplay())

    session: FetchSession = field(repr=False)
    """Connections, browser, and egress every lookup reuses; whoever made it closes it."""

    cache: cachetools.LRUCache[tuple[object, ...], str] = field(
        default_factory=lambda: cachetools.LRUCache[tuple[object, ...], str](
            maxsize=1024,
        ),
        repr=False,
    )
    """Rendered results, by lookup."""

    def summary(self, arguments: Mapping[str, PlainTree]) -> str:
        """Return ``PaperAuthor search '<query>'``, or the ids and operation.

        Args:
          arguments: What the model passed.

        Returns:
          label: Such as ``PaperAuthor papers 1741101``.

        """
        query = str(arguments.get("query", "")).strip()
        if query:
            return f"PaperAuthor search {query!r}"
        ids = summary_ids(arguments)
        if ids == "?":
            return "PaperAuthor"
        if arguments.get("operation") == "papers":
            return f"PaperAuthor papers {ids}"
        return f"PaperAuthor {ids}"

    def serialize_key(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> str | None:
        """Return ``None``: lookups are independent."""
        del context, arguments
        return None

    async def call(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> UncategorizedToolResult:
        """Search authors, look up ``ids``, or list one author's papers.

        Args:
          context: The calling agent's state.
          arguments: What the model passed, as ``PaperAuthor.Arguments``.

        Returns:
          result: One line per candidate, a block per author, or one line per
            paper.

        Raises:
          ValueError: The arguments contradict each other.

        """
        del context
        a = parse(arguments, PaperAuthor.Arguments)
        query = a.query.strip()
        # Author ids are integer strings, so a bundle splits on digits.
        ids = paper_ids(a.ids, looks_like_id=str.isdigit)
        if bool(query) == bool(ids):
            raise ValueError("Set exactly one of 'query' or 'ids'.")
        if a.operation is not None and len(ids) != 1:
            raise ValueError("'operation=papers' needs exactly one id in 'ids'.")
        years = a.year_from is not None or a.year_to is not None
        if a.operation is None and years:
            raise ValueError(
                "'year_from' / 'year_to' only apply to operation='papers'.",
            )
        if (
            a.year_from is not None
            and a.year_to is not None
            and a.year_from > a.year_to
        ):
            raise ValueError(
                f"'year_from' ({a.year_from}) must not exceed 'year_to' ({a.year_to}).",
            )
        if query:
            key: tuple[object, ...] = ("search", query, a.limit)
        elif a.operation == "papers":
            key = ("papers", ids[0], a.limit, a.year_from, a.year_to, a.abstract_chars)
        else:
            key = ("authors", tuple(ids))
        cached = self.cache.get(key)
        if cached is not None:
            return UncategorizedToolResult(call_id="", content=cached)
        try:
            if query:
                text = await self._search(query, limit=a.limit)
            elif a.operation == "papers":
                text = await self._papers(ids[0], a)
            else:
                records = await asyncio.to_thread(
                    author_metadata,
                    ids,
                    session=self.session,
                )
                text = "\n\n".join(
                    format_author_block(r) if r is not None else f"{label}: not found"
                    for label, r in zip(ids, records, strict=True)
                )
                if any(r is None for r in records):
                    return UncategorizedToolResult(call_id="", content=text)
        except PaperError as e:
            return error(str(e))
        self.cache[key] = text
        return UncategorizedToolResult(call_id="", content=text)

    async def _search(self, query: str, *, limit: int | None) -> str:
        """Return the authors named like ``query``, most prolific first."""
        found = await asyncio.to_thread(
            search_authors,
            query,
            limit=limit,
            session=self.session,
        )
        if not found.records:
            return "(no results)"
        return "\n".join(format_author_line(r) for r in found.records) + (
            truncation_notice(len(found.records), found.total)
        )

    async def _papers(self, author_id: str, a: PaperAuthor.Arguments) -> str:
        """Return one author's papers, one per line."""
        listing = await asyncio.to_thread(
            author_papers,
            author_id,
            limit=a.limit,
            year_from=a.year_from,
            year_to=a.year_to,
            session=self.session,
        )
        if not listing.records:
            return "(no results)"
        body = "\n".join(
            format_record(r, abstract_chars=a.abstract_chars) for r in listing.records
        )
        if not listing.complete:
            body += "\n... (more matches exist; raise 'limit' or narrow the years)"
        return body
