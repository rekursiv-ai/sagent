"""PaperDetails: paper metadata, and the citation graph around one paper.

Typical usage example:

  result = await PaperDetails(session=session).call(context, {"ids": ["2106.15928"]})
  result = await PaperDetails(session=session).call(
      context, {"ids": "2106.15928", "operation": "citations"},
  )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.resources import files
from typing import TYPE_CHECKING, Annotated, Literal

import asyncio

from wesearch.fetch import FetchSession
from wesearch.paper.details import (
    citations,
    metadata,
    metadata_batch,
    references,
)
from wesearch.paper.errors import PaperError
from wesearch.paper.ids import normalize_id, s2_wire_id
from wesearch.paper.render import format_block

import cachetools

from sagent.session.types import UncategorizedToolResult
from sagent.tools.display import ToolDisplay
from sagent.tools.paper_common import (
    error,
    listed,
    paper_ids,
    summary_ids,
)
from sagent.tools.types import Arg, parse, schema_of


if TYPE_CHECKING:
    from collections.abc import Mapping

    from treekle import PlainTree

    from sagent.tools.types import ToolContext


__all__ = [
    "PaperDetails",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class PaperDetails:
    """Look up papers, or walk the citation graph from one."""

    @dataclass(frozen=True, slots=True, kw_only=True)
    class Arguments:
        """What the model passes."""

        ids: Annotated[
            tuple[str, ...] | str,
            "Paper identifier(s): a single id as a bare string, or several as an"
            " array. Each is a DOI (10.xxxx/yyy, optional doi:/https://doi.org/"
            " prefix) or an arXiv id (2106.15928, arXiv:2106.15928, or legacy"
            " hep-th/9901001). For metadata, pass every id you need at once: they"
            " are looked up in ONE batched Semantic Scholar request (up to 500),"
            " dramatically more efficient against the 1 request/second rate limit"
            " than one call per id. Results come back in input order."
            " 'references'/'citations' operate on a single paper, so pass exactly"
            " one id with 'operation'.",
        ]
        operation: Annotated[
            Literal["references", "citations"] | None,
            "Omit for plain metadata lookup. 'references' lists papers this one"
            " cites; 'citations' lists papers that cite this one. Requires exactly"
            " one id in 'ids'.",
        ] = None
        source: Annotated[
            Literal[
                "s2",
                "openalex",
            ],
            "Citation-graph backend for references/citations. Default 's2'"
            " (Semantic Scholar; DOI + arXiv, flags influential edges). 'openalex'"
            " (DOI only; independent quota, broad non-CS coverage; no influential"
            " flag)."
            " Ignored for plain metadata (always Semantic Scholar).",
        ] = "s2"
        influential_only: Annotated[
            bool,
            "Citations only: restrict to Semantic Scholar's 'influential' subset"
            " (high-signal citations). Default false.",
        ] = False
        year_from: Annotated[
            int | None,
            "Citations only: drop citations published before this year (inclusive"
            " lower bound).",
            Arg(minimum=0),
        ] = None
        limit: Annotated[
            int | None,
            "Max references/citations to return. Omit to let Semantic Scholar"
            " decide (one default page). Ignored for metadata.",
            Arg(minimum=1),
        ] = None
        abstract_chars: Annotated[
            int | None,
            "Truncate every abstract to this many characters. Omit for full abstracts.",
            Arg(minimum=1),
        ] = None

    name = "PaperDetails"
    description = (files(__package__) / "assets" / "paperdetails.md").read_text(
        encoding="utf-8",
    )
    schema = schema_of(Arguments)

    display: ToolDisplay = field(default=ToolDisplay())

    session: FetchSession = field(repr=False)
    """Connections, browser, and egress every lookup reuses; whoever made it closes it."""

    # Paper metadata is effectively immutable on an agent's timescale.
    cache: cachetools.LRUCache[tuple[object, ...], str] = field(
        default_factory=lambda: cachetools.LRUCache[tuple[object, ...], str](
            maxsize=1024,
        ),
        repr=False,
    )
    """Rendered results, by lookup."""

    def summary(self, arguments: Mapping[str, PlainTree]) -> str:
        """Return ``PaperDetails [operation] <ids>``."""
        operation = arguments.get("operation")
        ids = summary_ids(arguments)
        if operation in ("references", "citations"):
            return f"PaperDetails {operation} {ids}"
        return f"PaperDetails {ids}"

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
        """Look up the papers ``ids`` names, or one paper's references or citations.

        Args:
          context: The calling agent's state.
          arguments: What the model passed, as ``PaperDetails.Arguments``.

        Returns:
          result: A block per paper, or one line per related paper.

        Raises:
          ValueError: The arguments contradict each other.

        """
        del context
        a = parse(arguments, PaperDetails.Arguments)
        ids = paper_ids(a.ids)
        if not ids:
            raise ValueError("'ids' is empty.")
        if a.operation is not None and len(ids) != 1:
            raise ValueError("'operation' (references/citations) needs exactly one id.")
        if a.operation != "citations" and (
            a.influential_only or a.year_from is not None
        ):
            raise ValueError(
                "'influential_only' and 'year_from' only apply to operation='citations'.",
            )
        try:
            if a.operation is None:
                return await self._metadata(ids, a.abstract_chars)
            kind, canonical = normalize_id(ids[0])
            key = (
                a.operation,
                a.source,
                canonical,
                a.influential_only,
                a.year_from,
                a.limit,
                a.abstract_chars,
            )
            cached = self.cache.get(key)
            if cached is not None:
                return UncategorizedToolResult(call_id="", content=cached)
            source = "openalex" if a.source == "openalex" else "s2"
            listing = await (
                asyncio.to_thread(
                    references,
                    kind,
                    canonical,
                    limit=a.limit,
                    source=source,
                    session=self.session,
                )
                if a.operation == "references"
                else asyncio.to_thread(
                    citations,
                    kind,
                    canonical,
                    limit=a.limit,
                    source=source,
                    influential_only=a.influential_only,
                    year_from=a.year_from,
                    session=self.session,
                )
            )
        except PaperError as e:
            return error(str(e))
        text = listed(listing, a.abstract_chars, more="exist")
        self.cache[key] = text
        return UncategorizedToolResult(call_id="", content=text)

    async def _metadata(
        self,
        raw_ids: list[str],
        abstract_chars: int | None,
    ) -> UncategorizedToolResult:
        """Look up every id in one batched request, in input order."""
        wire = [s2_wire_id(*normalize_id(raw)) for raw in raw_ids]
        key = ("metadata", abstract_chars, tuple(wire))
        cached = self.cache.get(key)
        if cached is not None:
            return UncategorizedToolResult(call_id="", content=cached)
        if len(wire) == 1:
            records = [
                await asyncio.to_thread(
                    metadata,
                    *normalize_id(raw_ids[0]),
                    session=self.session,
                ),
            ]
        else:
            records = await asyncio.to_thread(
                metadata_batch,
                wire,
                session=self.session,
            )
        text = "\n\n".join(
            format_block(r, abstract_chars=abstract_chars)
            if r is not None
            else f"{label}: not found"
            for label, r in zip(raw_ids, records, strict=True)
        )
        if all(r is not None for r in records):
            self.cache[key] = text
        return UncategorizedToolResult(call_id="", content=text)
