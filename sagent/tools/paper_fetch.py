"""PaperFetch: download papers' PDFs by identifier, to a cache on disk.

``wesearch`` owns the source cascade (arXiv, then an open-access URL) and
its rate limits; the tool owns its arguments and the cache, keyed by
identifier, so asking again returns the same path without a download.

Typical usage example:

  result = await PaperFetch(session=session).call(context, {"ids": ["2106.15928"]})
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import asyncio

from wesearch.fetch import FetchSession
from wesearch.paper.errors import PaperError
from wesearch.paper.fetch import batch_oa_urls, download, looks_like_pdf
from wesearch.paper.ids import id_slug, normalize_id, s2_wire_id

from sagent.lib.userdirs import cache_dir
from sagent.session.types import UncategorizedToolResult
from sagent.tools.display import ToolDisplay
from sagent.tools.paper_common import paper_ids, summary_ids
from sagent.tools.types import parse, schema_of
from sagent.tools.write import write_file


if TYPE_CHECKING:
    from collections.abc import Mapping

    from treekle import PlainTree
    from wesearch.paper.types import IdType

    from sagent.tools.types import ToolContext


__all__ = [
    "PaperFetch",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class PaperFetch:
    """Download papers' PDFs by identifier."""

    @dataclass(frozen=True, slots=True, kw_only=True)
    class Arguments:
        """What the model passes."""

        ids: Annotated[
            tuple[str, ...] | str,
            "Paper identifier(s): a single id as a bare string, or several as an"
            " array. Each is a DOI (10.xxxx/yyy, optional doi:/https://doi.org/"
            " prefix) or arXiv id (2106.15928, arXiv:2106.15928, or legacy"
            " hep-th/9901001). When several are given, the open-access URL lookups"
            " are resolved in ONE batched Semantic Scholar request (up to 500)"
            " instead of one per id -- markedly more efficient against the 1"
            " request/second rate limit -- then the PDFs download concurrently."
            " Always pass every id you need at once.",
        ]

    name = "PaperFetch"
    description = (files(__package__) / "assets" / "paperfetch.md").read_text(
        encoding="utf-8",
    )
    schema = schema_of(Arguments)

    display: ToolDisplay = field(default=ToolDisplay())

    directory: Path = field(
        default_factory=lambda: cache_dir() / "rekursiv-ai" / "wesearch" / "papers",
    )
    """Where PDFs are kept, one per identifier."""

    session: FetchSession = field(repr=False)
    """Connections, browser, and egress every download reuses; whoever made it closes it."""

    def summary(self, arguments: Mapping[str, PlainTree]) -> str:
        """Return ``PaperFetch <ids>``."""
        return f"PaperFetch {summary_ids(arguments)}"

    def serialize_key(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> str | None:
        """Return ``None``: downloads are independent."""
        del context, arguments
        return None

    async def call(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> UncategorizedToolResult:
        """Download each paper ``ids`` names, or find it in the cache.

        Args:
          context: The calling agent's state.
          arguments: What the model passed, as ``PaperFetch.Arguments``.

        Returns:
          result: One line per id: where its PDF is, or why it is not.

        Raises:
          ValueError: ``ids`` is empty, or names something that is no paper id.

        """
        del context
        ids = paper_ids(parse(arguments, PaperFetch.Arguments).ids)
        if not ids:
            raise ValueError("'ids' is empty.")
        try:
            parsed = [normalize_id(raw) for raw in ids]
        except PaperError as e:
            raise ValueError(str(e)) from e
        # One batched lookup of open-access URLs; ``None`` means it failed, and
        # each download looks its own up.
        urls = (
            await asyncio.to_thread(
                batch_oa_urls,
                [s2_wire_id(*p) for p in parsed],
                session=self.session,
            )
            if len(parsed) > 1
            else None
        )
        lines = await asyncio.gather(
            *(
                self._fetch(kind, canonical, url=url, looked_up=urls is not None)
                for (kind, canonical), url in zip(
                    parsed,
                    urls if urls is not None else [None] * len(parsed),
                    strict=True,
                )
            ),
        )
        return UncategorizedToolResult(
            call_id="",
            content="\n".join(text for text, _ in lines),
            is_error=any(failed for _, failed in lines),
        )

    async def _fetch(
        self,
        kind: IdType,
        canonical: str,
        *,
        url: str | None,
        looked_up: bool,
    ) -> tuple[str, bool]:
        """Return where one paper's PDF is, and whether getting it failed."""
        path = self.directory / f"{id_slug(kind, canonical)}.pdf"
        try:
            with path.open("rb") as f:
                if looks_like_pdf(f.read(128)):
                    return f"Cached: {path}", False
        except OSError:
            pass
        try:
            body, source = await asyncio.to_thread(
                download,
                kind,
                canonical,
                oa_url=url,
                oa_looked_up=looked_up,
                session=self.session,
            )
        except PaperError as e:
            return str(e), True
        await asyncio.to_thread(write_file, path, body)
        return f"Downloaded via {source}: {path}", False
