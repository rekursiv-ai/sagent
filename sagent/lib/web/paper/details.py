"""Paper metadata and citation-graph lookup.

Single/batched metadata (Semantic Scholar), backward edges (:func:`references`)
and forward edges (:func:`citations`) over a ``source``-selected graph backend
(S2 or OpenAlex), plus the Google Scholar cited-by pivot (:func:`cited_by`).
Sync -- a coroutine call site lifts these with ``asyncio.to_thread``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

from sagent.lib.custom_json import MutableJSON
from sagent.lib.web.paper.custom_types import IdType, PaperRecord
from sagent.lib.web.paper.errors import PaperError
from sagent.lib.web.paper.ids import s2_wire_id
from sagent.lib.web.paper.paginate import Page
from sagent.lib.web.paper.providers import (
    openalex,
    s2,
)


__all__ = [
    "GraphSource",
    "Listing",
    "citations",
    "metadata",
    "metadata_batch",
    "references",
]

# Citation-graph backends. S2 accepts DOI + arXiv and flags influential edges;
# OpenAlex accepts DOI only (its arXiv keying is unreliable) and has no
# influence flag, but offers an independent quota and broader non-CS coverage.
GraphSource = Literal["s2", "openalex"]


@dataclass(frozen=True, slots=True, kw_only=True)
class Listing:
    """A reference/citation/author-papers listing.

    Attributes:
      records: The paper records in the listing (trimmed to ``limit``).
      complete: True when the backend cursor was exhausted; False when more
        matches may exist beyond those returned.

    """

    records: list[PaperRecord]
    complete: bool


def metadata(kind: IdType, canonical: str) -> PaperRecord:
    """Fetch single-paper metadata from S2."""
    data = s2.get(
        f"/paper/{s2_wire_id(kind, canonical)}", {"fields": s2.S2_PAPER_FIELDS_STR}
    )
    return s2.paper_record_from(data)


def metadata_batch(wire_ids: list[str]) -> list[PaperRecord | None]:
    """Batch-fetch metadata for many wire ids; ``None`` per unresolved id."""
    records = s2.batch(wire_ids, s2.S2_PAPER_FIELDS_STR, endpoint="paper")
    return [s2.paper_record_from(r) if r is not None else None for r in records]


def references(
    kind: IdType, canonical: str, *, limit: int | None, source: GraphSource = "s2"
) -> Listing:
    """Fetch papers the given paper cites (backward citation edges).

    Args:
      kind: Seed identifier type.
      canonical: Bare seed identifier.
      limit: Max references, or ``None`` for the backend's default page.
      source: Citation-graph backend (``"s2"`` or ``"openalex"``).

    Returns:
      listing: A :class:`Listing` of cited papers.

    Raises:
      PaperError: On backend failure (e.g. an arXiv seed under ``"openalex"``).

    """
    if source == "openalex":
        records, complete = openalex.references(kind, canonical, limit=limit)
        return Listing(records=records, complete=complete)
    fields = ",".join(
        ("isInfluential", *(f"citedPaper.{f}" for f in s2.S2_PAPER_FIELDS))
    )
    page = s2.paginate(
        f"/paper/{s2_wire_id(kind, canonical)}/references",
        {"fields": fields},
        limit=limit,
    )
    return _edge_listing(page, inner_key="citedPaper")


def citations(
    kind: IdType,
    canonical: str,
    *,
    limit: int | None,
    source: GraphSource = "s2",
    influential_only: bool = False,
    year_from: int | None = None,
) -> Listing:
    """Fetch papers that cite the given paper (forward edges), with filters.

    Args:
      kind: Seed identifier type.
      canonical: Bare seed identifier.
      limit: Max citations, or ``None`` for the backend's default page.
      source: Citation-graph backend (``"s2"`` or ``"openalex"``).
      influential_only: S2 only -- restrict to S2's influential subset. OpenAlex
        has no influence flag, so a truthy value under ``"openalex"`` errors.
      year_from: Drop citations published before this year.

    Returns:
      listing: A :class:`Listing` of citing papers.

    Raises:
      PaperError: On backend failure, or ``influential_only`` under
        ``"openalex"``.

    """
    if source == "openalex":
        if influential_only:
            raise PaperError(
                "'influential_only' is S2-only; OpenAlex has no influence flag."
            )
        records, total, complete = openalex.citations(
            kind, canonical, limit=limit, year_from=year_from
        )
        return Listing(records=records, complete=complete or total <= len(records))
    fields = ",".join(
        ("isInfluential", *(f"citingPaper.{f}" for f in s2.S2_PAPER_FIELDS))
    )

    def keep(entry: MutableJSON) -> bool:
        if influential_only and not entry.get("isInfluential"):
            return False
        if year_from is None:
            return True
        inner = cast(MutableJSON, entry.get("citingPaper") or {})
        year = inner.get("year")
        return isinstance(year, int) and year >= year_from

    page = s2.paginate(
        f"/paper/{s2_wire_id(kind, canonical)}/citations",
        {"fields": fields},
        limit=limit,
        keep=keep,
    )
    return _edge_listing(page, inner_key="citingPaper")


def _edge_listing(page: Page, *, inner_key: str) -> Listing:
    """Extract paper records from S2 citation-edge entries into a Listing."""
    records: list[PaperRecord] = []
    for e in page.entries:
        inner = cast(MutableJSON, e.get(inner_key) or {})
        if not inner:
            continue
        records.append(
            s2.paper_record_from(
                inner, is_influential=cast(bool | None, e.get("isInfluential"))
            )
        )
    return Listing(records=records, complete=page.complete)
