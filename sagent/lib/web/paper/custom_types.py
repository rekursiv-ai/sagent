"""Backend-agnostic record types for :mod:`sagent.lib.web.paper`.

One common shape per concept so every backend (Semantic Scholar, OpenAlex,
SearXNG, Google Scholar) returns the same dataclass and a consumer works
against any source unchanged. All fields but the title/id are optional --
backends return sparse records (OpenAlex for very old works, Scholar with no
DOI) and we prefer ``None``/``""`` over fabricated values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


__all__ = [
    "AuthorRecord",
    "IdType",
    "PaperRecord",
]

# A paper identifier is either a DOI or an arXiv id; see :mod:`.ids`.
IdType = Literal["doi", "arxiv"]


@dataclass(frozen=True, slots=True, kw_only=True)
class PaperRecord:
    """Backend-agnostic paper record.

    All fields are optional except ``title`` -- backends occasionally return
    sparse records (e.g. OpenAlex for very old papers), and we prefer
    ``None``/``""`` over fabricating values.
    """

    title: str
    """Paper title."""

    authors: tuple[str, ...] = ()
    """Author display names in publication order."""

    year: int | None = None
    """Publication year."""

    venue: str | None = None
    """Publication venue (journal or conference)."""

    doi: str | None = None
    """DOI identifier (no prefix)."""

    arxiv_id: str | None = None
    """arXiv identifier (no prefix)."""

    abstract: str | None = None
    """Abstract text."""

    citation_count: int | None = None
    """Number of citing papers reported by the backend."""

    reference_count: int | None = None
    """Number of references reported by the backend."""

    open_access_pdf: str | None = None
    """URL of an open-access PDF, when available."""

    sources: tuple[str, ...] = field(default_factory=tuple)
    """Backends that returned this record (e.g. ``("s2",)`` or
    ``("s2", "openalex")``)."""

    is_influential: bool | None = None
    """Citation-only: S2's ``isInfluential`` flag (``None`` when unknown)."""

    def merge(self, other: PaperRecord) -> PaperRecord:
        """Combine two records of the same paper - prefer this record's values.

        Every field is taken from ``self`` unless it is empty (``None`` or an
        empty string/tuple), in which case ``other`` supplies it; ``sources`` is
        the union of both. Every field is listed explicitly, so the merge is
        obvious by inspection and a newly-added field is a visible edit here.

        Args:
          other: The lower-priority record to fill this record's gaps.

        Returns:
          merged: A new record with this record's values, gaps filled by
            ``other`` and the ``sources`` unioned.

        """
        return PaperRecord(
            title=self.title or other.title,
            authors=self.authors or other.authors,
            year=self.year if self.year is not None else other.year,
            venue=self.venue or other.venue,
            doi=self.doi or other.doi,
            arxiv_id=self.arxiv_id or other.arxiv_id,
            abstract=self.abstract or other.abstract,
            citation_count=(
                self.citation_count
                if self.citation_count is not None
                else other.citation_count
            ),
            reference_count=(
                self.reference_count
                if self.reference_count is not None
                else other.reference_count
            ),
            open_access_pdf=self.open_access_pdf or other.open_access_pdf,
            is_influential=(
                self.is_influential
                if self.is_influential is not None
                else other.is_influential
            ),
            sources=tuple(dict.fromkeys((*self.sources, *other.sources))),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class AuthorRecord:
    """Backend-agnostic author record.

    ``author_id`` is the Semantic Scholar opaque integer id (as a string).
    Other fields are optional -- sparse records are common for lesser-known
    authors, and we prefer ``None`` over fabricated values.
    """

    author_id: str
    """Semantic Scholar opaque integer id, as a string."""

    name: str
    """Display name."""

    aliases: tuple[str, ...] = ()
    """Alternate name spellings reported by the backend."""

    affiliations: tuple[str, ...] = ()
    """Institutional affiliations in backend-provided order."""

    homepage: str | None = None
    """Homepage URL, when available."""

    h_index: int | None = None
    """h-index reported by the backend."""

    citation_count: int | None = None
    """Total citations across the author's published work."""

    paper_count: int | None = None
    """Total published papers attributed to the author."""
