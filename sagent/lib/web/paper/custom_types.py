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
