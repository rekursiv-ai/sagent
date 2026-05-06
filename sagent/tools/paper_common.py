"""Shared helpers for the Paper* tool family.

Used by :mod:`.paper_details`, :mod:`.paper_search`, :mod:`.paper_author`,
and :mod:`.paper_fetch`:

- :func:`normalize_id` - detect DOI vs arXiv shape, return canonical form.
- :class:`PaperRecord` / :class:`AuthorRecord` - common record shapes
  across backends (S2, OpenAlex).
- :func:`format_record` / :func:`format_block` / :func:`format_author_line`
  / :func:`format_author_block` - list-line and metadata block renderings;
  agent consumes text, not JSON.
- :func:`openalex_reconstruct_abstract` - OpenAlex ships abstracts as
  an inverted index (word → token positions); fold back to plain text.
- :data:`S2_BASE`, :func:`s2_get`, :func:`s2_paper_to_record`
  - shared Semantic Scholar Graph API client + record conversion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, cast

import asyncio
import json
import os
import re

from sagent.custom_types import Message, TextMessage
from sagent.lib.json import MutableJSON
from sagent.lib.web.fetch import FetchError, fetch


_LIMIT_MAX = 1000


def short_id(raw: str) -> str:
    """Truncate an identifier to at most 40 characters for display.

    Args:
      raw: Original identifier string.

    Returns:
      short: The original or an ellipsis-prefixed tail.

    """
    return raw if len(raw) <= 40 else "…" + raw[-38:]


def clamp_limit(limit: int | None, default: int) -> int:
    """Clamp a caller-supplied limit to [1, 1000], falling back to default.

    Args:
      limit: Caller-supplied limit, or ``None`` to use the default.
      default: Value returned when ``limit`` is ``None``.

    Returns:
      clamped: The effective limit.

    """
    if limit is None:
        return default
    return max(1, min(int(limit), _LIMIT_MAX))


# -- Identifier detection --------------------------------------------------

# DOI shape: 10.<registrant>/<suffix>. Registrant is 4+ digits (ISO 26324).
# Suffix is opaque; may contain slashes, dots, colons, etc.
_DOI_RE = re.compile(r"^(10\.\d{4,})/(\S+)$")

# arXiv new-style id: NNNN.NNNNN with optional version (v1, v2, ...).
_ARXIV_NEW_RE = re.compile(r"^(\d{4}\.\d{4,5})(v\d+)?$")

# arXiv old-style id: <subject>/NNNNNNN (e.g. hep-th/9901001). Rare but
# S2 and arXiv both still honor these for papers pre-April-2007.
_ARXIV_OLD_RE = re.compile(r"^([a-z-]+(?:\.[A-Z]{2})?)/(\d{7})(v\d+)?$")


IdType = Literal["doi", "arxiv"]


def normalize_id(raw: str) -> tuple[IdType, str] | Message:
    """Parse a user-supplied identifier into (type, canonical).

    Accepts DOIs with or without the ``https://doi.org/`` / ``doi:``
    prefix, arXiv ids with or without ``arXiv:`` / ``arxiv.org/abs/``
    wrapping, bare new-style ids (``2106.15928``), and old-style ids
    (``hep-th/9901001``).

    Args:
      raw: User-supplied identifier string.

    Returns:
      kind: ``"doi"`` or ``"arxiv"``.
      canonical: Bare identifier with no prefix. Returns a ``Message``
        with ``text/x-error`` descriptor when the shape matches
        neither DOI nor arXiv.

    """
    s = raw.strip()
    if not s:
        return TextMessage("Empty identifier.", "text/x-error")

    # Strip common URL / scheme wrappers.
    lower = s.lower()
    for prefix in (
        "https://doi.org/",
        "http://doi.org/",
        "https://dx.doi.org/",
        "http://dx.doi.org/",
        "doi:",
    ):
        if lower.startswith(prefix):
            s = s[len(prefix) :]
            lower = s.lower()
            break
    for prefix in (
        "https://arxiv.org/abs/",
        "http://arxiv.org/abs/",
        "https://arxiv.org/pdf/",
        "http://arxiv.org/pdf/",
        "arxiv:",
        "arxiv.org/abs/",
    ):
        if lower.startswith(prefix):
            s = s[len(prefix) :]
            # arXiv PDF urls often end in ``.pdf`` - strip.
            if s.lower().endswith(".pdf"):
                s = s[:-4]
            return "arxiv", s

    if _DOI_RE.match(s):
        return "doi", s
    if _ARXIV_NEW_RE.match(s):
        return "arxiv", s
    if _ARXIV_OLD_RE.match(s):
        return "arxiv", s
    return TextMessage(
        (
            f"Unrecognized identifier shape: {raw!r}. "
            "Expected DOI (10.xxxx/yyy) or arXiv id (NNNN.NNNNN, "
            "arXiv:NNNN.NNNNN, or hep-th/NNNNNNN)."
        ),
        "text/x-error",
    )


def s2_wire_id(kind: IdType, canonical: str) -> str:
    """Build the prefixed form S2 accepts in URL path: ``DOI:...``/``ARXIV:...``.

    Args:
      kind: Identifier type.
      canonical: Bare identifier.

    Returns:
      wire_id: Prefixed identifier string for S2 API calls.

    """
    return f"DOI:{canonical}" if kind == "doi" else f"ARXIV:{canonical}"


def id_slug(kind: IdType, canonical: str) -> str:
    """Build a filesystem-safe slug for a paper id.

    Args:
      kind: Identifier type.
      canonical: Bare identifier.

    Returns:
      slug: String safe for use as a filename component.

    """
    prefix = "doi" if kind == "doi" else "arxiv"
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", canonical)
    return f"{prefix}_{safe}"


def papers_cache_dir() -> Path:
    """Return the default on-disk cache directory for downloaded PDFs.

    Returns:
      path: ``~/.sagent/papers/``.

    """
    return Path.home() / ".sagent" / "papers"


# -- Common paper record ---------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class PaperRecord:
    """Backend-agnostic paper record.

    All fields are optional except ``title`` - backends occasionally
    return sparse records (e.g. OpenAlex for very old papers), and we
    prefer ``None``/``""`` over fabricating values.
    """

    title: str
    authors: tuple[str, ...] = ()
    year: int | None = None
    venue: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    abstract: str | None = None
    citation_count: int | None = None
    reference_count: int | None = None
    open_access_pdf: str | None = None
    # Backends that returned this record, e.g. ``("s2",)`` or
    # ``("s2", "openalex")``. Emitted as ``sources: s2,openalex`` in the
    # formatted output when non-empty.
    sources: tuple[str, ...] = field(default_factory=tuple)
    # Citation-only: S2's isInfluential flag. ``None`` when unknown.
    is_influential: bool | None = None


def _format_authors(authors: tuple[str, ...], limit: int = 3) -> str:
    """Render first-``limit`` authors with ``+N`` suffix for the rest."""
    if not authors:
        return "unknown"
    shown = ", ".join(authors[:limit])
    extra = len(authors) - limit
    if extra > 0:
        return f"{shown} +{extra}"
    return shown


def _id_prefix(rec: PaperRecord) -> str:
    """Bracketed identifier prefix: ``[doi:... | arXiv:...]`` / subset."""
    parts: list[str] = []
    if rec.doi:
        parts.append(f"doi:{rec.doi}")
    if rec.arxiv_id:
        parts.append(f"arXiv:{rec.arxiv_id}")
    inner = " | ".join(parts) if parts else "no-id"
    return f"[{inner}]"


def _trim_abstract(abstract: str | None, cap: int | None) -> str | None:
    """Apply caller-supplied character cap to an abstract, if any."""
    if abstract is None:
        return None
    if cap is None or cap <= 0 or len(abstract) <= cap:
        return abstract
    return abstract[:cap].rstrip() + "..."


def format_record(
    rec: PaperRecord,
    abstract_chars: int | None = None,
) -> str:
    """Format a paper as one greppable line, followed by optional abstract.

    Line shape::

        [doi:... | arXiv:...] Title - Authors, Year, Venue - cites:N refs:N [OA] · sources: s2,openalex · influential

    Abstract, when present, follows on an indented continuation line so
    the hit record stays visually delineated even with multi-paragraph
    abstracts.

    Args:
      rec: Paper record to format.
      abstract_chars: Max characters for the abstract, or ``None`` for full.

    Returns:
      text: Formatted record string.

    """
    id_block = _id_prefix(rec)
    year_str = str(rec.year) if rec.year is not None else "?"
    venue_str = rec.venue or "?"
    meta_parts: list[str] = []
    if rec.citation_count is not None:
        meta_parts.append(f"cites:{rec.citation_count}")
    if rec.reference_count is not None:
        meta_parts.append(f"refs:{rec.reference_count}")
    if rec.open_access_pdf:
        meta_parts.append("OA")
    if rec.sources:
        meta_parts.append("sources: " + ",".join(rec.sources))
    if rec.is_influential is True:
        meta_parts.append("influential")
    meta = " · ".join(meta_parts)

    header = (
        f"{id_block} {rec.title} - {_format_authors(rec.authors)}, "
        f"{year_str}, {venue_str}"
    )
    if meta:
        header += f" - {meta}"
    abstract = _trim_abstract(rec.abstract, abstract_chars)
    if abstract:
        # Indent continuation lines for visual grouping.
        body = "\n".join(f"    {line}" for line in abstract.splitlines())
        return f"{header}\n    abstract:\n{body}"
    return header


def format_block(
    rec: PaperRecord,
    abstract_chars: int | None = None,
) -> str:
    """Render a multi-line metadata block for ``PaperDetails`` lookup.

    Args:
      rec: Paper record to format.
      abstract_chars: Max characters for the abstract, or ``None`` for full.

    Returns:
      block: Newline-joined key-value lines.

    """
    lines: list[str] = []
    if rec.arxiv_id:
        lines.append(f"id: arXiv:{rec.arxiv_id}")
    if rec.doi:
        lines.append(f"doi: {rec.doi}")
    lines.append(f"title: {rec.title}")
    lines.append(f"authors: {', '.join(rec.authors) if rec.authors else 'unknown'}")
    if rec.year is not None:
        lines.append(f"year: {rec.year}")
    if rec.venue:
        lines.append(f"venue: {rec.venue}")
    if rec.citation_count is not None:
        lines.append(f"citation_count: {rec.citation_count}")
    if rec.reference_count is not None:
        lines.append(f"reference_count: {rec.reference_count}")
    if rec.open_access_pdf:
        lines.append(f"open_access_pdf: {rec.open_access_pdf}")
    if rec.sources:
        lines.append(f"sources: {','.join(rec.sources)}")
    abstract = _trim_abstract(rec.abstract, abstract_chars)
    if abstract:
        lines.append(f"abstract: {abstract}")
    return "\n".join(lines)


# -- Author record --------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class AuthorRecord:
    """Backend-agnostic author record.

    ``author_id`` is the Semantic Scholar opaque integer id (as a string).
    Other fields are optional - sparse records are common for lesser-known
    authors, and we prefer ``None`` over fabricated values.
    """

    author_id: str
    name: str
    aliases: tuple[str, ...] = ()
    affiliations: tuple[str, ...] = ()
    homepage: str | None = None
    h_index: int | None = None
    citation_count: int | None = None
    paper_count: int | None = None


def format_author_line(rec: AuthorRecord) -> str:
    """Format one greppable line per author for search results.

    Shape::

        [author:1741101] Yoshua Bengio - h-index:245 cites:500000 papers:800 - Université de Montréal

    Args:
      rec: Author record to format.

    Returns:
      line: Single-line summary string.

    """
    id_block = f"[author:{rec.author_id}]"
    meta: list[str] = []
    if rec.h_index is not None:
        meta.append(f"h-index:{rec.h_index}")
    if rec.citation_count is not None:
        meta.append(f"cites:{rec.citation_count}")
    if rec.paper_count is not None:
        meta.append(f"papers:{rec.paper_count}")
    line = f"{id_block} {rec.name}"
    if meta:
        line += f" - {' '.join(meta)}"
    if rec.affiliations:
        # Primary affiliation only on the one-liner; the detail block
        # lists all of them.
        line += f" - {rec.affiliations[0]}"
    return line


def format_author_block(rec: AuthorRecord) -> str:
    """Render a multi-line metadata block for ``PaperAuthor`` details lookup.

    Args:
      rec: Author record to format.

    Returns:
      block: Newline-joined key-value lines.

    """
    lines: list[str] = [
        f"author_id: {rec.author_id}",
        f"name: {rec.name}",
    ]
    if rec.aliases:
        lines.append(f"aliases: {', '.join(rec.aliases)}")
    if rec.affiliations:
        lines.append(f"affiliations: {', '.join(rec.affiliations)}")
    if rec.homepage:
        lines.append(f"homepage: {rec.homepage}")
    if rec.h_index is not None:
        lines.append(f"h_index: {rec.h_index}")
    if rec.citation_count is not None:
        lines.append(f"citation_count: {rec.citation_count}")
    if rec.paper_count is not None:
        lines.append(f"paper_count: {rec.paper_count}")
    return "\n".join(lines)


# -- OpenAlex abstract reconstruction -------------------------------------


def openalex_reconstruct_abstract(
    inverted: dict[str, list[int]] | None,
) -> str | None:
    """Rebuild plain text from OpenAlex's inverted-index abstract shape.

    OpenAlex ships abstracts as ``{"word": [positions...]}`` - space-join
    tokens in position order.

    Args:
      inverted: Inverted-index mapping, or ``None``.

    Returns:
      abstract: Reconstructed plain text, or ``None`` when unavailable.

    """
    if not inverted:
        return None
    positions: dict[int, str] = {}
    for word, idxs in inverted.items():
        for i in idxs:
            positions[i] = word
    if not positions:
        return None
    return " ".join(positions[i] for i in sorted(positions))


# -- Truncation notice -----------------------------------------------------


def truncation_notice(shown: int, total: int) -> str:
    """Build a ``... showing N of M`` suffix for paginated output.

    Callers append when ``total > shown`` and the backend exposed a total.

    Args:
      shown: Number of results displayed.
      total: Total number of results available.

    Returns:
      notice: Truncation suffix, or empty string when no truncation.

    """
    if total > shown and total > 0:
        return f"\n... (showing {shown} of {total}; tighten filters for more)"
    return ""


# -- Semantic Scholar Graph API client ------------------------------------

S2_BASE = "https://api.semanticscholar.org/graph/v1"
S2_TIMEOUT = 60.0


def _s2_headers() -> dict[str, str]:
    headers: dict[str, str] = {"Accept": "application/json"}
    key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
    if key:
        headers["x-api-key"] = key
    return headers


async def s2_get(path: str, params: dict[str, str | int]) -> MutableJSON | Message:
    """GET an S2 Graph API path, injecting API key and normalizing errors.

    Args:
      path: API path relative to ``S2_BASE`` (e.g. ``/paper/search``).
      params: Query parameters.

    Returns:
      data: Parsed JSON response, or a ``Message`` with ``text/x-error``
        on 404 / 429 / other HTTP failures.

    """
    try:
        raw = cast(  # pyright: ignore[reportUnnecessaryCast] -- ty can't narrow to_thread through overloads
            bytes,
            await asyncio.to_thread(
                fetch,
                url=f"{S2_BASE}{path}",
                params=params,
                headers=_s2_headers(),
                timeout_sec=S2_TIMEOUT,
            ),
        )
    except FetchError as e:
        if e.status == 404:
            return TextMessage(f"Not found: {path}", "text/x-error")
        if e.status == 429:
            return TextMessage(
                "Semantic Scholar rate limit hit. Retry shortly.",
                "text/x-error",
            )
        return TextMessage(
            f"Semantic Scholar HTTP {e.status}: "
            f"{e.body[:200].decode(errors='replace')}",
            "text/x-error",
        )
    return cast(MutableJSON, json.loads(raw))


def s2_paper_to_record(
    data: MutableJSON,
    *,
    sources: tuple[str, ...] = ("s2",),
    is_influential: bool | None = None,
) -> PaperRecord:
    """Convert an S2 paper dict into a :class:`PaperRecord`.

    Shared across the S2 endpoints ``/paper/{id}``, ``/paper/search``,
    ``/paper/{id}/references``, ``/paper/{id}/citations``, and
    ``/author/{id}/papers``.

    Args:
      data: Raw S2 paper JSON object.
      sources: Backend tags to attach to the record.
      is_influential: S2's ``isInfluential`` flag, or ``None``.

    Returns:
      record: Populated paper record.

    """
    ids = cast(MutableJSON, data.get("externalIds") or {})
    authors_raw = cast(list[MutableJSON], data.get("authors") or [])
    authors = tuple(str(a.get("name") or "") for a in authors_raw if a.get("name"))
    oa = cast(MutableJSON, data.get("openAccessPdf") or {})
    doi = ids.get("DOI")
    arxiv = ids.get("ArXiv")
    return PaperRecord(
        title=str(data.get("title") or "(untitled)"),
        authors=authors,
        year=cast(int | None, data.get("year")),
        venue=(str(data["venue"]) if data.get("venue") else None),
        doi=(str(doi) if doi else None),
        arxiv_id=(str(arxiv) if arxiv else None),
        abstract=(str(data["abstract"]) if data.get("abstract") else None),
        citation_count=cast(int | None, data.get("citationCount")),
        reference_count=cast(int | None, data.get("referenceCount")),
        open_access_pdf=(str(oa["url"]) if oa.get("url") else None),
        sources=sources,
        is_influential=is_influential,
    )
