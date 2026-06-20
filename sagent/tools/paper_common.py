"""Shared helpers for the Paper* tool family.

Used by :mod:`.paper_details`, :mod:`.paper_search`, :mod:`.paper_author`,
and :mod:`.paper_fetch`, grouped by concern:

- **Identifiers** -- :func:`normalize_id` (detect DOI vs arXiv shape, return
  canonical form), :func:`s2_wire_id`, :func:`id_slug`, :func:`short_id`.
- **Tool arguments** -- :func:`resolve_id_args` / :func:`parse_optional_ids`
  (validate the ``ids`` list), :func:`validate_limit`,
  :func:`year_in_range` (shared client-side year filter),
  :func:`summary_ids`.
- **Records** -- :class:`PaperRecord` / :class:`AuthorRecord` (common shapes
  across S2 and OpenAlex), :func:`s2_paper_to_record`,
  :func:`openalex_reconstruct_abstract` (fold OpenAlex's inverted-index
  abstract back to text).
- **Rendering** -- :func:`format_record` / :func:`format_block` /
  :func:`format_author_line` / :func:`format_author_block` /
  :func:`truncation_notice`; the agent consumes text, not JSON.
- **S2 client** -- :data:`S2_BASE`, :data:`S2_PAPER_FIELDS` /
  :data:`S2_PAPER_FIELDS_STR`, :func:`s2_get`, :func:`s2_batch`,
  :func:`s2_paginate` (cursor walk, returning a :class:`Page`). The
  cross-process rate gate and 429 backoff live here too, behind these
  calls.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, MutableMapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, cast

import asyncio
import functools
import json
import os
import re
import time

from sagent.lib.custom_json import MutableJSON
from sagent.lib.ratelimit import FileStore, SystemClock, TokenBucketRateLimiter
from sagent.lib.web.fetch import FetchError, fetch
from sagent.types.runtime import ToolResult


def year_in_range(year: object, *, year_from: int | None, year_to: int | None) -> bool:
    """Whether ``year`` is a known int within the inclusive bounds.

    Shared by the client-side year filters in PaperDetails and PaperAuthor.
    A missing or non-int year is treated as out of range -- S2 omits the
    field on undated works, which a year filter should exclude.

    Args:
      year: Raw ``year`` value from an S2 record (may be any JSON type).
      year_from: Inclusive lower bound, or ``None`` for no lower bound.
      year_to: Inclusive upper bound, or ``None`` for no upper bound.

    Returns:
      in_range: True when ``year`` is an int satisfying both bounds.

    """
    if not isinstance(year, int):
        return False
    if year_from is not None and year < year_from:
        return False
    return not (year_to is not None and year > year_to)


def short_id(raw: str) -> str:
    """Truncate an identifier to at most 40 characters for display.

    Args:
      raw: Original identifier string.

    Returns:
      short: The original or an ellipsis-prefixed tail.

    """
    return raw if len(raw) <= 40 else "…" + raw[-38:]


# DOI shape: 10.<registrant>/<suffix>. Registrant is 4+ digits (ISO 26324).
# Suffix is opaque; may contain slashes, dots, colons, etc.
_DOI_RE = re.compile(r"^(10\.\d{4,})/(\S+)$")

# arXiv new-style id: NNNN.NNNNN with optional version (v1, v2, ...).
_ARXIV_NEW_RE = re.compile(r"^(\d{4}\.\d{4,5})(v\d+)?$")

# arXiv old-style id: <subject>/NNNNNNN (e.g. hep-th/9901001). Rare but
# S2 and arXiv both still honor these for papers pre-April-2007.
_ARXIV_OLD_RE = re.compile(r"^([a-z-]+(?:\.[A-Z]{2})?)/(\d{7})(v\d+)?$")

IdType = Literal["doi", "arxiv"]


def normalize_id(raw: str) -> tuple[IdType, str] | ToolResult:
    """Parse a user-supplied identifier into (type, canonical).

    Accepts DOIs with or without the ``https://doi.org/`` / ``doi:``
    prefix, arXiv ids with or without ``arXiv:`` / ``arxiv.org/abs/``
    wrapping, bare new-style ids (``2106.15928``), and old-style ids
    (``hep-th/9901001``).

    Args:
      raw: User-supplied identifier string.

    Returns:
      kind: ``"doi"`` or ``"arxiv"``.
      canonical: Bare identifier with no prefix. Returns a ``ToolResult``
        with ``is_error=True`` when the shape matches neither DOI nor
        arXiv.

    """
    s = raw.strip()
    if not s:
        return ToolResult(call_id="", content="Empty identifier.", is_error=True)

    # Strip common URL / scheme wrappers. A matched prefix pins the family,
    # so a ``doi:`` value is never re-interpreted as arXiv (or vice versa).
    lower = s.lower()
    forced: IdType | None = None
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
            forced = "doi"
            break
    if forced is None:
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
                forced = "arxiv"
                break

    if forced != "arxiv" and _DOI_RE.match(s):
        return "doi", s
    if forced != "doi" and (_ARXIV_NEW_RE.match(s) or _ARXIV_OLD_RE.match(s)):
        return "arxiv", s
    return ToolResult(
        call_id="",
        content=(
            f"Unrecognized identifier shape: {raw!r}. "
            "Expected DOI (10.xxxx/yyy) or arXiv id (NNNN.NNNNN, "
            "arXiv:NNNN.NNNNN, or hep-th/NNNNNNN)."
        ),
        is_error=True,
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


@dataclass(frozen=True, slots=True, kw_only=True)
class PaperRecord:
    """Backend-agnostic paper record.

    All fields are optional except ``title`` - backends occasionally
    return sparse records (e.g. OpenAlex for very old papers), and we
    prefer ``None``/``""`` over fabricating values.
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
    ``("s2", "openalex")``). Emitted as ``sources: s2,openalex`` in the
    formatted output when non-empty."""

    is_influential: bool | None = None
    """Citation-only: S2's ``isInfluential`` flag (``None`` when unknown)."""


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


@dataclass(frozen=True, slots=True, kw_only=True)
class AuthorRecord:
    """Backend-agnostic author record.

    ``author_id`` is the Semantic Scholar opaque integer id (as a string).
    Other fields are optional - sparse records are common for lesser-known
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


S2_BASE = "https://api.semanticscholar.org/graph/v1"
# Healthy S2 latency is sub-second to a few seconds even for a 100-item
# reference page; 10s clears the slow tail (batch POST, deep pagination) with
# wide margin. A larger value only prolongs a silent hang when S2 is wedged,
# blocking an interactive agent turn (live 2026-06-19: a stalled fetch sat
# idle ~60s against the old ceiling before the user interrupted it).
S2_TIMEOUT = 10.0

# The S2 paper fields every Paper* tool requests. Shared so the set stays in
# sync across tools; ``s2_paper_to_record`` consumes exactly these. Nested
# refs/cites endpoints prefix each with ``citedPaper.`` / ``citingPaper.``.
S2_PAPER_FIELDS: tuple[str, ...] = (
    "paperId",
    "externalIds",
    "title",
    "abstract",
    "authors",
    "year",
    "venue",
    "citationCount",
    "referenceCount",
    "openAccessPdf",
)
S2_PAPER_FIELDS_STR = ",".join(S2_PAPER_FIELDS)

# S2 now *requires* exponential backoff on 429 (API release notes); retry
# a throttled request a few times before surfacing the error to the agent.
_S2_MAX_RETRIES = 4
_S2_BACKOFF_BASE = 1.0


def _s2_headers() -> dict[str, str]:
    """Build S2 request headers, injecting ``x-api-key`` when present in env."""
    headers: dict[str, str] = {"Accept": "application/json"}
    key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
    if key:
        headers["x-api-key"] = key
    return headers


def summary_ids(args: Mapping[str, object]) -> str:
    """Render a short display label for an ``ids`` argument.

    Args:
      args: Tool arguments; reads the ``ids`` list.

    Returns:
      label: The first id (truncated), plus ``(+N more)`` when several,
        or ``"?"`` when none are present.

    """
    ids = parse_optional_ids(args)
    if isinstance(ids, ToolResult) or not ids:
        return "?"
    head = short_id(ids[0])
    return head if len(ids) == 1 else f"{head} (+{len(ids) - 1} more)"


def _is_paper_id(token: str) -> bool:
    """Whether ``token`` parses as a DOI or arXiv id (the default id shape)."""
    return not isinstance(normalize_id(token), ToolResult)


def parse_optional_ids(
    args: Mapping[str, object],
    *,
    looks_like_id: Callable[[str], bool] = _is_paper_id,
) -> list[str] | ToolResult:
    """Parse and validate the ``ids`` argument, allowing its absence.

    Shared shape-checker so every id-taking Paper* tool accepts ``ids``
    identically. A bare string is coerced to a single-element list -- the
    one-paper case is common and the array wrapper is pure ceremony there.
    Absence yields ``[]`` (the caller decides whether that is allowed). Size
    is not pre-checked -- S2 rejects an oversized batch with its own error,
    which the request path surfaces.

    Args:
      args: Tool arguments.
      looks_like_id: Predicate deciding whether a comma/newline bundle should
        be split (every token must satisfy it). Defaults to the DOI/arXiv
        shape; PaperAuthor passes ``str.isdigit`` for opaque author ids.

    Returns:
      ids: Stripped, non-empty identifier strings (possibly ``[]`` when
        ``ids`` is absent), or a ``ToolResult`` error on non-list,
        non-string input.

    """
    raw_ids = args.get("ids")
    if raw_ids is None:
        return []
    # Normalize to a list, then expand every string ELEMENT through bundle
    # recovery. Models emit a multi-id ``ids`` as a string when the wire
    # coerces the union-typed (``["array","string"]``) schema -- and that
    # string can arrive either bare (``"a,b"``) OR wrapped in a one-element
    # list (``["a,b"]``, live 2026-06-19: PaperAuthor hit ``/author/a,b`` as a
    # single id). Recovering per-element handles both shapes; a genuine
    # multi-element list is untouched because ``_split_id_bundle`` returns a
    # non-bundle element unchanged.
    items = [raw_ids] if isinstance(raw_ids, str) else raw_ids
    if not isinstance(items, list):
        return ToolResult(
            call_id="",
            content="'ids' must be a list of strings or a single string.",
            is_error=True,
        )
    expanded: list[str] = []
    for item in cast(list[object], items):
        if not isinstance(item, str):
            # Symmetric with the bare-scalar reject above: a non-string element
            # is a shape error, not something to silently ``str()``-coerce into
            # a bogus id (``7`` -> ``"7"``).
            return ToolResult(
                call_id="",
                content="'ids' must be a list of strings or a single string.",
                is_error=True,
            )
        expanded.extend(_split_id_bundle(item, looks_like_id=looks_like_id))
    return [x.strip() for x in expanded if x.strip()]


def _split_id_bundle(
    raw: str, *, looks_like_id: Callable[[str], bool] = _is_paper_id
) -> list[str]:
    """Recover a list of ids from a single string argument.

    Handles the two shapes a model emits when it means a batch but the wire
    delivers a scalar string:

    1. A JSON-encoded array of strings (``'["a", "b"]'``) -> its elements.
    2. A comma- or newline-joined bundle (``'a, b'``) -> split, but ONLY when
       every token satisfies ``looks_like_id``. A lone DOI can legitimately
       contain a comma, so an ambiguous split that yields any non-id token is
       rejected and the original string is returned untouched (one id).

    ``looks_like_id`` is injected because the validity test is namespace
    specific: paper tools accept DOI/arXiv shapes, but PaperAuthor's opaque
    integer author ids would be rejected by the paper-id check, leaving their
    comma bundles unrecovered. Each tool passes the predicate for its ids.

    A plain single id (no separators) returns ``[raw]`` -- the common case,
    unchanged.
    """
    s = raw.strip()
    if s.startswith("[") and s.endswith("]"):
        try:
            parsed = json.loads(s)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            return [str(x) for x in cast(list[object], parsed)]
    tokens = [t.strip() for t in re.split(r"[,\n]+", s) if t.strip()]
    if len(tokens) > 1 and all(looks_like_id(t) for t in tokens):
        return tokens
    return [raw]


def validate_limit(limit: int | None) -> int | None | ToolResult:
    """Reject a non-positive ``limit``; pass ``None`` and positives through.

    A ``limit`` below 1 is meaningless and, untrimmed, silently yields no
    results (the pagination loop never runs). This is input validation, not
    a provider-policy cap: no upper bound is imposed -- pagination satisfies
    any positive ``limit`` the backend can serve.

    Args:
      limit: Caller-supplied limit (``None`` when omitted).

    Returns:
      limit: The value unchanged, or a ``ToolResult`` error when ``< 1``.

    """
    if limit is not None and limit < 1:
        return ToolResult(
            call_id="", content="'limit' must be a positive integer.", is_error=True
        )
    return limit


def validate_abstract_chars(cap: int | None) -> int | None | ToolResult:
    """Reject a non-positive ``abstract_chars``; pass ``None`` and positives.

    The schema declares ``minimum: 1``, but the directive layer does not
    enforce minimums, so a ``0`` would otherwise reach :func:`_trim_abstract`
    and be silently read as "no truncation" -- contradicting the schema and
    diverging from :func:`validate_limit`. One validator per positive-int tool
    arg keeps the contract uniform.

    Args:
      cap: Caller-supplied abstract cap (``None`` when omitted).

    Returns:
      cap: The value unchanged, or a ``ToolResult`` error when ``< 1``.

    """
    if cap is not None and cap < 1:
        return ToolResult(
            call_id="",
            content="'abstract_chars' must be a positive integer.",
            is_error=True,
        )
    return cap


def resolve_id_args(
    args: Mapping[str, object],
    *,
    looks_like_id: Callable[[str], bool] = _is_paper_id,
) -> list[str] | ToolResult:
    """Resolve a required, non-empty ``ids`` list.

    Wraps :func:`parse_optional_ids` for tools where ``ids`` is mandatory.

    Args:
      args: Tool arguments. ``ids`` is a list of one or more identifiers.
      looks_like_id: Bundle-split predicate forwarded to
        :func:`parse_optional_ids`.

    Returns:
      ids: Stripped, non-empty identifier strings, or a ``ToolResult``
        error on missing, malformed, empty, or over-length input.

    """
    if "ids" not in args:
        return ToolResult(call_id="", content="'ids' is required.", is_error=True)
    ids = parse_optional_ids(args, looks_like_id=looks_like_id)
    if isinstance(ids, ToolResult):
        return ids
    if not ids:
        return ToolResult(call_id="", content="'ids' is empty.", is_error=True)
    return ids


async def s2_get(path: str, params: dict[str, str | int]) -> MutableJSON | ToolResult:
    """GET an S2 Graph API path, rate-gated, with key injection and backoff.

    Args:
      path: API path relative to ``S2_BASE`` (e.g. ``/paper/search``).
      params: Query parameters.

    Returns:
      data: Parsed JSON response, or a ``ToolResult`` error on
        404 / exhausted-429 / other HTTP failures.

    """
    result = await _s2_request(
        lambda: fetch(
            url=f"{S2_BASE}{path}",
            params=params,
            headers=_s2_headers(),
            timeout_sec=S2_TIMEOUT,
        ),
        what=path,
    )
    # GET metadata/search endpoints never return a top-level array.
    assert not isinstance(result, list), f"unexpected array from GET {path}"
    return result


async def s2_batch(
    ids: list[str],
    fields: str,
    *,
    endpoint: Literal["paper", "author"] = "paper",
) -> list[MutableJSON | None] | ToolResult:
    """Fetch metadata for many ids in one batched request.

    Uses S2's ``POST /{endpoint}/batch`` -- a single rate-gated call returns
    all requested records, far cheaper than one :func:`s2_get` per id against
    the 1 req/sec budget. The result list is positionally aligned with
    ``ids``; an entry is ``None`` when S2 could not resolve that id. The batch
    size is not pre-checked: S2 rejects an oversized batch with its own error.

    Args:
      ids: S2 wire ids. For ``endpoint="paper"`` these are paper ids
        (e.g. ``DOI:10.x/y``, ``ARXIV:1706.03762``); for
        ``endpoint="author"`` they are opaque author ids.
      fields: Comma-separated S2 field selector (e.g. ``"title,authors"``).
      endpoint: Which batch endpoint to hit -- ``"paper"`` or ``"author"``.

    Returns:
      records: One entry per input id, in order; ``None`` for unresolved
        ids. A ``ToolResult`` on an HTTP failure (incl. S2's size rejection).

    """
    if not ids:
        return []
    result = await _s2_request(
        lambda: fetch(
            url=f"{S2_BASE}/{endpoint}/batch",
            method="POST",
            params={"fields": fields},
            json={"ids": ids},
            headers=_s2_headers(),
            timeout_sec=S2_TIMEOUT,
        ),
        what=f"/{endpoint}/batch",
    )
    if isinstance(result, ToolResult):
        return result
    # S2 returns a JSON array aligned with the request, ``null`` per miss.
    return [cast(MutableJSON, p) if isinstance(p, dict) else None for p in result]


async def s2_batch_blocks(
    ids: list[str],
    *,
    fields: str,
    endpoint: Literal["paper", "author"],
    to_block: Callable[[MutableJSON], str],
    cache: MutableMapping[tuple[object, ...], str],
    cache_tag: tuple[object, ...],
    labels: list[str] | None = None,
) -> ToolResult:
    """Batch-fetch ``ids`` and render one block per id, cached when complete.

    The single batch path shared by PaperDetails and PaperAuthor metadata
    lookups. Each tool injects its wire ids, field set, endpoint, and a
    ``to_block`` renderer; the fetch, the per-id miss handling, and the
    process cache live here exactly once.

    Misses render as ``"<label>: not found"``. ``labels`` defaults to ``ids``,
    but a tool whose ``ids`` are internal wire forms (PaperDetails resolves
    ``10.x/y`` to ``DOI:10.x/y``) passes the user-facing spellings so the miss
    message never leaks wire syntax. The result is cached only when every id
    resolved: a transient miss -- a just-published paper, S2 indexing lag --
    must not pin a ``not found`` for the process lifetime, since the cache is
    capacity-bounded with no staleness window.

    Args:
      ids: S2 wire ids (paper or author, matching ``endpoint``).
      fields: Comma-separated S2 field selector.
      endpoint: Which batch endpoint to hit.
      to_block: Render one resolved record into its text block.
      cache: The calling tool's process cache.
      cache_tag: Key prefix distinguishing this batch shape (e.g. abstract
        cap) from other entries in the same cache.
      labels: Per-id display labels for miss lines; defaults to ``ids``.

    Returns:
      result: Blocks in input order, or an error ``ToolResult``.

    """
    assert ids, "s2_batch_blocks requires a non-empty id list"
    miss_labels = labels if labels is not None else ids
    key = (*cache_tag, tuple(ids))
    cached = cache.get(key)
    if cached is not None:
        return ToolResult(call_id="", content=cached)
    records = await s2_batch(ids, fields, endpoint=endpoint)
    if isinstance(records, ToolResult):
        return records
    blocks = [
        to_block(data) if data is not None else f"{label}: not found"
        for label, data in zip(miss_labels, records, strict=True)
    ]
    content = "\n\n".join(blocks)
    if all(r is not None for r in records):
        cache[key] = content
    return ToolResult(call_id="", content=content)


@dataclass(frozen=True, slots=True, kw_only=True)
class Page:
    """A paginated, filtered slice of an S2 list endpoint.

    Attributes:
      entries: Entries passing the filter, trimmed to the caller's ``limit``
        when one was given.
      complete: True when no matches remain unseen -- S2's cursor was walked
        to exhaustion. False when the caller's ``limit`` cut the walk short,
        or S2's hard offset ceiling did. So ``not complete`` means "more
        matches may exist; raise ``limit`` (up to the ceiling) to get them".

    """

    entries: list[MutableJSON]
    complete: bool


# Rows requested per page: large to gather a given number of matches in the
# fewest rate-gated calls (a latency choice). S2 caps an over-large request
# and signals its own depth ceiling with an HTTP error, which the walk below
# treats as "no more to serve" -- so we don't mirror S2's exact limits here.
_S2_PAGE_ROWS = 1000


async def _s2_attempt(do_fetch: Callable[[], bytes]) -> bytes | FetchError:
    """Run one gated S2 request with 429 backoff; return bytes or the error.

    The single source of truth for the rate gate + exponential-backoff retry
    that S2 requires. Callers decide how to render the outcome (``s2_get`` ->
    ``ToolResult``; ``_s2_get_page`` -> HTTP status), so the policy lives in
    exactly one place.

    Args:
      do_fetch: Thunk performing the blocking HTTP call, returning bytes.

    Returns:
      result: Response bytes, or the terminal :class:`FetchError` (its
        ``status`` and ``body`` intact for the caller to render).

    """
    for attempt in range(_S2_MAX_RETRIES + 1):
        await _s2_gate().acquire_async()
        try:
            return await asyncio.to_thread(do_fetch)
        except FetchError as e:
            if e.status == 429 and attempt < _S2_MAX_RETRIES:
                await asyncio.sleep(_S2_BACKOFF_BASE * 2**attempt)
                continue
            return e
        except (TimeoutError, OSError) as e:
            # fetch re-raises these on socket timeout / connection failure
            # (fetch.py exhausts its own retries first). Funnel them into the
            # same FetchError-rendered path as any HTTP error -- status 0 marks
            # "no response" -- so a timeout surfaces as a structured ToolResult
            # rather than a bare exception. This matters more since S2_TIMEOUT
            # was lowered to 10s: the condition now fires far more often.
            return FetchError(url="", status=0, headers={}, body=str(e).encode())
    raise AssertionError("_s2_attempt retry loop exited without returning")


def _s2_error_result(e: FetchError, what: str) -> ToolResult:
    """Render an S2 ``FetchError`` as a consistent ``ToolResult`` message.

    Shared so every S2 surface (``s2_get``, ``s2_batch``, ``s2_paginate``)
    reports the same wording for the same status.

    Args:
      e: The terminal fetch error.
      what: Short request label for the message.

    Returns:
      result: An error ``ToolResult``.

    """
    if e.status == 0:
        return ToolResult(
            call_id="",
            content=(
                f"Semantic Scholar request failed for {what} "
                "(timeout or connection error). Retry shortly."
            ),
            is_error=True,
        )
    if e.status == 404:
        return ToolResult(call_id="", content=f"Not found: {what}", is_error=True)
    if e.status == 429:
        return ToolResult(
            call_id="",
            content="Semantic Scholar rate limit hit. Retry shortly.",
            is_error=True,
        )
    body = e.body[:200].decode(errors="replace")
    return ToolResult(
        call_id="",
        content=f"Semantic Scholar HTTP {e.status} for {what}: {body}",
        is_error=True,
    )


async def _s2_get_page(
    path: str, params: dict[str, str | int]
) -> MutableJSON | FetchError:
    """Fetch one paginated S2 page; return its JSON or the error.

    Unlike :func:`s2_get`, which flattens every failure into an opaque
    ``ToolResult``, this returns the :class:`FetchError` so the caller can
    tell S2's depth-ceiling 400 apart from a transient 429/5xx and keep the
    response body. Malformed JSON surfaces as a synthetic ``FetchError`` with
    ``status=0``.

    Args:
      path: List endpoint path.
      params: Query params including ``offset``/``limit``.

    Returns:
      result: Parsed page object, or the :class:`FetchError`.

    """
    raw = await _s2_attempt(
        lambda: fetch(
            url=f"{S2_BASE}{path}",
            params=params,
            headers=_s2_headers(),
            timeout_sec=S2_TIMEOUT,
        )
    )
    if isinstance(raw, FetchError):
        return raw
    try:
        return cast(MutableJSON, json.loads(raw))
    except json.JSONDecodeError as e:
        return FetchError(url=path, status=0, headers={}, body=str(e).encode())


async def s2_paginate(
    path: str,
    params: dict[str, str | int],
    *,
    limit: int | None,
    keep: Callable[[MutableJSON], bool] = lambda _e: True,
) -> Page | ToolResult:
    """Walk an S2 ``offset``/``next`` cursor, collecting filtered entries.

    Requests a large page each iteration to minimize the number of rate-gated
    calls, and follows the cursor until one of:
      - ``limit`` post-filter matches are gathered (``complete=False``);
      - S2 exhausts the cursor (``complete=True``);
      - S2 refuses a deeper page (its depth ceiling) after we already have
        matches (``complete=False`` -- more may exist but S2 will not serve
        it). We rely on S2 to signal this rather than mirroring its limits.

    A client-side ``keep`` filter therefore never silently drops matches that
    lie beyond the first page; ``complete`` truthfully reports exhaustion.
    With ``limit=None`` a single page is fetched and ``complete`` reflects
    whether S2's cursor was already exhausted.

    Args:
      path: List endpoint path (e.g. ``/author/{id}/papers``).
      params: Query params (``fields`` etc.); ``offset``/``limit`` here are
        managed internally and overwrite any in ``params``.
      limit: Post-filter entries the caller wants, or ``None`` for one page.
      keep: Predicate selecting entries to retain. Defaults to keep-all.

    Returns:
      page: A :class:`Page` (entries + completeness), or a ``ToolResult``
        error from any underlying request.

    """
    kept: list[MutableJSON] = []
    offset = 0
    while True:
        page_params: dict[str, str | int] = {
            **params,
            "offset": offset,
            "limit": _S2_PAGE_ROWS,
        }
        data = await _s2_get_page(path, page_params)
        if isinstance(data, FetchError):
            # S2 answers a too-deep page with 400 ``offset + limit < 10000``
            # -- the only 400 a cursor walk can provoke, since it controls
            # only offset/limit -- so treat 400 (with results in hand) as the
            # depth ceiling (stop, more may exist). Any other error surfaces
            # with the shared, consistent message.
            if data.status == 400 and kept:
                return Page(entries=_cap(kept, limit), complete=False)
            return _s2_error_result(data, _paginate_label(path))
        rows = cast(list[MutableJSON], data.get("data") or [])
        kept.extend(e for e in rows if keep(e))
        nxt = data.get("next")
        enough = limit is not None and len(kept) >= limit
        exhausted = not isinstance(nxt, int) or not rows
        if limit is None or enough or exhausted:
            return Page(entries=_cap(kept, limit), complete=exhausted)
        # Not exhausted => ``nxt`` is an int offset. Guard against a cursor
        # that fails to advance (a server regression) looping forever.
        assert isinstance(nxt, int)
        if nxt <= offset:
            return Page(entries=_cap(kept, limit), complete=False)
        offset = nxt


def _cap(entries: list[MutableJSON], limit: int | None) -> list[MutableJSON]:
    """Trim ``entries`` to ``limit`` (no-op when ``limit`` is ``None``)."""
    return entries if limit is None else entries[:limit]


def _paginate_label(path: str) -> str:
    """Human label for a paginated path's error message.

    A 404 on a per-entity listing -- ``/paper/<id>/references``,
    ``/paper/<id>/citations``, or ``/author/<id>/papers`` -- means the ENTITY
    is unindexed, so render ``"<listing> for <id>"`` rather than the literal
    ``"paginating /paper/<id>/references"``, which reads as if pagination
    itself were missing. Other paths fall back to the literal form.
    """
    parts = path.strip("/").split("/")
    listings = {
        "paper": ("citations", "references"),
        "author": ("papers",),
    }
    if len(parts) == 3 and parts[2] in listings.get(parts[0], ()):
        return f"{parts[2]} for {parts[1]}"
    return f"paginating {path}"


async def _s2_request(
    do_fetch: Callable[[], bytes],
    *,
    what: str,
) -> MutableJSON | list[object] | ToolResult:
    """Run one gated, backed-off S2 request; parse JSON or return an error.

    Shared core of :func:`s2_get` and :func:`s2_batch`: acquires the
    cross-process rate gate so the 1 req/sec key is honored fleet-wide, and
    retries 429s with exponential backoff (which S2 requires).

    Args:
      do_fetch: Thunk performing the blocking HTTP call, returning bytes.
      what: Short request label used in error messages.

    Returns:
      data: Parsed JSON (object or array), or a ``ToolResult`` error on
        404 / exhausted-429 / other HTTP failures.

    """
    raw = await _s2_attempt(do_fetch)
    if isinstance(raw, FetchError):
        return _s2_error_result(raw, what)
    try:
        return cast("MutableJSON | list[object]", json.loads(raw))
    except json.JSONDecodeError as e:
        return ToolResult(
            call_id="",
            content=f"Semantic Scholar returned invalid JSON for {what}: {e}",
            is_error=True,
        )


@functools.cache
def _s2_gate() -> TokenBucketRateLimiter:
    """Return the process-wide S2 rate gate, constructed on first use.

    S2's authenticated tier is one request/second, cumulative across every
    endpoint and every holder of the key. Multiple sagent processes share one
    key, so the gate serializes across processes (not just coroutines) via a
    lockfile. ``S2_MIN_INTERVAL`` (env, default 1.0s) sets the spacing.

    Built lazily (not at import) so importing this module has no filesystem
    side effect -- tests stay hermetic -- and so an env override set before
    first use takes effect. The wall-clock source is mandatory with
    ``FileStore``: the persisted timestamp is compared across processes,
    which share no monotonic epoch.

    Returns:
      gate: The shared cross-process token-bucket limiter.

    """
    return TokenBucketRateLimiter(
        max_calls=1,
        per_seconds=float(os.environ.get("S2_MIN_INTERVAL", "1.0")),
        clock=SystemClock(source=time.time),
        store=FileStore(Path.home() / ".sagent" / "s2_ratelimit.lock"),
    )


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
