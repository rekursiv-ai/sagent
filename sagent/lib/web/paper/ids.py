"""Paper identifier parsing and canonicalization.

Detect whether a user-supplied string is a DOI or an arXiv id, strip common
URL/scheme wrappers, and produce the canonical bare form plus the wire/slug
spellings the backends and the PDF cache need.
"""

from __future__ import annotations

import re

from sagent.lib.web.paper.custom_types import IdType
from sagent.lib.web.paper.errors import InvalidIdError


__all__ = [
    "ARXIV_URL_RE",
    "id_slug",
    "looks_like_paper_id",
    "normalize_id",
    "s2_wire_id",
]

# DOI shape: 10.<registrant>/<suffix>. Registrant is 4+ digits (ISO 26324).
# Suffix is opaque; may contain slashes, dots, colons, etc.
_DOI_RE = re.compile(r"^(10\.\d{4,})/(\S+)$")

# arXiv new-style id: NNNN.NNNNN with optional version (v1, v2, ...).
_ARXIV_NEW_RE = re.compile(r"^(\d{4}\.\d{4,5})(v\d+)?$")

# arXiv old-style id: <subject>/NNNNNNN (e.g. hep-th/9901001). Rare but
# S2 and arXiv both still honor these for papers pre-April-2007.
_ARXIV_OLD_RE = re.compile(r"^([a-z-]+(?:\.[A-Z]{2})?)/(\d{7})(v\d+)?$")

# arXiv id embedded in an abs/pdf URL, for backends (SearXNG, Google Scholar)
# that surface no structured arXiv id but link to arxiv.org. Distinct from the
# anchored id regexes above, which match a bare id, not a URL.
ARXIV_URL_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/([\w.-]+/\d+|\d{4}\.\d{4,5})")


def normalize_id(raw: str) -> tuple[IdType, str]:
    """Parse a user-supplied identifier into ``(kind, canonical)``.

    Accepts DOIs with or without the ``https://doi.org/`` / ``doi:`` prefix,
    arXiv ids with or without ``arXiv:`` / ``arxiv.org/abs/`` wrapping, bare
    new-style ids (``2106.15928``), and old-style ids (``hep-th/9901001``).

    Args:
      raw: User-supplied identifier string.

    Returns:
      kind: ``"doi"`` or ``"arxiv"``.
      canonical: Bare identifier with no prefix.

    Raises:
      InvalidIdError: When the shape matches neither DOI nor arXiv.

    """
    doi_prefixes = (
        "https://doi.org/",
        "http://doi.org/",
        "https://dx.doi.org/",
        "http://dx.doi.org/",
        "doi:",
    )
    arxiv_prefixes = (
        "https://arxiv.org/abs/",
        "http://arxiv.org/abs/",
        "https://arxiv.org/pdf/",
        "http://arxiv.org/pdf/",
        "arxiv:",
        "arxiv.org/abs/",
    )
    s = raw.strip()
    if not s:
        raise InvalidIdError("Empty identifier.")

    # Strip common URL / scheme wrappers. A matched prefix pins the family,
    # so a ``doi:`` value is never re-interpreted as arXiv (or vice versa).
    lower = s.lower()
    forced: IdType | None = None
    for prefix in doi_prefixes:
        if lower.startswith(prefix):
            s = s[len(prefix) :]
            lower = s.lower()
            forced = "doi"
            break
    if forced is None:
        for prefix in arxiv_prefixes:
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
    raise InvalidIdError(
        f"Unrecognized identifier shape: {raw!r}. "
        "Expected DOI (10.xxxx/yyy) or arXiv id (NNNN.NNNNN, "
        "arXiv:NNNN.NNNNN, or hep-th/NNNNNNN)."
    )


def looks_like_paper_id(token: str) -> bool:
    """Whether ``token`` parses as a DOI or arXiv id.

    Args:
      token: The candidate identifier string to test.

    """
    try:
        _ = normalize_id(token)
    except InvalidIdError:
        return False
    return True


def s2_wire_id(kind: IdType, canonical: str) -> str:
    """Build the prefixed form S2 accepts in a URL path: ``DOI:...``/``ARXIV:...``.

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
