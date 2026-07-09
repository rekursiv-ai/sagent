"""Reciprocal-rank fusion of results from two backends.

Merges Semantic Scholar and OpenAlex hit lists into one ranked list: each
backend contributes ``weight / (offset + rank)`` to a paper's score, summed
across backends, so cross-backend agreement outranks either backend's lone top
hit. Duplicates (same DOI or normalized title) are merged for fields and the
``sources`` tag.
"""

from __future__ import annotations

import re

from sagent.lib.web.paper.custom_types import PaperRecord


__all__ = ["fuse"]

_WORD_PUNCT_RE = re.compile(r"[^\w\s]+")
_WS_RE = re.compile(r"\s+")


def _normalize_title(title: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace - for dedup."""
    lowered = title.lower()
    nopunct = _WORD_PUNCT_RE.sub(" ", lowered)
    return _WS_RE.sub(" ", nopunct).strip()


def _dedup_key(rec: PaperRecord) -> str:
    """Prefer DOI; fall back to normalized title for DOI-less records."""
    if rec.doi:
        return f"doi:{rec.doi.lower()}"
    return f"title:{_normalize_title(rec.title)}"


def _merge(first: PaperRecord, second: PaperRecord) -> PaperRecord:
    """Combine two records of the same paper - prefer non-null from first."""
    return PaperRecord(
        title=first.title or second.title,
        authors=first.authors or second.authors,
        year=first.year if first.year is not None else second.year,
        venue=first.venue or second.venue,
        doi=first.doi or second.doi,
        arxiv_id=first.arxiv_id or second.arxiv_id,
        abstract=first.abstract or second.abstract,
        citation_count=(
            first.citation_count
            if first.citation_count is not None
            else second.citation_count
        ),
        reference_count=(
            first.reference_count
            if first.reference_count is not None
            else second.reference_count
        ),
        open_access_pdf=first.open_access_pdf or second.open_access_pdf,
        sources=tuple(dict.fromkeys((*first.sources, *second.sources))),
    )


def fuse(s2_hits: list[PaperRecord], oa_hits: list[PaperRecord]) -> list[PaperRecord]:
    """Reciprocal-rank-fuse S2 and OpenAlex hits into one ranked list.

    A paper both backends rank well floats above either backend's lone top hit;
    an OpenAlex-only paper still scores by its single rank, so a throttled S2
    degrades to OpenAlex-ranked results rather than nothing.

    Args:
      s2_hits: Semantic Scholar results in rank order (best first).
      oa_hits: OpenAlex results in rank order (best first).

    Returns:
      fused: Papers ordered by descending fused score.

    References:
      https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf
        Cormack, Clarke, Büttcher. "Reciprocal Rank Fusion Outperforms
        Condorcet and Individual Rank Learning Methods." SIGIR 2009.

    """
    # RRF rank offset: ``weight / (offset + rank)`` adds ``offset`` phantom
    # slots before the real ranks (score half-life around ``rank == offset``).
    # The canonical 60 over-flattens with only two backends -- all of S2's top
    # ~27 would outrank an OpenAlex-only #1, burying the cross-pollinated hits
    # fusion exists to surface. At 10, a strong single-backend hit interleaves
    # into the other's top while respecting S2's lead.
    offset = 10.0
    # S2's relevance ranking is more precise than OpenAlex's broad text match,
    # so an S2 rank counts for more; an OpenAlex-only paper still scores.
    weights = ((s2_hits, 1.0), (oa_hits, 0.7))

    by_key: dict[str, PaperRecord] = {}
    score: dict[str, float] = {}
    for hits, weight in weights:
        for rank, rec in enumerate(hits, start=1):
            key = _dedup_key(rec)
            score[key] = score.get(key, 0.0) + weight / (offset + rank)
            by_key[key] = _merge(by_key[key], rec) if key in by_key else rec
    # Stable sort by descending score over insertion-ordered keys; the S2 loop
    # runs first so a coincidental score tie keeps S2's key first.
    ordered = sorted(by_key, key=lambda k: score[k], reverse=True)
    return [by_key[k] for k in ordered]
