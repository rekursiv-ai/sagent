"""Backend-agnostic cursor walker for paged list endpoints.

Every scholarly backend paginates differently on the wire -- Semantic Scholar
walks an integer ``offset``/``next`` cursor and signals its depth ceiling with a
400; OpenAlex walks a 1-based ``page`` bounded by ``meta.count`` -- but the
*contract* is identical: fetch pages of at most the endpoint's ceiling, keep the
rows a predicate accepts, stop at the caller's ``limit`` or at cursor
exhaustion, and report honestly whether the cursor was exhausted.

That contract lives here exactly once. A backend supplies a :class:`Cursor`
describing only its wire mechanics; it never writes a paging loop and never sets
a page size (this module is the sole writer of it, clamped to
:attr:`Cursor.page_size_max`). A new backend therefore cannot reintroduce the
page-size-overflow or lying-``complete`` bugs that independent per-backend loops
bred -- it has no loop to get wrong.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from sagent.lib.custom_json import MutableJSON
from sagent.lib.web.paper.errors import BackendError


__all__ = [
    "Cursor",
    "Page",
    "paginate",
]


def _keep_all(_entry: MutableJSON) -> bool:
    return True


@dataclass(frozen=True, slots=True, kw_only=True)
class Page:
    """A paginated, filtered slice of a list endpoint.

    Attributes:
      entries: Rows passing the ``keep`` predicate, trimmed to ``limit``.
      complete: True if and only if the cursor was walked to exhaustion. False
        when ``limit`` or a backend depth ceiling cut the walk short -- i.e.
        more matches may exist. Never derived from ``limit is None``.

    """

    entries: list[MutableJSON]
    complete: bool


@dataclass(frozen=True, slots=True, kw_only=True)
class Cursor:
    """A backend's paging mechanics -- everything :func:`paginate` needs.

    Attributes:
      fetch: Fetch one page given a start position and page size, returning the
        raw JSON body. The start is whatever :attr:`advance` yields (an integer
        offset, a 1-based page number, ...); the first call receives
        :attr:`start`.
      rows: Extract the row list from a page body.
      advance: Given the page body, the current start, and the page size that
        was requested, return the next start, or ``None`` when the cursor is
        exhausted. This is the SOLE source of the completeness signal. The
        requested size is passed so "this page was short, hence last" is
        distinguishable from a full page with more behind it.
      page_size_max: The endpoint's page-size ceiling; :func:`paginate` never
        requests more than this per call.
      start: The first-page start position (``0`` for offset APIs, ``1`` for
        page-number APIs).
      is_depth_ceiling: Whether a :class:`BackendError` is the backend's
        "cannot page deeper" signal (e.g. S2's 400) rather than a real failure.
        When it fires with rows already in hand, the walk stops incomplete.

    """

    fetch: Callable[[int, int], MutableJSON]
    rows: Callable[[MutableJSON], list[MutableJSON]]
    advance: Callable[[MutableJSON, int, int], int | None]
    page_size_max: int
    start: int = 0
    is_depth_ceiling: Callable[[BackendError], bool] = field(default=lambda _e: False)


def paginate(
    cursor: Cursor,
    *,
    limit: int | None,
    keep: Callable[[MutableJSON], bool] = _keep_all,
) -> Page:
    """Walk ``cursor`` collecting rows ``keep`` accepts, up to ``limit``.

    Requests pages of at most :attr:`Cursor.page_size_max` (never the raw
    ``limit``), following the cursor until ``limit`` kept rows are gathered, the
    cursor is exhausted, or the backend refuses a deeper page. ``complete``
    reflects cursor exhaustion alone, so a client-side ``keep`` filter never
    silently understates how much remains.

    Args:
      cursor: The backend's paging mechanics.
      limit: Kept rows the caller wants, or ``None`` for a single page.
      keep: Predicate selecting rows to retain. Defaults to keep-all.

    Returns:
      page: A :class:`Page` (kept rows trimmed to ``limit``, plus completeness).

    Raises:
      BackendError: Any error the cursor does not classify as its depth ceiling.

    """
    page_size = _page_size(limit, cursor.page_size_max)
    kept: list[MutableJSON] = []
    position = cursor.start
    while True:
        try:
            body = cursor.fetch(position, page_size)
        except BackendError as e:
            if cursor.is_depth_ceiling(e) and kept:
                return Page(entries=_cap(kept, limit), complete=False)
            raise
        kept.extend(row for row in cursor.rows(body) if keep(row))
        nxt = cursor.advance(body, position, page_size)
        exhausted = nxt is None
        enough = limit is not None and len(kept) >= limit
        if limit is None or enough or exhausted:
            return Page(entries=_cap(kept, limit), complete=exhausted)
        # ``advance`` returned a real next position; guard a non-advancing
        # cursor (a server regression) against looping forever.
        if nxt <= position:
            return Page(entries=_cap(kept, limit), complete=False)
        position = nxt


def _page_size(limit: int | None, page_size_max: int) -> int:
    """Page size to request: the whole limit when small, else the ceiling."""
    if limit is None:
        return page_size_max
    return min(limit, page_size_max)


def _cap(entries: list[MutableJSON], limit: int | None) -> list[MutableJSON]:
    """Trim ``entries`` to ``limit`` (no-op when ``limit`` is ``None``)."""
    return entries if limit is None else entries[:limit]
