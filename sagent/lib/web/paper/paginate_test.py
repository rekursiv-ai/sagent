"""Tests for sagent.lib.web.paper.paginate (the generic cursor walker)."""

from __future__ import annotations

from typing import cast
from unittest.mock import MagicMock

import pytest

from sagent.lib.custom_json import MutableJSON
from sagent.lib.web.paper.errors import BackendError
from sagent.lib.web.paper.paginate import Cursor, paginate


def _rows(body: MutableJSON) -> list[MutableJSON]:
    """Read the ``data`` row list from a fixture page body."""
    return cast(list[MutableJSON], body.get("data") or [])


def _offset_cursor(
    pages: list[list[MutableJSON]],
    *,
    page_size_max: int = 100,
    fetch: MagicMock | None = None,
) -> Cursor:
    """A 0-based offset cursor over pre-baked ``pages`` (short page = last)."""

    def do_fetch(offset: int, size: int) -> MutableJSON:
        del size
        idx = offset  # pages are indexed by page number for simplicity
        rows = pages[idx] if idx < len(pages) else []
        return cast(MutableJSON, {"data": rows})

    def advance(body: MutableJSON, position: int, size: int) -> int | None:
        rows = body.get("data") or []
        assert isinstance(rows, list)
        return position + 1 if len(rows) >= size else None

    return Cursor(
        fetch=fetch or do_fetch,
        rows=_rows,
        advance=advance,
        page_size_max=page_size_max,
    )


class TestPaginate:
    def test_single_page_under_limit_is_complete(self) -> None:
        cursor = _offset_cursor([[{"n": 1}, {"n": 2}]], page_size_max=10)
        page = paginate(cursor, limit=None)
        assert [r["n"] for r in page.entries] == [1, 2]
        assert page.complete  # short page (2 < 10) -> exhausted

    def test_limit_clamps_page_size_never_exceeds_max(self) -> None:
        sizes: list[int] = []

        def do_fetch(offset: int, size: int) -> MutableJSON:
            sizes.append(size)
            return cast(MutableJSON, {"data": [{"n": offset}] * size})  # always full

        cursor = _offset_cursor(
            [], page_size_max=200, fetch=MagicMock(side_effect=do_fetch)
        )
        paginate(cursor, limit=450)
        assert all(s <= 200 for s in sizes)
        assert sizes[0] == 200  # limit>max -> request the ceiling

    def test_walks_multiple_pages_to_limit_incomplete(self) -> None:
        # Three full 200-pages available; limit 450 spans them and stays
        # incomplete (a full final page means more may remain).
        pages: list[list[MutableJSON]] = [
            [{"n": i} for i in range(200)] for _ in range(3)
        ]
        cursor = _offset_cursor(pages, page_size_max=200)
        page = paginate(cursor, limit=450)
        assert len(page.entries) == 450
        assert not page.complete

    def test_full_page_exactly_at_limit_is_incomplete(self) -> None:
        # limit == a full page: enough is reached, but the cursor was not
        # exhausted, so complete must be False (the limit-clamp bug guard).
        pages: list[list[MutableJSON]] = [[{"n": i} for i in range(200)], [{"n": 200}]]
        cursor = _offset_cursor(pages, page_size_max=200)
        page = paginate(cursor, limit=200)
        assert len(page.entries) == 200
        assert not page.complete

    def test_keep_filter_does_not_understate_completeness(self) -> None:
        # A keep-filter dropping rows must not make a full page look short.
        pages: list[list[MutableJSON]] = [[{"n": i} for i in range(200)], [{"n": 200}]]
        cursor = _offset_cursor(pages, page_size_max=200)
        page = paginate(cursor, limit=None, keep=lambda r: cast(int, r["n"]) % 2 == 0)
        # Only one page fetched (limit=None), all-even kept, but complete
        # reflects the cursor (full page -> more), not the filtered count.
        assert not page.complete

    def test_depth_ceiling_stops_incomplete_with_results(self) -> None:
        def do_fetch(offset: int, size: int) -> MutableJSON:
            if offset == 0:
                return {"data": [{"n": i} for i in range(size)]}
            raise BackendError("too deep", status=400)

        cursor = Cursor(
            fetch=MagicMock(side_effect=do_fetch),
            rows=_rows,
            advance=lambda _b, pos, _s: pos + 1,  # always claims more
            page_size_max=200,
            is_depth_ceiling=lambda e: e.status == 400,
        )
        page = paginate(cursor, limit=1000)
        assert len(page.entries) == 200
        assert not page.complete  # ceiling hit -> more may exist

    def test_depth_ceiling_without_results_reraises(self) -> None:
        def do_fetch(_offset: int, _size: int) -> MutableJSON:
            raise BackendError("bad", status=400)

        cursor = Cursor(
            fetch=MagicMock(side_effect=do_fetch),
            rows=_rows,
            advance=lambda _b, pos, _s: pos + 1,
            page_size_max=200,
            is_depth_ceiling=lambda e: e.status == 400,
        )
        with pytest.raises(BackendError):
            paginate(cursor, limit=1000)

    def test_non_depth_ceiling_error_reraises(self) -> None:
        cursor = Cursor(
            fetch=MagicMock(side_effect=BackendError("boom", status=500)),
            rows=_rows,
            advance=lambda _b, _pos, _s: None,
            page_size_max=200,
        )
        with pytest.raises(BackendError):
            paginate(cursor, limit=None)

    def test_non_advancing_cursor_terminates(self) -> None:
        # advance returns a position <= current -> stop, do not loop forever.
        cursor = Cursor(
            fetch=MagicMock(return_value={"data": [{"n": 0}] * 200}),
            rows=_rows,
            advance=lambda _b, _pos, _s: 0,  # never advances
            page_size_max=200,
        )
        page = paginate(cursor, limit=1000)
        assert len(page.entries) == 200
        assert not page.complete

    def test_start_position_respected(self) -> None:
        seen: list[int] = []

        def do_fetch(position: int, _size: int) -> MutableJSON:
            seen.append(position)
            return {"data": []}

        cursor = Cursor(
            fetch=MagicMock(side_effect=do_fetch),
            rows=_rows,
            advance=lambda _b, _pos, _s: None,
            page_size_max=50,
            start=1,  # 1-based page APIs (OpenAlex)
        )
        paginate(cursor, limit=None)
        assert seen == [1]


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
