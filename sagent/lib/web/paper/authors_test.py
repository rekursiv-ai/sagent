"""Tests for sagent.lib.web.paper.authors (author search / metadata / papers)."""

from __future__ import annotations

from collections.abc import Callable
from unittest.mock import patch

from sagent.lib.custom_json import MutableJSON
from sagent.lib.web.paper.authors import author_metadata, author_papers, search_authors
from sagent.lib.web.paper.custom_types import AuthorRecord
from sagent.lib.web.paper.paginate import Page
from sagent.lib.web.paper.providers import s2


class TestAuthors:
    def test_search_sorts_by_h_index_and_caps(self) -> None:
        payload: MutableJSON = {
            "total": 3,
            "data": [
                {"authorId": "1", "name": "Low", "hIndex": 5},
                {"authorId": "2", "name": "High", "hIndex": 90},
                {"authorId": "3", "name": "Mid", "hIndex": 40},
            ],
        }
        with patch.object(s2, "get", return_value=payload):
            result = search_authors("x", limit=2)
        assert [r.name for r in result.records] == [
            "High",
            "Mid",
        ]  # h-index desc, capped
        assert result.total == 3

    def test_author_metadata_batch(self) -> None:
        with patch.object(
            s2, "batch", return_value=[{"authorId": "1", "name": "A"}, None]
        ):
            recs = author_metadata(["1", "2"])
        assert isinstance(recs[0], AuthorRecord)
        assert recs[1] is None

    def test_author_papers_year_filter(self) -> None:
        entries: list[MutableJSON] = [
            {"title": "new", "year": 2024},
            {"title": "old", "year": 2000},
        ]

        def fake(
            author_id: str,
            *,
            limit: int | None,
            keep: Callable[[MutableJSON], bool],
        ) -> Page:
            del author_id, limit
            return Page(entries=[e for e in entries if keep(e)], complete=True)

        with patch.object(s2, "author_papers", side_effect=fake):
            listing = author_papers("1", limit=None, year_from=2020)
        assert [r.title for r in listing.records] == ["new"]

    def test_author_papers_no_filter_keeps_all(self) -> None:
        entries: list[MutableJSON] = [{"title": "a"}, {"title": "b"}]

        def fake(
            author_id: str,
            *,
            limit: int | None,
            keep: Callable[[MutableJSON], bool],
        ) -> Page:
            del author_id, limit
            # No bounds -> keep predicate returns True for every entry.
            return Page(entries=[e for e in entries if keep(e)], complete=True)

        with patch.object(s2, "author_papers", side_effect=fake):
            listing = author_papers("1", limit=None)
        assert len(listing.records) == 2

    def test_author_papers_year_to_and_undated(self) -> None:
        # year_to upper bound + an undated (non-int year) work excluded.
        entries: list[MutableJSON] = [
            {"title": "in", "year": 2018},
            {"title": "toolate", "year": 2024},
            {"title": "undated"},
        ]

        def fake(
            author_id: str,
            *,
            limit: int | None,
            keep: Callable[[MutableJSON], bool],
        ) -> Page:
            del author_id, limit
            return Page(entries=[e for e in entries if keep(e)], complete=True)

        with patch.object(s2, "author_papers", side_effect=fake):
            listing = author_papers("1", limit=None, year_to=2020)
        assert [r.title for r in listing.records] == ["in"]


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
