"""Tests for sagent.lib.web.paper.providers.searxng (search + record mapping)."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import patch

import pytest

from sagent.lib.web.paper.errors import BackendError
from sagent.lib.web.paper.providers import searxng
from sagent.lib.web.search import PaperResult, SearchError


def _result(
    *,
    url: str = "https://example.com",
    title: str = "T",
    snippet: str = "",
    authors: tuple[str, ...] = (),
    journal: str = "",
    doi: str = "",
    pdf_url: str = "",
    published: datetime | None = None,
    citations: int | None = None,
) -> PaperResult:
    return PaperResult(
        url=url,
        title=title,
        snippet=snippet,
        authors=authors,
        journal=journal,
        doi=doi,
        pdf_url=pdf_url,
        published=published,
        citations=citations,
    )


_TARGET = "sagent.lib.web.paper.providers.searxng.searxng"


class TestSearch:
    def test_happy_path_maps_and_totals(self) -> None:
        hits = [
            _result(url="https://arxiv.org/abs/1706.03762", title="A"),
            _result(title="B"),
        ]
        with patch(_TARGET, return_value=hits):
            records, total = searxng.search(
                "q",
                limit=None,
                year_from=None,
                year_to=None,
                open_access_only=False,
            )
        assert total == 2
        assert [r.title for r in records] == ["A", "B"]
        assert records[0].arxiv_id == "1706.03762"

    def test_default_fetch_when_limit_none(self) -> None:
        with patch(_TARGET, return_value=[]) as mock:
            searxng.search(
                "q", limit=None, year_from=None, year_to=None, open_access_only=False
            )
        assert mock.call_args.kwargs["num_results"] == 20

    def test_limit_forwarded_as_num_results(self) -> None:
        with patch(_TARGET, return_value=[]) as mock:
            searxng.search(
                "q", limit=5, year_from=None, year_to=None, open_access_only=False
            )
        assert mock.call_args.kwargs["num_results"] == 5

    def test_year_filter_keeps_in_range(self) -> None:
        hits = [
            _result(title="old", published=datetime(2000, 1, 1)),  # noqa: DTZ001 -- year-only fixture
            _result(title="mid", published=datetime(2015, 1, 1)),  # noqa: DTZ001 -- year-only fixture
            _result(title="new", published=datetime(2025, 1, 1)),  # noqa: DTZ001 -- year-only fixture
        ]
        with patch(_TARGET, return_value=hits):
            records, total = searxng.search(
                "q",
                limit=None,
                year_from=2010,
                year_to=2020,
                open_access_only=False,
            )
        assert total == 1
        assert records[0].title == "mid"

    def test_open_access_only_keeps_pdf(self) -> None:
        hits = [
            _result(title="closed"),
            _result(title="open", pdf_url="https://x/pdf"),
        ]
        with patch(_TARGET, return_value=hits):
            records, total = searxng.search(
                "q",
                limit=None,
                year_from=None,
                year_to=None,
                open_access_only=True,
            )
        assert total == 1
        assert records[0].title == "open"

    def test_limit_caps_post_filter(self) -> None:
        hits = [_result(title=f"h{i}") for i in range(5)]
        with patch(_TARGET, return_value=hits):
            records, total = searxng.search(
                "q", limit=2, year_from=None, year_to=None, open_access_only=False
            )
        assert total == 2
        assert [r.title for r in records] == ["h0", "h1"]

    def test_search_error_raises_backend_error(self) -> None:
        with (
            patch(_TARGET, side_effect=SearchError("boom")),
            pytest.raises(BackendError, match="SearXNG"),
        ):
            searxng.search(
                "q", limit=None, year_from=None, year_to=None, open_access_only=False
            )

    def test_runtime_error_raises_backend_error(self) -> None:
        with (
            patch(_TARGET, side_effect=RuntimeError("boom")),
            pytest.raises(BackendError, match="SearXNG"),
        ):
            searxng.search(
                "q", limit=None, year_from=None, year_to=None, open_access_only=False
            )


class TestToRecord:
    def test_arxiv_id_from_url(self) -> None:
        rec = searxng._to_record(_result(url="https://arxiv.org/abs/1706.03762"))
        assert rec.arxiv_id == "1706.03762"

    def test_arxiv_id_none_for_non_arxiv_url(self) -> None:
        rec = searxng._to_record(_result(url="https://example.com/paper"))
        assert rec.arxiv_id is None

    def test_doi_kept_when_normalizes(self) -> None:
        rec = searxng._to_record(_result(doi="10.1234/x"))
        assert rec.doi == "10.1234/x"

    def test_doi_dropped_when_garbage(self) -> None:
        rec = searxng._to_record(_result(doi="garbage"))
        assert rec.doi is None

    def test_year_from_published(self) -> None:
        rec = searxng._to_record(_result(published=datetime(2019, 6, 1)))  # noqa: DTZ001 -- year-only fixture
        assert rec.year == 2019

    def test_year_none_when_no_published(self) -> None:
        rec = searxng._to_record(_result(published=None))
        assert rec.year is None

    def test_scalar_field_mapping(self) -> None:
        rec = searxng._to_record(
            _result(
                title="Title",
                journal="Nature",
                snippet="abstract",
                authors=("A", "B"),
                citations=42,
            )
        )
        assert rec.venue == "Nature"
        assert rec.abstract == "abstract"
        assert rec.authors == ("A", "B")
        assert rec.citation_count == 42
        assert rec.sources == ("searxng",)

    def test_empty_fields_become_none(self) -> None:
        rec = searxng._to_record(_result(title="", journal="", snippet=""))
        assert rec.title == "(untitled)"
        assert rec.venue is None
        assert rec.abstract is None


class TestYearInRange:
    def test_none_year_false(self) -> None:
        assert not searxng._year_in_range(None, 2000, 2020)

    def test_below_lo_false(self) -> None:
        assert not searxng._year_in_range(1999, 2000, 2020)

    def test_above_hi_false(self) -> None:
        assert not searxng._year_in_range(2021, 2000, 2020)

    def test_inside_true(self) -> None:
        assert searxng._year_in_range(2010, 2000, 2020)

    def test_no_bounds_true(self) -> None:
        assert searxng._year_in_range(2010, None, None)


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
