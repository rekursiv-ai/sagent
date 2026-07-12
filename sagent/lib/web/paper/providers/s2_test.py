"""Tests for sagent.lib.web.paper.providers.s2 (client, backoff, record mapping)."""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import json

import pytest

from sagent.lib.custom_json import MutableJSON
from sagent.lib.web.errors import FetchError
from sagent.lib.web.fetch import FetchSession
from sagent.lib.web.paper.errors import BackendError, NotFoundError, RateLimitError
from sagent.lib.web.paper.providers import s2


@pytest.fixture(autouse=True)
def mock_limiter() -> Iterator[MagicMock]:
    """Inject a mock shared gate so the S2 client never waits on real time."""
    limiter = MagicMock()
    with patch(
        "sagent.lib.web.paper.providers.s2.cross_process_limiter", return_value=limiter
    ):
        yield limiter


def _fetch_returning(payload: object) -> MagicMock:
    return MagicMock(return_value=(json.dumps(payload).encode(), FetchSession()))


class TestGet:
    def test_parses_object(self) -> None:
        with patch(
            "sagent.lib.web.paper.providers.s2.fetch",
            _fetch_returning({"title": "X"}),
        ):
            assert s2.get("/paper/DOI:10.1/x", {"fields": "title"}) == {"title": "X"}

    def test_404_raises_not_found(self) -> None:
        err = FetchError("u", 404, {}, b"missing")
        with (
            patch("sagent.lib.web.paper.providers.s2.fetch", side_effect=err),
            pytest.raises(NotFoundError),
        ):
            s2.get("/paper/DOI:10.1/x", {})

    def test_bad_json_raises_backend_error(self) -> None:
        with (
            patch(
                "sagent.lib.web.paper.providers.s2.fetch",
                MagicMock(return_value=(b"not json", FetchSession())),
            ),
            pytest.raises(BackendError),
        ):
            s2.get("/paper/search", {})

    def test_timeout_raises_backend_error_status_zero(self) -> None:
        with (
            patch(
                "sagent.lib.web.paper.providers.s2.fetch", side_effect=TimeoutError()
            ),
            pytest.raises(BackendError) as ei,
        ):
            s2.get("/paper/search", {})
        assert ei.value.status == 0


class TestBackoff:
    def test_429_retries_then_raises_rate_limit(self, mock_limiter: MagicMock) -> None:
        # Every attempt 429s: after the retry budget it surfaces RateLimitError,
        # and each retry records a growing backoff into the shared cooldown.
        err = FetchError("u", 429, {}, b"slow down")
        with (
            patch("sagent.lib.web.paper.providers.s2.fetch", side_effect=err),
            pytest.raises(RateLimitError),
        ):
            s2.get("/paper/search", {})
        # Two retries -> two backoff triggers (1s, 2s); acquire once per attempt.
        assert mock_limiter.trigger_cooldown.call_count == 2
        assert [c.args[0] for c in mock_limiter.trigger_cooldown.call_args_list] == [
            1.0,
            2.0,
        ]
        assert mock_limiter.acquire.call_count == 3

    def test_429_then_success_recovers(self, mock_limiter: MagicMock) -> None:
        err = FetchError("u", 429, {}, b"slow")
        ok = (json.dumps({"title": "ok"}).encode(), FetchSession())
        with patch("sagent.lib.web.paper.providers.s2.fetch", side_effect=[err, ok]):
            assert s2.get("/paper/x", {}) == {"title": "ok"}
        assert mock_limiter.trigger_cooldown.call_count == 1


class TestBatch:
    def test_aligns_and_nulls_misses(self) -> None:
        payload = [{"title": "A"}, None, {"title": "C"}]
        with patch(
            "sagent.lib.web.paper.providers.s2.fetch", _fetch_returning(payload)
        ):
            out = s2.batch(["DOI:1", "DOI:2", "DOI:3"], "title")
        assert out == [{"title": "A"}, None, {"title": "C"}]

    def test_empty_ids_no_fetch(self) -> None:
        with patch("sagent.lib.web.paper.providers.s2.fetch") as mock:
            assert s2.batch([], "title") == []
        mock.assert_not_called()

    def test_non_array_response_raises(self) -> None:
        # S2's batch endpoint must return an array; an object is a contract break.
        with (
            patch(
                "sagent.lib.web.paper.providers.s2.fetch",
                _fetch_returning({"unexpected": "object"}),
            ),
            pytest.raises(BackendError, match="non-array"),
        ):
            s2.batch(["DOI:1"], "title")


class TestPaginate:
    def test_walks_cursor_to_limit(self) -> None:
        pages = [
            (
                json.dumps({"data": [{"year": 2020}] * 3, "next": 3}).encode(),
                FetchSession(),
            ),
            (
                json.dumps({"data": [{"year": 2021}] * 3, "next": 6}).encode(),
                FetchSession(),
            ),
        ]
        with patch("sagent.lib.web.paper.providers.s2.fetch", side_effect=pages):
            page = s2.paginate("/paper/x/citations", {"fields": "year"}, limit=5)
        assert len(page.entries) == 5
        assert not page.complete

    def test_exhaustion_marks_complete(self) -> None:
        one = (
            json.dumps({"data": [{"year": 2020}], "next": None}).encode(),
            FetchSession(),
        )
        with patch("sagent.lib.web.paper.providers.s2.fetch", return_value=one):
            page = s2.paginate("/paper/x/references", {}, limit=10)
        assert page.complete

    def test_depth_ceiling_400_stops_with_results(self) -> None:
        first = (
            json.dumps({"data": [{"year": 2020}] * 3, "next": 3}).encode(),
            FetchSession(),
        )
        ceiling = FetchError("u", 400, {}, b"offset + limit < 10000")
        with patch(
            "sagent.lib.web.paper.providers.s2.fetch", side_effect=[first, ceiling]
        ):
            page = s2.paginate("/paper/x/citations", {}, limit=100)
        assert len(page.entries) == 3
        assert not page.complete

    def test_400_with_no_results_reraises(self) -> None:
        # A 400 before ANY page succeeded is a real error, not the depth ceiling.
        err = FetchError("u", 400, {}, b"bad request")
        with (
            patch("sagent.lib.web.paper.providers.s2.fetch", side_effect=err),
            pytest.raises(BackendError),
        ):
            s2.paginate("/paper/x/citations", {}, limit=100)

    def test_non_advancing_cursor_terminates(self) -> None:
        # A server regression where ``next`` does not advance past ``offset``
        # must terminate (not loop forever) and report incomplete.
        page_json = (
            json.dumps({"data": [{"year": 2020}] * 3, "next": 0}).encode(),
            FetchSession(),
        )
        with patch("sagent.lib.web.paper.providers.s2.fetch", return_value=page_json):
            page = s2.paginate("/paper/x/citations", {}, limit=100)
        assert len(page.entries) == 3
        assert not page.complete

    def test_author_papers_builds_endpoint(self) -> None:
        one = (
            json.dumps({"data": [{"title": "P"}], "next": None}).encode(),
            FetchSession(),
        )
        with patch("sagent.lib.web.paper.providers.s2.fetch", return_value=one) as mock:
            page = s2.author_papers("42", limit=None)
        assert [e.get("title") for e in page.entries] == ["P"]
        assert mock.call_args.kwargs["url"].endswith("/author/42/papers")


class TestSearchPaginate:
    def test_walks_offset_to_total_and_reports_total(self) -> None:
        pages = [
            (
                json.dumps({"data": [{"title": "A"}] * 2, "total": 3}).encode(),
                FetchSession(),
            ),
            (
                json.dumps({"data": [{"title": "B"}], "total": 3}).encode(),
                FetchSession(),
            ),
        ]
        with patch("sagent.lib.web.paper.providers.s2.fetch", side_effect=pages):
            page, total = s2.search_paginate({"query": "x"}, limit=5)
        assert [e.get("title") for e in page.entries] == ["A", "A", "B"]
        assert total == 3
        assert page.complete

    def test_caps_page_size_at_search_ceiling(self) -> None:
        one = (
            json.dumps({"data": [{"title": "A"}], "total": 1}).encode(),
            FetchSession(),
        )
        with patch("sagent.lib.web.paper.providers.s2.fetch", return_value=one) as mock:
            page, total = s2.search_paginate({"query": "x"}, limit=None)
        assert total == 1
        assert page.entries == [{"title": "A"}]
        assert mock.call_args.kwargs["request"].params["limit"] == 100


class TestSearchTotal:
    def test_extracts_total(self) -> None:
        assert s2.search_total({"total": 42}) == 42

    def test_missing_total_defaults_zero(self) -> None:
        assert s2.search_total({}) == 0


class TestRecordMapping:
    def test_paper_record_from_full(self) -> None:
        data: MutableJSON = {
            "title": "Attention",
            "externalIds": {"DOI": "10.1/x", "ArXiv": "1706.03762"},
            "authors": [{"name": "A"}, {"name": "B"}],
            "year": 2017,
            "venue": "NIPS",
            "abstract": "text",
            "citationCount": 100,
            "referenceCount": 20,
            "openAccessPdf": {"url": "http://x/pdf"},
        }
        rec = s2.paper_record_from(data)
        assert rec.title == "Attention"
        assert rec.doi == "10.1/x"
        assert rec.arxiv_id == "1706.03762"
        assert rec.authors == ("A", "B")
        assert rec.year == 2017
        assert rec.open_access_pdf == "http://x/pdf"
        assert rec.sources == ("s2",)

    def test_author_record_dict_affiliations(self) -> None:
        data: MutableJSON = {
            "authorId": "42",
            "name": "Yoshua Bengio",
            "affiliations": [{"name": "MILA"}, "UdeM"],
            "hIndex": 200,
        }
        rec = s2.author_record_from(data)
        assert rec.author_id == "42"
        assert rec.affiliations == ("MILA", "UdeM")
        assert rec.h_index == 200


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
