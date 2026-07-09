"""Tests for sagent.lib.web.paper.providers.openalex (client, filter, mapping)."""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import json

import pytest

from sagent.lib.custom_json import MutableJSON
from sagent.lib.web.fetch import FetchError
from sagent.lib.web.paper.errors import BackendError, NotFoundError, RateLimitError
from sagent.lib.web.paper.providers import openalex


@pytest.fixture(autouse=True)
def mock_limiter() -> Iterator[MagicMock]:
    """Inject a mock shared gate so the client never waits on real time."""
    limiter = MagicMock()
    with patch(
        "sagent.lib.web.paper.providers.openalex.cross_process_limiter",
        return_value=limiter,
    ):
        yield limiter


def _fetch_returning(payload: object) -> MagicMock:
    return MagicMock(return_value=json.dumps(payload).encode())


def _search(
    query: str = "attention",
    *,
    limit: int | None = None,
    year_from: int | None = None,
    year_to: int | None = None,
    open_access_only: bool = False,
) -> MagicMock:
    """Run ``search`` with a stub fetch and return that fetch mock for asserts."""
    fetch = _fetch_returning({"meta": {"count": 0}, "results": []})
    with patch("sagent.lib.web.paper.providers.openalex.fetch", fetch):
        openalex.search(
            query,
            limit=limit,
            year_from=year_from,
            year_to=year_to,
            open_access_only=open_access_only,
        )
    return fetch


class TestSearch:
    def test_happy_path_maps_records_and_total(self) -> None:
        work: MutableJSON = {"title": "Attention", "publication_year": 2017}
        payload = {"meta": {"count": 42}, "results": [work, {"title": "B"}]}
        with patch(
            "sagent.lib.web.paper.providers.openalex.fetch", _fetch_returning(payload)
        ):
            records, total = openalex.search(
                "attention",
                limit=None,
                year_from=None,
                year_to=None,
                open_access_only=False,
            )
        assert total == 42
        assert [r.title for r in records] == ["Attention", "B"]
        assert records[0].year == 2017

    def test_query_sanitizes_comma_and_pipe(self) -> None:
        fetch = _search("deep, learning | attention")
        flt = fetch.call_args.kwargs["params"]["filter"]
        assert "title_and_abstract.search:deep  learning   attention" in flt
        assert "," not in flt.split("title_and_abstract.search:")[1]
        assert "|" not in flt

    def test_limit_caps_at_per_page_max(self) -> None:
        fetch = _search(limit=500)
        assert fetch.call_args.kwargs["params"]["per-page"] == 200

    def test_limit_below_max_passthrough(self) -> None:
        fetch = _search(limit=10)
        assert fetch.call_args.kwargs["params"]["per-page"] == 10

    def test_limit_none_requests_full_page(self) -> None:
        # With no limit the walker fetches one full page (the ceiling), not a
        # bare default page -- so ``per-page`` is present and equals the max.
        fetch = _search(limit=None)
        assert fetch.call_args.kwargs["params"]["per-page"] == 200

    def test_filter_year_bounds_and_open_access(self) -> None:
        fetch = _search(year_from=2020, year_to=2023, open_access_only=True)
        flt = fetch.call_args.kwargs["params"]["filter"]
        assert "from_publication_date:2020-01-01" in flt
        assert "to_publication_date:2023-12-31" in flt
        assert "open_access.is_oa:true" in flt

    def test_api_key_present_when_env_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENALEX_API_KEY", "secret")
        fetch = _search()
        assert fetch.call_args.kwargs["params"]["api_key"] == "secret"

    def test_api_key_absent_when_env_unset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("OPENALEX_API_KEY", raising=False)
        fetch = _search()
        assert "api_key" not in fetch.call_args.kwargs["params"]


class TestHeaders:
    def test_mailto_in_user_agent_when_email_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENALEX_EMAIL", "me@example.com")
        headers = openalex._headers()
        assert headers["User-Agent"] == "loop-paper (mailto:me@example.com)"
        assert headers["Accept"] == "application/json"

    def test_plain_user_agent_when_email_unset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("OPENALEX_EMAIL", raising=False)
        assert openalex._headers()["User-Agent"] == "loop-paper"


class TestRequestErrors:
    def test_429_raises_rate_limit_mentions_budget(self) -> None:
        err = FetchError("u", 429, {}, b"slow down")
        with (
            patch("sagent.lib.web.paper.providers.openalex.fetch", side_effect=err),
            pytest.raises(RateLimitError) as ei,
        ):
            openalex.search(
                "x", limit=None, year_from=None, year_to=None, open_access_only=False
            )
        assert "daily credit budget" in str(ei.value)

    def test_non_429_raises_backend_error_with_status(self) -> None:
        err = FetchError("u", 500, {}, b"boom")
        with (
            patch("sagent.lib.web.paper.providers.openalex.fetch", side_effect=err),
            pytest.raises(BackendError) as ei,
        ):
            openalex.search(
                "x", limit=None, year_from=None, year_to=None, open_access_only=False
            )
        assert ei.value.status == 500

    def test_timeout_raises_backend_error_status_zero(self) -> None:
        with (
            patch(
                "sagent.lib.web.paper.providers.openalex.fetch",
                side_effect=TimeoutError(),
            ),
            pytest.raises(BackendError) as ei,
        ):
            openalex.search(
                "x", limit=None, year_from=None, year_to=None, open_access_only=False
            )
        assert ei.value.status == 0

    def test_oserror_raises_backend_error_status_zero(self) -> None:
        with (
            patch(
                "sagent.lib.web.paper.providers.openalex.fetch",
                side_effect=OSError("conn reset"),
            ),
            pytest.raises(BackendError) as ei,
        ):
            openalex.search(
                "x", limit=None, year_from=None, year_to=None, open_access_only=False
            )
        assert ei.value.status == 0

    def test_invalid_json_raises_backend_error(self) -> None:
        with (
            patch(
                "sagent.lib.web.paper.providers.openalex.fetch",
                MagicMock(return_value=b"not json"),
            ),
            pytest.raises(BackendError) as ei,
        ):
            openalex.search(
                "x", limit=None, year_from=None, year_to=None, open_access_only=False
            )
        assert "invalid JSON" in str(ei.value)


class TestFilter:
    def test_all_none_returns_none(self) -> None:
        assert (
            openalex._filter(year_from=None, year_to=None, open_access_only=False)
            is None
        )

    def test_composes_parts(self) -> None:
        flt = openalex._filter(year_from=2020, year_to=2021, open_access_only=True)
        assert flt == (
            "from_publication_date:2020-01-01,"
            "to_publication_date:2021-12-31,"
            "open_access.is_oa:true"
        )


class TestReconstructAbstract:
    def test_rebuilds_in_position_order(self) -> None:
        inverted = {"learning": [1], "Deep": [0], "rocks": [2]}
        assert openalex._reconstruct_abstract(inverted) == "Deep learning rocks"

    def test_none_returns_none(self) -> None:
        assert openalex._reconstruct_abstract(None) is None

    def test_empty_returns_none(self) -> None:
        assert openalex._reconstruct_abstract({}) is None

    def test_all_empty_positions_returns_none(self) -> None:
        assert openalex._reconstruct_abstract({"word": []}) is None


class TestWorkToRecord:
    def test_full_work(self) -> None:
        work: MutableJSON = {
            "title": "Attention Is All You Need",
            "authorships": [
                {"author": {"display_name": "Ashish Vaswani"}},
                {"author": {"display_name": "Noam Shazeer"}},
                {"author": {}},
            ],
            "publication_year": 2017,
            "primary_location": {"source": {"display_name": "NeurIPS"}},
            "doi": "https://doi.org/10.5555/attention",
            "ids": {"arxiv": "https://arxiv.org/abs/1706.03762"},
            "abstract_inverted_index": {"a": [0], "b": [1]},
            "cited_by_count": 100,
            "referenced_works_count": 20,
            "open_access": {"oa_url": "http://x/pdf"},
        }
        rec = openalex._work_to_record(work)
        assert rec.title == "Attention Is All You Need"
        assert rec.authors == ("Ashish Vaswani", "Noam Shazeer")
        assert rec.year == 2017
        assert rec.venue == "NeurIPS"
        assert rec.doi == "10.5555/attention"
        assert rec.arxiv_id == "1706.03762"
        assert rec.abstract == "a b"
        assert rec.citation_count == 100
        assert rec.reference_count == 20
        assert rec.open_access_pdf == "http://x/pdf"
        assert rec.sources == ("openalex",)

    def test_sparse_work(self) -> None:
        work: MutableJSON = {}
        rec = openalex._work_to_record(work)
        assert rec.title == "(untitled)"
        assert rec.authors == ()
        assert rec.year is None
        assert rec.doi is None
        assert rec.arxiv_id is None
        assert rec.abstract is None
        assert rec.venue is None
        assert rec.open_access_pdf is None

    def test_display_name_fallback_for_title(self) -> None:
        work: MutableJSON = {"display_name": "Fallback Title"}
        assert openalex._work_to_record(work).title == "Fallback Title"

    def test_doi_dx_prefix_stripped(self) -> None:
        work: MutableJSON = {"doi": "http://dx.doi.org/10.1/y"}
        assert openalex._work_to_record(work).doi == "10.1/y"

    def test_arxiv_no_match_leaves_none(self) -> None:
        work: MutableJSON = {"ids": {"arxiv": "!!!"}}
        assert openalex._work_to_record(work).arxiv_id is None


class TestReferences:
    def test_resolves_then_batches(self) -> None:
        resolve: MutableJSON = {
            "results": [
                {
                    "id": "https://openalex.org/W1",
                    "referenced_works": [
                        "https://openalex.org/W10",
                        "https://openalex.org/W11",
                    ],
                }
            ]
        }
        batch: MutableJSON = {
            "meta": {"count": 2},
            "results": [{"title": "ref-a"}, {"title": "ref-b"}],
        }
        fetch = MagicMock(
            side_effect=[json.dumps(resolve).encode(), json.dumps(batch).encode()]
        )
        with patch("sagent.lib.web.paper.providers.openalex.fetch", fetch):
            records, complete = openalex.references("doi", "10.1/x", limit=None)
        assert [r.title for r in records] == ["ref-a", "ref-b"]
        assert complete
        # Second call resolves the referenced ids via the ``openalex:`` filter.
        assert "openalex:W10|W11" in fetch.call_args.kwargs["params"]["filter"]

    def test_limit_truncates_and_marks_incomplete(self) -> None:
        resolve: MutableJSON = {
            "results": [
                {
                    "id": "https://openalex.org/W1",
                    "referenced_works": [
                        "https://openalex.org/W10",
                        "https://openalex.org/W11",
                        "https://openalex.org/W12",
                    ],
                }
            ]
        }
        batch: MutableJSON = {"meta": {"count": 1}, "results": [{"title": "ref-a"}]}
        fetch = MagicMock(
            side_effect=[json.dumps(resolve).encode(), json.dumps(batch).encode()]
        )
        with patch("sagent.lib.web.paper.providers.openalex.fetch", fetch):
            _, complete = openalex.references("doi", "10.1/x", limit=1)
        assert not complete  # 3 referenced, only 1 requested
        assert fetch.call_args.kwargs["params"]["filter"] == "openalex:W10"

    def test_arxiv_seed_rejected(self) -> None:
        with pytest.raises(BackendError, match="DOIs only"):
            openalex.references("arxiv", "1706.03762", limit=None)

    def test_unknown_doi_not_found(self) -> None:
        empty: MutableJSON = {"results": []}
        with (
            patch(
                "sagent.lib.web.paper.providers.openalex.fetch",
                _fetch_returning(empty),
            ),
            pytest.raises(NotFoundError),
        ):
            openalex.references("doi", "10.1/missing", limit=None)


class TestCitations:
    def test_cites_filter_and_total(self) -> None:
        resolve: MutableJSON = {"results": [{"id": "https://openalex.org/W1"}]}
        citing: MutableJSON = {
            "meta": {"count": 500},
            "results": [{"title": "citer"}],
        }
        fetch = MagicMock(
            side_effect=[json.dumps(resolve).encode(), json.dumps(citing).encode()]
        )
        with patch("sagent.lib.web.paper.providers.openalex.fetch", fetch):
            records, total, complete = openalex.citations("doi", "10.1/x", limit=1)
        assert [r.title for r in records] == ["citer"]
        assert total == 500
        assert not complete  # 1 of 500 -> more remain
        assert fetch.call_args.kwargs["params"]["filter"] == "cites:W1"

    def test_year_from_added_to_filter(self) -> None:
        resolve: MutableJSON = {"results": [{"id": "https://openalex.org/W1"}]}
        citing: MutableJSON = {"meta": {"count": 0}, "results": []}
        fetch = MagicMock(
            side_effect=[json.dumps(resolve).encode(), json.dumps(citing).encode()]
        )
        with patch("sagent.lib.web.paper.providers.openalex.fetch", fetch):
            openalex.citations("doi", "10.1/x", limit=None, year_from=2020)
        flt = fetch.call_args.kwargs["params"]["filter"]
        assert "cites:W1" in flt
        assert "from_publication_date:2020-01-01" in flt

    def test_arxiv_seed_rejected(self) -> None:
        with pytest.raises(BackendError, match="DOIs only"):
            openalex.citations("arxiv", "1706.03762", limit=None)

    def test_limit_over_page_max_paginates(self) -> None:
        # BUG 1: a limit above the 200 per-page ceiling must paginate, never
        # request per-page>200 (which OpenAlex 400s). Two 200-work pages satisfy
        # a limit of 250; no single request may set per-page above 200.
        resolve: MutableJSON = {"results": [{"id": "https://openalex.org/W1"}]}
        page1: MutableJSON = {
            "meta": {"count": 500},
            "results": [{"title": f"c{i}"} for i in range(200)],
        }
        page2: MutableJSON = {
            "meta": {"count": 500},
            "results": [{"title": f"c{200 + i}"} for i in range(200)],
        }
        fetch = MagicMock(
            side_effect=[
                json.dumps(resolve).encode(),
                json.dumps(page1).encode(),
                json.dumps(page2).encode(),
            ]
        )
        with patch("sagent.lib.web.paper.providers.openalex.fetch", fetch):
            records, total, complete = openalex.citations("doi", "10.1/x", limit=250)
        assert len(records) == 250
        assert total == 500
        assert not complete
        for call in fetch.call_args_list:
            per_page = call.kwargs["params"].get("per-page")
            assert per_page is None or per_page <= 200

    def test_limit_none_reports_honest_completeness(self) -> None:
        # BUG 2: with no limit, a single default page of 25 against a total of
        # 500 is NOT complete. ``complete`` must mean "cursor exhausted", never
        # be True merely because ``limit is None``.
        resolve: MutableJSON = {"results": [{"id": "https://openalex.org/W1"}]}
        # One page shorter than ``count`` -> cursor not exhausted.
        citing: MutableJSON = {
            "meta": {"count": 500},
            "results": [{"title": f"c{i}"} for i in range(200)],
        }
        fetch = MagicMock(
            side_effect=[json.dumps(resolve).encode(), json.dumps(citing).encode()]
        )
        with patch("sagent.lib.web.paper.providers.openalex.fetch", fetch):
            _, total, complete = openalex.citations("doi", "10.1/x", limit=None)
        assert total == 500
        assert not complete

    def test_exact_full_page_is_complete(self) -> None:
        # BUG D: a full final page whose length equals the requested size but
        # exhausts ``count`` must report complete=True. The len>=size heuristic
        # alone lies here; exhaustion must consult ``meta.count``.
        resolve: MutableJSON = {"results": [{"id": "https://openalex.org/W1"}]}
        citing: MutableJSON = {
            "meta": {"count": 200},
            "results": [{"title": f"c{i}"} for i in range(200)],
        }
        fetch = MagicMock(
            side_effect=[json.dumps(resolve).encode(), json.dumps(citing).encode()]
        )
        with patch("sagent.lib.web.paper.providers.openalex.fetch", fetch):
            records, total, complete = openalex.citations("doi", "10.1/x", limit=None)
        assert total == 200
        assert len(records) == 200
        assert complete  # cursor exhausted: 200 of 200 returned

    def test_multi_page_exact_total_is_complete(self) -> None:
        # Two full 200-pages reaching count=400 exactly: walking to limit=400
        # exhausts the cursor and reports complete=True (no lying full-page).
        resolve: MutableJSON = {"results": [{"id": "https://openalex.org/W1"}]}
        page1: MutableJSON = {
            "meta": {"count": 400},
            "results": [{"title": f"c{i}"} for i in range(200)],
        }
        page2: MutableJSON = {
            "meta": {"count": 400},
            "results": [{"title": f"c{200 + i}"} for i in range(200)],
        }
        fetch = MagicMock(
            side_effect=[
                json.dumps(resolve).encode(),
                json.dumps(page1).encode(),
                json.dumps(page2).encode(),
            ]
        )
        with patch("sagent.lib.web.paper.providers.openalex.fetch", fetch):
            records, total, complete = openalex.citations("doi", "10.1/x", limit=400)
        assert total == 400
        assert len(records) == 400
        assert complete  # 400 of 400 -> cursor exhausted


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
