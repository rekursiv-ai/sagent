"""Tests for sagent.lib.web.paper.search (backend dispatch + fused degradation)."""

from __future__ import annotations

from typing import cast
from unittest.mock import patch

import pytest

from sagent.lib.custom_json import MutableJSON
from sagent.lib.web.paper import search as search_mod
from sagent.lib.web.paper.custom_types import PaperRecord
from sagent.lib.web.paper.errors import PaperError
from sagent.lib.web.paper.providers import (
    openalex,
    s2,
)
from sagent.lib.web.paper.search import Source, search


def _rec(title: str, source: str) -> PaperRecord:
    return PaperRecord(title=title, sources=(source,))


class TestSearchDispatch:
    def test_single_backend_openalex(self) -> None:
        with patch.object(
            openalex, "search", return_value=([_rec("a", "openalex")], 1)
        ):
            result = search("q", source="openalex")
        assert [r.title for r in result.records] == ["a"]
        assert result.total == 1
        assert result.complete

    def test_transport_forwarded_to_provider(self) -> None:
        with patch.object(openalex, "search", return_value=([], 0)) as provider:
            search("q", source="openalex", transport="stdlib")
        assert provider.call_args.kwargs["transport"] == "stdlib"

    def test_unknown_source_raises(self) -> None:
        with pytest.raises(PaperError, match="Unknown search source"):
            search("q", source=cast(Source, "bogus"))


class TestFusedSearch:
    def test_both_backends_fused(self) -> None:
        s2_payload: MutableJSON = {
            "total": 9,
            "data": [{"title": "s", "externalIds": {"DOI": "10.1/s"}}],
        }
        with (
            patch.object(s2, "get", return_value=s2_payload),
            patch.object(openalex, "search", return_value=([_rec("o", "openalex")], 5)),
        ):
            result = search("q")  # fused default
        assert {r.title for r in result.records} == {"s", "o"}
        assert result.total == 9
        assert result.complete

    def test_total_never_below_returned_records(self) -> None:
        # Disjoint hits: each backend reports total=1, fused set has 2 records.
        # total must not be less than what's returned (max(1,1,2) == 2).
        with (
            patch.object(search_mod, "_s2_search", return_value=([_rec("s", "s2")], 1)),
            patch.object(openalex, "search", return_value=([_rec("o", "openalex")], 1)),
        ):
            result = search("q")
        assert len(result.records) == 2
        assert result.total >= len(result.records)

    def test_one_backend_error_is_partial_not_fatal(self) -> None:
        with (
            patch.object(search_mod, "_s2_search", side_effect=PaperError("s2 down")),
            patch.object(openalex, "search", return_value=([_rec("o", "openalex")], 3)),
        ):
            result = search("q")
        assert [r.title for r in result.records] == ["o"]
        assert not result.complete  # partial -> caller may decline to cache

    def test_total_failure_raises(self) -> None:
        with (
            patch.object(search_mod, "_s2_search", side_effect=PaperError("s2 down")),
            patch.object(openalex, "search", side_effect=PaperError("oa down")),
            pytest.raises(PaperError),
        ):
            search("q")


class TestS2SearchParams:
    def test_year_and_open_access_params(self) -> None:
        captured: dict[str, str | int] = {}

        def fake_get(
            path: str,
            params: dict[str, str | int],
            *,
            transport: object = "auto",
        ) -> MutableJSON:
            del transport
            del path
            captured.update(params)
            return {"total": 0, "data": []}

        with patch.object(s2, "get", side_effect=fake_get):
            search(
                "q",
                source="s2",
                limit=7,
                year_from=2020,
                year_to=2022,
                open_access_only=True,
            )
        assert captured["year"] == "2020-2022"
        assert captured["limit"] == 7
        assert captured["openAccessPdf"] == ""

    def test_open_year_bounds(self) -> None:
        captured: dict[str, str | int] = {}

        def fake_get(
            path: str,
            params: dict[str, str | int],
            *,
            transport: object = "auto",
        ) -> MutableJSON:
            del transport
            del path
            captured.update(params)
            return {"total": 0, "data": []}

        with patch.object(s2, "get", side_effect=fake_get):
            search("q", source="s2", year_from=2020)
        assert captured["year"] == "2020-"

    def test_limit_clamped_to_search_ceiling(self) -> None:
        # BUG 3: S2's /paper/search caps ``limit`` at 100 and 400s above it. A
        # caller asking for 200 must never send limit>100 on a single request.
        seen: list[int] = []

        def fake_get(
            path: str,
            params: dict[str, str | int],
            *,
            transport: object = "auto",
        ) -> MutableJSON:
            del transport
            del path
            lim = params.get("limit")
            if isinstance(lim, int):
                seen.append(lim)
            return {"total": 0, "data": []}

        with patch.object(s2, "get", side_effect=fake_get):
            search("q", source="s2", limit=200)
        assert seen  # a limit was sent
        assert all(lim <= 100 for lim in seen)

    def test_limit_over_ceiling_paginates(self) -> None:
        # A limit above 100 walks multiple 100-row pages (offset advances) and
        # collects the full requested count -- not just one clamped page.
        def fake_get(
            path: str,
            params: dict[str, str | int],
            *,
            transport: object = "auto",
        ) -> MutableJSON:
            del transport
            del path
            offset = int(params.get("offset", 0))
            rows = [{"title": f"p{offset + i}"} for i in range(100)]
            return cast("MutableJSON", {"total": 250, "data": rows})

        with patch.object(s2, "get", side_effect=fake_get):
            result = search("q", source="s2", limit=200)
        assert len(result.records) == 200
        assert result.total == 250


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
