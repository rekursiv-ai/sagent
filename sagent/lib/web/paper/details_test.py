"""Tests for sagent.lib.web.paper.details (metadata + citation graph)."""

from __future__ import annotations

from collections.abc import Callable
from unittest.mock import patch

import pytest

from sagent.lib.custom_json import MutableJSON
from sagent.lib.web.paper.custom_types import PaperRecord
from sagent.lib.web.paper.details import (
    citations,
    metadata,
    metadata_batch,
    references,
)
from sagent.lib.web.paper.errors import PaperError
from sagent.lib.web.paper.paginate import Page
from sagent.lib.web.paper.providers import (
    openalex,
    s2,
)


class TestCitations:
    def test_year_filter_keeps_recent(self) -> None:
        entries: list[MutableJSON] = [
            {"isInfluential": True, "citingPaper": {"title": "new", "year": 2024}},
            {"isInfluential": False, "citingPaper": {"title": "old", "year": 2010}},
        ]

        def fake(
            path: str,
            params: dict[str, str | int],
            *,
            limit: int | None,
            keep: Callable[[MutableJSON], bool],
        ) -> Page:
            del path, params, limit
            return Page(entries=[e for e in entries if keep(e)], complete=True)

        with patch.object(s2, "paginate", side_effect=fake):
            listing = citations("doi", "10.1/x", limit=None, year_from=2020)
        assert [r.title for r in listing.records] == ["new"]

    def test_influential_only_filters(self) -> None:
        entries: list[MutableJSON] = [
            {"isInfluential": True, "citingPaper": {"title": "keep"}},
            {"isInfluential": False, "citingPaper": {"title": "drop"}},
        ]

        def fake(
            path: str,
            params: dict[str, str | int],
            *,
            limit: int | None,
            keep: Callable[[MutableJSON], bool],
        ) -> Page:
            del path, params, limit
            return Page(entries=[e for e in entries if keep(e)], complete=False)

        with patch.object(s2, "paginate", side_effect=fake):
            listing = citations("doi", "10.1/x", limit=5, influential_only=True)
        assert [r.title for r in listing.records] == ["keep"]
        assert not listing.complete  # cursor not exhausted -> more may exist


class TestMetadata:
    def test_single(self) -> None:
        payload: MutableJSON = {"title": "T", "externalIds": {"DOI": "10.1/x"}}
        with patch.object(s2, "get", return_value=payload) as get:
            rec = metadata("doi", "10.1/x")
        assert rec.title == "T"
        assert get.call_args.args[0] == "/paper/DOI:10.1/x"

    def test_batch_aligns_and_nulls(self) -> None:
        with patch.object(s2, "batch", return_value=[{"title": "A"}, None]):
            recs = metadata_batch(["DOI:1", "DOI:2"])
        assert recs[0] is not None
        assert recs[0].title == "A"
        assert recs[1] is None


class TestReferences:
    def test_maps_edges(self) -> None:
        page = Page(
            entries=[{"citedPaper": {"title": "cited"}, "isInfluential": True}],
            complete=True,
        )
        with patch.object(s2, "paginate", return_value=page) as paginate:
            listing = references("arxiv", "1706.03762", limit=None)
        assert [r.title for r in listing.records] == ["cited"]
        assert listing.records[0].is_influential is True
        # References endpoint path + citedPaper.* fields.
        assert paginate.call_args.args[0] == "/paper/ARXIV:1706.03762/references"

    def test_skips_empty_inner_edge(self) -> None:
        # An edge row with no inner paper object is skipped, not mapped to a stub.
        page = Page(
            entries=[{"citedPaper": {}}, {"citedPaper": {"title": "real"}}],
            complete=True,
        )
        with patch.object(s2, "paginate", return_value=page):
            listing = references("doi", "10.1/x", limit=None)
        assert [r.title for r in listing.records] == ["real"]


class TestOpenAlexGraphSource:
    def test_references_dispatches_to_openalex(self) -> None:
        recs = [PaperRecord(title="ref", sources=("openalex",))]
        with patch.object(openalex, "references", return_value=(recs, True)) as oa_refs:
            listing = references("doi", "10.1/x", limit=None, source="openalex")
        assert [r.title for r in listing.records] == ["ref"]
        assert listing.complete
        assert oa_refs.call_args.args == ("doi", "10.1/x")

    def test_citations_dispatches_to_openalex(self) -> None:
        recs = [PaperRecord(title="citer", sources=("openalex",))]
        with patch.object(
            openalex, "citations", return_value=(recs, 500, False)
        ) as oa_cites:
            listing = citations("doi", "10.1/x", limit=1, source="openalex")
        assert [r.title for r in listing.records] == ["citer"]
        assert not listing.complete  # total 500 > 1 returned
        assert oa_cites.call_args.kwargs["year_from"] is None

    def test_openalex_citations_complete_when_all_returned(self) -> None:
        recs = [PaperRecord(title="c", sources=("openalex",))]
        with patch.object(openalex, "citations", return_value=(recs, 1, True)):
            listing = citations("doi", "10.1/x", limit=None, source="openalex")
        assert listing.complete

    def test_influential_only_rejected_for_openalex(self) -> None:
        with pytest.raises(PaperError, match="S2-only"):
            citations(
                "doi", "10.1/x", limit=None, source="openalex", influential_only=True
            )


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
