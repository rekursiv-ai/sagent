"""Tests for ``tools.paper_details``: the thin adapter over ``lib.web.paper``.

The adapter owns schema, arg validation, the process cache, and text
rendering; backend I/O / pagination / record shaping live in
:mod:`sagent.lib.web.paper` and are covered by its own tests. These tests mock
the library entry points where the adapter binds them
(``paper_details.metadata`` etc.) and assert only the adapter's behavior.
"""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import patch

import asyncio

import pytest

from sagent.lib.web.paper.custom_types import PaperRecord
from sagent.lib.web.paper.details import Listing
from sagent.lib.web.paper.errors import PaperError
from sagent.tools import paper_details
from sagent.tools.paper_details import (
    PaperDetails,
    _render_listing,
    _validate_details_args,
)


@pytest.fixture(autouse=True)
def _clear_cache() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction] -- autouse fixture
    """Clear the module result cache so a key is never pre-warmed by a sibling."""
    paper_details._cache.clear()
    yield
    paper_details._cache.clear()


# ---------------------------------------------------------------------------
# Tool metadata / summary
# ---------------------------------------------------------------------------


def test_paper_details_metadata() -> None:
    t = PaperDetails()
    assert t.name == "PaperDetails"
    assert t.tool_id == "application/x-tool-paperdetails"


def test_summary_default() -> None:
    t = PaperDetails()
    assert t.summary({"ids": ["10.1234/abc"]}) == "PaperDetails 10.1234/abc"


def test_summary_references() -> None:
    t = PaperDetails()
    out = t.summary({"ids": ["10.1234/abc"], "operation": "references"})
    assert out == "PaperDetails references 10.1234/abc"


def test_summary_citations() -> None:
    out = PaperDetails().summary({"ids": ["10.1234/abc"], "operation": "citations"})
    assert out == "PaperDetails citations 10.1234/abc"


def test_summary_missing_id() -> None:
    assert PaperDetails().summary({}) == "PaperDetails ?"


def test_prompt_empty() -> None:
    assert PaperDetails().prompt() == ""


def test_serialize_key_none() -> None:
    assert PaperDetails().serialize_key({"ids": ["10.1234/abc"]}) is None


# ---------------------------------------------------------------------------
# Arg validation (_validate_details_args)
# ---------------------------------------------------------------------------


def test_validate_details_args_empty_id() -> None:
    res = _validate_details_args("", "", influential_only=False, year_from=None)
    assert hasattr(res, "is_error")


def test_validate_details_args_bad_op() -> None:
    res = _validate_details_args(
        "10.1234/x",
        "frobnicate",
        influential_only=False,
        year_from=None,
    )
    assert hasattr(res, "is_error")


def test_validate_details_args_filter_without_citations() -> None:
    res = _validate_details_args(
        "10.1234/x",
        "references",
        influential_only=True,
        year_from=None,
    )
    assert hasattr(res, "is_error")


def test_validate_details_args_year_from_without_citations() -> None:
    res = _validate_details_args(
        "10.1234/x",
        "",
        influential_only=False,
        year_from=2020,
    )
    assert hasattr(res, "is_error")


def test_validate_details_args_unrecognized_id() -> None:
    res = _validate_details_args(
        "not-an-id", "", influential_only=False, year_from=None
    )
    assert hasattr(res, "is_error")


def test_validate_details_args_valid_metadata() -> None:
    res = _validate_details_args(
        "10.1234/x", "", influential_only=False, year_from=None
    )
    assert isinstance(res, tuple)
    op, kind, canonical = res
    assert op == ""
    assert kind == "doi"
    assert canonical == "10.1234/x"


def test_validate_details_args_valid_citations_with_filters() -> None:
    res = _validate_details_args(
        "10.1234/x",
        "citations",
        influential_only=True,
        year_from=2020,
    )
    assert isinstance(res, tuple)


# ---------------------------------------------------------------------------
# run(): arg-level rejections (no library call)
# ---------------------------------------------------------------------------


def test_run_missing_ids() -> None:
    result = asyncio.run(PaperDetails().run({}))
    assert result.is_error
    assert "'ids' is required" in result.content


def test_run_empty_ids_rejected() -> None:
    result = asyncio.run(PaperDetails().run({"ids": []}))
    assert result.is_error


def test_run_rejects_zero_limit() -> None:
    # LIM-001: limit=0 must error, not silently return no results.
    result = asyncio.run(
        PaperDetails().run(
            {"ids": ["10.1234/x"], "operation": "references", "limit": 0}
        )
    )
    assert result.is_error
    assert "limit" in result.content


def test_run_rejects_zero_abstract_chars() -> None:
    result = asyncio.run(
        PaperDetails().run({"ids": ["10.1234/x"], "abstract_chars": 0})
    )
    assert result.is_error
    assert "abstract_chars" in result.content


def test_run_invalid_id_returns_error() -> None:
    result = asyncio.run(PaperDetails().run({"ids": ["garbage"]}))
    assert result.is_error


def test_run_ids_with_operation_rejected() -> None:
    result = asyncio.run(
        PaperDetails().run(
            {"ids": ["10.1234/a", "10.1234/b"], "operation": "references"}
        ),
    )
    assert result.is_error
    assert "exactly one id" in result.content


# ---------------------------------------------------------------------------
# run(): metadata delegation to the library's metadata()
# ---------------------------------------------------------------------------


def test_run_metadata_success() -> None:
    rec = PaperRecord(title="T", year=2020, doi="10.1234/cached_meta", authors=("A",))
    with patch("sagent.tools.paper_details.metadata", return_value=rec) as m:
        result = asyncio.run(
            PaperDetails().run({"ids": ["10.1234/meta_ok"]}),
        )
    assert not result.is_error
    assert "title: T" in result.content
    assert m.call_args.args == ("doi", "10.1234/meta_ok")


def test_run_metadata_caches() -> None:
    rec = PaperRecord(title="C", doi="10.1234/unique_cache_doi_zz")
    with patch("sagent.tools.paper_details.metadata", return_value=rec) as m:
        _ = asyncio.run(PaperDetails().run({"ids": ["10.1234/unique_cache_doi_zz"]}))
        _ = asyncio.run(PaperDetails().run({"ids": ["10.1234/unique_cache_doi_zz"]}))
    assert m.call_count == 1


def test_run_metadata_error_mapped() -> None:
    with patch(
        "sagent.tools.paper_details.metadata",
        side_effect=PaperError("paper 10.1234/notfound_id not found"),
    ):
        result = asyncio.run(PaperDetails().run({"ids": ["10.1234/notfound_id"]}))
    assert result.is_error
    assert "not found" in result.content


# ---------------------------------------------------------------------------
# run(): references / citations delegation and rendering
# ---------------------------------------------------------------------------


def test_run_references() -> None:
    listing = Listing(
        records=[PaperRecord(title="Cited Paper", year=2019)], complete=True
    )
    with patch("sagent.tools.paper_details.references", return_value=listing) as m:
        result = asyncio.run(
            PaperDetails().run(
                {"ids": ["10.1234/ref_target"], "operation": "references"}
            ),
        )
    assert "Cited Paper" in result.content
    assert m.call_args.args == ("doi", "10.1234/ref_target")
    assert m.call_args.kwargs == {"limit": None, "source": "s2"}


def test_run_references_empty() -> None:
    with patch(
        "sagent.tools.paper_details.references",
        return_value=Listing(records=[], complete=True),
    ):
        result = asyncio.run(
            PaperDetails().run(
                {"ids": ["10.1234/empty_refs"], "operation": "references"}
            ),
        )
    assert result.content == "(no results)"


def test_run_citations_delegates_filters() -> None:
    listing = Listing(records=[PaperRecord(title="New", year=2022)], complete=True)
    with patch("sagent.tools.paper_details.citations", return_value=listing) as m:
        result = asyncio.run(
            PaperDetails().run(
                {
                    "ids": ["10.1234/cit_target"],
                    "operation": "citations",
                    "year_from": 2020,
                    "influential_only": True,
                    "limit": 5,
                }
            ),
        )
    assert "New" in result.content
    assert m.call_args.args == ("doi", "10.1234/cit_target")
    assert m.call_args.kwargs == {
        "limit": 5,
        "source": "s2",
        "influential_only": True,
        "year_from": 2020,
    }


def test_run_citations_openalex_source() -> None:
    listing = Listing(records=[PaperRecord(title="OA citer")], complete=True)
    with patch("sagent.tools.paper_details.citations", return_value=listing) as m:
        result = asyncio.run(
            PaperDetails().run(
                {
                    "ids": ["10.1234/oa_target"],
                    "operation": "citations",
                    "source": "openalex",
                    "limit": 3,
                }
            ),
        )
    assert "OA citer" in result.content
    assert m.call_args.kwargs["source"] == "openalex"


def test_run_references_openalex_source() -> None:
    listing = Listing(records=[PaperRecord(title="OA ref")], complete=True)
    with patch("sagent.tools.paper_details.references", return_value=listing) as m:
        result = asyncio.run(
            PaperDetails().run(
                {
                    "ids": ["10.1234/oa_ref_target"],
                    "operation": "references",
                    "source": "openalex",
                }
            ),
        )
    assert "OA ref" in result.content
    assert m.call_args.kwargs["source"] == "openalex"


def test_run_invalid_source_rejected() -> None:
    result = asyncio.run(
        PaperDetails().run(
            {"ids": ["10.1/x"], "operation": "citations", "source": "bogus"}
        )
    )
    assert result.is_error
    assert "Invalid source" in result.content


def test_run_citations_incomplete_notice() -> None:
    listing = Listing(records=[PaperRecord(title="Citing")], complete=False)
    with patch("sagent.tools.paper_details.citations", return_value=listing):
        result = asyncio.run(
            PaperDetails().run({"ids": ["10.1234/cit_more"], "operation": "citations"}),
        )
    assert "more matches exist; raise 'limit'" in result.content


# ---------------------------------------------------------------------------
# _render_listing (exact strings; bit-for-bit sensitive)
# ---------------------------------------------------------------------------


def test_render_listing_empty() -> None:
    assert _render_listing(Listing(records=[], complete=True), None) == "(no results)"


def test_render_listing_complete_no_notice() -> None:
    out = _render_listing(
        Listing(records=[PaperRecord(title="Solo")], complete=True), None
    )
    assert "Solo" in out
    assert "more matches" not in out


def test_render_listing_incomplete_notice() -> None:
    out = _render_listing(
        Listing(records=[PaperRecord(title="Solo")], complete=False), None
    )
    assert out.endswith("\n... (more matches exist; raise 'limit' to see them)")


# ---------------------------------------------------------------------------
# run(): metadata_batch delegation, miss line, cache semantics
# ---------------------------------------------------------------------------


def _batch_records() -> list[PaperRecord | None]:
    return [
        PaperRecord(title="First", doi="10.1234/a"),
        None,
        PaperRecord(title="Third", doi="10.1234/c"),
    ]


def test_run_ids_batches_one_call() -> None:
    with patch(
        "sagent.tools.paper_details.metadata_batch",
        return_value=_batch_records(),
    ) as m:
        result = asyncio.run(
            PaperDetails().run({"ids": ["10.1234/a", "10.1234/b", "10.1234/c"]}),
        )
    assert m.call_count == 1  # one batched call, not three
    assert m.call_args.args == (["DOI:10.1234/a", "DOI:10.1234/b", "DOI:10.1234/c"],)
    assert not result.is_error
    assert "title: First" in result.content
    # Miss label must echo the USER's id, not the internal S2 wire id.
    assert "10.1234/b: not found" in result.content
    assert "DOI:10.1234/b: not found" not in result.content
    assert "title: Third" in result.content


def test_run_ids_batch_error_mapped() -> None:
    with patch(
        "sagent.tools.paper_details.metadata_batch",
        side_effect=PaperError("batch too big"),
    ):
        result = asyncio.run(
            PaperDetails().run({"ids": ["10.1234/a", "10.1234/b"]}),
        )
    assert result.is_error
    assert "batch too big" in result.content


def test_run_ids_batch_caches_when_all_resolved() -> None:
    records: list[PaperRecord | None] = [
        PaperRecord(title="X", doi="10.1234/batch_cache_x"),
        PaperRecord(title="Y", doi="10.1234/batch_cache_y"),
        PaperRecord(title="Z", doi="10.1234/batch_cache_z"),
    ]
    with patch(
        "sagent.tools.paper_details.metadata_batch",
        return_value=records,
    ) as m:
        ids = [
            "10.1234/batch_cache_x",
            "10.1234/batch_cache_y",
            "10.1234/batch_cache_z",
        ]
        _ = asyncio.run(PaperDetails().run({"ids": ids}))
        _ = asyncio.run(PaperDetails().run({"ids": ids}))
    assert m.call_count == 1


def test_run_ids_batch_with_miss_not_cached() -> None:
    # A batch containing an unresolved id (indexing lag) must NOT be cached:
    # with no TTL, a cached "not found" would pin the miss for the whole
    # process. The repeat call must re-fetch.
    with patch(
        "sagent.tools.paper_details.metadata_batch",
        return_value=_batch_records(),
    ) as m:
        ids = ["10.1234/a", "10.1234/b", "10.1234/c"]
        first = asyncio.run(PaperDetails().run({"ids": ids}))
        _ = asyncio.run(PaperDetails().run({"ids": ids}))
    assert "not found" in first.content
    assert m.call_count == 2  # re-fetched, miss not pinned


def test_run_ids_batch_cache_keys_on_canonical_id() -> None:
    # Equivalent id spellings (arXiv: prefix vs bare) resolve to the same wire
    # id, so they share one cache entry -- the second call is a hit.
    records: list[PaperRecord | None] = [
        PaperRecord(title="P", arxiv_id="2401.00001"),
        PaperRecord(title="Q", arxiv_id="2401.00002"),
    ]
    with patch(
        "sagent.tools.paper_details.metadata_batch",
        return_value=records,
    ) as m:
        _ = asyncio.run(
            PaperDetails().run({"ids": ["arXiv:2401.00001", "arXiv:2401.00002"]})
        )
        _ = asyncio.run(PaperDetails().run({"ids": ["2401.00001", "2401.00002"]}))
    assert m.call_count == 1


# ---------------------------------------------------------------------------
# Google Scholar cited-by pivot (internal build only)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
