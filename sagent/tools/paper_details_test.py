"""Tests for ``tools.paper_details``: S2 metadata + citation graph."""

from __future__ import annotations

from unittest.mock import patch

import asyncio
import json

from sagent.lib.web.fetch import FetchError
from sagent.tools.paper_details import (
    PaperDetails,
    _validate_details_args,
)


def test_paper_details_metadata() -> None:
    t = PaperDetails()
    assert t.name == "PaperDetails"
    assert t.tool_id == "application/x-tool-paperdetails"


def test_summary_default() -> None:
    t = PaperDetails()
    assert t.summary({"id": "10.1234/abc"}) == "PaperDetails 10.1234/abc"


def test_summary_references() -> None:
    t = PaperDetails()
    out = t.summary({"id": "10.1234/abc", "operation": "references"})
    assert out == "PaperDetails references 10.1234/abc"


def test_summary_citations() -> None:
    out = PaperDetails().summary({"id": "10.1234/abc", "operation": "citations"})
    assert out == "PaperDetails citations 10.1234/abc"


def test_summary_missing_id() -> None:
    assert PaperDetails().summary({}) == "PaperDetails ?"


def test_prompt_empty() -> None:
    assert PaperDetails().prompt() == ""


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


def test_run_missing_id() -> None:
    result = asyncio.run(PaperDetails().run({}))
    assert result.is_error
    assert "'id' is required" in result.content


def _metadata_payload() -> bytes:
    return json.dumps(
        {
            "title": "T",
            "year": 2020,
            "externalIds": {"DOI": "10.1234/cached_meta"},
            "authors": [{"name": "A"}],
        }
    ).encode()


def test_run_metadata_success() -> None:
    with patch(
        "sagent.tools.paper_common.fetch",
        return_value=_metadata_payload(),
    ):
        result = asyncio.run(
            PaperDetails().run({"id": "10.1234/cached_meta"}),
        )
    assert not result.is_error
    assert "title: T" in result.content


def test_run_metadata_caches() -> None:
    with patch(
        "sagent.tools.paper_common.fetch",
        return_value=_metadata_payload(),
    ) as mock_fetch:
        _ = asyncio.run(PaperDetails().run({"id": "10.1234/unique_cache_doi_zz"}))
        _ = asyncio.run(PaperDetails().run({"id": "10.1234/unique_cache_doi_zz"}))
    assert mock_fetch.call_count == 1


def test_run_metadata_not_found() -> None:
    err = FetchError(
        url="https://api.semanticscholar.org/graph/v1/paper/DOI:10.1234/x",
        status=404,
        headers={},
        body=b"not found",
    )
    with patch("sagent.tools.paper_common.fetch", side_effect=err):
        result = asyncio.run(PaperDetails().run({"id": "10.1234/notfound_id"}))
    assert result.is_error


def test_run_references() -> None:
    payload = json.dumps(
        {
            "data": [
                {
                    "isInfluential": False,
                    "citedPaper": {
                        "title": "Cited Paper",
                        "year": 2019,
                        "externalIds": {"DOI": "10.1234/ref1"},
                    },
                }
            ]
        }
    ).encode()
    with patch("sagent.tools.paper_common.fetch", return_value=payload):
        result = asyncio.run(
            PaperDetails().run({"id": "10.1234/ref_target", "operation": "references"}),
        )
    assert "Cited Paper" in result.content


def test_run_references_empty() -> None:
    payload = json.dumps({"data": []}).encode()
    with patch("sagent.tools.paper_common.fetch", return_value=payload):
        result = asyncio.run(
            PaperDetails().run({"id": "10.1234/empty_refs", "operation": "references"}),
        )
    assert result.content == "(no results)"


def test_run_citations_with_year_filter() -> None:
    payload = json.dumps(
        {
            "data": [
                {
                    "isInfluential": True,
                    "citingPaper": {
                        "title": "Old",
                        "year": 2015,
                        "externalIds": {"DOI": "10.1234/old"},
                    },
                },
                {
                    "isInfluential": False,
                    "citingPaper": {
                        "title": "New",
                        "year": 2022,
                        "externalIds": {"DOI": "10.1234/new"},
                    },
                },
            ]
        }
    ).encode()
    with patch("sagent.tools.paper_common.fetch", return_value=payload):
        result = asyncio.run(
            PaperDetails().run(
                {
                    "id": "10.1234/cit_target",
                    "operation": "citations",
                    "year_from": 2020,
                }
            ),
        )
    assert "New" in result.content
    assert "Old" not in result.content


def test_run_citations_influential_only() -> None:
    payload = json.dumps(
        {
            "data": [
                {
                    "isInfluential": True,
                    "citingPaper": {
                        "title": "Important",
                        "externalIds": {"DOI": "10.1234/imp"},
                    },
                },
                {
                    "isInfluential": False,
                    "citingPaper": {
                        "title": "Citing",
                        "externalIds": {"DOI": "10.1234/citr"},
                    },
                },
            ]
        }
    ).encode()
    with patch("sagent.tools.paper_common.fetch", return_value=payload):
        result = asyncio.run(
            PaperDetails().run(
                {
                    "id": "10.1234/influential_target",
                    "operation": "citations",
                    "influential_only": True,
                }
            ),
        )
    assert "Important" in result.content
    assert "Citing" not in result.content


def test_run_invalid_id_returns_error() -> None:
    result = asyncio.run(PaperDetails().run({"id": "garbage"}))
    assert result.is_error


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
