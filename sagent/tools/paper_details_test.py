"""Tests for ``tools.paper_details``: S2 metadata + citation graph."""

from __future__ import annotations

from unittest.mock import patch

import asyncio
import json

import pytest

from sagent.lib.web.fetch import FetchError
from sagent.tools import paper_common
from sagent.tools.paper_details import (
    PaperDetails,
    _validate_details_args,
)


class _NoWaitGate:
    """Stand-in S2 gate that never blocks."""

    async def acquire_async(self) -> None:
        return None


@pytest.fixture(autouse=True)
def _no_rate_wait(  # pyright: ignore[reportUnusedFunction] -- autouse fixture
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stub the S2 gate so batched multi-call tests never sleep."""
    monkeypatch.setattr(paper_common, "_s2_gate", _NoWaitGate)


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


def test_run_missing_ids() -> None:
    result = asyncio.run(PaperDetails().run({}))
    assert result.is_error
    assert "'ids' is required" in result.content


def test_run_rejects_zero_limit() -> None:
    # LIM-001: limit=0 must error, not silently return no results.
    result = asyncio.run(
        PaperDetails().run(
            {"ids": ["10.1234/x"], "operation": "references", "limit": 0}
        )
    )
    assert result.is_error
    assert "limit" in result.content


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
            PaperDetails().run({"ids": ["10.1234/cached_meta"]}),
        )
    assert not result.is_error
    assert "title: T" in result.content


def test_run_metadata_caches() -> None:
    with patch(
        "sagent.tools.paper_common.fetch",
        return_value=_metadata_payload(),
    ) as mock_fetch:
        _ = asyncio.run(PaperDetails().run({"ids": ["10.1234/unique_cache_doi_zz"]}))
        _ = asyncio.run(PaperDetails().run({"ids": ["10.1234/unique_cache_doi_zz"]}))
    assert mock_fetch.call_count == 1


def test_run_metadata_not_found() -> None:
    err = FetchError(
        url="https://api.semanticscholar.org/graph/v1/paper/DOI:10.1234/x",
        status=404,
        headers={},
        body=b"not found",
    )
    with patch("sagent.tools.paper_common.fetch", side_effect=err):
        result = asyncio.run(PaperDetails().run({"ids": ["10.1234/notfound_id"]}))
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
            PaperDetails().run(
                {"ids": ["10.1234/ref_target"], "operation": "references"}
            ),
        )
    assert "Cited Paper" in result.content


def test_run_references_empty() -> None:
    payload = json.dumps({"data": []}).encode()
    with patch("sagent.tools.paper_common.fetch", return_value=payload):
        result = asyncio.run(
            PaperDetails().run(
                {"ids": ["10.1234/empty_refs"], "operation": "references"}
            ),
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
                    "ids": ["10.1234/cit_target"],
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
                    "ids": ["10.1234/influential_target"],
                    "operation": "citations",
                    "influential_only": True,
                }
            ),
        )
    assert "Important" in result.content
    assert "Citing" not in result.content


def test_run_invalid_id_returns_error() -> None:
    result = asyncio.run(PaperDetails().run({"ids": ["garbage"]}))
    assert result.is_error


def _batch_payload() -> bytes:
    return json.dumps(
        [
            {"title": "First", "externalIds": {"DOI": "10.1234/a"}},
            None,
            {"title": "Third", "externalIds": {"DOI": "10.1234/c"}},
        ]
    ).encode()


def test_run_ids_batches_one_request() -> None:
    with patch(
        "sagent.tools.paper_common.fetch",
        return_value=_batch_payload(),
    ) as mock_fetch:
        result = asyncio.run(
            PaperDetails().run({"ids": ["10.1234/a", "10.1234/b", "10.1234/c"]}),
        )
    assert mock_fetch.call_count == 1  # one batched call, not three
    assert not result.is_error
    assert "title: First" in result.content
    assert "10.1234/b: not found" in result.content
    assert "title: Third" in result.content


def test_run_ids_posts_batch_endpoint() -> None:
    captured: dict[str, object] = {}

    def fake_fetch(**kw: object) -> bytes:
        captured.update(kw)
        return b"[{}, {}]"

    with patch("sagent.tools.paper_common.fetch", side_effect=fake_fetch):
        _ = asyncio.run(PaperDetails().run({"ids": ["10.1234/a", "10.1234/b"]}))
    assert captured["method"] == "POST"
    assert str(captured["url"]).endswith("/paper/batch")


def test_run_ids_with_operation_rejected() -> None:
    result = asyncio.run(
        PaperDetails().run(
            {"ids": ["10.1234/a", "10.1234/b"], "operation": "references"}
        ),
    )
    assert result.is_error
    assert "exactly one id" in result.content


def test_run_empty_ids_rejected() -> None:
    result = asyncio.run(PaperDetails().run({"ids": []}))
    assert result.is_error


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
