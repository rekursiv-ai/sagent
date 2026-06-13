"""Tests for ``tools.paper_author``: Semantic Scholar author lookup."""

from __future__ import annotations

from typing import cast
from unittest.mock import patch

import asyncio
import json

from sagent.lib.json import MutableJSON
from sagent.tools.paper_author import (
    PaperAuthor,
    _s2_author_to_record,
    _validate_author_args,
)


def test_paper_author_metadata() -> None:
    t = PaperAuthor()
    assert t.name == "PaperAuthor"
    assert t.tool_id == "application/x-tool-paperauthor"


def test_summary_search() -> None:
    t = PaperAuthor()
    out = t.summary({"query": "Bengio"})
    assert "PaperAuthor search" in out
    assert "Bengio" in out


def test_summary_search_truncates() -> None:
    out = PaperAuthor().summary({"query": "x" * 60})
    assert "..." in out


def test_summary_id_no_op() -> None:
    out = PaperAuthor().summary({"ids": ["12345"]})
    assert out == "PaperAuthor 12345"


def test_summary_id_papers() -> None:
    out = PaperAuthor().summary({"ids": ["12345"], "operation": "papers"})
    assert out == "PaperAuthor papers 12345"


def test_summary_nothing() -> None:
    assert PaperAuthor().summary({}) == "PaperAuthor"


def test_prompt_empty() -> None:
    assert PaperAuthor().prompt() == ""


def test_validate_both_query_and_id() -> None:
    err = _validate_author_args("q", ["1"], "", year_from=None, year_to=None)
    assert err is not None
    assert err.is_error


def test_validate_neither_query_nor_id() -> None:
    err = _validate_author_args("", [], "", year_from=None, year_to=None)
    assert err is not None


def test_validate_unknown_op() -> None:
    err = _validate_author_args("", ["1"], "frob", year_from=None, year_to=None)
    assert err is not None
    assert "Unknown operation" in err.content


def test_validate_op_requires_id() -> None:
    err = _validate_author_args("", [], "papers", year_from=None, year_to=None)
    assert err is not None


def test_validate_year_only_for_papers() -> None:
    err = _validate_author_args("alice", [], "", year_from=2020, year_to=None)
    assert err is not None
    assert "year" in err.content.lower()


def test_validate_ok_for_search() -> None:
    err = _validate_author_args("q", [], "", year_from=None, year_to=None)
    assert err is None


def test_validate_ok_for_papers() -> None:
    err = _validate_author_args("", ["1"], "papers", year_from=2020, year_to=None)
    assert err is None


def test_s2_author_to_record_full() -> None:
    data: MutableJSON = {
        "authorId": "1741101",
        "name": "Bengio",
        "aliases": ["Yoshua Bengio"],
        "affiliations": ["Mila", {"name": "U Montreal"}],
        "homepage": "https://x",
        "hIndex": 200,
        "citationCount": 500_000,
        "paperCount": 800,
    }
    rec = _s2_author_to_record(data)
    assert rec.author_id == "1741101"
    assert rec.name == "Bengio"
    assert rec.aliases == ("Yoshua Bengio",)
    assert "Mila" in rec.affiliations
    assert "U Montreal" in rec.affiliations
    assert rec.homepage == "https://x"
    assert rec.h_index == 200


def test_s2_author_to_record_sparse() -> None:
    empty: MutableJSON = {}
    rec = _s2_author_to_record(empty)
    assert rec.author_id == ""
    assert rec.name == "(unknown)"
    assert rec.homepage is None


def test_s2_author_to_record_affiliations_dict_with_affiliation_key() -> None:
    data: MutableJSON = {
        "authorId": "1",
        "name": "X",
        "affiliations": [{"affiliation": "Foo Inst"}],
    }
    rec = _s2_author_to_record(data)
    assert "Foo Inst" in rec.affiliations


def test_run_invalid_args() -> None:
    result = asyncio.run(PaperAuthor().run({}))
    assert result.is_error


def test_run_search() -> None:
    payload = json.dumps(
        {
            "total": 1,
            "data": [
                {
                    "authorId": "42",
                    "name": "Searched",
                    "hIndex": 100,
                }
            ],
        }
    ).encode()
    with patch("sagent.tools.paper_common.fetch", return_value=payload):
        result = asyncio.run(
            PaperAuthor().run({"query": "unique_search_query_xyz"}),
        )
    assert "Searched" in result.content


def test_run_search_empty() -> None:
    payload = json.dumps({"total": 0, "data": []}).encode()
    with patch("sagent.tools.paper_common.fetch", return_value=payload):
        result = asyncio.run(
            PaperAuthor().run({"query": "no_results_unique_query"}),
        )
    assert result.content == "(no results)"


def test_run_search_sorted_by_hindex() -> None:
    payload = json.dumps(
        {
            "total": 2,
            "data": [
                {"authorId": "1", "name": "Lower", "hIndex": 10},
                {"authorId": "2", "name": "Higher", "hIndex": 100},
            ],
        }
    ).encode()
    with patch("sagent.tools.paper_common.fetch", return_value=payload):
        result = asyncio.run(
            PaperAuthor().run({"query": "sort_test_unique"}),
        )
    higher_idx = result.content.find("Higher")
    lower_idx = result.content.find("Lower")
    assert higher_idx >= 0
    assert higher_idx < lower_idx


def test_run_author_metadata() -> None:
    payload = json.dumps(
        {
            "authorId": "12345",
            "name": "Yoshua",
            "hIndex": 200,
            "paperCount": 800,
        }
    ).encode()
    with patch("sagent.tools.paper_common.fetch", return_value=payload):
        result = asyncio.run(PaperAuthor().run({"ids": ["unique_author_12345"]}))
    assert "Yoshua" in result.content
    assert "h_index: 200" in result.content


def test_run_papers() -> None:
    payload = json.dumps(
        {
            "data": [
                {
                    "title": "Paper",
                    "year": 2020,
                    "externalIds": {"DOI": "10.1234/p"},
                    "authors": [{"name": "A"}],
                }
            ]
        }
    ).encode()
    with patch("sagent.tools.paper_common.fetch", return_value=payload):
        result = asyncio.run(
            PaperAuthor().run({"ids": ["unique_papers_author"], "operation": "papers"}),
        )
    assert "Paper" in result.content


def test_run_papers_year_filter() -> None:
    payload = json.dumps(
        {
            "data": [
                {
                    "title": "Old",
                    "year": 2010,
                    "externalIds": {"DOI": "10.1234/old"},
                },
                {
                    "title": "New",
                    "year": 2023,
                    "externalIds": {"DOI": "10.1234/new"},
                },
            ]
        }
    ).encode()
    with patch("sagent.tools.paper_common.fetch", return_value=payload):
        result = asyncio.run(
            PaperAuthor().run(
                {
                    "ids": ["unique_year_filt_author"],
                    "operation": "papers",
                    "year_from": 2020,
                }
            ),
        )
    assert "New" in result.content
    assert "Old" not in result.content


def test_run_papers_year_filter_overfetches() -> None:
    # With a year filter active, fetch a wide page so matches past `limit`
    # aren't dropped before filtering.
    captured: dict[str, object] = {}

    def fake_fetch(**kw: object) -> bytes:
        captured.update(kw)
        return json.dumps({"data": []}).encode()

    with patch("sagent.tools.paper_common.fetch", side_effect=fake_fetch):
        _ = asyncio.run(
            PaperAuthor().run(
                {"ids": ["a"], "operation": "papers", "year_from": 2020, "limit": 5}
            ),
        )
    params = cast(dict[str, object], captured["params"])
    assert params["limit"] == 1000  # wide fetch, not the requested 5


def test_run_papers_no_results_after_filter() -> None:
    payload = json.dumps(
        {
            "data": [
                {
                    "title": "Old",
                    "year": 2010,
                    "externalIds": {"DOI": "10.1234/old"},
                },
            ]
        }
    ).encode()
    with patch("sagent.tools.paper_common.fetch", return_value=payload):
        result = asyncio.run(
            PaperAuthor().run(
                {
                    "ids": ["unique_empty_after_filt_author"],
                    "operation": "papers",
                    "year_from": 2030,
                }
            ),
        )
    assert result.content == "(no results)"


def test_run_caches_search() -> None:
    payload = json.dumps(
        {"total": 1, "data": [{"authorId": "1", "name": "Cached", "hIndex": 1}]}
    ).encode()
    with patch("sagent.tools.paper_common.fetch", return_value=payload) as mock_fetch:
        _ = asyncio.run(PaperAuthor().run({"query": "cache_search_unique_query_abc"}))
        _ = asyncio.run(PaperAuthor().run({"query": "cache_search_unique_query_abc"}))
    assert mock_fetch.call_count == 1


def test_run_ids_batches_author_metadata() -> None:
    # Many author ids resolve in ONE /author/batch call, aligned by input.
    batch_array = json.dumps(
        [
            {"authorId": "1", "name": "First", "hIndex": 10},
            None,
            {"authorId": "3", "name": "Third", "hIndex": 30},
        ]
    ).encode()
    captured: dict[str, object] = {}

    def fake_fetch(**kw: object) -> bytes:
        captured.update(kw)
        return batch_array

    with patch("sagent.tools.paper_common.fetch", side_effect=fake_fetch):
        result = asyncio.run(PaperAuthor().run({"ids": ["1", "2", "3"]}))
    assert captured["method"] == "POST"
    assert str(captured["url"]).endswith("/author/batch")
    assert "First" in result.content
    assert "2: not found" in result.content
    assert "Third" in result.content


def test_run_bare_string_id_coerced() -> None:
    # A single id may be passed as a bare string (coerced to a one-element
    # list), consistent with PaperDetails / PaperFetch. One id -> single GET.
    def fake_fetch(**kw: object) -> bytes:
        del kw
        return json.dumps({"authorId": "123", "name": "Solo"}).encode()

    with patch("sagent.tools.paper_common.fetch", side_effect=fake_fetch):
        result = asyncio.run(PaperAuthor().run({"ids": "123"}))
    assert not result.is_error
    assert "Solo" in result.content


def test_run_ids_wrong_scalar_type_errors() -> None:
    # A non-string, non-list scalar is still rejected (not coerced).
    result = asyncio.run(PaperAuthor().run({"ids": 123}))
    assert result.is_error
    assert "'ids' must be a list of strings or a single string" in result.content


def test_run_ids_with_query_rejected() -> None:
    result = asyncio.run(PaperAuthor().run({"ids": ["1"], "query": "x"}))
    assert result.is_error


def test_run_ids_multi_with_papers_rejected() -> None:
    result = asyncio.run(
        PaperAuthor().run({"ids": ["1", "2"], "operation": "papers"}),
    )
    assert result.is_error
    assert "exactly one id" in result.content


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
