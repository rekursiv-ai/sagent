"""Tests for ``tools.paper_author``: the S2 author-lookup adapter.

Backend I/O, record shaping, pagination, and fusion live in
:mod:`sagent.lib.web.paper` (covered by its own tests). These tests exercise only
the adapter: schema/metadata, arg validation, the id-bundle split, the process
cache, rendering, and library-error mapping. The library is mocked where the
adapter binds each name (``paper_author.search_authors`` etc.).
"""

from __future__ import annotations

from unittest.mock import patch

import asyncio

from sagent.lib.web.paper.authors import AuthorSearchResult
from sagent.lib.web.paper.custom_types import AuthorRecord, PaperRecord
from sagent.lib.web.paper.details import Listing
from sagent.lib.web.paper.errors import PaperError
from sagent.tools.paper_author import (
    PaperAuthor,
    _cache,
    _validate_author_args,
)
from sagent.types.runtime import ToolResult


def _clear_cache() -> None:
    _cache.clear()


# ---------------------------------------------------------------------------
# Tool metadata
# ---------------------------------------------------------------------------


def test_paper_author_metadata() -> None:
    t = PaperAuthor()
    assert t.name == "PaperAuthor"
    assert t.tool_id == "application/x-tool-paperauthor"


def test_prompt_empty() -> None:
    assert PaperAuthor().prompt() == ""


def test_serialize_key_none() -> None:
    assert PaperAuthor().serialize_key({"query": "x"}) is None


def test_summary_result_suppressed() -> None:
    assert PaperAuthor().summary_result(ToolResult(call_id="", content="x")) is None


# ---------------------------------------------------------------------------
# summary()
# ---------------------------------------------------------------------------


def test_summary_search() -> None:
    out = PaperAuthor().summary({"query": "Bengio"})
    assert "PaperAuthor search" in out
    assert "Bengio" in out


def test_summary_search_truncates() -> None:
    out = PaperAuthor().summary({"query": "x" * 60})
    assert "..." in out


def test_summary_id_no_op() -> None:
    assert PaperAuthor().summary({"ids": ["12345"]}) == "PaperAuthor 12345"


def test_summary_id_papers() -> None:
    out = PaperAuthor().summary({"ids": ["12345"], "operation": "papers"})
    assert out == "PaperAuthor papers 12345"


def test_summary_nothing() -> None:
    assert PaperAuthor().summary({}) == "PaperAuthor"


# ---------------------------------------------------------------------------
# _validate_author_args
# ---------------------------------------------------------------------------


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


def test_validate_op_with_no_id_fails() -> None:
    # No id at all trips the "required" guard before the papers-arity check.
    err = _validate_author_args("", [], "papers", year_from=None, year_to=None)
    assert err is not None
    assert "required" in err.content


def test_validate_op_rejects_multiple_ids() -> None:
    err = _validate_author_args("", ["1", "2"], "papers", year_from=None, year_to=None)
    assert err is not None
    assert "exactly one id" in err.content


def test_validate_year_only_for_papers() -> None:
    err = _validate_author_args("alice", [], "", year_from=2020, year_to=None)
    assert err is not None
    assert "year" in err.content.lower()


def test_validate_year_rejected_for_id_metadata() -> None:
    err = _validate_author_args("", ["1"], "", year_from=None, year_to=2020)
    assert err is not None
    assert "year" in err.content.lower()


def test_validate_ok_for_search() -> None:
    assert _validate_author_args("q", [], "", year_from=None, year_to=None) is None


def test_validate_ok_for_papers() -> None:
    err = _validate_author_args("", ["1"], "papers", year_from=2020, year_to=None)
    assert err is None


# ---------------------------------------------------------------------------
# run(): search
# ---------------------------------------------------------------------------


def test_run_invalid_args() -> None:
    _clear_cache()
    result = asyncio.run(PaperAuthor().run({}))
    assert result.is_error


def test_run_search() -> None:
    _clear_cache()
    ret = AuthorSearchResult(
        records=[AuthorRecord(author_id="42", name="Searched", h_index=100)],
        total=1,
    )
    with patch("sagent.tools.paper_author.search_authors", return_value=ret) as m:
        result = asyncio.run(PaperAuthor().run({"query": "q1"}))
    assert "Searched" in result.content
    assert m.call_args.args[0] == "q1"


def test_run_search_empty() -> None:
    _clear_cache()
    ret = AuthorSearchResult(records=[], total=0)
    with patch("sagent.tools.paper_author.search_authors", return_value=ret):
        result = asyncio.run(PaperAuthor().run({"query": "q2"}))
    assert result.content == "(no results)"


def test_run_search_truncation_notice() -> None:
    _clear_cache()
    ret = AuthorSearchResult(
        records=[AuthorRecord(author_id="1", name="Only", h_index=5)],
        total=10,
    )
    with patch("sagent.tools.paper_author.search_authors", return_value=ret):
        result = asyncio.run(PaperAuthor().run({"query": "q3"}))
    assert result.content == (
        "[author:1] Only - h-index:5\n... (showing 1 of 10; tighten filters for more)"
    )


def test_run_search_passes_limit() -> None:
    _clear_cache()
    ret = AuthorSearchResult(records=[], total=0)
    with patch("sagent.tools.paper_author.search_authors", return_value=ret) as m:
        _ = asyncio.run(PaperAuthor().run({"query": "q4", "limit": 7}))
    assert m.call_args.kwargs["limit"] == 7


# ---------------------------------------------------------------------------
# run(): single-author metadata
# ---------------------------------------------------------------------------


def test_run_author_metadata() -> None:
    _clear_cache()
    ret: list[AuthorRecord | None] = [
        AuthorRecord(author_id="12345", name="Yoshua", h_index=200, paper_count=800)
    ]
    with patch("sagent.tools.paper_author.author_metadata", return_value=ret) as m:
        result = asyncio.run(PaperAuthor().run({"ids": ["12345"]}))
    assert "name: Yoshua" in result.content
    assert "h_index: 200" in result.content
    assert m.call_args.args[0] == ["12345"]


def test_run_single_author_not_found() -> None:
    # The single-id path goes through the batch endpoint, so a miss renders as
    # "{id}: not found" (the id echoed), not a bare backend message.
    _clear_cache()
    ret: list[AuthorRecord | None] = [None]
    with patch("sagent.tools.paper_author.author_metadata", return_value=ret):
        result = asyncio.run(PaperAuthor().run({"ids": ["99999"]}))
    assert result.content == "99999: not found"


def test_run_bare_string_id_coerced() -> None:
    _clear_cache()
    ret: list[AuthorRecord | None] = [AuthorRecord(author_id="123", name="Solo")]
    with patch("sagent.tools.paper_author.author_metadata", return_value=ret):
        result = asyncio.run(PaperAuthor().run({"ids": "123"}))
    assert not result.is_error
    assert "name: Solo" in result.content


# ---------------------------------------------------------------------------
# run(): batched author metadata
# ---------------------------------------------------------------------------


def test_run_ids_batches_author_metadata() -> None:
    _clear_cache()
    ret: list[AuthorRecord | None] = [
        AuthorRecord(author_id="1", name="First", h_index=10),
        None,
        AuthorRecord(author_id="3", name="Third", h_index=30),
    ]
    with patch("sagent.tools.paper_author.author_metadata", return_value=ret) as m:
        result = asyncio.run(PaperAuthor().run({"ids": ["1", "2", "3"]}))
    assert m.call_args.args[0] == ["1", "2", "3"]
    assert "name: First" in result.content
    assert "2: not found" in result.content
    assert "name: Third" in result.content


def test_run_recovers_comma_joined_author_ids() -> None:
    # A comma-joined author-id STRING (the string-array wire shape) is recovered
    # into a batch via the ``str.isdigit`` predicate, since opaque author ids are
    # bare integers, not DOIs/arXiv ids.
    _clear_cache()
    ret: list[AuthorRecord | None] = [
        AuthorRecord(author_id="1741101", name="A", h_index=1),
        AuthorRecord(author_id="2064160", name="B", h_index=2),
    ]
    with patch("sagent.tools.paper_author.author_metadata", return_value=ret) as m:
        result = asyncio.run(PaperAuthor().run({"ids": "1741101,2064160"}))
    assert not result.is_error
    assert m.call_args.args[0] == ["1741101", "2064160"]


# ---------------------------------------------------------------------------
# run(): papers
# ---------------------------------------------------------------------------


def test_run_papers() -> None:
    _clear_cache()
    ret = Listing(
        records=[
            PaperRecord(title="Paper", year=2020, doi="10.1234/p", authors=("A",))
        ],
        complete=True,
    )
    with patch("sagent.tools.paper_author.author_papers", return_value=ret) as m:
        result = asyncio.run(
            PaperAuthor().run({"ids": ["7"], "operation": "papers"}),
        )
    assert "Paper" in result.content
    assert m.call_args.args[0] == "7"


def test_run_papers_empty() -> None:
    _clear_cache()
    ret = Listing(records=[], complete=True)
    with patch("sagent.tools.paper_author.author_papers", return_value=ret):
        result = asyncio.run(
            PaperAuthor().run({"ids": ["8"], "operation": "papers"}),
        )
    assert result.content == "(no results)"


def test_run_papers_incomplete_more_matches() -> None:
    _clear_cache()
    ret = Listing(
        records=[PaperRecord(title="P1", year=2021, doi="10.1/a")],
        complete=False,
    )
    with patch("sagent.tools.paper_author.author_papers", return_value=ret):
        result = asyncio.run(
            PaperAuthor().run({"ids": ["9"], "operation": "papers"}),
        )
    assert result.content.endswith(
        "\n... (more matches exist; raise 'limit' or narrow the years)"
    )


def test_run_papers_passes_year_bounds() -> None:
    _clear_cache()
    ret = Listing(records=[], complete=True)
    with patch("sagent.tools.paper_author.author_papers", return_value=ret) as m:
        _ = asyncio.run(
            PaperAuthor().run(
                {
                    "ids": ["10"],
                    "operation": "papers",
                    "year_from": 2020,
                    "year_to": 2023,
                    "limit": 5,
                }
            ),
        )
    assert m.call_args.kwargs == {"limit": 5, "year_from": 2020, "year_to": 2023}


# ---------------------------------------------------------------------------
# run(): cache behavior
# ---------------------------------------------------------------------------


def test_run_caches_search() -> None:
    _clear_cache()
    ret = AuthorSearchResult(
        records=[AuthorRecord(author_id="1", name="Cached", h_index=1)],
        total=1,
    )
    with patch("sagent.tools.paper_author.search_authors", return_value=ret) as m:
        _ = asyncio.run(PaperAuthor().run({"query": "cache_q"}))
        _ = asyncio.run(PaperAuthor().run({"query": "cache_q"}))
    assert m.call_count == 1


def test_run_ids_batch_caches_when_all_resolve() -> None:
    _clear_cache()
    ret: list[AuthorRecord | None] = [
        AuthorRecord(author_id="11", name="First", h_index=10),
        AuthorRecord(author_id="22", name="Second", h_index=20),
    ]
    with patch("sagent.tools.paper_author.author_metadata", return_value=ret) as m:
        _ = asyncio.run(PaperAuthor().run({"ids": ["11", "22"]}))
        _ = asyncio.run(PaperAuthor().run({"ids": ["11", "22"]}))
    assert m.call_count == 1


def test_run_ids_batch_miss_not_cached() -> None:
    # A batch with an unresolved id must not cache; the repeat re-fetches.
    _clear_cache()
    ret: list[AuthorRecord | None] = [
        AuthorRecord(author_id="33", name="Only", h_index=5),
        None,
    ]
    with patch("sagent.tools.paper_author.author_metadata", return_value=ret) as m:
        _ = asyncio.run(PaperAuthor().run({"ids": ["33", "44"]}))
        _ = asyncio.run(PaperAuthor().run({"ids": ["33", "44"]}))
    assert m.call_count == 2


# ---------------------------------------------------------------------------
# run(): arg-parsing / validation surfaced through run()
# ---------------------------------------------------------------------------


def test_run_rejects_zero_abstract_chars() -> None:
    _clear_cache()
    result = asyncio.run(PaperAuthor().run({"ids": ["1"], "abstract_chars": 0}))
    assert result.is_error
    assert "abstract_chars" in result.content


def test_run_ids_wrong_scalar_type_errors() -> None:
    _clear_cache()
    result = asyncio.run(PaperAuthor().run({"ids": 123}))
    assert result.is_error
    assert "'ids' must be a list of strings or a single string" in result.content


def test_run_ids_with_query_rejected() -> None:
    _clear_cache()
    result = asyncio.run(PaperAuthor().run({"ids": ["1"], "query": "x"}))
    assert result.is_error


def test_run_ids_multi_with_papers_rejected() -> None:
    _clear_cache()
    result = asyncio.run(
        PaperAuthor().run({"ids": ["1", "2"], "operation": "papers"}),
    )
    assert result.is_error
    assert "exactly one id" in result.content


# ---------------------------------------------------------------------------
# run(): library-error mapping
# ---------------------------------------------------------------------------


def test_run_search_paper_error_mapped() -> None:
    _clear_cache()
    with patch(
        "sagent.tools.paper_author.search_authors",
        side_effect=PaperError("s2 down"),
    ):
        result = asyncio.run(PaperAuthor().run({"query": "boom"}))
    assert result.is_error
    assert "s2 down" in result.content


def test_run_batch_paper_error_mapped() -> None:
    _clear_cache()
    with patch(
        "sagent.tools.paper_author.author_metadata",
        side_effect=PaperError("rate limited"),
    ):
        result = asyncio.run(PaperAuthor().run({"ids": ["1", "2"]}))
    assert result.is_error
    assert "rate limited" in result.content


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
