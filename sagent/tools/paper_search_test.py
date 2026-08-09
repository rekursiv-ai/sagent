"""Tests for ``tools.paper_search``: the thin adapter over ``lib.web.paper``.

Backend I/O, fusion, and record extraction now live in ``wesearch.paper``
and are covered by that library's own tests. These tests exercise ONLY the
adapter: schema/metadata, arg validation, the process cache, text rendering,
empty-result hints, and error mapping. The library boundary is mocked where the
adapter binds it (``paper_search.search``).
"""

from __future__ import annotations

from unittest.mock import patch

import asyncio

from wesearch.paper.custom_types import PaperRecord
from wesearch.paper.errors import PaperError
from wesearch.paper.search import SearchResult

from sagent.lib.custom_json import MutableJSON
from sagent.tools import paper_search
from sagent.tools.paper_search import PaperSearch, _empty_hint


def _result(
    records: list[PaperRecord], *, total: int | None = None, complete: bool = True
) -> SearchResult:
    """Build a library ``SearchResult`` for mocking ``paper_search.search``."""
    return SearchResult(
        records=records,
        total=len(records) if total is None else total,
        complete=complete,
    )


def _clear_cache() -> None:
    """Reset the process-wide result cache so tests stay hermetic."""
    paper_search._cache.clear()


# ---------------------------------------------------------------------------
# Metadata / summary / prompt
# ---------------------------------------------------------------------------


def test_paper_search_metadata() -> None:
    t = PaperSearch()
    assert t.name == "PaperSearch"
    assert t.tool_id == "application/x-tool-papersearch"


def test_summary_default() -> None:
    out = PaperSearch().summary({"query": "transformers"})
    assert "PaperSearch" in out
    assert "transformers" in out


def test_summary_with_source() -> None:
    out = PaperSearch().summary({"query": "x", "source": "openalex"})
    assert "(openalex)" in out


def test_summary_fused_source_no_suffix() -> None:
    out = PaperSearch().summary({"query": "x", "source": "fused"})
    assert "(fused)" not in out


def test_summary_keeps_long_query() -> None:
    q = "q" * 80
    assert PaperSearch().summary({"query": q}) == f"PaperSearch {q!r}"


def test_summary_empty_query() -> None:
    assert PaperSearch().summary({"query": ""}) == "PaperSearch"


def test_prompt_empty() -> None:
    assert PaperSearch().prompt() == ""


# ---------------------------------------------------------------------------
# Argument validation (adapter-owned, no library call)
# ---------------------------------------------------------------------------


def test_run_empty_query() -> None:
    result = asyncio.run(PaperSearch().run({"query": "  "}))
    assert result.is_error
    assert "required" in result.content


def test_run_invalid_source() -> None:
    result = asyncio.run(PaperSearch().run({"query": "x", "source": "bing"}))
    assert result.is_error
    assert "Invalid source" in result.content


def test_run_rejects_zero_limit() -> None:
    result = asyncio.run(PaperSearch().run({"query": "x", "limit": 0}))
    assert result.is_error
    assert "limit" in result.content


def test_run_rejects_inverted_year_range() -> None:
    result = asyncio.run(
        PaperSearch().run({"query": "x", "year_from": 2025, "year_to": 2020})
    )
    assert result.is_error
    assert "year_from" in result.content


def test_run_rejects_zero_abstract_chars() -> None:
    result = asyncio.run(PaperSearch().run({"query": "x", "abstract_chars": 0}))
    assert result.is_error
    assert "abstract_chars" in result.content


# ---------------------------------------------------------------------------
# Rendering: adapter formats library records into text
# ---------------------------------------------------------------------------


def test_run_renders_records() -> None:
    _clear_cache()
    rec = PaperRecord(
        title="Attention Is All You Need",
        authors=("Vaswani",),
        year=2017,
        doi="10.0/x",
        sources=("s2",),
    )
    with patch(
        "sagent.tools.paper_search.search",
        return_value=_result([rec]),
    ):
        result = asyncio.run(
            PaperSearch().run({"query": "transformers", "source": "s2"})
        )
    assert not result.is_error
    assert "Attention Is All You Need" in result.content
    assert "[doi:10.0/x]" in result.content
    assert "Vaswani" in result.content
    assert "2017" in result.content


def test_run_empty_renders_no_results() -> None:
    _clear_cache()
    with patch(
        "sagent.tools.paper_search.search",
        return_value=_result([], total=0),
    ):
        result = asyncio.run(PaperSearch().run({"query": "nothing", "source": "s2"}))
    assert result.content.startswith("(no results)")


def test_run_truncation_notice_when_total_exceeds_shown() -> None:
    _clear_cache()
    rec = PaperRecord(title="Only Shown", sources=("s2",))
    with patch(
        "sagent.tools.paper_search.search",
        return_value=_result([rec], total=42),
    ):
        result = asyncio.run(PaperSearch().run({"query": "x", "source": "s2"}))
    assert "... (showing 1 of 42; tighten filters for more)" in result.content


def test_run_abstract_truncated_to_cap() -> None:
    _clear_cache()
    rec = PaperRecord(title="T", abstract="x" * 100, sources=("s2",))
    with patch(
        "sagent.tools.paper_search.search",
        return_value=_result([rec]),
    ):
        result = asyncio.run(
            PaperSearch().run({"query": "x", "source": "s2", "abstract_chars": 10})
        )
    assert "..." in result.content
    # Cap applies to abstract chars, not the whole rendering.
    assert "x" * 100 not in result.content


def test_run_limit_caps_rendered_hits() -> None:
    _clear_cache()
    recs = [PaperRecord(title=f"P{i}", sources=("s2",)) for i in range(5)]
    with patch(
        "sagent.tools.paper_search.search",
        return_value=_result(recs, total=5),
    ):
        result = asyncio.run(
            PaperSearch().run({"query": "x", "source": "s2", "limit": 2})
        )
    assert "P0" in result.content
    assert "P1" in result.content
    assert "P4" not in result.content


# ---------------------------------------------------------------------------
# Empty-result hints (adapter-owned _empty_hint)
# ---------------------------------------------------------------------------


def test_empty_hint_multiterm_advises_dropping_terms() -> None:
    hint = _empty_hint([], "object centric slot attention")
    assert "every query term" in hint.lower()
    assert "drop" in hint.lower()


def test_empty_hint_single_term_no_and_advice() -> None:
    hint = _empty_hint([], "xyzzynonsense")
    assert "drop" not in hint.lower()
    # Single-term empties still get the author-tool note.
    assert "PaperAuthor" in hint


def test_empty_hint_points_to_paper_author() -> None:
    hint = _empty_hint([], "Andrews Sparks")
    assert "PaperAuthor" in hint


def test_empty_hint_absent_when_hits_present() -> None:
    hint = _empty_hint([PaperRecord(title="T")], "some multi term query")
    assert hint == ""


def test_run_multiterm_empty_appends_and_hint() -> None:
    _clear_cache()
    with patch(
        "sagent.tools.paper_search.search",
        return_value=_result([], total=0),
    ):
        result = asyncio.run(
            PaperSearch().run(
                {"query": "object-centric slot attention ARC", "source": "fused"}
            )
        )
    assert "(no results)" in result.content
    assert "every query term" in result.content.lower()
    assert "drop" in result.content.lower()


def test_run_empty_appends_author_hint() -> None:
    _clear_cache()
    with patch(
        "sagent.tools.paper_search.search",
        return_value=_result([], total=0),
    ):
        result = asyncio.run(
            PaperSearch().run({"query": "Andrews Sparks", "source": "s2"})
        )
    assert "(no results)" in result.content
    assert "PaperAuthor" in result.content


def test_run_hits_no_hint() -> None:
    _clear_cache()
    rec = PaperRecord(title="Hit", sources=("s2",))
    with patch(
        "sagent.tools.paper_search.search",
        return_value=_result([rec]),
    ):
        result = asyncio.run(PaperSearch().run({"query": "a b c", "source": "s2"}))
    assert "PaperAuthor" not in result.content
    assert "every query term" not in result.content.lower()


# ---------------------------------------------------------------------------
# Process cache
# ---------------------------------------------------------------------------


def test_run_caches_complete_results() -> None:
    _clear_cache()
    rec = PaperRecord(title="Cached", sources=("s2",))
    with patch(
        "sagent.tools.paper_search.search",
        return_value=_result([rec]),
    ) as mock_search:
        args: MutableJSON = {"query": "uniquecachetestkey1", "source": "s2"}
        first = asyncio.run(PaperSearch().run(args))
        second = asyncio.run(PaperSearch().run(args))
    assert mock_search.call_count == 1  # second served from cache
    assert first.content == second.content


def test_run_does_not_cache_partial_results() -> None:
    _clear_cache()
    rec = PaperRecord(title="Partial", sources=("openalex",))
    partial = _result([rec], complete=False)
    with patch("sagent.tools.paper_search.search", return_value=partial) as mock_search:
        args: MutableJSON = {"query": "partialcacheprobe", "source": "fused"}
        _ = asyncio.run(PaperSearch().run(args))
        _ = asyncio.run(PaperSearch().run(args))
    assert mock_search.call_count == 2  # partial result re-queried, not cached


# ---------------------------------------------------------------------------
# Error mapping: library exceptions -> ToolResult
# ---------------------------------------------------------------------------


def test_run_paper_error_maps_to_tool_error() -> None:
    _clear_cache()
    with patch(
        "sagent.tools.paper_search.search",
        side_effect=PaperError("all backends down"),
    ):
        result = asyncio.run(PaperSearch().run({"query": "x", "source": "fused"}))
    assert result.is_error
    assert "all backends down" in result.content


def test_run_paper_error_not_cached() -> None:
    _clear_cache()
    with patch(
        "sagent.tools.paper_search.search",
        side_effect=PaperError("boom"),
    ) as mock_search:
        args: MutableJSON = {"query": "errornotcachedprobe", "source": "fused"}
        _ = asyncio.run(PaperSearch().run(args))
        _ = asyncio.run(PaperSearch().run(args))
    assert mock_search.call_count == 2  # errors never cached


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
