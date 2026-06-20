"""Tests for ``tools.paper_search``: S2 + OpenAlex search + fusion."""

from __future__ import annotations

from unittest.mock import patch

import asyncio
import json

from sagent.lib.json import MutableJSON
from sagent.lib.web.fetch import FetchError
from sagent.tools.paper_common import PaperRecord
from sagent.tools.paper_search import (
    PaperSearch,
    _dedup_key,
    _fuse,
    _merge,
    _normalize_title,
    _openalex_filter,
    _openalex_work_to_record,
    _s2_year_param,
)


def test_paper_search_metadata() -> None:
    t = PaperSearch()
    assert t.name == "PaperSearch"
    assert t.tool_id == "application/x-tool-papersearch"


def test_summary_default() -> None:
    t = PaperSearch()
    out = t.summary({"query": "transformers"})
    assert "PaperSearch" in out
    assert "transformers" in out


def test_summary_with_source() -> None:
    out = PaperSearch().summary({"query": "x", "source": "openalex"})
    assert "(openalex)" in out


def test_summary_truncates_long_query() -> None:
    out = PaperSearch().summary({"query": "q" * 80})
    assert "..." in out


def test_summary_empty_query() -> None:
    assert PaperSearch().summary({"query": ""}) == "PaperSearch"


def test_prompt_empty() -> None:
    assert PaperSearch().prompt() == ""


def test_s2_year_param_both() -> None:
    assert _s2_year_param(2020, 2022) == "2020-2022"


def test_s2_year_param_from_only() -> None:
    assert _s2_year_param(2020, None) == "2020-"


def test_s2_year_param_to_only() -> None:
    assert _s2_year_param(None, 2022) == "-2022"


def test_s2_year_param_neither() -> None:
    assert _s2_year_param(None, None) is None


def test_openalex_filter_no_bounds() -> None:
    assert (
        _openalex_filter(year_from=None, year_to=None, open_access_only=False) is None
    )


def test_openalex_filter_year_from() -> None:
    out = _openalex_filter(year_from=2020, year_to=None, open_access_only=False)
    assert out == "from_publication_date:2020-01-01"


def test_openalex_filter_open_access() -> None:
    out = _openalex_filter(year_from=None, year_to=None, open_access_only=True)
    assert out == "open_access.is_oa:true"


def test_openalex_filter_combined() -> None:
    out = _openalex_filter(year_from=2020, year_to=2022, open_access_only=True)
    assert out is not None
    assert "from_publication_date:2020-01-01" in out
    assert "to_publication_date:2022-12-31" in out
    assert "open_access.is_oa:true" in out


def test_openalex_work_to_record_full() -> None:
    work: MutableJSON = {
        "title": "Attention",
        "authorships": [
            {"author": {"display_name": "A"}},
            {"author": {"display_name": "B"}},
            {"author": {}},
        ],
        "publication_year": 2017,
        "primary_location": {"source": {"display_name": "NeurIPS"}},
        "doi": "https://doi.org/10.0/abc",
        "ids": {"arxiv": "https://arxiv.org/abs/1706.03762"},
        "abstract_inverted_index": {"hello": [0], "world": [1]},
        "cited_by_count": 100_000,
        "referenced_works_count": 50,
        "open_access": {"oa_url": "https://oa/x"},
    }
    rec = _openalex_work_to_record(work)
    assert rec.title == "Attention"
    assert rec.authors == ("A", "B")
    assert rec.year == 2017
    assert rec.venue == "NeurIPS"
    assert rec.doi == "10.0/abc"
    assert rec.arxiv_id == "1706.03762"
    assert rec.abstract == "hello world"
    assert rec.citation_count == 100_000
    assert rec.open_access_pdf == "https://oa/x"
    assert rec.sources == ("openalex",)


def test_openalex_work_to_record_sparse() -> None:
    rec = _openalex_work_to_record({"display_name": "Bare"})
    assert rec.title == "Bare"
    assert rec.authors == ()
    assert rec.doi is None


def test_openalex_work_to_record_doi_no_prefix() -> None:
    rec = _openalex_work_to_record({"title": "X", "doi": "10.1/y"})
    assert rec.doi == "10.1/y"


def test_normalize_title_strips_punct_lowercase() -> None:
    assert _normalize_title("Hello, World!") == "hello world"


def test_normalize_title_collapses_whitespace() -> None:
    assert _normalize_title("a   b\t\nc") == "a b c"


def test_dedup_key_prefers_doi() -> None:
    rec = PaperRecord(title="T", doi="10.1/X")
    assert _dedup_key(rec) == "doi:10.1/x"


def test_dedup_key_falls_back_to_title() -> None:
    rec = PaperRecord(title="Some Paper!")
    assert _dedup_key(rec) == "title:some paper"


def test_merge_prefers_first() -> None:
    a = PaperRecord(title="A", year=2020, sources=("s2",))
    b = PaperRecord(title="A", year=2019, doi="10.1/x", sources=("openalex",))
    out = _merge(a, b)
    assert out.year == 2020  # first wins
    assert out.doi == "10.1/x"  # filled from second
    assert out.sources == ("s2", "openalex")


def test_fuse_orders_s2_first() -> None:
    s2 = [PaperRecord(title="X", doi="10.1/a")]
    oa = [PaperRecord(title="Y", doi="10.1/b")]
    fused = _fuse(s2, oa)
    assert fused[0].doi == "10.1/a"
    assert fused[1].doi == "10.1/b"


def test_fuse_dedups_by_doi() -> None:
    s2 = [PaperRecord(title="X", doi="10.1/a", citation_count=100, sources=("s2",))]
    oa = [PaperRecord(title="Y", doi="10.1/a", abstract="abs", sources=("openalex",))]
    fused = _fuse(s2, oa)
    assert len(fused) == 1
    assert fused[0].title == "X"
    assert fused[0].abstract == "abs"
    assert "s2" in fused[0].sources
    assert "openalex" in fused[0].sources


def test_run_empty_query() -> None:
    result = asyncio.run(PaperSearch().run({"query": "  "}))
    assert result.is_error
    assert "required" in result.content


def test_run_invalid_source() -> None:
    result = asyncio.run(PaperSearch().run({"query": "x", "source": "bing"}))
    assert result.is_error
    assert "Invalid source" in result.content


def _s2_search_payload() -> bytes:
    return json.dumps(
        {
            "total": 1,
            "data": [
                {
                    "title": "Hit",
                    "year": 2020,
                    "externalIds": {"DOI": "10.0/x"},
                    "authors": [{"name": "A"}],
                }
            ],
        }
    ).encode()


def test_run_s2_success() -> None:
    with patch(
        "sagent.tools.paper_common.fetch",
        return_value=_s2_search_payload(),
    ):
        result = asyncio.run(PaperSearch().run({"query": "transformers"}))
    assert not result.is_error
    assert "Hit" in result.content


def test_run_no_results() -> None:
    payload = json.dumps({"total": 0, "data": []}).encode()
    with patch("sagent.tools.paper_common.fetch", return_value=payload):
        result = asyncio.run(PaperSearch().run({"query": "nothing"}))
    assert result.content.startswith("(no results)")


def test_run_no_results_s2_appends_author_hint() -> None:
    # S2 (default) returning nothing must nudge toward the fused backend:
    # an author-surname query silently zero-hits on S2 but resolves via
    # OpenAlex (live 2026-06-19 ARC-AGI survey reference walk).
    payload = json.dumps({"total": 0, "data": []}).encode()
    with patch("sagent.tools.paper_common.fetch", return_value=payload):
        result = asyncio.run(PaperSearch().run({"query": "Andrews Sparks"}))
    assert "(no results)" in result.content
    assert 'source="fused"' in result.content
    assert "author" in result.content.lower()


def test_run_no_results_fused_no_hint() -> None:
    # Fused already includes OpenAlex; suggesting "retry with fused" would be
    # circular, so an empty fused result stays bare.
    s2_empty = json.dumps({"total": 0, "data": []}).encode()
    oa_empty = json.dumps({"meta": {"count": 0}, "results": []}).encode()

    def fake(**kw: object) -> bytes:
        url = str(kw.get("url", ""))
        return oa_empty if "openalex" in url else s2_empty

    with (
        patch("sagent.tools.paper_common.fetch", side_effect=fake),
        patch("sagent.tools.paper_search.fetch", side_effect=fake),
    ):
        result = asyncio.run(
            PaperSearch().run({"query": "nothing here", "source": "fused"})
        )
    assert result.content == "(no results)"


def test_run_hits_no_author_hint() -> None:
    # The hint is for empty results only; a populated result stays clean.
    with patch(
        "sagent.tools.paper_common.fetch",
        return_value=_s2_search_payload(),
    ):
        result = asyncio.run(PaperSearch().run({"query": "transformers"}))
    assert 'source="fused"' not in result.content


def test_openalex_uses_interactive_timeout() -> None:
    # The fused backend queries OpenAlex alongside S2; its timeout must be
    # bounded too, or one leg can silently hang an interactive turn even after
    # the other returned (live 2026-06-19 idle-hang regression guard).
    captured: dict[str, object] = {}

    def fake_fetch(**kw: object) -> bytes:
        captured.update(kw)
        return b'{"meta": {"count": 0}, "results": []}'

    with patch("sagent.tools.paper_search.fetch", side_effect=fake_fetch):
        _ = asyncio.run(
            PaperSearch().run({"query": "timeout_probe", "source": "openalex"})
        )
    timeout = captured["timeout_sec"]
    assert isinstance(timeout, float)
    assert timeout <= 15.0


def test_default_source_is_s2_only() -> None:
    # The module docstring and behavior must agree: default queries S2 ALONE,
    # not the fused both-backends path.
    with (
        patch(
            "sagent.tools.paper_common.fetch",
            return_value=_s2_search_payload(),
        ) as s2_fetch,
        patch("sagent.tools.paper_search.fetch") as oa_fetch,
    ):
        result = asyncio.run(PaperSearch().run({"query": "default_source_probe"}))
    assert not result.is_error
    assert s2_fetch.call_count == 1
    assert oa_fetch.call_count == 0


def test_run_rejects_zero_abstract_chars() -> None:
    # Schema declares minimum 1; 0 must be rejected, not silently treated as
    # "no truncation" (matches validate_limit's reject-0 contract).
    result = asyncio.run(PaperSearch().run({"query": "x", "abstract_chars": 0}))
    assert result.is_error
    assert "abstract_chars" in result.content


def test_run_openalex_path() -> None:
    payload = json.dumps(
        {
            "meta": {"count": 1},
            "results": [
                {"title": "OA Hit", "doi": "10.0/oa", "publication_year": 2021}
            ],
        }
    ).encode()
    with patch("sagent.tools.paper_search.fetch", return_value=payload):
        result = asyncio.run(
            PaperSearch().run({"query": "openalex_path_query", "source": "openalex"}),
        )
    assert "OA Hit" in result.content


def test_run_openalex_timeout_returns_error() -> None:
    # OpenAlex shares the same fetch path as S2: a socket timeout raises
    # TimeoutError/OSError (not FetchError). With the lowered timeout it must
    # render as a ToolResult, not escape (parity with the S2 fix).
    with patch(
        "sagent.tools.paper_search.fetch",
        side_effect=TimeoutError("slow"),
    ):
        result = asyncio.run(PaperSearch().run({"query": "x", "source": "openalex"}))
    assert result.is_error
    assert "OpenAlex" in result.content


def test_run_openalex_connection_error_returns_error() -> None:
    with patch(
        "sagent.tools.paper_search.fetch",
        side_effect=OSError("refused"),
    ):
        result = asyncio.run(PaperSearch().run({"query": "x", "source": "openalex"}))
    assert result.is_error


def test_run_openalex_rate_limit_returns_error() -> None:
    err = FetchError(
        url="https://api.openalex.org/works",
        status=429,
        headers={},
        body=b"slow down",
    )
    with patch("sagent.tools.paper_search.fetch", side_effect=err):
        result = asyncio.run(
            PaperSearch().run(
                {"query": "openalex_rate_limit_query", "source": "openalex"}
            ),
        )
    assert result.is_error
    assert "rate limit" in result.content.lower()


def test_run_openalex_other_http_error() -> None:
    err = FetchError(
        url="https://api.openalex.org/works",
        status=500,
        headers={},
        body=b"boom",
    )
    with patch("sagent.tools.paper_search.fetch", side_effect=err):
        result = asyncio.run(
            PaperSearch().run(
                {"query": "openalex_other_http_query", "source": "openalex"}
            ),
        )
    assert result.is_error
    assert "500" in result.content


def test_run_openalex_invalid_json_returns_error() -> None:
    # A non-JSON 200 body must surface a ToolResult error, not crash the tool.
    with patch("sagent.tools.paper_search.fetch", return_value=b"<html>"):
        result = asyncio.run(
            PaperSearch().run(
                {"query": "openalex_bad_json_query", "source": "openalex"}
            ),
        )
    assert result.is_error
    assert "invalid JSON" in result.content


def test_run_fused_combines_backends() -> None:
    s2_payload = _s2_search_payload()
    openalex_payload = json.dumps(
        {
            "meta": {"count": 1},
            "results": [{"title": "Other", "doi": "10.0/other"}],
        }
    ).encode()

    def s2_fetch(**_kw: object) -> bytes:
        return s2_payload

    def oa_fetch(**_kw: object) -> bytes:
        return openalex_payload

    with (
        patch("sagent.tools.paper_common.fetch", side_effect=s2_fetch),
        patch("sagent.tools.paper_search.fetch", side_effect=oa_fetch),
    ):
        result = asyncio.run(
            PaperSearch().run({"query": "fused_combines_query", "source": "fused"}),
        )
    assert "Hit" in result.content
    assert "Other" in result.content


def test_run_fused_all_fail_returns_error() -> None:
    err = FetchError(url="u", status=500, headers={}, body=b"")
    with (
        patch("sagent.tools.paper_common.fetch", side_effect=err),
        patch("sagent.tools.paper_search.fetch", side_effect=err),
    ):
        result = asyncio.run(
            PaperSearch().run({"query": "fused_all_fail_query", "source": "fused"}),
        )
    assert result.is_error


def test_run_caches_results() -> None:
    payload = _s2_search_payload()
    with patch("sagent.tools.paper_common.fetch", return_value=payload) as mock_fetch:
        _ = asyncio.run(PaperSearch().run({"query": "uniquecachetestkey1"}))
        _ = asyncio.run(PaperSearch().run({"query": "uniquecachetestkey1"}))
    assert mock_fetch.call_count == 1


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
