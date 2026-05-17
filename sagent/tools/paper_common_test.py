"""Tests for ``tools.paper_common``: shared helpers for Paper* tools."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast
from unittest.mock import patch

import asyncio

from sagent.lib.json import MutableJSON
from sagent.lib.web.fetch import FetchError
from sagent.tools.paper_common import (
    S2_BASE,
    AuthorRecord,
    PaperRecord,
    clamp_limit,
    format_author_block,
    format_author_line,
    format_block,
    format_record,
    id_slug,
    normalize_id,
    openalex_reconstruct_abstract,
    papers_cache_dir,
    s2_get,
    s2_paper_to_record,
    s2_wire_id,
    short_id,
    truncation_notice,
)
from sagent.types.history import ToolResult


if TYPE_CHECKING:
    import pytest


def test_short_id_short_unchanged() -> None:
    assert short_id("10.1/abc") == "10.1/abc"


def test_short_id_truncates_long() -> None:
    s = "x" * 60
    result = short_id(s)
    assert result.startswith("…")
    assert len(result) == 39


def test_clamp_limit_none_uses_default() -> None:
    assert clamp_limit(None, default=42) == 42


def test_clamp_limit_under_min_clamps_to_1() -> None:
    assert clamp_limit(0, default=10) == 1


def test_clamp_limit_over_max_clamps_to_1000() -> None:
    assert clamp_limit(99_999, default=10) == 1000


def test_clamp_limit_passes_through() -> None:
    assert clamp_limit(50, default=100) == 50


def test_normalize_id_doi_bare() -> None:
    result = normalize_id("10.1234/foo")
    assert result == ("doi", "10.1234/foo")


def test_normalize_id_doi_prefix() -> None:
    result = normalize_id("https://doi.org/10.1234/foo")
    assert result == ("doi", "10.1234/foo")


def test_normalize_id_doi_dx_prefix() -> None:
    assert normalize_id("https://dx.doi.org/10.1234/x") == ("doi", "10.1234/x")


def test_normalize_id_doi_short_prefix() -> None:
    assert normalize_id("doi:10.1234/x") == ("doi", "10.1234/x")


def test_normalize_id_arxiv_new_style() -> None:
    assert normalize_id("2106.15928") == ("arxiv", "2106.15928")


def test_normalize_id_arxiv_with_version() -> None:
    assert normalize_id("2106.15928v3") == ("arxiv", "2106.15928v3")


def test_normalize_id_arxiv_old_style() -> None:
    assert normalize_id("hep-th/9901001") == ("arxiv", "hep-th/9901001")


def test_normalize_id_arxiv_prefix_strip() -> None:
    assert normalize_id("arXiv:2106.15928") == ("arxiv", "2106.15928")


def test_normalize_id_arxiv_abs_url() -> None:
    assert normalize_id("https://arxiv.org/abs/2106.15928") == (
        "arxiv",
        "2106.15928",
    )


def test_normalize_id_arxiv_pdf_url_strips_pdf() -> None:
    assert normalize_id("https://arxiv.org/pdf/2106.15928.pdf") == (
        "arxiv",
        "2106.15928",
    )


def test_normalize_id_empty_returns_error() -> None:
    result = normalize_id("   ")
    assert isinstance(result, ToolResult)
    assert result.is_error
    assert "Empty" in result.content


def test_normalize_id_garbage_returns_error() -> None:
    result = normalize_id("not an id at all")
    assert isinstance(result, ToolResult)
    assert result.is_error


def test_s2_wire_id_doi() -> None:
    assert s2_wire_id("doi", "10.1/x") == "DOI:10.1/x"


def test_s2_wire_id_arxiv() -> None:
    assert s2_wire_id("arxiv", "2106.15928") == "ARXIV:2106.15928"


def test_id_slug_doi_replaces_unsafe() -> None:
    assert id_slug("doi", "10.1234/foo bar") == "doi_10.1234_foo_bar"


def test_id_slug_arxiv() -> None:
    assert id_slug("arxiv", "2106.15928") == "arxiv_2106.15928"


def test_id_slug_arxiv_old_style_safe_chars() -> None:
    assert id_slug("arxiv", "hep-th/9901001") == "arxiv_hep-th_9901001"


def test_papers_cache_dir_under_home() -> None:
    p = papers_cache_dir()
    assert p == Path.home() / ".sagent" / "papers"


def _make_record(
    *,
    title: str = "Attention Is All You Need",
    authors: tuple[str, ...] = ("Ashish", "Noam", "Niki", "Jakob"),
    year: int | None = 2017,
    venue: str | None = "NeurIPS",
    doi: str | None = "10.0/abc",
    arxiv_id: str | None = "1706.03762",
    abstract: str | None = "We propose a new architecture.",
    citation_count: int | None = 100_000,
    reference_count: int | None = 50,
    open_access_pdf: str | None = "https://x/pdf",
    sources: tuple[str, ...] = ("s2",),
    is_influential: bool | None = None,
) -> PaperRecord:
    return PaperRecord(
        title=title,
        authors=authors,
        year=year,
        venue=venue,
        doi=doi,
        arxiv_id=arxiv_id,
        abstract=abstract,
        citation_count=citation_count,
        reference_count=reference_count,
        open_access_pdf=open_access_pdf,
        sources=sources,
        is_influential=is_influential,
    )


def test_format_record_full() -> None:
    rec = _make_record()
    out = format_record(rec)
    assert "Attention Is All You Need" in out
    assert "doi:10.0/abc" in out
    assert "arXiv:1706.03762" in out
    assert "Ashish, Noam, Niki +1" in out  # 4 authors, limit=3 default
    assert "cites:100000" in out
    assert "OA" in out
    assert "sources: s2" in out
    assert "abstract:" in out


def test_format_record_truncates_abstract() -> None:
    rec = _make_record(abstract="x" * 100)
    out = format_record(rec, abstract_chars=10)
    assert "xxxxxxxxxx..." in out


def test_format_record_no_authors() -> None:
    rec = _make_record(authors=())
    out = format_record(rec)
    assert "unknown" in out


def test_format_record_no_ids() -> None:
    rec = _make_record(doi=None, arxiv_id=None)
    out = format_record(rec)
    assert "[no-id]" in out


def test_format_record_influential_marker() -> None:
    rec = _make_record(is_influential=True)
    out = format_record(rec)
    assert "influential" in out


def test_format_record_no_meta() -> None:
    rec = _make_record(
        citation_count=None,
        reference_count=None,
        open_access_pdf=None,
        sources=(),
    )
    out = format_record(rec)
    # No meta line trailing the header.
    assert " - cites:" not in out


def test_format_block_full() -> None:
    rec = _make_record()
    out = format_block(rec)
    assert "id: arXiv:1706.03762" in out
    assert "doi: 10.0/abc" in out
    assert "title: Attention Is All You Need" in out
    assert "year: 2017" in out
    assert "venue: NeurIPS" in out
    assert "abstract: We propose" in out


def test_format_block_sparse() -> None:
    rec = PaperRecord(title="Bare", authors=())
    out = format_block(rec)
    assert "title: Bare" in out
    assert "authors: unknown" in out


def test_format_author_line_full() -> None:
    rec = AuthorRecord(
        author_id="42",
        name="Yoshua Bengio",
        h_index=200,
        citation_count=500_000,
        paper_count=800,
        affiliations=("Mila",),
    )
    out = format_author_line(rec)
    assert "[author:42]" in out
    assert "Yoshua Bengio" in out
    assert "h-index:200" in out
    assert "Mila" in out


def test_format_author_line_minimal() -> None:
    rec = AuthorRecord(author_id="7", name="Anon")
    out = format_author_line(rec)
    assert out == "[author:7] Anon"


def test_format_author_block_full() -> None:
    rec = AuthorRecord(
        author_id="9",
        name="A",
        aliases=("Aa", "Aaa"),
        affiliations=("U1", "U2"),
        homepage="https://x",
        h_index=10,
        citation_count=100,
        paper_count=20,
    )
    out = format_author_block(rec)
    assert "author_id: 9" in out
    assert "aliases: Aa, Aaa" in out
    assert "affiliations: U1, U2" in out
    assert "homepage: https://x" in out
    assert "h_index: 10" in out


def test_openalex_reconstruct_basic() -> None:
    inv = {"hello": [0], "world": [1]}
    assert openalex_reconstruct_abstract(inv) == "hello world"


def test_openalex_reconstruct_unordered() -> None:
    inv = {"b": [1], "a": [0], "c": [2]}
    assert openalex_reconstruct_abstract(inv) == "a b c"


def test_openalex_reconstruct_none() -> None:
    assert openalex_reconstruct_abstract(None) is None


def test_openalex_reconstruct_empty() -> None:
    assert openalex_reconstruct_abstract({}) is None


def test_truncation_notice_appended_when_over() -> None:
    assert "showing 5 of 100" in truncation_notice(5, 100)


def test_truncation_notice_empty_when_equal() -> None:
    assert truncation_notice(5, 5) == ""


def test_truncation_notice_empty_when_total_zero() -> None:
    assert truncation_notice(5, 0) == ""


def test_s2_paper_to_record_full() -> None:
    data: MutableJSON = {
        "title": "T",
        "authors": [{"name": "A"}, {"name": "B"}, {}],
        "year": 2020,
        "venue": "V",
        "externalIds": {"DOI": "10.0/x", "ArXiv": "2001.0001"},
        "abstract": "abs",
        "citationCount": 5,
        "referenceCount": 10,
        "openAccessPdf": {"url": "https://oa/x"},
    }
    rec = s2_paper_to_record(data)
    assert rec.title == "T"
    assert rec.authors == ("A", "B")
    assert rec.year == 2020
    assert rec.venue == "V"
    assert rec.doi == "10.0/x"
    assert rec.arxiv_id == "2001.0001"
    assert rec.citation_count == 5
    assert rec.open_access_pdf == "https://oa/x"
    assert rec.sources == ("s2",)
    assert rec.is_influential is None


def test_s2_paper_to_record_sparse() -> None:
    empty: MutableJSON = {}
    rec = s2_paper_to_record(empty)
    assert rec.title == "(untitled)"
    assert rec.authors == ()
    assert rec.doi is None
    assert rec.year is None


def test_s2_paper_to_record_is_influential_set() -> None:
    minimal: MutableJSON = {"title": "X"}
    rec = s2_paper_to_record(minimal, is_influential=True)
    assert rec.is_influential is True


def test_s2_get_success() -> None:
    payload = b'{"data": [{"title": "x"}]}'
    with patch("sagent.tools.paper_common.fetch", return_value=payload):
        data = asyncio.run(s2_get("/paper/search", {"q": "x"}))
    assert not isinstance(data, ToolResult)
    inner = data["data"]
    assert isinstance(inner, list)


def test_s2_get_404_returns_tool_result() -> None:
    err = FetchError(
        url=f"{S2_BASE}/paper/x",
        status=404,
        headers={},
        body=b"Not Found",
    )
    with patch("sagent.tools.paper_common.fetch", side_effect=err):
        result = asyncio.run(s2_get("/paper/x", {}))
    assert isinstance(result, ToolResult)
    assert result.is_error
    assert "Not found" in result.content


def test_s2_get_429_returns_tool_result() -> None:
    err = FetchError(
        url=f"{S2_BASE}/paper/x",
        status=429,
        headers={},
        body=b"Too Many",
    )
    with patch("sagent.tools.paper_common.fetch", side_effect=err):
        result = asyncio.run(s2_get("/paper/x", {}))
    assert isinstance(result, ToolResult)
    assert "rate limit" in result.content.lower()


def test_s2_get_other_http_error() -> None:
    err = FetchError(url=f"{S2_BASE}/p", status=500, headers={}, body=b"server boom")
    with patch("sagent.tools.paper_common.fetch", side_effect=err):
        result = asyncio.run(s2_get("/p", {}))
    assert isinstance(result, ToolResult)
    assert "500" in result.content


def test_s2_get_uses_api_key_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEMANTIC_SCHOLAR_API_KEY", "secret-key")
    captured: dict[str, object] = {}

    def fake_fetch(**kw: object) -> bytes:
        captured.update(kw)
        return b"{}"

    with patch("sagent.tools.paper_common.fetch", side_effect=fake_fetch):
        _ = asyncio.run(s2_get("/p", {}))
    headers_obj = captured["headers"]
    assert isinstance(headers_obj, dict)
    headers = cast(dict[str, str], headers_obj)
    assert headers["x-api-key"] == "secret-key"


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
