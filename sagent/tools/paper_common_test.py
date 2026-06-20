"""Tests for ``tools.paper_common``: shared helpers for Paper* tools."""

from __future__ import annotations

from pathlib import Path
from typing import cast
from unittest.mock import patch

import asyncio
import json

import pytest

from sagent.lib.json import MutableJSON
from sagent.lib.ratelimit import FileStore, TokenBucketRateLimiter
from sagent.lib.web.fetch import FetchError
from sagent.tools import paper_common
from sagent.tools.paper_common import (
    S2_BASE,
    AuthorRecord,
    PaperRecord,
    format_author_block,
    format_author_line,
    format_block,
    format_record,
    id_slug,
    normalize_id,
    openalex_reconstruct_abstract,
    papers_cache_dir,
    parse_optional_ids,
    resolve_id_args,
    s2_batch,
    s2_get,
    s2_paginate,
    s2_paper_to_record,
    s2_wire_id,
    short_id,
    summary_ids,
    truncation_notice,
    validate_limit,
    year_in_range,
)
from sagent.types.runtime import ToolResult


@pytest.fixture(autouse=True)
def _neutralize_rate_gate(  # pyright: ignore[reportUnusedFunction] -- autouse fixture
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Point the module S2 gate at a fresh tmp-backed limiter, no waiting.

    Keeps ``s2_get`` tests hermetic and instant: a per-test lockfile (no
    cross-test coupling) and a stubbed sleep so a drained bucket never
    blocks the suite.
    """
    gate = TokenBucketRateLimiter(
        max_calls=1,
        per_seconds=1.0,
        store=FileStore(tmp_path / "s2.lock"),
    )

    async def _no_wait() -> None:
        return None

    monkeypatch.setattr(gate, "acquire_async", _no_wait)
    monkeypatch.setattr(paper_common, "_s2_gate", lambda: gate)


def test_short_id_short_unchanged() -> None:
    assert short_id("10.1/abc") == "10.1/abc"


def test_short_id_truncates_long() -> None:
    s = "x" * 60
    result = short_id(s)
    assert result.startswith("…")
    assert len(result) == 39


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


def test_normalize_id_arxiv_prefix_garbage_rejected() -> None:
    # A prefix-stripped value must still match the arXiv shape, not pass
    # through unvalidated.
    result = normalize_id("arxiv:not-an-id")
    assert isinstance(result, ToolResult)
    assert result.is_error


def test_normalize_id_prefix_pins_family() -> None:
    # A ``doi:``-prefixed arXiv-shaped value must not be re-read as arXiv,
    # and vice versa -- the prefix pins the family.
    assert isinstance(normalize_id("doi:2106.15928"), ToolResult)
    assert isinstance(normalize_id("arxiv:10.1145/foo"), ToolResult)


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


def test_s2_batch_aligns_results_with_input_ids() -> None:
    # S2 returns an array positionally aligned with the request, null per miss.
    payload = b'[{"title": "A"}, null, {"title": "C"}]'
    with patch("sagent.tools.paper_common.fetch", return_value=payload):
        out = asyncio.run(s2_batch(["DOI:a", "DOI:b", "DOI:c"], "title"))
    assert not isinstance(out, ToolResult)
    assert len(out) == 3
    first = out[0]
    assert first is not None
    assert first["title"] == "A"
    assert out[1] is None
    third = out[2]
    assert third is not None
    assert third["title"] == "C"


def test_s2_batch_empty_returns_empty_list() -> None:
    out = asyncio.run(s2_batch([], "title"))
    assert out == []


def test_s2_batch_posts_json_body() -> None:
    captured: dict[str, object] = {}

    def fake_fetch(**kw: object) -> bytes:
        captured.update(kw)
        return b"[{}]"

    with patch("sagent.tools.paper_common.fetch", side_effect=fake_fetch):
        _ = asyncio.run(s2_batch(["DOI:a"], "title"))
    assert captured["method"] == "POST"
    assert captured["json"] == {"ids": ["DOI:a"]}


def test_s2_paginate_walks_cursor_until_limit_matches() -> None:
    # Two pages; the filter keeps only even ids. Paginate must follow `next`
    # and gather `limit` matches that span both pages, not stop at page 1.
    pages = [
        json.dumps(
            {"offset": 0, "next": 3, "data": [{"n": 1}, {"n": 2}, {"n": 3}]}
        ).encode(),
        json.dumps({"offset": 3, "data": [{"n": 4}, {"n": 6}]}).encode(),
    ]
    calls = {"i": 0}

    def fake_fetch(**kw: object) -> bytes:
        del kw
        body = pages[calls["i"]]
        calls["i"] += 1
        return body

    with patch("sagent.tools.paper_common.fetch", side_effect=fake_fetch):
        page = asyncio.run(
            s2_paginate("/x", {}, limit=3, keep=lambda e: cast(int, e["n"]) % 2 == 0)
        )
    assert not isinstance(page, ToolResult)
    assert calls["i"] == 2  # walked both pages
    assert [cast(int, e["n"]) for e in page.entries] == [2, 4, 6]
    assert page.complete  # cursor exhausted (page 2 had no `next`)


def test_s2_paginate_stops_at_limit_marks_incomplete() -> None:
    # First page already yields more than `limit` matches; must stop and
    # report incomplete (more may exist) without fetching further.
    page1 = json.dumps(
        {"offset": 0, "next": 3, "data": [{"n": 1}, {"n": 2}, {"n": 3}]}
    ).encode()
    calls = {"i": 0}

    def fake_fetch(**kw: object) -> bytes:
        del kw
        calls["i"] += 1
        return page1

    with patch("sagent.tools.paper_common.fetch", side_effect=fake_fetch):
        page = asyncio.run(s2_paginate("/x", {}, limit=2))
    assert not isinstance(page, ToolResult)
    assert calls["i"] == 1
    assert len(page.entries) == 2
    assert not page.complete


def test_s2_paginate_trims_exhausted_final_page_to_limit() -> None:
    # PAG-002: an exhausted page (no `next`) with MORE rows than `limit`
    # must still trim to `limit` -- the over-delivery bug.
    payload = b'{"data": [{"n": 1}, {"n": 2}, {"n": 3}]}'
    with patch("sagent.tools.paper_common.fetch", return_value=payload):
        page = asyncio.run(s2_paginate("/x", {}, limit=2))
    assert not isinstance(page, ToolResult)
    assert [cast(int, e["n"]) for e in page.entries] == [1, 2]
    assert page.complete  # cursor exhausted


def test_s2_paginate_limit_none_incomplete_when_next_exists() -> None:
    # PAG-001: limit=None fetches one page; if S2 sent a `next` cursor there
    # is more, so `complete` must be False (no silent truncation).
    payload = b'{"next": 1000, "data": [{"n": 1}]}'
    with patch("sagent.tools.paper_common.fetch", return_value=payload):
        page = asyncio.run(s2_paginate("/x", {}, limit=None))
    assert not isinstance(page, ToolResult)
    assert not page.complete


def test_s2_paginate_limit_none_complete_when_exhausted() -> None:
    payload = b'{"data": [{"n": 1}]}'  # no `next` -> exhausted
    with patch("sagent.tools.paper_common.fetch", return_value=payload):
        page = asyncio.run(s2_paginate("/x", {}, limit=None))
    assert not isinstance(page, ToolResult)
    assert page.complete


def test_s2_paginate_deep_page_error_stops_gracefully() -> None:
    # When S2 refuses a deeper page (its own depth ceiling) AFTER we have
    # matches, stop with complete=False -- rely on S2's error, not a mirrored
    # offset constant.
    calls = {"i": 0}

    def fake_fetch(**kw: object) -> bytes:
        del kw
        calls["i"] += 1
        if calls["i"] == 1:
            return b'{"offset": 0, "next": 9000, "data": [{"n": 1}, {"n": 2}]}'
        raise FetchError(
            url="u", status=400, headers={}, body=b"offset + limit must be < 10000"
        )

    with patch("sagent.tools.paper_common.fetch", side_effect=fake_fetch):
        page = asyncio.run(s2_paginate("/x", {}, limit=1_000_000))
    assert not isinstance(page, ToolResult)
    assert len(page.entries) == 2
    assert not page.complete  # S2 said no more; honest partial


def test_s2_paginate_404_message_matches_s2_get() -> None:
    # D1: one shared error renderer -> s2_get and s2_paginate report a 404
    # with the same "Not found" wording, not divergent messages.
    err = FetchError(url="u", status=404, headers={}, body=b"")
    with patch("sagent.tools.paper_common.fetch", side_effect=err):
        via_get = asyncio.run(s2_get("/p", {}))
        via_page = asyncio.run(s2_paginate("/p", {}, limit=10))
    assert isinstance(via_get, ToolResult)
    assert isinstance(via_page, ToolResult)
    assert "Not found" in via_get.content
    assert "Not found" in via_page.content


def test_s2_paginate_transient_error_mid_walk_surfaces() -> None:
    # A non-400 error on a later page (e.g. 500) is a real failure, NOT the
    # depth ceiling -- it must surface, not masquerade as a graceful partial.
    calls = {"i": 0}

    def fake_fetch(**kw: object) -> bytes:
        del kw
        calls["i"] += 1
        if calls["i"] == 1:
            return b'{"offset": 0, "next": 1000, "data": [{"n": 1}]}'
        raise FetchError(url="u", status=500, headers={}, body=b"server boom")

    with patch("sagent.tools.paper_common.fetch", side_effect=fake_fetch):
        result = asyncio.run(s2_paginate("/x", {}, limit=1_000_000))
    assert isinstance(result, ToolResult)
    assert result.is_error
    assert "500" in result.content


def test_s2_paginate_first_page_error_surfaces() -> None:
    # An error on the FIRST page (no matches yet) is a real failure, surfaced.
    err = FetchError(url="u", status=400, headers={}, body=b"bad request")
    with patch("sagent.tools.paper_common.fetch", side_effect=err):
        result = asyncio.run(s2_paginate("/x", {}, limit=10))
    assert isinstance(result, ToolResult)
    assert result.is_error


def test_s2_paginate_non_advancing_cursor_terminates() -> None:
    # A server regression where `next` does not advance must not loop forever.
    calls = {"i": 0}

    def fake_fetch(**kw: object) -> bytes:
        del kw
        calls["i"] += 1
        return b'{"offset": 5, "next": 5, "data": [{"n": 1}]}'  # next == offset

    with patch("sagent.tools.paper_common.fetch", side_effect=fake_fetch):
        page = asyncio.run(
            s2_paginate("/x", {}, limit=1_000_000, keep=lambda _e: False)
        )
    assert not isinstance(page, ToolResult)
    assert not page.complete
    assert calls["i"] <= 3  # bailed on non-advancing cursor, did not spin


def test_parse_optional_ids_coerces_bare_string() -> None:
    assert parse_optional_ids({"ids": "10.1/x"}) == ["10.1/x"]


def test_parse_optional_ids_strips_list() -> None:
    assert parse_optional_ids({"ids": [" a ", "", "b"]}) == ["a", "b"]


def test_parse_optional_ids_absent_is_empty() -> None:
    assert parse_optional_ids({}) == []


def test_parse_optional_ids_rejects_non_string_scalar() -> None:
    result = parse_optional_ids({"ids": 7})
    assert isinstance(result, ToolResult)
    assert result.is_error


def test_parse_optional_ids_recovers_json_array_string() -> None:
    # The wire coerces a union-typed `ids` array to a STRING; recover it.
    arg = '["arXiv:2509.04439", "arXiv:2507.12821", "10.1/x"]'
    assert parse_optional_ids({"ids": arg}) == [
        "arXiv:2509.04439",
        "arXiv:2507.12821",
        "10.1/x",
    ]


def test_parse_optional_ids_recovers_json_array_no_spaces() -> None:
    arg = '["arXiv:2509.04439","arXiv:2507.12821"]'
    assert parse_optional_ids({"ids": arg}) == [
        "arXiv:2509.04439",
        "arXiv:2507.12821",
    ]


def test_parse_optional_ids_recovers_comma_joined_bundle() -> None:
    arg = "arXiv:2509.04439,arXiv:2507.12821,10.34190/icair.5.1.4311"
    assert parse_optional_ids({"ids": arg}) == [
        "arXiv:2509.04439",
        "arXiv:2507.12821",
        "10.34190/icair.5.1.4311",
    ]


def test_parse_optional_ids_recovers_newline_bundle() -> None:
    arg = "arXiv:2509.04439\narXiv:2507.12821"
    assert parse_optional_ids({"ids": arg}) == [
        "arXiv:2509.04439",
        "arXiv:2507.12821",
    ]


def test_parse_optional_ids_single_doi_with_comma_not_split() -> None:
    # A lone DOI containing a comma must NOT be split: the second token is
    # not a valid id, so the ambiguous split is rejected and the id is kept
    # whole (passed through for normalize_id to judge).
    assert parse_optional_ids({"ids": "10.1234/foo,bar"}) == ["10.1234/foo,bar"]


def test_parse_optional_ids_malformed_json_array_kept_whole() -> None:
    # Looks array-ish but isn't valid JSON: don't silently drop it, keep it
    # as one token so the caller surfaces a clear shape error.
    assert parse_optional_ids({"ids": "[not json"}) == ["[not json"]


def test_parse_optional_ids_single_id_unchanged() -> None:
    assert parse_optional_ids({"ids": "arXiv:2509.04439"}) == ["arXiv:2509.04439"]


def test_resolve_id_args_coerces_bare_string() -> None:
    assert resolve_id_args({"ids": "10.1/x"}) == ["10.1/x"]


def test_summary_ids_bare_string() -> None:
    """A bare-string id must label like a single-element list, not '?'."""
    assert summary_ids({"ids": "10.1/x"}) == "10.1/x"


def test_summary_ids_single_list() -> None:
    assert summary_ids({"ids": ["10.1/x"]}) == "10.1/x"


def test_summary_ids_multiple_appends_count() -> None:
    assert summary_ids({"ids": ["10.1/a", "10.1/b", "10.1/c"]}) == "10.1/a (+2 more)"


def test_summary_ids_absent_is_question_mark() -> None:
    assert summary_ids({}) == "?"


def test_validate_limit_rejects_non_positive() -> None:
    assert isinstance(validate_limit(0), ToolResult)
    assert isinstance(validate_limit(-3), ToolResult)


def test_validate_limit_passes_none_and_positive() -> None:
    assert validate_limit(None) is None
    assert validate_limit(5) == 5


def test_year_in_range() -> None:
    assert year_in_range(2021, year_from=2020, year_to=2022)
    assert not year_in_range(2019, year_from=2020, year_to=None)
    assert not year_in_range(2023, year_from=None, year_to=2022)
    assert year_in_range(2021, year_from=None, year_to=None)
    # Non-int / missing year is out of range (undated works excluded).
    assert not year_in_range(None, year_from=2020, year_to=None)
    assert not year_in_range("2021", year_from=None, year_to=None)


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
