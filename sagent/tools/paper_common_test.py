"""Tests for ``tools.paper_common``: sagent-tool adapter helpers.

Backend I/O, record mapping, the S2 client, and rate limiting now live in the
shared paper web library and are tested there. This suite covers only the
sagent-side concerns that remain: argument parsing/validation, identifier-arg
adaptation, and text rendering.
"""

from __future__ import annotations

from wesearch.paper.custom_types import AuthorRecord, PaperRecord

from sagent.lib.userdirs import data_dir
from sagent.tools.paper_common import (
    format_author_block,
    format_author_line,
    format_block,
    format_record,
    normalize_id_arg,
    papers_cache_dir,
    parse_optional_ids,
    resolve_id_args,
    short_id,
    summary_ids,
    truncation_notice,
    validate_abstract_chars,
    validate_limit,
)
from sagent.types.runtime import ToolResult


# ---------------------------------------------------------------------------
# normalize_id_arg -- the sagent-facing wrapper that maps a bad id to a
# ToolResult (the underlying normalize_id is covered by the library's ids_test).
# ---------------------------------------------------------------------------


def test_normalize_id_arg_ok() -> None:
    assert normalize_id_arg("2106.15928") == ("arxiv", "2106.15928")


def test_normalize_id_arg_error_is_tool_result() -> None:
    result = normalize_id_arg("not-an-id")
    assert isinstance(result, ToolResult)
    assert result.is_error


# ---------------------------------------------------------------------------
# short_id / papers_cache_dir
# ---------------------------------------------------------------------------


def test_short_id_short_unchanged() -> None:
    assert short_id("10.1/abc") == "10.1/abc"


def test_short_id_truncates_long() -> None:
    s = "x" * 60
    result = short_id(s)
    assert result.startswith("…")
    assert len(result) == 39


def test_papers_cache_dir_under_data_dir() -> None:
    assert papers_cache_dir() == data_dir("sagent") / "papers"


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


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
    out = format_record(_make_record())
    assert "Attention Is All You Need" in out
    assert "doi:10.0/abc" in out
    assert "arXiv:1706.03762" in out
    assert "Ashish, Noam, Niki +1" in out  # 4 authors, limit=3 default
    assert "cites:100000" in out
    assert "OA" in out
    assert "sources: s2" in out
    assert "abstract:" in out


def test_format_record_truncates_abstract() -> None:
    out = format_record(_make_record(abstract="x" * 100), abstract_chars=10)
    assert "xxxxxxxxxx..." in out


def test_format_record_no_authors() -> None:
    assert "unknown" in format_record(_make_record(authors=()))


def test_format_record_no_ids() -> None:
    assert "[no-id]" in format_record(_make_record(doi=None, arxiv_id=None))


def test_format_record_influential_marker() -> None:
    assert "influential" in format_record(_make_record(is_influential=True))


def test_format_record_no_meta() -> None:
    out = format_record(
        _make_record(
            citation_count=None, reference_count=None, open_access_pdf=None, sources=()
        )
    )
    assert " - cites:" not in out


def test_format_block_full() -> None:
    out = format_block(_make_record())
    assert "id: arXiv:1706.03762" in out
    assert "doi: 10.0/abc" in out
    assert "title: Attention Is All You Need" in out
    assert "year: 2017" in out
    assert "venue: NeurIPS" in out
    assert "abstract: We propose" in out


def test_format_block_sparse() -> None:
    out = format_block(PaperRecord(title="Bare", authors=()))
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
    assert format_author_line(AuthorRecord(author_id="7", name="Anon")) == (
        "[author:7] Anon"
    )


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


def test_truncation_notice_appended_when_over() -> None:
    assert truncation_notice(3, 10) == (
        "\n... (showing 3 of 10; tighten filters for more)"
    )


def test_truncation_notice_empty_when_equal() -> None:
    assert truncation_notice(5, 5) == ""


def test_truncation_notice_empty_when_total_zero() -> None:
    assert truncation_notice(5, 0) == ""


# ---------------------------------------------------------------------------
# Identifier argument parsing
# ---------------------------------------------------------------------------


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


def test_parse_optional_ids_rejects_non_string_list_element() -> None:
    result = parse_optional_ids({"ids": [7, "10.1/x"]})
    assert isinstance(result, ToolResult)
    assert result.is_error


def test_parse_optional_ids_recovers_json_array_string() -> None:
    arg = '["arXiv:2509.04439", "arXiv:2507.12821", "10.1/x"]'
    assert parse_optional_ids({"ids": arg}) == [
        "arXiv:2509.04439",
        "arXiv:2507.12821",
        "10.1/x",
    ]


def test_parse_optional_ids_recovers_json_array_no_spaces() -> None:
    arg = '["arXiv:2509.04439","arXiv:2507.12821"]'
    assert parse_optional_ids({"ids": arg}) == ["arXiv:2509.04439", "arXiv:2507.12821"]


def test_parse_optional_ids_recovers_comma_joined_bundle() -> None:
    arg = "arXiv:2509.04439,arXiv:2507.12821,10.34190/icair.5.1.4311"
    assert parse_optional_ids({"ids": arg}) == [
        "arXiv:2509.04439",
        "arXiv:2507.12821",
        "10.34190/icair.5.1.4311",
    ]


def test_parse_optional_ids_recovers_newline_bundle() -> None:
    arg = "arXiv:2509.04439\narXiv:2507.12821"
    assert parse_optional_ids({"ids": arg}) == ["arXiv:2509.04439", "arXiv:2507.12821"]


def test_parse_optional_ids_single_doi_with_comma_not_split() -> None:
    # A lone DOI containing a comma must NOT be split: the second token is not a
    # valid id, so the ambiguous split is rejected and the id is kept whole.
    assert parse_optional_ids({"ids": "10.1234/foo,bar"}) == ["10.1234/foo,bar"]


def test_parse_optional_ids_malformed_json_array_kept_whole() -> None:
    assert parse_optional_ids({"ids": "[not json"}) == ["[not json"]


def test_parse_optional_ids_single_id_unchanged() -> None:
    assert parse_optional_ids({"ids": "arXiv:2509.04439"}) == ["arXiv:2509.04439"]


def test_parse_optional_ids_list_wrapped_bundle_recovered() -> None:
    # The wire can deliver a bundle as a single LIST ELEMENT, not just a bare
    # string. Recover per-element too, using the author-id predicate.
    assert parse_optional_ids(
        {"ids": ["1741101,2064160"]}, looks_like_id=str.isdigit
    ) == ["1741101", "2064160"]


def test_parse_optional_ids_list_wrapped_json_array_recovered() -> None:
    assert parse_optional_ids(
        {"ids": ['["arXiv:2509.04439", "arXiv:2507.12821"]']}
    ) == ["arXiv:2509.04439", "arXiv:2507.12821"]


def test_parse_optional_ids_normal_list_elements_unsplit() -> None:
    assert parse_optional_ids({"ids": ["10.1/a", "10.1/b"]}) == ["10.1/a", "10.1/b"]


def test_resolve_id_args_coerces_bare_string() -> None:
    assert resolve_id_args({"ids": "10.1/x"}) == ["10.1/x"]


def test_resolve_id_args_missing_is_error() -> None:
    result = resolve_id_args({})
    assert isinstance(result, ToolResult)
    assert result.is_error


def test_resolve_id_args_empty_is_error() -> None:
    result = resolve_id_args({"ids": []})
    assert isinstance(result, ToolResult)
    assert result.is_error


# ---------------------------------------------------------------------------
# summary_ids / validators
# ---------------------------------------------------------------------------


def test_summary_ids_bare_string() -> None:
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


def test_validate_abstract_chars_rejects_non_positive() -> None:
    assert isinstance(validate_abstract_chars(0), ToolResult)
    assert isinstance(validate_abstract_chars(-1), ToolResult)


def test_validate_abstract_chars_passes_none_and_positive() -> None:
    assert validate_abstract_chars(None) is None
    assert validate_abstract_chars(200) == 200


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
