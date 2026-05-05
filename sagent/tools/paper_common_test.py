"""Tests for tools.paper_common."""

from __future__ import annotations

import pytest

from sagent.custom_types import MessageBase
from sagent.tools.paper_common import (
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
    s2_wire_id,
    truncation_notice,
)


class TestNormalizeId:
    def test_bare_doi(self) -> None:
        assert normalize_id("10.1145/3197517") == ("doi", "10.1145/3197517")

    def test_doi_https_prefix(self) -> None:
        assert normalize_id("https://doi.org/10.1145/3197517") == (
            "doi",
            "10.1145/3197517",
        )

    def test_doi_dx_prefix(self) -> None:
        assert normalize_id("https://dx.doi.org/10.1145/3197517") == (
            "doi",
            "10.1145/3197517",
        )

    def test_doi_scheme_prefix(self) -> None:
        assert normalize_id("doi:10.1145/3197517") == (
            "doi",
            "10.1145/3197517",
        )

    def test_doi_with_complex_suffix(self) -> None:
        # DOI suffix can contain punctuation, slashes, colons, etc.
        assert normalize_id("10.1109/TPAMI.2016.2577031") == (
            "doi",
            "10.1109/TPAMI.2016.2577031",
        )

    def test_arxiv_bare(self) -> None:
        assert normalize_id("2106.15928") == ("arxiv", "2106.15928")

    def test_arxiv_with_version(self) -> None:
        assert normalize_id("2106.15928v2") == ("arxiv", "2106.15928v2")

    def test_arxiv_scheme_prefix(self) -> None:
        assert normalize_id("arXiv:2106.15928") == ("arxiv", "2106.15928")

    def test_arxiv_lowercase_scheme(self) -> None:
        assert normalize_id("arxiv:2106.15928") == ("arxiv", "2106.15928")

    def test_arxiv_abs_url(self) -> None:
        assert normalize_id("https://arxiv.org/abs/2106.15928") == (
            "arxiv",
            "2106.15928",
        )

    def test_arxiv_pdf_url(self) -> None:
        assert normalize_id("https://arxiv.org/pdf/2106.15928.pdf") == (
            "arxiv",
            "2106.15928",
        )

    def test_arxiv_legacy_old_style(self) -> None:
        assert normalize_id("hep-th/9901001") == ("arxiv", "hep-th/9901001")

    def test_arxiv_legacy_with_subsection(self) -> None:
        assert normalize_id("math.AG/9901001") == (
            "arxiv",
            "math.AG/9901001",
        )

    def test_whitespace_trimmed(self) -> None:
        assert normalize_id("  10.1145/3197517  ") == (
            "doi",
            "10.1145/3197517",
        )

    def test_empty_rejected(self) -> None:
        result = normalize_id("")
        assert isinstance(result, MessageBase)
        assert result.descriptor == "text/x-error"
        assert "Empty" in str(result.content)

    def test_whitespace_only_rejected(self) -> None:
        result = normalize_id("   ")
        assert isinstance(result, MessageBase)
        assert result.descriptor == "text/x-error"
        assert "Empty" in str(result.content)

    def test_pmid_rejected(self) -> None:
        result = normalize_id("31395057")
        assert isinstance(result, MessageBase)
        assert result.descriptor == "text/x-error"
        assert "Unrecognized" in str(result.content)

    def test_nonsense_rejected(self) -> None:
        result = normalize_id("not-an-id")
        assert isinstance(result, MessageBase)
        assert result.descriptor == "text/x-error"
        assert "Unrecognized" in str(result.content)


class TestS2WireId:
    def test_doi(self) -> None:
        assert s2_wire_id("doi", "10.1145/3197517") == "DOI:10.1145/3197517"

    def test_arxiv(self) -> None:
        assert s2_wire_id("arxiv", "2106.15928") == "ARXIV:2106.15928"


class TestIdSlug:
    def test_doi_sanitized(self) -> None:
        slug = id_slug("doi", "10.1145/3197517.3201367")
        assert slug.startswith("doi_")
        assert "/" not in slug
        assert "." in slug  # dots are allowed

    def test_arxiv(self) -> None:
        slug = id_slug("arxiv", "2106.15928")
        assert slug == "arxiv_2106.15928"

    def test_arxiv_old_style(self) -> None:
        slug = id_slug("arxiv", "hep-th/9901001")
        assert "/" not in slug
        assert slug.startswith("arxiv_hep-th")


class TestPapersCacheDir:
    def test_under_home(self) -> None:
        p = papers_cache_dir()
        assert str(p).endswith(".sagent/papers")


class TestOpenalexAbstract:
    def test_basic(self) -> None:
        inv = {"Hello": [0, 2], "world": [1, 3]}
        assert openalex_reconstruct_abstract(inv) == "Hello world Hello world"

    def test_none(self) -> None:
        assert openalex_reconstruct_abstract(None) is None

    def test_empty(self) -> None:
        assert openalex_reconstruct_abstract({}) is None


class TestTruncationNotice:
    def test_fits(self) -> None:
        assert truncation_notice(10, 10) == ""

    def test_truncated(self) -> None:
        msg = truncation_notice(100, 4521)
        assert "100" in msg
        assert "4521" in msg

    def test_no_total(self) -> None:
        # total=0 means "unknown" - don't emit.
        assert truncation_notice(100, 0) == ""


class TestFormatRecord:
    def test_full(self) -> None:
        rec = PaperRecord(
            title="Attention Is All You Need",
            authors=("Vaswani", "Shazeer", "Parmar"),
            year=2017,
            venue="NeurIPS",
            doi="10.5555/3295222",
            arxiv_id="1706.03762",
            abstract="The dominant sequence transduction models...",
            citation_count=89_431,
            reference_count=42,
            open_access_pdf="https://arxiv.org/pdf/1706.03762",
            sources=("s2",),
        )
        out = format_record(rec)
        assert "[doi:10.5555/3295222 | arXiv:1706.03762]" in out
        assert "Attention Is All You Need" in out
        assert "Vaswani, Shazeer, Parmar" in out
        assert "2017, NeurIPS" in out
        assert "cites:89431" in out
        assert "refs:42" in out
        assert "OA" in out
        assert "sources: s2" in out
        assert "The dominant sequence transduction models" in out

    def test_author_truncation(self) -> None:
        rec = PaperRecord(
            title="T",
            authors=("A", "B", "C", "D", "E", "F", "G", "H"),
        )
        out = format_record(rec)
        assert "A, B, C +5" in out

    def test_abstract_cap(self) -> None:
        rec = PaperRecord(title="T", abstract="x" * 1000)
        out = format_record(rec, abstract_chars=50)
        # capped length + ellipsis tail
        assert "..." in out
        # Find the abstract block line prefixes and verify length
        assert out.count("x") < 1000

    def test_influential_tag(self) -> None:
        rec = PaperRecord(title="T", is_influential=True)
        assert "influential" in format_record(rec)

    def test_non_influential_no_tag(self) -> None:
        rec = PaperRecord(title="T", is_influential=False)
        assert "influential" not in format_record(rec)

    def test_sources_multiple(self) -> None:
        rec = PaperRecord(title="T", sources=("s2", "openalex"))
        assert "sources: s2,openalex" in format_record(rec)

    def test_no_abstract(self) -> None:
        rec = PaperRecord(title="T")
        out = format_record(rec)
        assert "abstract:" not in out


class TestFormatBlock:
    def test_rendering(self) -> None:
        rec = PaperRecord(
            title="T",
            authors=("A", "B"),
            year=2020,
            venue="V",
            doi="10.x/y",
            arxiv_id="2001.00001",
            abstract="abs",
            citation_count=5,
            reference_count=3,
            open_access_pdf="https://x/y",
            sources=("s2",),
        )
        out = format_block(rec)
        assert "id: arXiv:2001.00001" in out
        assert "doi: 10.x/y" in out
        assert "title: T" in out
        assert "authors: A, B" in out
        assert "year: 2020" in out
        assert "venue: V" in out
        assert "citation_count: 5" in out
        assert "reference_count: 3" in out
        assert "open_access_pdf: https://x/y" in out
        assert "sources: s2" in out
        assert "abstract: abs" in out

    def test_sparse(self) -> None:
        rec = PaperRecord(title="T")
        out = format_block(rec)
        assert "title: T" in out
        assert "authors: unknown" in out
        assert "doi:" not in out  # omitted when missing


class TestFormatAuthorLine:
    def test_full(self) -> None:
        rec = AuthorRecord(
            author_id="1741101",
            name="Yoshua Bengio",
            affiliations=("Mila", "Université de Montréal"),
            h_index=245,
            citation_count=500_000,
            paper_count=800,
        )
        out = format_author_line(rec)
        assert "[author:1741101]" in out
        assert "Yoshua Bengio" in out
        assert "h-index:245" in out
        assert "cites:500000" in out
        assert "papers:800" in out
        # Primary affiliation only.
        assert "Mila" in out
        assert "Université de Montréal" not in out

    def test_sparse(self) -> None:
        rec = AuthorRecord(author_id="1", name="Alice")
        out = format_author_line(rec)
        assert "[author:1]" in out
        assert "Alice" in out

    def test_no_metrics_no_dash(self) -> None:
        rec = AuthorRecord(author_id="1", name="Alice")
        out = format_author_line(rec)
        assert "h-index" not in out


class TestFormatAuthorBlock:
    def test_full(self) -> None:
        rec = AuthorRecord(
            author_id="1741101",
            name="Yoshua Bengio",
            aliases=("Y. Bengio", "Yoshua Bengio"),
            affiliations=("Mila", "Université de Montréal"),
            homepage="https://yoshuabengio.org",
            h_index=245,
            citation_count=500_000,
            paper_count=800,
        )
        out = format_author_block(rec)
        assert "author_id: 1741101" in out
        assert "name: Yoshua Bengio" in out
        assert "aliases: Y. Bengio, Yoshua Bengio" in out
        assert "affiliations: Mila, Université de Montréal" in out
        assert "homepage: https://yoshuabengio.org" in out
        assert "h_index: 245" in out
        assert "citation_count: 500000" in out
        assert "paper_count: 800" in out

    def test_sparse(self) -> None:
        rec = AuthorRecord(author_id="1", name="Alice")
        out = format_author_block(rec)
        assert "author_id: 1" in out
        assert "name: Alice" in out
        # Empty fields are omitted - no "aliases:" line.
        assert "aliases:" not in out
        assert "homepage:" not in out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
