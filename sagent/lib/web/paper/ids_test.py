"""Tests for sagent.lib.web.paper.ids."""

from __future__ import annotations

import pytest

from sagent.lib.web.paper.errors import InvalidIdError
from sagent.lib.web.paper.ids import (
    id_slug,
    looks_like_paper_id,
    normalize_id,
    s2_wire_id,
)


class TestNormalizeId:
    def test_doi_bare(self) -> None:
        assert normalize_id("10.1234/foo") == ("doi", "10.1234/foo")

    def test_doi_https_prefix(self) -> None:
        assert normalize_id("https://doi.org/10.1234/foo") == ("doi", "10.1234/foo")

    def test_doi_dx_prefix(self) -> None:
        assert normalize_id("https://dx.doi.org/10.1234/x") == ("doi", "10.1234/x")

    def test_doi_short_prefix(self) -> None:
        assert normalize_id("doi:10.1234/x") == ("doi", "10.1234/x")

    def test_arxiv_new_style(self) -> None:
        assert normalize_id("2106.15928") == ("arxiv", "2106.15928")

    def test_arxiv_with_version(self) -> None:
        assert normalize_id("2106.15928v3") == ("arxiv", "2106.15928v3")

    def test_arxiv_old_style(self) -> None:
        assert normalize_id("hep-th/9901001") == ("arxiv", "hep-th/9901001")

    def test_arxiv_prefix_strip(self) -> None:
        assert normalize_id("arXiv:2106.15928") == ("arxiv", "2106.15928")

    def test_arxiv_abs_url(self) -> None:
        assert normalize_id("https://arxiv.org/abs/2106.15928") == (
            "arxiv",
            "2106.15928",
        )

    def test_arxiv_pdf_url_strips_pdf(self) -> None:
        assert normalize_id("https://arxiv.org/pdf/2106.15928.pdf") == (
            "arxiv",
            "2106.15928",
        )

    def test_arxiv_prefix_garbage_rejected(self) -> None:
        # A prefix-stripped value must still match the arXiv shape.
        with pytest.raises(InvalidIdError):
            normalize_id("arxiv:not-an-id")

    def test_doi_prefix_pins_family(self) -> None:
        # A ``doi:``-prefixed arXiv-shaped value must not be re-read as arXiv.
        with pytest.raises(InvalidIdError):
            normalize_id("doi:2106.15928")

    def test_empty_rejected(self) -> None:
        with pytest.raises(InvalidIdError):
            normalize_id("   ")

    def test_garbage_rejected(self) -> None:
        with pytest.raises(InvalidIdError):
            normalize_id("not-an-identifier")


class TestLooksLikePaperId:
    def test_true_for_doi(self) -> None:
        assert looks_like_paper_id("10.1234/foo")

    def test_true_for_arxiv(self) -> None:
        assert looks_like_paper_id("2106.15928")

    def test_false_for_garbage(self) -> None:
        assert not looks_like_paper_id("nonsense")


class TestWireAndSlug:
    def test_wire_doi(self) -> None:
        assert s2_wire_id("doi", "10.1/x") == "DOI:10.1/x"

    def test_wire_arxiv(self) -> None:
        assert s2_wire_id("arxiv", "2106.15928") == "ARXIV:2106.15928"

    def test_slug_sanitizes(self) -> None:
        assert id_slug("doi", "10.1/a b") == "doi_10.1_a_b"

    def test_slug_arxiv(self) -> None:
        assert id_slug("arxiv", "hep-th/9901001") == "arxiv_hep-th_9901001"


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
