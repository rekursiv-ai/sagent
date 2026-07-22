"""Tests for sagent.lib.web.paper.fetch (the PDF download cascade)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from sagent.lib.custom_json import MutableJSON
from sagent.lib.web.errors import FetchError
from sagent.lib.web.fetch import FetchSession
from sagent.lib.web.paper import fetch as fetch_mod
from sagent.lib.web.paper.errors import NotFoundError
from sagent.lib.web.paper.providers import s2


_PDF = b"%PDF-1.5" + b"0" * 200
"""A body that passes ``looks_like_pdf`` (>=128 bytes, ``%PDF-`` magic)."""

_HTML = b"<html><body>not a pdf</body></html>"
"""A body that fails ``looks_like_pdf``."""


def _fetch_error(status: int) -> FetchError:
    return FetchError("http://x", status, {}, b"err")


class TestLooksLikePdf:
    def test_valid_magic_and_size(self) -> None:
        assert fetch_mod.looks_like_pdf(_PDF)

    def test_too_short_rejected(self) -> None:
        assert not fetch_mod.looks_like_pdf(b"%PDF-")

    def test_wrong_magic_rejected(self) -> None:
        assert not fetch_mod.looks_like_pdf(b"<html>" + b"0" * 200)

    def test_custom_magic_shorter_than_default(self) -> None:
        # A caller-supplied magic shorter than the 5-byte default must compare
        # against len(magic) bytes, not a hardcoded [:5] slice.
        assert fetch_mod.looks_like_pdf(b"%PDF" + b"0" * 200, pdf_magic=b"%PDF")


class TestOaUrlOf:
    def test_present_returns_url(self) -> None:
        paper: MutableJSON = {"openAccessPdf": {"url": "http://x/pdf"}}
        assert fetch_mod.oa_url_of(paper) == "http://x/pdf"

    def test_missing_returns_none(self) -> None:
        paper: MutableJSON = {"title": "no oa"}
        assert fetch_mod.oa_url_of(paper) is None

    def test_empty_url_returns_none(self) -> None:
        paper: MutableJSON = {"openAccessPdf": {"url": ""}}
        assert fetch_mod.oa_url_of(paper) is None


class TestBatchOaUrls:
    def test_aligns_urls_and_nones(self) -> None:
        papers: list[MutableJSON | None] = [
            {"openAccessPdf": {"url": "http://a/pdf"}},
            None,
            {"openAccessPdf": {"url": "http://c/pdf"}},
        ]
        with patch.object(s2, "batch", return_value=papers):
            out = fetch_mod.batch_oa_urls(["DOI:1", "DOI:2", "DOI:3"])
        assert out == ["http://a/pdf", None, "http://c/pdf"]

    def test_batch_failure_returns_none(self) -> None:
        with patch.object(s2, "batch", side_effect=RuntimeError("down")):
            assert fetch_mod.batch_oa_urls(["DOI:1"]) is None


class TestDownload:
    def test_arxiv_success(self) -> None:
        with patch.object(fetch_mod, "fetch", return_value=(_PDF, FetchSession())):
            body, source = fetch_mod.download("arxiv", "1706.03762")
        assert body == _PDF
        assert source == "arxiv"

    def test_arxiv_fails_then_open_access(self) -> None:
        # arXiv GET returns non-PDF (rejected); the supplied OA URL then wins.
        with patch.object(
            fetch_mod,
            "fetch",
            side_effect=[(_HTML, FetchSession()), (_PDF, FetchSession())],
        ):
            body, source = fetch_mod.download(
                "arxiv", "1706.03762", oa_url="http://oa/pdf", oa_looked_up=True
            )
        assert body == _PDF
        assert source == "open_access"

    def test_open_access_via_s2_lookup(self) -> None:
        # No oa_url pre-supplied and not looked up -> _s2_oa_lookup fires.
        s2_get = MagicMock(return_value={"openAccessPdf": {"url": "http://oa/pdf"}})
        with (
            patch.object(s2, "get", s2_get),
            patch.object(
                fetch_mod,
                "fetch",
                side_effect=[(_HTML, FetchSession()), (_PDF, FetchSession())],
            ),
        ):
            body, source = fetch_mod.download("arxiv", "1706.03762")
        assert source == "open_access"
        assert body == _PDF
        s2_get.assert_called_once()

    def test_batched_oa_url_skips_per_id_lookup(self) -> None:
        s2_get = MagicMock()
        with (
            patch.object(s2, "get", s2_get),
            patch.object(fetch_mod, "fetch", return_value=(_PDF, FetchSession())),
        ):
            body, source = fetch_mod.download(
                "doi", "10.1/x", oa_url="http://oa/pdf", oa_looked_up=True
            )
        assert source == "open_access"
        assert body == _PDF
        s2_get.assert_not_called()

    def test_looked_up_none_skips_lookup(self) -> None:
        # oa_looked_up with oa_url=None must not re-query S2 per id.
        s2_get = MagicMock()
        with (
            patch.object(s2, "get", s2_get),
            patch.object(fetch_mod, "fetch", side_effect=[(_HTML, FetchSession())]),
            pytest.raises(NotFoundError),
        ):
            fetch_mod.download("arxiv", "1706.03762", oa_url=None, oa_looked_up=True)
        s2_get.assert_not_called()

    def test_all_sources_fail_raises_not_found(self) -> None:
        with (
            patch.object(fetch_mod, "fetch", side_effect=_fetch_error(404)),
            pytest.raises(NotFoundError),
        ):
            fetch_mod.download("doi", "10.1/x", oa_url=None, oa_looked_up=True)

    def test_doi_kind_skips_arxiv_branch(self) -> None:
        # A doi kind never hits the arXiv PDF base; OA supplies the bytes.
        fetch_fn = MagicMock(return_value=(_PDF, FetchSession()))
        with patch.object(fetch_mod, "fetch", fetch_fn):
            body, source = fetch_mod.download(
                "doi", "10.1/x", oa_url="http://oa/pdf", oa_looked_up=True
            )
        assert source == "open_access"
        assert body == _PDF
        called = fetch_fn.call_args_list[0].args[0]
        assert "arxiv.org" not in called

    def test_oa_download_failure_returns_none(self) -> None:
        # OA URL present but GET raises -> _fetch_open_access swallows, None.
        with (
            patch.object(fetch_mod, "fetch", side_effect=OSError("boom")),
            pytest.raises(NotFoundError),
        ):
            fetch_mod.download(
                "doi", "10.1/x", oa_url="http://oa/pdf", oa_looked_up=True
            )

    def test_s2_lookup_exception_returns_none(self) -> None:
        # _s2_oa_lookup swallows any backend error -> no OA URL found.
        with (
            patch.object(s2, "get", side_effect=RuntimeError("down")),
            patch.object(fetch_mod, "fetch", side_effect=_fetch_error(404)),
            pytest.raises(NotFoundError),
        ):
            fetch_mod.download("arxiv", "1706.03762")


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
