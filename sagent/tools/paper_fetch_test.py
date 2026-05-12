"""Tests for ``tools.paper_fetch``: PDF download cascade."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import asyncio
import json

from sagent.lib.web.fetch import FetchError
from sagent.tools.paper_fetch import (
    _MIN_PDF_BYTES,
    PaperFetch,
    _looks_like_pdf,
    _s2_oa_lookup,
)


# Minimal stub of a PDF: starts with magic + padding > _MIN_PDF_BYTES.
_FAKE_PDF = b"%PDF-1.4\n" + b"x" * 200


def test_looks_like_pdf_true() -> None:
    assert _looks_like_pdf(_FAKE_PDF) is True


def test_looks_like_pdf_too_short() -> None:
    assert _looks_like_pdf(b"%PDF-") is False


def test_looks_like_pdf_wrong_magic() -> None:
    assert _looks_like_pdf(b"<html>" + b"x" * 200) is False


def test_s2_oa_lookup_returns_url() -> None:
    body = json.dumps({"openAccessPdf": {"url": "https://oa/x.pdf"}}).encode()
    with patch("sagent.tools.paper_fetch.fetch", return_value=body):
        url = _s2_oa_lookup("doi", "10.1234/x")
    assert url == "https://oa/x.pdf"


def test_s2_oa_lookup_no_url_returns_none() -> None:
    body = json.dumps({"openAccessPdf": None}).encode()
    with patch("sagent.tools.paper_fetch.fetch", return_value=body):
        assert _s2_oa_lookup("doi", "10.1234/x") is None


def test_s2_oa_lookup_non_dict_returns_none() -> None:
    body = b"[]"
    with patch("sagent.tools.paper_fetch.fetch", return_value=body):
        assert _s2_oa_lookup("doi", "10.1234/x") is None


def test_s2_oa_lookup_fetch_error_returns_none() -> None:
    err = FetchError(url="u", status=500, headers={}, body=b"x")
    with patch("sagent.tools.paper_fetch.fetch", side_effect=err):
        assert _s2_oa_lookup("doi", "10.1234/x") is None


def test_s2_oa_lookup_invalid_json_returns_none() -> None:
    with patch("sagent.tools.paper_fetch.fetch", return_value=b"not json"):
        assert _s2_oa_lookup("doi", "10.1234/x") is None


def test_paper_fetch_metadata(tmp_path: Path) -> None:
    t = PaperFetch(cache_dir=tmp_path)
    assert t.name == "PaperFetch"
    assert t.tool_id == "application/x-tool-paperfetch"


def test_summary_id(tmp_path: Path) -> None:
    out = PaperFetch(cache_dir=tmp_path).summary({"id": "10.1234/abc"})
    assert "PaperFetch" in out
    assert "10.1234/abc" in out


def test_summary_missing_id(tmp_path: Path) -> None:
    assert PaperFetch(cache_dir=tmp_path).summary({}) == "PaperFetch ?"


def test_prompt_empty(tmp_path: Path) -> None:
    assert PaperFetch(cache_dir=tmp_path).prompt() == ""


def test_run_empty_id(tmp_path: Path) -> None:
    result = asyncio.run(PaperFetch(cache_dir=tmp_path).run({"id": "  "}))
    assert result.is_error
    assert "'id' is required" in result.content


def test_run_invalid_id(tmp_path: Path) -> None:
    result = asyncio.run(PaperFetch(cache_dir=tmp_path).run({"id": "garbage"}))
    assert result.is_error


def test_run_cached_existing(tmp_path: Path) -> None:
    """An existing PDF in the cache short-circuits the cascade."""
    # The slug for arxiv id "1234.56789" is "arxiv_1234.56789".
    cache_file = tmp_path / "arxiv_1234.56789.pdf"
    _ = cache_file.write_bytes(_FAKE_PDF)
    result = asyncio.run(PaperFetch(cache_dir=tmp_path).run({"id": "1234.56789"}))
    assert "Cached" in result.content
    assert str(cache_file) in result.content


def test_run_arxiv_download_writes_cache(tmp_path: Path) -> None:
    """ArXiv source successfully downloads on first try."""
    with patch("sagent.tools.paper_fetch.fetch", return_value=_FAKE_PDF):
        result = asyncio.run(
            PaperFetch(cache_dir=tmp_path).run({"id": "1234.56789"}),
        )
    assert not result.is_error
    assert "Downloaded via arxiv" in result.content
    assert (tmp_path / "arxiv_1234.56789.pdf").exists()


def test_run_cascade_all_fail(tmp_path: Path) -> None:
    """All sources return None → tool error."""
    err = FetchError(url="u", status=500, headers={}, body=b"")
    with patch("sagent.tools.paper_fetch.fetch", side_effect=err):
        result = asyncio.run(
            PaperFetch(cache_dir=tmp_path).run({"id": "10.1234/nonexistent"}),
        )
    assert result.is_error
    assert "No source returned a PDF" in result.content


def test_run_open_access_fallback_for_doi(tmp_path: Path) -> None:
    """For DOI ids (no arxiv), the OA source is tried first."""
    oa_meta = json.dumps(
        {"openAccessPdf": {"url": "https://oa.example/x.pdf"}}
    ).encode()
    call_count = {"n": 0}

    def fake_fetch(url: str, **_kw: object) -> bytes:
        call_count["n"] += 1
        if "openAccessPdf" in url:
            return oa_meta
        return _FAKE_PDF

    with patch("sagent.tools.paper_fetch.fetch", side_effect=fake_fetch):
        result = asyncio.run(
            PaperFetch(cache_dir=tmp_path).run({"id": "10.1234/doi_oa_test"}),
        )
    assert not result.is_error
    assert "Downloaded via open_access" in result.content


def test_min_pdf_bytes_threshold_check(tmp_path: Path) -> None:
    """Cache short-circuit needs file > _MIN_PDF_BYTES."""
    cache_file = tmp_path / "arxiv_1234.56789.pdf"
    # Smaller than threshold: cache miss, falls through.
    _ = cache_file.write_bytes(b"%PDF-" + b"x" * (_MIN_PDF_BYTES - 10))
    err = FetchError(url="u", status=500, headers={}, body=b"")
    with patch("sagent.tools.paper_fetch.fetch", side_effect=err):
        result = asyncio.run(
            PaperFetch(cache_dir=tmp_path).run({"id": "1234.56789"}),
        )
    # Because file is too small, cascade ran and failed.
    assert result.is_error


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
