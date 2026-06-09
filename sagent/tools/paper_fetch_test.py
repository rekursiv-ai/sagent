"""Tests for ``tools.paper_fetch``: PDF download cascade."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

import asyncio
import json

from sagent.lib.web.fetch import FetchError
from sagent.tools import paper_common
from sagent.tools.paper_fetch import (
    _MIN_PDF_BYTES,
    PaperFetch,
    _looks_like_pdf,
    _s2_oa_lookup,
)


if TYPE_CHECKING:
    import pytest


# Minimal stub of a PDF: starts with magic + padding > _MIN_PDF_BYTES.
_FAKE_PDF = b"%PDF-1.4\n" + b"x" * 200


def test_looks_like_pdf_true() -> None:
    assert _looks_like_pdf(_FAKE_PDF) is True


def test_looks_like_pdf_too_short() -> None:
    assert _looks_like_pdf(b"%PDF-") is False


def test_looks_like_pdf_wrong_magic() -> None:
    assert _looks_like_pdf(b"<html>" + b"x" * 200) is False


# _s2_oa_lookup now routes through the gated paper_common.s2_get client, so
# tests patch paper_common.fetch and run the coroutine.
def test_s2_oa_lookup_returns_url() -> None:
    body = json.dumps({"openAccessPdf": {"url": "https://oa/x.pdf"}}).encode()
    with patch("sagent.tools.paper_common.fetch", return_value=body):
        url = asyncio.run(_s2_oa_lookup("doi", "10.1234/x"))
    assert url == "https://oa/x.pdf"


def test_s2_oa_lookup_no_url_returns_none() -> None:
    body = json.dumps({"openAccessPdf": None}).encode()
    with patch("sagent.tools.paper_common.fetch", return_value=body):
        assert asyncio.run(_s2_oa_lookup("doi", "10.1234/x")) is None


def test_s2_oa_lookup_http_error_returns_none() -> None:
    err = FetchError(url="u", status=500, headers={}, body=b"x")
    with patch("sagent.tools.paper_common.fetch", side_effect=err):
        assert asyncio.run(_s2_oa_lookup("doi", "10.1234/x")) is None


def test_s2_oa_lookup_invalid_json_returns_none() -> None:
    with patch("sagent.tools.paper_common.fetch", return_value=b"not json"):
        assert asyncio.run(_s2_oa_lookup("doi", "10.1234/x")) is None


def test_paper_fetch_metadata(tmp_path: Path) -> None:
    t = PaperFetch(cache_dir=tmp_path)
    assert t.name == "PaperFetch"
    assert t.tool_id == "application/x-tool-paperfetch"


def test_summary_id(tmp_path: Path) -> None:
    out = PaperFetch(cache_dir=tmp_path).summary({"ids": ["10.1234/abc"]})
    assert "PaperFetch" in out
    assert "10.1234/abc" in out


def test_summary_missing_id(tmp_path: Path) -> None:
    assert PaperFetch(cache_dir=tmp_path).summary({}) == "PaperFetch ?"


def test_prompt_empty(tmp_path: Path) -> None:
    assert PaperFetch(cache_dir=tmp_path).prompt() == ""


def test_run_empty_id(tmp_path: Path) -> None:
    result = asyncio.run(PaperFetch(cache_dir=tmp_path).run({"ids": ["  "]}))
    assert result.is_error
    assert "'ids' is empty" in result.content


def test_run_invalid_id(tmp_path: Path) -> None:
    result = asyncio.run(PaperFetch(cache_dir=tmp_path).run({"ids": ["garbage"]}))
    assert result.is_error


def test_run_cached_existing(tmp_path: Path) -> None:
    """An existing PDF in the cache short-circuits the cascade."""
    # The slug for arxiv id "1234.56789" is "arxiv_1234.56789".
    cache_file = tmp_path / "arxiv_1234.56789.pdf"
    _ = cache_file.write_bytes(_FAKE_PDF)
    result = asyncio.run(PaperFetch(cache_dir=tmp_path).run({"ids": ["1234.56789"]}))
    assert "Cached" in result.content
    assert str(cache_file) in result.content


def test_run_arxiv_download_writes_cache(tmp_path: Path) -> None:
    """ArXiv source successfully downloads on first try."""
    with patch("sagent.tools.paper_fetch.fetch", return_value=_FAKE_PDF):
        result = asyncio.run(
            PaperFetch(cache_dir=tmp_path).run({"ids": ["1234.56789"]}),
        )
    assert not result.is_error
    assert "Downloaded via arxiv" in result.content
    assert (tmp_path / "arxiv_1234.56789.pdf").exists()


def test_run_cascade_all_fail(tmp_path: Path) -> None:
    """All sources return None → tool error."""
    err = FetchError(url="u", status=500, headers={}, body=b"")
    with patch("sagent.tools.paper_fetch.fetch", side_effect=err):
        result = asyncio.run(
            PaperFetch(cache_dir=tmp_path).run({"ids": ["10.1234/nonexistent"]}),
        )
    assert result.is_error
    assert "No source returned a PDF" in result.content


def test_run_open_access_fallback_for_doi(tmp_path: Path) -> None:
    """For DOI ids (no arxiv), the OA source is tried first."""
    oa_meta = json.dumps(
        {"openAccessPdf": {"url": "https://oa.example/x.pdf"}}
    ).encode()

    # OA metadata lookup goes through the gated paper_common client; the PDF
    # download goes through paper_fetch.fetch.
    with (
        patch("sagent.tools.paper_common.fetch", return_value=oa_meta),
        patch("sagent.tools.paper_fetch.fetch", return_value=_FAKE_PDF),
    ):
        result = asyncio.run(
            PaperFetch(cache_dir=tmp_path).run({"ids": ["10.1234/doi_oa_test"]}),
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
            PaperFetch(cache_dir=tmp_path).run({"ids": ["1234.56789"]}),
        )
    # Because file is too small, cascade ran and failed.
    assert result.is_error


def test_cache_rejects_non_pdf_file(tmp_path: Path) -> None:
    """A cached file large enough but lacking the %PDF- magic is not served.

    The cache check must use the same magic-byte bar as fresh downloads, so a
    corrupted/truncated cache entry re-triggers the fetch cascade.
    """
    cache_file = tmp_path / "arxiv_1234.56789.pdf"
    _ = cache_file.write_bytes(b"<html>" + b"x" * 300)  # big but not a PDF
    err = FetchError(url="u", status=500, headers={}, body=b"")
    with patch("sagent.tools.paper_fetch.fetch", side_effect=err):
        result = asyncio.run(
            PaperFetch(cache_dir=tmp_path).run({"ids": ["1234.56789"]}),
        )
    assert result.is_error  # not served as "Cached:"


def test_run_ids_batches_oa_lookup(tmp_path: Path) -> None:
    """'ids' resolves OA URLs in ONE batched S2 call, then downloads each."""
    batch_array = json.dumps(
        [
            {"openAccessPdf": {"url": "https://oa.example/a.pdf"}},
            {"openAccessPdf": {"url": "https://oa.example/b.pdf"}},
        ]
    ).encode()
    common_calls = {"n": 0}

    def fake_common_fetch(**_kw: object) -> bytes:
        common_calls["n"] += 1
        return batch_array

    def fake_pdf_fetch(url: str, **_kw: object) -> bytes:
        del url
        return _FAKE_PDF

    with (
        patch(
            "sagent.tools.paper_common.fetch",
            side_effect=fake_common_fetch,
        ),
        patch(
            "sagent.tools.paper_fetch.fetch",
            side_effect=fake_pdf_fetch,
        ),
    ):
        result = asyncio.run(
            PaperFetch(cache_dir=tmp_path).run({"ids": ["10.1234/aa", "10.1234/bb"]}),
        )
    assert common_calls["n"] == 1  # one batched OA lookup, not two
    assert result.content.count("Downloaded via open_access") == 2
    assert (tmp_path / "doi_10.1234_aa.pdf").exists()
    assert (tmp_path / "doi_10.1234_bb.pdf").exists()


def test_run_ids_empty_rejected(tmp_path: Path) -> None:
    result = asyncio.run(PaperFetch(cache_dir=tmp_path).run({"ids": []}))
    assert result.is_error


def test_single_doi_oa_lookup_uses_s2_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The single-id OA lookup must pass through the shared S2 rate gate."""
    calls = {"n": 0}

    class CountingGate:
        async def acquire_async(self) -> None:
            calls["n"] += 1

    monkeypatch.setattr(paper_common, "_s2_gate", CountingGate)
    with (
        patch("sagent.tools.paper_common.fetch", return_value=b"{}"),
        patch("sagent.tools.paper_fetch.fetch", return_value=_FAKE_PDF),
    ):
        _ = asyncio.run(PaperFetch(cache_dir=tmp_path).run({"ids": ["10.1234/a"]}))
    assert calls["n"] >= 1  # the gate was acquired for the OA lookup


def test_run_ids_marks_error_when_all_downloads_fail(tmp_path: Path) -> None:
    """A batch where every id fails must report is_error, not silent success."""
    batch_array = b"[null, null]"
    err = FetchError(url="u", status=500, headers={}, body=b"")
    with (
        patch(
            "sagent.tools.paper_common.fetch",
            return_value=batch_array,
        ),
        patch("sagent.tools.paper_fetch.fetch", side_effect=err),
    ):
        result = asyncio.run(
            PaperFetch(cache_dir=tmp_path).run({"ids": ["10.1234/a", "10.1234/b"]}),
        )
    assert result.is_error
    assert result.content.count("No source returned a PDF") == 2


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
