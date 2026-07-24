"""Tests for ``tools.paper_fetch``: the thin adapter over ``paper.fetch``.

The source cascade (arXiv/open-access/source-only) and the batched open-access
URL resolve live in :mod:`wesearch.paper` and are tested there. These tests
cover only the adapter's concerns: schema/metadata, ``summary``, the on-disk PDF
cache, the single- and multi-id orchestration, result rendering, and error
mapping. The library surface is mocked where the adapter binds each name:

- ``patch("...paper_fetch.download", ...)`` -- the per-id download.
- ``patch("...paper_fetch.batch_oa_urls", ...)`` -- the batched OA resolve.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import asyncio

from wesearch.paper.errors import NotFoundError

from sagent.tools.paper_fetch import PaperFetch, _is_cached_pdf
from sagent.types.runtime import ToolResult


# A valid PDF: magic prefix + padding past the 128-byte floor _is_cached_pdf
# and the library's looks_like_pdf both enforce.
_FAKE_PDF = b"%PDF-1.5" + b"0" * 200


# ---------------------------------------------------------------------------
# Metadata / summary / prompt
# ---------------------------------------------------------------------------


def test_paper_fetch_metadata(tmp_path: Path) -> None:
    t = PaperFetch(cache_dir=tmp_path)
    assert t.name == "PaperFetch"
    assert t.tool_id == "application/x-tool-paperfetch"


def test_summary_id(tmp_path: Path) -> None:
    out = PaperFetch(cache_dir=tmp_path).summary({"ids": ["10.1234/abc"]})
    assert "PaperFetch" in out
    assert "10.1234/abc" in out


def test_summary_multi_id(tmp_path: Path) -> None:
    out = PaperFetch(cache_dir=tmp_path).summary({"ids": ["10.1234/a", "10.1234/b"]})
    assert "PaperFetch" in out
    assert "+1 more" in out


def test_summary_missing_id(tmp_path: Path) -> None:
    assert PaperFetch(cache_dir=tmp_path).summary({}) == "PaperFetch ?"


def test_prompt_empty(tmp_path: Path) -> None:
    assert PaperFetch(cache_dir=tmp_path).prompt() == ""


def test_summary_result_suppressed(tmp_path: Path) -> None:
    result = ToolResult(call_id="", content="anything")
    assert PaperFetch(cache_dir=tmp_path).summary_result(result) is None


# ---------------------------------------------------------------------------
# _is_cached_pdf
# ---------------------------------------------------------------------------


def test_is_cached_pdf_valid(tmp_path: Path) -> None:
    path = tmp_path / "arxiv_1234.56789.pdf"
    _ = path.write_bytes(_FAKE_PDF)
    assert _is_cached_pdf(path) is True


def test_is_cached_pdf_too_small(tmp_path: Path) -> None:
    path = tmp_path / "arxiv_1234.56789.pdf"
    _ = path.write_bytes(b"%PDF-tiny")
    assert _is_cached_pdf(path) is False


def test_is_cached_pdf_wrong_magic(tmp_path: Path) -> None:
    path = tmp_path / "arxiv_1234.56789.pdf"
    _ = path.write_bytes(b"<html>" + b"0" * 300)
    assert _is_cached_pdf(path) is False


def test_is_cached_pdf_missing(tmp_path: Path) -> None:
    assert _is_cached_pdf(tmp_path / "nope.pdf") is False


# ---------------------------------------------------------------------------
# Argument validation (delegated to paper_common, surfaced by run)
# ---------------------------------------------------------------------------


def test_run_empty_id(tmp_path: Path) -> None:
    result = asyncio.run(PaperFetch(cache_dir=tmp_path).run({"ids": ["  "]}))
    assert result.is_error
    assert "'ids' is empty" in result.content


def test_run_invalid_id(tmp_path: Path) -> None:
    result = asyncio.run(PaperFetch(cache_dir=tmp_path).run({"ids": ["garbage"]}))
    assert result.is_error


def test_run_ids_empty_rejected(tmp_path: Path) -> None:
    result = asyncio.run(PaperFetch(cache_dir=tmp_path).run({"ids": []}))
    assert result.is_error


# ---------------------------------------------------------------------------
# Cache short-circuit
# ---------------------------------------------------------------------------


def test_run_cached_existing(tmp_path: Path) -> None:
    """A valid cached PDF short-circuits to ``Cached: <path>`` without fetching."""
    cache_file = tmp_path / "arxiv_1234.56789.pdf"
    _ = cache_file.write_bytes(_FAKE_PDF)
    with patch("sagent.tools.paper_fetch.download") as download:
        result = asyncio.run(
            PaperFetch(cache_dir=tmp_path).run({"ids": ["1234.56789"]})
        )
    download.assert_not_called()
    assert not result.is_error
    assert result.content == f"Cached: {cache_file}"


def test_run_cache_too_small_refetches(tmp_path: Path) -> None:
    """A too-small cache file is not treated as cached; the library is called."""
    cache_file = tmp_path / "arxiv_1234.56789.pdf"
    _ = cache_file.write_bytes(b"%PDF-tiny")
    with patch(
        "sagent.tools.paper_fetch.download",
        return_value=(_FAKE_PDF, "arxiv"),
    ) as download:
        result = asyncio.run(
            PaperFetch(cache_dir=tmp_path).run({"ids": ["1234.56789"]})
        )
    download.assert_called_once()
    assert result.content == f"Downloaded via arxiv: {cache_file}"


def test_run_cache_garbage_refetches(tmp_path: Path) -> None:
    """A big-but-non-PDF cache entry re-triggers the fetch, not a cache hit."""
    cache_file = tmp_path / "arxiv_1234.56789.pdf"
    _ = cache_file.write_bytes(b"<html>" + b"0" * 300)
    with patch(
        "sagent.tools.paper_fetch.download",
        return_value=(_FAKE_PDF, "arxiv"),
    ) as download:
        result = asyncio.run(
            PaperFetch(cache_dir=tmp_path).run({"ids": ["1234.56789"]})
        )
    download.assert_called_once()
    assert "Downloaded via arxiv" in result.content


# ---------------------------------------------------------------------------
# Single-id path
# ---------------------------------------------------------------------------


def test_run_single_download_writes_cache(tmp_path: Path) -> None:
    """A single id fetches once, writes the bytes, and renders the source."""
    cache_file = tmp_path / "arxiv_1234.56789.pdf"
    with patch(
        "sagent.tools.paper_fetch.download",
        return_value=(_FAKE_PDF, "arxiv"),
    ) as download:
        result = asyncio.run(
            PaperFetch(cache_dir=tmp_path).run({"ids": ["1234.56789"]})
        )
    assert not result.is_error
    assert result.content == f"Downloaded via arxiv: {cache_file}"
    assert cache_file.read_bytes() == _FAKE_PDF
    # Single id: no OA pre-resolve, and no completed lookup claimed.
    download.assert_called_once_with(
        "arxiv", "1234.56789", oa_url=None, oa_looked_up=False
    )


def test_run_single_open_access_source(tmp_path: Path) -> None:
    """The exact rendered source label round-trips from the library."""
    with patch(
        "sagent.tools.paper_fetch.download",
        return_value=(_FAKE_PDF, "open_access"),
    ):
        result = asyncio.run(PaperFetch(cache_dir=tmp_path).run({"ids": ["10.1234/x"]}))
    cache_file = tmp_path / "doi_10.1234_x.pdf"
    assert result.content == f"Downloaded via open_access: {cache_file}"


def test_run_single_error_maps_to_tool_error(tmp_path: Path) -> None:
    """A ``PaperError`` becomes an is_error ToolResult carrying its text."""
    err = NotFoundError("No source returned a PDF for arxiv:1234.56789.")
    with patch("sagent.tools.paper_fetch.download", side_effect=err):
        result = asyncio.run(
            PaperFetch(cache_dir=tmp_path).run({"ids": ["1234.56789"]})
        )
    assert result.is_error
    assert result.content == "No source returned a PDF for arxiv:1234.56789."
    assert not (tmp_path / "arxiv_1234.56789.pdf").exists()


# ---------------------------------------------------------------------------
# Multi-id path: batched OA resolve, concurrent fetch, joined output
# ---------------------------------------------------------------------------


def test_run_multi_batches_oa_then_fetches(tmp_path: Path) -> None:
    """Several ids resolve OA URLs in ONE batch, then fetch each concurrently."""
    urls = ["https://oa.example/a.pdf", "https://oa.example/b.pdf"]

    def fake_download(
        kind: str, canonical: str, *, oa_url: str | None, oa_looked_up: bool
    ) -> tuple[bytes, str]:
        del kind, canonical, oa_url, oa_looked_up
        return _FAKE_PDF, "open_access"

    with (
        patch(
            "sagent.tools.paper_fetch.batch_oa_urls",
            return_value=urls,
        ) as batch,
        patch(
            "sagent.tools.paper_fetch.download",
            side_effect=fake_download,
        ) as download,
    ):
        result = asyncio.run(
            PaperFetch(cache_dir=tmp_path).run({"ids": ["10.1234/aa", "10.1234/bb"]})
        )

    batch.assert_called_once_with(["DOI:10.1234/aa", "DOI:10.1234/bb"])
    assert result.content.count("Downloaded via open_access") == 2
    assert (tmp_path / "doi_10.1234_aa.pdf").exists()
    assert (tmp_path / "doi_10.1234_bb.pdf").exists()
    # Batch succeeded: each fetch is told the OA lookup is complete.
    for c in download.call_args_list:
        assert c.kwargs["oa_looked_up"] is True
    passed_urls = {c.kwargs["oa_url"] for c in download.call_args_list}
    assert passed_urls == set(urls)


def test_run_multi_batch_failure_falls_back(tmp_path: Path) -> None:
    """A ``None`` batch result falls back to per-id lookups (oa_looked_up False)."""
    with (
        patch(
            "sagent.tools.paper_fetch.batch_oa_urls",
            return_value=None,
        ),
        patch(
            "sagent.tools.paper_fetch.download",
            return_value=(_FAKE_PDF, "arxiv"),
        ) as download,
    ):
        result = asyncio.run(
            PaperFetch(cache_dir=tmp_path).run({"ids": ["1234.11111", "1234.22222"]})
        )
    assert not result.is_error
    for c in download.call_args_list:
        assert c.kwargs["oa_url"] is None
        assert c.kwargs["oa_looked_up"] is False


def test_run_multi_joined_output_order(tmp_path: Path) -> None:
    """Per-id lines are newline-joined in input order (not completion order)."""
    sources = {"1234.11111": "arxiv", "10.1234/bb": "open_access"}

    def fake_download(
        _kind: str, canonical: str, *, oa_url: str | None, oa_looked_up: bool
    ) -> tuple[bytes, str]:
        del oa_url, oa_looked_up
        return _FAKE_PDF, sources[canonical]

    with (
        patch(
            "sagent.tools.paper_fetch.batch_oa_urls",
            return_value=[None, None],
        ),
        patch(
            "sagent.tools.paper_fetch.download",
            side_effect=fake_download,
        ),
    ):
        result = asyncio.run(
            PaperFetch(cache_dir=tmp_path).run({"ids": ["1234.11111", "10.1234/bb"]})
        )
    line_a = f"Downloaded via arxiv: {tmp_path / 'arxiv_1234.11111.pdf'}"
    line_b = f"Downloaded via open_access: {tmp_path / 'doi_10.1234_bb.pdf'}"
    assert result.content == f"{line_a}\n{line_b}"


def test_run_multi_error_if_any_id_fails(tmp_path: Path) -> None:
    """is_error is True if ANY id errors; the good id still renders its line."""
    err = NotFoundError("No source returned a PDF for doi:10.1234/bb.")

    def fake_download(
        _kind: str, canonical: str, *, oa_url: str | None, oa_looked_up: bool
    ) -> tuple[bytes, str]:
        del oa_url, oa_looked_up
        if canonical == "10.1234/bb":
            raise err
        return _FAKE_PDF, "arxiv"

    with (
        patch(
            "sagent.tools.paper_fetch.batch_oa_urls",
            return_value=[None, None],
        ),
        patch(
            "sagent.tools.paper_fetch.download",
            side_effect=fake_download,
        ),
    ):
        result = asyncio.run(
            PaperFetch(cache_dir=tmp_path).run({"ids": ["1234.11111", "10.1234/bb"]})
        )
    assert result.is_error
    assert "Downloaded via arxiv" in result.content
    assert "No source returned a PDF for doi:10.1234/bb." in result.content


def test_run_multi_all_fail(tmp_path: Path) -> None:
    """Every id failing reports is_error with one error line per id."""
    err = NotFoundError("No source returned a PDF.")
    with (
        patch(
            "sagent.tools.paper_fetch.batch_oa_urls",
            return_value=[None, None],
        ),
        patch("sagent.tools.paper_fetch.download", side_effect=err),
    ):
        result = asyncio.run(
            PaperFetch(cache_dir=tmp_path).run({"ids": ["10.1234/a", "10.1234/b"]})
        )
    assert result.is_error
    assert result.content.count("No source returned a PDF.") == 2


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
