"""Tests for PDF helper utilities."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import shutil
import subprocess as _sp

import pytest

from sagent.tools.lib.pdf import (
    PdfError,
    extract_pdf_pages,
    is_pdf,
    parse_page_range,
)


_MINIMAL_3PAGE_PDF = (
    b"%PDF-1.4\n"
    b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
    b"2 0 obj<</Type/Pages/Count 3/Kids[3 0 R 4 0 R 7 0 R]>>endobj\n"
    b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 100 100]/Contents 5 0 R/Resources<<>>>>endobj\n"
    b"4 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 100 100]/Contents 6 0 R/Resources<<>>>>endobj\n"
    b"7 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 100 100]/Contents 8 0 R/Resources<<>>>>endobj\n"
    b"5 0 obj<</Length 44>>stream\nBT /F1 12 Tf 10 50 Td (page one) Tj ET\nendstream endobj\n"
    b"6 0 obj<</Length 44>>stream\nBT /F1 12 Tf 10 50 Td (page two) Tj ET\nendstream endobj\n"
    b"8 0 obj<</Length 46>>stream\nBT /F1 12 Tf 10 50 Td (page three) Tj ET\nendstream endobj\n"
    b"xref\n0 9\n0000000000 65535 f\n"
    b"0000000009 00000 n\n0000000052 00000 n\n0000000107 00000 n\n"
    b"0000000176 00000 n\n0000000245 00000 n\n0000000316 00000 n\n"
    b"0000000387 00000 n\n0000000458 00000 n\n"
    b"trailer<</Size 9/Root 1 0 R>>\nstartxref\n531\n%%EOF\n"
)


class TestParsePageRange:
    def test_single_page(self) -> None:
        assert parse_page_range("5") == (5, 5)

    def test_range(self) -> None:
        assert parse_page_range("1-10") == (1, 10)

    def test_open_ended(self) -> None:
        assert parse_page_range("3-") == (3, None)

    def test_strip_whitespace(self) -> None:
        assert parse_page_range("  2-7  ") == (2, 7)

    def test_empty(self) -> None:
        assert parse_page_range("") is None
        assert parse_page_range("   ") is None

    def test_zero_invalid(self) -> None:
        assert parse_page_range("0") is None
        assert parse_page_range("0-5") is None

    def test_negative_invalid(self) -> None:
        assert parse_page_range("-5") is None

    def test_inverted_invalid(self) -> None:
        assert parse_page_range("10-3") is None

    def test_garbage(self) -> None:
        assert parse_page_range("abc") is None
        assert parse_page_range("1-abc") is None
        assert parse_page_range("1-2-3") is None


class TestIsPdf:
    def test_with_magic(self, tmp_path: Path) -> None:
        f = tmp_path / "real.pdf"
        f.write_bytes(b"%PDF-1.7\nfoo")
        assert is_pdf(f)

    def test_without_magic(self, tmp_path: Path) -> None:
        f = tmp_path / "fake.pdf"
        f.write_bytes(b"<html>")
        assert not is_pdf(f)

    def test_missing(self, tmp_path: Path) -> None:
        assert not is_pdf(tmp_path / "nope.pdf")


class TestExtractPdfPages:
    def test_pdftoppm_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        f = tmp_path / "a.pdf"
        f.write_bytes(b"%PDF-1.4")

        def _no_which(_name: str) -> str | None:
            return None

        monkeypatch.setattr(
            "sagent.tools.lib.pdf.shutil.which",
            _no_which,
        )
        with pytest.raises(PdfError, match="pdftoppm not installed"):
            extract_pdf_pages(f)

    def test_happy_path_renders_pages(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``-f`` / ``-l`` flags are forwarded; produced JPEGs are returned."""
        f = tmp_path / "doc.pdf"
        f.write_bytes(b"%PDF-1.4")
        captured: dict[str, object] = {}

        def _fake_which(name: str) -> str | None:
            return f"/usr/bin/{name}"

        def _fake_run(cmd: list[str], **_kw: object) -> _sp.CompletedProcess[str]:
            captured["cmd"] = cmd
            # cmd: [pdftoppm, -jpeg, -r, 100, -f, 2, -l, 4, src, prefix]
            prefix = cmd[-1]
            for i in (2, 3, 4):
                Path(f"{prefix}-{i:02d}.jpg").write_bytes(b"\xff\xd8\xff\xd9")
            return _sp.CompletedProcess(cmd, 0, stdout="", stderr="")

        monkeypatch.setattr(
            "sagent.tools.lib.pdf.shutil.which",
            _fake_which,
        )
        monkeypatch.setattr(
            "sagent.tools.lib.pdf.subprocess.run",
            _fake_run,
        )
        out = extract_pdf_pages(f, first=2, last=4)
        cmd = captured["cmd"]
        assert isinstance(cmd, list)
        assert "-f" in cmd
        assert "2" in cmd
        assert "-l" in cmd
        assert "4" in cmd
        assert len(out) == 3
        assert all(p.suffix == ".jpg" for p in out)

    def test_password_protected_classified(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        f = tmp_path / "locked.pdf"
        f.write_bytes(b"%PDF-1.4")

        def _fake_which(name: str) -> str | None:
            return f"/usr/bin/{name}"

        def _fake_run(cmd: list[str], **_kw: object) -> _sp.CompletedProcess[str]:
            return _sp.CompletedProcess(cmd, 1, stdout="", stderr="Incorrect password")

        monkeypatch.setattr(
            "sagent.tools.lib.pdf.shutil.which",
            _fake_which,
        )
        monkeypatch.setattr(
            "sagent.tools.lib.pdf.subprocess.run",
            _fake_run,
        )
        with pytest.raises(PdfError, match="password-protected"):
            extract_pdf_pages(f)


@pytest.mark.skipif(
    not (shutil.which("pdftoppm") and shutil.which("pdfinfo")),
    reason="requires poppler-utils",
)
class TestConcurrency:
    """Real pdftoppm under concurrent use - verifies no cross-talk."""

    def test_concurrent_extracts_disjoint_outputs(self, tmp_path: Path) -> None:
        """Four threads rasterizing the same PDF must produce disjoint files."""
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(_MINIMAL_3PAGE_PDF)

        def _run(_i: int) -> list[Path]:
            return extract_pdf_pages(pdf)

        with ThreadPoolExecutor(max_workers=4) as ex:
            results = list(ex.map(_run, range(4)))

        # Every call must produce three pages.
        assert all(len(r) == 3 for r in results)
        # Parent directories must be distinct (UUID subdirs).
        parents = {r[0].parent for r in results}
        assert len(parents) == 4
        # Every produced file must be a valid JPEG (SOI marker).
        for r in results:
            for page in r:
                assert page.read_bytes()[:2] == b"\xff\xd8"


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
