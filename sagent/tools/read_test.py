"""Tests for Read tool."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import cast

import json as _json
import re

from PIL import Image

import pytest

from sagent.custom_types import (
    JsonMessage,
    Message,
    MultipartMessage,
    TextMessage,
)
from sagent.lib.json import JSON, json_freeze
from sagent.tools.core import ToolState, tool_state_context
from sagent.tools.lib.bash import parse_bash
from sagent.tools.lib.pdf import PdfError
from sagent.tools.read import Read


read = Read()


def _text(r: Message) -> str:
    if isinstance(r, TextMessage):
        return r.content
    if isinstance(r, MultipartMessage):
        for p in r.content:
            if isinstance(p, TextMessage) and p.descriptor == "text/plain":
                return p.content
    return ""


def _error_text(r: Message) -> str:
    if isinstance(r, TextMessage) and r.descriptor == "text/x-error":
        return r.content
    return ""


def _msg(directive: JSON) -> Message:
    return MultipartMessage(
        (JsonMessage(directive, "application/x-tool-x"),),
        "multipart/x-tool-call",
    )


class TestRead:
    @pytest.mark.anyio
    async def test_read_file(self, tmp_path: Path) -> None:
        f = tmp_path / "test.txt"
        f.write_text("line0\nline1\nline2\n")
        response = await read.run(_msg(json_freeze({"file_path": str(f)})))
        assert "1\tline0" in _text(response)
        assert "3\tline2" in _text(response)

    @pytest.mark.anyio
    async def test_summary_result_reports_line_count(self, tmp_path: Path) -> None:
        """``Read.summary_result`` returns ``"{N} lines"`` for plain reads."""
        summary_read = Read()
        summary_read.emit_tool_summary = True
        f = tmp_path / "test.txt"
        f.write_text("a\nb\nc\n")
        response = await summary_read.run(_msg(json_freeze({"file_path": str(f)})))
        # Line-numbered output: 3 source lines → 3 newlines in body.
        assert summary_read.summary_result(response) == "3 lines"

    def test_summary_result_off_by_default(self) -> None:
        """``Read.summary_result`` returns ``None`` when the gate is off."""
        assert read.summary_result(TextMessage("x", "text/plain")) is None

    @pytest.mark.anyio
    async def test_relative_path_uses_bash_cwd(self, tmp_path: Path) -> None:
        f = tmp_path / "test.txt"
        f.write_text("cwd file\n")
        state = ToolState()
        state.bash_cwd = str(tmp_path)
        with tool_state_context(state):
            response = await read.run(_msg(json_freeze({"file_path": "test.txt"})))
        assert "1\tcwd file" in _text(response)

    @pytest.mark.anyio
    async def test_read_offset_limit(self, tmp_path: Path) -> None:
        f = tmp_path / "test.txt"
        f.write_text("a\nb\nc\nd\ne\n")
        response = await read.run(
            _msg(json_freeze({"file_path": str(f), "offset": 2, "limit": 2}))
        )
        assert "2\tb" in _text(response)
        assert "3\tc" in _text(response)
        assert "1\ta" not in _text(response)

    @pytest.mark.anyio
    async def test_read_missing(self) -> None:
        r = await read.run(_msg(json_freeze({"file_path": "/nonexistent"})))
        assert re.search(r"not found", _error_text(r))

    @pytest.mark.anyio
    async def test_read_empty_file(self, tmp_path: Path) -> None:
        f = tmp_path / "empty.txt"
        f.write_text("")
        response = await read.run(_msg(json_freeze({"file_path": str(f)})))
        assert "empty" in _text(response).lower()

    @pytest.mark.anyio
    async def test_read_image(self, tmp_path: Path) -> None:
        f = tmp_path / "test.png"
        buf = BytesIO()
        Image.new("RGB", (10, 10), (255, 0, 0)).save(buf, format="PNG")
        f.write_bytes(buf.getvalue())
        response = await read.run(_msg(json_freeze({"file_path": str(f)})))
        assert "image" in _text(response).lower()
        # Image part: text/plain description + image/png attachment.
        assert isinstance(response, MultipartMessage)
        assert len(response.content) == 2
        assert response.content[1].descriptor == "image/png"
        assert response.content[1].content == buf.getvalue()


class TestReadEdgeCases:
    @pytest.mark.anyio
    async def test_read_notebook(self, tmp_path: Path) -> None:
        nb = {
            "cells": [
                {
                    "cell_type": "code",
                    "source": ["print('hello')"],
                    "outputs": [{"text": ["hello\n"]}],
                },
                {
                    "cell_type": "markdown",
                    "source": ["# Title"],
                    "outputs": [],
                },
            ],
        }
        f = tmp_path / "test.ipynb"
        f.write_text(_json.dumps(nb))
        response = await read.run(_msg(json_freeze({"file_path": str(f)})))
        assert "hello" in _text(response)
        assert "Title" in _text(response)
        assert "Cell 1" in _text(response)

    @pytest.mark.anyio
    async def test_read_offset_zero_clamps(self, tmp_path: Path) -> None:
        f = tmp_path / "clamp.txt"
        f.write_text("a\nb\n")
        response = await read.run(_msg(json_freeze({"file_path": str(f), "offset": 0})))
        assert "1\ta" in _text(response)

    @pytest.mark.anyio
    async def test_read_more_lines_hint(self, tmp_path: Path) -> None:
        f = tmp_path / "long.txt"
        f.write_text("\n".join(f"line{i}" for i in range(10)))
        response = await read.run(
            _msg(json_freeze({"file_path": str(f), "offset": 1, "limit": 3}))
        )
        assert "more lines" in _text(response).lower()

    @pytest.mark.anyio
    async def test_read_pdf_missing_magic(self, tmp_path: Path) -> None:
        """A file named ``.pdf`` without the magic header is rejected."""
        f = tmp_path / "fake.pdf"
        f.write_bytes(b"not a pdf")
        r = await read.run(_msg(json_freeze({"file_path": str(f)})))
        assert re.search(r"%PDF-", _error_text(r))


class TestCheckUnchanged:
    @pytest.mark.anyio
    async def test_unchanged_file_skipped(self, tmp_path: Path) -> None:
        f = tmp_path / "unch.txt"
        f.write_text("content\n")
        await read.run(_msg(json_freeze({"file_path": str(f)})))
        r2 = await read.run(_msg(json_freeze({"file_path": str(f)})))
        assert "unchanged" in _text(r2).lower()

    @pytest.mark.anyio
    async def test_different_params_rereads(self, tmp_path: Path) -> None:
        f = tmp_path / "params.txt"
        f.write_text("a\nb\nc\nd\ne\n")
        await read.run(
            _msg(json_freeze({"file_path": str(f), "offset": 1, "limit": 2}))
        )
        r2 = await read.run(
            _msg(json_freeze({"file_path": str(f), "offset": 3, "limit": 2}))
        )
        assert "3\tc" in _text(r2)


class TestReadPDF:
    """Tests for Read's PDF branch using rasterize-via-pdftoppm."""

    @pytest.mark.anyio
    async def test_pdf_with_pages_range(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        f = tmp_path / "test.pdf"
        f.write_bytes(b"%PDF-1.4")
        captured: dict[str, object] = {}

        def _fake_extract(
            path: Path,
            *,
            first: int | None = None,
            last: int | None = None,
            **_: object,
        ) -> list[Path]:
            captured["path"] = path
            captured["first"] = first
            captured["last"] = last
            out = [tmp_path / f"page-{i:02d}.jpg" for i in range(1, 4)]
            for o in out:
                o.write_bytes(b"\xff\xd8\xff\xd9")  # minimal JPEG magic
            return out

        monkeypatch.setattr(
            "sagent.tools.read.extract_pdf_pages",
            _fake_extract,
        )
        response = await read.run(
            _msg(json_freeze({"file_path": str(f), "pages": "2-4"}))
        )
        assert captured["first"] == 2
        assert captured["last"] == 4
        # PDF pages: text/plain description + image/jpeg pages.
        assert isinstance(response, MultipartMessage)
        image_parts = [p for p in response.content if p.descriptor == "image/jpeg"]
        assert len(image_parts) == 3

    @pytest.mark.anyio
    async def test_pdf_single_page(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        f = tmp_path / "test.pdf"
        f.write_bytes(b"%PDF-1.4")
        captured: dict[str, object] = {}

        def _fake_extract(
            _path: Path,
            *,
            first: int | None = None,
            last: int | None = None,
            **_: object,
        ) -> list[Path]:
            captured["first"] = first
            captured["last"] = last
            out = tmp_path / "page-03.jpg"
            out.write_bytes(b"\xff\xd8\xff\xd9")
            return [out]

        monkeypatch.setattr(
            "sagent.tools.read.extract_pdf_pages",
            _fake_extract,
        )
        response = await read.run(
            _msg(json_freeze({"file_path": str(f), "pages": "3"}))
        )
        assert captured["first"] == 3
        assert captured["last"] == 3
        assert isinstance(response, MultipartMessage)
        image_parts = [p for p in response.content if p.descriptor == "image/jpeg"]
        assert len(image_parts) == 1

    @pytest.mark.anyio
    async def test_pdf_bad_range(self, tmp_path: Path) -> None:
        f = tmp_path / "test.pdf"
        f.write_bytes(b"%PDF-1.4")
        r = await read.run(_msg(json_freeze({"file_path": str(f), "pages": "bogus"})))
        assert re.search(r"Invalid pages spec", _error_text(r))

    @pytest.mark.anyio
    async def test_pdf_too_many_pages_without_range(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        f = tmp_path / "huge.pdf"
        f.write_bytes(b"%PDF-1.4")

        def _count_500(_p: Path) -> int:
            return 500

        monkeypatch.setattr(
            "sagent.tools.read.get_pdf_page_count",
            _count_500,
        )
        r = await read.run(_msg(json_freeze({"file_path": str(f)})))
        assert re.search(r"500 pages", _error_text(r))

    @pytest.mark.anyio
    async def test_pdf_pdftoppm_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        f = tmp_path / "test.pdf"
        f.write_bytes(b"%PDF-1.4")

        def _fail(*_a: object, **_k: object) -> list[Path]:
            raise PdfError("pdftoppm not installed")

        def _count_3(_p: Path) -> int:
            return 3

        monkeypatch.setattr(
            "sagent.tools.read.extract_pdf_pages",
            _fail,
        )
        monkeypatch.setattr(
            "sagent.tools.read.get_pdf_page_count",
            _count_3,
        )
        r = await read.run(_msg(json_freeze({"file_path": str(f)})))
        assert re.search(r"pdftoppm", _error_text(r))


class TestReadLastLines:
    @pytest.mark.anyio
    async def test_last_lines_only(self, tmp_path: Path) -> None:
        f = tmp_path / "f.txt"
        f.write_text("\n".join(f"line{i}" for i in range(1, 21)) + "\n")
        r = await read.run(_msg(json_freeze({"file_path": str(f), "last_lines": 5})))
        assert "line16" in _text(r)
        assert "line20" in _text(r)
        assert "line15" not in _text(r)

    @pytest.mark.anyio
    async def test_last_lines_composes_with_limit(self, tmp_path: Path) -> None:
        """``last_lines=10, limit=3`` returns the first 3 of the final 10."""
        f = tmp_path / "f.txt"
        f.write_text("\n".join(f"line{i}" for i in range(1, 21)) + "\n")
        r = await read.run(
            _msg(json_freeze({"file_path": str(f), "last_lines": 10, "limit": 3}))
        )
        # Final 10 = lines 11-20; first 3 of those = lines 11-13.
        assert "line11" in _text(r)
        assert "line13" in _text(r)
        assert "line14" not in _text(r)

    @pytest.mark.anyio
    async def test_last_lines_larger_than_file(self, tmp_path: Path) -> None:
        f = tmp_path / "f.txt"
        f.write_text("a\nb\nc\n")
        r = await read.run(_msg(json_freeze({"file_path": str(f), "last_lines": 100})))
        assert "a" in _text(r)
        assert "c" in _text(r)

    @pytest.mark.anyio
    async def test_numeric_string_params(self, tmp_path: Path) -> None:
        """Numeric-string args coerce via int_val (provider may stringify)."""
        f = tmp_path / "f.txt"
        f.write_text("aaa\nbbb\nccc\nddd\n")
        r = await read.run(
            _msg(
                json_freeze(
                    {
                        "file_path": str(f),
                        "offset": cast("int", "2"),
                        "limit": cast("int", "1"),
                    }
                )
            )
        )
        # Line-number-prefixed content avoids false-positive substring
        # matches against the "... Use offset=N to continue" trailer.
        assert "2\tbbb" in _text(r)
        assert "1\taaa" not in _text(r)
        assert "3\tccc" not in _text(r)

    @pytest.mark.anyio
    async def test_uncoercible_params_fall_back_to_defaults(
        self, tmp_path: Path
    ) -> None:
        f = tmp_path / "f.txt"
        f.write_text("hello")
        r = await read.run(
            _msg(
                json_freeze({"file_path": str(f), "offset": cast("int", "not-an-int")})
            )
        )
        assert "hello" in _text(r)

    @pytest.mark.anyio
    async def test_file_not_found(self) -> None:
        r = await read.run(_msg(json_freeze({"file_path": "/nonexistent/xyz.txt"})))
        assert re.search(r"not found", _error_text(r))

    @pytest.mark.anyio
    async def test_read_directory(self, tmp_path: Path) -> None:
        r = await read.run(_msg(json_freeze({"file_path": str(tmp_path)})))
        assert re.search(r"is a directory", _error_text(r))

    @pytest.mark.anyio
    async def test_notebook_invalid_json(self, tmp_path: Path) -> None:
        nb = tmp_path / "bad.ipynb"
        nb.write_text("not json")
        r = await read.run(_msg(json_freeze({"file_path": str(nb)})))
        assert "Invalid notebook" in _text(r)

    @pytest.mark.anyio
    async def test_binary_file(self, tmp_path: Path) -> None:
        b = tmp_path / "bin.dat"
        b.write_bytes(b"\x00\x01\x02\x03")
        r = await read.run(_msg(json_freeze({"file_path": str(b)})))
        assert "Binary file" in _text(r)

    @pytest.mark.anyio
    async def test_non_utf8_text(self, tmp_path: Path) -> None:
        f = tmp_path / "latin.txt"
        f.write_bytes(b"caf\xe9")  # latin-1, invalid utf-8  # codespell:ignore caf
        r = await read.run(_msg(json_freeze({"file_path": str(f)})))
        assert "Non-UTF-8" in _text(r)

    @pytest.mark.anyio
    async def test_empty_file(self, tmp_path: Path) -> None:
        f = tmp_path / "empty.txt"
        f.write_text("")
        r = await read.run(_msg(json_freeze({"file_path": str(f)})))
        assert "empty" in _text(r)

    @pytest.mark.anyio
    async def test_last_lines_participates_in_cache_key(
        self,
        tmp_path: Path,
    ) -> None:
        """Two reads with different ``last_lines`` must not dedup."""
        with tool_state_context(ToolState()):
            f = tmp_path / "f.txt"
            f.write_text("\n".join(f"line{i}" for i in range(1, 21)) + "\n")
            r1 = await read.run(
                _msg(json_freeze({"file_path": str(f), "last_lines": 5}))
            )
            r2 = await read.run(
                _msg(json_freeze({"file_path": str(f), "last_lines": 10}))
            )
            assert "unchanged" not in _text(r2)
            # r1 = last 5, r2 = last 10. r2 must include lines r1 doesn't.
            assert "line11" in _text(r2)
            assert "line11" not in _text(r1)


# -- Read.bash_match ---------------------------------------------------


def _match(command: str) -> str | None:
    trees = parse_bash(command)
    assert trees is not None
    return read.bash_match(trees)


_CAT_NUDGE = "cat via Bash is a bad UX. Use the Read tool."
_HEAD_NUDGE = "head via Bash is a bad UX. Use the Read tool."
_TAIL_NUDGE = "tail via Bash is a bad UX. Use the Read tool."


class TestBashMatchCat:
    def test_single_file(self) -> None:
        assert _match("cat file.txt") == _CAT_NUDGE

    def test_multi_file_bails(self) -> None:
        assert _match("cat a.txt b.txt") is None

    def test_flag_bails(self) -> None:
        assert _match("cat -n file.txt") is None

    def test_pipe_to_head(self) -> None:
        assert _match("cat file.txt | head") == _CAT_NUDGE

    def test_pipe_to_tail(self) -> None:
        assert _match("cat file.txt | tail") == _CAT_NUDGE

    def test_pipe_to_head_n(self) -> None:
        assert _match("cat file.txt | head -50") == _CAT_NUDGE

    def test_pipe_to_grep_not_matched(self) -> None:
        # cat | grep is handled by Grep's bash_match, not Read's.
        assert _match("cat file.txt | grep pattern") is None


class TestBashMatchHead:
    def test_no_flags(self) -> None:
        assert _match("head file.txt") == _HEAD_NUDGE

    def test_dash_n(self) -> None:
        assert _match("head -n 20 file.txt") == _HEAD_NUDGE

    def test_dash_num(self) -> None:
        assert _match("head -5 file.txt") == _HEAD_NUDGE

    def test_bytes_flag_bails(self) -> None:
        assert _match("head -c 100 file.txt") is None


class TestBashMatchTail:
    def test_dash_n(self) -> None:
        assert _match("tail -n 10 file.txt") == _TAIL_NUDGE

    def test_dash_num(self) -> None:
        assert _match("tail -20 file.txt") == _TAIL_NUDGE

    def test_default_count(self) -> None:
        assert _match("tail file.txt") == _TAIL_NUDGE

    def test_bytes_flag_bails(self) -> None:
        assert _match("tail -c 100 file.txt") is None


class TestBashMatchCdPrefix:
    # Fixed-string hints - cd prefix no longer affects content.
    def test_cat_joined(self) -> None:
        assert _match("cd src && cat file.py") == _CAT_NUDGE

    def test_head_joined(self) -> None:
        assert _match("cd src && head -5 file.py") == _HEAD_NUDGE

    def test_tail_joined(self) -> None:
        assert _match("cd src && tail -n 20 file.py") == _TAIL_NUDGE


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
