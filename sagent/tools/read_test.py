"""Tests for the Read tool: text, images, PDFs, notebooks, and shell reads."""

from __future__ import annotations

from dataclasses import replace
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING

import json
import os
import time

from PIL import Image

import pypdfium2 as pdfium
import pytest

from sagent.session.types import LineSpan, ReadResult, ToolResult
from sagent.tools.command_analyzer import invocations
from sagent.tools.read import Read, _count
from sagent.tools.types import ToolContext, parse, settled


if TYPE_CHECKING:
    from treekle import PlainTree


def test_summary_names_the_file_and_the_lines_asked_for() -> None:
    tool = Read()
    assert tool.summary({"file_path": "/a/b.py"}) == "Read /a/b.py"
    assert tool.summary({"file_path": "b.py", "offset": 10, "limit": 10}) == (
        "Read b.py:10-19"
    )
    assert tool.summary({"file_path": "b.py", "offset": 10}) == "Read b.py:10+"
    assert tool.summary({"file_path": "b.py", "limit": 5}) == "Read b.py:1-5"
    assert tool.summary({"file_path": "log", "last_lines": 50, "offset": 3}) == (
        "Read log:last-50"
    )
    assert tool.summary({}) == "Read "


def test_reads_of_one_file_share_a_key_however_it_is_named(tmp_path: Path) -> None:
    (tmp_path / "real.py").write_text("x")
    (tmp_path / "link.py").symlink_to(tmp_path / "real.py")
    context = ToolContext(cwd=tmp_path)
    key = Read().serialize_key(context, {"file_path": "real.py"})
    assert key == str(tmp_path / "real.py")
    assert Read().serialize_key(context, {"file_path": "link.py"}) == key
    assert (
        Read().serialize_key(context, {"file_path": str(tmp_path / "real.py")}) == key
    )
    assert Read().serialize_key(context, {}) == str(tmp_path)


@pytest.mark.asyncio
async def test_text_and_notebooks_are_utf8_whatever_the_locale(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "f.txt").write_bytes("café\r\nb\n".encode())
    (tmp_path / "n.ipynb").write_bytes(
        json.dumps({"cells": [{"source": "café"}]}, ensure_ascii=False).encode(),
    )
    real = Path.read_text

    def latin1_default(
        self: Path,
        encoding: str | None = None,
        errors: str | None = None,
    ) -> str:
        return real(
            self,
            encoding=encoding or "latin-1",
            errors=errors,
        )

    monkeypatch.setattr(Path, "read_text", latin1_default)
    assert (await _read(tmp_path, file_path="f.txt")).text() == "1\tcafé\n2\tb\n"
    assert (await _read(tmp_path, file_path="n.ipynb")).text() == (
        "--- Cell 1 (code) ---\ncafé\n"
    )


@pytest.mark.asyncio
async def test_text_comes_back_numbered(tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_text("alpha\nbeta\n")
    context = ToolContext(cwd=tmp_path)
    result = await settled(Read().call(context, {"file_path": "f.txt"}))
    assert result.text() == "1\talpha\n2\tbeta\n"
    assert [item for item in result.stdout if isinstance(item, LineSpan)] == []
    assert result.is_error is False
    assert tmp_path / "f.txt" in context.files


@pytest.mark.asyncio
async def test_a_window_reads_from_offset_for_limit_lines(tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_text("".join(f"line{i}\n" for i in range(1, 11)))
    result = await _read(tmp_path, file_path="f.txt", offset=3, limit=2)
    assert result.text() == (
        "3\tline3\n4\tline4\n\n... (6 more lines. Use offset=5 to continue.)"
    )
    assert result.stdout[1:] == (LineSpan(start=3, count=2),)


@pytest.mark.asyncio
async def test_last_lines_reads_the_tail_and_composes_with_limit(
    tmp_path: Path,
) -> None:
    (tmp_path / "f.txt").write_text("".join(f"line{i}\n" for i in range(1, 11)))
    tail = await _read(tmp_path, file_path="f.txt", last_lines=2, offset=1)
    assert tail.text() == "9\tline9\n10\tline10\n"
    assert tail.stdout[1:] == (LineSpan(start=9, count=2),)
    head_of_tail = await _read(tmp_path, file_path="f.txt", last_lines=4, limit=1)
    assert head_of_tail.text() == (
        "7\tline7\n\n... (3 more lines. Use offset=8 to continue.)"
    )
    whole = await _read(tmp_path, file_path="f.txt", last_lines=99)
    assert whole.text().startswith("1\tline1\n")
    assert [item for item in whole.stdout if isinstance(item, LineSpan)] == []


@pytest.mark.asyncio
async def test_a_full_result_says_where_to_resume(tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_text("".join(f"line{i}\n" for i in range(1, 101)))
    context = ToolContext(cwd=tmp_path, max_result_chars=100)
    result = await settled(Read().call(context, {"file_path": "f.txt"}))
    kept = result.text().count("\tline")
    assert 0 < kept < 100
    assert result.text().endswith(
        f"\n... ({100 - kept} more lines. Use offset={kept + 1} to continue.)",
    )
    assert result.stdout[1:] == (LineSpan(start=1, count=kept),)
    assert len(result.text()) <= context.max_result_chars


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("name", "content", "arguments", "message"),
    [
        (
            "f.txt",
            b"a\nb\n",
            {"offset": 3},
            "[offset 3 is past the end: {} has 2 lines]",
        ),
        ("f.txt", b"", {}, "[File exists but is empty: {}]"),
        ("f.bin", b"\x00\x01\x02", {}, "[Binary file: {} (3 bytes).]"),
        (
            "f.txt",
            b"\xff\xfe bad",
            {},
            "[{} is not UTF-8. Use Bash with an explicit decoder.]",
        ),
    ],
)
async def test_a_file_with_nothing_to_number_says_why(
    tmp_path: Path,
    name: str,
    content: bytes,
    arguments: dict[str, PlainTree],
    message: str,
) -> None:
    (tmp_path / name).write_bytes(content)
    result = await _read(tmp_path, file_path=name, **arguments)
    assert result.text() == message.format(tmp_path / name)
    assert result.is_error is False


@pytest.mark.asyncio
async def test_a_missing_path_or_a_directory_is_an_error(tmp_path: Path) -> None:
    missing = await _read(tmp_path, file_path="nope.txt")
    assert (missing.is_error, missing.result) == (True, ReadResult(exit_code=1))
    assert missing.text() == f"File not found: {tmp_path / 'nope.txt'}"
    directory = await _read(tmp_path, file_path=".")
    assert (directory.is_error, directory.result) == (True, ReadResult(exit_code=1))
    assert "is a directory. Use List or Glob." in directory.text()


@pytest.mark.asyncio
async def test_an_unreadable_file_is_an_error_not_an_exception(tmp_path: Path) -> None:
    locked = tmp_path / "locked.txt"
    locked.write_text("x")
    locked.chmod(0)
    try:
        result = await _read(tmp_path, file_path="locked.txt")
    finally:
        locked.chmod(0o600)
    assert (result.is_error, result.result) == (True, ReadResult(exit_code=1))
    assert result.text().startswith("Error reading locked.txt: ")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "arguments",
    [{"offset": 0}, {"limit": 0}, {"last_lines": 0}, {"pages": 2}],
)
async def test_arguments_out_of_bounds_are_refused(
    tmp_path: Path,
    arguments: dict[str, PlainTree],
) -> None:
    (tmp_path / "f.txt").write_text("x\n")
    with pytest.raises(ValueError, match="must be"):
        await _read(tmp_path, file_path="f.txt", **arguments)


@pytest.mark.asyncio
async def test_the_same_window_of_an_unchanged_file_is_not_sent_twice(
    tmp_path: Path,
) -> None:
    f = tmp_path / "f.txt"
    f.write_text("hi\n")
    context = ToolContext(cwd=tmp_path)
    first = await settled(Read().call(context, {"file_path": "f.txt"}))
    again = await settled(Read().call(context, {"file_path": "f.txt"}))
    other = await settled(Read().call(context, {"file_path": "f.txt", "limit": 5}))
    assert first.text() == "1\thi\n"
    assert again.text() == f"[File unchanged since last read: {f}]"
    assert other.text() == "1\thi\n"
    f.write_text("bye\n")
    later = f.stat().st_mtime + 100
    os.utime(f, (later, later))
    edited = await settled(Read().call(context, {"file_path": "f.txt", "limit": 5}))
    assert edited.text() == "1\tbye\n"


@pytest.mark.asyncio
async def test_a_write_during_the_read_is_not_taken_for_what_was_seen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    f = tmp_path / "f.txt"
    f.write_text("old\n")
    os.utime(f, (1_000_000, 1_000_000))
    original = Path.read_bytes

    def racing_read(self: Path) -> bytes:
        data = original(self)
        if self == f and data == b"old\n":
            f.write_text("new\n")
            os.utime(f, (2_000_000, 2_000_000))
        return data

    context = ToolContext(cwd=tmp_path)
    monkeypatch.setattr(Path, "read_bytes", racing_read)
    first = await settled(Read().call(context, {"file_path": "f.txt"}))
    monkeypatch.setattr(Path, "read_bytes", original)
    second = await settled(Read().call(context, {"file_path": "f.txt"}))
    assert "new" in first.text() + second.text()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("name", "kind", "mime"),
    [
        ("pic.png", "PNG", "image/png"),
        ("pic.JPG", "JPEG", "image/jpeg"),
        ("pic.gif", "GIF", "image/gif"),
        ("pic.bmp", "BMP", "image/bmp"),
        ("pic.webp", "WEBP", "image/webp"),
    ],
)
async def test_an_image_comes_back_as_itself(
    tmp_path: Path,
    name: str,
    kind: str,
    mime: str,
) -> None:
    image = _png(kind)
    (tmp_path / name).write_bytes(image)
    context = ToolContext(cwd=tmp_path)
    result = await settled(Read().call(context, {"file_path": name}))
    assert result.call_id == ""
    assert result.text() == f"[image: {tmp_path / name}]"
    (attachment,) = result.attachments
    assert (attachment.mime_descriptor, attachment.data) == (mime, image)
    assert tmp_path / name in context.files


@pytest.mark.asyncio
async def test_an_svg_is_read_as_text(tmp_path: Path) -> None:
    (tmp_path / "pic.svg").write_text("<svg/>")
    result = await _read(tmp_path, file_path="pic.svg")
    assert result.text() == "1\t<svg/>"
    assert not result.attachments


@pytest.mark.asyncio
async def test_an_image_over_the_cap_is_refused(tmp_path: Path) -> None:
    (tmp_path / "pic.png").write_bytes(_png())
    size = (tmp_path / "pic.png").stat().st_size
    result = await settled(
        Read(max_image_bytes=size - 1).call(
            ToolContext(cwd=tmp_path),
            {"file_path": "pic.png"},
        ),
    )
    assert (result.is_error, result.result) == (True, ReadResult(exit_code=1))
    assert not result.attachments
    assert result.text() == (
        f"{tmp_path / 'pic.png'} is {size:,} bytes, over the {size - 1:,} a read"
        " takes. Shrink it with Bash first."
    )
    exact = await settled(
        Read(max_image_bytes=size).call(
            ToolContext(cwd=tmp_path),
            {"file_path": "pic.png"},
        ),
    )
    assert exact.attachments


@pytest.mark.asyncio
async def test_a_notebook_reads_as_cells_and_their_outputs(tmp_path: Path) -> None:
    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "source": ["print('hi')\n", "1 / 0"],
                "outputs": [
                    {"output_type": "stream", "text": ["hi\n"]},
                    {"output_type": "execute_result", "data": {"text/plain": "42"}},
                    {
                        "output_type": "error",
                        "traceback": ["Trace", "ZeroDivisionError"],
                    },
                    {"output_type": "display_data", "data": "not text"},
                    "bogus",
                ],
            },
            {"cell_type": "markdown", "source": "# heading", "outputs": "bogus"},
            {"source": "x"},
            {"cell_type": "raw", "source": 5},
        ],
    }
    (tmp_path / "n.ipynb").write_text(json.dumps(notebook))
    context = ToolContext(cwd=tmp_path)
    result = await settled(Read().call(context, {"file_path": "n.ipynb"}))
    assert result.text() == (
        "--- Cell 1 (code) ---\nprint('hi')\n1 / 0\n"
        "[output] hi\n\n"
        "[output] 42\n"
        "[output] TraceZeroDivisionError\n"
        "--- Cell 2 (markdown) ---\n# heading\n"
        "--- Cell 3 (code) ---\nx\n"
        "--- Cell 4 (raw) ---\n\n"
    )
    assert tmp_path / "n.ipynb" in context.files


@pytest.mark.asyncio
async def test_an_empty_notebook_says_so(tmp_path: Path) -> None:
    empty: tuple[dict[str, list[object]], ...] = ({"cells": []}, {})
    for notebook in empty:
        (tmp_path / "n.ipynb").write_text(json.dumps(notebook))
        result = await _read(tmp_path, file_path="n.ipynb")
        assert result.text() == "(empty notebook)"
        assert result.is_error is False


@pytest.mark.asyncio
async def test_a_long_notebook_says_how_much_was_left_out(tmp_path: Path) -> None:
    cells = [{"cell_type": "code", "source": f"x = {i}"} for i in range(50)]
    (tmp_path / "n.ipynb").write_text(json.dumps({"cells": cells}))
    context = ToolContext(cwd=tmp_path, max_result_chars=200)
    result = await settled(Read().call(context, {"file_path": "n.ipynb"}))
    kept = result.text().count("--- Cell")
    assert 0 < kept < 50
    shown = result.text().count("x = ")
    assert result.text().endswith(
        f"\n... ({100 - kept - shown} more notebook sections omitted.)",
    )
    assert len(result.text()) <= context.max_result_chars


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        b"not json",
        b"\xff\xfe\x00bad",
        b"[1, 2]",
        b'{"cells": "x"}',
        b'{"cells": ["x"]}',
    ],
    ids=["json", "utf8", "list", "cells", "cell"],
)
async def test_an_unreadable_notebook_is_an_error(
    tmp_path: Path,
    payload: bytes,
) -> None:
    (tmp_path / "n.ipynb").write_bytes(payload)
    result = await _read(tmp_path, file_path="n.ipynb")
    assert (result.is_error, result.result) == (True, ReadResult(exit_code=1))
    assert result.text().startswith(
        f"{tmp_path / 'n.ipynb'} is not a Jupyter notebook: ",
    )


@pytest.mark.asyncio
async def test_a_pdf_comes_back_as_a_jpeg_per_page(tmp_path: Path) -> None:
    _pdf(tmp_path / "doc.pdf", pages=3)
    context = ToolContext(cwd=tmp_path)
    result = await settled(Read().call(context, {"file_path": "doc.pdf"}))
    assert result.call_id == ""
    assert result.text() == "[PDF: doc.pdf, pages 1-3 of 3]"
    assert [a.mime_descriptor for a in result.attachments] == ["image/jpeg"] * 3
    assert all(a.data.startswith(b"\xff\xd8") for a in result.attachments)
    assert tmp_path / "doc.pdf" in context.files


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("pages", "note", "count"),
    [
        ("2", "[PDF: doc.pdf, pages 2-2 of 4]", 1),
        ("2-3", "[PDF: doc.pdf, pages 2-3 of 4]", 2),
        ("3-", "[PDF: doc.pdf, pages 3-4 of 4]", 2),
        (" 1-9999 ", "[PDF: doc.pdf, pages 1-4 of 4]", 4),
    ],
)
async def test_a_page_range_picks_the_pages(
    tmp_path: Path,
    pages: str,
    note: str,
    count: int,
) -> None:
    _pdf(tmp_path / "doc.pdf", pages=4)
    result = await _read(tmp_path, file_path="doc.pdf", pages=pages)
    assert result.text() == note
    assert len(result.attachments) == count


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("pages", "message"),
    [
        ("garbage", 'Invalid pages \'garbage\': use "N", "N-M", or "N-".'),
        ("  ", 'Invalid pages \'  \': use "N", "N-M", or "N-".'),
        ("5", "has no pages 5 (it has 4)."),
        ("0", "has no pages 0 (it has 4)."),
        ("3-2", "has no pages 3-2 (it has 4)."),
        ("5-9", "has no pages 5-9 (it has 4)."),
    ],
)
async def test_a_page_range_the_pdf_cannot_meet_is_an_error(
    tmp_path: Path,
    pages: str,
    message: str,
) -> None:
    _pdf(tmp_path / "doc.pdf", pages=4)
    result = await _read(tmp_path, file_path="doc.pdf", pages=pages)
    assert (result.is_error, result.result) == (True, ReadResult(exit_code=1))
    assert result.text().endswith(message)


@pytest.mark.asyncio
async def test_a_long_pdf_needs_a_range(tmp_path: Path) -> None:
    _pdf(tmp_path / "doc.pdf", pages=3)
    tool = Read(max_pdf_pages=2)
    refused = await settled(
        tool.call(ToolContext(cwd=tmp_path), {"file_path": "doc.pdf"}),
    )
    assert (refused.is_error, refused.result) == (True, ReadResult(exit_code=1))
    assert refused.text().endswith(
        'has 3 pages, over 2. Pass pages="1-N" to read some.',
    )
    ranged = await settled(
        tool.call(
            ToolContext(cwd=tmp_path),
            {"file_path": "doc.pdf", "pages": "1-3"},
        ),
    )
    assert len(ranged.attachments) == 3
    at_cap = await settled(
        Read(max_pdf_pages=3).call(
            ToolContext(cwd=tmp_path),
            {"file_path": "doc.pdf"},
        ),
    )
    assert len(at_cap.attachments) == 3


@pytest.mark.asyncio
async def test_the_byte_budget_stops_at_the_first_page_that_does_not_fit(
    tmp_path: Path,
) -> None:
    """A later small page is not sent past a large one: the pages stay a range."""
    _pdf(tmp_path / "doc.pdf", pages=3, large=frozenset({2}))
    sizes = [
        len(
            (await _read(tmp_path, file_path="doc.pdf", pages=str(n)))
            .attachments[0]
            .data,
        )
        for n in (1, 2, 3)
    ]
    assert sizes[1] > sizes[2]
    result = await settled(
        Read(max_rendered_bytes=sizes[0] + sizes[2]).call(
            ToolContext(cwd=tmp_path),
            {"file_path": "doc.pdf", "pages": "1-3"},
        ),
    )
    assert len(result.attachments) == 1
    assert result.text() == (
        '[PDF: doc.pdf, pages 1-1 of 3] Pass pages="2-3" for the rest.'
    )


@pytest.mark.asyncio
async def test_pages_past_the_byte_budget_are_left_for_another_read(
    tmp_path: Path,
) -> None:
    _pdf(tmp_path / "doc.pdf", pages=4)
    one = await _read(tmp_path, file_path="doc.pdf", pages="1")
    per_page = len(one.attachments[0].data)
    tool = Read(max_rendered_bytes=per_page * 2)
    result = await settled(
        tool.call(
            ToolContext(cwd=tmp_path),
            {"file_path": "doc.pdf", "pages": "1-"},
        ),
    )
    assert len(result.attachments) == 2
    assert result.text() == (
        '[PDF: doc.pdf, pages 1-2 of 4] Pass pages="3-4" for the rest.'
    )
    alone = await settled(
        Read(max_rendered_bytes=per_page - 1).call(
            ToolContext(cwd=tmp_path),
            {"file_path": "doc.pdf", "pages": "2"},
        ),
    )
    assert (alone.is_error, alone.result) == (True, ReadResult(exit_code=1))
    assert (
        alone.text() == f"Page 2 of {tmp_path / 'doc.pdf'} alone is too large to send."
    )


@pytest.mark.asyncio
async def test_a_pdf_that_is_not_one_or_too_big_is_an_error(tmp_path: Path) -> None:
    for name, head in (("fake.pdf", b"not a pdf"), ("almost.pdf", b"%PDFX-1.4")):
        (tmp_path / name).write_bytes(head)
        fake = await _read(tmp_path, file_path=name)
        assert (fake.is_error, fake.result) == (True, ReadResult(exit_code=1))
        assert fake.text().endswith("is not a PDF: it lacks the %PDF- header.")
    (tmp_path / "broken.pdf").write_bytes(b"%PDF-1.4\n%EOF\n")
    broken = await _read(tmp_path, file_path="broken.pdf")
    assert (broken.is_error, broken.result) == (True, ReadResult(exit_code=1))
    assert f"{tmp_path / 'broken.pdf'} is not a readable PDF: " in broken.text()
    _pdf(tmp_path / "doc.pdf", pages=1)
    size = (tmp_path / "doc.pdf").stat().st_size
    big = await settled(
        Read(max_pdf_bytes=size - 1).call(
            ToolContext(cwd=tmp_path),
            {"file_path": "doc.pdf"},
        ),
    )
    assert (big.is_error, big.result) == (True, ReadResult(exit_code=1))
    assert big.text().endswith(
        f"is {size:,} bytes, over the {size - 1:,} a read opens.",
    )
    at_cap = await settled(
        Read(max_pdf_bytes=size).call(
            ToolContext(cwd=tmp_path),
            {"file_path": "doc.pdf"},
        ),
    )
    assert len(at_cap.attachments) == 1


@pytest.mark.parametrize(
    ("command", "arguments"),
    [
        ("cat foo.txt", {"file_path": "foo.txt"}),
        ("cat -n foo.txt", {"file_path": "foo.txt"}),
        ("cat foo.txt | less", {"file_path": "foo.txt"}),
        ("head foo.txt", {"file_path": "foo.txt", "limit": 10}),
        ("head -n 5 foo.txt", {"file_path": "foo.txt", "limit": 5}),
        ("head -5 foo.txt", {"file_path": "foo.txt", "limit": 5}),
        ("head -n5 foo.txt", {"file_path": "foo.txt", "limit": 5}),
        ("head -q -n 3 foo.txt", {"file_path": "foo.txt", "limit": 3}),
        ("head -n 3 -q foo.txt", {"file_path": "foo.txt", "limit": 3}),
        ("head foo.txt -n 3", {"file_path": "foo.txt", "limit": 3}),
        ("head -n 1 foo.txt", {"file_path": "foo.txt", "limit": 1}),
        ("sed -n '1,3p' foo.txt", {"file_path": "foo.txt", "offset": 1, "limit": 3}),
        ("head --lines=5 foo.txt", {"file_path": "foo.txt", "limit": 5}),
        ("head --lines 5 foo.txt", {"file_path": "foo.txt", "limit": 5}),
        ("tail -n 5 foo.txt", {"file_path": "foo.txt", "last_lines": 5}),
        ("tail -n +5 foo.txt", {"file_path": "foo.txt", "offset": 5}),
        ("cat foo.txt | head -20", {"file_path": "foo.txt", "limit": 20}),
        ("cat -n foo.txt | head", {"file_path": "foo.txt", "limit": 10}),
        ("cat foo.txt | tail -5", {"file_path": "foo.txt", "last_lines": 5}),
        ("cat foo.txt | tail -n +3", {"file_path": "foo.txt", "offset": 3}),
        (
            "sed -n '10,20p' foo.txt",
            {"file_path": "foo.txt", "offset": 10, "limit": 11},
        ),
        ("sed -n '10,$p' foo.txt", {"file_path": "foo.txt", "offset": 10}),
        ("sed -n '7p' foo.txt", {"file_path": "foo.txt", "offset": 7, "limit": 1}),
        ("sed -n '7,7p' foo.txt", {"file_path": "foo.txt", "offset": 7, "limit": 1}),
        ("cat /etc/hosts", {"file_path": "/etc/hosts"}),
        ("cat ~/notes.md", {"file_path": "~/notes.md"}),
        ("cat X.txt", {"file_path": "X.txt"}),
    ],
)
def test_a_plain_shell_read_becomes_the_read_that_shows_the_same_lines(
    command: str,
    arguments: dict[str, PlainTree],
) -> None:
    assert _from_shell(command) == arguments
    parse(arguments, Read.Arguments)


@pytest.mark.parametrize(
    "command",
    [
        "cat a b",
        "cat -s foo.txt",
        "head -5 foo.txt | tail -2",
        "head -n -5 foo.txt",
        "head -n 0 foo.txt",
        "head -n abc foo.txt",
        "tail -n 0 foo.txt",
        "cat foo.txt | head -n 0",
        "sed -n '0p' foo.txt",
        "sed -n '5,3p' foo.txt",
        "sed -n '1p' a b",
        "cat *.py",
        "cat $f",
        "for f in *.py; do cat $f; done",
        "cat foo.txt | head -5 | head -2",
        # ``-`` is the input, which for a Bash call is empty, not a file.
        "cat -",
        "head -",
        "tail -n 3 -",
        "sed -n '1p' -",
        # After ``--`` a word is a file, and here that file is named ``-n``.
        "cat -- -n",
    ],
)
def test_a_read_no_call_matches_is_refused_with_advice(command: str) -> None:
    assert _from_shell(command) == (
        "Read each file with the Read tool, all in one block: one round trip,"
        " and lines numbered to cite."
    )


def test_a_read_after_an_unknowable_cd_is_refused() -> None:
    found = invocations("cd - && cat foo.txt")
    assert found is not None
    assert isinstance(Read().from_shell(found[0]), str)


def test_a_read_after_a_known_cd_names_the_file_from_there() -> None:
    found = invocations("cd /srv && cat foo.txt")
    assert found is not None
    assert Read().from_shell(found[0]) == {"file_path": "/srv/foo.txt"}


@pytest.mark.parametrize(
    "command",
    [
        "ls -la",
        "cat",
        "head",
        "head --bytes 5 foo.txt",
        "head -c 5 foo.txt",
        "tail -f /var/log/syslog",
        "cat -A foo.txt",
        "cat foo.txt | sort",
        "cat foo.txt > out",
        "FOO=1 cat foo.txt",
        "echo x | cat",
        "sed 's/a/b/' foo.txt",
        "sed -n '/x/p' foo.txt",
        "sed -n -i '1p' foo.txt",
        "sed -n '1p;2p' foo.txt",
        # One deny list serves every read, and ``cat -e`` marks line ends.
        "sed -n -e '2,3p' foo.txt",
        "sed '1p' notes.txt",
        "sed --line-length=9 '1p' foo.txt",
        # The word after each flag is its value, so neither command names a file.
        "sed -n --expression '1p'",
        "sed -n --file 1p",
        "head -n 5",
        "head --lines 5",
        "tail -n 3",
        *(
            f"{exe} {flag} foo.txt"
            for exe, flag in (
                ("tail", "-F"),
                ("tail", "--follow"),
                ("tail", "--retry"),
                ("cat", "-v"),
                ("cat", "-e"),
                ("cat", "-E"),
                ("cat", "-t"),
                ("cat", "-T"),
                ("cat", "--show-all"),
                ("cat", "--show-nonprinting"),
            )
        ),
    ],
)
def test_a_command_that_is_no_plain_read_is_not_reads_job(command: str) -> None:
    assert _from_shell(command) is None


@pytest.mark.parametrize(
    ("command", "arguments"),
    [
        ("head -n", None),
        ("head -3 foo.txt", {"file_path": "foo.txt", "limit": 3}),
        ("head --lines=+3 foo.txt", {"file_path": "foo.txt", "limit": 3}),
        ("tail --lines=+3 foo.txt", {"file_path": "foo.txt", "offset": 3}),
        ("tail -n+4 foo.txt", {"file_path": "foo.txt", "offset": 4}),
        ("tail -5 foo.txt", {"file_path": "foo.txt", "last_lines": 5}),
        ("tail -n 2 -n 6 foo.txt", {"file_path": "foo.txt", "last_lines": 6}),
        ("tail -n +2 -6 foo.txt", {"file_path": "foo.txt", "last_lines": 6}),
        ("sed -n '07,7p' foo.txt", {"file_path": "foo.txt", "offset": 7, "limit": 1}),
        ("head foo.txt -n", {"file_path": "foo.txt", "limit": 10}),
        ("tail foo.txt -n", {"file_path": "foo.txt", "last_lines": 10}),
        ("head --lines=1=2 foo.txt", "refused"),
        ("head - foo.txt", "refused"),
        ("tail -n -3 foo.txt", {"file_path": "foo.txt", "last_lines": 3}),
        # A count's letters are no digits: an ``X`` in it reads as no count.
        ("head -n X3 foo.txt", "refused"),
        ("cat -s foo.txt", "refused"),
    ],
)
def test_each_spelling_of_a_count_and_its_sign_is_read(
    command: str,
    arguments: dict[str, PlainTree] | str | None,
) -> None:
    refused = (
        "Read each file with the Read tool, all in one block: one round trip,"
        " and lines numbered to cite."
    )
    assert _from_shell(command) == (refused if arguments == "refused" else arguments)


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (("foo.txt",), (["foo.txt"], 10, "")),
        (("-5", "foo.txt"), (["foo.txt"], 5, "")),
        (("-n", "5", "foo.txt"), (["foo.txt"], 5, "")),
        (("-n", "+5", "foo.txt"), (["foo.txt"], 5, "+")),
        (("-n", "-5", "foo.txt"), (["foo.txt"], 5, "-")),
    ],
    ids=["no-count", "bare-number", "unsigned", "plus", "minus"],
)
def test_a_count_is_returned_with_the_operands_and_its_sign(
    args: tuple[str, ...],
    expected: tuple[list[str], int, str],
) -> None:
    assert _count(args) == expected


@pytest.mark.parametrize("command", ["head -n", "tail -n"])
def test_a_count_flag_with_no_word_after_it_reads_nothing(command: str) -> None:
    assert _from_shell(command) is None


@pytest.mark.parametrize(
    "command",
    ["cat 'a`b'", "cat '[ab]'", "cat 'a?'", "cat 'a$'", "cat 'a*'"],
)
def test_a_path_glob_or_expansion_would_read_is_refused(command: str) -> None:
    assert _from_shell(command) == (
        "Read each file with the Read tool, all in one block: one round trip,"
        " and lines numbered to cite."
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(("at", "binary"), [(8191, True), (8192, False)])
async def test_a_nul_in_the_first_8192_bytes_marks_the_file_binary(
    tmp_path: Path,
    at: int,
    binary: bool,
) -> None:
    (tmp_path / "f").write_bytes(b"a" * at + b"\x00")
    result = await _read(tmp_path, file_path="f")
    assert result.text().startswith("[Binary file") is binary


@pytest.mark.asyncio
async def test_line_ends_read_as_a_text_file_s_do(tmp_path: Path) -> None:
    (tmp_path / "f").write_bytes(b"a\r\nb\rc\n")
    assert (await _read(tmp_path, file_path="f")).text() == "1\ta\n2\tb\n3\tc\n"


@pytest.mark.asyncio
async def test_a_read_unchanged_since_the_last_is_said_so_for_no_call(
    tmp_path: Path,
) -> None:
    (tmp_path / "f").write_text("a\n")
    context = ToolContext(cwd=tmp_path)
    _ = await settled(Read().call(context, {"file_path": "f"}))
    again = await settled(Read().call(context, {"file_path": "f"}))
    assert again.text().startswith("[File unchanged")
    assert again.call_id == ""


@pytest.mark.asyncio
async def test_a_cut_read_never_passes_its_cap(tmp_path: Path) -> None:
    # Resuming at 10,000 adds the digit the note must fit.
    (tmp_path / "f").write_text("".join(f"{i}\n" for i in range(10_000)))
    numbered = [f"{n}\t{n - 1}\n" for n in range(9_990, 10_001)]

    def cut(kept: int) -> str:
        left, resume = 11 - kept, 9_990 + kept
        note = f"\n... ({left:,} more lines. Use offset={resume} to continue.)"
        return "".join(numbered[:kept]) + note

    whole = "".join(numbered)
    thresholds = [*(len(cut(k)) for k in range(1, 11)), len(whole)]
    caps = {threshold + delta for threshold in thresholds for delta in (-1, 0, 1)}
    for cap in sorted(c for c in caps if c >= len(cut(1))):
        context = ToolContext(cwd=tmp_path, max_result_chars=cap)
        text = (
            await settled(Read().call(context, {"file_path": "f", "offset": 9_990}))
        ).text()
        assert len(text) <= cap, cap
        kept = text.count("\t")
        if kept < 11:
            assert text == cut(kept), cap
            assert kept == 10 or len(cut(kept + 1)) > cap, cap
        else:
            assert text == whole, cap
    context = ToolContext(cwd=tmp_path, max_result_chars=len(cut(1)) - 1)
    result = await settled(Read().call(context, {"file_path": "f", "offset": 9_990}))
    assert result.text() == cut(1)


@pytest.mark.asyncio
async def test_a_whole_read_that_fits_its_cap_keeps_every_line(tmp_path: Path) -> None:
    (tmp_path / "f").write_text("a\nb\n")
    context = ToolContext(cwd=tmp_path, max_result_chars=len("1\ta\n2\tb\n"))
    result = await settled(Read().call(context, {"file_path": "f"}))
    assert result.text() == "1\ta\n2\tb\n"


@pytest.mark.asyncio
async def test_a_page_is_rendered_at_twice_72_dpi_as_a_quality_85_jpeg(
    tmp_path: Path,
) -> None:
    _pdf(tmp_path / "doc.pdf", pages=1)
    result = await _read(tmp_path, file_path="doc.pdf")
    (attachment,) = result.attachments
    doc = pdfium.PdfDocument(str(tmp_path / "doc.pdf"))
    try:
        page = doc[0]
        image = page.render(scale=2).to_pil()
        page.close()
    finally:
        doc.close()
    want = BytesIO()
    image.save(want, format="JPEG", quality=85, optimize=True)
    assert attachment.data == want.getvalue()
    assert Image.open(BytesIO(attachment.data)).size == (72, 72)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("days", "note"),
    [
        (0, ""),
        (1, ""),
        (
            2,
            (
                "[This memory is 2 days old. Memories are point-in-time observations,"
                " not live state: claims about code behavior or file:line citations may"
                " be outdated. Verify against current code before asserting as fact.]\n"
            ),
        ),
    ],
)
async def test_a_memory_older_than_a_day_says_how_old_it_is(
    tmp_path: Path,
    days: int,
    note: str,
) -> None:
    """Models do not reason about staleness from a date the way they do from an age."""
    (tmp_path / "MEMORY.md").write_text("- [role](user_role_lead.md) -- Leads\n")
    memory = tmp_path / "user_role_lead.md"
    memory.write_text("Leads the team.\n")
    then = time.time() - days * 86_400 - 1
    os.utime(memory, (then, then))
    context = ToolContext(cwd=tmp_path)
    result = await settled(Read().call(context, {"file_path": "user_role_lead.md"}))
    again = await settled(Read().call(context, {"file_path": "user_role_lead.md"}))
    assert result.text() == f"{note}1\tLeads the team.\n"
    assert again.text().startswith("[File unchanged since last read: ")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "name",
    ["user_role.md", "notes_role.md", "user_role.txt"],
    ids=["no-index", "no-type", "not-markdown"],
)
async def test_a_file_that_is_no_memory_carries_no_age(
    tmp_path: Path,
    name: str,
) -> None:
    if name != "user_role.md":
        (tmp_path / "MEMORY.md").write_text("")
    (tmp_path / name).write_text("x\n")
    os.utime(tmp_path / name, (0, 0))
    assert (await _read(tmp_path, file_path=name)).text() == "1\tx\n"


async def _read(
    cwd: Path,
    **arguments: PlainTree,
) -> ToolResult:
    """Read with ``arguments`` in a fresh context at ``cwd``; the runtime names the call."""
    streaming_result = Read().call(ToolContext(cwd=cwd), arguments)
    result = await settled(streaming_result)
    # ``settled`` records a raise instead of raising; a refusal test expects the raise.
    _ = await streaming_result.result
    assert result.call_id == ""
    return result


def _from_shell(command: str) -> PlainTree:
    """Return what Read makes of the first command in ``command`` but ``echo``."""
    found = invocations(command)
    assert found is not None
    result = Read().from_shell(next(i for i in found if i.exe != "echo"))
    return result if result is None or isinstance(result, str) else dict(result)


def _png(kind: str = "PNG") -> bytes:
    """Return a 4x4 red image, as a PNG unless ``kind`` names another format."""
    buf = BytesIO()
    Image.new("RGB", (4, 4), (255, 0, 0)).save(buf, format=kind)
    return buf.getvalue()


def _pdf(path: Path, *, pages: int, large: frozenset[int] = frozenset()) -> None:
    """Write a renderable PDF of ``pages`` blank pages; those in ``large`` are big."""
    doc = pdfium.PdfDocument.new()
    try:
        for n in range(1, pages + 1):
            side = 400 if n in large else 36
            doc.new_page(side, side)
        doc.save(str(path))
    finally:
        doc.close()


@pytest.mark.asyncio
async def test_a_read_says_which_file_and_why_it_showed_no_lines(
    tmp_path: Path,
) -> None:
    (tmp_path / "a.txt").write_text("one\ntwo\n")
    (tmp_path / "empty.txt").write_text("")
    (tmp_path / "bin").write_bytes(b"\x00\x01")
    (tmp_path / "latin").write_bytes(b"\xe9t\xe9")
    (tmp_path / "p.png").write_bytes(_png())
    (tmp_path / "n.ipynb").write_text(json.dumps({"cells": []}))
    _pdf(tmp_path / "d.pdf", pages=1)
    context = ToolContext(cwd=tmp_path)

    async def read(name: str, **arguments: PlainTree) -> tuple[object, bool]:
        done = await settled(Read().call(context, {"file_path": name, **arguments}))
        return done.result, done.is_error

    text = ReadResult(path=str(tmp_path / "a.txt"))
    assert await read("a.txt") == (text, False)
    assert await read("a.txt") == (replace(text, status="unchanged"), False)
    assert await read("a.txt", offset=9) == (replace(text, status="past_end"), False)
    assert await read("empty.txt") == (
        ReadResult(path=str(tmp_path / "empty.txt"), status="empty"),
        False,
    )
    assert await read("bin") == (
        ReadResult(path=str(tmp_path / "bin"), status="binary"),
        False,
    )
    assert await read("latin") == (
        ReadResult(path=str(tmp_path / "latin"), status="undecodable"),
        False,
    )
    for name in ("p.png", "n.ipynb", "d.pdf"):
        assert await read(name) == (ReadResult(path=str(tmp_path / name)), False)
    assert await read("missing") == (ReadResult(exit_code=1), True)


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
