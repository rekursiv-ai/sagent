"""Tests for the Read tool: text, images, PDFs, notebooks, and shell reads."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING

import json
import os

from PIL import Image

import pypdfium2 as pdfium
import pytest

from sagent.session.types import FileReadResult, UncategorizedToolResult
from sagent.tools.command_analyzer import invocations
from sagent.tools.read import Read
from sagent.tools.types import ToolContext, parse


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
    result = await Read().call(context, {"file_path": "f.txt"})
    assert isinstance(result, FileReadResult)
    assert result.content == "1\talpha\n2\tbeta\n"
    assert result.path == str(tmp_path / "f.txt")
    assert result.ranges == ()
    assert result.is_error is None
    assert tmp_path / "f.txt" in context.files


@pytest.mark.asyncio
async def test_a_window_reads_from_offset_for_limit_lines(tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_text("".join(f"line{i}\n" for i in range(1, 11)))
    result = await _read(tmp_path, file_path="f.txt", offset=3, limit=2)
    assert isinstance(result, FileReadResult)
    assert result.content == (
        "3\tline3\n4\tline4\n\n... (6 more lines. Use offset=5 to continue.)"
    )
    assert result.ranges == ((3, 2),)


@pytest.mark.asyncio
async def test_last_lines_reads_the_tail_and_composes_with_limit(
    tmp_path: Path,
) -> None:
    (tmp_path / "f.txt").write_text("".join(f"line{i}\n" for i in range(1, 11)))
    tail = await _read(tmp_path, file_path="f.txt", last_lines=2, offset=1)
    assert tail.text() == "9\tline9\n10\tline10\n"
    assert isinstance(tail, FileReadResult)
    assert tail.ranges == ((9, 2),)
    head_of_tail = await _read(tmp_path, file_path="f.txt", last_lines=4, limit=1)
    assert head_of_tail.text() == (
        "7\tline7\n\n... (3 more lines. Use offset=8 to continue.)"
    )
    whole = await _read(tmp_path, file_path="f.txt", last_lines=99)
    assert whole.text().startswith("1\tline1\n")
    assert isinstance(whole, FileReadResult)
    assert whole.ranges == ()


@pytest.mark.asyncio
async def test_a_full_result_says_where_to_resume(tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_text("".join(f"line{i}\n" for i in range(1, 101)))
    context = ToolContext(cwd=tmp_path, max_result_chars=100)
    result = await Read().call(context, {"file_path": "f.txt"})
    assert isinstance(result, FileReadResult)
    kept = result.text().count("\tline")
    assert 0 < kept < 100
    assert result.text().endswith(
        f"\n... ({100 - kept} more lines. Use offset={kept + 1} to continue.)",
    )
    assert result.ranges == ((1, kept),)
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
    assert isinstance(result, UncategorizedToolResult)
    assert result.text() == message.format(tmp_path / name)
    assert result.is_error is None


@pytest.mark.asyncio
async def test_a_missing_path_or_a_directory_is_an_error(tmp_path: Path) -> None:
    missing = await _read(tmp_path, file_path="nope.txt")
    assert missing.is_error is True
    assert missing.text() == f"File not found: {tmp_path / 'nope.txt'}"
    directory = await _read(tmp_path, file_path=".")
    assert directory.is_error is True
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
    assert result.is_error is True
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
    first = await Read().call(context, {"file_path": "f.txt"})
    again = await Read().call(context, {"file_path": "f.txt"})
    other = await Read().call(context, {"file_path": "f.txt", "limit": 5})
    assert first.text() == "1\thi\n"
    assert again.text() == f"[File unchanged since last read: {f}]"
    assert other.text() == "1\thi\n"
    f.write_text("bye\n")
    later = f.stat().st_mtime + 100
    os.utime(f, (later, later))
    edited = await Read().call(context, {"file_path": "f.txt", "limit": 5})
    assert edited.text() == "1\tbye\n"


@pytest.mark.asyncio
async def test_a_write_during_the_read_is_not_taken_for_what_was_seen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    f = tmp_path / "f.txt"
    f.write_text("old\n")
    os.utime(f, (1_000_000, 1_000_000))
    original = Path.read_text

    def racing_read(
        self: Path,
        encoding: str | None = None,
        errors: str | None = None,
    ) -> str:
        text = original(self, encoding=encoding, errors=errors)
        if self == f and text == "old\n":
            f.write_text("new\n")
            os.utime(f, (2_000_000, 2_000_000))
        return text

    context = ToolContext(cwd=tmp_path)
    monkeypatch.setattr(Path, "read_text", racing_read)
    first = await Read().call(context, {"file_path": "f.txt"})
    monkeypatch.setattr(Path, "read_text", original)
    second = await Read().call(context, {"file_path": "f.txt"})
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
    result = await Read().call(context, {"file_path": name})
    assert isinstance(result, FileReadResult)
    assert result.call_id == ""
    assert result.content == f"[image: {tmp_path / name}]"
    assert result.path == str(tmp_path / name)
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
    result = await Read(max_image_bytes=size - 1).call(
        ToolContext(cwd=tmp_path),
        {"file_path": "pic.png"},
    )
    assert result.is_error is True
    assert not result.attachments
    assert result.text() == (
        f"{tmp_path / 'pic.png'} is {size:,} bytes, over the {size - 1:,} a read"
        " takes. Shrink it with Bash first."
    )
    exact = await Read(max_image_bytes=size).call(
        ToolContext(cwd=tmp_path),
        {"file_path": "pic.png"},
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
    result = await Read().call(context, {"file_path": "n.ipynb"})
    assert isinstance(result, FileReadResult)
    assert result.path == str(tmp_path / "n.ipynb")
    assert result.content == (
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
        assert result.is_error is None


@pytest.mark.asyncio
async def test_a_long_notebook_says_how_much_was_left_out(tmp_path: Path) -> None:
    cells = [{"cell_type": "code", "source": f"x = {i}"} for i in range(50)]
    (tmp_path / "n.ipynb").write_text(json.dumps({"cells": cells}))
    context = ToolContext(cwd=tmp_path, max_result_chars=200)
    result = await Read().call(context, {"file_path": "n.ipynb"})
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
    assert result.is_error is True
    assert result.text().startswith(
        f"{tmp_path / 'n.ipynb'} is not a Jupyter notebook: ",
    )


@pytest.mark.asyncio
async def test_a_pdf_comes_back_as_a_jpeg_per_page(tmp_path: Path) -> None:
    _pdf(tmp_path / "doc.pdf", pages=3)
    context = ToolContext(cwd=tmp_path)
    result = await Read().call(context, {"file_path": "doc.pdf"})
    assert isinstance(result, FileReadResult)
    assert result.call_id == ""
    assert result.content == "[PDF: doc.pdf, pages 1-3 of 3]"
    assert result.path == str(tmp_path / "doc.pdf")
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
    assert result.is_error is True
    assert result.text().endswith(message)


@pytest.mark.asyncio
async def test_a_long_pdf_needs_a_range(tmp_path: Path) -> None:
    _pdf(tmp_path / "doc.pdf", pages=3)
    tool = Read(max_pdf_pages=2)
    refused = await tool.call(ToolContext(cwd=tmp_path), {"file_path": "doc.pdf"})
    assert refused.is_error is True
    assert refused.text().endswith(
        'has 3 pages, over 2. Pass pages="1-N" to read some.',
    )
    ranged = await tool.call(
        ToolContext(cwd=tmp_path),
        {"file_path": "doc.pdf", "pages": "1-3"},
    )
    assert len(ranged.attachments) == 3
    at_cap = await Read(max_pdf_pages=3).call(
        ToolContext(cwd=tmp_path),
        {"file_path": "doc.pdf"},
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
    result = await Read(max_rendered_bytes=sizes[0] + sizes[2]).call(
        ToolContext(cwd=tmp_path),
        {"file_path": "doc.pdf", "pages": "1-3"},
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
    result = await tool.call(
        ToolContext(cwd=tmp_path),
        {"file_path": "doc.pdf", "pages": "1-"},
    )
    assert len(result.attachments) == 2
    assert result.text() == (
        '[PDF: doc.pdf, pages 1-2 of 4] Pass pages="3-4" for the rest.'
    )
    alone = await Read(max_rendered_bytes=per_page - 1).call(
        ToolContext(cwd=tmp_path),
        {"file_path": "doc.pdf", "pages": "2"},
    )
    assert alone.is_error is True
    assert (
        alone.text() == f"Page 2 of {tmp_path / 'doc.pdf'} alone is too large to send."
    )


@pytest.mark.asyncio
async def test_a_pdf_that_is_not_one_or_too_big_is_an_error(tmp_path: Path) -> None:
    for name, head in (("fake.pdf", b"not a pdf"), ("almost.pdf", b"%PDFX-1.4")):
        (tmp_path / name).write_bytes(head)
        fake = await _read(tmp_path, file_path=name)
        assert fake.is_error is True
        assert fake.text().endswith("is not a PDF: it lacks the %PDF- header.")
    (tmp_path / "broken.pdf").write_bytes(b"%PDF-1.4\n%EOF\n")
    broken = await _read(tmp_path, file_path="broken.pdf")
    assert broken.is_error is True
    assert f"{tmp_path / 'broken.pdf'} is not a readable PDF: " in broken.text()
    _pdf(tmp_path / "doc.pdf", pages=1)
    size = (tmp_path / "doc.pdf").stat().st_size
    big = await Read(max_pdf_bytes=size - 1).call(
        ToolContext(cwd=tmp_path),
        {"file_path": "doc.pdf"},
    )
    assert big.is_error is True
    assert big.text().endswith(
        f"is {size:,} bytes, over the {size - 1:,} a read opens.",
    )
    at_cap = await Read(max_pdf_bytes=size).call(
        ToolContext(cwd=tmp_path),
        {"file_path": "doc.pdf"},
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


async def _read(
    cwd: Path,
    **arguments: PlainTree,
) -> FileReadResult | UncategorizedToolResult:
    """Read with ``arguments`` in a fresh context at ``cwd``; the runtime names the call."""
    result = await Read().call(ToolContext(cwd=cwd), arguments)
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


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
