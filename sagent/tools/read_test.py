"""Tests for ``tools.read``: text/image/PDF/notebook reading."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import json
import os

from PIL import Image

import pypdfium2 as pdfium
import pytest

from sagent.testing import FakeAgent, with_fake_agent
from sagent.tools.lib.bash import parse_bash
from sagent.tools.lib.pdf import MAX_PDF_BYTES, extract_pdf_pages
from sagent.tools.read import (
    _ASSUMED_CHARS_PER_LINE,
    Read,
    _default_line_limit,
)


read = Read()


def _png_bytes() -> bytes:
    buf = BytesIO()
    Image.new("RGB", (4, 4), (255, 0, 0)).save(buf, format="PNG")
    return buf.getvalue()


@pytest.mark.asyncio
async def test_read_basic_text(tmp_path: Path) -> None:
    f = tmp_path / "a.txt"
    f.write_text("line0\nline1\nline2\n")
    with with_fake_agent() as agent:
        agent.tool_state.bash_cwd = str(tmp_path)
        result = await read.run({"file_path": str(f)})
    assert "1\tline0" in result.content
    assert "3\tline2" in result.content


@pytest.mark.asyncio
async def test_read_relative_uses_bash_cwd(tmp_path: Path) -> None:
    f = tmp_path / "rel.txt"
    f.write_text("cwd file\n")
    with with_fake_agent() as agent:
        agent.tool_state.bash_cwd = str(tmp_path)
        result = await read.run({"file_path": "rel.txt"})
    assert "1\tcwd file" in result.content


@pytest.mark.asyncio
async def test_read_offset_limit(tmp_path: Path) -> None:
    f = tmp_path / "f.txt"
    f.write_text("a\nb\nc\nd\ne\n")
    with with_fake_agent() as agent:
        agent.tool_state.bash_cwd = str(tmp_path)
        result = await read.run({"file_path": str(f), "offset": 2, "limit": 2})
    assert "2\tb" in result.content
    assert "3\tc" in result.content
    assert "1\ta" not in result.content


@pytest.mark.asyncio
async def test_read_more_lines_tail(tmp_path: Path) -> None:
    f = tmp_path / "f.txt"
    f.write_text("a\nb\nc\nd\ne\n")
    with with_fake_agent() as agent:
        agent.tool_state.bash_cwd = str(tmp_path)
        result = await read.run({"file_path": str(f), "last_lines": 2})
    assert "4\td" in result.content
    assert "5\te" in result.content
    assert "1\ta" not in result.content


@pytest.mark.asyncio
async def test_read_truncation_marker(tmp_path: Path) -> None:
    f = tmp_path / "f.txt"
    f.write_text("\n".join(f"L{i}" for i in range(100)) + "\n")
    with with_fake_agent() as agent:
        agent.tool_state.bash_cwd = str(tmp_path)
        result = await read.run({"file_path": str(f), "limit": 5})
    assert "more lines" in result.content


@pytest.mark.asyncio
async def test_read_offset_beyond_eof(tmp_path: Path) -> None:
    f = tmp_path / "f.txt"
    f.write_text("a\nb\n")
    with with_fake_agent() as agent:
        agent.tool_state.bash_cwd = str(tmp_path)
        result = await read.run({"file_path": str(f), "offset": 99})
    assert "beyond EOF" in result.content


@pytest.mark.asyncio
async def test_read_empty_file(tmp_path: Path) -> None:
    f = tmp_path / "empty.txt"
    f.write_text("")
    with with_fake_agent() as agent:
        agent.tool_state.bash_cwd = str(tmp_path)
        result = await read.run({"file_path": str(f)})
    assert "empty" in result.content


@pytest.mark.asyncio
async def test_read_not_found(tmp_path: Path) -> None:
    with with_fake_agent() as agent:
        agent.tool_state.bash_cwd = str(tmp_path)
        result = await read.run({"file_path": str(tmp_path / "missing.txt")})
    assert result.is_error
    assert "not found" in result.content.lower()


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["offset", "limit", "last_lines"])
@pytest.mark.parametrize("value", [0, -1, -5])
async def test_read_rejects_subminimum_windowing(
    field: str,
    value: int,
    tmp_path: Path,
) -> None:
    """Schema declares ``minimum: 1``; runtime must enforce.

    Pre-fix ``offset=0`` fell through to ``max(1, offset)`` and
    silently became ``offset=1``; ``limit=0`` was treated as
    "unbounded"; negatives produced garbage windows. All three are
    schema violations.
    """
    f = tmp_path / "a.txt"
    f.write_text("a\nb\nc\n")
    with with_fake_agent() as agent:
        agent.tool_state.bash_cwd = str(tmp_path)
        result = await read.run({"file_path": str(f), field: value})
    assert result.is_error
    assert field in result.content


@pytest.mark.asyncio
async def test_read_directory_errors(tmp_path: Path) -> None:
    sub = tmp_path / "dir"
    sub.mkdir()
    with with_fake_agent() as agent:
        agent.tool_state.bash_cwd = str(tmp_path)
        result = await read.run({"file_path": str(sub)})
    assert result.is_error
    assert "directory" in result.content


@pytest.mark.asyncio
async def test_read_binary_file(tmp_path: Path) -> None:
    f = tmp_path / "bin.dat"
    f.write_bytes(b"\x00\x01\x02hello")
    with with_fake_agent() as agent:
        agent.tool_state.bash_cwd = str(tmp_path)
        result = await read.run({"file_path": str(f)})
    assert "Binary file" in result.content


@pytest.mark.asyncio
async def test_read_invalid_utf8(tmp_path: Path) -> None:
    f = tmp_path / "bad.txt"
    f.write_bytes(b"\xff\xfe" + b"a" * 100)
    with with_fake_agent() as agent:
        agent.tool_state.bash_cwd = str(tmp_path)
        result = await read.run({"file_path": str(f)})
    assert "Non-UTF-8" in result.content


@pytest.mark.asyncio
async def test_read_unchanged_dedup(tmp_path: Path) -> None:
    f = tmp_path / "f.txt"
    f.write_text("hi\n")
    with with_fake_agent() as agent:
        agent.tool_state.bash_cwd = str(tmp_path)
        first = await read.run({"file_path": str(f)})
        assert "hi" in first.content
        second = await read.run({"file_path": str(f)})
    assert "File unchanged" in second.content


@pytest.mark.asyncio
async def test_read_png_image(tmp_path: Path) -> None:
    f = tmp_path / "image.png"
    f.write_bytes(_png_bytes())
    with with_fake_agent() as agent:
        agent.tool_state.bash_cwd = str(tmp_path)
        result = await read.run({"file_path": str(f)})
    assert result.attachments
    assert result.attachments[0].descriptor == "image/png"
    assert "[image:" in result.content


@pytest.mark.asyncio
async def test_read_svg_as_text(tmp_path: Path) -> None:
    f = tmp_path / "image.svg"
    f.write_text("<svg/>")
    with with_fake_agent() as agent:
        agent.tool_state.bash_cwd = str(tmp_path)
        result = await read.run({"file_path": str(f)})
    assert not result.attachments
    assert "<svg/>" in result.content


@pytest.mark.asyncio
async def test_read_notebook(tmp_path: Path) -> None:
    nb = {
        "cells": [
            {
                "cell_type": "code",
                "source": ["print('hi')\n"],
                "outputs": [{"text": ["hi\n"]}],
            },
            {
                "cell_type": "markdown",
                "source": "# heading",
                "outputs": [{"text": "no list"}],
            },
        ],
    }
    f = tmp_path / "nb.ipynb"
    f.write_text(json.dumps(nb))
    with with_fake_agent() as agent:
        agent.tool_state.bash_cwd = str(tmp_path)
        result = await read.run({"file_path": str(f)})
    assert "Cell 1" in result.content
    assert "print('hi')" in result.content
    assert "[output] hi" in result.content
    assert "# heading" in result.content


@pytest.mark.asyncio
async def test_read_notebook_invalid_json(tmp_path: Path) -> None:
    f = tmp_path / "bad.ipynb"
    f.write_text("not json")
    with with_fake_agent() as agent:
        agent.tool_state.bash_cwd = str(tmp_path)
        result = await read.run({"file_path": str(f)})
    assert "Invalid notebook JSON" in result.content


@pytest.mark.asyncio
async def test_read_notebook_non_utf8(tmp_path: Path) -> None:
    f = tmp_path / "bad.ipynb"
    f.write_bytes(b"\xff\xfe\xfd_not_utf8")
    with with_fake_agent() as agent:
        agent.tool_state.bash_cwd = str(tmp_path)
        result = await read.run({"file_path": str(f)})
    assert "Non-UTF-8 notebook" in result.content


@pytest.mark.asyncio
async def test_read_notebook_non_dict(tmp_path: Path) -> None:
    f = tmp_path / "bad.ipynb"
    f.write_text("[1, 2, 3]")
    with with_fake_agent() as agent:
        agent.tool_state.bash_cwd = str(tmp_path)
        result = await read.run({"file_path": str(f)})
    assert "Not a valid Jupyter notebook" in result.content


@pytest.mark.asyncio
async def test_read_notebook_empty(tmp_path: Path) -> None:
    f = tmp_path / "empty.ipynb"
    f.write_text(json.dumps({"cells": []}))
    with with_fake_agent() as agent:
        agent.tool_state.bash_cwd = str(tmp_path)
        result = await read.run({"file_path": str(f)})
    assert result.content == "(empty notebook)"


@pytest.mark.asyncio
async def test_read_pdf_missing_header(tmp_path: Path) -> None:
    f = tmp_path / "fake.pdf"
    f.write_text("not a pdf")
    with with_fake_agent() as agent:
        agent.tool_state.bash_cwd = str(tmp_path)
        result = await read.run({"file_path": str(f)})
    assert result.is_error
    assert "Not a PDF" in result.content


@pytest.mark.asyncio
async def test_read_pdf_invalid_page_range(tmp_path: Path) -> None:
    # File has the PDF header so it passes ``is_pdf``, then fails page-range parse.
    f = tmp_path / "x.pdf"
    f.write_bytes(b"%PDF-1.4\n%EOF")
    with with_fake_agent() as agent:
        agent.tool_state.bash_cwd = str(tmp_path)
        result = await read.run({"file_path": str(f), "pages": "garbage"})
    assert result.is_error
    assert "Invalid pages spec" in result.content


@pytest.mark.asyncio
async def test_read_pdf_extract_error(tmp_path: Path) -> None:
    """Malformed PDF surfaces an ``extract_pdf_pages`` PdfError."""
    f = tmp_path / "x.pdf"
    # Has %PDF- header but no actual page structure; pdftoppm errors.
    f.write_bytes(b"%PDF-1.4\n%EOF\n")
    with with_fake_agent() as agent:
        agent.tool_state.bash_cwd = str(tmp_path)
        result = await read.run({"file_path": str(f), "pages": "1"})
    assert result.is_error
    assert "corrupt or invalid PDF" in result.content


@pytest.mark.asyncio
async def test_read_pdf_too_large(tmp_path: Path) -> None:
    """PDF over ``MAX_PDF_BYTES`` errors with a size hint."""
    f = tmp_path / "big.pdf"
    f.write_bytes(b"%PDF-1.4\n")
    # Sparse truncate is O(1); no physical bytes written for the gap.
    os.truncate(f, MAX_PDF_BYTES + 1)
    with with_fake_agent() as agent:
        agent.tool_state.bash_cwd = str(tmp_path)
        result = await read.run({"file_path": str(f)})
    assert result.is_error
    assert "too large" in result.content


def _build_pdf(path: Path, n_pages: int) -> None:
    """Write a renderable ``n_pages``-page PDF via pypdfium2."""
    doc = pdfium.PdfDocument.new()
    try:
        for _ in range(n_pages):
            doc.new_page(72, 72)
        doc.save(str(path))
    finally:
        doc.close()


@pytest.mark.asyncio
async def test_read_pdf_byte_budget_is_provider_specific(tmp_path: Path) -> None:
    """The rendered-byte bound follows the ACTIVE model's request ceiling.

    A model with a small ``max_request_bytes`` must truncate a PDF read more
    aggressively than a large-ceiling model -- the bound is not a single
    hardcoded constant. Same PDF, two ceilings, different page counts.
    """
    f = tmp_path / "doc.pdf"
    _build_pdf(f, 6)
    one, _ = extract_pdf_pages(f, first=1, last=1)
    per_page = len(one[0])

    # Tiny ceiling: only a couple pages fit (after headroom).
    small = FakeAgent()
    small.max_request_bytes = per_page * 3
    with with_fake_agent(agent=small) as agent:
        agent.tool_state.bash_cwd = str(tmp_path)
        small_result = await read.run({"file_path": str(f), "pages": "1-6"})

    # Large ceiling: all 6 pages fit.
    large = FakeAgent()
    large.max_request_bytes = per_page * 100
    with with_fake_agent(agent=large) as agent:
        agent.tool_state.bash_cwd = str(tmp_path)
        large_result = await read.run({"file_path": str(f), "pages": "1-6"})

    assert len(large_result.attachments) == 6, "large ceiling reads all pages"
    assert len(small_result.attachments) < len(large_result.attachments), (
        "small-ceiling model must truncate more than large-ceiling model"
    )
    assert "pages=" in small_result.content  # continuation hint on the truncated read


@pytest.mark.asyncio
async def test_read_pdf_partial_on_byte_budget_surfaces_continuation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A read that busts the rendered-byte budget returns the pages that fit
    plus a VISIBLE continuation hint naming the remaining range.

    The model must not mistake a partial read for a complete one (silent
    truncation = data loss), so the result text states which pages were
    rendered and how to read the rest.
    """
    f = tmp_path / "dense.pdf"
    _build_pdf(f, 4)
    del monkeypatch  # budget now derives from the active model ceiling
    # Active-model ceiling that admits some but not all pages. The rendered
    # budget is ``(ceiling // 2) * 3 // 4`` raw bytes; size the ceiling so a
    # couple of pages fit but not all four.
    one, _ = extract_pdf_pages(f, first=1, last=1)
    per_page = len(one[0])
    small = FakeAgent()
    small.max_request_bytes = per_page * 7
    with with_fake_agent(agent=small) as agent:
        agent.tool_state.bash_cwd = str(tmp_path)
        result = await read.run({"file_path": str(f), "pages": "1-4"})
    assert not result.is_error
    assert result.attachments, "partial read must still return the pages that fit"
    assert len(result.attachments) < 4
    # Continuation hint names the next page to resume from.
    assert "pages=" in result.content
    assert "truncate" in result.content.lower() or "remaining" in result.content.lower()


@pytest.mark.asyncio
async def test_read_pdf_partial_truncation_does_not_depend_on_recount(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Truncation is surfaced from the renderer's total, not a re-open.

    Previously ``_read_pdf`` called ``get_pdf_page_count`` a SECOND time to
    compute the resume range; a transient ``None`` there silently marked a
    partial read as complete (data loss). The total now flows out of
    ``extract_pdf_pages``, so even if a re-count would fail, truncation is
    still reported. Pin it by making any post-render ``get_pdf_page_count``
    return ``None``.
    """
    f = tmp_path / "dense.pdf"
    _build_pdf(f, 4)
    one, _ = extract_pdf_pages(f, first=1, last=1)
    per_page = len(one[0])

    # If _read_pdf re-counts pages, this would force a false "complete".
    # Use "1-" (last=None) -- the case that previously triggered the recount.
    def _no_count(_p: Path) -> int | None:
        return None

    monkeypatch.setattr("sagent.tools.read.get_pdf_page_count", _no_count)
    small = FakeAgent()
    small.max_request_bytes = per_page * 7  # admits some, not all 4 pages
    with with_fake_agent(agent=small) as agent:
        agent.tool_state.bash_cwd = str(tmp_path)
        result = await read.run({"file_path": str(f), "pages": "1-"})
    assert not result.is_error
    assert len(result.attachments) < 4
    assert "pages=" in result.content
    assert "truncate" in result.content.lower() or "remaining" in result.content.lower()


@pytest.mark.asyncio
async def test_read_pdf_over_range_last_is_not_truncation(tmp_path: Path) -> None:
    """An over-range ``last`` that fits the byte budget is a COMPLETE read.

    ``extract_pdf_pages`` clamps ``hi`` to the real page count, so a request
    like ``pages="1-9999"`` on a short PDF renders every page. The truncation
    check must compare against the clamped page count, not the caller's raw
    ``last`` -- otherwise a complete read is mislabeled byte-truncated and the
    resume hint points past the end of the document.
    """
    f = tmp_path / "short.pdf"
    _build_pdf(f, 3)
    large = FakeAgent()
    large.max_request_bytes = 1 << 30  # ample: all pages fit
    with with_fake_agent(agent=large) as agent:
        agent.tool_state.bash_cwd = str(tmp_path)
        result = await read.run({"file_path": str(f), "pages": "1-9999"})
    assert not result.is_error
    assert len(result.attachments) == 3, "all real pages render"
    assert "truncate" not in result.content.lower()
    assert "9999" not in result.content


def test_summary_minimal() -> None:
    assert read.summary({"file_path": "/tmp/x.txt"}) == "Read x.txt"  # noqa: S108


def test_summary_offset_limit_window() -> None:
    # End line is inclusive: offset=5, limit=20 covers lines 5..24.
    # Pre-fix the summary printed ``5-25``, claiming a line that the
    # tool never reads.
    s = read.summary({"file_path": "x.txt", "offset": 5, "limit": 20})
    assert s == "Read x.txt:5-24"


def test_summary_last_lines() -> None:
    s = read.summary({"file_path": "x.txt", "last_lines": 10})
    assert s == "Read x.txt:last-10"


def test_summary_only_offset() -> None:
    s = read.summary({"file_path": "x.txt", "offset": 5})
    assert s == "Read x.txt:5+"


def test_summary_only_limit() -> None:
    s = read.summary({"file_path": "x.txt", "limit": 20})
    assert s == "Read x.txt:1-20"


def test_summary_no_path() -> None:
    assert read.summary({}) == "Read ?"


def test_prompt_empty() -> None:
    assert read.prompt() == ""


@pytest.mark.parametrize(
    ("command", "exe"),
    [
        ("cat foo.txt", "cat"),
        ("head -n 5 foo.txt", "head"),
        ("head -5 foo.txt", "head"),
        ("tail -n 5 foo.txt", "tail"),
        ("cat foo.txt | head", "cat"),
        # ``-n`` numbers lines, which Read does for every line it returns.
        ("cat -n foo.txt", "cat"),
        ("cat -n foo.txt | head", "cat"),
        # An unparsed count costs the worked example, not the nudge.
        ("head -n abc foo.txt", "head"),
    ],
)
def test_bash_match_nudges(command: str, exe: str) -> None:
    trees = parse_bash(command)
    assert trees is not None
    hint = read.bash_match(trees) or ""
    assert hint.startswith(f"{exe} via Bash is a bad UX."), command


@pytest.mark.parametrize(
    ("command", "call"),
    [
        ("cat foo.txt", "Try: Read file_path='foo.txt'"),
        ("head -n 5 foo.txt", "Try: Read file_path='foo.txt' limit=5"),
        ("tail -n 5 foo.txt", "Try: Read file_path='foo.txt' last_lines=5"),
        ("sed -n '10,20p' foo.txt", "Try: Read file_path='foo.txt' offset=10 limit=11"),
        ("sed -n '10,$p' foo.txt", "Try: Read file_path='foo.txt' offset=10"),
        # A shaping sink bounds the output, and Read expresses that bound
        # directly. Dropping it advertised the WHOLE file for a command
        # that asked for 20 lines -- on a large file the difference is
        # the entire reason the caller wrote ``| head``.
        ("cat foo.txt | head -20", "Try: Read file_path='foo.txt' limit=20"),
        ("cat foo.txt | head", "Try: Read file_path='foo.txt' limit=10"),
        ("cat foo.txt | tail -5", "Try: Read file_path='foo.txt' last_lines=5"),
        # ``less``/``cat`` page rather than truncate: no bound to render.
        ("cat foo.txt | less", "Try: Read file_path='foo.txt'"),
    ],
)
def test_bash_match_suggests_a_concrete_call(command: str, call: str) -> None:
    trees = parse_bash(command)
    assert trees is not None
    assert call in (read.bash_match(trees) or ""), command


@pytest.mark.parametrize(
    "command",
    [
        # No operand at all.
        "head",
        "head -n",
        # ``-c``/``--bytes`` window BYTES; Read windows lines.
        "head --bytes 5 foo.txt",
        "head -c 5 foo.txt",
        # ``-f`` follows a growing file; Read returns a snapshot.
        "tail -f /var/log/syslog",
        # ``-A`` renders nonprinting characters, which Read does not.
        "cat -A foo.txt",
        # A transforming sink does work Read cannot.
        "cat foo.txt | sort",
        "grep x foo.txt | head",
    ],
)
def test_bash_match_no_nudge(command: str) -> None:
    trees = parse_bash(command)
    assert trees is not None
    assert read.bash_match(trees) is None, command


def test_bash_match_unknown_command_no_nudge() -> None:
    trees = parse_bash("ls -la")
    assert trees is not None
    assert read.bash_match(trees) is None


def test_bash_match_env_prefix_skipped() -> None:
    trees = parse_bash("FOO=1 cat foo.txt")
    assert trees is not None
    assert read.bash_match(trees) is None


@pytest.mark.asyncio
async def test_read_notebook_non_dict_cell(tmp_path: Path) -> None:
    f = tmp_path / "nb.ipynb"
    # Cells list contains a string, not a dict; iterator must skip it.
    f.write_text(
        json.dumps({"cells": ["bogus", {"cell_type": "code", "source": "ok"}]})
    )
    with with_fake_agent() as agent:
        agent.tool_state.bash_cwd = str(tmp_path)
        result = await read.run({"file_path": str(f)})
    assert "ok" in result.content


@pytest.mark.asyncio
async def test_read_notebook_non_list_outputs(tmp_path: Path) -> None:
    f = tmp_path / "nb.ipynb"
    f.write_text(
        json.dumps(
            {"cells": [{"cell_type": "code", "source": "x", "outputs": "bogus"}]}
        )
    )
    with with_fake_agent() as agent:
        agent.tool_state.bash_cwd = str(tmp_path)
        result = await read.run({"file_path": str(f)})
    assert "Cell 1" in result.content


@pytest.mark.asyncio
async def test_read_notebook_non_dict_output(tmp_path: Path) -> None:
    f = tmp_path / "nb.ipynb"
    f.write_text(
        json.dumps(
            {
                "cells": [
                    {
                        "cell_type": "code",
                        "source": "x",
                        "outputs": ["bogus_string", {"text": "real"}],
                    }
                ]
            }
        )
    )
    with with_fake_agent() as agent:
        agent.tool_state.bash_cwd = str(tmp_path)
        result = await read.run({"file_path": str(f)})
    assert "[output] real" in result.content


@pytest.mark.asyncio
async def test_read_notebook_output_no_text(tmp_path: Path) -> None:
    f = tmp_path / "nb.ipynb"
    # Output without a 'text' key gets skipped.
    f.write_text(
        json.dumps(
            {
                "cells": [
                    {
                        "cell_type": "code",
                        "source": "x",
                        "outputs": [{"data": "not text"}],
                    }
                ]
            }
        )
    )
    with with_fake_agent() as agent:
        agent.tool_state.bash_cwd = str(tmp_path)
        result = await read.run({"file_path": str(f)})
    assert "[output]" not in result.content


@pytest.mark.asyncio
async def test_notebook_outputs_beyond_stream_text_are_kept(tmp_path: Path) -> None:
    """A cell's VALUE and its traceback are its most useful outputs.

    Only ``output['text']`` was read -- the ``stream`` shape. An
    ``execute_result`` carries ``data['text/plain']`` and an ``error``
    carries ``traceback``, so reading a notebook to see what a cell
    returned or how it failed showed neither.
    """
    nb = {
        "cells": [
            {
                "cell_type": "code",
                "source": "1 / 0",
                "outputs": [
                    {"output_type": "execute_result", "data": {"text/plain": "42"}},
                    {
                        "output_type": "error",
                        "ename": "ZeroDivisionError",
                        "traceback": ["Traceback", "ZeroDivisionError: division"],
                    },
                ],
            }
        ]
    }
    nb_path = tmp_path / "n.ipynb"
    nb_path.write_text(json.dumps(nb), encoding="utf-8")
    with with_fake_agent():
        result = await read.run({"file_path": str(nb_path)})
    assert "42" in result.content, result.content
    assert "ZeroDivisionError" in result.content, result.content


@pytest.mark.asyncio
async def test_an_unreadable_file_returns_an_error_not_an_exception(
    tmp_path: Path,
) -> None:
    """``Tool.run`` must return a ToolResult, never raise.

    A mode-000 file raised PermissionError straight out of the envelope,
    which the agent loop sees as a crash rather than a tool error.
    """
    locked = tmp_path / "locked.txt"
    locked.write_text("x", encoding="utf-8")
    locked.chmod(0o000)
    try:
        with with_fake_agent():
            result = await read.run({"file_path": str(locked)})
    finally:
        locked.chmod(0o600)
    assert result.is_error, result.content
    assert "locked.txt" in result.content, result.content


@pytest.mark.parametrize("ceiling", [20_000, 60_000, 150_000, 600_000])
def test_the_default_line_cap_stays_under_the_result_ceiling(ceiling: int) -> None:
    """The derived cap must fit ONE reply at every budget, not a floor.

    ``ContextBudget.from_model`` bottoms out at ``persist_threshold=20_000``,
    where a 2000-line floor is ~160_000 chars -- 8x the ceiling the cap
    exists to respect, and Read is exempt from disk offload, so nothing
    else bounds it.
    """
    with with_fake_agent() as agent:
        agent.max_result_chars = ceiling
        lines = _default_line_limit()
    assert lines * _ASSUMED_CHARS_PER_LINE <= ceiling, (
        f"{lines} lines x {_ASSUMED_CHARS_PER_LINE} chars exceeds {ceiling}"
    )
    assert lines >= 1, "a positive ceiling must still allow lines"


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
