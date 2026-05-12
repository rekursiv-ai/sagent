"""Tests for ``tools.read``: text/image/PDF/notebook reading."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import json
import os

from PIL import Image

import pytest

from sagent.agent.runtime import BytesMessage, ToolResult
from sagent.testing import with_fake_agent
from sagent.tools.lib.bash import parse_bash
from sagent.tools.lib.pdf import MAX_PDF_BYTES
from sagent.tools.read import Read


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
async def test_read_svg_mime(tmp_path: Path) -> None:
    f = tmp_path / "image.svg"
    f.write_text("<svg/>")
    with with_fake_agent() as agent:
        agent.tool_state.bash_cwd = str(tmp_path)
        result = await read.run({"file_path": str(f)})
    assert result.attachments[0].descriptor == "image/svg+xml"


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


def test_summary_minimal() -> None:
    assert read.summary({"file_path": "/tmp/x.txt"}) == "Read x.txt"  # noqa: S108


def test_summary_offset_limit_window() -> None:
    s = read.summary({"file_path": "x.txt", "offset": 5, "limit": 20})
    assert s == "Read x.txt:5-25"


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


def test_summary_result_off() -> None:
    assert read.summary_result(ToolResult(call_id="", content="x")) is None


def test_summary_result_lines_when_emit() -> None:
    r = Read()
    r.emit_tool_summary = True
    assert r.summary_result(ToolResult(call_id="", content="a\nb\nc\n")) == "3 lines"


def test_summary_result_binary_marker() -> None:
    r = Read()
    r.emit_tool_summary = True
    out = r.summary_result(
        ToolResult(
            call_id="",
            content="[image: x.png]",
            attachments=(BytesMessage(b"", "image/png"),),
        ),
    )
    assert out == "binary"


def test_summary_result_unchanged_marker() -> None:
    r = Read()
    r.emit_tool_summary = True
    assert (
        r.summary_result(ToolResult(call_id="", content="[File unchanged since…]"))
        == "unchanged"
    )


def test_summary_result_error_skipped() -> None:
    r = Read()
    r.emit_tool_summary = True
    assert r.summary_result(ToolResult(call_id="", content="!", is_error=True)) is None


def test_summary_result_empty_text() -> None:
    r = Read()
    r.emit_tool_summary = True
    assert r.summary_result(ToolResult(call_id="", content="")) is None


def test_prompt_empty() -> None:
    assert read.prompt() == ""


def test_bash_match_cat_file() -> None:
    trees = parse_bash("cat foo.txt")
    assert trees is not None
    assert read.bash_match(trees) == "cat via Bash is a bad UX. Use the Read tool."


def test_bash_match_cat_with_flag_no_nudge() -> None:
    trees = parse_bash("cat -n foo.txt")
    assert trees is not None
    assert read.bash_match(trees) is None


def test_bash_match_head_n_flag() -> None:
    trees = parse_bash("head -n 5 foo.txt")
    assert trees is not None
    assert read.bash_match(trees) == "head via Bash is a bad UX. Use the Read tool."


def test_bash_match_head_short_flag() -> None:
    trees = parse_bash("head -5 foo.txt")
    assert trees is not None
    assert read.bash_match(trees) == "head via Bash is a bad UX. Use the Read tool."


def test_bash_match_tail_n_flag() -> None:
    trees = parse_bash("tail -n 5 foo.txt")
    assert trees is not None
    assert read.bash_match(trees) == "tail via Bash is a bad UX. Use the Read tool."


def test_bash_match_head_no_arg_no_nudge() -> None:
    trees = parse_bash("head")
    assert trees is not None
    assert read.bash_match(trees) is None


def test_bash_match_head_long_flag_no_nudge() -> None:
    trees = parse_bash("head --bytes 5 foo.txt")
    assert trees is not None
    assert read.bash_match(trees) is None


def test_bash_match_head_bad_n_value() -> None:
    trees = parse_bash("head -n abc foo.txt")
    assert trees is not None
    assert read.bash_match(trees) is None


def test_bash_match_head_n_no_value() -> None:
    trees = parse_bash("head -n")
    assert trees is not None
    assert read.bash_match(trees) is None


def test_bash_match_head_c_no_nudge() -> None:
    trees = parse_bash("head -c 5 foo.txt")
    assert trees is not None
    assert read.bash_match(trees) is None


def test_bash_match_pipeline_cat_head() -> None:
    trees = parse_bash("cat foo.txt | head")
    assert trees is not None
    assert read.bash_match(trees) == "cat via Bash is a bad UX. Use the Read tool."


def test_bash_match_pipeline_other_no_nudge() -> None:
    trees = parse_bash("cat foo.txt | sort")
    assert trees is not None
    assert read.bash_match(trees) is None


def test_bash_match_pipeline_grep_first_no_nudge() -> None:
    trees = parse_bash("grep x foo.txt | head")
    assert trees is not None
    assert read.bash_match(trees) is None


def test_bash_match_pipeline_cat_flag_no_nudge() -> None:
    trees = parse_bash("cat -n foo.txt | head")
    assert trees is not None
    assert read.bash_match(trees) is None


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


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
