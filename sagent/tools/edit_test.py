"""Tests for the Edit tool: exact replacements, and ``sed -i`` taken on."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import os
import stat

import pytest

from sagent.session.types import FileEditResult, Splice, UncategorizedToolResult
from sagent.tools.command_analyzer import invocations
from sagent.tools.edit import Edit
from sagent.tools.types import ToolContext, parse


if TYPE_CHECKING:
    from treekle import PlainTree


def test_summary_names_the_file() -> None:
    assert Edit().summary({"file_path": "/src/a.py"}) == "Edit /src/a.py"
    assert Edit().summary({}) == "Edit "


def test_edits_of_one_file_share_a_key_however_it_is_named(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x")
    (tmp_path / "b.py").symlink_to(tmp_path / "a.py")
    context = ToolContext(cwd=tmp_path)
    assert Edit().serialize_key(context, {"file_path": "b.py"}) == str(
        tmp_path / "a.py",
    )
    assert Edit().serialize_key(context, {}) == str(tmp_path)


@pytest.mark.asyncio
async def test_a_unique_string_is_replaced_and_the_splice_says_where(
    tmp_path: Path,
) -> None:
    (tmp_path / "a.py").write_text("one\nx = 1\nthree\n")
    context = ToolContext(cwd=tmp_path)
    result = await _edit(context, "a.py", "x = 1", "x = 2")
    assert result == FileEditResult(
        call_id="",
        path=str(tmp_path / "a.py"),
        edits=(Splice(before="x = 1", after="x = 2", start=2, count=1),),
    )
    assert result.text() == f"Edited {tmp_path / 'a.py'}."
    assert (tmp_path / "a.py").read_text() == "one\nx = 2\nthree\n"


@pytest.mark.asyncio
async def test_replace_all_replaces_each_and_records_each_line(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("foo\nbar\nfoo foo\nend\nfoo\n")
    result = await _edit(
        ToolContext(cwd=tmp_path),
        "a.py",
        "foo",
        "baz",
        replace_all=True,
    )
    assert isinstance(result, FileEditResult)
    assert [(s.start, s.count) for s in result.edits] == [
        (1, 1),
        (3, 1),
        (3, 1),
        (5, 1),
    ]
    assert (tmp_path / "a.py").read_text() == "baz\nbar\nbaz baz\nend\nbaz\n"


@pytest.mark.asyncio
async def test_a_leading_blank_line_is_counted(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("\nx\n")
    result = await _edit(ToolContext(cwd=tmp_path), "a.py", "x", "y")
    assert isinstance(result, FileEditResult)
    assert [s.start for s in result.edits] == [2]


@pytest.mark.asyncio
async def test_a_string_across_lines_counts_each_line_it_spans(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("a\nb\nc\nd\nb\nc\n")
    result = await _edit(
        ToolContext(cwd=tmp_path),
        "a.py",
        "b\nc\n",
        "",
        replace_all=True,
    )
    assert isinstance(result, FileEditResult)
    assert [(s.start, s.count) for s in result.edits] == [(2, 3), (5, 3)]
    assert (tmp_path / "a.py").read_text() == "a\nd\n"


@pytest.mark.asyncio
async def test_a_file_need_not_be_read_first(tmp_path: Path) -> None:
    """The exact ``old_string`` is the proof the model knows what it replaces."""
    (tmp_path / "a.py").write_text("x = 1\n")
    context = ToolContext(cwd=tmp_path)
    result = await _edit(context, "a.py", "1", "2")
    assert result.is_error is None
    assert tmp_path / "a.py" in context.files


@pytest.mark.asyncio
async def test_edits_in_a_row_each_see_the_last(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("a = 1\n")
    context = ToolContext(cwd=tmp_path)
    for old, new in (("1", "2"), ("2", "3"), ("3", "4")):
        result = await _edit(context, "a.py", old, new)
        assert result.is_error is None, result.text()
    assert (tmp_path / "a.py").read_text() == "a = 4\n"


@pytest.mark.asyncio
async def test_a_file_changed_since_it_was_read_is_not_edited(tmp_path: Path) -> None:
    f = tmp_path / "a.py"
    f.write_text("x = 1\n")
    context = ToolContext(cwd=tmp_path)
    context.files.saw(f)
    f.write_text("x = 1  # theirs\n")
    later = f.stat().st_mtime + 100
    os.utime(f, (later, later))
    result = await _edit(context, "a.py", "1", "2")
    assert result.is_error is True
    assert result.text() == (
        f"{f} changed since it was read, by the user, a linter, or another agent."
        " Read it again before editing it."
    )
    assert f.read_text() == "x = 1  # theirs\n"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("setup", "old", "replace_all", "message"),
    [
        ("text", "", False, "old_string is empty."),
        ("missing", "x", False, "{} is not a file."),
        ("directory", "x", False, "{} is not a file."),
        ("text", "absent", False, "old_string not found in file."),
        (
            "text",
            "o",
            False,
            (
                "old_string found 2 times. Set replace_all, or give more context to"
                " make it unique."
            ),
        ),
        ("text", "nowhere", True, "old_string not found in file."),
        ("binary", "x", False, "{} is not UTF-8. Use Bash with an explicit decoder."),
    ],
)
async def test_an_edit_that_cannot_be_made_changes_nothing(
    tmp_path: Path,
    setup: str,
    old: str,
    replace_all: bool,
    message: str,
) -> None:
    f = tmp_path / "f"
    if setup == "text":
        f.write_text("one two\n")
    elif setup == "directory":
        f.mkdir()
    elif setup == "binary":
        f.write_bytes(b"x\xff\xfe")
    before = f.read_bytes() if f.is_file() else None
    result = await _edit(
        ToolContext(cwd=tmp_path),
        "f",
        old,
        "new",
        replace_all=replace_all,
    )
    assert result.is_error is True
    assert result.text() == message.format(f)
    assert (f.read_bytes() if f.is_file() else None) == before


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("before", "old", "new", "after"),
    [
        (b"a\r\nb\r\n", "b", "c", b"a\r\nc\r\n"),
        (b"a\r\nb\r\nc\r\n", "a\nb\n", "x\ny\nz\n", b"x\r\ny\r\nz\r\nc\r\n"),
        (b"a\r\nb\r\n", "a\r\nb", "c", b"c\r\n"),
        (b"a\r\nb\nc\r\n", "b", "x", b"a\r\nx\nc\r\n"),
        (b"a\rb\r", "a", "x", b"x\rb\r"),
    ],
    ids=["crlf", "crlf-lines", "crlf-given", "mixed", "cr"],
)
async def test_an_edit_keeps_the_files_line_endings(
    tmp_path: Path,
    before: bytes,
    old: str,
    new: str,
    after: bytes,
) -> None:
    r"""Read shows ``\\n``; the file keeps what it had, and the rest is untouched."""
    (tmp_path / "f").write_bytes(before)
    result = await _edit(ToolContext(cwd=tmp_path), "f", old, new)
    assert result.is_error is None, result.text()
    assert (tmp_path / "f").read_bytes() == after


@pytest.mark.asyncio
async def test_an_edit_reads_and_writes_utf8_whatever_the_locale(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "f").write_bytes("café\nb\n".encode())
    _latin1_locale(monkeypatch)
    result = await _edit(ToolContext(cwd=tmp_path), "f", "b", "c")
    assert result.is_error is None, result.text()
    assert (tmp_path / "f").read_bytes() == "café\nc\n".encode()


@pytest.mark.asyncio
async def test_an_edited_file_keeps_its_mode(tmp_path: Path) -> None:
    f = tmp_path / "run.sh"
    f.write_text("echo 1\n")
    f.chmod(0o750)
    await _edit(ToolContext(cwd=tmp_path), "run.sh", "1", "2")
    assert stat.S_IMODE(f.stat().st_mode) == 0o750


@pytest.mark.asyncio
async def test_arguments_of_the_wrong_type_are_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="'replace_all' must be boolean"):
        await Edit().call(
            ToolContext(cwd=tmp_path),
            {"file_path": "f", "old_string": "a", "new_string": "b", "replace_all": 1},
        )


@pytest.mark.parametrize(
    ("command", "arguments"),
    [
        (
            "sed -i 's/foo/bar/' file.txt",
            {
                "file_path": "file.txt",
                "old_string": "foo",
                "new_string": "bar",
                "replace_all": False,
            },
        ),
        (
            "sed -i 's/foo/bar/g' file.txt",
            {
                "file_path": "file.txt",
                "old_string": "foo",
                "new_string": "bar",
                "replace_all": True,
            },
        ),
        (
            "sed --in-place 's/foo bar/baz/' file.txt",
            {
                "file_path": "file.txt",
                "old_string": "foo bar",
                "new_string": "baz",
                "replace_all": False,
            },
        ),
        (
            "sed -i 's/drop//' file.txt",
            {
                "file_path": "file.txt",
                "old_string": "drop",
                "new_string": "",
                "replace_all": False,
            },
        ),
        (
            "cd /srv && sed -i 's/a/b/' f.txt",
            {
                "file_path": "/srv/f.txt",
                "old_string": "a",
                "new_string": "b",
                "replace_all": False,
            },
        ),
        (
            "cd /srv && cd sub && sed -i 's/a/b/' f.txt",
            {
                "file_path": "/srv/sub/f.txt",
                "old_string": "a",
                "new_string": "b",
                "replace_all": False,
            },
        ),
        (
            "cd /srv && sed -i 's/a/b/' /etc/hosts",
            {
                "file_path": "/etc/hosts",
                "old_string": "a",
                "new_string": "b",
                "replace_all": False,
            },
        ),
    ],
)
def test_a_literal_sed_in_place_becomes_the_edit_that_makes_it(
    command: str,
    arguments: dict[str, PlainTree],
) -> None:
    assert _from_shell(command) == arguments
    parse(arguments, Edit.Arguments)


@pytest.mark.parametrize(
    "command",
    [
        "sed 's/foo/bar/' file.txt",
        "sed -i.bak 's/foo/bar/' file.txt",
        "sed -i 's/foo/bar/gi' file.txt",
        "sed -E -i 's/foo/bar/' file.txt",
        "sed -i -e 's/foo/bar/' file.txt",
        "sed -i 's/foo/bar/'",
        "sed -i 's/foo/bar/' a.txt b.txt",
        "sed -i '/foo/d' file.txt",
        "sed -i 's/f.o/bar/' file.txt",
        "sed -i 's/ab*/bar/' file.txt",
        "sed -i 's/[ab]/bar/' file.txt",
        "sed -i 's/^foo/bar/' file.txt",
        "sed -i 's/foo$/bar/' file.txt",
        "sed -i 's/a\\/b/c/' file.txt",
        "sed -i 's//bar/' file.txt",
        "sed -i 's/foo/&&/' file.txt",
        "sed -i 's/foo/\\1/' file.txt",
        # Bash fills these in before sed sees them; Edit would write them literally.
        'sed -i "s/foo/$HOME/" file.txt',
        'sed -i "s/foo/$(date)/" file.txt',
        'sed -i "s/foo/bar/" "$F"',
        "sed -i 's/a/b/' *.txt",
        "cd - && sed -i 's/a/b/' f.txt",
        "FOO=1 sed -i 's/a/b/' f",
        "sed -i 's/a/b/' f > out",
        "awk 'NR==1' file.txt",
        "echo hi | grep x",
    ],
)
def test_a_command_no_edit_matches_stays_a_shell_command(command: str) -> None:
    assert _from_shell(command) is None


async def _edit(
    context: ToolContext,
    file_path: str,
    old: str,
    new: str,
    *,
    replace_all: bool = False,
) -> FileEditResult | UncategorizedToolResult:
    """Replace ``old`` with ``new`` in ``file_path``; the runtime names the call."""
    result = await Edit().call(
        context,
        {
            "file_path": file_path,
            "old_string": old,
            "new_string": new,
            "replace_all": replace_all,
        },
    )
    assert result.call_id == ""
    return result


def _latin1_locale(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make text reads that name no encoding decode as a Latin-1 locale would."""
    real = Path.read_text

    def read_text(
        self: Path,
        encoding: str | None = None,
        errors: str | None = None,
    ) -> str:
        return real(
            self,
            encoding=encoding or "latin-1",
            errors=errors,
        )

    monkeypatch.setattr(Path, "read_text", read_text)


def _from_shell(command: str) -> PlainTree:
    """Return what Edit makes of the first ``sed`` in ``command``, else its last command."""
    found = invocations(command)
    assert found is not None
    result = Edit().from_shell(next((i for i in found if i.exe == "sed"), found[-1]))
    return result if result is None or isinstance(result, str) else dict(result)


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
