"""Tests for the Write tool and the atomic ``write_file`` it shares."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import os
import stat

import pytest

from sagent.session.types import FileWriteResult, UncategorizedToolResult
from sagent.tools.read import Read
from sagent.tools.types import ToolContext
from sagent.tools.write import Write, write_file


if TYPE_CHECKING:
    from treekle import PlainTree


def test_summary_names_the_file() -> None:
    assert Write().summary({"file_path": "/a/b/notes.md"}) == "Write /a/b/notes.md"
    assert Write().summary({}) == "Write "


def test_writes_to_one_file_share_a_key_however_it_is_named(tmp_path: Path) -> None:
    (tmp_path / "real.md").write_text("x")
    (tmp_path / "link.md").symlink_to(tmp_path / "real.md")
    context = ToolContext(cwd=tmp_path)
    key = Write().serialize_key(context, {"file_path": "link.md"})
    assert key == str(tmp_path / "real.md")
    assert Write().serialize_key(context, {}) == str(tmp_path)


def test_the_schema_requires_the_path_and_the_content() -> None:
    assert Write.schema["required"] == ["file_path", "content"]


@pytest.mark.asyncio
@pytest.mark.parametrize("arguments", [{"file_path": "a"}, {"content": "x"}])
async def test_a_call_missing_an_argument_is_refused(
    tmp_path: Path,
    arguments: dict[str, PlainTree],
) -> None:
    with pytest.raises(ValueError, match="Missing argument"):
        await Write().call(ToolContext(cwd=tmp_path), arguments)


@pytest.mark.asyncio
async def test_a_new_file_is_made_with_its_directories(tmp_path: Path) -> None:
    context = ToolContext(cwd=tmp_path)
    result = await _write(context, "deep/er/notes.md", "hello\n")
    target = tmp_path / "deep" / "er" / "notes.md"
    assert result == FileWriteResult(call_id="", path=str(target), content="hello\n")
    assert result.text() == f"Wrote {target}."
    assert target.read_text() == "hello\n"
    assert target in context.files
    assert not context.files.changed(target)


@pytest.mark.asyncio
async def test_a_file_just_written_may_be_written_again(tmp_path: Path) -> None:
    context = ToolContext(cwd=tmp_path)
    await _write(context, "a.md", "one")
    again = await _write(context, "a.md", "two")
    assert isinstance(again, FileWriteResult)
    assert (tmp_path / "a.md").read_text() == "two"


@pytest.mark.asyncio
async def test_an_existing_file_must_be_read_before_it_is_replaced(
    tmp_path: Path,
) -> None:
    (tmp_path / "a.md").write_text("theirs")
    context = ToolContext(cwd=tmp_path)
    refused = await _write(context, "a.md", "mine")
    assert refused.is_error is True
    assert (
        refused.text()
        == f"{tmp_path / 'a.md'} exists and has not been read. Read it first."
    )
    assert (tmp_path / "a.md").read_text() == "theirs"
    context.files.saw(tmp_path / "a.md")
    replaced = await _write(context, "a.md", "mine")
    assert replaced.is_error is None
    assert (tmp_path / "a.md").read_text() == "mine"


@pytest.mark.asyncio
async def test_a_file_read_by_one_spelling_may_be_written_by_another(
    tmp_path: Path,
) -> None:
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.py").write_text("old")
    context = ToolContext(cwd=tmp_path)
    await Read().call(context, {"file_path": "sub/../a.py"})
    result = await _write(context, "a.py", "new")
    assert result.is_error is None, result.text()
    assert (tmp_path / "a.py").read_text() == "new"


@pytest.mark.asyncio
async def test_a_file_changed_since_it_was_read_is_not_replaced(tmp_path: Path) -> None:
    f = tmp_path / "a.md"
    f.write_text("seen")
    context = ToolContext(cwd=tmp_path)
    context.files.saw(f)
    f.write_text("someone else's")
    later = f.stat().st_mtime + 100
    os.utime(f, (later, later))
    result = await _write(context, "a.md", "mine")
    assert result.is_error is True
    assert result.text() == (
        f"{f} changed since it was read, by the user, a linter, or another agent."
        " Read it again before writing it."
    )
    assert f.read_text() == "someone else's"


@pytest.mark.asyncio
async def test_a_directory_is_not_written(tmp_path: Path) -> None:
    (tmp_path / "d").mkdir()
    result = await _write(ToolContext(cwd=tmp_path), "d", "x")
    assert result.is_error is True
    assert result.text() == f"{tmp_path / 'd'} is a directory, not a file."


@pytest.mark.asyncio
async def test_a_replaced_file_keeps_its_mode(tmp_path: Path) -> None:
    f = tmp_path / "secret.env"
    f.write_text("A=1")
    f.chmod(0o600)
    context = ToolContext(cwd=tmp_path)
    context.files.saw(f)
    await _write(context, "secret.env", "A=2")
    assert stat.S_IMODE(f.stat().st_mode) == 0o600
    assert f.read_text() == "A=2"


@pytest.mark.parametrize(("umask", "mode"), [(0o027, 0o640), (0, 0o666)])
def test_a_new_file_is_read_write_less_the_umask(
    tmp_path: Path,
    umask: int,
    mode: int,
) -> None:
    old = os.umask(umask)
    try:
        write_file(tmp_path / "new", b"x")
    finally:
        os.umask(old)
    assert stat.S_IMODE((tmp_path / "new").stat().st_mode) == mode


def test_a_write_that_cannot_start_raises_what_stopped_it(tmp_path: Path) -> None:
    locked = tmp_path / "locked"
    locked.mkdir(mode=0o500)
    try:
        with pytest.raises(PermissionError):
            write_file(locked / "f", b"x")
    finally:
        locked.chmod(0o700)
    assert list(locked.iterdir()) == []


def test_a_link_is_written_through_and_stays_a_link(tmp_path: Path) -> None:
    target = tmp_path / "target.md"
    target.write_text("old")
    target.chmod(0o600)
    link = tmp_path / "link.md"
    link.symlink_to(target)
    write_file(link, b"new")
    assert link.is_symlink()
    assert target.read_bytes() == b"new"
    assert stat.S_IMODE(target.stat().st_mode) == 0o600


def test_a_dangling_link_is_written_through_to_its_target(tmp_path: Path) -> None:
    """``echo x > link`` makes the file the link names; it keeps the link."""
    link = tmp_path / "link.md"
    link.symlink_to(tmp_path / "sub" / "target.md")
    write_file(link, b"new")
    assert link.is_symlink()
    assert (tmp_path / "sub" / "target.md").read_bytes() == b"new"


def test_every_byte_lands_when_the_kernel_takes_a_few_at_a_time(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_write = os.write

    def short_write(fd: int, data: bytes | memoryview) -> int:
        return real_write(fd, bytes(data[:3]))

    monkeypatch.setattr(os, "write", short_write)
    write_file(tmp_path / "f", b"0123456789")
    assert (tmp_path / "f").read_bytes() == b"0123456789"


def test_a_failed_write_leaves_the_old_file_and_no_scrap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    f = tmp_path / "f"
    f.write_bytes(b"old")

    def full_disk(fd: int, data: bytes | memoryview) -> int:
        del fd, data
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(os, "write", full_disk)
    with pytest.raises(OSError, match="No space left"):
        write_file(f, b"new")
    assert f.read_bytes() == b"old"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["f"]


def test_a_write_is_synced_before_it_replaces_the_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A crash after the rename must not leave an empty file in place."""
    order: list[str] = []
    real_fsync, real_replace = os.fsync, Path.replace

    def fsync(fd: int) -> None:
        order.append("fsync")
        real_fsync(fd)

    def replace(self: Path, target: Path) -> Path:
        order.append("replace")
        return real_replace(self, target)

    monkeypatch.setattr(os, "fsync", fsync)
    monkeypatch.setattr(Path, "replace", replace)
    write_file(tmp_path / "f", b"x")
    assert order == ["fsync", "replace"]


async def _write(
    context: ToolContext,
    file_path: str,
    content: str,
) -> FileWriteResult | UncategorizedToolResult:
    """Write ``content`` to ``file_path``; the runtime names the call."""
    result = await Write().call(context, {"file_path": file_path, "content": content})
    assert result.call_id == ""
    return result


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
