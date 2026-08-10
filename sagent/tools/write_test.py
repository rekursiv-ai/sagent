"""Tests for ``tools.write``: file creation / overwrite tool."""

from __future__ import annotations

from pathlib import Path

import os
import stat
import time

import pytest

from sagent.testing import with_fake_agent
from sagent.tools.write import Write


write = Write()


@pytest.mark.asyncio
async def test_write_creates_new_file(tmp_path: Path) -> None:
    f = tmp_path / "new.txt"
    with with_fake_agent() as agent:
        agent.tool_state.bash_cwd = str(tmp_path)
        result = await write.run({"file_path": str(f), "content": "hello"})
    assert not result.is_error
    assert f.read_text() == "hello"
    assert "Wrote 5 bytes" in result.content


@pytest.mark.asyncio
async def test_write_relative_path(tmp_path: Path) -> None:
    with with_fake_agent() as agent:
        agent.tool_state.bash_cwd = str(tmp_path)
        result = await write.run({"file_path": "rel.txt", "content": "hi"})
    assert not result.is_error
    assert (tmp_path / "rel.txt").read_text() == "hi"


@pytest.mark.asyncio
async def test_write_overwrite_requires_prior_read(tmp_path: Path) -> None:
    f = tmp_path / "exists.txt"
    f.write_text("old")
    with with_fake_agent() as agent:
        agent.tool_state.bash_cwd = str(tmp_path)
        result = await write.run({"file_path": str(f), "content": "new"})
    assert result.is_error
    assert "not yet read" in result.content


@pytest.mark.asyncio
async def test_write_overwrite_after_read_succeeds(tmp_path: Path) -> None:
    f = tmp_path / "exists.txt"
    f.write_text("old\n")
    with with_fake_agent() as agent:
        agent.tool_state.bash_cwd = str(tmp_path)
        agent.tool_state.mark_read(str(f), content="old\n")
        result = await write.run({"file_path": str(f), "content": "new\n"})
    assert not result.is_error
    assert f.read_text() == "new\n"


@pytest.mark.asyncio
async def test_write_stale_file_errors(tmp_path: Path) -> None:
    f = tmp_path / "x.txt"
    f.write_text("v0\n")
    with with_fake_agent() as agent:
        agent.tool_state.bash_cwd = str(tmp_path)
        agent.tool_state.mark_read(str(f), content="v0\n", mtime=1.0)
        f.write_text("changed\n")
        new_mtime = time.time() + 100
        os.utime(f, (new_mtime, new_mtime))
        result = await write.run({"file_path": str(f), "content": "v1\n"})
    assert result.is_error
    assert "modified since read" in result.content


@pytest.mark.asyncio
async def test_write_then_write_to_new_file_succeeds(tmp_path: Path) -> None:
    f = tmp_path / "new.txt"
    with with_fake_agent() as agent:
        agent.tool_state.bash_cwd = str(tmp_path)
        r1 = await write.run({"file_path": str(f), "content": "v1"})
        assert not r1.is_error, r1.content
        r2 = await write.run({"file_path": str(f), "content": "v2"})
        assert not r2.is_error, r2.content
    assert f.read_text() == "v2"


@pytest.mark.asyncio
async def test_write_to_directory_errors(tmp_path: Path) -> None:
    sub = tmp_path / "dir"
    sub.mkdir()
    with with_fake_agent() as agent:
        agent.tool_state.bash_cwd = str(tmp_path)
        result = await write.run({"file_path": str(sub), "content": "x"})
    assert result.is_error
    assert "directory" in result.content


@pytest.mark.asyncio
async def test_write_preserves_file_mode(tmp_path: Path) -> None:
    f = tmp_path / "secret.txt"
    f.write_text("v0\n")
    f.chmod(0o600)
    with with_fake_agent() as agent:
        agent.tool_state.bash_cwd = str(tmp_path)
        agent.tool_state.mark_read(str(f), content="v0\n")
        result = await write.run({"file_path": str(f), "content": "v1\n"})
    assert not result.is_error
    assert stat.S_IMODE(f.stat().st_mode) == 0o600


def test_summary_path_basename() -> None:
    assert write.summary({"file_path": "/tmp/foo.txt"}) == "Write foo.txt"  # noqa: S108


def test_summary_no_path() -> None:
    assert write.summary({}) == "Write ?"


def test_prompt_empty() -> None:
    assert write.prompt() == ""


def test_schema_required() -> None:
    schema = write.directive_schema
    assert schema["required"] == ("file_path", "content")


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
