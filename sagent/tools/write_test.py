"""Tests for Write tool."""

from __future__ import annotations

from pathlib import Path

import asyncio

import pytest

from sagent.custom_types import (
    JsonMessage,
    Message,
    MultipartMessage,
    TextMessage,
)
from sagent.lib.json import JSON, json_freeze
from sagent.tools.core import ToolState, tool_state_context
from sagent.tools.read import Read
from sagent.tools.write import Write


read = Read()
write = Write()


def _text(r: Message) -> str:
    if isinstance(r, TextMessage):
        return r.content
    if isinstance(r, MultipartMessage):
        for p in r.content:
            if isinstance(p, TextMessage) and p.descriptor == "text/plain":
                return p.content
    return ""


def _msg(directive: JSON) -> Message:
    return MultipartMessage(
        (JsonMessage(directive, "application/x-tool-x"),),
        "multipart/x-tool-call",
    )


class TestWrite:
    @pytest.mark.anyio
    async def test_write_file(self, tmp_path: Path) -> None:
        f = tmp_path / "out.txt"
        response = await write.run(
            _msg(json_freeze({"file_path": str(f), "content": "hello"}))
        )
        assert f.read_text() == "hello"
        assert "5 bytes" in _text(response)

    @pytest.mark.anyio
    async def test_summary_result_reports_byte_count(self, tmp_path: Path) -> None:
        """Write.summary_result extracts the byte count from the OK message."""
        summary_write = Write()
        summary_write.emit_tool_summary = True
        f = tmp_path / "out.txt"
        response = await summary_write.run(
            _msg(json_freeze({"file_path": str(f), "content": "hello"}))
        )
        assert summary_write.summary_result(response) == "wrote 5 bytes"

    def test_summary_result_off_by_default(self) -> None:
        """Write.summary_result returns None when the gate is off."""
        assert (
            write.summary_result(TextMessage("Wrote 5 bytes to /x", "text/plain"))
            is None
        )

    @pytest.mark.anyio
    async def test_write_creates_dirs(self, tmp_path: Path) -> None:
        f = tmp_path / "sub" / "dir" / "out.txt"
        await write.run(_msg(json_freeze({"file_path": str(f), "content": "nested"})))
        assert f.read_text() == "nested"

    @pytest.mark.anyio
    async def test_relative_path_uses_bash_cwd(self, tmp_path: Path) -> None:
        state = ToolState()
        state.bash_cwd = str(tmp_path)
        with tool_state_context(state):
            await write.run(
                _msg(json_freeze({"file_path": "relative.txt", "content": "ok"}))
            )
        assert (tmp_path / "relative.txt").read_text() == "ok"

    @pytest.mark.anyio
    async def test_write_guard_rejects(self, tmp_path: Path) -> None:
        f = tmp_path / "existing.txt"
        f.write_text("original")
        r = await write.run(_msg(json_freeze({"file_path": str(f), "content": "new"})))
        assert r.descriptor == "text/x-error"
        assert "not yet read" in str(r.content)
        assert f.read_text() == "original"

    @pytest.mark.anyio
    async def test_write_rejects_directory(self, tmp_path: Path) -> None:
        r = await write.run(
            _msg(json_freeze({"file_path": str(tmp_path), "content": "x"}))
        )
        assert r.descriptor == "text/x-error"
        assert "is a directory" in str(r.content)

    @pytest.mark.anyio
    async def test_write_rejects_stale_file(self, tmp_path: Path) -> None:
        f = tmp_path / "stale.txt"
        f.write_text("v1")
        await read.run(_msg(json_freeze({"file_path": str(f)})))
        # Mutate the file out-of-band.
        await asyncio.sleep(0.01)
        f.write_text("v2")
        r = await write.run(_msg(json_freeze({"file_path": str(f), "content": "v3"})))
        assert r.descriptor == "text/x-error"
        assert "modified since read" in str(r.content)


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
