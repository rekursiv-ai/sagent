"""Tests for ``sandboxed_tools.py``: path-restricted Write and Edit.

These exercise the sandbox boundary directly, without going through a
full agent. The goal is to prove: (a) in-sandbox writes/edits delegate
to the underlying tool, (b) out-of-sandbox writes/edits reject with a
clear error before opening the file, (c) ``..`` path traversal cannot
escape, (d) symlinks pointing outside the sandbox cannot escape.
"""

from __future__ import annotations

from pathlib import Path

import asyncio
import os

import pytest
import sandboxed_tools

from sagent.testing import with_fake_agent


def _run(coro):
    """Sync wrapper used by tests to drive an async tool call."""
    return asyncio.run(coro)


def test_sandboxed_write_rejects_absolute_path_outside_sandbox(tmp_path: Path) -> None:
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    outside = tmp_path / "outside.txt"
    tool = sandboxed_tools.SandboxedWrite(sandbox_root=sandbox)
    with with_fake_agent():
        result = _run(tool.run({"file_path": str(outside), "content": "hi"}))
    assert result.is_error
    assert "outside the statistician sandbox" in result.content
    assert not outside.exists()


def test_sandboxed_write_accepts_path_inside_sandbox(tmp_path: Path) -> None:
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    target = sandbox / "experiment.py"
    tool = sandboxed_tools.SandboxedWrite(sandbox_root=sandbox)
    with with_fake_agent():
        result = _run(tool.run({"file_path": str(target), "content": "print('ok')\n"}))
    assert not result.is_error, f"unexpected error: {result.content}"
    assert target.read_text() == "print('ok')\n"


def test_sandboxed_write_blocks_dot_dot_traversal(tmp_path: Path) -> None:
    """``../outside.txt`` resolves above the sandbox and must reject."""
    sandbox = tmp_path / "sandbox"
    nested = sandbox / "nested"
    nested.mkdir(parents=True)
    escape_target = tmp_path / "pwned.txt"
    tool = sandboxed_tools.SandboxedWrite(sandbox_root=sandbox)
    with with_fake_agent():
        result = _run(
            tool.run(
                {"file_path": str(nested / ".." / ".." / "pwned.txt"), "content": "x"}
            )
        )
    assert result.is_error
    assert not escape_target.exists()


def test_sandboxed_write_blocks_symlink_escape(tmp_path: Path) -> None:
    """A symlink inside the sandbox pointing outside still rejects."""
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    outside_target = tmp_path / "outside.txt"
    outside_target.write_text("original\n")
    link_inside = sandbox / "trojan.txt"
    os.symlink(outside_target, link_inside)

    tool = sandboxed_tools.SandboxedWrite(sandbox_root=sandbox)
    with with_fake_agent():
        result = _run(tool.run({"file_path": str(link_inside), "content": "PWNED\n"}))
    assert result.is_error
    # outside_target must be untouched:
    assert outside_target.read_text() == "original\n"


def test_sandboxed_write_rejects_missing_file_path() -> None:
    tool = sandboxed_tools.SandboxedWrite(sandbox_root=Path("/tmp"))
    with with_fake_agent():
        result = _run(tool.run({"content": "hi"}))
    assert result.is_error
    assert "'file_path' is required" in result.content


def test_sandboxed_edit_accepts_in_sandbox_target(tmp_path: Path) -> None:
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    target = sandbox / "f.py"
    target.write_text("x = 1\ny = 2\n")
    tool = sandboxed_tools.SandboxedEdit(sandbox_root=sandbox)
    # Ensure the read-cache lookup the Edit tool performs starts clean.
    # The FakeAgent fresh per-test gets its own ToolState.
    with with_fake_agent() as agent:
        # mark_read so the Edit tool's read-before-write check passes
        from sagent.tools.core import mark_read

        mark_read(str(target), content=target.read_text())
        result = _run(
            tool.run(
                {
                    "file_path": str(target),
                    "old_string": "x = 1",
                    "new_string": "x = 42",
                }
            )
        )
        del agent  # silence unused
    assert not result.is_error, f"unexpected error: {result.content}"
    assert target.read_text() == "x = 42\ny = 2\n"


def test_sandboxed_edit_rejects_out_of_sandbox(tmp_path: Path) -> None:
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    outside = tmp_path / "production.py"
    outside.write_text("DANGER = True\n")
    tool = sandboxed_tools.SandboxedEdit(sandbox_root=sandbox)
    with with_fake_agent():
        result = _run(
            tool.run(
                {
                    "file_path": str(outside),
                    "old_string": "DANGER = True",
                    "new_string": "DANGER = False",
                }
            )
        )
    assert result.is_error
    # File contents unchanged:
    assert outside.read_text() == "DANGER = True\n"


def test_sandbox_root_must_be_absolute() -> None:
    with pytest.raises(ValueError, match="must be absolute"):
        sandboxed_tools.SandboxedWrite(sandbox_root="relative/path")
    with pytest.raises(ValueError, match="must be absolute"):
        sandboxed_tools.SandboxedEdit(sandbox_root="relative/path")
