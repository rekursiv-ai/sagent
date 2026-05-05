"""Tests for Edit tool."""

from __future__ import annotations

from pathlib import Path

import asyncio
import re

import pytest

from sagent.custom_types import (
    JsonMessage,
    Message,
    MessageBase,
    MultipartMessage,
    TextMessage,
)
from sagent.lib.json import JSON, json_freeze
from sagent.tools.core import (
    ToolState,
    tool_state_context,
)
from sagent.tools.edit import Edit
from sagent.tools.lib.bash import parse_bash
from sagent.tools.read import Read


read = Read()
edit = Edit()


def _msg_read(file_path: str) -> Message:
    return MultipartMessage(
        (
            JsonMessage(
                json_freeze({"file_path": file_path}),
                "application/x-tool-read",
            ),
        ),
        "multipart/x-tool-call",
    )


def _msg_edit(directive: JSON) -> Message:
    return MultipartMessage(
        (JsonMessage(directive, "application/x-tool-edit"),),
        "multipart/x-tool-call",
    )


def _error_text(r: Message) -> str:
    if isinstance(r, TextMessage) and r.descriptor == "text/x-error":
        return r.content
    return ""


def _txt(msg: Message) -> str:
    if isinstance(msg, TextMessage):
        return msg.content
    if isinstance(msg, MultipartMessage):
        for p in msg.content:
            if isinstance(p, TextMessage):
                return p.content
    return ""


class TestEdit:
    @pytest.mark.anyio
    async def test_replace_unique(self, tmp_path: Path) -> None:
        f = tmp_path / "test.py"
        f.write_text("x = 1\ny = 2\n")
        await read.run(_msg_read(str(f)))
        response = await edit.run(
            _msg_edit(
                json_freeze(
                    {
                        "file_path": str(f),
                        "old_string": "x = 1",
                        "new_string": "x = 42",
                    }
                )
            )
        )
        assert f.read_text() == "x = 42\ny = 2\n"
        assert "1 occurrence" in _txt(response)

    @pytest.mark.anyio
    async def test_ambiguous_reject(self, tmp_path: Path) -> None:
        f = tmp_path / "test.py"
        f.write_text("x = 1\nx = 1\n")
        await read.run(_msg_read(str(f)))
        r = await edit.run(
            _msg_edit(
                json_freeze(
                    {
                        "file_path": str(f),
                        "old_string": "x = 1",
                        "new_string": "x = 42",
                    }
                )
            )
        )
        assert re.search(r"2 times", _error_text(r))
        assert f.read_text() == "x = 1\nx = 1\n"

    @pytest.mark.anyio
    async def test_replace_all(self, tmp_path: Path) -> None:
        f = tmp_path / "test.py"
        f.write_text("x = 1\nx = 1\n")
        await read.run(_msg_read(str(f)))
        response = await edit.run(
            _msg_edit(
                json_freeze(
                    {
                        "file_path": str(f),
                        "old_string": "x = 1",
                        "new_string": "x = 42",
                        "replace_all": True,
                    }
                )
            )
        )
        assert f.read_text() == "x = 42\nx = 42\n"
        assert "2 occurrence" in _txt(response)

    @pytest.mark.anyio
    async def test_replace_all_string_false_is_false(self, tmp_path: Path) -> None:
        f = tmp_path / "test.py"
        f.write_text("x = 1\nx = 1\n")
        r = await edit.run(
            _msg_edit(
                json_freeze(
                    {
                        "file_path": str(f),
                        "old_string": "x = 1",
                        "new_string": "x = 42",
                        "replace_all": "false",
                    }
                )
            )
        )
        assert "2 times" in _error_text(r)
        assert f.read_text() == "x = 1\nx = 1\n"

    @pytest.mark.anyio
    async def test_relative_path_uses_bash_cwd(self, tmp_path: Path) -> None:
        f = tmp_path / "test.py"
        f.write_text("x = 1\n")
        state = ToolState()
        state.bash_cwd = str(tmp_path)
        with tool_state_context(state):
            await edit.run(
                _msg_edit(
                    json_freeze(
                        {
                            "file_path": "test.py",
                            "old_string": "x = 1",
                            "new_string": "x = 2",
                        }
                    )
                )
            )
        assert f.read_text() == "x = 2\n"

    @pytest.mark.anyio
    async def test_not_found(self, tmp_path: Path) -> None:
        f = tmp_path / "test.py"
        f.write_text("x = 1\n")
        await read.run(_msg_read(str(f)))
        r = await edit.run(
            _msg_edit(
                json_freeze(
                    {
                        "file_path": str(f),
                        "old_string": "NOPE",
                        "new_string": "x",
                    }
                )
            )
        )
        assert re.search(r"not found", _error_text(r))

    @pytest.mark.anyio
    async def test_edit_without_prior_read(self, tmp_path: Path) -> None:
        f = tmp_path / "allowed.py"
        f.write_text("x = 1\n")
        response = await edit.run(
            _msg_edit(
                json_freeze(
                    {
                        "file_path": str(f),
                        "old_string": "x = 1",
                        "new_string": "x = 42",
                    }
                )
            )
        )
        assert f.read_text() == "x = 42\n"
        assert "1 occurrence" in _txt(response)


class TestEditEdgeCases:
    @pytest.mark.anyio
    async def test_edit_file_not_found(self) -> None:
        r = await edit.run(
            _msg_edit(
                json_freeze(
                    {
                        "file_path": "/nonexistent/x.py",
                        "old_string": "x",
                        "new_string": "y",
                    }
                )
            )
        )
        assert re.search(r"not found", _error_text(r))


class TestEditConcurrency:
    """Lock-based serialization of mutating ops on the same file.

    These tests exercise ``get_file_write_lock`` - the module-level
    ``asyncio.Lock`` registry that Edit (and Write) acquire before
    dispatching ``_run`` to the thread pool. Without it, two parallel
    Edits on the same file could both pass ``check_stale`` and then
    clobber each other's writes.
    """

    @pytest.mark.anyio
    async def test_same_file_edits_serialize(self, tmp_path: Path) -> None:
        """Two concurrent Edits on the same file must both apply.

        Without the lock, the second Edit's ``check_stale`` (run under
        a stale cached mtime) might race with the first Edit's write
        and last-writer-wins - losing one change silently.
        """
        f = tmp_path / "serial.py"
        f.write_text("A\nB\n")
        state = ToolState()
        with tool_state_context(state):
            await read.run(_msg_read(str(f)))
            results = await asyncio.gather(
                edit.run(
                    _msg_edit(
                        json_freeze(
                            {
                                "file_path": str(f),
                                "old_string": "A",
                                "new_string": "A-edited",
                            }
                        )
                    )
                ),
                edit.run(
                    _msg_edit(
                        json_freeze(
                            {
                                "file_path": str(f),
                                "old_string": "B",
                                "new_string": "B-edited",
                            }
                        )
                    )
                ),
            )
        # Both must succeed.
        for r in results:
            assert isinstance(r, MessageBase), r
        text = f.read_text()
        assert "A-edited" in text
        assert "B-edited" in text

    @pytest.mark.anyio
    async def test_preserves_file_mode(self, tmp_path: Path) -> None:
        """Atomic rename creates a fresh inode - the write path must
        carry the original file's mode forward or we silently flip
        e.g. ``0o600`` → ``0o644``.
        """
        f = tmp_path / "secret.txt"
        f.write_text("hello\n")
        f.chmod(0o600)
        state = ToolState()
        with tool_state_context(state):
            await read.run(_msg_read(str(f)))
            response = await edit.run(
                _msg_edit(
                    json_freeze(
                        {
                            "file_path": str(f),
                            "old_string": "hello",
                            "new_string": "world",
                        }
                    )
                )
            )
        assert isinstance(response, MessageBase)
        assert f.read_text() == "world\n"
        assert (f.stat().st_mode & 0o777) == 0o600, (
            f"mode changed: {oct(f.stat().st_mode & 0o777)}"
        )

    @pytest.mark.anyio
    async def test_different_files_edit_in_parallel(
        self,
        tmp_path: Path,
    ) -> None:
        """Different paths use different locks → no serialization."""
        a = tmp_path / "a.py"
        b = tmp_path / "b.py"
        a.write_text("foo\n")
        b.write_text("foo\n")
        state = ToolState()
        with tool_state_context(state):
            await read.run(_msg_read(str(a)))
            await read.run(_msg_read(str(b)))
            results = await asyncio.gather(
                edit.run(
                    _msg_edit(
                        json_freeze(
                            {
                                "file_path": str(a),
                                "old_string": "foo",
                                "new_string": "bar",
                            }
                        )
                    )
                ),
                edit.run(
                    _msg_edit(
                        json_freeze(
                            {
                                "file_path": str(b),
                                "old_string": "foo",
                                "new_string": "bar",
                            }
                        )
                    )
                ),
            )
        for r in results:
            assert isinstance(r, MessageBase)
        assert a.read_text() == "bar\n"
        assert b.read_text() == "bar\n"


class TestDescribe:
    def test_label_has_filename(self) -> None:
        label = edit.summary(
            _msg_edit(
                json_freeze(
                    {
                        "file_path": "/home/user/foo.py",
                        "old_string": "a",
                        "new_string": "b",
                    }
                )
            )
        )
        assert "foo.py" in label

    def test_label_is_short_no_diff(self) -> None:
        """help() should return a short label without a diff."""
        label = edit.summary(
            _msg_edit(
                json_freeze(
                    {
                        "file_path": "/home/user/foo.py",
                        "old_string": "a",
                        "new_string": "b",
                    }
                )
            )
        )
        assert label == "Edit foo.py"
        assert "@@" not in label

    @pytest.mark.anyio
    async def test_run_returns_diff(self, tmp_path: Path) -> None:
        """run() should return a multipart result including a text/x-diff."""
        f = tmp_path / "f.py"
        f.write_text("line1\nunique\nline3\n")
        result = await edit.run(
            _msg_edit(
                json_freeze(
                    {
                        "file_path": str(f),
                        "old_string": "unique",
                        "new_string": "replaced",
                    }
                )
            )
        )
        # Result is multipart with diff inside.
        assert isinstance(result, MultipartMessage)
        descriptors = [p.descriptor for p in result.content]
        assert "text/x-diff" in descriptors
        diff_part = next(p for p in result.content if p.descriptor == "text/x-diff")
        assert isinstance(diff_part, TextMessage)
        assert "@@ -2" in diff_part.content  # offset=1 → line 2


# -- Edit.bash_match ---------------------------------------------------


def _match(command: str) -> str | None:
    trees = parse_bash(command)
    assert trees is not None
    return edit.bash_match(trees)


_NUDGE = "sed via Bash is a bad UX. Use the Edit tool."


class TestBashMatchFires:
    def test_simple_replace_all(self) -> None:
        assert _match("sed -i 's/foo/bar/g' file.txt") == _NUDGE

    def test_single_replace(self) -> None:
        assert _match("sed -i 's/foo/bar/' file.txt") == _NUDGE


class TestBashMatchBails:
    def test_no_in_place(self) -> None:
        # Without -i, sed streams to stdout - not an Edit.
        assert _match("sed 's/foo/bar/' file.txt") is None

    def test_in_place_with_backup(self) -> None:
        assert _match("sed -i.bak 's/foo/bar/g' file.txt") is None

    def test_address_range(self) -> None:
        assert _match("sed -i '1,10 s/foo/bar/g' file.txt") is None

    def test_alt_delimiter(self) -> None:
        # s!X!Y! with ! delimiter - we only handle /.
        assert _match("sed -i 's!foo!bar!g' file.txt") is None

    def test_case_insensitive(self) -> None:
        # 'i' flag on s///i - Edit is exact-match only.
        assert _match("sed -i 's/foo/bar/gi' file.txt") is None

    def test_multiple_commands(self) -> None:
        # -e with multiple -e or combined script.
        assert _match("sed -i -e 's/a/b/' -e 's/c/d/' file.txt") is None

    def test_no_file(self) -> None:
        assert _match("sed -i 's/foo/bar/g'") is None

    def test_escaped_delimiter(self) -> None:
        # We don't handle escaped slashes in old/new.
        r = _match(r"sed -i 's/foo\/baz/bar/g' file.txt")
        assert r is None


class TestBashMatchCdPrefix:
    def test_joined_path(self) -> None:
        # Fixed-string hint - cd prefix no longer affects the hint
        # content; the LLM handles path resolution itself.
        assert _match("cd src && sed -i 's/foo/bar/g' file.txt") == _NUDGE


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
