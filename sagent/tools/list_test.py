"""Tests for the List tool (directory listing) and its ``bash_match``."""

from __future__ import annotations

from pathlib import Path

import pytest

from sagent.custom_types import (
    JsonMessage,
    Message,
    MultipartMessage,
    TextMessage,
)
from sagent.lib.json import JSON, json_freeze
from sagent.tools.core import ToolState, tool_state_context
from sagent.tools.lib.bash import parse_bash
from sagent.tools.list import List


list_tool = List()


def _msg(directive: JSON) -> Message:
    return MultipartMessage(
        (JsonMessage(directive, "application/x-tool-list"),),
        "multipart/x-tool-call",
    )


def _txt(msg: Message) -> str:
    if isinstance(msg, TextMessage):
        return msg.content
    if isinstance(msg, MultipartMessage):
        for p in msg.content:
            if isinstance(p, TextMessage):
                return p.content
    return ""


def _match(command: str) -> str | None:
    trees = parse_bash(command)
    assert trees is not None
    return list_tool.bash_match(trees)


class TestList:
    @pytest.mark.anyio
    async def test_basic(self, tmp_path: Path) -> None:
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "b").mkdir()
        r = await list_tool.run(_msg(json_freeze({"path": str(tmp_path)})))
        assert "a.txt" in _txt(r)
        assert "b/" in _txt(r)

    @pytest.mark.anyio
    async def test_hidden_excluded_by_default(self, tmp_path: Path) -> None:
        (tmp_path / ".hidden").write_text("")
        (tmp_path / "visible").write_text("")
        r = await list_tool.run(_msg(json_freeze({"path": str(tmp_path)})))
        assert "visible" in _txt(r)
        assert ".hidden" not in _txt(r)

    @pytest.mark.anyio
    async def test_hidden_included_with_flag(self, tmp_path: Path) -> None:
        (tmp_path / ".hidden").write_text("")
        r = await list_tool.run(
            _msg(json_freeze({"path": str(tmp_path), "show_hidden": True}))
        )
        assert ".hidden" in _txt(r)

    @pytest.mark.anyio
    async def test_long(self, tmp_path: Path) -> None:
        (tmp_path / "x.txt").write_text("abc")
        r = await list_tool.run(
            _msg(json_freeze({"path": str(tmp_path), "long": True}))
        )
        assert "x.txt" in _txt(r)
        # Size column should appear.
        assert "3" in _txt(r)

    @pytest.mark.anyio
    async def test_not_a_dir(self, tmp_path: Path) -> None:
        f = tmp_path / "f.txt"
        f.write_text("")
        r = await list_tool.run(_msg(json_freeze({"path": str(f)})))
        assert r.descriptor == "text/x-error"
        assert "Not a directory" in _txt(r)

    @pytest.mark.anyio
    async def test_missing(self) -> None:
        r = await list_tool.run(_msg(json_freeze({"path": "/nonexistent/path/xyz"})))
        assert r.descriptor == "text/x-error"
        assert "Not found" in _txt(r)

    @pytest.mark.anyio
    async def test_relative_path_uses_bash_cwd(self, tmp_path: Path) -> None:
        state = ToolState()
        state.bash_cwd = str(tmp_path)
        (tmp_path / "f.txt").write_text("")
        with tool_state_context(state):
            r = await list_tool.run(_msg(json_freeze({"path": "."})))
            assert "f.txt" in _txt(r)

    @pytest.mark.anyio
    async def test_show_hidden_string_false_is_false(self, tmp_path: Path) -> None:
        (tmp_path / ".hidden").write_text("")
        r = await list_tool.run(
            _msg(
                json_freeze(
                    {
                        "path": str(tmp_path),
                        "show_hidden": "false",
                    }
                )
            )
        )
        assert ".hidden" not in _txt(r)

    @pytest.mark.anyio
    async def test_max_results_truncates(self, tmp_path: Path) -> None:
        for i in range(10):
            (tmp_path / f"f{i}.txt").write_text("")
        r = await list_tool.run(
            _msg(json_freeze({"path": str(tmp_path), "max_results": 3}))
        )
        assert "more" in _txt(r)

    @pytest.mark.anyio
    async def test_empty_directory(self, tmp_path: Path) -> None:
        r = await list_tool.run(_msg(json_freeze({"path": str(tmp_path)})))
        assert "empty directory" in _txt(r)

    @pytest.mark.anyio
    async def test_iterdir_oserror(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def _raise(_self: Path) -> None:
            raise OSError("denied")

        monkeypatch.setattr(Path, "iterdir", _raise)
        r = await list_tool.run(_msg(json_freeze({"path": str(tmp_path)})))
        assert r.descriptor == "text/x-error"
        assert "Error reading" in _txt(r)

    @pytest.mark.anyio
    async def test_long_stat_error_via_broken_symlink(
        self,
        tmp_path: Path,
    ) -> None:
        # Dangling symlink → ``stat`` errors on follow, ``long=True``
        # falls into the except branch and prints ``?`` for mtime.
        link = tmp_path / "dangling"
        link.symlink_to(tmp_path / "nope")
        r = await list_tool.run(
            _msg(json_freeze({"path": str(tmp_path), "long": True}))
        )
        assert "dangling" in _txt(r)


_NUDGE = "ls via Bash is a bad UX. Use the List tool."
_NUDGE_GLOB = "ls glob via Bash is a bad UX. Use the Glob tool."


class TestBashMatchLs:
    def test_bare(self) -> None:
        assert _match("ls") == _NUDGE

    def test_dir(self) -> None:
        assert _match("ls src") == _NUDGE

    def test_long_flag(self) -> None:
        assert _match("ls -l") == _NUDGE

    def test_all_long_flags(self) -> None:
        assert _match("ls -la") == _NUDGE

    def test_unknown_flag_bails(self) -> None:
        # ``-h`` (human-readable sizes) has no List equivalent.
        assert _match("ls -h") is None

    def test_long_flag_bails(self) -> None:
        assert _match("ls --color=always") is None

    def test_glob_positional_routes_to_glob(self) -> None:
        assert _match("ls src/*.py") == _NUDGE_GLOB

    def test_cd_prefix(self) -> None:
        # Fixed-string hint - cd prefix doesn't affect the content.
        assert _match("cd src && ls tests") == _NUDGE

    def test_multi_positional_bails(self) -> None:
        assert _match("ls dir1 dir2") is None


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
