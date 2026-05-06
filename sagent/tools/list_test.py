"""Tests for the List tool (directory listing) and its ``bash_match``."""

from __future__ import annotations

from pathlib import Path

import os
import time

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


class TestListSort:
    @pytest.mark.anyio
    async def test_sort_name_default(self, tmp_path: Path) -> None:
        (tmp_path / "b").write_text("")
        (tmp_path / "a").write_text("")
        (tmp_path / "c").write_text("")
        r = await list_tool.run(_msg(json_freeze({"path": str(tmp_path)})))
        assert _txt(r).splitlines() == ["a", "b", "c"]

    @pytest.mark.anyio
    async def test_sort_name_desc(self, tmp_path: Path) -> None:
        (tmp_path / "a").write_text("")
        (tmp_path / "c").write_text("")
        (tmp_path / "b").write_text("")
        r = await list_tool.run(
            _msg(json_freeze({"path": str(tmp_path), "sort": "name_desc"}))
        )
        assert _txt(r).splitlines() == ["c", "b", "a"]

    @pytest.mark.anyio
    async def test_sort_mtime_desc(self, tmp_path: Path) -> None:
        a = tmp_path / "a"
        b = tmp_path / "b"
        c = tmp_path / "c"
        a.write_text("")
        b.write_text("")
        c.write_text("")
        now = time.time()
        os.utime(a, (now - 300, now - 300))
        os.utime(b, (now - 100, now - 100))
        os.utime(c, (now, now))
        r = await list_tool.run(
            _msg(json_freeze({"path": str(tmp_path), "sort": "mtime_desc"}))
        )
        assert _txt(r).splitlines() == ["c", "b", "a"]

    @pytest.mark.anyio
    async def test_sort_size_desc(self, tmp_path: Path) -> None:
        (tmp_path / "small").write_text("a")
        (tmp_path / "big").write_text("a" * 100)
        (tmp_path / "med").write_text("a" * 10)
        r = await list_tool.run(
            _msg(json_freeze({"path": str(tmp_path), "sort": "size_desc"}))
        )
        assert _txt(r).splitlines() == ["big", "med", "small"]

    @pytest.mark.anyio
    async def test_sort_invalid(self, tmp_path: Path) -> None:
        r = await list_tool.run(
            _msg(json_freeze({"path": str(tmp_path), "sort": "bogus"}))
        )
        assert r.descriptor == "text/x-error"
        assert "unknown sort" in _txt(r)

    @pytest.mark.anyio
    async def test_sort_name_case_insensitive(self, tmp_path: Path) -> None:
        # GNU ``ls`` mirror: 'apple' < 'Banana' < 'Cherry' (case-folded),
        # not raw ASCII order which would put all uppercase before lowercase.
        (tmp_path / "Banana").write_text("")
        (tmp_path / "Cherry").write_text("")
        (tmp_path / "apple").write_text("")
        r = await list_tool.run(_msg(json_freeze({"path": str(tmp_path)})))
        assert _txt(r).splitlines() == ["apple", "Banana", "Cherry"]


_NUDGE = "ls via Bash is a bad UX. Use the List tool."
_NUDGE_GLOB = "ls glob via Bash is a bad UX. Use the Glob tool."


class TestBashMatchLs:
    def test_bare(self) -> None:
        assert _match("ls") == _NUDGE

    def test_dir(self) -> None:
        assert _match("ls src") == _NUDGE

    def test_long_flag(self) -> None:
        assert (
            _match("ls -l")
            == "ls via Bash is a bad UX. Use the List tool with long=true."
        )

    def test_all_long_flags(self) -> None:
        assert _match("ls -la") == (
            "ls via Bash is a bad UX. Use the List tool with"
            " long=true, show_hidden=true."
        )

    def test_capital_a(self) -> None:
        assert _match("ls -A") == (
            "ls via Bash is a bad UX. Use the List tool with show_hidden=true."
        )

    def test_unknown_flag_bails(self) -> None:
        # ``-h`` (human-readable sizes) has no List equivalent.
        assert _match("ls -h") is None

    def test_long_flag_bails(self) -> None:
        assert _match("ls --color=always") is None

    def test_glob_positional_routes_to_glob(self) -> None:
        assert _match("ls src/*.py") == _NUDGE_GLOB

    def test_cd_prefix(self) -> None:
        assert _match("cd src && ls tests") == _NUDGE

    def test_multi_positional_bails(self) -> None:
        assert _match("ls dir1 dir2") is None


class TestBashMatchLsSort:
    def test_t_flag(self) -> None:
        assert _match("ls -t") == (
            "ls via Bash is a bad UX. Use the List tool with sort='mtime_desc'."
        )

    def test_tr_flags(self) -> None:
        assert _match("ls -tr") == (
            "ls via Bash is a bad UX. Use the List tool with sort='mtime'."
        )

    def test_capital_s(self) -> None:
        assert _match("ls -S") == (
            "ls via Bash is a bad UX. Use the List tool with sort='size_desc'."
        )

    def test_capital_s_reverse(self) -> None:
        assert _match("ls -Sr") == (
            "ls via Bash is a bad UX. Use the List tool with sort='size'."
        )

    def test_r_alone(self) -> None:
        assert _match("ls -r") == (
            "ls via Bash is a bad UX. Use the List tool with sort='name_desc'."
        )

    def test_lat_combo(self) -> None:
        assert _match("ls -lat") == (
            "ls via Bash is a bad UX. Use the List tool with"
            " sort='mtime_desc', long=true, show_hidden=true."
        )

    def test_t_and_size_conflict_bails(self) -> None:
        assert _match("ls -tS") is None


class TestBashMatchLsHead:
    def test_lat_head_5(self) -> None:
        assert _match("ls -lat ~/Downloads | head -5") == (
            "ls via Bash is a bad UX. Use the List tool with"
            " sort='mtime_desc', long=true, show_hidden=true, max_results=5."
        )

    def test_head_n_form(self) -> None:
        assert _match("ls -t | head -n 3") == (
            "ls via Bash is a bad UX. Use the List tool with"
            " sort='mtime_desc', max_results=3."
        )

    def test_head_clustered(self) -> None:
        assert _match("ls -t | head -n3") == (
            "ls via Bash is a bad UX. Use the List tool with"
            " sort='mtime_desc', max_results=3."
        )

    def test_bare_head_defaults_10(self) -> None:
        assert _match("ls | head") == (
            "ls via Bash is a bad UX. Use the List tool with max_results=10."
        )

    def test_head_byte_mode_bails(self) -> None:
        assert _match("ls | head -c 100") is None

    def test_head_with_file_arg_bails(self) -> None:
        assert _match("ls | head extra.txt") is None

    def test_pipe_to_other_bails(self) -> None:
        assert _match("ls | wc -l") is None

    def test_head_negative_count_bails(self) -> None:
        # ``head -n -5`` = "all but last 5" -- semantics don't map to max_results.
        assert _match("ls | head -n -5") is None

    def test_head_plus_count_bails(self) -> None:
        assert _match("ls | head -n +5") is None

    def test_head_zero_count_bails(self) -> None:
        assert _match("ls | head -n 0") is None


class TestBashMatchLsTail:
    def test_tail_flips_default_to_name_desc(self) -> None:
        assert _match("ls | tail -5") == (
            "ls via Bash is a bad UX. Use the List tool with"
            " sort='name_desc', max_results=5."
        )

    def test_tail_flips_mtime_desc_to_mtime(self) -> None:
        # ``ls -t | tail -5`` = oldest 5 entries.
        assert _match("ls -t | tail -5") == (
            "ls via Bash is a bad UX. Use the List tool with"
            " sort='mtime', max_results=5."
        )

    def test_tail_flips_size_desc_to_size(self) -> None:
        assert _match("ls -S | tail -3") == (
            "ls via Bash is a bad UX. Use the List tool with"
            " sort='size', max_results=3."
        )

    def test_tail_n_form(self) -> None:
        assert _match("ls -lat | tail -n 5") == (
            "ls via Bash is a bad UX. Use the List tool with"
            " sort='mtime', long=true, show_hidden=true, max_results=5."
        )

    def test_bare_tail_defaults_10(self) -> None:
        assert _match("ls | tail") == (
            "ls via Bash is a bad UX. Use the List tool with"
            " sort='name_desc', max_results=10."
        )

    def test_tail_plus_count_bails(self) -> None:
        # ``tail -n +5`` = "from line 5 onward" -- not "last 5".
        assert _match("ls | tail -n +5") is None

    def test_tail_negative_count_bails(self) -> None:
        assert _match("ls | tail -n -5") is None


class TestBashMatchLsCdPipeline:
    def test_cd_then_ls_pipe_head(self) -> None:
        assert _match("cd ~/Downloads && ls -lat | head -5") == (
            "ls via Bash is a bad UX. Use the List tool with"
            " sort='mtime_desc', long=true, show_hidden=true, max_results=5."
        )

    def test_cd_then_ls_pipe_tail(self) -> None:
        assert _match("cd src && ls -t | tail -3") == (
            "ls via Bash is a bad UX. Use the List tool with"
            " sort='mtime', max_results=3."
        )

    def test_cd_then_bare_ls(self) -> None:
        assert _match("cd src && ls -lat") == (
            "ls via Bash is a bad UX. Use the List tool with"
            " sort='mtime_desc', long=true, show_hidden=true."
        )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
