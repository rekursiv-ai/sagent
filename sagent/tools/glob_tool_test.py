"""Tests for Glob tool."""

from __future__ import annotations

from pathlib import (
    Path,
    Path as _Path,
)

import pytest

from sagent.custom_types import (
    JsonMessage,
    Message,
    MultipartMessage,
    TextMessage,
)
from sagent.lib.json import JSON, json_freeze
from sagent.tools.glob_tool import Glob
from sagent.tools.lib.bash import parse_bash


glob_tool = Glob()
summary_glob = Glob()
summary_glob.emit_tool_summary = True


def _msg(directive: JSON) -> Message:
    return MultipartMessage(
        (JsonMessage(directive, "application/x-tool-x"),),
        "multipart/x-tool-call",
    )


def _text(r: Message) -> str:
    if isinstance(r, TextMessage):
        return r.content
    if isinstance(r, MultipartMessage):
        for p in r.content:
            if isinstance(p, TextMessage) and p.descriptor == "text/plain":
                return p.content
    return ""


class TestGlob:
    @pytest.mark.anyio
    async def test_glob_files(self, tmp_path: Path) -> None:
        (tmp_path / "a.py").write_text("")
        (tmp_path / "b.py").write_text("")
        (tmp_path / "c.txt").write_text("")
        response = await glob_tool.run(
            _msg(json_freeze({"pattern": "*.py", "path": str(tmp_path)}))
        )
        assert "a.py" in _text(response)
        assert "b.py" in _text(response)
        assert "c.txt" not in _text(response)

    @pytest.mark.anyio
    async def test_glob_no_match(self, tmp_path: Path) -> None:
        response = await glob_tool.run(
            _msg(json_freeze({"pattern": "*.xyz", "path": str(tmp_path)}))
        )
        assert "no matches" in _text(response).lower()

    @pytest.mark.anyio
    async def test_summary_result_reports_match_count(self, tmp_path: Path) -> None:
        """Glob.summary_result returns ``"{N} matches"`` for hit results."""
        (tmp_path / "a.py").write_text("")
        (tmp_path / "b.py").write_text("")
        response = await summary_glob.run(
            _msg(json_freeze({"pattern": "*.py", "path": str(tmp_path)}))
        )
        assert summary_glob.summary_result(response) == "2 matches"

    @pytest.mark.anyio
    async def test_summary_result_reports_no_matches(self, tmp_path: Path) -> None:
        """Glob.summary_result returns ``"no matches"`` when nothing hit."""
        response = await summary_glob.run(
            _msg(json_freeze({"pattern": "*.xyz", "path": str(tmp_path)}))
        )
        assert summary_glob.summary_result(response) == "no matches"

    @pytest.mark.anyio
    async def test_summary_result_off_by_default(self, tmp_path: Path) -> None:
        """Glob.summary_result returns None when the gate is off."""
        response = await glob_tool.run(
            _msg(json_freeze({"pattern": "*.py", "path": str(tmp_path)}))
        )
        assert glob_tool.summary_result(response) is None


class TestGlobName:
    def test_tool_name_is_glob(self) -> None:
        assert glob_tool.name == "Glob"


class TestGlobEdgeCases:
    @pytest.mark.anyio
    async def test_max_results_truncates(self, tmp_path: Path) -> None:
        for i in range(10):
            (tmp_path / f"f{i}.txt").write_text("")
        r = await glob_tool.run(
            _msg(
                json_freeze(
                    {"pattern": "*.txt", "path": str(tmp_path), "max_results": 3}
                )
            )
        )
        assert "more" in _text(r).lower()


class TestGlobSort:
    @pytest.mark.anyio
    async def test_sort_name(self, tmp_path: Path) -> None:
        (tmp_path / "b.py").write_text("")
        (tmp_path / "a.py").write_text("")
        (tmp_path / "c.py").write_text("")
        r = await glob_tool.run(
            _msg(
                json_freeze({"pattern": "*.py", "path": str(tmp_path), "sort": "name"})
            )
        )
        names = [Path(line).name for line in _text(r).splitlines()]
        assert names == ["a.py", "b.py", "c.py"]

    @pytest.mark.anyio
    async def test_sort_invalid(self, tmp_path: Path) -> None:
        r = await glob_tool.run(
            _msg(
                json_freeze({"pattern": "*.py", "path": str(tmp_path), "sort": "bogus"})
            )
        )
        assert r.descriptor == "text/x-error"
        assert "unknown sort" in _text(r)

    @pytest.mark.anyio
    async def test_long_includes_size(self, tmp_path: Path) -> None:
        (tmp_path / "x.py").write_text("abcde")
        r = await glob_tool.run(
            _msg(json_freeze({"pattern": "*.py", "path": str(tmp_path), "long": True}))
        )
        # Size column appears.
        assert "5" in _text(r)
        assert "x.py" in _text(r)


class TestGlobOSError:
    @pytest.mark.anyio
    async def test_glob_oserror_stat(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        f = tmp_path / "a.txt"
        f.write_text("hi")

        _real_stat = _Path.stat

        def _raise_stat(self: _Path, follow_symlinks: bool = True) -> object:
            if self.name == "a.txt":
                raise OSError("stat error")
            return _real_stat(self, follow_symlinks=follow_symlinks)

        monkeypatch.setattr(_Path, "stat", _raise_stat)
        response = await glob_tool.run(
            _msg(json_freeze({"pattern": "*.txt", "path": str(tmp_path)}))
        )
        assert response is not None


class TestAbsolutePattern:
    @pytest.mark.anyio
    async def test_absolute_with_metachars(self, tmp_path: Path) -> None:
        (tmp_path / "a.py").write_text("")
        (tmp_path / "b.py").write_text("")
        # Pattern with absolute root + wildcard suffix.
        response = await glob_tool.run(
            _msg(json_freeze({"pattern": str(tmp_path / "*.py")}))
        )
        assert "a.py" in _text(response)

    @pytest.mark.anyio
    async def test_absolute_no_metachars(self, tmp_path: Path) -> None:
        f = tmp_path / "exact.txt"
        f.write_text("")
        response = await glob_tool.run(_msg(json_freeze({"pattern": str(f)})))
        assert "exact.txt" in _text(response)

    @pytest.mark.anyio
    async def test_absolute_nonexistent(self) -> None:
        response = await glob_tool.run(
            _msg(json_freeze({"pattern": "/does/not/exist"}))
        )
        assert "no matches" in _text(response).lower()


# -- Glob.bash_match ---------------------------------------------------


def _match(command: str) -> str | None:
    trees = parse_bash(command)
    assert trees is not None
    return glob_tool.bash_match(trees)


_NUDGE = "find via Bash is a bad UX. Use the Glob tool."


class TestBashMatchFires:
    def test_basic(self) -> None:
        assert _match("find src -name *.py") == _NUDGE

    def test_no_path(self) -> None:
        assert _match("find -name *.py") == _NUDGE

    def test_dot_path(self) -> None:
        assert _match("find . -name *.py") == _NUDGE

    def test_type_f_accepted(self) -> None:
        assert _match("find . -type f -name *.py") == _NUDGE

    def test_type_d_accepted(self) -> None:
        assert _match("find . -type d -name docs") == _NUDGE


class TestBashMatchBails:
    def test_exec_bails(self) -> None:
        assert _match("find . -name *.py -exec rm {} ;") is None

    def test_delete_bails(self) -> None:
        assert _match("find . -name *.pyc -delete") is None

    def test_newer_bails(self) -> None:
        assert _match("find . -newer foo.py") is None

    def test_mtime_bails(self) -> None:
        assert _match("find . -mtime -7") is None

    def test_maxdepth_bails(self) -> None:
        assert _match("find . -maxdepth 2 -name *.py") is None

    def test_no_name_bails(self) -> None:
        assert _match("find . -type f") is None

    def test_pipe_bails(self) -> None:
        assert _match("find . -name *.py | xargs grep foo") is None

    def test_unknown_flag_bails(self) -> None:
        assert _match("find . -name *.py -follow") is None


class TestBashMatchLsNotMatched:
    def test_ls_handled_by_list_tool(self) -> None:
        # Glob no longer owns ``ls`` - the List tool does.
        assert _match("ls src") is None


class TestBashMatchCdPrefix:
    def test_find_joined(self) -> None:
        # Fixed-string nudge - cd prefix doesn't affect the content.
        assert _match("cd src && find . -name *.py") == _NUDGE


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
