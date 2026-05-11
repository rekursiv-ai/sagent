"""Tests for Grep tool."""

from __future__ import annotations

from pathlib import Path

import subprocess
import subprocess as _sp

import pytest

from sagent.custom_types import (
    JsonMessage,
    Message,
    MultipartMessage,
    TextMessage,
)
from sagent.lib.json import JSON, json_freeze
from sagent.tools.grep import Grep
from sagent.tools.lib.bash import parse_bash

import sagent.tools.grep as gm


def _msg(directive: JSON) -> MultipartMessage:
    return MultipartMessage(
        (JsonMessage(directive, "application/x-tool-x"),),
        "multipart/x-tool-call",
    )


grep = Grep()
summary_grep = Grep()
summary_grep.emit_tool_summary = True


def _text(r: Message) -> str:
    if isinstance(r, TextMessage):
        return r.content
    if isinstance(r, MultipartMessage):
        for p in r.content:
            if isinstance(p, TextMessage) and p.descriptor == "text/plain":
                return p.content
    return ""


class TestGrep:
    @pytest.mark.anyio
    async def test_grep_pattern(self, tmp_path: Path) -> None:
        f = tmp_path / "test.py"
        f.write_text("def foo():\n    pass\ndef bar():\n    pass\n")
        response = await grep.run(
            _msg(
                json_freeze(
                    {
                        "pattern": "def \\w+",
                        "path": str(tmp_path),
                        "output_mode": "content",
                    }
                )
            )
        )
        assert "foo" in _text(response)
        assert "bar" in _text(response)

    @pytest.mark.anyio
    async def test_grep_no_match(self, tmp_path: Path) -> None:
        f = tmp_path / "test.py"
        f.write_text("hello\n")
        response = await grep.run(
            _msg(json_freeze({"pattern": "NOPE", "path": str(tmp_path)}))
        )
        assert "no matches" in _text(response).lower()

    @pytest.mark.anyio
    async def test_summary_result_reports_hit_count(self, tmp_path: Path) -> None:
        """Grep.summary_result returns ``"{N} hits"`` for non-empty output."""
        (tmp_path / "a.py").write_text("def foo(): pass\n")
        (tmp_path / "b.py").write_text("def bar(): pass\n")
        response = await summary_grep.run(
            _msg(json_freeze({"pattern": "def", "path": str(tmp_path)}))
        )
        # Default output_mode="files_with_matches" → 2 lines, one path each.
        assert summary_grep.summary_result(response) == "2 hits"

    @pytest.mark.anyio
    async def test_summary_result_reports_no_matches(self, tmp_path: Path) -> None:
        """Grep.summary_result returns ``"no matches"`` when empty."""
        (tmp_path / "test.py").write_text("hello\n")
        response = await summary_grep.run(
            _msg(json_freeze({"pattern": "NOPE", "path": str(tmp_path)}))
        )
        assert summary_grep.summary_result(response) == "no matches"

    @pytest.mark.anyio
    async def test_summary_result_off_by_default(self, tmp_path: Path) -> None:
        """Grep.summary_result returns None when the gate is off."""
        (tmp_path / "a.py").write_text("def foo(): pass\n")
        response = await grep.run(
            _msg(json_freeze({"pattern": "def", "path": str(tmp_path)}))
        )
        assert grep.summary_result(response) is None

    @pytest.mark.anyio
    async def test_grep_files_with_matches(self, tmp_path: Path) -> None:
        (tmp_path / "a.py").write_text("def foo(): pass\n")
        (tmp_path / "b.py").write_text("x = 1\n")
        response = await grep.run(
            _msg(
                json_freeze(
                    {
                        "pattern": "def",
                        "path": str(tmp_path),
                        "output_mode": "files_with_matches",
                    }
                )
            )
        )
        assert "a.py" in _text(response)
        assert "b.py" not in _text(response)

    @pytest.mark.anyio
    async def test_grep_count(self, tmp_path: Path) -> None:
        (tmp_path / "a.py").write_text("x\nx\nx\n")
        response = await grep.run(
            _msg(
                json_freeze(
                    {"pattern": "x", "path": str(tmp_path), "output_mode": "count"}
                )
            )
        )
        assert ":3" in _text(response)

    @pytest.mark.anyio
    async def test_grep_file_type(self, tmp_path: Path) -> None:
        (tmp_path / "a.py").write_text("target\n")
        (tmp_path / "b.js").write_text("target\n")
        response = await grep.run(
            _msg(
                json_freeze(
                    {"pattern": "target", "path": str(tmp_path), "file_type": "py"}
                )
            )
        )
        assert "a.py" in _text(response)
        assert "b.js" not in _text(response)

    @pytest.mark.anyio
    async def test_grep_context(self, tmp_path: Path) -> None:
        (tmp_path / "a.txt").write_text("aaa\nbbb\nccc\nddd\neee\n")
        response = await grep.run(
            _msg(
                json_freeze(
                    {
                        "pattern": "ccc",
                        "path": str(tmp_path),
                        "output_mode": "content",
                        "context_before": 1,
                        "context_after": 1,
                    }
                )
            )
        )
        assert "bbb" in _text(response)
        assert "ddd" in _text(response)

    @pytest.mark.anyio
    async def test_grep_multiline(self, tmp_path: Path) -> None:
        (tmp_path / "a.txt").write_text("start\nmiddle\nend\n")
        response = await grep.run(
            _msg(
                json_freeze(
                    {
                        "pattern": "start.*end",
                        "path": str(tmp_path),
                        "multiline": True,
                        "output_mode": "content",
                    }
                )
            )
        )
        assert "start" in _text(response)


class TestGrepPythonFallback:
    @pytest.mark.anyio
    async def test_content(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(gm, "_RG_PATH", None)
        (tmp_path / "a.py").write_text("def foo():\n    pass\n")
        r = await grep.run(
            _msg(
                json_freeze(
                    {"pattern": "def", "path": str(tmp_path), "output_mode": "content"}
                )
            )
        )
        assert "foo" in _text(r)

    @pytest.mark.anyio
    async def test_files_with_matches(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(gm, "_RG_PATH", None)
        (tmp_path / "a.py").write_text("target\n")
        (tmp_path / "b.py").write_text("nope\n")
        r = await grep.run(
            _msg(
                json_freeze(
                    {
                        "pattern": "target",
                        "path": str(tmp_path),
                        "output_mode": "files_with_matches",
                    }
                )
            )
        )
        assert "a.py" in _text(r)
        assert "b.py" not in _text(r)

    @pytest.mark.anyio
    async def test_count(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(gm, "_RG_PATH", None)
        (tmp_path / "a.py").write_text("x\nx\nx\n")
        r = await grep.run(
            _msg(
                json_freeze(
                    {"pattern": "x", "path": str(tmp_path), "output_mode": "count"}
                )
            )
        )
        assert ":3" in _text(r)

    @pytest.mark.anyio
    async def test_context(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(gm, "_RG_PATH", None)
        (tmp_path / "a.txt").write_text("aaa\nbbb\nccc\n")
        r = await grep.run(
            _msg(
                json_freeze(
                    {
                        "pattern": "bbb",
                        "path": str(tmp_path),
                        "context_before": 1,
                        "context_after": 1,
                        "output_mode": "content",
                    }
                )
            )
        )
        assert "aaa" in _text(r)
        assert "ccc" in _text(r)

    @pytest.mark.anyio
    async def test_multiline(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(gm, "_RG_PATH", None)
        (tmp_path / "a.txt").write_text("start\nmiddle\nend\n")
        r = await grep.run(
            _msg(
                json_freeze(
                    {
                        "pattern": "start.*end",
                        "path": str(tmp_path),
                        "multiline": True,
                        "output_mode": "content",
                    }
                )
            )
        )
        assert "start" in _text(r)

    @pytest.mark.anyio
    async def test_multiline_files_with_matches(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(gm, "_RG_PATH", None)
        (tmp_path / "a.txt").write_text("start\nend\n")
        r = await grep.run(
            _msg(
                json_freeze(
                    {
                        "pattern": "start.*end",
                        "path": str(tmp_path),
                        "multiline": True,
                        "output_mode": "files_with_matches",
                    }
                )
            )
        )
        assert "a.txt" in _text(r)

    @pytest.mark.anyio
    async def test_multiline_count(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(gm, "_RG_PATH", None)
        (tmp_path / "a.txt").write_text("ab\ncd\nab\n")
        r = await grep.run(
            _msg(
                json_freeze(
                    {
                        "pattern": "ab",
                        "path": str(tmp_path),
                        "multiline": True,
                        "output_mode": "count",
                    }
                )
            )
        )
        assert "a.txt" in _text(r)

    @pytest.mark.anyio
    async def test_case_insensitive(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(gm, "_RG_PATH", None)
        (tmp_path / "a.txt").write_text("Hello World\n")
        r = await grep.run(
            _msg(
                json_freeze(
                    {
                        "pattern": "hello",
                        "path": str(tmp_path),
                        "case_insensitive": True,
                        "output_mode": "content",
                    }
                )
            )
        )
        assert "Hello" in _text(r)

    @pytest.mark.anyio
    async def test_file_type(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(gm, "_RG_PATH", None)
        (tmp_path / "a.py").write_text("target\n")
        (tmp_path / "b.js").write_text("target\n")
        r = await grep.run(
            _msg(
                json_freeze(
                    {"pattern": "target", "path": str(tmp_path), "file_type": "py"}
                )
            )
        )
        assert "a.py" in _text(r)
        assert "b.js" not in _text(r)

    @pytest.mark.anyio
    async def test_glob_filter(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(gm, "_RG_PATH", None)
        (tmp_path / "a.py").write_text("target\n")
        (tmp_path / "b.txt").write_text("target\n")
        r = await grep.run(
            _msg(
                json_freeze(
                    {"pattern": "target", "path": str(tmp_path), "glob_filter": "*.py"}
                )
            )
        )
        assert "a.py" in _text(r)

    @pytest.mark.anyio
    async def test_single_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(gm, "_RG_PATH", None)
        f = tmp_path / "a.txt"
        f.write_text("hello world\n")
        r = await grep.run(
            _msg(
                json_freeze(
                    {"pattern": "hello", "path": str(f), "output_mode": "content"}
                )
            )
        )
        assert "hello" in _text(r)

    @pytest.mark.anyio
    async def test_no_matches(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(gm, "_RG_PATH", None)
        (tmp_path / "a.txt").write_text("hello\n")
        r = await grep.run(
            _msg(json_freeze({"pattern": "NOPE", "path": str(tmp_path)}))
        )
        assert "no matches" in _text(r).lower()


class TestGrepRg:
    """Tests for _grep_rg via mocked subprocess and _RG_PATH."""

    @pytest.mark.anyio
    async def test_rg_content_mode(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(gm, "_RG_PATH", "/usr/bin/rg")

        def _fake_run(cmd: list[str], **_kw: object) -> _sp.CompletedProcess[str]:
            result = _sp.CompletedProcess(cmd, 0, stdout="", stderr="")
            result.stdout = "file.py:1:match line\n"
            result.stderr = ""
            return result

        monkeypatch.setattr(subprocess, "run", _fake_run)
        r = await grep.run(
            _msg(json_freeze({"pattern": "match", "path": str(tmp_path)}))
        )
        assert "match line" in _text(r)

    @pytest.mark.anyio
    async def test_rg_files_with_matches(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(gm, "_RG_PATH", "/usr/bin/rg")

        def _fake_run(cmd: list[str], **_kw: object) -> _sp.CompletedProcess[str]:
            result = _sp.CompletedProcess(cmd, 0, stdout="", stderr="")
            result.stdout = "a.py\n"
            result.stderr = ""
            return result

        monkeypatch.setattr(subprocess, "run", _fake_run)
        r = await grep.run(
            _msg(
                json_freeze(
                    {
                        "pattern": "match",
                        "path": str(tmp_path),
                        "output_mode": "files_with_matches",
                    }
                )
            )
        )
        assert "a.py" in _text(r)

    @pytest.mark.anyio
    async def test_rg_count_mode(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(gm, "_RG_PATH", "/usr/bin/rg")

        def _fake_run(cmd: list[str], **_kw: object) -> _sp.CompletedProcess[str]:
            result = _sp.CompletedProcess(cmd, 0, stdout="", stderr="")
            result.stdout = "a.py:3\n"
            result.stderr = ""
            return result

        monkeypatch.setattr(subprocess, "run", _fake_run)
        r = await grep.run(
            _msg(
                json_freeze(
                    {"pattern": "x", "path": str(tmp_path), "output_mode": "count"}
                )
            )
        )
        assert ":3" in _text(r)

    @pytest.mark.anyio
    async def test_rg_no_matches(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(gm, "_RG_PATH", "/usr/bin/rg")

        def _fake_run(cmd: list[str], **_kw: object) -> _sp.CompletedProcess[str]:
            result = _sp.CompletedProcess(cmd, 1, stdout="", stderr="")
            result.stdout = ""
            result.stderr = ""
            return result

        monkeypatch.setattr(subprocess, "run", _fake_run)
        r = await grep.run(
            _msg(json_freeze({"pattern": "NOPE", "path": str(tmp_path)}))
        )
        assert "no matches" in _text(r).lower()

    @pytest.mark.anyio
    async def test_rg_offset(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(gm, "_RG_PATH", "/usr/bin/rg")

        def _fake_run(cmd: list[str], **_kw: object) -> _sp.CompletedProcess[str]:
            result = _sp.CompletedProcess(cmd, 0, stdout="", stderr="")
            result.stdout = "a.py:1:line1\na.py:2:line2\na.py:3:line3\n"
            result.stderr = ""
            return result

        monkeypatch.setattr(subprocess, "run", _fake_run)
        r = await grep.run(
            _msg(json_freeze({"pattern": "line", "path": str(tmp_path), "offset": 1}))
        )
        assert "line1" not in _text(r)
        assert "line2" in _text(r)

    @pytest.mark.anyio
    async def test_rg_offset_reset_with_context(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(gm, "_RG_PATH", "/usr/bin/rg")
        captured_cmd: list[str] = []

        def _fake_run(cmd: list[str], **_kw: object) -> _sp.CompletedProcess[str]:
            captured_cmd.extend(cmd)
            result = _sp.CompletedProcess(cmd, 0, stdout="", stderr="")
            result.stdout = "a.py:1:line1\n"
            result.stderr = ""
            return result

        monkeypatch.setattr(subprocess, "run", _fake_run)
        await grep.run(
            _msg(
                json_freeze(
                    {
                        "pattern": "line",
                        "path": str(tmp_path),
                        "offset": 2,
                        "context_before": 1,
                    }
                )
            )
        )
        assert "-B" in captured_cmd

    @pytest.mark.anyio
    async def test_rg_case_insensitive(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(gm, "_RG_PATH", "/usr/bin/rg")
        captured: list[str] = []

        def _fake_run(cmd: list[str], **_kw: object) -> _sp.CompletedProcess[str]:
            captured.extend(cmd)
            result = _sp.CompletedProcess(cmd, 0, stdout="", stderr="")
            result.stdout = "a.py:1:Hello\n"
            result.stderr = ""
            return result

        monkeypatch.setattr(subprocess, "run", _fake_run)
        await grep.run(
            _msg(
                json_freeze(
                    {
                        "pattern": "hello",
                        "path": str(tmp_path),
                        "case_insensitive": True,
                    }
                )
            )
        )
        assert "-i" in captured

    @pytest.mark.anyio
    async def test_rg_multiline(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(gm, "_RG_PATH", "/usr/bin/rg")
        captured: list[str] = []

        def _fake_run(cmd: list[str], **_kw: object) -> _sp.CompletedProcess[str]:
            captured.extend(cmd)
            result = _sp.CompletedProcess(cmd, 0, stdout="", stderr="")
            result.stdout = "a.py:1:match\n"
            result.stderr = ""
            return result

        monkeypatch.setattr(subprocess, "run", _fake_run)
        await grep.run(
            _msg(
                json_freeze(
                    {"pattern": "match", "path": str(tmp_path), "multiline": True}
                )
            )
        )
        assert "-U" in captured

    @pytest.mark.anyio
    async def test_rg_glob_and_type(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(gm, "_RG_PATH", "/usr/bin/rg")
        captured: list[str] = []

        def _fake_run(cmd: list[str], **_kw: object) -> _sp.CompletedProcess[str]:
            captured.extend(cmd)
            result = _sp.CompletedProcess(cmd, 0, stdout="", stderr="")
            result.stdout = ""
            result.stderr = ""
            return result

        monkeypatch.setattr(subprocess, "run", _fake_run)
        await grep.run(
            _msg(
                json_freeze(
                    {
                        "pattern": "x",
                        "path": str(tmp_path),
                        "glob_filter": "*.py",
                        "file_type": "py",
                    }
                )
            )
        )
        assert "--glob" in captured
        assert "--type" in captured

    @pytest.mark.anyio
    async def test_context_after_in_rg(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Ensure context_after triggers -A flag in _grep_rg."""
        monkeypatch.setattr(gm, "_RG_PATH", "/usr/bin/rg")
        captured: list[str] = []

        def _fake_run(cmd: list[str], **_kw: object) -> _sp.CompletedProcess[str]:
            captured.extend(cmd)
            result = _sp.CompletedProcess(cmd, 0, stdout="", stderr="")
            result.stdout = "a.py:1:line1\n"
            result.stderr = ""
            return result

        monkeypatch.setattr(subprocess, "run", _fake_run)
        await grep.run(
            _msg(
                json_freeze(
                    {"pattern": "line", "path": str(tmp_path), "context_after": 2}
                )
            )
        )
        assert "-A" in captured


class TestGrepRgOffsetReset:
    @pytest.mark.anyio
    async def test_offset_with_context_resets(self, tmp_path: Path) -> None:
        (tmp_path / "a.py").write_text("a\nb\nc\nd\ne\n")
        response = await grep.run(
            _msg(
                json_freeze(
                    {
                        "pattern": "b",
                        "path": str(tmp_path),
                        "context_before": 1,
                        "context_after": 1,
                        "offset": 5,
                    }
                )
            )
        )
        assert _text(response) != ""


class TestGrepPythonAdditional:
    @pytest.mark.anyio
    async def test_multiline_offset_skips(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(gm, "_RG_PATH", None)
        (tmp_path / "a.txt").write_text("start\nend\n")
        (tmp_path / "b.txt").write_text("start\nend\n")
        r = await grep.run(
            _msg(
                json_freeze(
                    {
                        "pattern": "start.*end",
                        "path": str(tmp_path),
                        "multiline": True,
                        "offset": 1,
                    }
                )
            )
        )
        assert _text(r) != "(no matches)"

    @pytest.mark.anyio
    async def test_multiline_files_with_matches_offset(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(gm, "_RG_PATH", None)
        (tmp_path / "a.txt").write_text("start\nend\n")
        (tmp_path / "b.txt").write_text("start\nend\n")
        r = await grep.run(
            _msg(
                json_freeze(
                    {
                        "pattern": "start.*end",
                        "path": str(tmp_path),
                        "multiline": True,
                        "output_mode": "files_with_matches",
                        "offset": 1,
                    }
                )
            )
        )
        assert _text(r) != "(no matches)"

    @pytest.mark.anyio
    async def test_multiline_match_truncated(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(gm, "_RG_PATH", None)
        content = "start\n" + "x" * 300 + "\nend"
        (tmp_path / "a.txt").write_text(content)
        r = await grep.run(
            _msg(
                json_freeze(
                    {
                        "pattern": "start[\\s\\S]*end",
                        "path": str(tmp_path),
                        "multiline": True,
                        "output_mode": "content",
                    }
                )
            )
        )
        assert "..." in _text(r)

    @pytest.mark.anyio
    async def test_files_with_matches_offset_skips(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(gm, "_RG_PATH", None)
        (tmp_path / "a.py").write_text("target\n")
        (tmp_path / "b.py").write_text("target\n")
        r = await grep.run(
            _msg(
                json_freeze(
                    {
                        "pattern": "target",
                        "path": str(tmp_path),
                        "output_mode": "files_with_matches",
                        "offset": 1,
                    }
                )
            )
        )
        lines = [x for x in _text(r).split("\n") if x]
        assert len(lines) == 1

    @pytest.mark.anyio
    async def test_content_count_mode(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(gm, "_RG_PATH", None)
        (tmp_path / "a.py").write_text("x\ny\nx\n")
        r = await grep.run(
            _msg(
                json_freeze(
                    {"pattern": "x", "path": str(tmp_path), "output_mode": "count"}
                )
            )
        )
        assert ":2" in _text(r)

    @pytest.mark.anyio
    async def test_keep_first_truncates_multiline(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(gm, "_RG_PATH", None)
        (tmp_path / "a.txt").write_text("foo\nbar\nfoo\nbar\n")
        r = await grep.run(
            _msg(
                json_freeze(
                    {
                        "pattern": "foo",
                        "path": str(tmp_path),
                        "keep_first": 1,
                        "output_mode": "content",
                    }
                )
            )
        )
        assert "foo" in _text(r)

    @pytest.mark.anyio
    async def test_oserror_file_skipped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(gm, "_RG_PATH", None)
        f = tmp_path / "a.py"
        f.write_text("target\n")
        f.chmod(0o000)
        try:
            r = await grep.run(
                _msg(json_freeze({"pattern": "target", "path": str(tmp_path)}))
            )
            assert _text(r) is not None
        finally:
            f.chmod(0o644)

    @pytest.mark.anyio
    async def test_context_offset_skips(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(gm, "_RG_PATH", None)
        (tmp_path / "a.py").write_text("a\nb\nc\nd\ne\n")
        r = await grep.run(
            _msg(
                json_freeze(
                    {
                        "pattern": "[a-e]",
                        "path": str(tmp_path),
                        "offset": 2,
                        "output_mode": "content",
                    }
                )
            )
        )
        assert _text(r) != "(no matches)"

    @pytest.mark.anyio
    async def test_non_file_entry_skipped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(gm, "_RG_PATH", None)
        (tmp_path / "subdir").mkdir()
        (tmp_path / "a.txt").write_text("hello\n")
        r = await grep.run(
            _msg(
                json_freeze(
                    {
                        "pattern": "hello",
                        "path": str(tmp_path),
                        "glob_filter": "*",
                        "output_mode": "content",
                    }
                )
            )
        )
        assert "hello" in _text(r)

    @pytest.mark.anyio
    async def test_multiline_no_match_continues(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(gm, "_RG_PATH", None)
        (tmp_path / "a.txt").write_text("nope\n")
        (tmp_path / "b.txt").write_text("start\nend\n")
        r = await grep.run(
            _msg(
                json_freeze(
                    {"pattern": "start.*end", "path": str(tmp_path), "multiline": True}
                )
            )
        )
        assert "b.txt" in _text(r)

    @pytest.mark.anyio
    async def test_multiline_max_results_files_with_matches_break(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(gm, "_RG_PATH", None)
        for i in range(5):
            (tmp_path / f"f{i}.txt").write_text("start\nend\n")
        r = await grep.run(
            _msg(
                json_freeze(
                    {
                        "pattern": "start.*end",
                        "path": str(tmp_path),
                        "multiline": True,
                        "output_mode": "files_with_matches",
                        "keep_first": 2,
                    }
                )
            )
        )
        lines = [x for x in _text(r).split("\n") if x]
        assert len(lines) == 2

    @pytest.mark.anyio
    async def test_multiline_keep_first_content_break(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(gm, "_RG_PATH", None)
        content = "\n".join(f"start{i}\nend{i}" for i in range(10))
        (tmp_path / "a.txt").write_text(content)
        r = await grep.run(
            _msg(
                json_freeze(
                    {
                        "pattern": r"start\d",
                        "path": str(tmp_path),
                        "multiline": True,
                        "keep_first": 1,
                    }
                )
            )
        )
        lines = [x for x in _text(r).split("\n") if x]
        assert len(lines) == 1


class TestKeepLast:
    @pytest.mark.anyio
    async def test_keep_last_python(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(gm, "_RG_PATH", None)
        (tmp_path / "a.txt").write_text("\n".join(f"hit{i}" for i in range(10)))
        r = await grep.run(
            _msg(
                json_freeze(
                    {
                        "pattern": r"hit\d",
                        "path": str(tmp_path),
                        "output_mode": "content",
                        "keep_last": 3,
                    }
                )
            )
        )
        lines = [x for x in _text(r).split("\n") if x]
        assert len(lines) == 3
        assert "hit7" in lines[0]
        assert "hit9" in lines[2]

    @pytest.mark.anyio
    async def test_keep_last_files_with_matches_python(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(gm, "_RG_PATH", None)
        for i in range(5):
            (tmp_path / f"f{i}.txt").write_text("hit\n")
        r = await grep.run(
            _msg(json_freeze({"pattern": "hit", "path": str(tmp_path), "keep_last": 2}))
        )
        lines = [x for x in _text(r).split("\n") if x]
        assert len(lines) == 2
        assert lines[0].endswith("f3.txt")
        assert lines[1].endswith("f4.txt")

    @pytest.mark.anyio
    @pytest.mark.skipif(gm._RG_PATH is None, reason="requires ripgrep")
    async def test_keep_last_rg(self, tmp_path: Path) -> None:
        (tmp_path / "a.txt").write_text("\n".join(f"hit{i}" for i in range(10)))
        r = await grep.run(
            _msg(
                json_freeze(
                    {
                        "pattern": r"hit\d",
                        "path": str(tmp_path),
                        "output_mode": "content",
                        "keep_last": 3,
                    }
                )
            )
        )
        lines = [x for x in _text(r).split("\n") if x]
        assert len(lines) == 3
        assert "hit9" in lines[-1]

    @pytest.mark.anyio
    async def test_keep_last_overrides_keep_first(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(gm, "_RG_PATH", None)
        (tmp_path / "a.txt").write_text("\n".join(f"hit{i}" for i in range(10)))
        r = await grep.run(
            _msg(
                json_freeze(
                    {
                        "pattern": r"hit\d",
                        "path": str(tmp_path),
                        "output_mode": "content",
                        "keep_first": 5,
                        "keep_last": 2,
                    }
                )
            )
        )
        lines = [x for x in _text(r).split("\n") if x]
        assert len(lines) == 2
        assert "hit8" in lines[0]
        assert "hit9" in lines[1]


class TestGrepExclude:
    @pytest.mark.anyio
    async def test_exclude_glob_python(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(gm, "_RG_PATH", None)
        (tmp_path / "keep.py").write_text("foo\n")
        (tmp_path / "skip.test.py").write_text("foo\n")
        r = await grep.run(
            _msg(
                json_freeze(
                    {
                        "pattern": "foo",
                        "path": str(tmp_path),
                        "exclude": "*.test.py",
                        "output_mode": "files_with_matches",
                    }
                )
            )
        )
        assert "keep.py" in _text(r)
        assert "skip.test.py" not in _text(r)

    @pytest.mark.anyio
    async def test_exclude_glob_rg(self, tmp_path: Path) -> None:
        if gm._RG_PATH is None:
            pytest.skip("ripgrep not available")
        (tmp_path / "keep.py").write_text("foo\n")
        (tmp_path / "skip.test.py").write_text("foo\n")
        r = await grep.run(
            _msg(
                json_freeze(
                    {
                        "pattern": "foo",
                        "path": str(tmp_path),
                        "exclude": "*.test.py",
                        "output_mode": "files_with_matches",
                    }
                )
            )
        )
        assert "keep.py" in _text(r)
        assert "skip.test.py" not in _text(r)


class TestGrepPcre:
    @pytest.mark.anyio
    async def test_pcre_lookahead_rg(self, tmp_path: Path) -> None:
        if gm._RG_PATH is None:
            pytest.skip("ripgrep not available")
        f = tmp_path / "a.txt"
        f.write_text("foobar\nfoobaz\n")
        # PCRE2 lookahead: match 'foo' only when followed by 'bar'.
        r = await grep.run(
            _msg(
                json_freeze(
                    {
                        "pattern": "foo(?=bar)",
                        "path": str(tmp_path),
                        "pcre": True,
                        "output_mode": "content",
                    }
                )
            )
        )
        assert "foobar" in _text(r)
        assert "foobaz" not in _text(r)


# -- Grep.bash_match ---------------------------------------------------


def _match(command: str) -> str | None:
    trees = parse_bash(command)
    assert trees is not None
    return grep.bash_match(trees)


# Post-simplification: the nudge is a fixed string - we verify the
# matcher recognizes each supported bash shape but no longer assert
# on reconstructed Grep(pattern=..., path=..., glob=..., pcre=...)
# content (the LLM re-derives those from the original command).
_NUDGE = "grep via Bash is a bad UX. Use the Grep tool."


class TestBashMatchFiresForSimpleGrep:
    def test_pattern_and_path(self) -> None:
        assert _match("grep foo src/") == _NUDGE

    def test_pattern_only(self) -> None:
        assert _match("grep foo") == _NUDGE

    def test_bundled_short_flags(self) -> None:
        assert _match("grep -rln foo src/") == _NUDGE

    def test_no_filenames_flag_rh(self) -> None:
        # Regression: ``-h`` used to disqualify the nudge entirely.
        assert _match("grep -rh foo src/") == _NUDGE

    def test_always_filenames_flag_capital_h(self) -> None:
        assert _match("grep -rH foo src/") == _NUDGE

    def test_silent_flag(self) -> None:
        assert _match("grep -s foo file") == _NUDGE

    def test_value_flag_context(self) -> None:
        assert _match("grep -C 5 foo file") == _NUDGE

    def test_case_insensitive_flag(self) -> None:
        assert _match("grep -i foo file") == _NUDGE

    def test_extended_regex_flag(self) -> None:
        assert _match("grep -E foo file") == _NUDGE

    def test_include_long_flag_equals(self) -> None:
        assert _match("grep --include=*.py foo src/") == _NUDGE

    def test_include_long_flag_space(self) -> None:
        assert _match("grep --include *.py foo src/") == _NUDGE

    def test_cd_prefix(self) -> None:
        assert _match("cd src && grep foo .") == _NUDGE

    def test_cd_prefix_joined_path(self) -> None:
        assert _match("cd src && grep foo file.py") == _NUDGE

    def test_pcre_flag(self) -> None:
        assert _match("grep -P '(?=lookahead)foo' file") == _NUDGE

    def test_bre_alternation(self) -> None:
        assert (
            _match("grep -n 'AssistantMessage\\|StreamingMessage\\|streaming'")
            == _NUDGE
        )

    def test_bre_alternation_with_path(self) -> None:
        assert _match("grep -rn 'Foo\\|Bar' src/") == _NUDGE

    def test_ere_alternation(self) -> None:
        assert _match("grep -E 'Foo|Bar' src/") == _NUDGE

    def test_bre_other_metacharacters(self) -> None:
        assert _match(r"grep 'foo\+bar\?' src/") == _NUDGE

    def test_exclude_long_flag_equals(self) -> None:
        assert _match("grep --exclude=*.test.py foo src/") == _NUDGE

    def test_exclude_long_flag_space(self) -> None:
        assert _match("grep --exclude *.test.py foo src/") == _NUDGE

    def test_stderr_redirect_merge(self) -> None:
        # ``2>&1`` doesn't change grep's stdout - nudge must still fire.
        assert _match("grep -rn 'Foo\\|Bar' src/ 2>&1") == _NUDGE

    def test_stderr_redirect_devnull(self) -> None:
        assert _match("grep -rn 'Foo\\|Bar' src/ 2>/dev/null") == _NUDGE

    def test_pipe_to_head(self) -> None:
        assert _match("grep -rn 'Foo\\|Bar' src/ | head -30") == _NUDGE

    def test_pipe_to_tail(self) -> None:
        assert _match("grep -rn 'Foo\\|Bar' src/ | tail -n 50") == _NUDGE

    def test_pipe_to_less(self) -> None:
        assert _match("grep -rn 'Foo\\|Bar' src/ | less") == _NUDGE

    def test_stderr_redirect_then_pipe_to_head(self) -> None:
        # The real-world shape that slipped through: ``2>&1 | head -N``.
        assert _match("grep -rn 'Foo\\|Bar' src/ 2>&1 | head -30") == _NUDGE


class TestBashMatchPipelineShapes:
    def test_find_pipe_xargs_grep(self) -> None:
        assert _match("find src -name *.py | xargs grep foo") == _NUDGE

    def test_find_print0_xargs_0_grep(self) -> None:
        assert _match("find src -name *.py -print0 | xargs -0 grep foo") == _NUDGE

    def test_find_no_path_no_glob(self) -> None:
        assert _match("find . -type f | xargs grep foo") == _NUDGE

    def test_cat_pipe_grep(self) -> None:
        assert _match("cat file.txt | grep foo") == _NUDGE

    def test_cat_multi_file_bails(self) -> None:
        assert _match("cat a.txt b.txt | grep foo") is None

    def test_find_exec_bails(self) -> None:
        assert _match("find . -name *.py -exec grep foo {} ;") is None

    def test_xargs_with_replace_bails(self) -> None:
        # -I{} changes invocation pattern; can't round-trip.
        assert _match("find . -name *.py | xargs -I{} grep foo {}") is None

    def test_grep_pipe_wc_l(self) -> None:
        assert _match("grep foo file | wc -l") == _NUDGE

    def test_grep_rn_pipe_wc_l(self) -> None:
        assert _match("grep -rn 'pattern' src/ | wc -l") == _NUDGE

    def test_three_command_pipeline_bails(self) -> None:
        assert _match("find . | xargs grep foo | head") is None


class TestBashMatchBailsOnUnsupportedShapes:
    def test_unknown_long_flag(self) -> None:
        assert _match("grep --color=always foo file") is None

    def test_pipeline_wc_non_line_bails(self) -> None:
        assert _match("grep foo file | wc -c") is None
        assert _match("grep foo file | wc -w") is None
        assert _match("grep foo file | wc") is None

    def test_compound_after_non_cd(self) -> None:
        # Unwrapping only handles ``cd X && CMD``.
        assert _match("echo hi && grep foo .") is None

    def test_compound_three_commands(self) -> None:
        assert _match("cd src && cd tests && grep foo .") is None

    def test_redirect(self) -> None:
        assert _match("grep foo file > out.txt") is None

    def test_env_prefix(self) -> None:
        assert _match("LC_ALL=C grep foo file") is None

    def test_not_grep(self) -> None:
        assert _match("rg foo src/") is None
        assert _match("cat file") is None

    def test_missing_value_for_context(self) -> None:
        assert _match("grep -C") is None

    def test_too_many_positionals(self) -> None:
        # 3 positionals (pattern + 2 paths) is an ambiguous shape.
        assert _match("grep foo file1 file2") is None

    def test_zero_positionals(self) -> None:
        assert _match("grep -l") is None


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
