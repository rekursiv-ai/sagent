"""Tests for debug_log."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

import json

from sagent.lib import debug_log


if TYPE_CHECKING:
    import pytest


class TestTrace:
    def test_writes_when_debug_enabled(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        log = tmp_path / "debug.log"
        monkeypatch.setenv("SAGENT_DEBUG", "1")
        monkeypatch.setenv("SAGENT_DEBUG_LOG", str(log))
        debug_log.trace("test_event", key="val")
        lines = log.read_text().splitlines()
        assert len(lines) == 1
        rec = json.loads(lines[0])
        assert rec["event"] == "test_event"
        assert rec["key"] == "val"

    def test_noop_when_debug_disabled(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        log = tmp_path / "debug.log"
        monkeypatch.delenv("SAGENT_DEBUG", raising=False)
        monkeypatch.setenv("SAGENT_DEBUG_LOG", str(log))
        debug_log.trace("should_not_appear")
        assert not log.exists()


class TestTraceError:
    def test_always_writes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        log = tmp_path / "debug.log"
        monkeypatch.setenv("SAGENT_DEBUG_LOG", str(log))
        debug_log.trace_error("err_event", code=42)
        rec = json.loads(log.read_text())
        assert rec["event"] == "err_event"
        assert rec["code"] == 42


class TestWrite:
    def test_swallows_exceptions(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SAGENT_DEBUG_LOG", "/proc/0/impossible")
        debug_log._write("boom", {"x": 1})


class TestSummarizeMessages:
    def test_non_dict(self) -> None:
        result = debug_log.summarize_messages([42, "hello"])
        assert result[0] == {"role": "?", "raw": "42"}
        assert result[1]["role"] == "?"
        assert cast(str, result[1]["raw"]).startswith("'hello'")

    def test_str_content(self) -> None:
        result = debug_log.summarize_messages([{"role": "user", "content": "hi there"}])
        assert result == [{"role": "user", "text": "hi there"}]

    def test_list_content(self) -> None:
        msgs = [{"role": "assistant", "content": [{"type": "text", "text": "yo"}]}]
        result = debug_log.summarize_messages(msgs)
        assert result[0]["role"] == "assistant"
        assert result[0]["blocks"] == [{"type": "text", "preview": "yo"}]

    def test_other_content(self) -> None:
        result = debug_log.summarize_messages([{"role": "system", "content": None}])
        assert result == [{"role": "system", "content": None}]


class TestRoleSequence:
    def test_mixed(self) -> None:
        msgs: list[object] = [
            {"role": "user"},
            {"role": "assistant"},
            99,
            {"role": 123},
        ]
        assert debug_log.role_sequence(msgs) == ["user", "assistant", "?", "?"]


class TestSummarizeBlock:
    def test_non_mapping(self) -> None:
        assert debug_log._summarize_block("not a dict") == {"type": "?"}

    def test_text(self) -> None:
        b: dict[str, object] = {"type": "text", "text": "hello world"}
        assert debug_log._summarize_block(b) == {
            "type": "text",
            "preview": "hello world",
        }

    def test_text_non_str(self) -> None:
        b: dict[str, object] = {"type": "text", "text": 12_345}
        assert debug_log._summarize_block(b) == {"type": "text", "preview": "12345"}

    def test_tool_use(self) -> None:
        b: dict[str, object] = {"type": "tool_use", "name": "grep", "id": "abc"}
        assert debug_log._summarize_block(b) == {
            "type": "tool_use",
            "name": "grep",
            "id": "abc",
        }

    def test_tool_result_str(self) -> None:
        b: dict[str, object] = {
            "type": "tool_result",
            "content": "ok",
            "is_error": False,
        }
        assert debug_log._summarize_block(b) == {
            "type": "tool_result",
            "preview": "ok",
            "is_error": False,
        }

    def test_tool_result_non_str(self) -> None:
        b: dict[str, object] = {
            "type": "tool_result",
            "content": {"k": "v"},
            "is_error": True,
        }
        result = debug_log._summarize_block(b)
        assert result["type"] == "tool_result"
        assert result["is_error"] is True
        assert json.loads(cast(str, result["preview"])) == {"k": "v"}

    def test_unknown_type(self) -> None:
        assert debug_log._summarize_block({"type": "image"}) == {"type": "image"}


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
