"""Tests for ``lib.debug_log``: JSONL trace + wire-message summarization."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import json

from sagent.lib.debug_log import (
    _write,
    log_path,
    role_sequence,
    summarize_messages,
    trace,
    trace_error,
)
from sagent.lib.userdirs import data_dir


if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def _read_records(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def test_log_path_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SAGENT_DEBUG_LOG", raising=False)
    p = log_path()
    assert p.name == "debug.log"
    assert p.parent == data_dir("sagent")


def test_log_path_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "custom.log"
    monkeypatch.setenv("SAGENT_DEBUG_LOG", str(target))
    assert log_path() == target


def test_trace_gated_off(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "debug.log"
    monkeypatch.setenv("SAGENT_DEBUG_LOG", str(target))
    monkeypatch.delenv("SAGENT_DEBUG", raising=False)
    trace("event", x=1)
    assert not target.exists()


def test_trace_writes_when_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "debug.log"
    monkeypatch.setenv("SAGENT_DEBUG_LOG", str(target))
    monkeypatch.setenv("SAGENT_DEBUG", "1")
    trace("hello", x=1, y="z")
    records = _read_records(target)
    assert len(records) == 1
    rec = records[0]
    assert rec["event"] == "hello"
    assert rec["x"] == 1
    assert rec["y"] == "z"
    assert "ts" in rec


def test_trace_error_always_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "debug.log"
    monkeypatch.setenv("SAGENT_DEBUG_LOG", str(target))
    monkeypatch.delenv("SAGENT_DEBUG", raising=False)
    trace_error("boom", code=400)
    records = _read_records(target)
    assert len(records) == 1
    assert records[0]["event"] == "boom"
    assert records[0]["code"] == 400


def test_write_drops_reserved_key_collisions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """User payload must not overwrite the reserved ``ts``/``event`` keys."""
    target = tmp_path / "debug.log"
    monkeypatch.setenv("SAGENT_DEBUG_LOG", str(target))
    _write("real_event", {"event": "hijacked", "ts": "not-a-number", "extra": 1})
    records = _read_records(target)
    assert records[0]["event"] == "real_event"
    assert isinstance(records[0]["ts"], (int, float))
    assert records[0]["extra"] == 1


def test_trace_does_not_raise_on_unserializable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "debug.log"
    monkeypatch.setenv("SAGENT_DEBUG_LOG", str(target))
    monkeypatch.setenv("SAGENT_DEBUG", "1")
    # ``object()`` triggers default=str fallback rather than blowing up.
    trace("e", obj=object())
    assert target.exists()


def test_summarize_messages_text_string() -> None:
    out = summarize_messages([{"role": "user", "content": "hello world"}])
    assert out == [{"role": "user", "text": "hello world"}]


def test_summarize_messages_text_truncates() -> None:
    long = "a" * 500
    out = summarize_messages([{"role": "user", "content": long}])
    assert isinstance(out[0]["text"], str)
    assert len(out[0]["text"]) == 200  # _MAX_PREVIEW


def test_summarize_messages_block_text() -> None:
    out = summarize_messages(
        [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
    )
    assert out == [{"role": "user", "blocks": [{"type": "text", "preview": "hi"}]}]


def test_summarize_messages_block_tool_use() -> None:
    out = summarize_messages(
        [
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "name": "Bash", "id": "abc"},
                ],
            }
        ]
    )
    assert out[0]["blocks"] == [{"type": "tool_use", "name": "Bash", "id": "abc"}]


def test_summarize_messages_block_tool_result_string() -> None:
    out = summarize_messages(
        [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "content": "ok", "is_error": False},
                ],
            }
        ]
    )
    block = out[0]["blocks"]
    assert isinstance(block, list)
    assert block[0] == {
        "type": "tool_result",
        "preview": "ok",
        "is_error": False,
    }


def test_summarize_messages_block_tool_result_complex() -> None:
    out = summarize_messages(
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "content": [{"x": 1}],
                        "is_error": True,
                    }
                ],
            }
        ]
    )
    block = out[0]["blocks"]
    assert isinstance(block, list)
    entry = cast("dict[str, object]", block[0])
    assert entry["is_error"] is True
    assert isinstance(entry["preview"], str)


def test_summarize_messages_block_unknown_type() -> None:
    out = summarize_messages([{"role": "u", "content": [{"type": "weird", "x": 1}]}])
    assert out[0]["blocks"] == [{"type": "weird"}]


def test_summarize_messages_block_not_dict() -> None:
    out = summarize_messages([{"role": "u", "content": ["not-a-dict"]}])
    assert out[0]["blocks"] == [{"type": "?"}]


def test_summarize_messages_other_content() -> None:
    out = summarize_messages([{"role": "u", "content": 42}])
    assert out == [{"role": "u", "content": None}]


def test_summarize_messages_non_mapping() -> None:
    out = summarize_messages(["not a dict"])
    assert out[0]["role"] == "?"
    raw = out[0]["raw"]
    assert isinstance(raw, str)


def test_role_sequence_extracts() -> None:
    msgs = [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
        "junk",
        {"content": "no role"},
        {"role": 123, "content": "bad role type"},
    ]
    assert role_sequence(msgs) == ["user", "assistant", "?", "?", "?"]


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
