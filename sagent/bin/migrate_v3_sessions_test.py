"""Tests for ``bin.migrate_v3_sessions``: v3 -> v4 session migration."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import base64
import json

from sagent.bin.migrate_v3_sessions import (
    iter_v4_records,
    main,
    migrate_file,
)


if TYPE_CHECKING:
    from pathlib import Path


def _records(*lines: str) -> list[dict[str, object]]:
    return list(iter_v4_records(lines))


def test_iter_v4_records_meta() -> None:
    meta = {"kind": "meta", "version": 3, "session_id": "abc"}
    out = _records(json.dumps(meta))
    assert len(out) == 1
    assert out[0]["kind"] == "meta"
    assert out[0]["session_id"] == "abc"
    # ``version`` is dropped on the v4 side.
    assert "version" not in out[0]


def test_iter_v4_records_user_text() -> None:
    rec = {
        "kind": "message",
        "descriptor": "text/x-user-message",
        "content": "hello there",
        "_id": 5,
        "_parent_id": 3,
        "_timestamp": 1_700_000_000,
    }
    out = _records(json.dumps(rec))
    assert out == [
        {
            "kind": "history",
            "type": "user",
            "text": "hello there",
            "attachments": [],
            "id": 5,
            "parent_id": 3,
            "timestamp": 1_700_000_000.0,
        }
    ]


def test_iter_v4_records_user_multipart_with_image() -> None:
    image_bytes = b"\x89PNG\x0d\x0a\x1a\x0a"
    rec = {
        "kind": "message",
        "descriptor": "multipart/x-user-message",
        "content": [
            {"descriptor": "text/plain", "content": "look:"},
            {
                "descriptor": "image/png",
                "content": {"_bytes": base64.b64encode(image_bytes).decode()},
            },
        ],
        "_id": 7,
        "_parent_id": -1,
        "_timestamp": 1_700_000_001,
    }
    out = _records(json.dumps(rec))
    converted = out[0]
    assert converted["type"] == "user"
    assert converted["text"] == "look:"
    atts = converted["attachments"]
    assert isinstance(atts, list)
    att0 = cast("dict[str, str]", atts[0])
    assert att0["mime"] == "image/png"
    assert base64.b64decode(att0["data"]) == image_bytes


def test_iter_v4_records_user_multipart_skips_bad_image() -> None:
    rec = {
        "kind": "message",
        "descriptor": "multipart/x-user-message",
        "content": [
            {"descriptor": "text/plain", "content": "hi"},
            {"descriptor": "image/png", "content": "not-bytes-shape"},
            {"descriptor": "image/png", "content": {"_bytes": "%%% invalid %%%"}},
        ],
    }
    out = _records(json.dumps(rec))
    assert out[0]["attachments"] == []


def test_iter_v4_records_assistant_multipart() -> None:
    rec = {
        "kind": "message",
        "descriptor": "multipart/x-model-message",
        "content": [
            {"descriptor": "text/plain", "content": "reply A"},
            {"descriptor": "text/plain", "content": "reply B"},
            {
                "descriptor": "application/x-thinking-structured",
                "content": {"type": "thinking", "text": "internal"},
            },
            {
                "descriptor": "multipart/x-tool-call",
                "content": [
                    {"descriptor": "text/x-queue-id", "content": "q1"},
                    {
                        "descriptor": "application/x-tool-Echo",
                        "content": {"msg": "hi"},
                    },
                ],
            },
        ],
        "_id": 10,
        "_parent_id": 7,
        "_timestamp": 1_700_000_002_000_000_000,  # ns-scale -> downshift.
    }
    out = _records(json.dumps(rec))
    converted = out[0]
    assert converted["type"] == "assistant"
    assert converted["text"] == "reply A\nreply B"
    thinking = converted["thinking_blocks"]
    assert isinstance(thinking, list)
    thinking0 = cast("dict[str, object]", thinking[0])
    assert thinking0["text"] == "internal"
    tcs = converted["tool_calls"]
    assert isinstance(tcs, list)
    tc0 = cast("dict[str, object]", tcs[0])
    assert tc0["id"] == "q1"
    assert tc0["name"] == "Echo"
    assert tc0["args"] == {"msg": "hi"}
    # ns -> seconds downshift.
    assert isinstance(converted["timestamp"], float)
    assert converted["timestamp"] == 1_700_000_002.0


def test_iter_v4_records_tool_call_without_id_or_name_dropped() -> None:
    rec: dict[str, object] = {
        "kind": "message",
        "descriptor": "multipart/x-model-message",
        "content": [
            {"descriptor": "multipart/x-tool-call", "content": []},
        ],
    }
    out = _records(json.dumps(rec))
    assert out[0]["tool_calls"] == []


def test_iter_v4_records_tool_result_multi_text_and_hint() -> None:
    rec = {
        "kind": "message",
        "descriptor": "multipart/x-tool-result",
        "content": [
            {"descriptor": "text/x-queue-id", "content": "q-7"},
            {"descriptor": "text/plain", "content": "ok part 1"},
            {"descriptor": "text/plain", "content": "ok part 2"},
            {"descriptor": "text/x-hint-tool-use-nudge", "content": "use grep"},
            {"descriptor": "application/x-file-stat", "content": {"x": 1}},
        ],
    }
    out = _records(json.dumps(rec))
    converted = out[0]
    assert converted["type"] == "tool_result"
    assert converted["call_id"] == "q-7"
    assert converted["content"] == "ok part 1\nok part 2"
    assert converted["hint"] == "use grep"
    assert converted["is_error"] is False


def test_iter_v4_records_tool_result_error() -> None:
    rec = {
        "kind": "message",
        "descriptor": "multipart/x-tool-result",
        "content": [
            {"descriptor": "text/x-queue-id", "content": "q-1"},
            {"descriptor": "text/x-error", "content": "boom"},
        ],
    }
    out = _records(json.dumps(rec))
    assert out[0]["is_error"] is True
    assert out[0]["content"] == "boom"


def test_iter_v4_records_clear_passthrough() -> None:
    rec = {"kind": "clear", "_timestamp": 100}
    out = _records(json.dumps(rec))
    assert out == [{"kind": "clear", "_timestamp": 100}]


def test_iter_v4_records_unknown_kind_skipped() -> None:
    rec = {"kind": "weird"}
    out = _records(json.dumps(rec))
    assert out == []


def test_iter_v4_records_unknown_descriptor_skipped() -> None:
    rec = {"kind": "message", "descriptor": "weird/x-something", "content": "x"}
    out = _records(json.dumps(rec))
    assert out == []


def test_iter_v4_records_blank_and_malformed_lines() -> None:
    out = _records("", "   ", "not json", json.dumps({"kind": "clear"}))
    assert out == [{"kind": "clear", "_timestamp": 0}]


def test_iter_v4_records_non_dict_record_skipped() -> None:
    out = _records(json.dumps(["a list"]))
    assert out == []


def test_iter_v4_records_legacy_seconds_timestamp_preserved() -> None:
    rec = {
        "kind": "message",
        "descriptor": "text/x-user-message",
        "content": "x",
        "_timestamp": 1_500_000_000,
    }
    out = _records(json.dumps(rec))
    assert out[0]["timestamp"] == 1_500_000_000.0


def test_iter_v4_records_non_numeric_timestamp_zero() -> None:
    rec = {
        "kind": "message",
        "descriptor": "text/x-user-message",
        "content": "x",
        "_timestamp": "not a number",
    }
    out = _records(json.dumps(rec))
    assert out[0]["timestamp"] == 0.0


def test_migrate_file_writes_v4_jsonl(tmp_path: Path) -> None:
    src = tmp_path / "session.jsonl"
    src.write_text(
        json.dumps(
            {"kind": "message", "descriptor": "text/x-user-message", "content": "hi"}
        )
        + "\n"
    )
    dst = tmp_path / "session.v4.jsonl"
    n = migrate_file(src, dst)
    assert n == 1
    out = [json.loads(line) for line in dst.read_text().splitlines() if line]
    assert out[0]["type"] == "user"


def test_main_handles_missing_path(tmp_path: Path, capsys: object) -> None:
    del capsys  # logging output is not asserted on.
    missing = tmp_path / "nope.jsonl"
    rc = main([str(missing)])
    assert rc == 1


def test_main_directory_walk(tmp_path: Path) -> None:
    nested = tmp_path / "sess-1"
    nested.mkdir()
    (nested / "session.jsonl").write_text(
        json.dumps(
            {"kind": "message", "descriptor": "text/x-user-message", "content": "hi"}
        )
        + "\n"
    )
    rc = main([str(tmp_path)])
    assert rc == 0
    assert (nested / "session.v4.jsonl").exists()


def test_main_skip_existing(tmp_path: Path) -> None:
    nested = tmp_path / "sess-1"
    nested.mkdir()
    (nested / "session.jsonl").write_text(
        json.dumps(
            {"kind": "message", "descriptor": "text/x-user-message", "content": "hi"}
        )
        + "\n"
    )
    dst = nested / "session.v4.jsonl"
    dst.write_text("PRE-EXISTING\n")
    rc = main([str(tmp_path)])
    assert rc == 0
    # Without --overwrite the existing file stays intact.
    assert "PRE-EXISTING" in dst.read_text()


def test_main_empty_dir(tmp_path: Path) -> None:
    rc = main([str(tmp_path)])
    assert rc == 0


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
