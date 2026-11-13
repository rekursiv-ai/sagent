"""Tests for reading old sagent session logs as today's records.

The tests after ``_AT`` read one old record at a time; those before it, a whole log.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import base64
import os

from treekle import json_dumps

import pytest

from sagent.lib.userdirs import data_dir
from sagent.model.cost import TokenCost, TokenCount
from sagent.model.types import Response
from sagent.runtime.tape import ContextSplice
from sagent.session import legacy_journal, projects
from sagent.session.jsonl import JsonlJournal
from sagent.session.legacy_journal import (
    convert,
    find_old,
    is_legacy,
    upgrade,
)
from sagent.session.types import (
    AgentSpawnResult,
    AgentToAgentMessage,
    AssistantMessage,
    Attachment,
    BashResult,
    ContextClear,
    EditResult,
    ReadResult,
    SystemMessage,
    Thinking,
    ToolCall,
    ToolResult,
    UncategorizedToolResult,
    UserMessage,
    WebFetchResult,
    WriteResult,
)


if TYPE_CHECKING:
    from pathlib import Path

    from treekle import PlainTree

    from sagent.session.types import SessionRecord


_PNG = base64.b64encode(b"\x89PNG").decode()
_DETACHED = "[detached: tool still running; real result arrives in a later message]"
_CANCELLED = "[cancelled: tool killed before completion; no result will follow]"


def _log(tmp_path: Path, *records: PlainTree | str) -> Path:
    """Write ``records`` as an old ``session.jsonl``; a string is written as is."""
    path = tmp_path / "session.jsonl"
    _ = path.write_text(
        "".join((r if isinstance(r, str) else json_dumps(r)) + "\n" for r in records),
    )
    return path


def _ref(n: int, sid: str = "s") -> dict[str, PlainTree]:
    return {"session_id": sid, "ordinal": n}


def _user(n: int, text: str) -> dict[str, PlainTree]:
    return {"kind": "history", "ref": _ref(n), "type": "user", "text": text}


def _call(n: int, call_id: str, text: str = "") -> dict[str, PlainTree]:
    return {
        "kind": "history",
        "ref": _ref(n),
        "type": "assistant",
        "text": text,
        "tool_calls": [{"id": call_id, "name": "Read", "args": {"file_path": "a"}}],
    }


def _said(n: int, text: str) -> dict[str, PlainTree]:
    return {"kind": "history", "ref": _ref(n), "type": "assistant", "text": text}


def _result(
    n: int | None,
    call_id: str,
    content: str,
    **extra: PlainTree,
) -> dict[str, PlainTree]:
    record: dict[str, PlainTree] = {
        "kind": "history",
        "type": "tool_result",
        "call_id": call_id,
        "content": content,
        **extra,
    }
    if n is not None:
        record["ref"] = _ref(n)
    return record


def _history(path: Path) -> tuple[SessionRecord, ...]:
    clear, *history = convert(path)
    assert isinstance(clear, ContextClear)
    assert clear.history == ()
    return tuple(history)


def _texts(path: Path) -> list[str]:
    return [
        r.content or ""
        if isinstance(
            r,
            UserMessage | AssistantMessage | AgentToAgentMessage | SystemMessage,
        )
        else r.text()
        if isinstance(r, ToolResult)
        else f"call {r.call_id}"
        if isinstance(r, ToolCall)
        else type(r).__name__
        for r in _history(path)
    ]


@pytest.mark.parametrize(
    ("lines", "legacy"),
    [
        ([{"kind": "meta", "session_id": "s"}], True),
        ([{"descriptor": "multipart/x-user-turn", "content": "x"}], True),
        ([{"py/object": "ContextClear", "kind": "x"}], False),
        ([{"session_id": "s", "model_id": "m"}], True),
        (["not json", {"kind": "meta"}], True),
        ([[1, 2]], False),
        ([], False),
    ],
    ids=[
        "tape",
        "descriptor",
        "today",
        "kindless-meta",
        "skips-junk",
        "array",
        "empty",
    ],
)
def test_an_old_log_is_told_from_todays(
    tmp_path: Path,
    lines: list[PlainTree | str],
    legacy: bool,
) -> None:
    assert is_legacy(_log(tmp_path, *lines)) is legacy


def test_a_journal_not_yet_written_is_not_old(tmp_path: Path) -> None:
    """``--session`` may name a new file: there is nothing old to upgrade."""
    assert not is_legacy(tmp_path / "session.jsonl")


def test_a_tape_converts_to_what_its_model_last_saw(tmp_path: Path) -> None:
    """A detached call's stub is replaced by its real result, as the old loader showed it."""
    path = _log(
        tmp_path,
        {
            "kind": "meta",
            "session_id": "s",
            "bash_cwd": "/work",
            "spend": {"request": 0.25, "response": 0.5},
        },
        {"kind": "tool_state", "bash_cwd": "/elsewhere"},
        _user(0, "go"),
        _call(1, "c1", text="reading"),
        _result(2, "c1", _DETACHED, result_kind="pending"),
        {
            "kind": "context_splice",
            "ref": _ref(3),
            "mask": [[_ref(2), _ref(2)]],
            "insert_after": _ref(1),
            "payload": [_result(None, "c1", "the file")],
            "strategy": "detached_splice",
            "paired_externally": ["c1"],
        },
        _said(4, "done"),
        {"kind": "runtime_event", "type": "notice_message", "text": "busy"},
    )

    clear, *history = convert(path)

    assert isinstance(clear, ContextClear)
    assert clear.extra == {"spend_usd": 0.75, "converted_from": str(path)}
    assert _texts(path) == ["go", "reading", "call c1", "the file", "done"]
    (result,) = [r for r in history if isinstance(r, ToolResult)]
    assert result.state == "done"


def test_a_meta_record_without_its_kind_is_still_read(tmp_path: Path) -> None:
    path = _log(
        tmp_path,
        {"session_id": "s7", "model_id": "m", "total_cost_usd": 0.5},
        _user(0, "a"),
    )
    clear, *_ = convert(path)
    assert isinstance(clear, ContextClear)
    assert clear.extra["spend_usd"] == 0.5


def test_a_summary_hides_what_it_replaced(tmp_path: Path) -> None:
    path = _log(
        tmp_path,
        _user(0, "old"),
        _said(1, "old reply"),
        {
            "kind": "context_splice",
            "ref": _ref(2),
            "mask": [[_ref(0), _ref(1)]],
            "insert_after": None,
            "payload": [{"type": "user", "text": "summary"}],
            "strategy": "summary",
        },
        {"kind": "history", "ref": _ref(3), "type": "compact_complete"},
        _said(4, "after"),
    )
    assert _texts(path) == ["summary", "after"]


def _cut(
    n: int,
    hide: tuple[int, int] | None,
    *said: str,
) -> dict[str, PlainTree]:
    """Return a splice at ``n`` hiding ``hide`` inclusive and showing ``said``."""
    return {
        "kind": "context_splice",
        "ref": _ref(n),
        "mask": [[_ref(hide[0]), _ref(hide[1])]] if hide is not None else [],
        "insert_after": None,
        "payload": [{"type": "user", "text": text} for text in said],
        "strategy": "barrier",
    }


# Two writers once took one position. Each case is the old loader's test of the same
# name, in ``agent/session_io_test.py``.
@pytest.mark.parametrize(
    ("records", "shown", "hidden"),
    [
        (
            (_user(0, "a"), _cut(1, (0, 99), "barrier"), _user(0, "later-real")),
            ("later-real",),
            (),
        ),
        (
            (_user(0, "a"), _user(0, "a-dup"), _cut(1, (0, 10))),
            (),
            ("a-dup",),
        ),
        (
            (
                _user(0, "real"),
                _cut(1, (0, 0), "poison"),
                _cut(2, (1, 1)),
                _cut(1, (0, 0), "poison-dup"),
            ),
            ("real",),
            ("poison",),
        ),
        (
            (_user(0, "shared"), _user(1, "said-to-A"), _cut(1, None, "said-to-B")),
            ("said-to-A", "said-to-B"),
            (),
        ),
        (
            (_user(0, "a"), _user(0, "a-dup"), _cut(2, (0, 1), "barrier")),
            ("barrier",),
            ("a-dup",),
        ),
        (
            (_user(0, "a"), _cut(1, (0, 0), "barrier"), _user(0, "later-real")),
            ("barrier", "later-real"),
            (),
        ),
        (
            (
                _user(0, "a"),
                _user(0, "dup"),
                _user(1, "b"),
                _cut(1, (0, 0), "summary"),
            ),
            ("summary",),
            ("dup",),
        ),
        (
            (_user(0, "first"), _user(0, "second")),
            ("first", "second"),
            (),
        ),
    ],
    ids=[
        "relocated-record-does-not-land-inside-an-existing-mask",
        "carried-mask-does-not-overlap-the-range-it-extends",
        "duplicated-dead-splice-is-not-resurrected",
        "second-writers-splice-is-relocated-not-dropped",
        "duplicate-written-before-a-barrier-stays-masked",
        "duplicate-written-after-a-barrier-survives-it",
        "relocated-splice-still-masks-a-relocated-record-it-followed",
        "keeps-both-records-when-a-ref-collides",
    ],
)
def test_a_position_two_writers_took_keeps_each_record_s_fate(
    tmp_path: Path,
    records: tuple[dict[str, PlainTree], ...],
    shown: tuple[str, ...],
    hidden: tuple[str, ...],
) -> None:
    said = "\n".join(_texts(_log(tmp_path, *records)))
    assert [text for text in shown if text not in said] == []
    assert [text for text in hidden if text in said] == []


@pytest.mark.parametrize(
    ("override", "texts"),
    [
        ({"suppresses": [_ref(1)], "inject_after": _ref(0)}, ["a\n\nnew", "c"]),
        ({"barrier": True, "inject_after": _ref(1)}, ["new", "c"]),
        ({"barrier": True}, ["new", "c"]),
        ({"inject_after": _ref(1)}, ["a", "b", "new", "c"]),
    ],
    ids=["suppresses", "barrier-to-anchor", "barrier-before-itself", "insert-only"],
)
def test_an_override_hides_as_it_said(
    tmp_path: Path,
    override: dict[str, PlainTree],
    texts: list[str],
) -> None:
    path = _log(
        tmp_path,
        _user(0, "a"),
        _said(1, "b"),
        {
            "kind": "context_override",
            "ref": _ref(2),
            "payload": [{"type": "user", "text": "new"}],
            **override,
        },
        _said(3, "c"),
    )
    assert _texts(path) == texts


@pytest.mark.parametrize("kind", ["clear", "context_clear"])
def test_a_clear_hides_everything_before_it(tmp_path: Path, kind: str) -> None:
    path = _log(tmp_path, _user(0, "a"), _said(1, "b"), {"kind": kind}, _user(3, "c"))
    assert _texts(path) == ["c"]


def test_an_update_rewrites_the_result_it_names(tmp_path: Path) -> None:
    path = _log(
        tmp_path,
        _call(0, "c1"),
        _result(1, "c1", _DETACHED, id=7, timestamp=1_790_000_000),
        {"kind": "update", "id": 7, "content": "real", "is_error": True},
        {"kind": "update", "id": 99, "content": "nowhere"},
    )
    (result,) = [r for r in _history(path) if isinstance(r, ToolResult)]
    assert (result.text(), result.is_error, result.state) == ("real", True, "done")
    assert result.timestamp == "2026-09-21T14:13:20+00:00"


def test_records_without_refs_take_the_next_ordinal(tmp_path: Path) -> None:
    path = _log(
        tmp_path,
        {"kind": "meta", "session_id": "s"},
        {"kind": "history", "type": "user", "text": "a"},
        _said(5, "b"),
        {"kind": "history", "type": "user", "text": "c"},
    )
    assert _texts(path) == ["a", "b", "c"]


def test_a_taken_ref_moves_on_and_a_masked_splice_again_is_dropped(
    tmp_path: Path,
) -> None:
    hide = {
        "kind": "context_splice",
        "ref": _ref(2),
        "mask": [[_ref(1), _ref(1)]],
        "insert_after": _ref(0),
        "payload": [{"type": "assistant", "text": "edited"}],
    }
    path = _log(
        tmp_path,
        _user(0, "a"),
        _said(1, "b"),
        hide,
        {
            "kind": "context_splice",
            "ref": _ref(3),
            "mask": [[_ref(1), _ref(2)]],
            "insert_after": _ref(0),
            "payload": [{"type": "assistant", "text": "edited again"}],
        },
        hide,
        _user(3, "second writer"),
    )
    assert _texts(path) == ["a", "edited again", "second writer"]


def test_what_cannot_be_read_is_skipped(tmp_path: Path) -> None:
    path = _log(
        tmp_path,
        "{not json",
        "[1, 2]",
        "",
        {
            "kind": "history",
            "ref": {"session_id": "s", "ordinal": True},
            "type": "user",
            "text": "a",
        },
        {
            "kind": "history",
            "ref": {"session_id": "s", "ordinal": -1},
            "type": "nonsense",
        },
        {
            "kind": "context_splice",
            "ref": _ref(5),
            "mask": [
                [_ref(0), _ref(1, sid="other")],
                [_ref(1), _ref(0)],
                [_ref(0)],
                "x",
            ],
            "payload": "x",
        },
        {
            "kind": "history",
            "ref": _ref(6),
            "type": "assistant",
            "text": "t",
            "tool_calls": [{"id": "", "name": "Read"}, {"id": "c", "name": ""}, "x"],
            "thinking_blocks": [{"type": "unknown"}, "x"],
        },
        {
            "kind": "history",
            "ref": _ref(7),
            "type": "user",
            "text": "pic",
            "attachments": [
                {"mime": "image/png", "data": "!!"},
                {"mime": "", "data": _PNG},
                {"mime": "image/png", "data": _PNG},
            ],
        },
    )
    history = _history(path)
    assert _texts(path) == ["a", "t", "pic"]
    pic = history[-1]
    assert isinstance(pic, UserMessage)
    assert pic.attachments == (
        Attachment(mime_descriptor="image/png", data=b"\x89PNG"),
    )


def test_a_payload_not_valid_alone_is_repaired(tmp_path: Path) -> None:
    path = _log(
        tmp_path,
        _user(0, "a"),
        {
            "kind": "context_splice",
            "ref": _ref(1),
            "mask": [[_ref(0), _ref(0)]],
            "payload": [
                {"type": "user", "text": "one"},
                {"type": "user", "text": "two"},
                {
                    "type": "assistant",
                    "text": "",
                    "tool_calls": [{"id": "c9", "name": "Bash"}],
                },
            ],
        },
    )
    assert _texts(path) == ["one\n\ntwo", "call c9", "[interrupted]"]


def test_a_repaired_payload_keeps_where_it_goes_and_what_answers_it_later(
    tmp_path: Path,
) -> None:
    placed = _log(
        tmp_path,
        _user(0, "a"),
        _said(1, "b"),
        {
            "kind": "context_splice",
            "ref": _ref(2),
            "mask": [[_ref(0), _ref(0)]],
            "insert_after": _ref(1),
            "payload": [
                {"type": "assistant", "tool_calls": [{"id": "c8", "name": "Bash"}]},
            ],
        },
        _user(3, "c"),
    )
    assert _texts(placed) == ["b", "call c8", "[interrupted]", "c"]
    answered = _log(
        tmp_path,
        _user(0, "a"),
        {
            "kind": "context_splice",
            "ref": _ref(1),
            "mask": [[_ref(0), _ref(0)]],
            "payload": [
                {"type": "user", "text": "one"},
                {"type": "assistant", "text": "x"},
                {"type": "assistant", "tool_calls": [{"id": "c9", "name": "Bash"}]},
            ],
            "paired_externally": ["c9"],
        },
        _result(2, "c9", "done"),
    )
    assert _texts(answered)[-2:] == ["call c9", "done"]


def test_a_payload_entry_no_model_saw_is_left_out(tmp_path: Path) -> None:
    path = _log(
        tmp_path,
        _user(0, "a"),
        {
            "kind": "context_splice",
            "ref": _ref(1),
            "mask": [[_ref(0), _ref(0)]],
            "payload": [{"type": "compact_started"}, {"type": "user", "text": "kept"}],
        },
    )
    assert _texts(path) == ["kept"]


def test_a_spawn_result_names_the_child_its_label_started(tmp_path: Path) -> None:
    path = _log(
        tmp_path,
        {
            "kind": "history",
            "ref": _ref(0),
            "type": "assistant",
            "tool_calls": [
                {"id": "sp", "name": "AgentSpawn", "args": {"label": "rev"}},
            ],
        },
        {"kind": "persistent_agent", "label": "rev", "session_dir": "/p/s/3640f519"},
        _result(1, "sp", "started"),
    )
    (result,) = [r for r in _history(path) if isinstance(r, ToolResult)]
    assert result.result == AgentSpawnResult(agent_id="3640f519", agent_kind="rev")


def test_an_unanswered_call_is_answered_interrupted(tmp_path: Path) -> None:
    path = _log(tmp_path, _user(0, "go"), _call(1, "c1"))
    assert _texts(path) == ["go", "call c1", "[interrupted]"]


@pytest.mark.parametrize(
    ("meta", "spent"),
    [
        ({"own_spend": {"request": 1.0}, "spend": {"request": 9.0}}, 9.0),
        ({"own_spend": {"request": 1.0}}, 1.0),
        ({"spend": {"request": 1.0, "response": 2.0, "note": "x", "flag": True}}, 3.0),
        ({"total_cost_usd": 4.5}, 4.5),
        ({"total_cost_usd": True}, 0.0),
        ({}, 0.0),
    ],
)
def test_the_spend_is_what_the_agent_and_its_children_spent(
    tmp_path: Path,
    meta: dict[str, PlainTree],
    spent: float,
) -> None:
    clear, *_ = convert(_log(tmp_path, {"kind": "meta", **meta}))
    assert isinstance(clear, ContextClear)
    assert clear.extra["spend_usd"] == spent


def test_a_v3_log_converts_through_the_tape(tmp_path: Path) -> None:
    call = {
        "descriptor": "multipart/x-tool-call",
        "content": [
            {"descriptor": "text/x-queue-id", "content": "c1"},
            {"descriptor": "application/x-tool-read", "content": {"file_path": "a"}},
        ],
    }
    path = _log(
        tmp_path,
        {"kind": "meta", "session_id": "v3", "bash_cwd": "/w"},
        {
            "kind": "message",
            "descriptor": "text/x-user-message",
            "content": "old",
            "_id": 1,
        },
        {"kind": "clear"},
        {
            "kind": "message",
            "descriptor": "text/x-user-message",
            "content": "hi",
            "_timestamp": 1_790_000_000_000_000_000,
        },
        {
            "kind": "message",
            "descriptor": "multipart/x-user-message",
            "content": [
                {"descriptor": "text/plain", "content": "look"},
                {"descriptor": "image/png", "content": {"_bytes": _PNG}},
            ],
        },
        {
            "kind": "message",
            "descriptor": "multipart/x-model-message",
            "content": [
                {"descriptor": "text/plain", "content": "reading"},
                {
                    "descriptor": "application/x-thinking-anthropic",
                    "content": {"type": "thinking", "thinking": "t", "signature": "s"},
                },
                call,
            ],
        },
        {
            "kind": "message",
            "descriptor": "multipart/x-tool-result",
            "content": [
                {"descriptor": "text/x-queue-id", "content": "c1"},
                {"descriptor": "text/plain", "content": "line"},
                {"descriptor": "text/x-error", "content": "failed"},
                {"descriptor": "application/x-file-stat", "content": {}},
            ],
        },
        {"kind": "message", "descriptor": "application/x-unknown", "content": "x"},
    )

    assert _texts(path) == [
        "hi\n\nlook",
        "Thinking",
        "reading",
        "call c1",
        "line\nfailed",
    ]
    history = _history(path)
    user = history[0]
    assert isinstance(user, UserMessage)
    assert user.attachments == (
        Attachment(mime_descriptor="image/png", data=b"\x89PNG"),
    )
    (called,) = [r for r in history if isinstance(r, ToolCall)]
    assert (called.name, called.arguments) == ("read", {"file_path": "a"})
    (result,) = [r for r in history if isinstance(r, ToolResult)]
    assert result.is_error is True


def test_a_line_of_no_record_is_logged_with_its_file(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    path = _log(tmp_path, "{nope", _user(0, "u"))
    assert _texts(path) == ["u"]
    assert caplog.messages == [f"skipping a line of {path} that holds no record"]


@pytest.mark.parametrize(("held", "named"), [("x", "x"), (1, True), (None, None)])
def test_an_update_names_its_result_by_an_integer_id_alone(
    tmp_path: Path,
    held: PlainTree,
    named: PlainTree,
) -> None:
    path = _log(
        tmp_path,
        _call(0, "c"),
        _result(1, "c", "old", id=held),
        {"kind": "update", "id": named, "content": "new"},
    )
    assert _texts(path)[-1] == "old"


def test_an_update_to_a_stand_in_marks_what_it_stood_for(tmp_path: Path) -> None:
    path = _log(
        tmp_path,
        _call(0, "c"),
        _result(1, "c", "old", id=4),
        {"kind": "update", "id": 4, "content": _DETACHED},
    )
    result = _history(path)[-1]
    assert isinstance(result, ToolResult)
    assert result.state == "pending"


def test_a_ref_less_record_takes_the_sessions_id_from_ordinal_zero(
    tmp_path: Path,
) -> None:
    hide = {
        "kind": "context_splice",
        "ref": _ref(9, "m"),
        "mask": [[_ref(1, "m"), _ref(1, "m")]],
    }
    path = _log(
        tmp_path,
        {"kind": "meta", "session_id": "m"},
        {"kind": "history", "type": "user", "text": "zero"},
        {"kind": "history", "type": "assistant", "text": "one"},
        {"kind": "history", "type": "assistant", "text": "two"},
        hide,
    )
    assert _texts(path) == ["zero", "two"]


def test_a_mask_in_another_session_hides_nothing_here(tmp_path: Path) -> None:
    path = _log(
        tmp_path,
        _user(0, "kept"),
        {
            "kind": "context_splice",
            "ref": _ref(1),
            "mask": [[_ref(0, "t"), _ref(0, "t")]],
        },
    )
    assert _texts(path) == ["kept"]


@pytest.mark.parametrize(
    "bad",
    [
        [_ref(0), {"ordinal": 0}],
        [{"ordinal": 0}, _ref(0)],
        [_ref(0)],
    ],
)
def test_a_malformed_range_is_skipped_and_those_after_it_kept(
    tmp_path: Path,
    bad: PlainTree,
) -> None:
    path = _log(
        tmp_path,
        _user(0, "a"),
        _user(1, "b"),
        {"kind": "context_splice", "ref": _ref(2), "mask": [bad, [_ref(1), _ref(1)]]},
    )
    assert _texts(path) == ["a"]


def test_a_barrier_hides_up_to_its_anchor_or_before_itself(tmp_path: Path) -> None:
    anchored = _log(
        tmp_path,
        _user(0, "a"),
        _said(1, "b"),
        _user(2, "c"),
        {
            "kind": "context_override",
            "ref": _ref(3),
            "barrier": True,
            "inject_after": _ref(0),
        },
    )
    assert _texts(anchored) == ["b", "c"]
    first = _log(
        tmp_path,
        {"kind": "context_override", "ref": _ref(0), "barrier": True},
        _user(1, "b"),
    )
    assert _texts(first) == ["b"]


def test_a_kindless_record_without_a_session_id_is_no_meta(tmp_path: Path) -> None:
    path = _log(
        tmp_path,
        {"kind": "meta", "session_id": "m", "total_cost_usd": 2.5},
        {"note": "x"},
    )
    clear, *_ = convert(path)
    assert isinstance(clear, ContextClear)
    assert clear.extra["spend_usd"] == 2.5
    assert clear.timestamp is not None
    assert clear.timestamp.endswith("+00:00")


def test_a_v3_log_keeps_times_ids_lines_and_result_attachments(tmp_path: Path) -> None:
    def part(descriptor: str, content: PlainTree) -> dict[str, PlainTree]:
        return {"descriptor": descriptor, "content": content}

    nameless = part("multipart/x-tool-call", [part("text/x-queue-id", "c0")])
    path = _log(
        tmp_path,
        {
            "kind": "message",
            "descriptor": "multipart/x-user-message",
            "content": [part("text/plain", "a"), part("text/plain", "b")],
            "_timestamp": 1_790_000_000,
        },
        {
            "kind": "message",
            "descriptor": "multipart/x-model-message",
            "content": [
                part("text/plain", "go"),
                nameless,
                part(
                    "multipart/x-tool-call",
                    [part("text/x-queue-id", "c1"), part("application/x-tool-ls", {})],
                ),
            ],
        },
        {
            "kind": "message",
            "descriptor": "multipart/x-tool-result",
            "_id": 7,
            "content": [
                part("text/x-queue-id", "c1"),
                part("text/plain", "old"),
                part("image/png", {"_bytes": _PNG}),
            ],
        },
        {"kind": "update", "id": 7, "content": "new"},
    )
    user, said, call, result = _history(path)
    assert user == UserMessage(timestamp="2026-09-21T14:13:20+00:00", content="a\nb")
    assert said == AssistantMessage(content="go")
    assert isinstance(call, ToolCall)
    assert call.name == "ls"
    assert isinstance(result, ToolResult)
    assert (result.text(), result.is_error) == ("new", False)
    assert result.attachments == (
        Attachment(mime_descriptor="image/png", data=b"\x89PNG"),
    )


def test_a_tape_s_splices_keep_what_made_them() -> None:
    tape = legacy_journal._tape(
        [
            _user(0, "a"),
            {"kind": "clear", "ref": _ref(1)},
            {"kind": "context_splice", "ref": _ref(2), "strategy": "summary"},
        ],
        "s",
    )
    assert [r.strategy for r in tape if isinstance(r, ContextSplice)] == [
        "clear",
        "summary",
    ]


def test_a_repaired_splice_keeps_what_made_it() -> None:
    tape = legacy_journal._tape(
        [
            _user(0, "a"),
            {
                "kind": "context_splice",
                "ref": _ref(1),
                "mask": [[_ref(0), _ref(0)]],
                "strategy": "summary",
                "payload": [
                    {
                        "type": "assistant",
                        "text": "",
                        "tool_calls": [{"id": "c9", "name": "Bash"}],
                    },
                ],
            },
        ],
        "s",
    )
    assert [r.strategy for r in tape if isinstance(r, ContextSplice)] == ["summary"]


def test_an_update_names_a_tool_result_alone(tmp_path: Path) -> None:
    path = _log(
        tmp_path,
        _user(0, "go"),
        {**_said(1, "said"), "id": 3},
        {"kind": "update", "id": 3, "content": "new"},
    )
    assert _texts(path) == ["go", "said"]


def test_records_sharing_a_ref_move_just_past_everything_else() -> None:
    tape = legacy_journal._tape(
        [
            _user(3, "a"),
            {"kind": "context_splice", "ref": _ref(1), "mask": [[_ref(5), _ref(7)]]},
            _said(3, "b"),
            _said(3, "c"),
        ],
        "s",
    )
    assert [r.ref.ordinal for r in tape] == [1, 3, 8, 9]
    assert legacy_journal._tape([], "s") == []


def test_an_old_turn_costs_nothing_and_stopped_for_no_reason() -> None:
    turn = legacy_journal._Entries().read({"type": "assistant", "text": "hi"})
    assert isinstance(turn, Response)
    assert (turn.records, turn.tokens, turn.spend, turn.stop_reason) == (
        (AssistantMessage(content="hi"),),
        TokenCount(),
        TokenCost(),
        "",
    )


def test_an_upgrade_a_crash_cut_short_finishes(tmp_path: Path) -> None:
    log = _session(tmp_path / "abc", _user(0, "hello"))
    before = log.read_bytes()
    os.link(log, projects.legacy_log(log.parent))

    assert upgrade(log)

    assert projects.legacy_log(log.parent).read_bytes() == before
    assert not is_legacy(log)


def test_an_upgrade_never_overwrites_another_log_kept(tmp_path: Path) -> None:
    log = _session(tmp_path / "abc", _user(0, "hello"))
    projects.legacy_log(log.parent).write_text("other")
    with pytest.raises(FileExistsError):
        upgrade(log)
    assert is_legacy(log)


def _session(directory: Path, *records: PlainTree) -> Path:
    """Write ``records`` as the old session ``directory``."""
    directory.mkdir(parents=True)
    return _log(directory, {"kind": "meta", "session_id": directory.name}, *records)


async def _window(path: Path) -> list[SessionRecord]:
    async with JsonlJournal(path) as journal:
        return list(journal)


@pytest.mark.asyncio
async def test_an_old_session_is_converted_in_place(tmp_path: Path) -> None:
    log = _session(tmp_path / "abc", _user(0, "hello"))

    assert upgrade(log)

    assert not is_legacy(log)
    async with JsonlJournal(log) as journal:
        clear, said = list(journal)
        journal.append(UserMessage(content="again"))
    assert isinstance(clear, ContextClear)
    assert said == UserMessage(content="hello")
    assert await _window(log) == [clear, said, UserMessage(content="again")]
    assert log.stat().st_mode & 0o777 == 0o600
    assert sorted(p.name for p in log.parent.iterdir()) == [
        "session.jsonl",
        "session.legacy.jsonl",
    ]


def test_converting_an_old_session_keeps_its_log_whole_beside_it(
    tmp_path: Path,
) -> None:
    """A conversion keeps one window; every window the log held stays readable."""
    log = _session(
        tmp_path / "abc",
        _user(0, "FIRST WINDOW"),
        {"kind": "clear", "ref": _ref(1)},
        _user(2, "second window"),
    )
    before = log.read_bytes()

    assert upgrade(log)

    assert "FIRST WINDOW" not in log.read_text()
    assert projects.legacy_log(log.parent).read_bytes() == before


def test_an_old_session_with_nothing_said_is_left_as_it_is(tmp_path: Path) -> None:
    log = _session(tmp_path / "abc")
    before = log.read_bytes()

    assert not upgrade(log)

    assert log.read_bytes() == before


def test_old_sessions_are_found_by_id_prefix_until_converted(tmp_path: Path) -> None:
    found = _session(tmp_path / "_p" / "ab1200000000")
    _session(tmp_path / "_p" / "ab1200000000" / "ab99-0000-0000")
    converted = _session(tmp_path / "_p" / "ab3400000000", _user(0, "hi"))
    assert upgrade(converted)
    loose = _session(tmp_path / "ab5600000000")

    assert find_old("ab", root=tmp_path) == [found, loose]
    assert find_old("ab1", root=tmp_path) == [found]
    assert find_old("*", root=tmp_path) == []


def test_old_sessions_in_a_project_holding_a_log_of_its_own_are_found(
    tmp_path: Path,
) -> None:
    """An old sagent once wrote a session into the project directory itself."""
    project = tmp_path / "_home_me"
    _session(project)
    log = _session(project / "00b3879bbdea")
    assert find_old("00b3", root=tmp_path) == [log]


def test_old_sessions_are_found_under_the_per_user_root() -> None:
    root = data_dir() / "rekursiv-ai" / "sagent" / "projects"
    log = _session(root / "_w" / "feed00000000")
    assert find_old("fe") == [log]


_AT = "2026-09-21T14:13:20+00:00"


def _read(*records: dict[str, PlainTree]) -> list[SessionRecord]:
    """Read ``records`` in order as one session; a message as its history record."""
    entries = legacy_journal._Entries()
    out: list[SessionRecord] = []
    for record in records:
        entry = (
            record
            if record.get("kind") == "history"
            else legacy_journal._history(record)
        )
        assert entry is not None
        event = entries.read(entry)
        if isinstance(event, Response):
            out.extend(event.records)
        elif event is not None:
            out.append(event)
    return out


def _entry(kind: str, **fields: PlainTree) -> dict[str, PlainTree]:
    return {"kind": "history", "type": kind, **fields}


def _turn(*calls: dict[str, PlainTree]) -> dict[str, PlainTree]:
    return _entry("assistant", tool_calls=list(calls))


def _answer(call_id: str, content: str, **fields: PlainTree) -> dict[str, PlainTree]:
    return _entry("tool_result", call_id=call_id, content=content, **fields)


def _results(records: list[SessionRecord]) -> list[ToolResult]:
    return [r for r in records if isinstance(r, ToolResult)]


def _part(descriptor: str, content: PlainTree = None) -> dict[str, PlainTree]:
    return {"descriptor": descriptor, "content": content}


def _v3(
    descriptor: str,
    content: PlainTree,
    **fields: PlainTree,
) -> dict[str, PlainTree]:
    return {"kind": "message", "descriptor": descriptor, "content": content, **fields}


@pytest.mark.parametrize(
    "record",
    [
        {"kind": "message"},
        {"kind": "message", "descriptor": "application/x-unknown", "content": "x"},
        {"kind": "message", "role": "system", "content": "x"},
    ],
    ids=["untagged", "unknown-descriptor", "unknown-role"],
)
def test_a_message_no_model_saw_is_no_entry(record: dict[str, PlainTree]) -> None:
    assert legacy_journal._history(record) is None


def test_an_entry_no_model_saw_reads_as_nothing() -> None:
    assert legacy_journal._Entries().read(_entry("compact_complete")) is None


def test_every_kind_keeps_its_time_and_attachments() -> None:
    seen: dict[str, PlainTree] = {
        "timestamp": 1_790_000_000,
        "attachments": [{"mime": "image/png", "data": _PNG}],
    }
    png = (Attachment(mime_descriptor="image/png", data=b"\x89PNG"),)

    user, note, peer, said, call, result = _read(
        _entry("user", text="u", **seen),
        _entry("system_message", text="n", **seen),
        _entry("agent_send", text="p", source="Agent", **seen),
        _entry(
            "assistant",
            text="said",
            timestamp=1_790_000_000,
            tool_calls=[{"id": "c1", "name": "Grep", "args": {}}],
        ),
        _answer("c1", "out", **seen),
    )

    assert user == UserMessage(timestamp=_AT, content="u", attachments=png)
    assert note == SystemMessage(timestamp=_AT, content="n", subtype="reminder")
    assert peer == AgentToAgentMessage(
        timestamp=_AT,
        content="p",
        attachments=png,
        sender="Agent",
    )
    assert said == AssistantMessage(timestamp=_AT, content="said")
    assert call == ToolCall(timestamp=_AT, call_id="c1", name="Grep")
    assert result == ToolResult(
        timestamp=_AT,
        call_id="c1",
        state="done",
        stdout=("out", *png),
        result=UncategorizedToolResult(),
        extra={"is_error": False},
    )


def test_a_peer_with_no_source_names_no_sender() -> None:
    (peer,) = _read(_entry("agent_send", text="p"))
    assert peer == AgentToAgentMessage(content="p")


def test_an_attachment_that_is_no_strict_base64_is_dropped(
    caplog: pytest.LogCaptureFixture,
) -> None:
    (user,) = _read(
        _entry(
            "user",
            attachments=[
                {"mime": "image/png", "data": "iVBO!!Rw=="},
                {"mime": "", "data": _PNG},
                "x",
            ],
        ),
    )
    assert user == UserMessage(content="")
    assert caplog.messages == ["dropping an attachment that does not decode"]


def test_a_value_of_the_wrong_type_reads_as_none_given() -> None:
    user, said, _, result = _read(
        _entry("user", text=5),
        _entry("assistant", text="x", tool_calls=[{"id": "c", "name": "Grep"}]),
        _answer("c", "out", is_error="yes"),
    )
    assert user == UserMessage(content="")
    assert said == AssistantMessage(content="x")
    assert isinstance(result, ToolResult)
    assert result.is_error is False


def test_a_turn_splits_into_thinking_text_and_calls_each_sealed() -> None:
    records = _read(
        _entry(
            "assistant",
            text="a",
            thought_signature="g1",
            thinking_blocks=[
                {"type": "thinking", "thinking": "hmm", "signature": "s1"},
                {"thinking": "old"},
                {"type": "redacted_thinking", "data": "r1"},
                {
                    "type": "reasoning",
                    "id": "rs_1",
                    "encrypted_content": "e1",
                    "summary": [{"text": "sum"}, {"text": "med"}],
                    "content": [{"text": "why"}, {"text": " not"}],
                },
                {"type": "reasoning", "text": "plain"},
                {"type": "unknown"},
                "x",
            ],
            tool_calls=[
                {"id": "c1", "name": "Read", "args": {}, "thought_signature": "g2"},
                {"id": "", "name": "Read"},
                {"id": "c2", "name": ""},
                "x",
            ],
        ),
    )
    assert records == [
        Thinking(content="hmm", encrypted="s1"),
        Thinking(content="old"),
        Thinking(encrypted="r1"),
        Thinking(
            content="why not",
            summary="summed",
            encrypted="e1",
            extra={"reasoning_id": "rs_1"},
        ),
        Thinking(content="plain"),
        AssistantMessage(content="a", extra={"thoughtSignature": "g1"}),
        ToolCall(call_id="c1", name="Read", extra={"thoughtSignature": "g2"}),
    ]


def test_an_empty_turn_says_nothing() -> None:
    turn = legacy_journal._Entries().read(_entry("assistant", text=""))
    assert isinstance(turn, Response)
    assert turn.records == ()


@pytest.mark.parametrize(
    ("fields", "content", "stub"),
    [
        ({"result_kind": "pending"}, "x", "pending"),
        ({"result_kind": "cancelled"}, "x", "cancelled"),
        ({"result_kind": "final"}, _DETACHED, None),
        ({}, _DETACHED, "pending"),
        ({}, "[Running in background: Bash as job c1]", "pending"),
        ({}, _CANCELLED, "cancelled"),
        ({}, "plain", None),
    ],
)
def test_a_stand_in_result_keeps_what_it_stood_for(
    fields: dict[str, PlainTree],
    content: str,
    stub: str | None,
) -> None:
    (result,) = _results(_read(_answer("c1", content, **fields)))
    assert (result.state, result.result is None) == (stub or "done", stub is not None)


def test_a_result_is_typed_by_the_call_it_answers() -> None:
    """A result names only its call; the call says what the result was."""
    records = _read(
        _turn(
            {"id": "b", "name": "Bash", "args": {"command": "ls"}},
            {"id": "b0", "name": "Bash"},
            {"id": "r", "name": "Read", "args": {"file_path": "a.py"}},
            {"id": "w", "name": "Write", "args": {"path": "b.py", "content": "z"}},
            {"id": "e", "name": "Edit", "args": {"file_path": "a.py"}},
            {"id": "f", "name": "WebFetch", "args": {"url": "https://x.org"}},
            {"id": "g", "name": "Grep"},
        ),
        *(_answer(i, "out") for i in ("b", "b0", "r", "w", "e", "f", "g", "nope")),
    )
    assert [r.result for r in _results(records)] == [
        BashResult(command=("ls",)),
        BashResult(),
        ReadResult(path="a.py"),
        WriteResult(path="b.py", content="z"),
        EditResult(path="a.py"),
        WebFetchResult(url="https://x.org"),
        UncategorizedToolResult(),
        UncategorizedToolResult(),
    ]
    assert {r.text() for r in _results(records)} == {"out"}


def test_a_failed_call_did_not_do_what_it_names() -> None:
    """A refused edit changed nothing; typing it as an edit hid the refusal."""
    records = _read(
        _turn({"id": "e", "name": "Edit", "args": {"file_path": "a.py"}}),
        _answer("e", "modified since read", is_error=True),
    )
    (result,) = _results(records)
    assert result.result == UncategorizedToolResult(exit_code=1)
    assert (result.is_error, result.extra) == (True, {"is_error": True})


def test_an_entry_s_spawn_result_names_its_child() -> None:
    entries = legacy_journal._Entries(children={"rev": "3640f519"})
    spawn = {"label": "rev", "prompt": "review x", "model_id": "luna"}
    _ = entries.read(
        _turn(
            {"id": "s1", "name": "AgentSpawn", "args": spawn},
            {"id": "s2", "name": "AgentSpawn", "args": {"prompt": "count"}},
        ),
    )

    results = [entries.read(_answer(i, "ok")) for i in ("s1", "s2")]

    assert [r.result for r in results if isinstance(r, ToolResult)] == [
        AgentSpawnResult(
            agent_id="3640f519",
            agent_kind="rev",
            prompt="review x",
            model="luna",
        ),
        AgentSpawnResult(prompt="count"),
    ]


def test_an_update_keeps_what_it_rewrote() -> None:
    png = (Attachment(mime_descriptor="image/png", data=b"\x89PNG"),)
    assert legacy_journal._Entries().result(
        "c",
        "new",
        failed=False,
        timestamp=_AT,
        attachments=png,
    ) == ToolResult(
        timestamp=_AT,
        call_id="c",
        state="done",
        stdout=("new", *png),
        result=UncategorizedToolResult(),
        extra={"is_error": False},
    )


@pytest.mark.parametrize(
    ("raw", "when"),
    [
        (1_790_000_000.0, _AT),
        (1_790_000_000_000_000_000, _AT),
        (130811.8, None),
        (True, None),
        ("x", None),
        (1_000_000_000, None),
        (1_000_000_000.5, "2001-09-09T01:46:40.500000+00:00"),
        (1_000_000_000_000_000, None),
        (1_790_000_000_000, None),
        (1_000_000_000_500_000_000, "2001-09-09T01:46:40.500000+00:00"),
        (1_000_000_000_000_000_000, None),
        (1e300, None),
    ],
    ids=[
        "seconds",
        "nanoseconds",
        "monotonic",
        "bool",
        "string",
        "exactly-2001",
        "just-after",
        "1e15-is-seconds",
        "milliseconds",
        "nanoseconds-past-2001",
        "nanoseconds-at-2001",
        "neither",
    ],
)
def test_a_time_is_kept_only_when_it_names_a_date(
    raw: PlainTree,
    when: str | None,
) -> None:
    assert legacy_journal._time(raw) == when


def test_a_v3_session_reads_like_history() -> None:
    call = _part(
        "multipart/x-tool-call",
        [
            _part("text/x-queue-id", "c1"),
            _part("application/x-tool-read", {"file_path": "a"}),
        ],
    )
    records = _read(
        _v3("text/x-user-message", "hi", _timestamp=1_790_000_000_000_000_000),
        _v3("multipart/x-user-turn", "turn"),
        _v3(
            "multipart/x-user-message",
            [
                _part("text/plain", "look"),
                _part("text/markdown", "here"),
                _part("image/png", {"_bytes": _PNG}),
            ],
        ),
        _v3(
            "multipart/x-assistant-turn",
            [
                _part("text/x-queue-id", "msg_1"),
                _part("text/plain", "reading"),
                _part("application/x-thinking", {"thinking": "t", "signature": "s"}),
                _part("multipart/x-tool-call", [_part("text/x-queue-id", "c0")]),
                call,
            ],
        ),
        _v3(
            "multipart/x-tool-result",
            [
                _part("text/x-queue-id", "c1"),
                _part("text/plain", "line"),
                _part("text/x-error", "failed"),
                _part("image/png", {"_bytes": _PNG}),
                _part("application/x-file-stat", {}),
            ],
        ),
        _v3("multipart/x-model-message", [_part("text/markdown", "ok")]),
        _v3("multipart/x-tool-result", "raw"),
    )
    png = Attachment(mime_descriptor="image/png", data=b"\x89PNG")
    assert records == [
        UserMessage(timestamp=_AT, content="hi"),
        UserMessage(content="turn"),
        UserMessage(content="look\nhere", attachments=(png,)),
        Thinking(content="t", encrypted="s"),
        AssistantMessage(content="reading"),
        ToolCall(call_id="c1", name="read", arguments={"file_path": "a"}),
        ToolResult(
            call_id="c1",
            state="done",
            stdout=("line\nfailed", png),
            result=UncategorizedToolResult(exit_code=1),
            extra={"is_error": True},
        ),
        AssistantMessage(content="ok"),
        ToolResult(
            call_id="",
            state="done",
            stdout=("raw",),
            result=UncategorizedToolResult(),
            extra={"is_error": False},
        ),
    ]


def test_a_role_session_reads_like_history() -> None:
    records = _read(
        {"kind": "message", "role": "user", "content": "hi"},
        {
            "kind": "message",
            "role": "assistant",
            "content": "",
            "thinking_blocks": [{"thinking": "hm", "signature": "s"}],
            "tool_calls": [
                {"id": "t1", "name": "Bash", "input": {"command": "ls"}},
                {"id": "t2", "name": "Bash", "args": {"command": "pwd"}},
            ],
        },
        {
            "kind": "message",
            "role": "tool",
            "tool_call_id": "t1",
            "content": "[exit code: 2]",
            "is_error": False,
        },
    )
    assert records == [
        UserMessage(content="hi"),
        Thinking(content="hm", encrypted="s"),
        ToolCall(call_id="t1", name="Bash", arguments={"command": "ls"}),
        ToolCall(call_id="t2", name="Bash", arguments={"command": "pwd"}),
        ToolResult(
            call_id="t1",
            state="done",
            stdout=("[exit code: 2]",),
            result=BashResult(command=("ls",)),
            extra={"is_error": False},
        ),
    ]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(3, 3), (True, None), ("3", None)],
)
def test_an_int_is_an_int_and_never_a_bool(
    raw: PlainTree,
    expected: int | None,
) -> None:
    assert legacy_journal._int(raw) == expected


def test_a_bool_is_a_bool_alone() -> None:
    assert (legacy_journal._bool(True), legacy_journal._bool(1)) == (True, False)


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
