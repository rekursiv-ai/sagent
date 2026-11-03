"""Tests for reading old sagent session logs as today's records."""

from __future__ import annotations

from typing import TYPE_CHECKING

import base64
import os

from treekle import json_dumps

import pytest

from sagent.lib.userdirs import data_dir
from sagent.model.cost import TokenCost, TokenCount
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
    AgentToAgentMessage,
    AssistantMessage,
    Attachment,
    ContextClear,
    SystemMessage,
    Thinking,
    ToolCall,
    ToolResult,
    UncategorizedToolResult,
    UserMessage,
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
    (result,) = [r for r in history if isinstance(r, UncategorizedToolResult)]
    assert result.stub is None


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
        _result(1, "c1", _DETACHED, id=7),
        {"kind": "update", "id": 7, "content": "real", "is_error": True},
        {"kind": "update", "id": 99, "content": "nowhere"},
    )
    (result,) = [r for r in _history(path) if isinstance(r, UncategorizedToolResult)]
    assert (result.content, result.is_error, result.stub) == ("real", True, None)


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


def test_thinking_keeps_the_seal_each_provider_reads(tmp_path: Path) -> None:
    path = _log(
        tmp_path,
        _user(0, "q"),
        {
            "kind": "history",
            "ref": _ref(1),
            "type": "assistant",
            "text": "a",
            "thought_signature": "g1",
            "thinking_blocks": [
                {"type": "thinking", "thinking": "hmm", "signature": "s1"},
                {"type": "redacted_thinking", "data": "r1"},
                {
                    "type": "reasoning",
                    "id": "rs_1",
                    "encrypted_content": "e1",
                    "summary": [{"text": "sum"}, {"text": "med"}],
                    "content": [{"text": "why"}, {"text": " not"}],
                },
                {"type": "reasoning", "text": "plain"},
            ],
            "tool_calls": [
                {"id": "c1", "name": "Read", "args": {}, "thought_signature": "g2"},
            ],
        },
        _result(2, "c1", "ok"),
    )
    history = _history(path)
    assert [r for r in history if isinstance(r, Thinking)] == [
        Thinking(content="hmm", encrypted="s1"),
        Thinking(encrypted="r1"),
        Thinking(
            content="why not",
            summary="summed",
            encrypted="e1",
            extra={"reasoning_id": "rs_1"},
        ),
        Thinking(content="plain"),
    ]
    (said,) = [r for r in history if isinstance(r, AssistantMessage)]
    (called,) = [r for r in history if isinstance(r, ToolCall)]
    assert said.extra == {"thoughtSignature": "g1"}
    assert called.extra == {"thoughtSignature": "g2"}


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


def test_a_v3_result_without_an_error_part_is_no_error(tmp_path: Path) -> None:
    path = _log(
        tmp_path,
        {
            "kind": "message",
            "descriptor": "multipart/x-model-message",
            "content": [
                {
                    "descriptor": "multipart/x-tool-call",
                    "content": [
                        {"descriptor": "text/x-queue-id", "content": "c1"},
                        {"descriptor": "application/x-tool-ls", "content": {}},
                    ],
                },
            ],
        },
        {
            "kind": "message",
            "descriptor": "multipart/x-tool-result",
            "content": [
                {"descriptor": "text/x-queue-id", "content": "c1"},
                {"descriptor": "text/plain", "content": "ok"},
            ],
        },
    )
    result = _history(path)[-1]
    assert isinstance(result, UncategorizedToolResult)
    assert (result.text(), result.is_error) == ("ok", False)


def test_an_unanswered_call_is_answered_interrupted(tmp_path: Path) -> None:
    path = _log(tmp_path, _user(0, "go"), _call(1, "c1"))
    assert _texts(path) == ["go", "call c1", "[interrupted]"]


@pytest.mark.parametrize(
    ("extra", "content", "stub"),
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
    tmp_path: Path,
    extra: dict[str, PlainTree],
    content: str,
    stub: str | None,
) -> None:
    path = _log(tmp_path, _call(0, "c1"), _result(1, "c1", content, **extra))
    (result,) = [r for r in _history(path) if isinstance(r, UncategorizedToolResult)]
    assert result.stub == stub


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


@pytest.mark.parametrize(
    ("raw", "when"),
    [
        (1_790_000_000.0, "2026-09-21T14:13:20+00:00"),
        (1_790_000_000_000_000_000, "2026-09-21T14:13:20+00:00"),
        (130811.8, None),
        (True, None),
        ("x", None),
    ],
    ids=["seconds", "nanoseconds", "monotonic", "bool", "string"],
)
def test_a_time_is_kept_only_when_it_names_a_date(
    tmp_path: Path,
    raw: PlainTree,
    when: str | None,
) -> None:
    path = _log(
        tmp_path,
        {
            "kind": "history",
            "ref": _ref(0),
            "type": "user",
            "text": "a",
            "timestamp": raw,
        },
    )
    (user,) = _history(path)
    assert isinstance(user, UserMessage)
    assert user.timestamp == when


def test_peers_and_notes_keep_their_kind(tmp_path: Path) -> None:
    path = _log(
        tmp_path,
        _user(0, "go"),
        _said(1, "ok"),
        {"kind": "history", "ref": _ref(2), "type": "system_message", "text": "note"},
        {
            "kind": "history",
            "ref": _ref(3),
            "type": "agent_send",
            "source": "Agent",
            "text": "hi",
        },
    )
    history = _history(path)
    assert [type(r) for r in history] == [
        UserMessage,
        AssistantMessage,
        SystemMessage,
        AgentToAgentMessage,
    ]
    peer = history[-1]
    assert isinstance(peer, AgentToAgentMessage)
    assert peer.sender == "Agent"


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
    (result,) = [r for r in history if isinstance(r, UncategorizedToolResult)]
    assert result.is_error is True


def test_every_kind_keeps_its_time_and_attachments(tmp_path: Path) -> None:
    when, at = 1_790_000_000, "2026-09-21T14:13:20+00:00"
    seen = {"timestamp": when, "attachments": [{"mime": "image/png", "data": _PNG}]}
    path = _log(
        tmp_path,
        {**_user(0, "u"), **seen},
        {**_call(1, "c1", "said"), "timestamp": when},
        {**_result(2, "c1", "out"), **seen},
        {
            "kind": "history",
            "ref": _ref(3),
            "type": "system_message",
            "text": "n",
            **seen,
        },
        {"kind": "history", "ref": _ref(4), "type": "agent_send", "text": "p", **seen},
    )
    png = (Attachment(mime_descriptor="image/png", data=b"\x89PNG"),)
    user, said, call, result, note, peer = _history(path)
    assert user == UserMessage(timestamp=at, content="u", attachments=png)
    assert said == AssistantMessage(timestamp=at, content="said")
    assert isinstance(call, ToolCall)
    assert call.timestamp == at
    assert isinstance(result, UncategorizedToolResult)
    assert (result.timestamp, result.attachments) == (at, png)
    assert note == SystemMessage(timestamp=at, content="n")
    assert peer == AgentToAgentMessage(timestamp=at, content="p", attachments=png)


def test_an_attachment_that_is_no_strict_base64_is_dropped(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    path = _log(
        tmp_path,
        {**_user(0, "u"), "attachments": [{"mime": "image/png", "data": "iVBO!!Rw=="}]},
    )
    (user,) = _history(path)
    assert isinstance(user, UserMessage)
    assert user.attachments == ()
    assert caplog.messages == ["dropping an attachment that does not decode"]


def test_a_line_of_no_record_is_logged_with_its_file(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    path = _log(tmp_path, "{nope", _user(0, "u"))
    assert _texts(path) == ["u"]
    assert caplog.messages == [f"skipping a line of {path} that holds no record"]


def test_a_value_of_the_wrong_type_reads_as_none_given(tmp_path: Path) -> None:
    path = _log(
        tmp_path,
        {"kind": "history", "ref": _ref(0), "type": "user", "text": 5},
        _call(1, "c", "x"),
        _result(2, "c", "out", is_error="yes"),
    )
    user, said, _, result = _history(path)
    assert user == UserMessage(content="")
    assert said == AssistantMessage(content="x")
    assert isinstance(result, UncategorizedToolResult)
    assert result.is_error is False


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
    assert isinstance(result, UncategorizedToolResult)
    assert result.stub == "pending"


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


def test_a_turn_without_a_signature_carries_none(tmp_path: Path) -> None:
    (said,) = _history(_log(tmp_path, _said(0, "x")))
    assert isinstance(said, AssistantMessage)
    assert said.extra == {}


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
    assert isinstance(result, UncategorizedToolResult)
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


@pytest.mark.parametrize(
    ("raw", "when"),
    [
        (1_000_000_000, None),
        (1_000_000_000.5, "2001-09-09T01:46:40.500000+00:00"),
        (1_000_000_000_000_000, None),
        (1_790_000_000_000, None),
        (1_000_000_000_500_000_000, "2001-09-09T01:46:40.500000+00:00"),
        (1_000_000_000_000_000_000, None),
        (1e300, None),
    ],
    ids=[
        "exactly-2001",
        "just-after",
        "1e15-is-seconds",
        "milliseconds",
        "nanoseconds",
        "nanoseconds-at-2001",
        "neither",
    ],
)
def test_a_time_s_bounds(tmp_path: Path, raw: PlainTree, when: str | None) -> None:
    path = _log(tmp_path, {**_user(0, "a"), "timestamp": raw})
    (user,) = _history(path)
    assert isinstance(user, UserMessage)
    assert user.timestamp == when


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
    turn = legacy_journal._turn({}, text="hi", timestamp=None)
    assert (turn.tokens, turn.spend, turn.stop_reason) == (
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


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
