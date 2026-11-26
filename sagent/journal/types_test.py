"""Tests for the session records: what the model reads of each, and how each is stored."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, fields, is_dataclass, replace
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Final, cast

import json
import re
import time

from treekle import PlainTree, ReadError, from_plain, immutable, json_loads

import pytest

from sagent.journal import types
from sagent.journal.types import (
    FORMER_RESULT_KINDS,
    RENAMED,
    RENAMED_KINDS,
    AgentIdle,
    AgentMessage,
    AgentSelfResult,
    AgentSendResult,
    AgentSpawnResult,
    AssistantMessage,
    Attachment,
    BackgroundTaskResult,
    BashResult,
    Cancelled,
    ContextClear,
    ContextCompaction,
    ContextState,
    EditResult,
    Failed,
    GlobResult,
    GrepResult,
    IncompleteRecord,
    LineSpan,
    ListResult,
    McpResult,
    MemoryResult,
    ModelIdle,
    Outcome,
    PaperAuthorResult,
    PaperDetailsResult,
    PaperFetchResult,
    PaperSearchResult,
    PlanResult,
    PlayAudioResult,
    ReadResult,
    RecallResult,
    RemoteToolResult,
    RequestMeasurement,
    SessionContext,
    SessionRecord,
    SessionResult,
    SkillResult,
    Splice,
    Succeeded,
    SystemMessage,
    Thinking,
    TokenUsage,
    ToolCall,
    ToolMessage,
    UncategorizedRecord,
    UncategorizedToolResult,
    UserMessage,
    WebBrowserResult,
    WebFetchResult,
    WebSearchHit,
    WebSearchResult,
    WriteResult,
    as_sent,
    current_plain,
    first_name,
    former_kind,
    labeled_text,
    late_messages,
    now_timestamp,
    pairings,
    parsed_timestamp,
    record_classes,
    renamed_records,
    stated_timestamp,
    stored_names,
    stored_plain,
    stored_record,
    timestamp_as_written,
)
from sagent.lib.userdirs import base_dir_variables


_CWD: Final[Path] = Path(__file__).resolve().parent


def _json() -> dict[str, PlainTree]:
    """Return a JSON field as a provider sends it: plain dicts and lists, nested."""
    return {"k": [1, {"n": 2}]}


_ARGUMENT_ECHO_RESULTS: Final = (
    BashResult(command=("bash", "-c", "ls")),
    WebBrowserResult(action="open"),
    GrepResult(pattern="x", path="/r"),
    GlobResult(pattern="*.py", path="/r"),
    ListResult(path="/r"),
    SkillResult(skill="tdd"),
    AgentSendResult(to="kid", delay_sec=2.5),
    BackgroundTaskResult(operation="cancel", job_id="j1"),
    SessionResult(action="list", agent="kid"),
    MemoryResult(action="save", name="n"),
    PlayAudioResult(path="/r/a.wav"),
    PaperSearchResult(query="q", source="s2"),
    PaperDetailsResult(operation="citations", source="s2"),
    PaperAuthorResult(operation="papers"),
    RemoteToolResult(endpoint="gpu1"),
    McpResult(server="srv", tool="echo"),
    UncategorizedToolResult(),
    AgentSelfResult(),
    PaperFetchResult(),
    PlanResult(),
    RecallResult(),
)
"""One result of each class whose fields only repeat the call's arguments."""


# Every JSON field is given plain, so each record's own ``__post_init__`` must
# freeze it: a default is frozen already, and would show nothing.
def _every_record() -> list[SessionRecord]:
    return [
        SessionContext(
            cwd=Path("/w"),
            model="m",
            effort="high",
            encoding={"crlf": True},
            extra=_json(),
        ),
        UserMessage(
            content="hi",
            attachments=(Attachment(mime_descriptor="image/png", data=b"\x89PNG"),),
            extra=_json(),
        ),
        AssistantMessage(content="hello", extra={"id": "a1"}),
        Thinking(content="hm", encrypted="sealed", summary="s", extra=_json()),
        ToolCall(
            call_id="1",
            name="Read",
            arguments={"path": "a.py", "n": [1, 2]},
            extra=_json(),
        ),
        ToolMessage(
            call_id="1",
            wire_id="ab12",
            state="done",
            stdout=("out\n",),
            stderr=("warn\n",),
            result=UncategorizedToolResult(exit_code=2),
            extra=_json(),
        ),
        ToolMessage(
            call_id="2",
            state="done",
            stdout=(
                "Edited c.py.",
                Splice(
                    before="a",
                    after="b",
                    start=3,
                    count=1,
                    bare=frozenset({"after"}),
                ),
                Attachment(mime_descriptor="image/png", data=b"png"),
            ),
            result=UncategorizedToolResult(),
            shown="[pruned]",
        ),
        ToolMessage(call_id="3", state="pending", shown="[running]"),
        ToolMessage(call_id="4", state="cancelled", stdout=("so far\n",)),
        ToolMessage(call_id="5", state="failed", stderr=("ValueError: no",)),
        *(
            ToolMessage(call_id=type(result).__name__, state="done", result=result)
            for result in _ARGUMENT_ECHO_RESULTS
        ),
        ToolMessage(
            call_id="6",
            state="done",
            stdout=("3\tx\n", LineSpan(start=3, count=1)),
            result=ReadResult(path="/r/a.py", status="past_end", exit_code=1),
        ),
        ToolMessage(
            call_id="7",
            state="done",
            stdout=("Wrote /r/a.py.",),
            result=WriteResult(path="/r/a.py", content="x\n"),
        ),
        ToolMessage(
            call_id="8",
            state="done",
            stdout=("Edited /r/a.py.", Splice(before="a", after="b")),
            result=EditResult(path="/r/a.py"),
        ),
        ToolMessage(
            call_id="9",
            state="done",
            stdout=("t\nu\ns", WebSearchHit(url="u", title="t", snippet="s")),
            result=WebSearchResult(query="q", duration_sec=0.5),
        ),
        ToolMessage(
            call_id="10",
            state="done",
            stdout=("page",),
            result=WebFetchResult(url="http://a", code=200, duration_sec=0.5, size=4),
        ),
        ToolMessage(
            call_id="11",
            state="done",
            stdout=("three TODOs",),
            result=AgentSpawnResult(
                agent_id="Agent_0",
                agent_kind="persistent",
                prompt="count",
                model="scripted",
                state="completed",
                tokens=15,
                duration_sec=1.5,
                tool_calls=1,
                output_file="/s/session.jsonl",
            ),
        ),
        SystemMessage(
            content="reminder",
            subtype="rules",
            due_after=ModelIdle,
            extra=_json(),
        ),
        TokenUsage(
            info={"request": 1},
            rate_limits={"5h": 0.5},
            extra=_json(),
            measurement=RequestMeasurement(
                tokenizer="provider/model",
                model="provider/model",
                frame="frame",
                prefix="prefix",
                records=2,
                prompt=101,
                block="block",
                block_records=1,
                block_tokens=31,
            ),
        ),
        ContextState(kind="k", content="c", extra={"full": [True]}),
        ContextCompaction(summary="sum", extra={"tokens": {"before": 9}}),
        ContextCompaction(
            summary="The user asked.",
            provider="anthropic",
            block={
                "type": "compaction",
                "content": "The user asked.",
                "signature": "s",
            },
            replaces="abc",
        ),
        ContextClear(
            cleared_session_id="old",
            system_prompt="sys",
            summary="kept",
            history=(UserMessage(content="before"),),
            extra=_json(),
        ),
        AgentMessage(
            content="ping",
            sender="a",
            recipient="b",
            due_after=AgentIdle,
            extra=_json(),
        ),
        UncategorizedRecord(kind="custom", payload={"k": "v"}),
        IncompleteRecord(text="{torn"),
    ]


@pytest.fixture
def pacific(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Run in Los Angeles' time zone."""
    isolated = base_dir_variables()
    try:
        with monkeypatch.context() as local:
            local.setenv("TZ", "America/Los_Angeles")
            time.tzset()
            yield
    finally:
        time.tzset()
        assert base_dir_variables() == isolated


@pytest.mark.usefixtures("pacific")
def test_turn_context_text_states_each_setting_on_its_own_line() -> None:
    context = SessionContext(
        cwd=Path("/w"),
        environment="Platform: linux",
        model="opus",
        timestamp=datetime(2026, 10, 23, tzinfo=UTC),
        agents_md="Be terse.",
    )
    assert context.text() == (
        "Working directory: /w\n"
        "Platform: linux\n"
        "Model: opus\n"
        "Current time: Thu 2026-10-22 17:00 -07:00 (PDT)\n"
        "Be terse."
    )


def test_turn_context_text_names_the_session_s_user_first() -> None:
    """Who a message with no sender is from."""
    assert SessionContext(user="ann", cwd=Path("/w")).text() == (
        "User: ann\nWorking directory: /w"
    )


@pytest.mark.usefixtures("pacific")
def test_the_time_reads_where_the_user_is() -> None:
    winter = datetime(2026, 12, 1, 8, 30, tzinfo=UTC)
    assert SessionContext(timestamp=winter).text() == (
        "Current time: Tue 2026-12-01 00:30 -08:00 (PST)"
    )


def test_a_timestamp_is_now_here_to_the_second() -> None:
    before = datetime.now().astimezone().replace(microsecond=0)
    when = now_timestamp()
    after = datetime.now().astimezone()
    assert when.utcoffset() == before.utcoffset()
    assert when.microsecond == 0
    assert before <= when <= after


def test_a_stored_time_reads_as_a_datetime_and_is_stored_as_it_was() -> None:
    """A journal written before a time was a ``datetime`` reads, and writes the same."""
    stored = {
        "py/object": "UserMessage",
        "timestamp": "2026-11-19T02:03:04-08:00",
        "content": "hi",
    }
    read = stored_record(stored, UserMessage)
    assert read.timestamp == datetime(
        2026,
        11,
        19,
        2,
        3,
        4,
        tzinfo=timezone(timedelta(hours=-8)),
    )
    assert read.extra == {}
    assert stored_plain(read)["timestamp"] == "2026-11-19T02:03:04-08:00"


@pytest.mark.parametrize(
    ("spelled", "when"),
    [
        ("2026-11-19T10:03:04.123Z", datetime(2026, 11, 19, 10, 3, 4, 123_000, UTC)),
        # A time with no offset is read as UTC, so every time read compares.
        ("2026-11-19T10:03:04", datetime(2026, 11, 19, 10, 3, 4, tzinfo=UTC)),
        ("soon", None),
    ],
    ids=["another-spelling", "no-offset", "no-time"],
)
@pytest.mark.parametrize(
    "kind",
    [
        SessionContext,
        UserMessage,
        AssistantMessage,
        Thinking,
        SystemMessage,
        TokenUsage,
        ContextState,
        ContextClear,
        AgentMessage,
    ],
)
def test_a_stored_time_spelled_another_way_keeps_its_spelling_beside_the_record(
    kind: type,
    spelled: str,
    when: datetime | None,
) -> None:
    stored = {"py/object": kind.__name__, "context_id": 3, "timestamp": spelled}
    read = stored_record(stored)
    assert isinstance(read, kind)
    assert not isinstance(read, IncompleteRecord | UncategorizedRecord)
    assert (read.timestamp, read.extra) == (when, {"$timestamp_as_written": spelled})
    # It is stored key for key as a record whose time ``isoformat`` spells: the
    # spelling is the stored time, where a time is stored, and nothing beside it.
    again = stored_plain(read)
    at_some_time = replace(read, timestamp=datetime(2026, 1, 1, tzinfo=UTC), extra={})
    spelled_alike = stored_plain(at_some_time)
    assert list(again.items()) == list((spelled_alike | {"timestamp": spelled}).items())
    assert stored_record(again) == read


def test_a_spelling_kept_beside_a_record_is_the_time_it_is_stored_with() -> None:
    """As a converter keeps a line's own spelling: the stored record holds it once."""
    record = UserMessage(
        timestamp=datetime(2026, 11, 19, 10, 3, 4, tzinfo=UTC),
        extra={"$timestamp_as_written": "2026-11-19T10:03:04.000Z", "k": 1},
    )
    stored = stored_plain(record)
    assert (stored["timestamp"], stored["extra"]) == (
        "2026-11-19T10:03:04.000Z",
        {"k": 1},
    )
    assert stored_record(stored) == record


def test_a_spelling_is_kept_only_while_it_names_the_record_s_time() -> None:
    """A record given another time, or none, is stored with that, not the spelling."""
    read = stored_record(
        {"py/object": "UserMessage", "timestamp": "2026-11-19T10:03:04Z"},
        UserMessage,
    )
    later = stored_plain(replace(read, timestamp=datetime(2026, 11, 20, tzinfo=UTC)))
    assert (later["timestamp"], later["extra"]) == ("2026-11-20T00:00:00+00:00", {})
    cleared = stored_plain(replace(read, timestamp=None))
    assert ("timestamp" in cleared, cleared["extra"]) == (False, {})
    # The same instant in another zone is still the time the source spelled.
    west = timezone(timedelta(hours=-8))
    moved = replace(read, timestamp=datetime(2026, 11, 19, 2, 3, 4, tzinfo=west))
    assert stored_plain(moved)["timestamp"] == "2026-11-19T10:03:04Z"
    assert timestamp_as_written(None, {"$timestamp_as_written": 5}) is None


def test_the_records_that_keep_a_time_s_spelling_are_those_with_a_timestamp() -> None:
    # The cached set is worked out once per process; this works it out here.
    stamped = types._stamped_tags.__wrapped__()
    assert {"UserMessage", "ToolCall", "ToolMessage", "ContextClear"} <= stamped
    assert {"Attachment", "ReadResult", "Splice"}.isdisjoint(stamped)
    assert stamped == types._stamped_tags()


def test_a_former_tag_reads_as_the_tag_its_class_bears_now() -> None:
    # The cached map is worked out once per process; this works it out here.
    names = types._tags_now.__wrapped__()
    assert names["ToolResult"] == "ToolMessage"
    assert names["AgentToAgentMessage"] == "AgentMessage"
    assert names == types._tags_now()


@pytest.mark.parametrize(
    ("entry", "labeled"),
    [
        (
            AgentMessage(sender="reviewer", content="finding"),
            "[from reviewer]: finding",
        ),
        # Labeled once: its own label is not written again.
        (
            AgentMessage(sender="reviewer", content="[from reviewer]: x"),
            "[from reviewer]: x",
        ),
        # Another sender's label is not mistaken for this one.
        (
            AgentMessage(sender="b", content="[from a]: x"),
            "[from b]: [from a]: x",
        ),
        (AgentMessage(sender="a"), "[from a]: "),
        # A person and an agent of one name are told apart.
        (UserMessage(sender="bob", content="hi"), "[from user bob]: hi"),
        (
            UserMessage(sender="bob", content="[from user bob]: hi"),
            "[from user bob]: hi",
        ),
        (
            UserMessage(sender="bob", content="[from bob]: hi"),
            "[from user bob]: [from bob]: hi",
        ),
        # Who names no sender is not named.
        (UserMessage(content="mine"), "mine"),
        (UserMessage(), ""),
        (AgentMessage(sender=None, content="unsigned"), "unsigned"),
    ],
)
def test_a_message_s_text_is_labeled_with_who_sent_it(
    entry: UserMessage | AgentMessage,
    labeled: str,
) -> None:
    assert labeled_text(entry) == labeled


@pytest.mark.parametrize(
    ("spelled", "when"),
    [
        ("2026-11-19T10:03:04Z", datetime(2026, 11, 19, 10, 3, 4, tzinfo=UTC)),
        ("2026-11-19T02:03:04-08:00", datetime(2026, 11, 19, 10, 3, 4, tzinfo=UTC)),
        # No offset is read as UTC, so every time read compares with every other.
        ("2026-11-19T10:03:04", datetime(2026, 11, 19, 10, 3, 4, tzinfo=UTC)),
        ("soon", None),
        ("", None),
    ],
)
def test_a_spelled_time_is_read_with_an_offset_or_as_none(
    spelled: str,
    when: datetime | None,
) -> None:
    read = parsed_timestamp(spelled)
    assert read == when
    assert read is None or read.utcoffset() is not None


def test_the_stored_shape_keeps_a_time_as_it_is_stored() -> None:
    """``current_plain`` returns what ``stored_plain`` writes; only a read parses it."""
    stored = {"py/object": "UserMessage", "timestamp": "2026-11-19T10:03:04Z"}
    assert current_plain(stored) == stored


def test_a_record_states_its_time_as_it_is_written() -> None:
    when = datetime(2026, 11, 19, 10, 3, 4, tzinfo=UTC)
    spelled = {"$timestamp_as_written": "2026-11-19T10:03:04Z"}
    assert stated_timestamp(UserMessage(timestamp=when)) == "2026-11-19T10:03:04+00:00"
    assert stated_timestamp(UserMessage(timestamp=when, extra=spelled)) == (
        "2026-11-19T10:03:04Z"
    )
    # A record of no other type keeps the spelling in its payload.
    assert stated_timestamp(
        UncategorizedRecord(timestamp=when, kind="odd", payload=spelled),
    ) == ("2026-11-19T10:03:04Z")
    assert stated_timestamp(UserMessage()) is None
    assert stated_timestamp(IncompleteRecord(text="{torn")) is None


def test_a_window_s_first_records_are_each_stored_as_they_were_spelled() -> None:
    """A ``UserMessage`` is stored there with no tag; every other record with one."""
    when = datetime(2026, 11, 19, 10, 3, 4, tzinfo=UTC)
    spelled = {"$timestamp_as_written": "2026-11-19T10:03:04Z"}
    window = ContextClear(
        history=(
            UserMessage(timestamp=when, extra=spelled),
            AssistantMessage(timestamp=when, extra=spelled),
            IncompleteRecord(text="{torn"),
        ),
    )
    stored = stored_plain(window)
    message, answered, torn = from_plain(stored["history"], list[dict[str, object]])
    assert [
        (held.get("py/object"), held["timestamp"], held["extra"])
        for held in (message, answered)
    ] == [
        (None, "2026-11-19T10:03:04Z", {}),
        ("AssistantMessage", "2026-11-19T10:03:04Z", {}),
    ]
    assert torn == {"py/object": "IncompleteRecord", "text": "{torn"}
    assert stored_record(stored) == window


def test_a_window_whose_history_holds_no_record_does_not_read() -> None:
    with pytest.raises(ReadError):
        _ = stored_record({"py/object": "ContextClear", "history": ["no record"]})


def test_a_time_in_plain_data_under_py_dict_is_no_record_s() -> None:
    """A host's ``Call`` holds its arguments so: whatever they look like, they are data."""
    held = {"py/object": "UserMessage", "timestamp": "2026-11-19T10:03:04Z"}
    stored = {
        "py/object": "ToolCall",
        "call_id": "1",
        "name": "Read",
        # Its own copy: a reader that took it for a record would change it in place.
        "arguments": {"py/dict": dict(held)},
    }
    assert stored_record(stored, ToolCall).arguments == held


def test_a_call_s_arguments_are_no_record_whatever_their_keys() -> None:
    arguments = {
        "py/object": "UserMessage",
        "timestamp": "t",
        "extra": {"$timestamp_as_written": "kept"},
    }
    stored = stored_plain(ToolCall(call_id="1", name="Read", arguments=arguments))
    assert stored_record(stored, ToolCall).arguments == arguments


def test_a_call_and_its_result_keep_a_time_spelled_another_way() -> None:
    call = stored_record(
        {"py/object": "ToolCall", "timestamp": "soon", "call_id": "1", "name": "Read"},
        ToolCall,
    )
    result = stored_record(
        {
            "py/object": "ToolMessage",
            "timestamp": "soon",
            "call_id": "1",
            "state": "cancelled",
        },
        ToolMessage,
    )
    assert (call.timestamp, call.extra) == (None, {"$timestamp_as_written": "soon"})
    assert (result.timestamp, result.extra) == (None, {"$timestamp_as_written": "soon"})


def test_a_record_of_no_other_type_keeps_such_a_time_in_its_payload() -> None:
    """It has no ``extra``; a ``timestamp`` its payload holds is its source's own key."""
    payload: dict[str, PlainTree] = {"timestamp": "soon", "k": 1}
    stored = {
        "py/object": "UncategorizedRecord",
        "timestamp": "soon",
        "kind": "odd",
        "payload": payload,
    }
    read = stored_record(stored, UncategorizedRecord)
    assert (read.timestamp, read.payload) == (
        None,
        {**payload, "$timestamp_as_written": "soon"},
    )
    # Stored again it is as it was, its payload the source's own.
    again = stored_plain(read)
    assert list(again.items()) == list(stored.items())


def test_a_payload_s_own_timestamp_is_not_the_time_a_record_is_stored_with() -> None:
    record = UncategorizedRecord(
        timestamp=datetime(2026, 11, 19, 10, 3, 4, tzinfo=UTC),
        kind="odd",
        payload={"timestamp": "2026-11-19T10:03:04.000Z"},
    )
    stored = stored_plain(record)
    assert stored["timestamp"] == "2026-11-19T10:03:04+00:00"
    assert stored_record(stored) == record


def test_times_read_from_a_session_that_spells_them_both_ways_sort() -> None:
    stored = [
        {
            "py/object": "UserMessage",
            "timestamp": "2026-11-19T02:03:05-08:00",
            "content": "later",
        },
        {
            "py/object": "UserMessage",
            "timestamp": "2026-11-19T10:03:04",
            "content": "sooner",
        },
    ]
    read = [stored_record(each, UserMessage) for each in stored]
    times = [(r.timestamp, r.content) for r in read if r.timestamp is not None]
    # By the instant, not by the text: 02:03:05 at -08:00 is 10:03:05 UTC.
    assert [content for _, content in sorted(times)] == ["sooner", "later"]


def test_turn_context_text_skips_what_it_does_not_state() -> None:
    assert SessionContext().text() == ""
    assert SessionContext(model="opus").text() == "Model: opus"
    assert SessionContext(environment="", agents_md="").text() == ""


@pytest.mark.parametrize(
    ("tools", "shown"),
    [
        (("Read", "Bash"), "Tools you may call: Read, Bash"),
        ((), "Tools you may call: none"),
    ],
    ids=["some", "none"],
)
def test_turn_context_text_names_the_tools_the_model_may_call(
    tools: tuple[str, ...],
    shown: str,
) -> None:
    assert SessionContext(tools=tools).text() == shown


def test_a_stored_record_keeps_its_fields_in_declaration_order() -> None:
    """``timestamp`` is declared before ``content``, though it sorts after it."""
    stored = stored_plain(
        UserMessage(timestamp=datetime(2026, 10, 23, tzinfo=UTC), content="hi"),
    )
    assert [key for key in stored if not key.startswith("py/")] == [
        "timestamp",
        "content",
        "extra",
    ]


def test_a_stored_record_is_read_as_the_class_asked_for() -> None:
    with pytest.raises(ReadError):
        _ = stored_record(stored_plain(UserMessage(content="hi")), ToolCall)


@pytest.mark.parametrize(
    "plain",
    [
        {"py/object": "UserMessage", "content": "hi", "added_later": 1},
        {
            "py/object": "UserMessage",
            "attachments": [{"mime_descriptor": "a", "data": "", "added_later": 1}],
        },
    ],
)
def test_a_record_with_a_field_its_class_lacks_does_not_read(
    plain: dict[str, object],
) -> None:
    with pytest.raises(ReadError, match="added_later"):
        _ = stored_record(plain)


def test_a_result_keeps_the_id_its_call_went_by_on_the_wire() -> None:
    """A journal written before ``wire_id`` reads it as ``""``."""
    sent = ToolMessage(
        call_id="toolu_1",
        wire_id="ab12",
        state="done",
        result=UncategorizedToolResult(),
    )
    assert stored_record(json_loads(json.dumps(stored_plain(sent)))) == sent
    old = {"py/object": "ToolResult", "call_id": "toolu_1", "state": "done"}
    assert stored_record(old) == replace(sent, wire_id="", result=None)


def test_text_joins_the_text_items_of_stdout_then_stderr() -> None:
    """Items that are not text, such as a ``Splice`` or an ``Attachment``, are skipped."""
    image = Attachment(mime_descriptor="image/png", data=b"png")
    result = ToolMessage(
        call_id="1",
        state="done",
        stdout=("a\n", Splice(after="x"), "b\n", image),
        stderr=("warn",),
        result=UncategorizedToolResult(),
    )

    assert result.text() == "a\nb\nwarn"
    assert ToolMessage(call_id="1", state="done").text() == ""


def test_text_is_shown_when_set_and_the_output_is_kept() -> None:
    """A prune or a placeholder sets ``shown``; ``stdout`` is unchanged."""
    read = ToolMessage(
        call_id="1",
        state="done",
        stdout=("body",),
        result=UncategorizedToolResult(),
    )
    pruned = replace(read, shown="[pruned]")

    assert (read.text(), pruned.text()) == ("body", "[pruned]")
    assert pruned.stdout == ("body",)
    assert ToolMessage(call_id="2", state="pending", shown="[running]").text() == (
        "[running]"
    )
    assert replace(read, shown="").text() == ""


@pytest.mark.parametrize(
    ("result", "failed"),
    [
        (
            ToolMessage(call_id="1", state="done", result=UncategorizedToolResult()),
            False,
        ),
        (
            ToolMessage(
                call_id="1",
                state="done",
                result=UncategorizedToolResult(exit_code=2),
            ),
            True,
        ),
        (
            ToolMessage(
                call_id="1",
                state="done",
                result=UncategorizedToolResult(exit_code=-9),
            ),
            True,
        ),
        (ToolMessage(call_id="1", state="done"), False),
        (ToolMessage(call_id="1", state="pending"), False),
        (ToolMessage(call_id="1", state="failed"), True),
        (ToolMessage(call_id="1", state="cancelled"), True),
        (ToolMessage(call_id="1", state="done", result=ReadResult(path="a")), False),
        (ToolMessage(call_id="1", state="done", result=ReadResult(exit_code=1)), True),
        (ToolMessage(call_id="1", state="done", result=BashResult(exit_code=2)), False),
    ],
    ids=[
        "zero",
        "nonzero",
        "signal",
        "no-result",
        "pending",
        "failed",
        "cancelled",
        "read-zero",
        "read-nonzero",
        "bash-nonzero",
    ],
)
def test_is_error_for_a_failed_or_cancelled_call_or_a_nonzero_exit_code(
    result: ToolMessage,
    failed: bool,
) -> None:
    assert result.is_error is failed


def test_attachments_are_the_attachment_items_of_stdout() -> None:
    one = Attachment(mime_descriptor="image/png", data=b"1")
    two = Attachment(mime_descriptor="image/jpeg", data=b"2")
    result = ToolMessage(call_id="1", state="done", stdout=("[image]", one, "x", two))

    assert result.attachments == (one, two)
    assert result.attached((two,)).stdout == ("[image]", "x", two)
    assert result.attached(()).attachments == ()


def test_a_restated_result_keeps_its_result_and_drops_attachments() -> None:
    image = Attachment(mime_descriptor="image/png", data=b"png")
    edit = Splice(after="x")
    result = ToolMessage(
        call_id="1",
        state="done",
        stdout=("Edited a.py.", edit, image),
        stderr=("warn",),
        result=UncategorizedToolResult(exit_code=1),
    )

    restated = result.restated("[pruned]")

    assert restated == replace(
        result,
        shown="[pruned]",
        stdout=("Edited a.py.", edit),
    )
    assert restated.text() == "[pruned]"
    assert restated.is_error


def _formerly(tag: str, **fields: object) -> ToolMessage:
    """Return the ``ToolMessage`` read from a legacy record of class ``tag``."""
    read = stored_record({"py/object": tag, "call_id": "1", **fields})
    assert isinstance(read, ToolMessage)
    return read


@pytest.mark.parametrize("tag", sorted(FORMER_RESULT_KINDS))
def test_a_result_stored_under_a_former_class_reads_as_a_tool_result(tag: str) -> None:
    assert RENAMED_KINDS[tag] == "ToolMessage"
    read = _formerly(tag)
    assert (read.call_id, read.state) == (
        "1",
        "failed" if tag == "ToolErrorResult" else "done",
    )
    assert json.dumps(stored_plain(read)).startswith('{"py/object": "ToolMessage"')


@pytest.mark.parametrize("tag", sorted(FORMER_RESULT_KINDS))
def test_a_former_result_keeps_its_call_id_context_and_shown_text(
    tag: str,
) -> None:
    read = _formerly(
        tag,
        context_id=3,
        timestamp="2026-10-23T00:00:00+00:00",
        wire_id="w",
        shown="[pruned]",
        extra={"k": [1]},
    )
    assert (read.context_id, read.timestamp, read.wire_id) == (
        3,
        datetime(2026, 10, 23, tzinfo=UTC),
        "w",
    )
    assert read.text() == "[pruned]"
    assert read.extra == {"k": (1,)}


def test_a_former_result_keeps_a_time_spelled_another_way() -> None:
    read = _formerly("BashResult", timestamp="t", extra={"k": [1]})
    assert (read.timestamp, read.extra) == (
        None,
        {"k": (1,), "$timestamp_as_written": "t"},
    )


# Each former class whose only output was its ``content`` text.
_TEXT_CLASSES = sorted(
    FORMER_RESULT_KINDS
    - {
        *("ShellCommandResult", "BashResult", "FileWriteResult", "WriteResult"),
        *("FileEditResult", "EditResult", "WebSearchResults", "WebSearchResult"),
        "ToolErrorResult",
    },
)


@pytest.mark.parametrize("tag", _TEXT_CLASSES)
def test_a_former_text_result_reads_as_its_content_on_stdout(tag: str) -> None:
    read = _formerly(tag, content="what the model read")
    assert read.text() == "what the model read"
    assert (read.stdout, read.stderr) == (("what the model read",), ())
    result_class = record_classes()[RENAMED.get(tag, tag)]
    assert read.result == result_class()
    assert _formerly(tag).stdout == ()


_SPAWNED: Final = AgentSpawnResult(
    agent_id="Agent_0",
    agent_kind="persistent",
    prompt="count",
    model="scripted",
    state="completed",
    tokens=15,
    duration_sec=1.5,
    tool_calls=1,
    output_file="/s/session.jsonl",
)


@pytest.mark.parametrize(
    ("tag", "fields", "expected_result"),
    [
        (
            "ReadResult",
            {"path": "/a.py", "status": "empty"},
            ReadResult(path="/a.py", status="empty"),
        ),
        ("FileReadResult", {"path": "/a.py"}, ReadResult(path="/a.py")),
        (
            "WriteResult",
            {"path": "a.py", "content": "the file's text"},
            WriteResult(path="a.py", content="the file's text"),
        ),
        ("FileWriteResult", {}, WriteResult()),
        ("EditResult", {"path": "a.py"}, EditResult(path="a.py")),
        ("FileEditResult", {"path": "a.py"}, EditResult(path="a.py")),
        (
            "WebFetchResult",
            {"url": "http://a", "code": 200, "duration_sec": 0.5, "size": 9},
            WebFetchResult(url="http://a", code=200, duration_sec=0.5, size=9),
        ),
        (
            "WebSearchResult",
            {"query": "q", "duration_sec": 0.5},
            WebSearchResult(query="q", duration_sec=0.5),
        ),
        ("WebSearchResults", {"query": "q"}, WebSearchResult(query="q")),
        (
            "AgentSpawnResult",
            {
                f.name: getattr(_SPAWNED, f.name)
                for f in fields(_SPAWNED)
                if f.name not in {"exit_code", "shown"}
            },
            _SPAWNED,
        ),
        ("AgentStatusResult", {"agent_id": "kid"}, AgentSpawnResult(agent_id="kid")),
    ],
)
def test_a_former_record_s_fields_become_its_result(
    tag: str,
    fields: Mapping[str, object],
    expected_result: object,
) -> None:
    """A legacy record's non-text fields move to its result class."""
    read = _formerly(tag, **fields)
    failed = _formerly(tag, is_error=True, **fields)

    assert (read.result, read.is_error, read.extra) == (expected_result, False, {})
    assert isinstance(failed.result, type(expected_result))
    assert (getattr(failed.result, "exit_code", None), failed.is_error) == (1, True)
    assert stored_record(stored_plain(read)) == read


@pytest.mark.parametrize(
    "result",
    _ARGUMENT_ECHO_RESULTS,
    ids=[type(result).__name__ for result in _ARGUMENT_ECHO_RESULTS],
)
def test_a_legacy_record_s_fields_become_its_result_s_fields(result: object) -> None:
    not_copied = {"py/object", "exit_code"}
    fields = {k: v for k, v in stored_plain(result).items() if k not in not_copied}

    read = _formerly(type(result).__name__, **fields)

    assert (read.result, read.extra) == (result, {})
    assert not read.is_error
    marked = _formerly(type(result).__name__, is_error=True, **fields)
    assert marked.is_error
    assert marked.state == ("failed" if isinstance(result, BashResult) else "done")


def test_a_stored_function_result_reads_as_an_uncategorized_one() -> None:
    """``FunctionResult`` had one field, its tool's name; ``ToolCall.name`` has it."""
    record = _formerly("FunctionResult", name="add", content="3")
    nested = stored_record(
        {
            "py/object": "ToolResult",
            "call_id": "1",
            "state": "done",
            "stdout": ["3"],
            "result": {"py/object": "FunctionResult", "name": "add", "exit_code": 2},
        },
    )

    assert record == ToolMessage(
        call_id="1",
        state="done",
        stdout=("3",),
        result=UncategorizedToolResult(),
    )
    assert nested == replace(record, result=UncategorizedToolResult(exit_code=2))


def test_a_former_tool_error_reads_as_a_failed_call_with_the_error_on_stderr() -> None:
    read = _formerly("ToolErrorResult", content="ValueError: no", is_error=True)
    assert (read.state, read.stdout, read.stderr, read.result) == (
        "failed",
        (),
        ("ValueError: no",),
        None,
    )
    assert read.text() == "ValueError: no"
    assert read.is_error


@pytest.mark.parametrize(
    ("exit_code", "status"),
    [(None, ""), (0, ""), (2, "\n[exit 2]")],
)
def test_a_former_shell_result_reads_its_streams_and_a_failing_exit(
    exit_code: int | None,
    status: str,
) -> None:
    read = _formerly(
        "BashResult",
        command={"py/tuple": ["bash", "-c", "ls"]},
        stdout="out\n",
        stderr="err",
        **({} if exit_code is None else {"exit_code": exit_code}),
    )
    assert read.text() == f"out\nerr{status}"
    # Its streams hold what the command wrote, and how it exited is read after them.
    assert (read.stdout, read.stderr) == (("out\n",), ("err",))
    assert read.result == BashResult(command=("bash", "-c", "ls"), exit_code=exit_code)
    # How its command exited is no failure of the call, and was never marked one.
    assert (read.is_error, read.extra) == (False, {})
    # No shell marked itself an error; a row some other writer marked one keeps the
    # mark as a call that failed, and still says what ran.
    marked = _formerly("BashResult", is_error=True, stdout="out")
    assert (marked.state, marked.is_error, marked.result, marked.text()) == (
        "failed",
        True,
        BashResult(),
        "out",
    )
    assert former_kind(marked) == "BashResult"
    silent = _formerly("ShellCommandResult", exit_code=3)
    assert (silent.stdout, silent.stderr, silent.text()) == ((), (), "[exit 3]")
    empty = _formerly("BashResult", stdout="", stderr="")
    assert (empty.stdout, empty.stderr) == ((), ())
    # Only a shell record had an exit code; one on another record is no result's.
    stray = _formerly("ReadResult", path="a.py", exit_code=7)
    assert (stray.result, stray.extra) == (ReadResult(path="a.py"), {"exit_code": 7})


@pytest.mark.parametrize(
    ("exit_code", "note"),
    [(None, ""), (0, ""), (3, "\n[exit 3]"), (-9, "\n[exit -9]")],
    ids=["killed", "zero", "nonzero", "signal"],
)
def test_a_failing_command_s_exit_is_read_after_its_streams_and_held_in_neither(
    exit_code: int | None,
    note: str,
) -> None:
    ran = ToolMessage(
        call_id="1",
        state="done",
        stdout=("out\n",),
        stderr=("err",),
        result=BashResult(exit_code=exit_code),
    )

    assert ran.text() == f"out\nerr{note}"
    assert (ran.stdout, ran.stderr) == (("out\n",), ("err",))
    # Output that ends its last line is not given a blank line before the note.
    ended = replace(ran, stderr=("err\n",))
    assert ended.text() == f"out\nerr\n{note.removeprefix(chr(10))}"
    # What a prune or a compaction left is read as it is.
    assert ran.restated("[pruned]").text() == "[pruned]"


def test_only_a_bash_result_adds_its_exit_code_to_the_text() -> None:
    assert [
        ToolMessage(call_id="1", state="done", stdout=("out",), result=result).text()
        for result in (
            ReadResult(exit_code=3),
            UncategorizedToolResult(exit_code=3),
            BashResult(exit_code=3),
        )
    ] == ["out", "out", "out\n[exit 3]"]


def test_former_write_and_edit_results_read_as_their_summary_line() -> None:
    written = _formerly("WriteResult", path="a.py", content="the file's text")
    edit = {"py/object": "Splice", "before": "a", "after": "b", "start": 3}
    image = {"py/object": "Attachment", "mime_descriptor": "image/png", "data": ""}
    edited = _formerly(
        "EditResult",
        path="a.py",
        edits={"py/tuple": [edit]},
        attachments={"py/tuple": [image]},
    )

    assert (written.text(), written.stdout) == ("Wrote a.py.", ("Wrote a.py.",))
    assert written.result == WriteResult(path="a.py", content="the file's text")
    assert edited.text() == "Edited a.py."
    assert edited.stdout == (
        "Edited a.py.",
        Splice(before="a", after="b", start=3),
        Attachment(mime_descriptor="image/png", data=b""),
    )
    assert (written.extra, edited.extra) == ({}, {})
    assert _formerly("FileWriteResult").text() == "Wrote the file."
    assert _formerly("FileEditResult").text() == "Edited the file."


def test_a_former_web_search_reads_as_its_hits_title_url_snippet() -> None:
    """A ``WebSearchResults`` record tagged each hit ``WebSearchResult``."""
    hits = [
        {"py/object": "WebSearchHit", "url": "u1", "title": "t1", "snippet": "s1"},
        {"py/object": "WebSearchHit", "url": "u2"},
        {"py/object": "WebSearchHit", "title": "t3", "snippet": "s3"},
    ]
    first = [{**hit, "py/object": "WebSearchResult"} for hit in hits]

    read = _formerly("WebSearchResult", query="q", content={"py/tuple": hits})
    bare = [{k: v for k, v in hit.items() if k != "py/object"} for hit in hits]

    assert read.text() == "t1\nu1\ns1\n\nu2\n\nt3\ns3"
    assert read.stdout == (
        read.text(),
        WebSearchHit(url="u1", title="t1", snippet="s1"),
        WebSearchHit(url="u2"),
        WebSearchHit(title="t3", snippet="s3"),
    )
    assert _formerly("WebSearchResult", query="q", content=bare) == read
    assert _formerly("WebSearchResults", content={"py/tuple": first}).stdout == (
        read.stdout
    )
    assert _formerly("WebSearchResult").text() == ""
    image = {"py/object": "Attachment", "mime_descriptor": "image/png", "data": ""}
    pictured = _formerly("WebSearchResult", content=bare, attachments=[image])
    assert pictured.stdout == (
        *read.stdout,
        Attachment(mime_descriptor="image/png", data=b""),
    )


def test_a_former_result_keeps_its_attachments_after_its_text() -> None:
    image = {"py/object": "Attachment", "mime_descriptor": "image/png", "data": ""}
    read = _formerly(
        "ReadResult",
        path="a.png",
        content="[image: a.png]",
        ranges={"py/tuple": [{"py/tuple": [1, None]}]},
        attachments={"py/tuple": [image]},
    )

    assert read.stdout == (
        "[image: a.png]",
        LineSpan(start=1),
        Attachment(mime_descriptor="image/png", data=b""),
    )
    assert (read.result, read.extra) == (ReadResult(path="a.png"), {})


def test_a_former_record_converts_to_exactly_the_fields_it_had() -> None:
    assert current_plain(
        {"py/object": "ToolErrorResult", "call_id": "1", "content": ""},
    ) == {
        "py/object": "ToolMessage",
        "call_id": "1",
        "state": "failed",
        "stdout": [],
        "stderr": [],
    }


def test_plain_data_under_py_dict_is_never_read_as_a_record() -> None:
    held = {"py/object": "ShellCommandResult", "k": [{"py/object": "FunctionResult"}]}

    assert current_plain({"py/object": "ToolCall", "arguments": {"py/dict": held}}) == {
        "py/object": "ToolCall",
        "arguments": {"py/dict": held},
    }


@pytest.mark.parametrize("held", ["rs_1", ""])
def test_a_reasoning_id_once_kept_in_extra_reads_as_its_field(held: str) -> None:
    """Journals and stored rows written before the field name OpenAI's seal so."""
    plain = {
        "py/object": "Thinking",
        "encrypted": "gAAA",
        "extra": {"reasoning_id": held, "kept": 1},
    }

    assert stored_record(plain) == Thinking(
        encrypted="gAAA",
        reasoning_id=held,
        extra={"kept": 1},
    )
    assert stored_record(stored_plain(stored_record(plain))) == stored_record(plain)


def test_a_reasoning_id_stated_both_ways_keeps_the_field() -> None:
    plain = {
        "py/object": "Thinking",
        "reasoning_id": "rs_new",
        "extra": {"reasoning_id": "rs_old"},
    }

    assert stored_record(plain) == Thinking(reasoning_id="rs_new")


@pytest.mark.parametrize(
    ("held", "reasoning_id", "extra"),
    [
        (
            {"type": "reasoning", "id": "rs_1", "k": 1},
            "rs_1",
            {"type": "reasoning", "k": 1},
        ),
        ({"type": "reasoning"}, "", {"type": "reasoning", "$id_absent": True}),
        ({"type": "reasoning", "id": 7}, None, {"type": "reasoning", "id": 7}),
        ({"type": "thinking", "id": "x"}, None, {"type": "thinking", "id": "x"}),
    ],
    ids=["id", "no-id", "malformed-id", "not-reasoning"],
)
def test_a_reasoning_item_the_codex_reader_once_kept_whole_names_its_sealer(
    held: dict[str, PlainTree],
    reasoning_id: str | None,
    extra: dict[str, PlainTree],
) -> None:
    """The codex reader once kept a reasoning item's ``id`` in ``extra``."""
    plain = {"py/object": "Thinking", "encrypted": "gAAA", "extra": held}

    assert stored_record(plain) == Thinking(
        encrypted="gAAA",
        reasoning_id=reasoning_id,
        extra=extra,
    )


def test_only_reasoning_moves_an_id_out_of_extra() -> None:
    plain = {
        "py/object": "UserMessage",
        "content": "hi",
        "extra": {"type": "reasoning", "id": "x"},
    }

    assert stored_record(plain) == UserMessage(
        content="hi",
        extra={"type": "reasoning", "id": "x"},
    )


def test_a_reasoning_item_read_now_is_left_as_stored() -> None:
    plain = {
        "py/object": "Thinking",
        "reasoning_id": "",
        "extra": {"type": "reasoning", "$id_absent": True},
    }

    assert stored_record(plain) == Thinking(
        reasoning_id="",
        extra={"type": "reasoning", "$id_absent": True},
    )


def test_a_former_field_the_record_lacks_is_kept_in_extra_as_json() -> None:
    read = _formerly(
        "WebFetchResult",
        url="http://a",
        seen={"sizes": {"py/tuple": [1, 2]}, "hops": [{"py/tuple": ["a"]}, "b"]},
    )
    assert read.result == WebFetchResult(url="http://a")
    assert read.extra == {"seen": {"sizes": (1, 2), "hops": (("a",), "b")}}


@pytest.mark.parametrize(
    ("fields", "state", "result", "failed"),
    [
        ({"stub": "pending", "shown": "[running]"}, "pending", None, False),
        ({"stub": "cancelled", "is_error": True}, "cancelled", None, True),
        (
            {"stub": "pending", "is_error": True},
            "pending",
            UncategorizedToolResult(exit_code=1),
            True,
        ),
        (
            {"is_error": True, "shown": "no such file"},
            "done",
            UncategorizedToolResult(exit_code=1),
            True,
        ),
        ({}, "done", UncategorizedToolResult(), False),
    ],
    ids=["pending", "cancelled", "marked", "error", "plain"],
)
def test_a_former_placeholder_and_a_former_error_keep_their_state(
    fields: dict[str, object],
    state: str,
    result: UncategorizedToolResult | None,
    failed: bool,
) -> None:
    for tag in ("ToolResult", "UncategorizedToolResult"):
        read = _formerly(tag, **fields)
        assert (read.state, read.result, read.is_error) == (state, result, failed)
        assert (read.shown, read.extra) == (fields.get("shown"), {})
    # Only a done call has its tool's result class; another keeps the fields in ``extra``.
    typed = _formerly("ReadResult", path="a.py", **fields)
    assert (typed.state, typed.is_error) == (state, failed)
    assert isinstance(typed.result, ReadResult) is (state == "done")
    assert typed.extra == ({} if state == "done" else {"path": "a.py"})


@pytest.mark.parametrize("code", [0, 2, -9])
def test_a_stored_bare_exit_code_reads_as_an_uncategorized_tool_result(
    code: int,
) -> None:
    """``result`` was once the bare exit code of a tool with no result class."""
    stored = {
        "py/object": "ToolResult",
        "call_id": "1",
        "state": "done",
        "stdout": ["out"],
        "result": code,
    }
    expected = ToolMessage(
        call_id="1",
        state="done",
        stdout=("out",),
        result=UncategorizedToolResult(exit_code=code),
    )

    assert stored_record(stored) == expected
    assert stored_record(immutable(stored)) == expected
    cleared = stored_record({"py/object": "ContextClear", "history": [stored]})
    assert isinstance(cleared, ContextClear)
    assert cleared.history == (expected,)
    assert stored_record(stored_plain(expected)) == expected


_OLD_ROWS: Final = (
    {
        "py/object": "EditResult",
        "call_id": "1",
        "path": "a.py",
        "edits": [{"before": "a", "after": "b", "start": 3}],
    },
    {
        "py/object": "ReadResult",
        "call_id": "1",
        "content": "[image: a.png]",
        "ranges": [[1, None]],
        "attachments": [{"mime_descriptor": "image/png", "data": ""}],
    },
    {
        "py/object": "WebSearchResult",
        "call_id": "1",
        "content": [{"url": "u", "title": "t", "snippet": "s"}],
    },
    {
        "py/object": "ContextClear",
        "system_prompt": "S",
        "history": [{"py/object": "BashResult", "call_id": "1", "stdout": "x"}],
    },
)
"""Rows as a store kept them: each item untagged where its field named its class."""


def test_an_old_item_stored_untagged_reads_as_its_class() -> None:
    edited, read, searched, cleared = map(stored_record, _OLD_ROWS)

    assert isinstance(edited, ToolMessage)
    assert edited.stdout == ("Edited a.py.", Splice(before="a", after="b", start=3))
    assert isinstance(read, ToolMessage)
    assert read.stdout == (
        "[image: a.png]",
        LineSpan(start=1),
        Attachment(mime_descriptor="image/png", data=b""),
    )
    assert isinstance(searched, ToolMessage)
    assert searched.stdout == ("t\nu\ns", WebSearchHit(url="u", title="t", snippet="s"))
    assert isinstance(cleared, ContextClear)
    assert cleared.history == (
        ToolMessage(call_id="1", state="done", stdout=("x",), result=BashResult()),
    )


@pytest.mark.parametrize(
    "row",
    _OLD_ROWS,
    ids=[str(row["py/object"]) for row in _OLD_ROWS],
)
def test_an_old_row_reads_the_same_frozen_as_not(row: Mapping[str, object]) -> None:
    """A store hands its rows back frozen: each list a tuple, each dict read-only."""
    assert stored_record(immutable(row)) == stored_record(row)


@pytest.mark.parametrize("tag", sorted(FORMER_RESULT_KINDS))
def test_former_kind_of_a_legacy_record_is_its_class_name_after_renames(
    tag: str,
) -> None:
    """A store groups rows by kind, so a legacy row and a new one must match."""
    current_name = {"WebSearchResults": "WebSearchResult"}.get(
        tag,
        RENAMED.get(tag, tag),
    )
    assert former_kind(_formerly(tag)) == current_name


@pytest.mark.parametrize(
    ("result", "kind"),
    [
        (ToolMessage(call_id="1", state="done", result=BashResult()), "BashResult"),
        (
            ToolMessage(call_id="1", state="done", result=UncategorizedToolResult()),
            "UncategorizedToolResult",
        ),
        (
            ToolMessage(
                call_id="1",
                state="done",
                result=UncategorizedToolResult(exit_code=3),
            ),
            "UncategorizedToolResult",
        ),
        (ToolMessage(call_id="1", state="done"), "UncategorizedToolResult"),
        (ToolMessage(call_id="1", state="failed"), "ToolErrorResult"),
        (ToolMessage(call_id="1", state="pending"), "ToolMessage"),
        (ToolMessage(call_id="1", state="cancelled"), "ToolMessage"),
    ],
    ids=["bash", "zero", "nonzero", "no-result", "failed", "pending", "cancelled"],
)
def test_former_kind_is_the_result_class_name_or_follows_the_state(
    result: ToolMessage,
    kind: str,
) -> None:
    assert former_kind(result) == kind


def test_every_result_class_names_a_former_result_kind() -> None:
    """A store groups rows by ``former_kind`` and reads only ``FORMER_RESULT_KINDS``."""
    result_class_names = {
        name
        for name, kind in record_classes().items()
        if is_dataclass(kind)
        and any(field.name == "exit_code" for field in fields(kind))
    }
    assert result_class_names
    assert result_class_names <= FORMER_RESULT_KINDS


def test_record_classes_names_every_dataclass_the_module_defines_but_the_outcomes() -> (
    None
):
    classes = record_classes()
    members = dict[str, object](vars(types))
    defined = {
        name
        for name in types.__all__
        if isinstance(kind := members[name], type) and is_dataclass(kind)
    }
    # No record holds an outcome, nor how an entry pairs.
    assert set(classes) == defined - {"Succeeded", "Cancelled", "Failed", "Pairing"}
    assert all(classes[name] is members[name] for name in classes)


def test_an_outcome_is_one_of_three_endings() -> None:
    endings: list[Outcome[int]] = [
        Succeeded(value=1),
        Cancelled(),
        Failed(exception=RuntimeError("no")),
    ]

    assert [type(each) for each in endings] == [Succeeded, Cancelled, Failed]


def test_a_failure_holds_what_was_raised() -> None:
    error = RuntimeError("no")
    assert Failed(exception=error).exception is error


@pytest.mark.parametrize("state", [AgentIdle, ModelIdle])
def test_a_message_waiting_for_a_state_is_stored_and_read_back_by_its_name(
    state: type[AgentIdle | ModelIdle],
) -> None:
    held = UserMessage(content="later", due_after=state)

    plain = stored_plain(held)

    assert plain["due_after"] == {"py/type": state.__name__}
    assert stored_record(plain) == held


def test_the_model_s_idle_says_how_its_work_stopped() -> None:
    assert ModelIdle(outcome=Succeeded(value=None)) != ModelIdle(outcome=Cancelled())


@dataclass(frozen=True, slots=True, kw_only=True)
class _Turn:
    """A model turn held as one entry, as a context holds a reply."""

    records: tuple[AssistantMessage | Thinking | ToolCall, ...]


@pytest.mark.parametrize(
    ("entries", "late"),
    [
        # A context: each reply one entry.
        (
            [
                UserMessage(content="go"),
                _Turn(records=(ToolCall(call_id="c1", name="Bash"),)),
                ToolMessage(call_id="c1", state="pending", shown="[detached]"),
                _Turn(records=(AssistantMessage(content="waiting"),)),
                ToolMessage(
                    call_id="c1",
                    state="pending",
                    stdout=("[Bash c1 running]\nhalf",),
                ),
                ToolMessage(
                    call_id="c1",
                    state="failed",
                    stdout=("boom", Attachment(mime_descriptor="image/png", data=b"p")),
                ),
            ],
            {4, 5},
        ),
        # A log: a reply is a run of records; a note between them is no one's turn.
        (
            [
                UserMessage(content="go"),
                AssistantMessage(content="looking"),
                ToolCall(call_id="c0", name="Read"),
                TokenUsage(),
                ToolCall(call_id="c1", name="Bash"),
                ToolMessage(call_id="c0", state="done"),
                ToolMessage(call_id="c1", state="pending", shown="[detached]"),
                AssistantMessage(content="waiting"),
                ToolMessage(
                    call_id="c1",
                    state="failed",
                    stdout=("boom", Attachment(mime_descriptor="image/png", data=b"p")),
                ),
            ],
            {8},
        ),
    ],
    ids=["context", "log"],
)
def test_a_call_s_messages_after_its_first_answer_are_late(
    entries: list[SessionRecord | _Turn],
    late: set[int],
) -> None:
    assert late_messages(entries) == late


def test_a_message_of_a_call_no_entry_answered_is_late_only_if_orphans_are_news() -> (
    None
):
    entries = [
        UserMessage(content="[summary]"),
        ToolMessage(call_id="c1", state="pending", stdout=("[Bash c1 running]\nhalf",)),
        ToolMessage(
            call_id="c1",
            state="failed",
            stdout=("boom", Attachment(mime_descriptor="image/png", data=b"p")),
        ),
        ToolMessage(
            call_id="c1",
            state="failed",
            stdout=("boom", Attachment(mime_descriptor="image/png", data=b"p")),
        ),
    ]
    assert late_messages(entries) == {1, 2}
    assert late_messages(entries, orphans_are_news=False) == {2}
    # After its call's result, a message of the call is a repeat: no news.
    assert pairings(entries)[0][3].role == "repeat"


def test_each_entry_is_told_where_it_stands_among_the_calls() -> None:
    entries: list[SessionRecord | _Turn] = [
        _Turn(
            records=(ToolCall(call_id="a", name="R"), ToolCall(call_id="x", name="R")),
        ),
        SystemMessage(content="n"),
        ToolMessage(call_id="z", state="done"),
        AgentMessage(sender="peer", content="hi"),
        ToolMessage(call_id="a", state="done"),
        ToolMessage(call_id="x", state="done"),
        ToolMessage(call_id="a", state="done"),
        _Turn(records=(ToolCall(call_id="b", name="R"),)),
        UserMessage(content="stop"),
        _Turn(
            records=(ToolCall(call_id="a", name="R"), ToolCall(call_id="c", name="R")),
        ),
    ]

    found, still_open = pairings(entries, paired_externally=frozenset({"x"}))

    assert [p.role for p in found] == [
        "reply",
        "note",
        "orphan",
        "aside",
        "answer",
        "elsewhere",
        "repeat",
        "reply",
        "interrupt",
        "reply",
    ]
    assert [p.waiting for p in found][1:5] == [("a",)] * 4
    assert (found[8].left_open, found[9].reused, still_open) == (
        ("b",),
        ("a",),
        ("a", "c"),
    )
    # The next reply leaves what one before it left open unanswered.
    assert pairings([*entries, _Turn(records=())])[0][-1].left_open == ("a", "c")


def test_a_late_message_is_sent_as_the_user_s_text_headed_by_its_call() -> None:
    entries: list[SessionRecord] = [
        ToolCall(call_id="c1", name="Bash"),
        ToolMessage(call_id="c1", state="pending", shown="[detached]"),
        AssistantMessage(content="waiting"),
        ToolMessage(call_id="c1", state="pending", stdout=("[Bash c1 running]\nhalf",)),
        ToolMessage(
            call_id="c1",
            state="failed",
            stdout=("boom", Attachment(mime_descriptor="image/png", data=b"p")),
        ),
    ]
    assert as_sent(entries) == [
        *entries[:3],
        UserMessage(content="[Bash c1 running]\nhalf"),
        UserMessage(
            content="[call c1 failed]\nboom",
            attachments=(Attachment(mime_descriptor="image/png", data=b"p"),),
        ),
    ]
    assert as_sent(entries[:3]) == entries[:3]


@pytest.mark.parametrize(
    "record",
    _every_record(),
    ids=[type(r).__name__ for r in _every_record()],
)
def test_every_record_round_trips_through_its_stored_form(
    record: SessionRecord,
) -> None:
    text = json.dumps(stored_plain(record))
    assert text.startswith(f'{{"py/object": "{type(record).__name__}"')
    assert stored_record(json_loads(text)) == record


def _held(value: object) -> bool:
    """Return whether ``value`` holds no dict or list at any depth."""
    if isinstance(value, (dict, list)):
        return False
    if isinstance(value, Mapping):
        return all(_held(item) for item in cast("Mapping[str, object]", value).values())
    if isinstance(value, tuple):
        return all(_held(item) for item in cast("tuple[object, ...]", value))
    return True


# Built in the test, not at collection: a record made while pytest collects is frozen
# before any test runs, so no test is seen to reach ``__post_init__``.
@pytest.mark.parametrize(
    "index",
    range(len(_every_record())),
    ids=[type(r).__name__ for r in _every_record()],
)
def test_a_record_holds_its_json_frozen(index: int) -> None:
    record = _every_record()[index]
    values = [cast("object", getattr(record, item.name)) for item in fields(record)]
    json_fields: list[Mapping[object, object]] = [
        value for value in values if isinstance(value, Mapping)
    ]
    assert all(type(value) is MappingProxyType for value in json_fields)
    assert all(_held(value) for value in json_fields)


@pytest.mark.parametrize(
    "record",
    _every_record(),
    ids=[type(r).__name__ for r in _every_record()],
)
def test_every_record_stores_its_json_plain(record: SessionRecord) -> None:
    stored = stored_plain(record)

    text = json.dumps(stored)
    assert all(tag not in text for tag in ("py/dict", "py/list", "py/mappingproxy"))
    back = stored_record(stored)
    assert back == record
    assert is_dataclass(back)
    assert not isinstance(back, type)
    assert all(_held(cast("object", getattr(back, item.name))) for item in fields(back))


def test_a_stored_record_names_every_class_it_holds_alone() -> None:
    """A union member nested in history is found by its tag, so it carries a name too."""
    record = ContextClear(
        history=(
            UserMessage(content="u"),
            AssistantMessage(content="a"),
            ToolMessage(call_id="c", state="done", stdout=(Splice(after="y"),)),
        ),
    )

    stored = stored_plain(record)

    tags = re.findall(r'"py/object": "([^"]+)"', json.dumps(stored))
    assert tags == ["ContextClear", "AssistantMessage", "ToolMessage", "Splice"]
    assert stored_record(stored) == record


@dataclass(frozen=True, slots=True, kw_only=True)
class _Widened:
    text: str


def test_a_class_outside_the_vocabulary_stores_only_where_classes_name_it() -> None:
    """A caller widening the vocabulary names its classes; the default refuses them."""
    classes = {**record_classes(), "_Widened": _Widened}

    stored = stored_plain(_Widened(text="t"), classes=classes)

    assert stored == {"py/object": "_Widened", "text": "t"}
    assert stored_record(stored, _Widened, classes=classes) == _Widened(text="t")
    with pytest.raises(TypeError, match="_Widened"):
        _ = stored_plain(_Widened(text="t"))


def test_the_round_trip_cases_cover_every_record_kind() -> None:
    covered = {type(r).__name__ for r in _every_record()}
    named_by_due_after = {
        r.due_after.__name__
        for r in _every_record()
        if isinstance(r, (UserMessage, AgentMessage, SystemMessage, ToolMessage))
        and r.due_after is not None
    }
    inside_tool_results = {
        type(item).__name__
        for r in _every_record()
        if isinstance(r, ToolMessage)
        for item in (*r.stdout, r.result)
        if is_dataclass(item)
    }
    assert covered | inside_tool_results | named_by_due_after | {
        "RequestMeasurement",
    } == set(record_classes())


def test_a_session_preserves_interleaved_tool_lifecycles() -> None:
    # Two calls run at once, so their answers arrive out of order; only the
    # call id ties each result to the call that asked for it.
    records = (
        AssistantMessage(content="Starting both."),
        ToolCall(call_id="a", name="Read", arguments={"path": "/a"}),
        ToolMessage(
            call_id="a",
            state="pending",
            stdout=(Attachment(mime_descriptor="image/png", data=b"png"),),
        ),
        UserMessage(content="Also check B."),
        ToolCall(call_id="b", name="Read", arguments={"path": "/b"}),
        ToolMessage(
            call_id="b",
            state="done",
            stdout=("B",),
            result=UncategorizedToolResult(),
        ),
        ToolMessage(
            call_id="a",
            state="done",
            stdout=("A",),
            result=UncategorizedToolResult(),
        ),
    )

    restored = stored_record(stored_plain(ContextClear(history=records)), ContextClear)

    assert restored.history == records
    assert [
        item.call_id for item in restored.history if isinstance(item, ToolMessage)
    ] == ["a", "b", "a"]


def test_reasoning_is_a_sibling_of_the_reply_not_nested_in_it() -> None:
    # Axiom 3: both come off one source line, and both are their own record --
    # a nested one would be unreachable by the index a context names.
    thinking = Thinking(context_id=0, content="Check the file.", encrypted="sig")
    assistant = AssistantMessage(context_id=0, content="Found it.")
    user = UserMessage(context_id=0, content="Check this.")

    records = (user, thinking, assistant)
    restored = stored_record(stored_plain(ContextClear(history=records)), ContextClear)

    assert restored.history == records
    assert [type(record).__name__ for record in restored.history] == [
        "UserMessage",
        "Thinking",
        "AssistantMessage",
    ]
    assert not hasattr(restored.history[-1], "thinking")


def test_a_record_names_its_settings_by_index_rather_than_copying_them() -> None:
    # Axiom 5: settings live in one context, and a record points at it, so a
    # session that switches providers mid-stream has one context per turn.
    records = (
        AssistantMessage(context_id=0, content="First turn."),
        AssistantMessage(context_id=4, content="Second turn."),
    )

    restored = stored_record(stored_plain(ContextClear(history=records)), ContextClear)

    assert restored.history == records
    first, second = restored.history
    assert isinstance(first, AssistantMessage)
    assert isinstance(second, AssistantMessage)
    assert (first.context_id, second.context_id) == (0, 4)


def test_these_record_classes_are_renamed() -> None:
    assert dict(renamed_records()) == {
        "AgentToAgentMessage": "AgentMessage",
        "ToolResult": "ToolMessage",
        "TurnContext": "SessionContext",
    }
    # Each class bears one of its two names, never both and never neither.
    for old, new in renamed_records().items():
        assert (old in record_classes()) is not (new in record_classes())


@pytest.mark.parametrize(("old", "new"), sorted(renamed_records().items()))
def test_a_renamed_record_is_stored_under_both_names_and_first_under_the_old(
    old: str,
    new: str,
) -> None:
    assert stored_names(old) == (old, new)
    assert stored_names(new) == (new, old)
    assert (first_name(old), first_name(new)) == (old, old)


def test_a_record_never_renamed_is_stored_under_its_one_name() -> None:
    assert stored_names("UserMessage") == ("UserMessage",)
    assert first_name("UserMessage") == "UserMessage"


@pytest.mark.parametrize(("old", "new"), sorted(renamed_records().items()))
def test_a_former_record_tag_reads_as_the_new_one_once_a_class_bears_it(
    old: str,
    new: str,
) -> None:
    # Each has a state, as every tool call's record now has: one with none is
    # the placeholder of a former format, which is converted, not only renamed.
    held = {"py/object": old, "state": "done"}
    stored = {**held, "held": held}
    read = cast("dict[str, object]", current_plain(stored))
    assert (read["py/object"] == new) is (new in record_classes())
    # Told the names, it renames the tag wherever a record is held.
    assert current_plain(stored, {old: new}) == {
        "py/object": new,
        "state": "done",
        "held": {"py/object": new, "state": "done"},
    }


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
