"""Tests for the session records: what the model reads of each, and how each is stored."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import fields, is_dataclass
from types import MappingProxyType
from typing import cast

import json
import time

from treekle import from_plain, json_dumps, json_loads, to_plain

import pytest

from sagent.session import types
from sagent.session.types import (
    CONCRETE_TYPES,
    AgentStatusResult,
    AgentToAgentMessage,
    AssistantMessage,
    Attachment,
    ContextClear,
    ContextCompaction,
    ContextState,
    FileEditResult,
    FileReadResult,
    FileWriteResult,
    IncompleteRecord,
    SessionRecord,
    ShellCommandResult,
    Splice,
    SystemMessage,
    Thinking,
    TokenUsage,
    ToolCall,
    ToolResult,
    TurnContext,
    UncategorizedRecord,
    UncategorizedToolResult,
    UserMessage,
    WebFetchResult,
    WebSearchResult,
    WebSearchResults,
    record_classes,
)


def _every_record() -> list[SessionRecord]:
    return [
        TurnContext(cwd="/w", model="m", effort="high", encoding={"crlf": True}),
        UserMessage(
            content="hi",
            attachments=(Attachment(mime_descriptor="image/png", data=b"\x89PNG"),),
        ),
        AssistantMessage(content="hello", extra={"id": "a1"}),
        Thinking(content="hm", encrypted="sealed", summary="s"),
        ToolCall(call_id="1", name="Read", arguments={"path": "a.py", "n": [1, 2]}),
        UncategorizedToolResult(call_id="1", content="x", stub="pending"),
        ShellCommandResult(call_id="2", command=("ls", "-la"), stdout="o", exit_code=0),
        FileReadResult(call_id="3", path="a.py", content="c", ranges=((1, None),)),
        FileWriteResult(call_id="4", path="b.py", content="w"),
        FileEditResult(
            call_id="5",
            path="c.py",
            edits=(
                Splice(
                    before="a",
                    after="b",
                    start=3,
                    count=1,
                    bare=frozenset({"after"}),
                ),
            ),
        ),
        WebSearchResults(
            call_id="6",
            query="q",
            content=(WebSearchResult(url="u", title="t", snippet="s"),),
        ),
        WebFetchResult(call_id="7", url="u", content="page", code=200),
        AgentStatusResult(call_id="8", agent_id="a", content="done", tokens=5),
        SystemMessage(content="reminder", subtype="rules"),
        TokenUsage(info={"request": 1}, rate_limits={"5h": 0.5}),
        ContextState(kind="k", content="c"),
        ContextCompaction(summary="sum"),
        ContextClear(
            cleared_session_id="old",
            system_prompt="sys",
            summary="kept",
            history=(UserMessage(content="before"),),
        ),
        AgentToAgentMessage(content="ping", sender="a", recipient="b"),
        UncategorizedRecord(kind="custom", payload={"k": "v"}),
        IncompleteRecord(text="{torn"),
    ]


@pytest.fixture
def pacific(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Run in Los Angeles' time zone."""
    monkeypatch.setenv("TZ", "America/Los_Angeles")
    time.tzset()
    yield
    monkeypatch.undo()
    time.tzset()


@pytest.mark.usefixtures("pacific")
def test_turn_context_text_states_each_setting_on_its_own_line() -> None:
    context = TurnContext(
        cwd="/w",
        environment="Platform: linux",
        model="opus",
        timestamp="2026-10-23T00:00:00Z",
        agents_md="Be terse.",
    )
    assert context.text() == (
        "Working directory: /w\n"
        "Platform: linux\n"
        "Model: opus\n"
        "Current time: Thu 2026-10-22 17:00 -07:00 (PDT)\n"
        "Be terse."
    )


@pytest.mark.usefixtures("pacific")
@pytest.mark.parametrize(
    ("timestamp", "shown"),
    [
        ("2026-12-01T08:30:00+00:00", "Tue 2026-12-01 00:30 -08:00 (PST)"),
        ("2026-10-23T09:00:00", "2026-10-23T09:00:00"),
        ("yesterday", "yesterday"),
    ],
    ids=["winter", "no-offset", "no-time"],
)
def test_the_time_reads_where_the_user_is(timestamp: str, shown: str) -> None:
    assert TurnContext(timestamp=timestamp).text() == f"Current time: {shown}"


def test_turn_context_text_skips_what_it_does_not_state() -> None:
    assert TurnContext().text() == ""
    assert TurnContext(model="opus").text() == "Model: opus"


def test_a_bare_tool_result_reads_as_nothing() -> None:
    assert ToolResult(call_id="1").text() == ""
    assert UncategorizedToolResult(call_id="1").text() == ""
    assert UncategorizedToolResult(call_id="1", content="out").text() == "out"


@pytest.mark.parametrize(
    ("exit_code", "status"),
    [(None, ""), (0, ""), (2, "\n[exit 2]")],
)
def test_a_shell_result_reads_its_streams_and_a_failing_exit(
    exit_code: int | None,
    status: str,
) -> None:
    result = ShellCommandResult(
        call_id="1",
        stdout="out\n",
        stderr="err",
        exit_code=exit_code,
    )
    assert result.text() == f"out\nerr{status}"


def test_file_results_read_as_what_was_done() -> None:
    assert FileReadResult(call_id="1", content="body").text() == "body"
    assert FileReadResult(call_id="1").text() == ""
    assert (
        FileWriteResult(call_id="1", path="a.py", content="x").text() == "Wrote a.py."
    )
    assert FileEditResult(call_id="1", path="a.py").text() == "Edited a.py."


def test_search_hits_read_as_title_url_snippet_blocks() -> None:
    results = WebSearchResults(
        call_id="1",
        content=(
            WebSearchResult(url="u1", title="t1", snippet="s1"),
            WebSearchResult(url="u2", title="t2", snippet="s2"),
        ),
    )
    assert results.text() == "t1\nu1\ns1\n\nt2\nu2\ns2"
    assert WebSearchResults(call_id="1").text() == ""


def test_no_result_reads_none_for_what_its_source_did_not_say() -> None:
    hits = WebSearchResults(
        call_id="1",
        content=(WebSearchResult(url="u1"), WebSearchResult(title="t2", snippet="s2")),
    )
    assert hits.text() == "u1\n\nt2\ns2"
    assert FileWriteResult(call_id="1").text() == "Wrote the file."
    assert FileEditResult(call_id="1").text() == "Edited the file."


def test_fetch_and_agent_results_read_as_their_content() -> None:
    assert WebFetchResult(call_id="1", content="page").text() == "page"
    assert WebFetchResult(call_id="1").text() == ""
    assert AgentStatusResult(call_id="1", content="done").text() == "done"
    assert AgentStatusResult(call_id="1").text() == ""


def test_record_classes_names_every_dataclass_the_module_defines() -> None:
    classes = record_classes()
    members = dict[str, object](vars(types))
    defined = {
        name
        for name in types.__all__
        if isinstance(kind := members[name], type) and is_dataclass(kind)
    }
    assert set(classes) == defined
    assert all(classes[name] is members[name] for name in classes)


@pytest.mark.parametrize(
    "record",
    _every_record(),
    ids=[type(r).__name__ for r in _every_record()],
)
def test_every_record_round_trips_through_its_stored_form(
    record: SessionRecord,
) -> None:
    classes = record_classes()
    text = json_dumps(record, classes=classes, sort=False)
    assert f'"{type(record).__name__}"' in text
    assert (
        cast(SessionRecord, json_loads(text, SessionRecord, classes=classes)) == record
    )


def _held(value: object) -> bool:
    """Return whether ``value`` holds no dict or list at any depth."""
    if isinstance(value, (dict, list)):
        return False
    if isinstance(value, Mapping):
        return all(_held(item) for item in cast("Mapping[str, object]", value).values())
    if isinstance(value, tuple):
        return all(_held(item) for item in cast("tuple[object, ...]", value))
    return True


@pytest.mark.parametrize(
    "record",
    _every_record(),
    ids=[type(r).__name__ for r in _every_record()],
)
def test_a_record_holds_its_json_frozen(record: SessionRecord) -> None:
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
    stored = to_plain(
        record,
        sort=False,
        default_concrete_types=CONCRETE_TYPES,
    )

    text = json.dumps(stored)
    assert all(tag not in text for tag in ("py/dict", "py/list", "py/mappingproxy"))
    back = from_plain(stored, SessionRecord, default_concrete_types=CONCRETE_TYPES)
    assert back == record
    assert is_dataclass(back)
    assert not isinstance(back, type)
    assert all(_held(cast("object", getattr(back, item.name))) for item in fields(back))


def test_the_round_trip_cases_cover_every_record_kind() -> None:
    covered = {type(r).__name__ for r in _every_record()}
    containers = {"Attachment", "Splice", "WebSearchResult", "ToolResult"}
    assert covered | containers == set(record_classes())


def test_a_session_preserves_interleaved_tool_lifecycles() -> None:
    # Two calls run at once, so their answers arrive out of order; only the
    # call id ties each result to the call that asked for it.
    records = (
        AssistantMessage(content="Starting both."),
        ToolCall(call_id="a", name="Read", arguments={"path": "/a"}),
        UncategorizedToolResult(
            call_id="a",
            attachments=(Attachment(mime_descriptor="image/png", data=b"png"),),
        ),
        UserMessage(content="Also check B."),
        ToolCall(call_id="b", name="Read", arguments={"path": "/b"}),
        FileReadResult(call_id="b", content="B"),
        FileReadResult(call_id="a", content="A"),
    )

    assert [item.call_id for item in records if isinstance(item, ToolResult)] == [
        "a",
        "b",
        "a",
    ]


def test_reasoning_is_a_sibling_of_the_reply_not_nested_in_it() -> None:
    # Axiom 3: both come off one source line, and both are their own record --
    # a nested one would be unreachable by the index a context names.
    thinking = Thinking(context_id=0, content="Check the file.", encrypted="sig")
    assistant = AssistantMessage(context_id=0, content="Found it.")
    user = UserMessage(context_id=0, content="Check this.")

    records = (user, thinking, assistant)

    assert [type(record).__name__ for record in records] == [
        "UserMessage",
        "Thinking",
        "AssistantMessage",
    ]
    assert not hasattr(assistant, "thinking")


def test_a_record_names_its_settings_by_index_rather_than_copying_them() -> None:
    # Axiom 5: settings live in one context, and a record points at it, so a
    # session that switches providers mid-stream has one context per turn.
    records = (
        AssistantMessage(context_id=0, content="First turn."),
        AssistantMessage(context_id=4, content="Second turn."),
    )

    assert [record.context_id for record in records] == [0, 4]


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
