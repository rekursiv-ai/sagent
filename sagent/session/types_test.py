"""Tests for the session records: what the model reads of each, and how each is stored."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, fields, is_dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Final, cast

import json
import re
import time

from treekle import PlainTree, ReadError, json_loads

import pytest

from sagent.session import types
from sagent.session.types import (
    AgentSpawnResult,
    AgentToAgentMessage,
    AssistantMessage,
    Attachment,
    BashResult,
    ContextClear,
    ContextCompaction,
    ContextState,
    EditResult,
    IncompleteRecord,
    ReadResult,
    SessionRecord,
    Splice,
    SystemMessage,
    Thinking,
    TokenUsage,
    ToolCall,
    ToolErrorResult,
    ToolResult,
    TurnContext,
    UncategorizedRecord,
    UncategorizedToolResult,
    UserMessage,
    WebFetchResult,
    WebSearchHit,
    WebSearchResult,
    WriteResult,
    record_classes,
    stored_plain,
    stored_record,
)


_CWD: Final = Path(__file__).resolve().parent


def _json() -> dict[str, PlainTree]:
    """Return a JSON field as a provider sends it: plain dicts and lists, nested."""
    return {"k": [1, {"n": 2}]}


# Every JSON field is given plain, so each record's own ``__post_init__`` must
# freeze it: a default is frozen already, and would show nothing.
def _every_record() -> list[SessionRecord]:
    return [
        TurnContext(
            cwd="/w",
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
        UncategorizedToolResult(
            call_id="1",
            content="x",
            stub="pending",
            extra=_json(),
        ),
        BashResult(call_id="2", command=("ls", "-la"), stdout="o", exit_code=0),
        ReadResult(call_id="3", path="a.py", content="c", ranges=((1, None),)),
        WriteResult(call_id="4", path="b.py", content="w"),
        EditResult(
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
        WebSearchResult(
            call_id="6",
            query="q",
            content=(WebSearchHit(url="u", title="t", snippet="s"),),
        ),
        WebFetchResult(call_id="7", url="u", content="page", code=200),
        AgentSpawnResult(call_id="8", agent_id="a", content="done", tokens=5),
        ToolErrorResult(call_id="9", content="ValueError: no", shown="[pruned]"),
        ToolResult(call_id="10", stub="pending", shown="[running]", is_error=True),
        types.GrepResult(call_id="11", pattern="p", path="src", content="a.py"),
        types.GlobResult(call_id="12", pattern="*.py", content="a.py"),
        types.ListResult(call_id="13", path="src", content="a.py"),
        types.SkillResult(call_id="14", skill="tdd", content="body"),
        types.AgentSendResult(call_id="15", to="parent", delay_sec=5, content="ok"),
        types.BackgroundTaskResult(call_id="16", operation="list", content="none"),
        types.AgentSelfResult(call_id="17", content="model: x"),
        types.SessionResult(call_id="18", action="read", agent="a", content="r"),
        types.MemoryResult(call_id="19", action="save", name="n", content="saved"),
        types.PlayAudioResult(call_id="20", path="a.wav", content="Played"),
        types.PaperSearchResult(call_id="21", query="q", source="s2", content="p"),
        types.PaperDetailsResult(call_id="22", operation="references", content="p"),
        types.PaperAuthorResult(call_id="23", operation="papers", content="p"),
        types.PaperFetchResult(call_id="24", content="a.pdf"),
        types.RemoteToolResult(call_id="25", endpoint="e", content="ok"),
        types.McpResult(call_id="26", server="s", tool="t", content="ok"),
        types.FunctionResult(call_id="27", name="act", content="acted"),
        SystemMessage(content="reminder", subtype="rules", extra=_json()),
        TokenUsage(info={"request": 1}, rate_limits={"5h": 0.5}, extra=_json()),
        ContextState(kind="k", content="c", extra={"full": [True]}),
        ContextCompaction(summary="sum", extra={"tokens": {"before": 9}}),
        ContextClear(
            cleared_session_id="old",
            system_prompt="sys",
            summary="kept",
            history=(UserMessage(content="before"),),
            extra=_json(),
        ),
        AgentToAgentMessage(content="ping", sender="a", recipient="b", extra=_json()),
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
    assert TurnContext(environment="", agents_md="").text() == ""


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
    assert TurnContext(tools=tools).text() == shown


def test_a_stored_record_keeps_its_fields_in_declaration_order() -> None:
    """``timestamp`` is declared before ``content``, though it sorts after it."""
    stored = stored_plain(UserMessage(timestamp="t", content="hi"))
    assert [key for key in stored if not key.startswith("py/")] == [
        "timestamp",
        "content",
        "extra",
    ]


def test_a_stored_record_is_read_as_the_class_asked_for() -> None:
    with pytest.raises(ReadError):
        _ = stored_record(stored_plain(UserMessage(content="hi")), ToolCall)


def test_a_bare_tool_result_reads_as_nothing() -> None:
    assert ToolResult(call_id="1").text() == ""
    assert UncategorizedToolResult(call_id="1").text() == ""
    assert UncategorizedToolResult(call_id="1", content="out").text() == "out"


def test_what_a_result_shows_stands_in_for_its_body_its_fields_kept() -> None:
    """A prune or a stand-in sets ``shown``; the type and its fields stay."""
    read = ReadResult(call_id="1", path="a.py", content="body")
    pruned = replace(read, shown="[pruned]")

    assert (read.text(), pruned.text()) == ("body", "[pruned]")
    assert (pruned.path, pruned.content) == ("a.py", "body")
    assert ToolResult(call_id="2", stub="pending", shown="[running]").text() == (
        "[running]"
    )
    assert ToolResult(call_id="3", shown="").text() == ""


def test_an_error_no_tool_answered_reads_as_its_content() -> None:
    assert ToolErrorResult(call_id="1", content="ValueError: no").text() == (
        "ValueError: no"
    )


def _tool_result_classes() -> set[type[ToolResult]]:
    """Return ``ToolResult`` and every subclass of it the module defines."""
    members = dict[str, object](vars(types))
    return {
        kind
        for name in types.__all__
        if isinstance(kind := members[name], type) and issubclass(kind, ToolResult)
    }


@pytest.mark.parametrize(
    "kind",
    types.tool_result_classes(),
    ids=[kind.__name__ for kind in types.tool_result_classes()],
)
def test_a_restated_result_keeps_its_type_and_fields_and_drops_attachments(
    kind: type[ToolResult],
) -> None:
    image = Attachment(mime_descriptor="image/png", data=b"png")
    result = kind(call_id="1", is_error=True, attachments=(image,))

    restated = result.restated("[pruned]")

    assert type(restated) is kind
    assert restated == replace(result, shown="[pruned]", attachments=())
    assert restated.text() == "[pruned]"


@pytest.mark.parametrize(
    ("old", "kind"),
    [
        ("BashResult", types.BashResult),
        ("ReadResult", types.ReadResult),
        ("WriteResult", types.WriteResult),
        ("EditResult", types.EditResult),
        ("AgentSpawnResult", types.AgentSpawnResult),
    ],
)
def test_a_record_stored_under_its_old_name_reads_as_the_renamed_type(
    old: str,
    kind: type[ToolResult],
) -> None:
    assert stored_record({"py/object": old, "call_id": "1"}) == kind(call_id="1")
    assert json.dumps(stored_plain(kind(call_id="1"))).startswith(
        f'{{"py/object": "{kind.__name__}"',
    )


def test_an_old_web_search_reads_as_one_result_of_hits() -> None:
    """Its old name for the results is the name of one hit now; hits nest only there."""
    old = {
        "py/object": "WebSearchResults",
        "call_id": "1",
        "content": {
            "py/tuple": [
                {"py/object": "WebSearchResult", "title": "t", "url": "u"},
            ],
        },
    }

    assert stored_record(old) == types.WebSearchResult(
        call_id="1",
        content=(types.WebSearchHit(title="t", url="u"),),
    )


# Each tool's result whose only reading is the text it holds.
_TEXT_RESULTS = tuple(
    kind
    for kind in types.tool_result_classes()
    if any(f.name == "content" and f.default == "" for f in fields(kind))
)


@pytest.mark.parametrize(
    "kind",
    _TEXT_RESULTS,
    ids=[kind.__name__ for kind in _TEXT_RESULTS],
)
def test_a_text_result_reads_as_the_text_it_holds(kind: type[ToolResult]) -> None:
    result = stored_record(
        {"py/object": kind.__name__, "call_id": "1", "content": "what the model read"},
    )

    assert isinstance(result, ToolResult)
    assert type(result) is kind
    assert result.text() == "what the model read"


def test_every_tool_result_type_is_in_the_union_and_stored_by_name() -> None:
    """A type left out of either is one a match or a journal silently misses."""
    union = set[type[ToolResult]](types.tool_result_classes())
    classes = record_classes()

    assert _tool_result_classes() == union
    assert all(classes[kind.__name__] is kind for kind in union)


@pytest.mark.parametrize(
    ("exit_code", "status"),
    [(None, ""), (0, ""), (2, "\n[exit 2]")],
)
def test_a_shell_result_reads_its_streams_and_a_failing_exit(
    exit_code: int | None,
    status: str,
) -> None:
    result = BashResult(
        call_id="1",
        stdout="out\n",
        stderr="err",
        exit_code=exit_code,
    )
    assert result.text() == f"out\nerr{status}"


def test_file_results_read_as_what_was_done() -> None:
    assert ReadResult(call_id="1", content="body").text() == "body"
    assert ReadResult(call_id="1").text() == ""
    assert WriteResult(call_id="1", path="a.py", content="x").text() == "Wrote a.py."
    assert EditResult(call_id="1", path="a.py").text() == "Edited a.py."


def test_search_hits_read_as_title_url_snippet_blocks() -> None:
    results = WebSearchResult(
        call_id="1",
        content=(
            WebSearchHit(url="u1", title="t1", snippet="s1"),
            WebSearchHit(url="u2", title="t2", snippet="s2"),
        ),
    )
    assert results.text() == "t1\nu1\ns1\n\nt2\nu2\ns2"
    assert WebSearchResult(call_id="1").text() == ""


def test_no_result_reads_none_for_what_its_source_did_not_say() -> None:
    hits = WebSearchResult(
        call_id="1",
        content=(WebSearchHit(url="u1"), WebSearchHit(title="t2", snippet="s2")),
    )
    assert hits.text() == "u1\n\nt2\ns2"
    assert WriteResult(call_id="1").text() == "Wrote the file."
    assert EditResult(call_id="1").text() == "Edited the file."


def test_fetch_and_agent_results_read_as_their_content() -> None:
    assert WebFetchResult(call_id="1", content="page").text() == "page"
    assert WebFetchResult(call_id="1").text() == ""
    assert AgentSpawnResult(call_id="1", content="done").text() == "done"
    assert AgentSpawnResult(call_id="1").text() == ""


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
            EditResult(call_id="c", edits=(Splice(after="y"),)),
        ),
    )

    stored = stored_plain(record)

    tags = re.findall(r'"py/object": "([^"]+)"', json.dumps(stored))
    assert tags == ["ContextClear", "AssistantMessage", "EditResult"]
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
    containers = {"Attachment", "Splice", "WebSearchHit"}
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
        ReadResult(call_id="b", content="B"),
        ReadResult(call_id="a", content="A"),
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
