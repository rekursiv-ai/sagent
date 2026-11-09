"""Tests for the wire: each message crosses as one line and reads back equal."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from typing import (
    TYPE_CHECKING,
    Final,
    TypeAliasType,
    cast,
    get_args,
    get_type_hints,
)
from unittest.mock import AsyncMock, Mock, call, patch

import asyncio
import contextlib
import json
import re
import secrets

from treekle import PlainTree, json_dumps, json_loads
from wesearch.fetch.gate import Admission, GatePolicy

import pytest

from sagent.host.wire import (
    MAX_LINE_BYTES,
    VERSION,
    Ack,
    Bye,
    Call,
    Cancel,
    DaemonControl,
    DaemonStatus,
    Describe,
    Described,
    GateAnswer,
    GateCall,
    Hello,
    Message,
    Note,
    Pause,
    Paused,
    Peers,
    Ping,
    Refused,
    Result,
    Resume,
    Tools,
    ToolsChanged,
    ToolSpec,
    UnreadLineError,
    Welcome,
    _implied,
    decoded,
    encoded,
    ping,
    received,
)
from sagent.session.types import (
    Attachment,
    BashResult,
    EditResult,
    ReadResult,
    Splice,
    UserMessage,
    WebSearchHit,
    WebSearchResult,
    record_classes,
    stored_plain,
)
from sagent.tools.read import Read
from sagent.tools.types import checked


if TYPE_CHECKING:
    from collections.abc import Callable


_SPEC = ToolSpec(description="Run it.", schema={"type": "object", "required": ["x"]})
_TOKEN = secrets.token_urlsafe(16)


@pytest.mark.parametrize(
    "message",
    [
        DaemonControl(
            source_version="source",
            listen=("127.0.0.1:0",),
            tools=("Read",),
            ttl_sec=90,
        ),
        DaemonStatus(pid=1, source_version="source", listen=(), restarting=True),
        Hello(version=VERSION, token=_TOKEN, session_id="s", client="c"),
        Hello(
            version=VERSION,
            token=_TOKEN,
            session_id="s",
            client="c",
            tools=("Read",),
        ),
        Welcome(
            name="gpu1",
            resumed=True,
            calls=("1", "2"),
            tools={"Bash": _SPEC.digest},
        ),
        Refused(reason="no"),
        Refused(reason="withdrawn", id="note"),
        Pause(),
        Paused(),
        Resume(),
        Peers(labels=("kid", "private"), public=("kid",)),
        ToolsChanged(tools={}, peer_messaging=False),
        Call(
            id="1",
            agent="a",
            window="w",
            tool="Bash",
            arguments={"command": "ls", "n": [1, 2]},
        ),
        Call(
            id="2",
            agent="a",
            window="w",
            tool="AgentSpawn",
            arguments={"prompt": "go"},
            address_book=("a", "reviewer"),
            granted=("AgentSpawn", "Read"),
        ),
        Note(id="n1", sender="a", recipient="a_0", content="more?", kind="ask"),
        Note(
            id="n2",
            sender="a_0",
            recipient="a",
            content="[a_0 is idle]",
            kind="tell",
        ),
        Peers(labels=("a_0", "a_1")),
        Result(
            id="1",
            record=BashResult(call_id="", stdout="hi\n", exit_code=0),
        ),
        Result(
            id="2",
            record=ReadResult(
                call_id="",
                path="/a.png",
                attachments=(
                    Attachment(mime_descriptor="image/png", data=b"\x00\xff"),
                ),
            ),
        ),
        Result(id="3", record=ReadResult(call_id="", wire_id="3", path="/a")),
        Ack(id="1"),
        Cancel(id="1"),
        Bye(),
        Ping(),
        Tools(tools={"Read": _SPEC}),
        ToolsChanged(tools={"Read": _SPEC.digest}),
        Describe(id="d1", tool="Read"),
        Described(id="d1", spec=_SPEC),
        Described(id="d2", spec=None),
    ],
)
@pytest.mark.asyncio
async def test_each_message_reads_back_as_it_was_sent(message: Message) -> None:
    reader = asyncio.StreamReader()
    reader.feed_data(encoded(message))
    reader.feed_eof()
    back = await received(reader)
    assert back == message
    assert await received(reader) is None


@pytest.mark.asyncio
async def test_a_message_is_one_line() -> None:
    line = encoded(Result(id="1", record=BashResult(call_id="", stdout="a\nb")))
    assert line.endswith(b"\n")
    assert line.count(b"\n") == 1


@pytest.mark.asyncio
async def test_a_session_record_is_no_message() -> None:
    reader = asyncio.StreamReader()
    reader.feed_data(
        json_dumps(stored_plain(UserMessage(content="hi"))).encode() + b"\n",
    )
    with pytest.raises(ValueError, match="UserMessage"):
        _ = await received(reader)


@pytest.mark.asyncio
async def test_a_line_nested_too_deep_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # How deep overflows the parser's stack depends on the platform.
    def overflowing(_: bytes) -> None:
        raise RecursionError

    monkeypatch.setattr("sagent.host.wire.json_loads", overflowing)
    reader = asyncio.StreamReader()
    reader.feed_data(b"[[[\n")
    with pytest.raises(ValueError, match="nested too deep"):
        _ = await received(reader)


@pytest.mark.asyncio
async def test_a_line_that_is_no_json_is_refused() -> None:
    reader = asyncio.StreamReader()
    reader.feed_data(b"not json\n")
    with pytest.raises(ValueError, match="Expecting value"):
        _ = await received(reader)


_DEEP = b"[" * 100_000 + b"]" * 100_000


@pytest.mark.parametrize(
    ("line", "named"),
    [
        (b'{"py/object": "Result", "id": "7", "record": 5}', "7"),
        (b'{"py/object": "ResultOfANewerSagent", "id": "7"}', "7"),
        (b'{"py/object": "Result", "id": 7, "record": 5}', None),
        (b'["id", "7"]', None),
        (b"not json", None),
        (_DEEP, None),
    ],
)
@pytest.mark.asyncio
async def test_a_line_that_is_no_message_names_the_id_it_holds_as_plain_json(
    line: bytes,
    named: str | None,
) -> None:
    """So the reader can answer for that id, and go on to the next line."""
    reader = asyncio.StreamReader(limit=MAX_LINE_BYTES)
    reader.feed_data(line + b"\n" + encoded(Ping()))
    with pytest.raises(UnreadLineError) as raised:
        _ = await received(reader)
    assert raised.value.id == named
    assert isinstance(raised.value.__cause__, ValueError)
    assert str(raised.value) == str(raised.value.__cause__)
    assert await received(reader) == Ping()


@pytest.mark.asyncio
async def test_a_line_over_the_readers_limit_names_no_id() -> None:
    reader = asyncio.StreamReader(limit=8)
    reader.feed_data(b'{"id": "7"}\n')
    with pytest.raises(ValueError, match="limit") as raised:
        _ = await received(reader)
    assert not isinstance(raised.value, UnreadLineError)


_CALL = Call(
    id="1",
    agent="a",
    window="w",
    tool="Bash",
    arguments={"command": "ls"},
    address_book=("a",),
)
_RESULT = Result(
    id="1",
    record=ReadResult(
        call_id="",
        path="/a.png",
        attachments=(Attachment(mime_descriptor="image/png", data=b"\x00"),),
    ),
)
_NOTE = Note(id="n1", sender="a", recipient="b", content="hi", kind="tell")
_HELLO = Hello(version=VERSION, token="", session_id="s", client="c", tools=("Read",))
_ADDED: Final = {"py/object": "AddedByANewerSagent", "deep": [1, "x"]}
"""A field a newer sagent writes, holding a class this one does not know."""


type _Plain = dict[str, object]


def _top(plain: _Plain) -> _Plain:
    return plain


def _record(plain: _Plain) -> _Plain:
    return cast("_Plain", plain["record"])


def _first(field: str) -> Callable[[_Plain], _Plain]:
    def first(plain: _Plain) -> _Plain:
        return cast("list[_Plain]", _record(plain)[field])[0]

    return first


def _field(field: str) -> Callable[[_Plain], _Plain]:
    def inner(plain: _Plain) -> _Plain:
        return cast("_Plain", plain[field])

    return inner


def _with_added(message: Message, at: Callable[[_Plain], _Plain]) -> bytes:
    """Return ``message``'s line, an unknown field added at ``at``."""
    plain = cast("_Plain", json.loads(encoded(message)))
    at(plain)["added_later"] = _ADDED
    return (json.dumps(plain) + "\n").encode()


@pytest.mark.parametrize(
    ("message", "at"),
    [
        (_CALL, _top),
        (_RESULT, _top),
        (Welcome(name="gpu1", resumed=True, calls=("1",), tools={}), _top),
        (_RESULT, _record),
        (_RESULT, _first("attachments")),
        (
            Result(id="1", record=EditResult(call_id="", edits=(Splice(after="x"),))),
            _first("edits"),
        ),
        (
            Result(
                id="1",
                record=WebSearchResult(call_id="", content=(WebSearchHit(url="u"),)),
            ),
            _first("content"),
        ),
        (Described(id="d1", spec=_SPEC), _field("spec")),
        (GateCall(op="trip", policy=GatePolicy()), _field("policy")),
        (GateAnswer(admission=Admission()), _field("admission")),
    ],
)
def test_a_field_a_newer_sagent_added_is_ignored(
    message: Message,
    at: Callable[[_Plain], _Plain],
) -> None:
    back = decoded(_with_added(message, at))
    assert back == message
    # What is written is what was written before, the field not kept.
    assert encoded(back) == encoded(message)


def test_what_a_message_is_written_as_is_unchanged() -> None:
    assert encoded(_CALL) == (
        b'{"py/object": "Call", "id": "1", "agent": "a", "window": "w", '
        b'"tool": "Bash", "arguments": {"py/dict": {"command": "ls"}}, '
        b'"address_book": ["a"]}\n'
    )
    assert encoded(_RESULT) == (
        b'{"py/object": "Result", "id": "1", "record": {"py/object": "ReadResult", '
        b'"path": "/a.png", "call_id": "", "attachments": [{"mime_descriptor": '
        b'"image/png", "data": {"py/b64": "AA=="}}], "extra": {}}}\n'
    )


def test_a_field_of_plain_data_is_kept_whatever_it_holds() -> None:
    """Arguments are the model's; what looks like a record there is data."""
    held = {"py/object": "BashResult", "call_id": "", "added_later": 1}
    call = Call(id="1", agent="a", window="w", tool="Bash", arguments={"a": held})
    assert decoded(encoded(call)) == call


@pytest.mark.parametrize(
    "change",
    [
        {"py/object": "MessageOfANewerSagent"},
        {"py/object": ["Call"]},
        {"record": {"py/object": "ResultOfANewerSagent", "call_id": ""}},
        {
            "record": {
                "py/object": "ReadResult",
                "call_id": "",
                "attachments": [{"py/object": "AttachmentOfANewerSagent"}],
            },
        },
        {"record": {"py/object": "ReadResult", "call_id": "", "attachments": 5}},
        {"record": {"py/object": "ReadResult", "call_id": "", "attachments": [5]}},
        {"record": {"py/object": "ReadResult", "call_id": "", "attachments": {}}},
    ],
)
def test_an_unknown_kind_or_class_or_a_misshapen_line_is_refused(
    change: dict[str, object],
) -> None:
    plain = cast("_Plain", json.loads(encoded(_RESULT)))
    with pytest.raises(ValueError, match=r"cannot read|missing|expected"):
        _ = decoded((json.dumps({**plain, **change}) + "\n").encode())


_RECORD: Final = {"py/object": "Ping"}
"""A record no text field may hold, which treekle builds below the root all the same."""


@pytest.mark.parametrize(
    ("message", "field", "value"),
    [
        (_CALL, "id", _RECORD),
        (_CALL, "agent", {"py/path": "/etc/passwd"}),
        (_CALL, "address_book", [_RECORD]),
        (_CALL, "granted", ["Read", {"py/object": "DaemonControl", "stop": True}]),
        (_NOTE, "sender", _RECORD),
        (_NOTE, "kind", _RECORD),
        (_HELLO, "tools", [_RECORD]),
        (_HELLO, "version", {"py/path": "/1"}),
        (_RESULT, "id", {"py/b64": "AA=="}),
        (Peers(labels=("a",)), "public", {"py/tuple": [_RECORD]}),
    ],
)
def test_a_field_holding_what_its_type_is_not_is_refused(
    message: Message,
    field: str,
    value: object,
) -> None:
    plain = cast("_Plain", json.loads(encoded(message)))
    line = (json.dumps({**plain, field: value}) + "\n").encode()
    with pytest.raises(ValueError, match=rf"^'{type(message).__name__}\.{field}' must"):
        _ = decoded(line)


def test_a_record_inside_a_message_is_checked_field_by_field() -> None:
    plain = cast("_Plain", json.loads(encoded(_RESULT)))
    record = cast("_Plain", plain["record"])
    line = json.dumps({**plain, "record": {**record, "path": _RECORD}}) + "\n"
    with pytest.raises(ValueError, match=r"^'Result\.record\.path' must be string"):
        _ = decoded(line.encode())


def test_a_line_nested_too_deep_in_its_records_is_refused() -> None:
    depth = 100_000
    line = encoded(_RESULT).replace(
        b'"attachments": [',
        b'"attachments": [' + b"[" * depth + b"]" * depth + b", ",
    )
    with pytest.raises(ValueError, match=r"nested too deep|cannot read"):
        _ = decoded(line)


_MESSAGES = Message.__value__  # pyright: ignore[reportAny] -- Typeshed types an alias's value as Any.


_LINE_CLASSES: Final = [
    *cast("tuple[type, ...]", get_args(_MESSAGES)),
    *record_classes().values(),
    ToolSpec,
    GatePolicy,
    Admission,
]
"""Every class a line may name."""


@pytest.mark.parametrize("kind", _LINE_CLASSES)
def test_the_fields_of_each_class_a_line_may_name_resolve(kind: type) -> None:
    """The reader looks up each class's fields' types to know what it holds."""
    _ = get_type_hints(kind)


class _Unnamed:
    """A value no field's type names."""


@pytest.mark.parametrize("kind", _LINE_CLASSES)
def test_each_field_a_line_may_name_is_of_a_type_the_reader_checks(kind: type) -> None:
    """A type ``checked`` does not know fails every line, not only a misfit's."""
    assert is_dataclass(kind)
    hints = cast("dict[str, object]", get_type_hints(kind, include_extras=True))
    for f in fields(kind):
        with contextlib.suppress(ValueError):
            _ = checked(_Unnamed(), hints[f.name], key=f.name)


_SENT_BY_ITS_OWN_HOST: Final = frozenset({("Tools", "tools")})
"""Fields the reader need not see into: only a host the server started itself sends
them, of the server's own version."""


def _records_in(hint: object, seen: frozenset[object] = frozenset()) -> set[type]:
    """Return every record class ``hint`` names, however deep."""
    if isinstance(hint, TypeAliasType):
        if hint in seen:
            return set()
        seen, hint = seen | {hint}, cast(object, hint.__value__)
    found = {hint} if isinstance(hint, type) and is_dataclass(hint) else set[type]()
    for arg in cast("tuple[object, ...]", get_args(hint)):
        found |= _records_in(arg, seen)
    return found


# Treekle writes a record untagged where its field's type names its class alone, so the
# reader must know that class to drop a newer sagent's fields from it. A field of
# several classes is tagged, and read by its tag.
def _unseen(kind: type) -> list[str]:
    """Return each field of ``kind`` that holds one record class the reader cannot see."""
    return [
        f"{kind.__name__}.{name}"
        for name, hint in cast("dict[str, object]", get_type_hints(kind)).items()
        if (kind.__name__, name) not in _SENT_BY_ITS_OWN_HOST
        and len(held := _records_in(hint)) == 1
        and {_implied(hint)} != held
    ]


@pytest.mark.parametrize("kind", _LINE_CLASSES)
def test_a_record_a_field_holds_untagged_is_one_the_reader_sees(kind: type) -> None:
    assert _unseen(kind) == []


@dataclass(frozen=True, slots=True, kw_only=True)
class _Unseen:
    """A message whose fields hold a record in shapes the reader does not see into."""

    by_name: Mapping[str, Attachment]
    maybe: tuple[Attachment, ...] | None
    seen: tuple[Attachment, ...]
    either: ReadResult | BashResult
    plain: str


def test_a_field_whose_shape_hides_its_record_is_named() -> None:
    assert _unseen(_Unseen) == ["_Unseen.by_name", "_Unseen.maybe"]


def test_a_spec_is_the_same_however_its_schema_was_read() -> None:
    read = Read()
    local = ToolSpec.from_tool(read)
    reread = json_loads(json_dumps(read.schema), dict[str, PlainTree])
    assert local.digest == ToolSpec(description=read.description, schema=reread).digest
    assert local.digest != ToolSpec(description="other", schema=read.schema).digest
    assert local.digest != ToolSpec(description=read.description, schema={}).digest
    assert re.fullmatch(r"[0-9a-f]{64}", local.digest)


@pytest.mark.asyncio
async def test_ping_says_ping_until_cancelled() -> None:
    write = Mock()
    writer = Mock(spec=asyncio.StreamWriter, write=write)
    sleep = AsyncMock(side_effect=[None, None, asyncio.CancelledError])
    with (
        patch("sagent.host.wire.asyncio.sleep", sleep),
        pytest.raises(asyncio.CancelledError),
    ):
        await ping(writer, every_sec=2.5)
    assert sleep.await_args_list == [call(2.5), call(2.5), call(2.5)]
    assert write.call_args_list == [call(encoded(Ping())), call(encoded(Ping()))]


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
