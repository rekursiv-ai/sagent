"""Tests for the wire: each message crosses as one line and reads back equal."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, Mock, call, patch

import asyncio
import re
import secrets

from treekle import PlainTree, json_dumps, json_loads

import pytest

from sagent.host.wire import (
    VERSION,
    Ack,
    Bye,
    Call,
    Cancel,
    DaemonControl,
    DaemonStatus,
    Describe,
    Described,
    Hello,
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
    Welcome,
    encoded,
    ping,
    received,
)
from sagent.session.types import (
    Attachment,
    BashResult,
    ReadResult,
    UserMessage,
    stored_plain,
)
from sagent.tools.read import Read


if TYPE_CHECKING:
    from sagent.host.wire import Message


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
async def test_a_line_that_is_no_json_is_refused() -> None:
    reader = asyncio.StreamReader()
    reader.feed_data(b"not json\n")
    with pytest.raises(ValueError, match="Expecting value"):
        _ = await received(reader)


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
