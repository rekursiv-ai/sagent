"""Tests for the wire: each message crosses as one line and reads back equal."""

from __future__ import annotations

from typing import TYPE_CHECKING

import asyncio
import secrets
import socket

from treekle import json_dumps

import pytest

from sagent.host.wire import (
    VERSION,
    Ack,
    Bye,
    Call,
    Cancel,
    Hello,
    Ping,
    Refused,
    Result,
    Tools,
    ToolSpec,
    Welcome,
    encoded,
    ping,
    received,
)
from sagent.session.types import (
    Attachment,
    FileReadResult,
    ShellCommandResult,
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
        Hello(version=VERSION, token=_TOKEN, session_id="s", client="c"),
        Hello(
            version=VERSION,
            token=_TOKEN,
            session_id="s",
            client="c",
            tools=("Read",),
        ),
        Welcome(name="gpu1", resumed=True, calls=("1", "2"), tools={"Bash": _SPEC}),
        Refused(reason="no"),
        Call(
            id="1",
            agent="a",
            window="w",
            tool="Bash",
            arguments={"command": "ls", "n": [1, 2]},
        ),
        Result(
            id="1",
            record=ShellCommandResult(call_id="", stdout="hi\n", exit_code=0),
        ),
        Result(
            id="2",
            record=FileReadResult(
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
    line = encoded(Result(id="1", record=ShellCommandResult(call_id="", stdout="a\nb")))
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
    assert local.same(ToolSpec(description=read.description, schema=dict(read.schema)))
    assert not local.same(ToolSpec(description="other", schema=read.schema))
    assert not local.same(ToolSpec(description=read.description, schema={}))


@pytest.mark.asyncio
async def test_ping_says_ping_until_cancelled() -> None:
    theirs, ours = socket.socketpair()
    reader, mine = await asyncio.open_unix_connection(sock=ours)
    _theirs_reader, writer = await asyncio.open_unix_connection(sock=theirs)
    pinging = asyncio.create_task(ping(writer, every_sec=0))
    assert [await received(reader), await received(reader)] == [Ping(), Ping()]
    pinging.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pinging
    writer.close()
    mine.close()


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
