"""What a client, a tool server and a tool host say to each other: one JSON line each.

A message is a dataclass, written as a session record is (``stored_plain``), so
a tool's result crosses whole and typed: a ``ShellCommandResult`` arrives as
one, its attachments' bytes included.

::

    client                    server                    host (one per session)
      Hello    ──────────▶      checks token, version
               ◀──────────    Welcome | Refused
      Call     ──────────▶      ────────────────▶         runs the tool
               ◀──────────      ◀────────────────       Result
      Cancel   ──────────▶      ────────────────▶
      Ack      ──────────▶      forgets the result
               ◀──────────    Tools                ◀──  Tools (on start)
      Ping     ◀─────────▶    Ping                      (each way, to stay alive)
      Bye      ──────────▶      ends the session

Typical usage example:

  writer.write(encoded(Call(id="1", agent="a", window="w", tool="Bash", arguments={})))
  message = await received(reader)          # ``None`` once the peer has gone.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import asyncio
import json

from treekle import PlainTree, json_loads, mutable

from sagent.session.types import (
    AnyToolResult,
    record_classes,
    stored_plain,
    stored_record,
)


if TYPE_CHECKING:
    from sagent.tools.types import Tool


__all__ = [
    "MAX_LINE_BYTES",
    "PING_SEC",
    "SILENT_PINGS",
    "VERSION",
    "Ack",
    "Bye",
    "Call",
    "Cancel",
    "Hello",
    "Message",
    "Ping",
    "Refused",
    "Result",
    "ToolSpec",
    "Tools",
    "Welcome",
    "encoded",
    "ping",
    "received",
]


VERSION: Final = 1
"""The protocol; a client and server of different versions refuse each other."""

MAX_LINE_BYTES: Final = 256 << 20
"""The longest line read: a result carrying a PDF's pages is many megabytes."""

PING_SEC: Final = 30.0
"""How often each side says something when it has nothing to say."""

SILENT_PINGS: Final = 3
"""How many pings a peer may miss before it is taken for gone."""


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolSpec:
    """A tool as the model is shown it."""

    description: str
    schema: Mapping[str, PlainTree]

    @classmethod
    def from_tool(cls, tool: Tool) -> ToolSpec:
        """Return how ``tool`` is shown to the model."""
        return cls(description=tool.description, schema=tool.schema)

    def same(self, other: ToolSpec) -> bool:
        """Return whether both show the model one tool, however each was read."""
        return self.description == other.description and mutable(
            self.schema,
        ) == mutable(other.schema)


@dataclass(frozen=True, slots=True, kw_only=True)
class Hello:
    """A client's first line: who it is, and what it may use."""

    version: int
    token: str
    session_id: str
    """The client session, which a reconnect names to take up its calls again."""

    client: str
    """The client's machine, as it calls itself."""

    tools: tuple[str, ...] | None = None
    """The tools it asks for; ``None`` for every one its string grants."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Welcome:
    """The server's answer to a ``Hello`` it accepts."""

    name: str
    """The server's machine, which names the endpoint."""

    resumed: bool
    """Whether the session was already running."""

    calls: tuple[str, ...]
    """The calls it still holds: running, or done and not acked."""

    tools: Mapping[str, ToolSpec]
    """The tools granted, by name."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Refused:
    """The server's answer to a ``Hello`` it turns away; it closes after."""

    reason: str


@dataclass(frozen=True, slots=True, kw_only=True)
class Call:
    """Run ``tool`` with ``arguments`` for the client agent ``agent``."""

    id: str
    agent: str
    """The calling agent's label, which its working directory and seen files go by."""

    window: str
    """The calling agent's context window (``FileTracker.window``): a new one, after
    a clear or by a new agent of that label, starts at home knowing no files."""

    tool: str
    arguments: Mapping[str, PlainTree]


@dataclass(frozen=True, slots=True, kw_only=True)
class Result:
    """What a call did."""

    id: str
    record: AnyToolResult


@dataclass(frozen=True, slots=True, kw_only=True)
class Ack:
    """The client holds the result of call ``id``; the server may forget it."""

    id: str


@dataclass(frozen=True, slots=True, kw_only=True)
class Cancel:
    """Stop call ``id``; its result says it was cancelled."""

    id: str


@dataclass(frozen=True, slots=True, kw_only=True)
class Bye:
    """The client is quitting: end its session, and its calls."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Ping:
    """Nothing to say; a peer silent for three pings is gone."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Tools:
    """The tools served, by name: a host's on starting, a client's when they change."""

    tools: Mapping[str, ToolSpec]


type Message = (
    Hello | Welcome | Refused | Call | Result | Ack | Cancel | Bye | Ping | Tools
)

_KINDS: Final = (Hello, Welcome, Refused, Call, Result, Ack, Cancel, Bye, Ping, Tools)


async def ping(writer: asyncio.StreamWriter, *, every_sec: float) -> None:
    """Say ``Ping`` every ``every_sec``, so a silent peer reads as gone."""
    while True:
        await asyncio.sleep(every_sec)
        writer.write(encoded(Ping()))


def encoded(message: Message) -> bytes:
    """Return ``message`` as one line: its JSON, newline included."""
    return (json.dumps(stored_plain(message, classes=_classes())) + "\n").encode()


async def received(reader: asyncio.StreamReader) -> Message | None:
    """Return the next message ``reader`` holds; ``None`` once the peer has closed.

    Args:
      reader: A stream opened with a limit of at least ``MAX_LINE_BYTES``.

    Returns:
      message: The message; ``None`` at the end of the stream.

    Raises:
      ValueError: A line is no message.

    """
    line = await reader.readline()
    if not line:
        return None
    message = stored_record(json_loads(line), Message, classes=_classes())
    assert isinstance(message, _KINDS)
    return message


def _classes() -> dict[str, type]:
    """Return every class a line may name: the messages and the session records."""
    return {
        **record_classes(),
        ToolSpec.__name__: ToolSpec,
        **{kind.__name__: kind for kind in _KINDS},
    }
