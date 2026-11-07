"""What a client, a tool server and a tool host say to each other: one JSON line each.

A message is a dataclass, written as a session record is (``stored_plain``), so
a tool's result crosses whole and typed: a ``BashResult`` arrives as
one, its attachments' bytes included.

::

    client                    server                    host (one per session)
                                                   ◀──  Tools (specs, on start)
      Hello    ──────────▶      checks token, version
               ◀──────────    Welcome (digests) | Refused
      Describe ──────────▶      answers from the specs
               ◀──────────    Described
      Call     ──────────▶      ────────────────▶         runs the tool
               ◀──────────      ◀────────────────       Result
      Cancel   ──────────▶      ────────────────▶
      Ack      ──────────▶      forgets the result
               ◀──────────    ToolsChanged (digests, as grants change)
      Note     ──────────▶      ────────────────▶         to a hosted agent
               ◀──────────      ◀────────────────       Note (to a client agent)
               ◀──────────      ◀────────────────       Peers (its agents' labels)
      Ping     ◀─────────▶    Ping                      (each way, to stay alive)
      Bye      ──────────▶      ends the session

An owner on the daemon's own socket may instead send one request, alone on its
connection: ``DaemonControl`` (answered ``DaemonStatus``) or ``GateCall``
(``GateAnswer``).

A client is sent a tool's digest, never its schema; it asks ``Describe`` only
for a tool whose digest matches none of its own. A ``Note`` is acked as a
``Result`` is, and kept until then by whoever sent it, across a drop.

Typical usage example:

  writer.write(encoded(Call(id="1", agent="a", window="w", tool="Bash", arguments={})))
  message = await received(reader)          # ``None`` once the peer has gone.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final, Literal

import asyncio
import hashlib
import json

from treekle import PlainTree, json_loads, mutable
from wesearch.fetch.gate import Admission, GatePolicy

from sagent.agent.team import MessageKind
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
    "DaemonControl",
    "DaemonStatus",
    "Describe",
    "Described",
    "GateAnswer",
    "GateCall",
    "Hello",
    "Message",
    "Note",
    "Pause",
    "Paused",
    "Peers",
    "Ping",
    "Refused",
    "Result",
    "Resume",
    "ToolSpec",
    "Tools",
    "ToolsChanged",
    "Welcome",
    "decoded",
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

    @property
    def digest(self) -> str:
        """Return the hex SHA-256 of the spec, however its schema was read.

        Two machines whose digests of a tool match show the model one tool, so
        a client sends no schema it already has.
        """
        shown = {"description": self.description, "schema": mutable(self.schema)}
        return hashlib.sha256(json.dumps(shown, sort_keys=True).encode()).hexdigest()


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
    """What it still holds: calls running, and results and notes not acked."""

    tools: Mapping[str, str]
    """The tools granted: each name's ``ToolSpec.digest``."""

    peer_messaging: bool = False
    """Whether the token also grants cross-session messaging."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Refused:
    """A refused handshake, or an identified message rejected by the directory."""

    reason: str
    id: str = ""


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
    address_book: tuple[str, ...] = ()
    """The labels of the calling agent's team, its own first: whom an agent spawned
    there may message."""

    granted: tuple[str, ...] = ()
    """The tools the client is granted, which the server sets: all an agent spawned
    there may have."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Note:
    """A message between agents of the two machines; acked by ``Ack`` ``id``."""

    id: str
    sender: str
    recipient: str
    content: str
    kind: MessageKind
    """How the sending team booked it, as the receiving team books it too."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Peers:
    """Private team labels, optionally narrowed for cross-session discovery."""

    labels: tuple[str, ...]
    public: tuple[str, ...] | None = None


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
class DaemonControl:
    """An owner-only request to inspect, configure, or replace the daemon."""

    source_version: str = ""
    listen: tuple[str, ...] | None = None
    tools: tuple[str, ...] | None = None
    ttl_sec: float | None = None
    stop: bool = False


@dataclass(frozen=True, slots=True, kw_only=True)
class DaemonStatus:
    """The serving process, its code fingerprint, and enabled TLS listeners."""

    pid: int
    source_version: str
    listen: tuple[str, ...]
    restarting: bool = False


@dataclass(frozen=True, slots=True, kw_only=True)
class GateCall:
    """An owner's call of the daemon's ``Gate``: ``op`` names the method."""

    op: Literal["admitted", "trip", "hold", "release_probe", "reset"]
    """``admitted`` stays open until the caller is admitted, or refused; each
    other changes the key, waking every caller waiting on it."""

    key: str = ""
    policy: GatePolicy | None = None
    """For ``admitted`` and ``trip``."""

    until: float = 0.0
    """For ``hold``."""

    challenge_url: str = ""
    """For ``trip``."""

    forget: bool = False
    """For ``reset``."""


@dataclass(frozen=True, slots=True, kw_only=True)
class GateAnswer:
    """What a ``GateCall`` returned: an ``admitted``'s admission; else nothing."""

    admission: Admission = field(default_factory=Admission)


@dataclass(frozen=True, slots=True, kw_only=True)
class Pause:
    """Buffer worker output after acknowledging the daemon's handoff boundary."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Paused:
    """All preceding worker output has been sent; subsequent output is buffered."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Resume:
    """Flush buffered worker output and continue on the inherited connection."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Ping:
    """Nothing to say; a peer silent for three pings is gone."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Tools:
    """The tools a host serves, by name: its first line to the server."""

    tools: Mapping[str, ToolSpec]


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolsChanged:
    """The current tool digests and an optional cross-session permission change."""

    tools: Mapping[str, str]
    peer_messaging: bool | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class Describe:
    """Send the spec of ``tool``, whose digest the client does not know."""

    id: str
    tool: str


@dataclass(frozen=True, slots=True, kw_only=True)
class Described:
    """The answer to ``Describe`` ``id``: the spec; ``None`` for a tool not granted."""

    id: str
    spec: ToolSpec | None


type Message = (
    DaemonControl
    | DaemonStatus
    | GateCall
    | GateAnswer
    | Hello
    | Welcome
    | Refused
    | Call
    | Result
    | Ack
    | Cancel
    | Bye
    | Ping
    | Tools
    | ToolsChanged
    | Describe
    | Described
    | Note
    | Peers
    | Pause
    | Paused
    | Resume
)

_KINDS: Final = (
    DaemonControl,
    DaemonStatus,
    GateCall,
    GateAnswer,
    Hello,
    Welcome,
    Refused,
    Call,
    Result,
    Ack,
    Cancel,
    Bye,
    Ping,
    Tools,
    ToolsChanged,
    Describe,
    Described,
    Note,
    Peers,
    Pause,
    Paused,
    Resume,
)


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
    return decoded(line)


def decoded(line: bytes) -> Message:
    """Return the message one line holds.

    Args:
      line: One line, as ``encoded`` wrote it.

    Returns:
      message: The message.

    Raises:
      ValueError: The line is no message.

    """
    message = stored_record(json_loads(line), Message, classes=_classes())
    assert isinstance(message, _KINDS)
    return message


def _classes() -> dict[str, type]:
    """Return every class a line may name: the messages and the session records."""
    return {
        **record_classes(),
        ToolSpec.__name__: ToolSpec,
        **{kind.__name__: kind for kind in (GatePolicy, Admission)},
        **{kind.__name__: kind for kind in _KINDS},
    }
