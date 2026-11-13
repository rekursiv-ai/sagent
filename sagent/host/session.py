#!/bin/sh
# ruff: noqa: EXE003, D300, D205 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 -P "$0" "$@"
One client session's tools, run as sagent on this machine runs them.

A tool server starts one per client session, as ``python -m`` of this module,
and speaks to it on a socket the host inherits as ``--fd``: ``Call``,
``Cancel`` and ``Note`` in; ``Tools``, then each ``Result``, ``Note`` and
``Peers``, out. What a tool prints goes to stdout and stderr, never into the
lines. Each client agent has its own working directory and seen files, as an
agent here does; a new context window of it starts afresh. The socket's end
cancels every call, each cleaning up as it would here.

The agents a call spawns here join this host's team. The calling agent's team,
named by each call's address book, are its outside members, reached by
``Note``: a child tells its parent there it is idle, and either side may
message the other. Such a child takes after this machine's own agent, as
sagent here would start one, with only the tools the client is granted.

Typical usage example:

  python -m sagent.host.session --fd 5 --tools Bash,Read --spill /state/spill
  await serve(reader, writer, tools={"Bash": Bash()}, new_context=ToolContext)
'''
# fmt: on

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import argparse
import asyncio
import contextlib
import logging
import socket
import uuid

from wesearch.fetch import FetchSession

from sagent.agent.agent import Agent
from sagent.agent.team import RESERVED, Team
from sagent.host.shared import host_gate
from sagent.host.wire import (
    MAX_LINE_BYTES,
    Ack,
    Call,
    Cancel,
    Note,
    Pause,
    Paused,
    Peers,
    Refused,
    Result,
    Resume,
    Tools,
    ToolSpec,
    encoded,
    received,
)
from sagent.providers.catalog import ModelCatalog, default_provider
from sagent.runtime.exceptions import describe
from sagent.runtime.types import Markers
from sagent.session.types import ToolResult
from sagent.tools.builtin import builders
from sagent.tools.remote import RemoteSession, saved, tls_link
from sagent.tools.types import ToolContext, settled


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence

    from sagent.agent.team import MessageKind
    from sagent.tools.types import Tool


__all__ = [
    "main",
    "serve",
]


logger = logging.getLogger(__name__)


async def serve(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    *,
    tools: Mapping[str, Tool],
    new_context: Callable[[str], ToolContext],
    team: Team,
    stand_in: Callable[[Sequence[Tool]], Agent],
) -> None:
    """Say which tools are served, then run each call ``reader`` asks for.

    Returns once ``reader`` ends, every call it left running cancelled.

    Args:
      reader: The server's lines: ``Call``, ``Cancel`` and ``Note``.
      writer: Where ``Tools``, then each ``Result``, ``Note`` and ``Peers``, go.
      tools: What is served, by name.
      new_context: A fresh state for the agent of a label: at home, no file seen.
      team: The agents here, which reach each caller's address book.
      stand_in: This machine's agent, unopened, with the tools given: what a
        child of a caller takes after.

    """
    writer.write(
        encoded(Tools(tools={n: ToolSpec.from_tool(t) for n, t in tools.items()})),
    )
    output = _Output(writer=writer)
    wire = _Wire(writer=output, team=team)
    team.changes.add(wire.peers)
    calls = _Calls(writer=output, tools=tools, new_context=new_context, wire=wire)
    try:
        while (message := await received(reader)) is not None:
            if isinstance(message, Call):
                wire.reach(message.address_book)
                team.stand_in = partial(
                    stand_in,
                    [t for name, t in tools.items() if name in message.granted],
                )
                calls.start(message)
            elif isinstance(message, Cancel):
                calls.cancel(message.id)
            elif isinstance(message, Note):
                wire.receive(message)
            elif isinstance(message, Peers):
                wire.peered(message.labels)
            elif isinstance(message, Pause):
                output.pause()
            elif isinstance(message, Resume):
                output.resume()
            elif isinstance(message, Ack):
                wire.pending.pop(message.id, None)
            elif isinstance(message, Refused):
                note = wire.pending.pop(message.id, None)
                if note is not None:
                    team.unreach(note.recipient)
                    if note.sender in team:
                        team.tell(
                            note.recipient,
                            recipient=note.sender,
                            content=f"[message refused] {message.reason}",
                        )
    finally:
        team.changes.discard(wire.peers)
        output.resume()
        await calls.close()


def main(argv: Sequence[str] | None = None) -> int:
    """Run one session's tools on the inherited socket until it ends.

    Args:
      argv: The flags; ``None`` for the command line's.

    Returns:
      code: The process exit code.

    """
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n", 2)[2] if __doc__ else None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_arguments(parser)
    flags = cast("Flags", parser.parse_args(argv))
    asyncio.run(_main(flags))
    return 0


class Flags(Protocol):
    """Parsed command-line flags."""

    fd: int
    tools: str
    spill: Path


@dataclass(slots=True, kw_only=True)
class _Output:
    """Hold output in the worker while its daemon replaces itself."""

    writer: asyncio.StreamWriter
    held: list[bytes] = field(default_factory=list[bytes])
    paused: bool = False

    def write(self, line: bytes) -> None:
        """Write now, or retain the line until the replacement resumes us."""
        if self.paused:
            self.held.append(line)
        else:
            self.writer.write(line)

    def pause(self) -> None:
        """Mark the last output the old daemon must consume."""
        self.writer.write(encoded(Paused()))
        self.paused = True

    def resume(self) -> None:
        """Flush in order, then send subsequent output directly."""
        for line in self.held:
            self.writer.write(line)
        self.held.clear()
        self.paused = False


@dataclass(slots=True, kw_only=True)
class _Wire:
    """The team's way to the callers: notes up, and the labels of its agents."""

    writer: _Output
    team: Team
    _told: Peers = field(default_factory=lambda: Peers(labels=()))
    """The private and public labels last written."""

    _peers: tuple[str, ...] = ()
    pending: dict[str, Note] = field(default_factory=dict[str, Note])

    def peered(self, labels: tuple[str, ...]) -> None:
        """Replace the directory's cross-session routes, not the caller's own team."""
        for label in set(self._peers) - set(labels):
            self.team.unreach(label)
        self._peers = labels
        self.reach(labels)

    def reach(self, labels: Iterable[str]) -> None:
        """Make each label of another machine's team, not one here, reachable."""
        for label in labels:
            if label not in self.team and label not in RESERVED:
                self.team.reach(label, relay=self)

    def receive(self, note: Note) -> None:
        """Put ``note`` before the agent here it names; log one for no agent.

        Args:
          note: From an agent of a caller's team, which it makes reachable.

        """
        self.reach([note.sender])
        try:
            self.team.receive(
                sender=note.sender,
                recipient=note.recipient,
                content=note.content,
                kind=note.kind,
            )
        except ValueError:
            logger.warning(
                "note from %s to %r, who is not here",
                note.sender,
                note.recipient,
            )

    def send(
        self,
        *,
        sender: str,
        recipient: str,
        content: str,
        kind: MessageKind,
    ) -> None:
        """Write a ``Note``, after the labels of the agents here if they changed.

        Args:
          sender: The agent here.
          recipient: The agent of a caller's team.
          content: What it says.
          kind: How the team here booked it.

        """
        self.peers()
        note = Note(
            id=uuid.uuid4().hex,
            sender=sender,
            recipient=recipient,
            content=content,
            kind=kind,
        )
        self.pending[note.id] = note
        self.writer.write(encoded(note))

    def peers(self) -> None:
        """Write ``Peers`` if the team's labels changed since last written."""
        labels = tuple(self.team)
        public = tuple(label for label in labels if self.team.peer_messaging(label))
        peers = Peers(labels=labels, public=public if public != labels else None)
        if peers != self._told:
            self._told = peers
            self.writer.write(encoded(peers))


class _Calls:
    """The calls running, and each agent's state, for one session."""

    def __init__(
        self,
        *,
        writer: _Output,
        tools: Mapping[str, Tool],
        new_context: Callable[[str], ToolContext],
        wire: _Wire,
    ) -> None:
        self._writer = writer
        self._tools = tools
        self._new_context = new_context
        self._wire = wire
        self._agents: dict[str, _Agent] = {}
        self._running: dict[str, asyncio.Task[None]] = {}

    # A task cancelled before its first step never enters ``_run``: the callback
    # answers it, so every call gets a result.
    def start(self, call: Call) -> None:
        """Run ``call``; its result is written when it ends."""
        task = self._running[call.id] = asyncio.create_task(self._run(call))
        task.add_done_callback(partial(self._finished, call.id))

    def _finished(self, call_id: str, task: asyncio.Task[None]) -> None:
        self._running.pop(call_id, None)
        if task.cancelled():
            cancelled = ToolResult(
                call_id="",
                state="cancelled",
                shown=Markers().cancelled,
            )
            self._writer.write(encoded(Result(id=call_id, record=cancelled)))

    def cancel(self, call_id: str) -> None:
        """Stop the call ``call_id``, if it runs; its result says it was cancelled."""
        task = self._running.get(call_id)
        if task is not None:
            task.cancel()

    async def close(self) -> None:
        """Cancel every call, and wait for each to clean up."""
        tasks = list(self._running.values())
        for task in tasks:
            task.cancel()
        for task in tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task

    def _context(self, call: Call) -> ToolContext:
        """Return the state of ``call``'s agent; a new window's starts afresh."""
        kept = self._agents.get(call.agent)
        if kept is None or kept.window != call.window:
            kept = _Agent(window=call.window, context=self._new_context(call.agent))
            self._agents[call.agent] = kept
        return kept.context

    # The labels go first: a spawn's result names a child the caller may message.
    async def _run(self, call: Call) -> None:
        record = await _result(
            self._tools.get(call.tool),
            self._context(call),
            call=call,
        )
        self._wire.peers()
        self._writer.write(encoded(Result(id=call.id, record=record)))


@dataclass(frozen=True, slots=True, kw_only=True)
class _Agent:
    """A client agent's state here, for one context window of it."""

    window: str
    context: ToolContext


async def _result(
    tool: Tool | None,
    context: ToolContext,
    *,
    call: Call,
) -> ToolResult:
    """Return what ``tool`` did, a failure or a cancel as the runtime records it."""
    if tool is None:
        return ToolResult(
            call_id="",
            state="failed",
            stderr=(f"Unknown tool: {call.tool}",),
        )
    try:
        streaming_result = tool.call(context, call.arguments)
    except Exception as exc:  # noqa: BLE001 -- A tool's failure is its call's result, as the runtime records it.
        return ToolResult(call_id="", state="failed", stderr=(describe(exc),))
    result = await settled(streaming_result)
    if result.state == "cancelled":
        return replace(result, shown=Markers().cancelled)
    return result


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register flags on ``parser``."""
    parser.add_argument(
        "--fd",
        type=int,
        required=True,
        help="The inherited socket the server speaks on.",
    )
    parser.add_argument(
        "--tools",
        required=True,
        help="The tools served, comma-separated.",
    )
    parser.add_argument(
        "--spill",
        type=Path,
        required=True,
        help="Where a tool writes the whole of an output it cut.",
    )


async def _main(flags: Flags) -> None:
    """Serve the tools ``flags`` names on the socket ``flags.fd``."""
    home = Path.cwd()
    reader, writer = await asyncio.open_unix_connection(
        sock=socket.socket(fileno=flags.fd),
        limit=MAX_LINE_BYTES,
    )
    async with (
        ModelCatalog() as catalog,
        Team(catalog=catalog) as team,
        FetchSession() as fetch,
        RemoteSession(tuple(tls_link(r) for r in saved()), team=team) as remote,
    ):
        # An agent spawned here bills the same accounts as every sagent on the host.
        catalog.gate = fetch.gate = host_gate()
        built = builders(cwd=home, team=team, remote=remote)
        new_context = partial(
            _new_context,
            home=home,
            spill_dir=flags.spill,
            fetch_session=fetch,
        )
        await serve(
            reader,
            writer,
            tools={name: built[name]() for name in flags.tools.split(",") if name},
            new_context=new_context,
            team=team,
            stand_in=partial(_stand_in, catalog=catalog, new_context=new_context),
        )
    writer.close()


def _stand_in(
    tools: Sequence[Tool],
    *,
    catalog: ModelCatalog,
    new_context: Callable[[str], ToolContext],
) -> Agent:
    """Return this machine's agent as sagent here starts one, with ``tools``."""
    return Agent(
        model=catalog.model("default", current=catalog.name(default_provider())),
        tools=tools,
        tool_context=new_context(""),
    )


def _new_context(
    label: str,
    *,
    home: Path,
    spill_dir: Path,
    fetch_session: FetchSession,
) -> ToolContext:
    return ToolContext(
        cwd=home,
        label=label,
        spill_dir=spill_dir,
        fetch_session=fetch_session,
    )


if __name__ == "__main__":
    raise SystemExit(main())
# vim: ft=python
