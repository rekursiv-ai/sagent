"""For tests: a host run in this process, and a tool that runs until let go.

Typical usage example:

  gate = Gate()
  host = LocalHost(tools={"Gate": gate})
  server = Server(dirs, listen=[...], tools=["Gate"], start_host=host)
  await gate.started.wait()                 # A call is running there.
  gate.opened.set()                         # It returns.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING

import asyncio
import contextlib
import socket

from treekle import PlainTree

from sagent.agent.agent import Agent
from sagent.agent.team import Team
from sagent.host.session import serve
from sagent.host.wire import MAX_LINE_BYTES
from sagent.model.scripted import ScriptedProvider
from sagent.providers.catalog import ModelCatalog
from sagent.session.types import UncategorizedToolResult
from sagent.tools.display import ToolDisplay
from sagent.tools.types import (
    Tool,
    ToolContext,
    ToolOutput,
    ToolStreamingResult,
    whole_output,
)


if TYPE_CHECKING:
    from pathlib import Path


__all__ = [
    "Gate",
    "LocalHost",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class Gate:
    """A tool whose call runs until ``opened`` is set, or it is cancelled."""

    name = "Gate"
    description = "Wait for the gate."
    schema: Mapping[str, PlainTree] = field(default_factory=lambda: {"type": "object"})
    display: ToolDisplay = field(default=ToolDisplay())
    started: asyncio.Event = field(default_factory=asyncio.Event)
    """Set once a call is running."""

    opened: asyncio.Event = field(default_factory=asyncio.Event)
    """Set to let each call return."""

    stopped: asyncio.Event = field(default_factory=asyncio.Event)
    """Set once a call was cancelled while it waited."""

    def summary(self, arguments: Mapping[str, PlainTree]) -> str:
        """Return ``Gate``."""
        del arguments
        return "Gate"

    def serialize_key(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> str | None:
        """Return ``None``: calls run beside each other."""
        del context, arguments
        return None

    def call(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> ToolStreamingResult[UncategorizedToolResult, str, str]:
        """Wait until the gate opens, then return the caller's label.

        Args:
          context: The calling agent's state.
          arguments: Ignored.

        Returns:
          streaming_result: ``open for LABEL`` on standard output.

        """
        del arguments
        return whole_output(self._wait(context))

    async def _wait(
        self,
        context: ToolContext,
    ) -> ToolOutput[UncategorizedToolResult, str, str]:
        """Wait until the gate opens, then return ``open for LABEL`` as the output."""
        self.started.set()
        try:
            _ = await self.opened.wait()
        except asyncio.CancelledError:
            self.stopped.set()
            raise
        return ToolOutput(
            stdout=(f"open for {context.label}",),
            result=UncategorizedToolResult(),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class _Running:
    """A host serving in a task of this process."""

    reader: asyncio.StreamReader
    writer: asyncio.StreamWriter
    task: asyncio.Task[None]

    async def stop(self) -> None:
        self.writer.write_eof()
        with contextlib.suppress(asyncio.CancelledError):
            await self.task


@dataclass(frozen=True, slots=True, kw_only=True)
class LocalHost:
    """Start each session's host in this process, serving ``tools``."""

    tools: Mapping[str, Tool]
    team: Team = field(default_factory=lambda: Team(catalog=ModelCatalog(factories={})))
    """The team of every host started: the one its agent tools were built with."""

    stand_in: Callable[[Sequence[Tool]], Agent] = field(
        default=lambda tools: Agent(model=ScriptedProvider()["scripted"], tools=tools),
    )
    """This machine's agent, with the tools given: what a caller's child takes after."""

    started: list[_Running] = field(default_factory=list[_Running])
    """Each host started, oldest first."""

    async def __call__(self, names: Sequence[str], directory: Path) -> _Running:
        """Start a host serving ``names``, each agent at home in ``directory``."""
        theirs, ours = socket.socketpair()
        reader, writer = await asyncio.open_unix_connection(
            sock=ours,
            limit=MAX_LINE_BYTES,
        )
        host_reader, host_writer = await asyncio.open_unix_connection(
            sock=theirs,
            limit=MAX_LINE_BYTES,
        )
        task = asyncio.create_task(
            self._served(
                host_reader,
                host_writer,
                tools={n: self.tools[n] for n in names if n in self.tools},
                directory=directory,
            ),
        )
        running = _Running(reader=reader, writer=writer, task=task)
        self.started.append(running)
        return running

    async def _served(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        *,
        tools: Mapping[str, Tool],
        directory: Path,
    ) -> None:
        try:
            await serve(
                reader,
                writer,
                tools=tools,
                new_context=partial(_context, home=directory),
                team=self.team,
                stand_in=self.stand_in,
            )
        finally:
            writer.close()


def _context(label: str, *, home: Path) -> ToolContext:
    return ToolContext(cwd=home, label=label)
