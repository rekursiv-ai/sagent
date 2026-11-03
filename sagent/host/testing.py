"""For tests: a host run in this process, and a tool that runs until let go.

Typical usage example:

  gate = Gate()
  host = LocalHost(tools={"Gate": gate})
  server = Server(dirs, listen=[...], tools=["Gate"], start_host=host)
  await gate.started.wait()                 # A call is running there.
  gate.opened.set()                         # It returns.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING

import asyncio
import contextlib
import socket

from treekle import PlainTree

from sagent.host.session import serve
from sagent.host.wire import MAX_LINE_BYTES
from sagent.session.types import UncategorizedToolResult
from sagent.tools.display import ToolDisplay
from sagent.tools.types import Tool, ToolContext


if TYPE_CHECKING:
    from collections.abc import Sequence
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

    async def call(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> UncategorizedToolResult:
        """Wait for the gate; return who called.

        Args:
          context: The calling agent's state; its label is the answer.
          arguments: Ignored.

        Returns:
          result: ``open for LABEL``.

        """
        del arguments
        self.started.set()
        try:
            _ = await self.opened.wait()
        except asyncio.CancelledError:
            self.stopped.set()
            raise
        return UncategorizedToolResult(call_id="", content=f"open for {context.label}")


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
            _served(
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
        )
    finally:
        writer.close()


def _context(label: str, *, home: Path) -> ToolContext:
    return ToolContext(cwd=home, label=label)
