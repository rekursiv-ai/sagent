"""Background jobs: any tool call run detached, and the tool that manages them.

A call may carry ``background: true`` to run detached, or ``delay: N`` to start
N seconds later. The runtime offers each tool with those two keys added to its
schema, strips them before the tool runs, answers the call at once with a
pending stub, and delivers the real result later. ``BackgroundTask`` lists,
cancels, or waits for those jobs.

Typical usage example:

  tool = offered(Read())                     # what the model is shown
  delay = background_delay(call.arguments)   # None: run it now
  result = await tool.call(context, without_controls(call.arguments))
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib.resources import files
from typing import TYPE_CHECKING, Annotated, Literal, Protocol

import math
import time

from treekle import PlainTree

from sagent.session.types import UncategorizedToolResult
from sagent.tools.display import ToolDisplay
from sagent.tools.types import Tool, parse, schema_of


if TYPE_CHECKING:
    from sagent.runtime.types import Job
    from sagent.session.types import AnyToolResult
    from sagent.tools.types import ToolContext


__all__ = [
    "BackgroundTask",
    "JobTable",
    "Offered",
    "background_delay",
    "backgroundable",
    "offered",
    "without_controls",
]


class JobTable(Protocol):
    """What ``BackgroundTask`` reads and asks of the runtime running it."""

    @property
    def jobs(self) -> Mapping[str, Job]:
        """Return the calls still running, by call id."""
        ...

    async def foreground(self, call_id: str) -> AnyToolResult | None:
        """Wait for the background call ``call_id``; ``None`` if there is none."""
        ...


@dataclass(frozen=True, slots=True, kw_only=True)
class BackgroundTask:
    """List, cancel, or wait for background jobs."""

    @dataclass(frozen=True, slots=True, kw_only=True)
    class Arguments:
        """What the model passes."""

        operation: Annotated[
            Literal["list", "cancel", "foreground"],
            "Which operation to perform.",
        ]
        id: Annotated[str, "The job id; required for cancel and foreground."] = ""

    name = "BackgroundTask"
    description = (files(__package__) / "assets" / "backgroundtask.md").read_text(
        encoding="utf-8",
    )
    schema = schema_of(Arguments)

    display: ToolDisplay = field(default=ToolDisplay())

    def summary(self, arguments: Mapping[str, PlainTree]) -> str:
        """Return ``BackgroundTask <operation>``, then the job id if any."""
        job = arguments.get("id")
        return f"BackgroundTask {arguments.get('operation', '?')}" + (
            "" if job in (None, "") else f" {job}"
        )

    def serialize_key(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> str | None:
        """Return ``None``: managing jobs touches no resource another call uses."""
        del context, arguments
        return None

    async def call(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> UncategorizedToolResult:
        """Refuse: the jobs are the runtime's, which calls :meth:`manage` instead."""
        del context, arguments
        return _error("BackgroundTask runs only inside an agent's runtime.")

    async def manage(
        self,
        table: JobTable,
        arguments: Mapping[str, PlainTree],
    ) -> AnyToolResult:
        """Do ``operation`` on ``table``'s background jobs.

        Args:
          table: The runtime running this call.
          arguments: What the model passed, as ``BackgroundTask.Arguments``.

        Returns:
          result: The job listing, the cancellation, the job's own result, or
            an error naming what was wrong.

        """
        a = parse(arguments, BackgroundTask.Arguments)
        jobs = {
            cid: job
            for cid, job in table.jobs.items()
            if job.state in ("sleeping", "detached")
        }
        if a.operation == "list":
            return _listing(jobs)
        if not a.id:
            return _error(f"{a.operation} requires an id.")
        job = jobs.get(a.id)
        unknown = _error(f"No background job {a.id!r}; list shows them.")
        # A call the model waits on already is no background job to manage.
        if a.operation == "cancel" or (job is None and a.id in table.jobs):
            if job is None:
                return unknown
            job.cancel()
            return UncategorizedToolResult(
                call_id="",
                content=f"Cancelling {job.call.name} ({a.id}).",
            )
        # A job already done may still be asked for: its result is given again.
        result = await table.foreground(a.id)
        return unknown if result is None else result


@dataclass(frozen=True, slots=True, kw_only=True)
class Offered:
    """A tool as the model is shown it: its schema also takes the control keys."""

    tool: Tool
    schema: Mapping[str, PlainTree]

    @property
    def name(self) -> str:
        """Return the tool's name."""
        return self.tool.name

    @property
    def description(self) -> str:
        """Return the tool's description."""
        return self.tool.description

    @property
    def display(self) -> ToolDisplay:
        """Return the tool's display."""
        return self.tool.display

    def summary(self, arguments: Mapping[str, PlainTree]) -> str:
        """Return the tool's label for a call, without the control keys."""
        return self.tool.summary(without_controls(arguments))

    def serialize_key(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> str | None:
        """Return the tool's key for a call, without the control keys."""
        return self.tool.serialize_key(context, without_controls(arguments))

    async def call(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> AnyToolResult:
        """Run the tool, without the control keys."""
        return await self.tool.call(context, without_controls(arguments))


def backgroundable(tool: Tool) -> bool:
    """Return whether ``tool`` takes the control keys.

    A tool declaring either keeps its own (AgentSend's ``delay`` schedules a
    message); taking it over would run the call late and drop its argument.

    Args:
      tool: The tool.

    Returns:
      backgroundable: ``False`` for ``BackgroundTask`` and for a tool that
        declares ``background`` or ``delay``.

    """
    properties = tool.schema.get("properties")
    declared = isinstance(properties, Mapping) and any(
        key in properties for key in ("background", "delay")
    )
    return not declared and not isinstance(tool, BackgroundTask)


def offered(tool: Tool) -> Tool:
    """Return ``tool`` as the model is shown it: with the control keys if it takes them.

    Args:
      tool: The tool.

    Returns:
      offered: An ``Offered`` whose schema adds ``background`` and ``delay``,
        or ``tool`` itself when it is not backgroundable.

    """
    if not backgroundable(tool):
        return tool
    raw = tool.schema.get("properties")
    properties = dict(raw) if isinstance(raw, Mapping) else {}
    properties["background"] = {
        "type": "boolean",
        "description": "Run in the background; the result arrives in a later message.",
    }
    properties["delay"] = {
        "type": "number",
        "minimum": 0,
        "description": "Seconds to wait before running; implies background.",
    }
    return Offered(tool=tool, schema={**tool.schema, "properties": properties})


def background_delay(arguments: Mapping[str, PlainTree]) -> float | None:
    """Return how long to wait before running a call in the background.

    Args:
      arguments: What the model passed, control keys included.

    Returns:
      delay: Seconds to wait, or ``None`` to run the call now, in the foreground.

    Raises:
      TypeError: ``background`` is not a boolean, or ``delay`` not a number.
      ValueError: ``delay`` is negative, or not a number at all (NaN).

    """
    background = arguments.get("background", False)
    delay = arguments.get("delay", 0)
    if not isinstance(background, bool):
        raise TypeError(f"'background' must be a boolean, got {background!r}.")
    if isinstance(delay, bool) or not isinstance(delay, int | float):
        raise TypeError(f"'delay' must be seconds, at least 0, got {delay!r}.")
    if math.isnan(delay) or delay < 0:
        raise ValueError(f"'delay' must be seconds, at least 0, got {delay!r}.")
    return float(delay) if background or delay > 0 else None


def without_controls(arguments: Mapping[str, PlainTree]) -> dict[str, PlainTree]:
    """Return ``arguments`` without ``background`` and ``delay``."""
    return {k: v for k, v in arguments.items() if k not in ("background", "delay")}


def _listing(jobs: Mapping[str, Job]) -> UncategorizedToolResult:
    """Return one row per job: id, tool, phase, and seconds since it was asked for."""
    if not jobs:
        return UncategorizedToolResult(call_id="", content="No background jobs.")
    width = max(len(cid) for cid in jobs)
    now = time.time()
    rows = [
        f"{cid:<{width}}  {job.call.name:<20}  {_phase(job):<8}  {now - job.started:.0f}s"
        for cid, job in jobs.items()
    ]
    header = f"{'ID':<{width}}  {'TOOL':<20}  {'PHASE':<8}  ELAPSED"
    return UncategorizedToolResult(call_id="", content="\n".join((header, *rows)))


def _phase(job: Job) -> str:
    """Return ``sleeping`` until a delayed job starts, then ``running``."""
    return "sleeping" if job.state == "sleeping" else "running"


def _error(text: str) -> UncategorizedToolResult:
    return UncategorizedToolResult(call_id="", content=text, is_error=True)
