"""Background jobs: any tool call run detached, and the tool that manages them.

A call may carry ``background: true`` to run detached, or ``delay: N`` to start
N seconds later; the runtime answers such a call at once with a pending stub,
and delivers the real result later. A call may also carry ``wake_every: N``, to
have what it wrote told every N seconds while it runs, in place of its tool's
schedule. The runtime offers each tool with those keys (:func:`controls`) added to
its schema, and strips them before the tool runs. ``BackgroundTask`` lists,
cancels, or waits for the jobs run apart.

Typical usage example:

  tool = offered(Read())                     # what the model is shown
  delay = background_delay(call.arguments)   # None: run it now
  streaming_result = tool.call(context, without_controls(call.arguments))
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib.resources import files
from typing import TYPE_CHECKING, Annotated, Literal, Protocol

import math

from treekle import PlainTree

from sagent.session.types import BackgroundTaskResult, ToolMessage
from sagent.tools.display import ToolDisplay
from sagent.tools.types import (
    Tool,
    ToolOutput,
    ToolStreamingResult,
    json_schema,
    parse,
    whole_output,
)


if TYPE_CHECKING:
    from sagent.runtime.types import Job
    from sagent.tools.types import ToolContext


__all__ = [
    "BackgroundTask",
    "JobTable",
    "Offered",
    "background_delay",
    "backgroundable",
    "controls",
    "offered",
    "wake_period",
    "without_controls",
]


class JobTable(Protocol):
    """What ``BackgroundTask`` reads and asks of the runtime running it."""

    @property
    def jobs(self) -> Mapping[str, Job]:
        """Return the calls still running, by call id."""
        ...

    async def foreground(self, call_id: str) -> ToolMessage | None:
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
    schema = json_schema(Arguments)

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

    def call(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> ToolStreamingResult[BackgroundTaskResult, str, str]:
        """Refuse: the jobs are the runtime's, which calls :meth:`manage` instead.

        Args:
          context: The calling agent's state; not read.
          arguments: What the model passed; not read.

        Returns:
          streaming_result: An error saying that ``BackgroundTask`` runs only
            inside an agent's runtime.

        """
        del context, arguments
        return whole_output(
            ToolOutput(
                stderr=("BackgroundTask runs only inside an agent's runtime.",),
                result=BackgroundTaskResult(exit_code=1),
            ),
        )

    async def manage(
        self,
        table: JobTable,
        arguments: Mapping[str, PlainTree],
    ) -> ToolMessage:
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
            if job.status in ("sleeping", "detached")
        }
        if a.operation == "list":
            return _listing(jobs)
        if not a.id:
            return ToolMessage(
                call_id="",
                state="done",
                stderr=(f"{a.operation} requires an id.",),
                result=BackgroundTaskResult(exit_code=1),
            )
        job = jobs.get(a.id)
        unknown = ToolMessage(
            call_id="",
            state="done",
            stderr=(f"No background job {a.id!r}; list shows them.",),
            result=BackgroundTaskResult(exit_code=1),
        )
        # A call the model waits on already is no background job to manage.
        if a.operation == "cancel" or (job is None and a.id in table.jobs):
            if job is None:
                return unknown
            job.cancel()
            return ToolMessage(
                call_id="",
                state="done",
                stdout=(f"Cancelling {job.call.name} ({a.id}).",),
                result=BackgroundTaskResult(operation="cancel", job_id=a.id),
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

    def call(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> ToolStreamingResult[object, object, object]:
        """Run the tool, without the control keys."""
        return self.tool.call(context, without_controls(arguments))


def backgroundable(tool: Tool) -> bool:
    """Return whether ``tool`` takes the control keys.

    A tool declaring any of them keeps its own (AgentSend's ``delay`` schedules a
    message); taking it over would run the call late and drop its argument.

    Args:
      tool: The tool.

    Returns:
      backgroundable: ``False`` for ``BackgroundTask`` and for a tool that
        declares one of the control keys.

    """
    properties = tool.schema.get("properties")
    declared = isinstance(properties, Mapping) and bool(controls(properties))
    return not declared and not isinstance(tool, BackgroundTask)


def offered(tool: Tool) -> Tool:
    """Return ``tool`` as the model is shown it: with the control keys if it takes them.

    Args:
      tool: The tool.

    Returns:
      offered: An ``Offered`` whose schema adds the control keys, or ``tool`` itself
        when it is not backgroundable.

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
    properties["wake_every"] = {
        "type": "number",
        "minimum": 1,
        "description": (
            "Seconds between wakes while the call runs: each tells you what it"
            " wrote since the last, or that it wrote nothing. Omit for your usual"
            " schedule."
        ),
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
      ValueError: ``delay`` is negative, or no finite number (NaN, infinity).

    """
    background = arguments.get("background", False)
    delay = arguments.get("delay", 0)
    if not isinstance(background, bool):
        raise TypeError(f"'background' must be a boolean, got {background!r}.")
    if isinstance(delay, bool) or not isinstance(delay, int | float):
        raise TypeError(f"'delay' must be seconds, at least 0, got {delay!r}.")
    if not math.isfinite(delay) or delay < 0:
        raise ValueError(f"'delay' must be seconds, at least 0, got {delay!r}.")
    return float(delay) if background or delay > 0 else None


def wake_period(arguments: Mapping[str, PlainTree]) -> float | None:
    """Return how often a call asks to have what it wrote told.

    Args:
      arguments: What the model passed, control keys included.

    Returns:
      period: Seconds between looks, or ``None`` for its tool's own schedule.

    Raises:
      TypeError: ``wake_every`` is not a number.
      ValueError: ``wake_every`` is under a second, or no finite number (NaN,
        infinity).

    """
    every = arguments.get("wake_every")
    if every is None:
        return None
    if isinstance(every, bool) or not isinstance(every, int | float):
        raise TypeError(f"'wake_every' must be seconds, at least 1, got {every!r}.")
    if not math.isfinite(every) or every < 1:
        raise ValueError(f"'wake_every' must be seconds, at least 1, got {every!r}.")
    return float(every)


def controls(arguments: Mapping[str, PlainTree]) -> dict[str, PlainTree]:
    """Return the keys of ``arguments`` that are the runtime's to read, never the tool's."""
    return {
        k: v for k, v in arguments.items() if k in ("background", "delay", "wake_every")
    }


def without_controls(arguments: Mapping[str, PlainTree]) -> dict[str, PlainTree]:
    """Return ``arguments`` without :func:`controls`."""
    kept = controls(arguments)
    return {k: v for k, v in arguments.items() if k not in kept}


def _listing(jobs: Mapping[str, Job]) -> ToolMessage[BackgroundTaskResult]:
    """Return one row per job: id, tool, status, and seconds since it was asked for."""
    listing = "No background jobs."
    if jobs:
        width = max(len(cid) for cid in jobs)
        rows = [
            f"{cid:<{width}}  {job.call.name:<20}  "
            f"{'sleeping' if job.status == 'sleeping' else 'running':<8}  "
            f"{job.elapsed_sec:.0f}s"
            for cid, job in jobs.items()
        ]
        header = f"{'ID':<{width}}  {'TOOL':<20}  {'STATUS':<8}  ELAPSED"
        listing = "\n".join((header, *rows))
    return ToolMessage(
        call_id="",
        state="done",
        stdout=(listing,),
        result=BackgroundTaskResult(operation="list"),
    )
