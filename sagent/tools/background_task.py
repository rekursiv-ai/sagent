"""``BackgroundTask`` tool: manage background tasks.

Conceptual model is bash job control: ``&`` (background), ``jobs``
(list), ``kill %N`` (cancel), ``fg %N`` (foreground). Any tool call
can be backgrounded by setting ``background: true`` in its
parameters; the dispatch layer in ``agent.py`` handles this before
the directive reaches the tool. This tool manages the resulting
tasks.

The dataclass that represents a tracked task (``BackgroundTaskEntry``)
and the per-tool schema wrapper (``BackgroundAwareTool``) live in
``agent/background.py`` -- the agent layer needs them at module load
time, and keeping them under ``tools/`` would force every tool module
to load before ``agent/`` is fully initialized.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, cast

import asyncio
import dataclasses
import time

from sagent.agent.background import BackgroundTaskEntry
from sagent.agent.session_io import append_persistent_agent_lifecycle
from sagent.agent.state import agent_registry
from sagent.lib.json import JSON, json_freeze
from sagent.lib.lazy_import import lazy_import
from sagent.tools.core import current_agent_var, load_tool_description
from sagent.types.runtime import (
    CANCELLED_PLACEHOLDER,
    DETACHED_PLACEHOLDER,
    RUNNING_PREFIX,
    AssistantMessage,
    DetachedResult,
    ModelContextEvent,
    RuntimeEvent,
    ToolResult,
)


if TYPE_CHECKING:
    from collections.abc import Sequence

    from sagent.agent import Agent
    from sagent.tools.core import AgentLike
    from sagent.types.tape import TapeRef

agent_lib = lazy_import("sagent.agent")

__all__ = [
    "BackgroundTask",
    "BackgroundTaskEntry",
    "cancel_persistent_subagent",
    "shutdown_persistent_subagent",
]


class BackgroundTask:
    """Tool: list, cancel, or foreground background tasks."""

    name: str = "BackgroundTask"
    tool_id: str = "application/x-tool-backgroundtask"
    clearable_results: bool = False
    description: str = load_tool_description("BackgroundTask")
    directive_schema: JSON = json_freeze(
        {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["list", "cancel", "foreground"],
                    "description": "Which operation to perform.",
                },
                "id": {
                    "type": "string",
                    "description": "Job id (required for cancel and foreground).",
                },
            },
            "required": ["operation"],
        }
    )

    def summary(self, args: Mapping[str, object]) -> str:
        """Return a short label for this background-task operation.

        Args:
          args: Directive with ``operation`` and optional ``id``.

        Returns:
          label: ``BackgroundTask <op> [<id>]`` line shown before invocation.

        """
        op = str(args.get("operation", "?"))
        job_id = args.get("id")
        if job_id:
            return f"BackgroundTask {op} {job_id}"
        return f"BackgroundTask {op}"

    def summary_result(self, result: ToolResult) -> str | None:
        """No receipt line: the operation's content is self-explanatory.

        Args:
          result: Completed ``ToolResult`` (ignored).

        Returns:
          receipt: Always ``None`` (no receipt line).

        """
        del result
        return None

    def prompt(self) -> str:
        """Return background-task usage guidance for the system prompt.

        Returns:
          contribution: Per-request prompt fragment describing ``background``
              / ``delay`` directives and ``BackgroundTask`` operations.

        """
        return (
            "Any tool call can include `background: true` to run it "
            "asynchronously (result delivered to inbox on completion). "
            "Add `delay: N` (seconds) to sleep before executing "
            "(implies background). Use BackgroundTask to list, cancel, "
            "or foreground running jobs."
        )

    def serialize_key(self, args: Mapping[str, object]) -> str | None:
        """Run in parallel: background-task control has no shared resource."""
        del args
        return None

    async def run(self, args: Mapping[str, object]) -> ToolResult:
        """Dispatch a list, cancel, or foreground operation.

        Args:
          args: Directive with ``operation`` (``list``/``cancel``/
              ``foreground``) and optional ``id``.

        Returns:
          result: Listing table, cancel confirmation, or the foregrounded
              task's ``ToolResult``; error when ``id`` is missing or unknown.

        """
        op = str(args.get("operation", ""))
        job_id = str(args.get("id", ""))

        agent = current_agent_var.get(None)
        if agent is None:
            return ToolResult(
                call_id="", content="BackgroundTask: no active agent", is_error=True
            )

        if op == "list":
            return self._list(agent)
        if op == "cancel":
            return self._cancel(agent, job_id)
        if op == "foreground":
            return await self._foreground(agent, job_id)
        return ToolResult(
            call_id="", content=f"Unknown operation: {op!r}", is_error=True
        )

    def _list(self, agent: AgentLike) -> ToolResult:
        """Render the current background-task table."""
        # Hidden infra tasks (REPL pump, daemons) are filtered out --
        # they're internal and not user/model-actionable.
        jobs = {q: j for q, j in agent.background.items() if not j.hidden}
        if not jobs:
            return ToolResult(call_id="", content="No background tasks.")
        lines: list[str] = []
        now = time.time()
        for qid, job in jobs.items():
            elapsed = now - job.started
            phase = (
                "cancelled"
                if job.task.cancelled()
                else (
                    "completed"
                    if job.task.done()
                    else (
                        "sleeping"
                        if job.delay_sec > 0 and elapsed < job.delay_sec
                        else "running"
                    )
                )
            )
            lines.append(f"{qid}  {job.tool_name:<20s}  {phase:<10s}  {elapsed:.0f}s")
        header = f"{'ID':<8s}  {'TOOL':<20s}  {'PHASE':<10s}  ELAPSED"
        return ToolResult(call_id="", content=header + "\n" + "\n".join(lines))

    def _cancel(self, agent: AgentLike, job_id: str) -> ToolResult:
        """Cancel a tracked background task by job id."""
        if not job_id:
            return ToolResult(
                call_id="", content="cancel requires an id", is_error=True
            )
        job = agent.background.get(job_id)
        if job is None or job.hidden:
            return ToolResult(
                call_id="", content=f"No such job: {job_id}", is_error=True
            )
        if job.kind == "persistent_subagent":
            # Strip the ``persistent:`` prefix to recover the child's label;
            # ``cancel_persistent_subagent`` owns the full graceful path
            # (lifecycle write + child shutdown + cancel_background).
            label = job.queue_id.removeprefix("persistent:")
            _ = cancel_persistent_subagent(agent, label)
        else:
            _ = job.task.cancel()
            # Explicit-bg entries live in the agent's bg registry; cohort-
            # detached entries land in ``runtime.detached`` and clean
            # themselves up when the task completes.
            agent.cancel_background(job_id)
        return ToolResult(
            call_id="",
            content=f"Cancelled: {job.tool_name} ({job_id})",
        )

    async def _foreground(self, agent: AgentLike, job_id: str) -> ToolResult:
        """Resolve a tracked background task to its result.

        Tracked tasks post results via ``DetachedResult`` to the
        runtime inbox; the task object itself returns ``None``. If the
        result has already been spliced into history (splice fires when
        the runtime drains the ``DetachedResult``), read it from there;
        otherwise wait for the event.

        Args:
          agent: The current agent.
          job_id: Queue id of the registered task to foreground.

        Returns:
          result: The completed task's tool result, or an error result.

        """
        if not job_id:
            return ToolResult(
                call_id="", content="foreground requires an id", is_error=True
            )
        job = agent.background.get(job_id)
        if job is None or job.hidden:
            return ToolResult(
                call_id="", content=f"No such job: {job_id}", is_error=True
            )
        if job.kind == "persistent_subagent":
            return ToolResult(
                call_id="",
                content=(
                    "Persistent subagent jobs cannot be foregrounded; use AgentSend "
                    "to message the child or BackgroundTask cancel to stop it."
                ),
                is_error=True,
            )
        completed = False
        call_id = job.call_id or job.queue_id
        try:
            spliced = _find_history_result(agent, call_id)
            if spliced is None:
                spliced = await _await_detached(agent, call_id, job)
            completed = True
            return ToolResult(
                call_id="", content=spliced.content, is_error=spliced.is_error
            )
        finally:
            if completed:
                agent.cancel_background(job_id)


def shutdown_persistent_subagent(agent: AgentLike, job: BackgroundTaskEntry) -> None:
    """Shut down a persistent child through its public lifecycle hook."""
    child = agent_registry.get(job.queue_id)
    if child is None:
        return
    parent_agent = agent if isinstance(agent, _get_agent_class()) else None
    child_agent = child if isinstance(child, _get_agent_class()) else None
    if job.persistent_run_id and parent_agent is not None and child_agent is not None:
        append_persistent_agent_lifecycle(
            parent_agent,
            child_agent,
            job.queue_id,
            job.persistent_run_id,
            state="cancelled",
            notify_on_asleep=job.notify_on_asleep,
        )
    child.shutdown(force=True)


def cancel_persistent_subagent(agent: AgentLike, label: str) -> bool:
    """Gracefully cancel a persistent subagent and clear parent bookkeeping.

    Single helper used by every cancel path (``BackgroundTask cancel``,
    ``/kill <label>``). Writes the terminal ``cancelled`` lifecycle
    record when the parent owns a matching ``BackgroundTaskEntry``,
    then forces the child to shut down and unregisters the parent
    bookkeeping. Falls back to a registry-only shutdown when the parent
    has no matching bg entry -- the child still stops, but no lifecycle
    record is written because the parent never owned the run id.

    Args:
      agent: Parent agent.
      label: Persistent child's registry label.

    Returns:
      cancelled: True when a child was found and shut down; False when
          no child matched ``label``.

    """
    queue_id = f"persistent:{label}"
    job = agent.background.get(queue_id)
    if job is not None and job.kind == "persistent_subagent":
        shutdown_persistent_subagent(agent, job)
        agent.cancel_background(queue_id)
        return True
    child = agent_registry.get(label)
    if child is None:
        return False
    child.shutdown(force=True)
    return True


def _get_agent_class() -> type[Agent]:
    """Resolve the concrete ``Agent`` class lazily."""
    return cast(type["Agent"], agent_lib.Agent)


def _find_history_result(agent: AgentLike, call_id: str) -> ToolResult | None:
    """Return the most recent non-placeholder result matching ``call_id``."""
    for entry in reversed(agent.runtime.context().messages):
        if (
            isinstance(entry, ToolResult)
            and entry.call_id == call_id
            and not _is_background_placeholder(entry.content)
        ):
            return entry
    return None


def _is_background_placeholder(content: str) -> bool:
    """Return true for background placeholders awaiting detached content."""
    return content == DETACHED_PLACEHOLDER or content.startswith(RUNNING_PREFIX)


async def _await_detached(
    agent: AgentLike,
    call_id: str,
    job: BackgroundTaskEntry,
) -> ToolResult:
    """Wait for a ``DetachedResult`` matching ``call_id`` and return it."""
    fut: asyncio.Future[ToolResult] = asyncio.get_running_loop().create_future()

    def on_event(event: RuntimeEvent) -> None:
        if fut.done():
            return
        if isinstance(event, DetachedResult) and event.call_id == call_id:
            fut.set_result(_tool_result_from_detached(event))

    agent.runtime.observers.append(on_event)
    try:
        try:
            await asyncio.shield(job.task)
        except asyncio.CancelledError:
            if fut.done():
                return fut.result()
            event = _drain_queued_detached(agent, call_id)
            if event is not None:
                return event
            if job.task.cancelled():
                return ToolResult(
                    call_id=call_id, content=CANCELLED_PLACEHOLDER, is_error=True
                )
            raise
        except Exception as exc:  # noqa: BLE001
            if fut.done():
                return fut.result()
            return ToolResult(
                call_id=call_id,
                content=f"{type(exc).__name__}: {exc}",
                is_error=True,
            )
        if fut.done():
            return fut.result()
        event = await _drain_detached(agent, call_id)
        if event is not None:
            return event
        return await fut
    finally:
        if not fut.done():
            fut.cancel()
        if on_event in agent.runtime.observers:
            agent.runtime.observers.remove(on_event)


async def _drain_detached(agent: AgentLike, call_id: str) -> ToolResult | None:
    """Drain one inbox batch and return a matching detached result if present."""
    return _splice_from_inbox_items(agent, call_id, await agent.runtime.inbox.drain())


def _drain_queued_detached(agent: AgentLike, call_id: str) -> ToolResult | None:
    """Drain already-queued inbox items without blocking for new input."""
    return _splice_from_inbox_items(
        agent,
        call_id,
        agent.runtime.inbox.drain_nowait(),
    )


def _splice_from_inbox_items(
    agent: AgentLike,
    call_id: str,
    items: list[RuntimeEvent],
) -> ToolResult | None:
    """Splice one detached result from ``items`` and restore the rest."""
    keep: list[RuntimeEvent] = []
    result: ToolResult | None = None
    for item in items:
        if isinstance(item, DetachedResult) and item.call_id == call_id:
            result = _splice_detached(agent, item)
        else:
            keep.append(item)
    if keep:
        agent.runtime.inbox.push_front(*keep)
    return result


def _splice_detached(agent: AgentLike, event: DetachedResult) -> ToolResult:
    """Splice a detached event into history and return the foreground result."""
    spliced = _replace_placeholder(agent, event)
    if spliced is not None:
        return spliced
    return _tool_result_from_detached(event)


def _replace_placeholder(agent: AgentLike, event: DetachedResult) -> ToolResult | None:
    """Replace the latest visible placeholder for ``event.call_id``."""
    resolved = agent.runtime.context()
    parent_origin = _find_parent_origin(
        resolved.messages, resolved.origins, event.call_id
    )
    if parent_origin is None:
        return None
    for entry, origin in reversed(
        tuple(zip(resolved.messages, resolved.origins, strict=True))
    ):
        if (
            isinstance(entry, ToolResult)
            and entry.call_id == event.call_id
            and _is_background_placeholder(entry.content)
        ):
            real = dataclasses.replace(
                entry,
                content=event.content,
                is_error=event.is_error,
            )
            agent.runtime.append_splice(
                mask=((origin, origin),),
                insert_after=parent_origin,
                payload=(real,),
                strategy="foreground_detached_splice",
                paired_externally=frozenset({event.call_id}),
            )
            return real
    return None


def _find_parent_origin(
    messages: Sequence[ModelContextEvent],
    origins: Sequence[TapeRef],
    call_id: str,
) -> TapeRef | None:
    """Return the visible assistant origin for ``call_id``, if present."""
    for entry, origin in reversed(tuple(zip(messages, origins, strict=True))):
        if isinstance(entry, AssistantMessage) and any(
            tool_call.id == call_id for tool_call in entry.tool_calls
        ):
            return origin
    return None


def _tool_result_from_detached(event: DetachedResult) -> ToolResult:
    """Convert a detached runtime event into a foreground tool result."""
    return ToolResult(
        call_id=event.call_id,
        content=event.content,
        is_error=event.is_error,
    )
