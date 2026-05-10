"""BackgroundTask tool: manage background tasks.

Conceptual model is bash job control: `&` (background), `jobs`
(list), `kill %N` (cancel), `fg %N` (foreground).  Any tool call
can be backgrounded by setting `background: true` in its
parameters; the dispatch layer in agent.py handles this before
the directive reaches the tool.  This tool manages the resulting
tasks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

import asyncio
import dataclasses
import time

from sagent.custom_types import Message, TextMessage, Tool
from sagent.lib.json import JSON, json_freeze, json_unfreeze
from sagent.lib.message import get_directive
from sagent.tools.core import current_agent_var, load_tool_description


if TYPE_CHECKING:
    from sagent.agent.agent import Agent


@dataclasses.dataclass(kw_only=True, slots=True)
class BackgroundTaskEntry:
    """A long-running task tracked by the Agent.

    Three flavors share this dataclass:

    - **Tool** (``kind="tool"``, ``hidden=False``). User-scheduled tool
      invocations spawned via ``background: true``. Listed by the
      ``BackgroundTask`` tool, cancellable by the model, foregroundable.
    - **Persistent subagent** (``kind="persistent_subagent"``,
      ``hidden=False``). Child agent running its own ``serve_forever``.
      Shut down via ``child.shutdown(force=True)`` rather than raw cancel
      so its driver loop exits cleanly.
    - **Hidden infra** (``hidden=True``). REPL input pump, watchdogs,
      daemons. Filtered out of the model-facing tool listing; survives
      ``/abort all``; cancelled only at agent shutdown.

    ``task`` is typed ``Task[Any]`` to accommodate the three flavors --
    tool runs return ``None`` (the worker posts its result via inbox),
    subagent runs return ``None``, hidden infra returns ``None``.

    Attributes:
      task: The asyncio task to track / cancel.
      tool_name: Display name surfaced by ``BackgroundTask list``.
      queue_id: Stable identifier for cancel / foreground operations.
      started: Wall-clock seconds when the task began.
      delay_sec: Sleep before invocation (tool only); 0 for immediate.
      hidden: True for infra; user-invisible.
      kind: Dispatch hint for ``abort_all_bg`` / ``shutdown(force=True)``.

    """

    task: asyncio.Task[Any]
    tool_name: str
    queue_id: str
    started: float
    delay_sec: float = 0.0
    hidden: bool = False
    kind: Literal["tool", "persistent_subagent"] = "tool"


_BG_FIELDS: JSON = json_freeze(
    {
        "background": {
            "type": "boolean",
            "description": "Run this tool asynchronously. Result delivered to inbox on completion.",
        },
        "delay": {
            "type": "integer",
            "minimum": 0,
            "description": (
                "Seconds to sleep before executing (implies background). Must be ≥ 0."
            ),
        },
    }
)


class BackgroundAwareTool:
    """Proxy that injects background/delay into a tool's schema."""

    def __init__(self, tool: Tool) -> None:
        self._tool = tool

    def __getattr__(self, name: str) -> object:
        return getattr(self._tool, name)

    @property
    def directive_schema(self) -> JSON:
        """Return the wrapped tool's schema with background/delay fields injected."""
        schema = json_unfreeze(self._tool.directive_schema)
        if not isinstance(schema, dict):
            return self._tool.directive_schema
        raw_props = schema.get("properties")
        if not isinstance(raw_props, dict):
            return self._tool.directive_schema
        props = cast(dict[str, object], raw_props)
        bg = json_unfreeze(_BG_FIELDS)
        if isinstance(bg, dict):
            props.update(cast(dict[str, object], bg))
        return json_freeze(schema)


class BackgroundTask:
    """Tool: list, cancel, or foreground background tasks."""

    name: str = "BackgroundTask"
    tool_id: str = "application/x-tool-backgroundtask"
    description: str = load_tool_description("BackgroundTask")
    supports_microcompaction: bool = True
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

    def summary(self, msg: Message) -> str:
        """Return a short label for this background-task operation.

        Args:
          msg: Tool call message.

        Returns:
          label: Operation name and optional job ID.

        """
        directive = get_directive(msg)
        op = str(directive.get("operation", "?"))
        job_id = directive.get("id")
        if job_id:
            return f"BackgroundTask {op} {job_id}"
        return f"BackgroundTask {op}"

    def summary_result(self, result: Message) -> str | None:
        del result
        return None

    def prompt(self) -> str:
        """Return background-task usage guidance for the system prompt.

        Returns:
          prompt: Instructions for backgrounding, delaying, and managing tasks.

        """
        return (
            "Any tool call can include `background: true` to run it "
            "asynchronously (result delivered to inbox on completion). "
            "Add `delay: N` (seconds) to sleep before executing "
            "(implies background). Use BackgroundTask to list, cancel, "
            "or foreground running jobs."
        )

    async def run(self, msg: Message) -> Message:
        """Dispatch a list, cancel, or foreground operation.

        Args:
          msg: Tool call message with ``operation`` and optional ``id``.

        Returns:
          result: Operation result or error message.

        """
        directive = get_directive(msg)
        op = str(directive.get("operation", ""))
        job_id = str(directive.get("id", ""))

        agent = current_agent_var.get(None)
        if agent is None:
            return TextMessage("BackgroundTask: no active agent", "text/x-error")

        if op == "list":
            return self._list(agent)
        if op == "cancel":
            return self._cancel(agent, job_id)
        if op == "foreground":
            return await self._foreground(agent, job_id)
        return TextMessage(f"Unknown operation: {op!r}", "text/x-error")

    def _list(self, agent: Agent) -> Message:
        # Hidden infra tasks (REPL pump, daemons) are filtered out --
        # they're internal and not user/model-actionable.
        jobs = {q: j for q, j in agent.background.items() if not j.hidden}
        if not jobs:
            return TextMessage("No background tasks.", "text/plain")
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
        return TextMessage(
            header + "\n" + "\n".join(lines),
            "text/plain",
        )

    def _cancel(self, agent: Agent, job_id: str) -> Message:
        if not job_id:
            return TextMessage("cancel requires an id", "text/x-error")
        jobs = agent.background
        job = jobs.get(job_id)
        if job is None or job.hidden:
            return TextMessage(f"No such job: {job_id}", "text/x-error")
        job.task.cancel()
        jobs.pop(job_id, None)
        return TextMessage(
            f"Cancelled: {job.tool_name} ({job_id})",
            "text/plain",
        )

    async def _foreground(self, agent: Agent, job_id: str) -> Message:
        if not job_id:
            return TextMessage("foreground requires an id", "text/x-error")
        jobs = agent.background
        job = jobs.get(job_id)
        if job is None or job.hidden:
            return TextMessage(f"No such job: {job_id}", "text/x-error")
        try:
            result = await job.task
        except Exception as e:  # noqa: BLE001 -- task errors are heterogeneous
            jobs.pop(job_id, None)
            return TextMessage(
                f"Job {job_id} failed: {type(e).__name__}: {e}",
                "text/x-error",
            )
        jobs.pop(job_id, None)
        return result
