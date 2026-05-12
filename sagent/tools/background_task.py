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
from typing import TYPE_CHECKING

import time

from sagent.agent.background import BackgroundTaskEntry
from sagent.agent.runtime import ToolResult
from sagent.lib.json import JSON, json_freeze
from sagent.tools.core import current_agent_var, load_tool_description


if TYPE_CHECKING:
    from sagent.tools.core import AgentLike

__all__ = ["BackgroundTask", "BackgroundTaskEntry"]


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
        """Await a tracked background task and return its result."""
        if not job_id:
            return ToolResult(
                call_id="", content="foreground requires an id", is_error=True
            )
        job = agent.background.get(job_id)
        if job is None or job.hidden:
            return ToolResult(
                call_id="", content=f"No such job: {job_id}", is_error=True
            )
        try:
            result = await job.task
        except Exception as e:  # noqa: BLE001 -- task errors are heterogeneous
            agent.cancel_background(job_id)
            return ToolResult(
                call_id="",
                content=f"Job {job_id} failed: {type(e).__name__}: {e}",
                is_error=True,
            )
        agent.cancel_background(job_id)
        # The task's result is a ToolResult from ``_AgentTool.run``; return it.
        if isinstance(result, ToolResult):
            return result
        return ToolResult(call_id="", content=str(result))
