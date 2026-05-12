"""Background-task primitives owned by the agent layer.

``BackgroundTaskEntry`` is the registry record the Agent keeps for every
long-running coroutine it tracks (user-scheduled tool invocations,
persistent subagents, detached cohort members, hidden infra like the
REPL input pump). ``BackgroundAwareTool`` is the per-tool wrapper that
injects ``background``/``delay`` properties into a tool's directive
schema so the LLM can ask for asynchronous execution.

Both live here -- not under ``tools/`` -- because ``agent/compaction.py``
and ``agent/agent.py`` need them at module-load time, and pulling
``tools/`` in that early triggers the
``providers → agent → tools → providers`` import cycle (``tools/__init__.py``
eagerly loads every tool, several of which import ``providers``).

The ``BackgroundTask`` tool itself (the one that lists/cancels/foregrounds
jobs) stays in ``tools/`` since it's an LLM-facing tool, not a runtime
primitive.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, cast

import asyncio
import dataclasses

from sagent.agent.runtime import ToolResult
from sagent.custom_types import Tool
from sagent.lib.json import JSON, MutableJSON, MutableJSONValue, json_freeze


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
    - **Detached** (``kind="detached"``, ``hidden=False``). Cohort-decayed
      tasks owned by ``runtime.detached``; the Agent's ``background``
      property synthesizes these on demand from the runtime view.
    - **Hidden infra** (``hidden=True``). REPL input pump, watchdogs,
      daemons. Filtered out of the model-facing tool listing.
    """

    task: asyncio.Task[Any]
    """The asyncio task to track / cancel."""

    tool_name: str
    """Display name surfaced by ``BackgroundTask list``."""

    queue_id: str
    """Stable identifier for cancel / foreground operations."""

    started: float
    """Wall-clock seconds when the task began."""

    delay_sec: float = 0.0
    """Sleep before invocation (tool only); 0 for immediate."""

    hidden: bool = False
    """True for infra; user-invisible."""

    kind: Literal["tool", "persistent_subagent", "detached"] = "tool"
    """Dispatch hint for shutdown semantics."""


_BG_FIELDS: JSON = json_freeze(
    {
        "background": {
            "type": "boolean",
            "description": (
                "Run this tool asynchronously. Result delivered to inbox on completion."
            ),
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
    """Proxy that injects ``background``/``delay`` into a tool's schema.

    Args:
      tool: The wrapped rich tool whose schema is extended.

    """

    name: str
    """Forwarded from the wrapped tool."""

    tool_id: str
    """Forwarded from the wrapped tool."""

    description: str
    """Forwarded from the wrapped tool."""

    supports_microcompaction: bool
    """Forwarded from the wrapped tool."""

    directive_schema: JSON
    """Wrapped tool's schema with ``background`` / ``delay`` properties
    merged into ``properties``."""

    def __init__(self, tool: Tool) -> None:
        self._tool = tool
        self.name = tool.name
        self.tool_id = tool.tool_id
        self.description = tool.description
        self.supports_microcompaction = tool.supports_microcompaction
        schema: MutableJSON = cast(MutableJSON, dict(tool.directive_schema))
        raw_props = schema.get("properties")
        if isinstance(raw_props, Mapping):
            props: MutableJSON = cast(MutableJSON, dict(raw_props))
            props.update(cast(MutableJSON, dict(_BG_FIELDS)))
            schema["properties"] = cast(MutableJSONValue, props)
            self.directive_schema = json_freeze(schema)
        else:
            self.directive_schema = tool.directive_schema

    def summary(self, args: Mapping[str, object]) -> str:
        """Forward to the wrapped tool's pre-execution label.

        Args:
          args: Directive arguments destined for the tool.

        Returns:
          label: Short label rendered before invocation.

        """
        return self._tool.summary(args)

    def summary_result(self, result: ToolResult) -> str | None:
        """Forward to the wrapped tool's post-execution receipt.

        Args:
          result: Completed tool result.

        Returns:
          receipt: Short receipt line, or ``None`` to suppress it.

        """
        return self._tool.summary_result(result)

    def prompt(self) -> str:
        """Forward to the wrapped tool's system-prompt contribution.

        Returns:
          contribution: Per-request prompt fragment, ``""`` if none.

        """
        return self._tool.prompt()

    async def run(self, args: Mapping[str, object]) -> ToolResult:
        """Forward execution to the wrapped tool.

        Args:
          args: Directive arguments parsed from the model output.

        Returns:
          result: The wrapped tool's ``ToolResult``.

        """
        return await self._tool.run(args)
