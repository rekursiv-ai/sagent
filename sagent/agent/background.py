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

from sagent.lib.json import (
    JSON,
    MutableJSON,
    MutableJSONValue,
    bool_val,
    int_val,
    json_freeze,
)
from sagent.types.history import ToolResult
from sagent.types.tools import Tool


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
                "Run this tool asynchronously; result delivered at a later turn."
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

    directive_schema: JSON
    """Wrapped tool's schema with ``background`` / ``delay`` properties
    merged into ``properties``."""

    clearable_results: bool
    """Forwarded from the wrapped tool."""

    def __init__(self, tool: Tool) -> None:
        self._tool = tool
        self.name = tool.name
        self.tool_id = tool.tool_id
        self.clearable_results = tool.clearable_results
        self.description = tool.description
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

    def prompt(self) -> str | None:
        """Forward to the wrapped tool's system-prompt contribution.

        Returns:
          contribution: Per-request prompt fragment, ``""`` for none,
              or ``None`` to leave the per-section cache unchanged.

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


def split_bg_args(
    args: Mapping[str, object],
) -> tuple[bool, float, dict[str, object]]:
    """Pop ``background`` / ``delay`` from ``args``; return the rest.

    The two keys are advertised to the LLM by ``BackgroundAwareTool``
    but aren't part of the raw tool's schema. Strip before dispatch so
    the inner tool doesn't see unexpected kwargs and validation
    doesn't flag them.

    Args:
      args: Directive arguments as parsed from the model output.

    Returns:
      background: True when the LLM set ``background: true`` or
          ``delay > 0`` (delay implies background).
      delay_sec: Seconds to sleep before executing (``0.0`` for no delay).
      clean_args: ``args`` minus ``background`` / ``delay``.

    """
    clean = {k: v for k, v in args.items() if k not in ("background", "delay")}
    delay_sec = float(int_val(args.get("delay"), 0))
    background = bool_val(args.get("background"), default=False) or delay_sec > 0
    return background, max(0.0, delay_sec), clean
