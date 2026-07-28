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

from sagent.lib.custom_json import (
    JSON,
    MutableJSON,
    MutableJSONValue,
    bool_val,
    int_val,
    json_freeze,
)
from sagent.types.runtime import ToolResult
from sagent.types.tools import Tool


@dataclasses.dataclass(kw_only=True, slots=True)
class BackgroundTaskEntry:
    """A long-running task tracked by the Agent.

    Four flavors share this dataclass:

    - **Tool** (``kind="tool"``, ``hidden=False``). User-scheduled tool
      invocations spawned via ``background: true``. Listed by the
      ``BackgroundTask`` tool, cancellable by the model, foregroundable.
    - **Subagent** (``kind="subagent"``, ``hidden=False``). Child agent
      running its own ``serve_forever``. A ``lifecycle="serviced"``
      subagent is shut down via ``child.shutdown(force=True)`` rather than
      raw cancel so its driver loop exits cleanly; a ``lifecycle="oneshot"``
      subagent self-stops after its first result.
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
    """Human-facing identifier for cancel / foreground operations."""

    call_id: str = ""
    """Provider/runtime call id, when distinct from ``queue_id``."""

    started: float
    """Wall-clock seconds when the task began."""

    delay_sec: float = 0.0
    """Sleep before invocation (tool only); 0 for immediate."""

    hidden: bool = False
    """True for infra; user-invisible."""

    kind: Literal["tool", "subagent", "detached"] = "tool"
    """Dispatch hint for shutdown semantics."""

    lifecycle: Literal["oneshot", "serviced"] = "serviced"
    """Serving policy for a ``kind="subagent"`` entry.

    ``"serviced"`` subagents own their ``serve_forever`` (exempt from
    raw-cancel and shutdown-cancel); ``"oneshot"`` subagents self-stop
    after their first result. Ignored for non-subagent kinds."""

    persistent_run_id: str = ""
    """Lifecycle run id for serviced subagents.

    Empty for other kinds; ``__post_init__`` rejects an empty value when
    ``kind == "subagent"`` and ``lifecycle == "serviced"`` so the driver
    bookkeeping always has a run id to key off."""

    notify_on_asleep: bool = True
    """Whether subagent idle pings are enabled."""

    def __post_init__(self) -> None:
        if (
            self.kind == "subagent"
            and self.lifecycle == "serviced"
            and not self.persistent_run_id
        ):
            raise ValueError(
                "persistent_run_id is required when kind='subagent'"
                " and lifecycle='serviced'",
            )


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


def bg_augmented_schema(directive_schema: JSON) -> JSON:
    """Return ``directive_schema`` with ``background``/``delay`` advertised.

    The single source of truth for background-field injection, shared by
    the runtime path (:class:`BackgroundAwareTool`) and the CLI MCP bridge
    (``providers.lib.mcp_bridge``). Both advertise the same two control
    fields and strip them with :func:`split_bg_args` before dispatch, so
    the schema they validate against must be produced identically -- a
    second copy is how the two paths drift.

    Args:
      directive_schema: A tool's own object-typed (or schemaless) schema.

    Returns:
      schema: A frozen copy with ``background`` / ``delay`` merged into
          ``properties``.

    Raises:
      ValueError: If the schema declares a non-object ``type``; injecting
          ``properties`` there yields a schema strict validators reject.

    """
    schema: MutableJSON = cast(MutableJSON, dict(directive_schema))
    schema_type = schema.get("type")
    if schema_type is not None and schema_type != "object":
        raise ValueError(
            "bg_augmented_schema requires an object-typed directive_schema"
            f" (or no ``type``); got type={schema_type!r}",
        )
    raw_props = schema.get("properties")
    # Inject even when the inner schema is schemaless (no ``properties``):
    # JSON Schema allows ``type: object`` without ``properties``, but
    # skipping injection there would silently disable backgrounding.
    props: MutableJSON = (
        cast(MutableJSON, dict(raw_props))
        if isinstance(raw_props, Mapping)
        else cast(MutableJSON, {})
    )
    props.update(cast(MutableJSON, dict(_BG_FIELDS)))
    schema["properties"] = cast(MutableJSONValue, props)
    return json_freeze(schema)


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
        try:
            self.directive_schema = bg_augmented_schema(tool.directive_schema)
        except ValueError as exc:
            raise ValueError(f"tool {tool.name!r}: {exc}") from exc

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

    def serialize_key(self, args: Mapping[str, object]) -> str | None:
        """Forward to the wrapped tool's serialization key.

        Strips the injected ``background`` / ``delay`` keys first, so
        the inner tool keys off its own schema, symmetric with ``run``.

        Args:
          args: Directive arguments parsed from the model output.

        Returns:
          key: The inner tool's serialization key, or ``None``.

        """
        _, _, clean_args = split_bg_args(args)
        return self._tool.serialize_key(clean_args)

    async def run(self, args: Mapping[str, object]) -> ToolResult:
        """Forward execution to the wrapped tool with bg keys stripped.

        Symmetric with schema injection: this wrapper advertises
        ``background`` / ``delay`` to the model, so it also owns
        removing them before the inner tool's schema validation sees
        them. ``_AgentTool.run`` already strips on the production path;
        this strip covers direct invocation (alt-drivers, test
        scaffolding) so the inner tool never receives unexpected kwargs.

        Trusts the inner tool to satisfy the ``Tool.run`` "must not
        raise" contract (errors populate ``ToolResult(is_error=True)``);
        this wrapper does not add a defensive guard.

        Args:
          args: Directive arguments parsed from the model output.

        Returns:
          result: The wrapped tool's ``ToolResult``.

        """
        _, _, clean_args = split_bg_args(args)
        return await self._tool.run(clean_args)


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
    # Negative ``delay`` is meaningless; coerce to zero rather than
    # waiting an unbounded duration backwards or raising mid-dispatch.
    delay_sec = max(0.0, float(int_val(args.get("delay"), 0)))
    background = bool_val(args.get("background"), default=False) or delay_sec > 0
    return background, delay_sec, clean
