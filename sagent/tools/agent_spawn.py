"""Agent factory tool: let an agent spawn a configured sub-agent.

This is ``tools.AgentSpawn`` (the tool), distinct from
:class:`sagent.agent.Agent` (the runtime class).
Python namespaces disambiguate.

The factory's job is to convert an LLM-emitted tool call into a
fresh ``Agent`` instance with the right knobs resolved, run it to
completion, and return its final output. Every knob follows one
rule: ``LLM arg → factory arg → parent → hard default``. ``None``
at any layer means "fall through."

The factory also holds Python objects (``Model`` instances, tool
instances) that the LLM addresses by string name - the factory
itself *is* the registry for those objects.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, cast

import asyncio
import dataclasses
import logging
import time
import uuid

from sagent.agent.background import BackgroundTaskEntry
from sagent.lib.json import JSON, bool_val, json_freeze
from sagent.lib.lazy_import import lazy_import
from sagent.providers import PROVIDER_NAMES, build_provider
from sagent.tools.core import (
    agent_counter_var,
    agent_label_var,
    agent_path_var,
    agent_registry,
    current_agent_var,
    get_tool_state,
    load_tool_description,
    max_depth_var,
    opt_int,
    opt_str,
)
from sagent.types.compactor import Compactor
from sagent.types.history import (
    AssistantMessage,
    HistoryEntry,
    ToolResult,
    UserMessage,
)
from sagent.types.model import Model, ModelSpec
from sagent.types.runtime import (
    AgentIdle,
    ChildDoneEvent,
    ChildEvent,
    ModelResponseError,
    ModelResponsePartial,
    ModelResponseThinking,
    RuntimeEvent,
    ToolLabel,
)
from sagent.types.tools import Tool


agent_lib = lazy_import("sagent.agent")

if TYPE_CHECKING:
    from sagent.agent import (
        Agent as _Agent,
        SystemPromptArg,
    )

# Prevent GC of persistent agent tasks. Keyed by label; cleaned
# up in the wrapper's ``finally`` block.
_persistent_tasks: dict[str, asyncio.Task[None]] = {}


def _current_agent() -> _Agent | None:
    """Resolve the currently-executing concrete ``Agent``.

    ``current_agent_var`` is typed as the minimal ``AgentLike`` Protocol
    so ``tools.core`` avoids a circular import with the concrete class.
    Inside ``AgentSpawn`` we always need the full surface; non-``Agent``
    holders (e.g. ``FakeAgent`` in unit tests) return ``None`` so callers
    fall through to "no parent" semantics.
    """
    agent = current_agent_var.get()
    if agent is None:
        return None
    cls = _get_agent_class()
    return agent if isinstance(agent, cls) else None


def _get_agent_class() -> type[_Agent]:
    """Resolve ``sagent.agent.Agent`` lazily.

    ``tools.AgentSpawn`` is re-exported from ``tools/__init__.py``. That
    re-export runs while ``sagent.agent`` is mid-initialization
    (``sagent.agent`` -> ``tools.core`` -> ``tools/__init__.py`` ->
    ``tools.agent_spawn``). A top-level attribute import of ``Agent``
    here would hit the partially-initialized module and fail. Deferring
    the lookup until ``__call__`` time sidesteps the cycle; the class
    is guaranteed to be resolved by then.
    """
    return agent_lib.Agent


class AgentSpawn:
    """Factory tool: spawn a child agent with per-invocation knobs.

    Constructor args mirror :class:`sagent.agent.Agent.__init__`; any
    arg left as ``None`` falls through to the parent (or ultimately a
    hard default) at call time.

    Model selection is exposed to the LLM as four independent strings
    - ``provider``, ``auth``, ``model_id``, ``account`` - mirroring
    ``cli.py``'s CLI flags. Each follows the standard ``LLM arg →
    factory arg → parent.model_spec.<field>`` fallthrough. When every
    field matches the parent's spec the child simply reuses
    ``parent.model``; otherwise a fresh ``Model`` is built via
    :func:`providers.build_provider`.
    """

    name: str = "AgentSpawn"
    tool_id: str = "application/x-tool-agentspawn"
    description: str = load_tool_description("agentspawn")
    supports_microcompaction: bool = False
    emit_tool_summary: bool = False
    directive_schema: JSON = json_freeze(
        {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Task instruction for the child agent.",
                },
                "system": {
                    "type": "string",
                    "description": (
                        "Override the child's system prompt. Defaults to"
                        " inheriting the parent agent's system."
                    ),
                },
                "provider": {
                    "type": "string",
                    "description": (
                        "Provider class name from ``sagent.providers``"
                        " (e.g. "
                        + ", ".join(f"``{n}``" for n in PROVIDER_NAMES)
                        + "). Defaults to inheriting the parent's provider."
                    ),
                },
                "auth": {
                    "type": "string",
                    "description": (
                        "Auth method suffix - dispatches to"
                        " a zero-argument ``<Provider>.from_<auth>()``"
                        " (for example, ``env`` for API-key environment"
                        " variables). Defaults to inheriting the parent's auth."
                    ),
                },
                "model_id": {
                    "type": "string",
                    "description": (
                        "Model ID for the chosen provider (e.g."
                        " ``claude-sonnet-4-6``, ``gemini-3.1-pro-preview``,"
                        " ``gpt-5.5``). Defaults to inheriting the parent's"
                        " model id."
                    ),
                },
                "account": {
                    "type": "string",
                    "description": (
                        "Credential account name. Defaults to inheriting"
                        " the parent's account."
                    ),
                },
                "tools": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Tool names to grant the child. Defaults to the"
                        " parent's full toolset. Pass [] for no tools."
                    ),
                },
                "max_tool_call_rounds": {
                    "type": "integer",
                    "minimum": 1,
                    "description": (
                        "Cap on the child's tool-call rounds (model responses"
                        " that include one or more tool calls). Must be ≥ 1."
                    ),
                },
                "max_depth": {
                    "type": "integer",
                    "minimum": 0,
                    "description": (
                        "Cap on the child's own sub-spawning. 0 makes"
                        " the child a leaf; omit for unbounded."
                    ),
                },
                "persistent": {
                    "type": "boolean",
                    "description": (
                        "Run the child as a persistent agent via"
                        " serve_forever(). Returns immediately with"
                        " the child's label. Send messages via"
                        " AgentSend; manage via BackgroundTask."
                    ),
                },
                "notify_on_asleep": {
                    "type": "boolean",
                    "description": (
                        "Persistent only. When true, the parent's"
                        " inbox receives a UserMessage every time the"
                        " child becomes idle (drained inbox, no work"
                        " in flight). Lets the parent detect 'child"
                        " has finished processing my message' without"
                        " polling. Default false."
                    ),
                },
                "label": {
                    "type": "string",
                    "description": (
                        "Label for the child agent (used for AgentSend"
                        " routing). Auto-generated if omitted."
                    ),
                },
            },
            "required": ["prompt"],
        }
    )

    def __init__(
        self,
        *,
        provider: str | None = None,
        auth: str | None = None,
        model_id: str | None = None,
        account: str | None = None,
        system: SystemPromptArg | None = None,
        tools: list[Tool] | None = None,
        max_tool_call_rounds: int | None = None,
        max_depth: int | None = None,
        compactor: Compactor | None = None,
        max_attempts: int | None = None,
        thinking: str | None = None,
        effort: str | None = None,
        session_root_dir: str | Path | None = None,
        verbosity: int = 1,
    ) -> None:
        self._provider = provider
        self._auth = auth
        self._model_id = model_id
        self._account = account
        self._system = system
        self._tools = tools
        self._max_tool_call_rounds = max_tool_call_rounds
        self._max_depth = max_depth
        self._compactor = compactor
        self._max_attempts = max_attempts
        self._thinking = thinking
        self._effort = effort
        self._session_root_dir = (
            Path(session_root_dir) if session_root_dir is not None else None
        )
        self._verbosity = verbosity
        self.on_persistent_spawn: (
            Callable[[str, asyncio.Queue[RuntimeEvent | None]], None] | None
        ) = None
        self.on_persistent_stop: Callable[[str], None] | None = None

    def summary(self, args: Mapping[str, object]) -> str:
        """Return a short label summarizing this spawn call.

        Args:
          args: Parsed tool directive mapping.

        Returns:
          label: Compact one-line label for renderer display.

        """
        prompt_arg = args.get("prompt")
        prompt = prompt_arg if isinstance(prompt_arg, str) else ""
        preview = prompt.replace("\n", " ").strip()
        if len(preview) > 60:
            preview = preview[:57] + "..."
        model_arg = args.get("model_id")
        suffix = f" [{model_arg}]" if isinstance(model_arg, str) and model_arg else ""
        return f"{self.name} {preview}{suffix}" if preview else self.name

    def prompt(self) -> str:
        """Return spawn-budget info for the system prompt.

        Returns:
          text: Depth/cap reminder text, or empty when no cap is active.

        """
        depth = get_tool_state().depth
        cap = max_depth_var.get()
        if cap is None:
            return ""
        remaining = cap - depth
        if remaining <= 0:
            return (
                f"Spawn budget: depth {depth}/{cap} -- you are a leaf agent. "
                "Do not call AgentSpawn; it will fail. Do the work directly."
            )
        s = "s" if remaining != 1 else ""
        return (
            f"Spawn budget: depth {depth}/{cap} -- "
            f"{remaining} generation{s} of sub-spawning available."
        )

    def summary_result(self, result: ToolResult) -> str | None:
        """Return a one-line receipt for the child result.

        Args:
          result: The child's completed ``ToolResult``.

        Returns:
          receipt: Compact line-count receipt, or ``None`` when summaries
            are disabled for this tool.

        """
        if not self.emit_tool_summary:
            return None
        text = result.content.strip()
        if not text:
            return "completed with no output"
        return f"{len(text.splitlines())}L"

    async def run(self, args: Mapping[str, object]) -> ToolResult:
        """Spawn and run a child agent per the directive.

        Args:
          args: Parsed tool directive mapping.

        Returns:
          result: Child's final assistant message wrapped as a
            ``ToolResult``, or an error.

        """
        prompt = str(args.get("prompt", ""))
        system = opt_str(args, "system")
        provider = opt_str(args, "provider")
        auth = opt_str(args, "auth")
        model_id = opt_str(args, "model_id")
        account = opt_str(args, "account")
        tools_raw = args.get("tools")
        tools: list[str] | None
        if isinstance(tools_raw, (list, tuple)):
            tools = [
                str(t) for t in cast("list[object] | tuple[object, ...]", tools_raw)
            ]
        else:
            tools = None
        max_rounds = opt_int(args, "max_tool_call_rounds")
        max_depth = opt_int(args, "max_depth")
        persistent = bool_val(args.get("persistent"), False)
        notify_on_asleep = bool_val(args.get("notify_on_asleep"), False)
        custom_label = opt_str(args, "label")
        parent_agent = _current_agent()
        parent_depth = get_tool_state().depth

        # Effective max_depth = ``min`` over every non-None cap in
        # play: LLM-supplied, factory construction, and the ambient
        # ``max_depth_var`` set by an ancestor. Semantics is "tighten
        # only" - neither the LLM nor the factory can *raise* a cap an
        # ancestor already put in place.
        caps = [
            c
            for c in (max_depth, self._max_depth, max_depth_var.get())
            if c is not None
        ]
        eff_max_depth: int | None = min(caps) if caps else None

        if eff_max_depth is not None and parent_depth > eff_max_depth:
            return ToolResult(
                call_id="",
                content=f"max_depth {eff_max_depth} exceeded (current depth {parent_depth})",
                is_error=True,
            )

        resolved = self._resolve_model(
            provider=provider,
            auth=auth,
            model_id=model_id,
            account=account,
            parent_agent=parent_agent,
        )
        if isinstance(resolved, ToolResult):
            return resolved
        child_model, child_spec = resolved
        child_tools = self._resolve_tools(tools, parent_agent)
        if isinstance(child_tools, ToolResult):
            return child_tools

        child = self._build_child(
            system=system,
            child_model=child_model,
            child_spec=child_spec,
            child_tools=child_tools,
            max_rounds=max_rounds,
            parent_agent=parent_agent,
        )

        parent_path = agent_path_var.get("")
        try:
            child_idx = next(agent_counter_var.get())
        except LookupError:
            child_idx = 0
        child_path = f"{parent_path}_{child_idx}" if parent_path else str(child_idx)
        label = custom_label or f"Agent_{child_path}"

        if persistent:
            return self._spawn_persistent(
                child, label, prompt, notify_on_asleep=notify_on_asleep
            )

        return await self._execute_child(
            child,
            prompt=prompt,
            label=label,
            child_path=child_path,
            eff_max_depth=eff_max_depth,
        )

    def _build_child(
        self,
        *,
        system: str | None,
        child_model: Model,
        child_spec: ModelSpec | None,
        child_tools: list[Tool],
        max_rounds: int | None,
        parent_agent: _Agent | None,
    ) -> _Agent:
        """Build a child Agent with inherited knobs."""
        child_system = self._resolve_system(system, parent_agent)
        child_max_rounds = (
            max_rounds if max_rounds is not None else self._max_tool_call_rounds
        )
        init_kwargs: dict[str, object] = {
            "model": child_model,
            "model_spec": child_spec,
            "system": child_system,
            "tools": child_tools,
            "compactor": self._inherit("compactor", parent_agent),
            "max_tool_call_rounds": child_max_rounds,
            "session_dir": self._child_session_dir(parent_agent),
        }
        for name in ("max_attempts", "thinking", "effort"):
            resolved = self._inherit(name, parent_agent)
            if resolved is not None:
                init_kwargs[name] = resolved
        return _get_agent_class()(**init_kwargs)  # pyright: ignore[reportArgumentType]  # ty: ignore[invalid-argument-type] -- dynamic kwargs from directive

    async def _execute_child(
        self,
        child: _Agent,
        *,
        prompt: str,
        label: str,
        child_path: str,
        eff_max_depth: int | None,
    ) -> ToolResult:
        """Run a non-persistent child with contextvar isolation."""
        parent_agent = _current_agent()
        assert parent_agent is not None
        depth_token = max_depth_var.set(eff_max_depth)
        path_token = agent_path_var.set(child_path)
        label_token = agent_label_var.set(label)
        agent_token = current_agent_var.set(child)
        agent_registry[label] = child
        try:
            forwarder = _build_forwarder(label, self._verbosity, parent_agent)
            if forwarder is not None:
                child.runtime.observers.append(forwarder)
            try:
                async for _event in child.run(UserMessage(text=prompt)):
                    pass
            finally:
                if forwarder is not None and forwarder in child.runtime.observers:
                    child.runtime.observers.remove(forwarder)
                if forwarder is not None:
                    forwarder.emit_done()
            return _last_assistant_result(child.history)
        finally:
            agent_registry.pop(label, None)
            current_agent_var.reset(agent_token)
            agent_label_var.reset(label_token)
            agent_path_var.reset(path_token)
            max_depth_var.reset(depth_token)

    def _spawn_persistent(
        self,
        child: _Agent,
        label: str,
        prompt: str,
        *,
        notify_on_asleep: bool = False,
    ) -> ToolResult:
        """Start a persistent child agent via ``serve_forever()``.

        Registers the child in ``agent_registry``, attaches the parent
        forwarder as an observer, seeds the child's inbox with the prompt,
        spawns ``serve_forever`` as a visible bg job under
        ``parent._bg``. Returns immediately with the label.

        Rejects duplicate labels: a persistent agent's label is its
        addressable identity for ``AgentSend``. Silently overwriting
        ``agent_registry[label]`` would orphan the prior agent (whose
        background task keeps running but becomes unreachable) and --
        because the prior agent's cleanup ``finally`` does
        ``agent_registry.pop(label, None)`` -- eventually pop the NEW
        entry too, leaving both agents unreachable. The caller must
        kill the prior agent first.
        """
        if label in agent_registry:
            return ToolResult(
                call_id="",
                content=(
                    f"Persistent agent {label!r} is already running."
                    " Kill it via BackgroundTask before spawning a"
                    " replacement with the same label."
                ),
                is_error=True,
            )
        child._persistent = True  # noqa: SLF001 -- cross-layer flag
        child.name = label
        if self._session_root_dir is not None:
            child.session_dir = self._session_root_dir / label
        agent_registry[label] = child
        parent_agent = _current_agent()
        forwarder = _build_forwarder(
            label,
            self._verbosity,
            parent_agent,
            notify_on_asleep=notify_on_asleep,
        )
        if forwarder is not None:
            child.runtime.observers.append(forwarder)
        bg_key = f"persistent:{label}"
        external_queue: asyncio.Queue[RuntimeEvent | None] | None = (
            asyncio.Queue() if self.on_persistent_spawn is not None else None
        )

        async def _run() -> None:
            try:
                await child.serve_forever()
            except Exception:
                _logger.exception(
                    "persistent agent %r crashed in serve_forever",
                    label,
                )
            finally:
                if forwarder is not None and forwarder in child.runtime.observers:
                    child.runtime.observers.remove(forwarder)
                if forwarder is not None:
                    forwarder.emit_done()
                agent_registry.pop(label, None)
                _persistent_tasks.pop(label, None)
                if parent_agent is not None:
                    parent_agent.cancel_background(bg_key)
                if external_queue is not None:
                    external_queue.put_nowait(None)
                if self.on_persistent_stop is not None:
                    self.on_persistent_stop(label)

        child.runtime.inbox.push_back(UserMessage(text=prompt))
        task = asyncio.create_task(_run())
        _persistent_tasks[label] = task
        if parent_agent is not None:
            parent_agent.register_background(
                bg_key,
                BackgroundTaskEntry(
                    task=task,
                    tool_name="persistent-agent",
                    queue_id=label,
                    started=time.time(),
                    hidden=False,
                    kind="persistent_subagent",
                ),
            )
        if self.on_persistent_spawn is not None and external_queue is not None:
            self.on_persistent_spawn(label, external_queue)
        return ToolResult(
            call_id="",
            content=f"Persistent agent started: {label}",
        )

    def _inherit(self, name: str, parent_agent: _Agent | None) -> object:
        """Generic ``factory arg → parent → None`` fall-through.

        Used for knobs that are NOT LLM-settable: ``compactor``,
        ``max_attempts``, ``thinking``, ``effort``. The factory stores
        these as ``self._<name>`` (private); the parent exposes them
        as ``<name>`` (public property). Convention is rigid across
        all inheritable knobs so a single arg is enough. When every
        layer is ``None`` the child uses ``Agent.__init__``'s default.
        """
        factory_val = getattr(self, f"_{name}")
        if factory_val is not None:
            return factory_val
        if parent_agent is None:
            return None
        return getattr(parent_agent, name)

    def _resolve_system(
        self,
        llm_arg: str | None,
        parent_agent: _Agent | None,
    ) -> SystemPromptArg:
        """Resolve system prompt via LLM arg → factory → parent fallthrough."""
        if llm_arg is not None:
            return llm_arg
        if self._system is not None:
            return self._system
        if parent_agent is not None:
            return parent_agent.system
        return ""

    def _resolve_model(
        self,
        *,
        provider: str | None,
        auth: str | None,
        model_id: str | None,
        account: str | None,
        parent_agent: _Agent | None,
    ) -> tuple[Model, ModelSpec | None] | ToolResult:
        """Resolve ``(model, model_spec)`` for the child.

        Per-field fallthrough: ``LLM arg → factory arg →
        parent.model_spec.<field>``. When the resolved tuple equals
        the parent's spec, reuse ``parent.model`` without rebuilding.

        When the parent has no ``model_spec`` (e.g. test harnesses
        that inject a raw ``Model``) and the LLM / factory supplied
        no model strings at all, inherit ``parent.model`` as-is and
        return a ``None`` spec. If the LLM / factory *did* ask for a
        switch but the resulting trio is missing fields, that's an
        error - we can't build a provider without all three.
        """
        parent_spec = parent_agent.model_spec if parent_agent is not None else None
        llm_asked = any(x is not None for x in (provider, auth, model_id, account))
        factory_asked = any(
            x is not None
            for x in (self._provider, self._auth, self._model_id, self._account)
        )

        p = _pick_field(
            provider, self._provider, parent_spec.provider if parent_spec else None
        )
        a = _pick_field(auth, self._auth, parent_spec.auth if parent_spec else None)
        m = _pick_field(
            model_id, self._model_id, parent_spec.model_id if parent_spec else None
        )
        ac = _pick_field(
            account, self._account, parent_spec.account if parent_spec else None
        )

        if (
            parent_agent is not None
            and parent_spec is not None
            and (p, a, m, ac)
            == (
                parent_spec.provider,
                parent_spec.auth,
                parent_spec.model_id,
                parent_spec.account,
            )
        ):
            return parent_agent.model, parent_spec

        if not llm_asked and not factory_asked and parent_agent is not None:
            return parent_agent.model, parent_spec

        if p is None or a is None or m is None:
            return ToolResult(
                call_id="",
                content=(
                    "Cannot build a model: need provider, auth, and"
                    f" model_id; got provider={p!r}, auth={a!r},"
                    f" model_id={m!r}. The parent agent has no model_spec"
                    " to inherit from."
                ),
                is_error=True,
            )
        built_provider = build_provider(p, a, account=ac)
        new_model = built_provider.model(m)
        new_spec = ModelSpec(
            provider=p,
            auth=a,
            model_id=new_model.model_id,
            account=ac,
        )
        return new_model, new_spec

    def _resolve_tools(
        self,
        names: list[str] | None,
        parent_agent: _Agent | None,
    ) -> list[Tool] | ToolResult:
        """Resolve LLM-supplied tool names to tool instances.

        - ``names is None``: inherit. Use factory's ``self._tools``
          if set, else parent's full toolset (including AgentSpawn,
          so children can spawn further subagents).
        - ``names == []``: explicit empty; child runs with no tools.
          Honored - don't silently upgrade to inherit.
        - ``names == ["Read", ...]``: subset by ``.name``; error on
          unknown.

        Bundling rule: ``BackgroundTask`` rides along whenever
        ``AgentSpawn`` is granted. Any agent that can create
        persistent / background work must be able to list, cancel,
        and foreground that work -- decoupling the two is how
        runaway children become uncancellable.
        """
        available: list[Tool]
        if self._tools is not None:
            available = list(self._tools)
        elif parent_agent is not None:
            available = list(parent_agent.tools)
        else:
            available = []

        if names is None:
            return _bundle_background_task(available)

        by_name = {t.name: t for t in available}
        missing = [n for n in names if n not in by_name]
        if missing:
            return ToolResult(
                call_id="",
                content=f"Unknown tools: {missing}. Available: {list(by_name)}",
                is_error=True,
            )
        return _bundle_background_task([by_name[n] for n in names])

    def _child_session_dir(
        self,
        parent_agent: _Agent | None,
    ) -> Path | None:
        """Per-child subdir under ``session_root_dir``, or None.

        Shape: ``<root>/<parent_session_id>/<child_uuid>/``.

        Resolution order for root:
        1. Explicit ``session_root_dir`` from factory construction.
        2. Parent agent's ``session_dir`` (inherits logging from parent).
        3. ``None`` → ephemeral (no transcript).
        """
        root = self._session_root_dir
        if root is None and parent_agent is not None:
            root = parent_agent.session_dir
        if root is None:
            return None
        parent_id = parent_agent.session_id if parent_agent is not None else "root"
        return root / parent_id / str(uuid.uuid4())


def _pick_field(
    llm_val: str | None, fac_val: str | None, spec_val: str | None
) -> str | None:
    """Return the first non-None value among LLM, factory, and spec."""
    if llm_val is not None:
        return llm_val
    if fac_val is not None:
        return fac_val
    return spec_val


def _bundle_background_task(tools: list[Tool]) -> list[Tool]:
    """Append ``BackgroundTask`` when ``AgentSpawn`` is present.

    Cancel/foreground are non-negotiable companion capabilities to
    spawn: every code path that resolves a non-empty child toolset
    runs through this gate. The fresh ``BackgroundTask()`` is
    stateless -- safe to mint on demand when the parent didn't
    carry one. Returns ``tools`` unchanged if either ``AgentSpawn``
    is absent or ``BackgroundTask`` is already present.
    """
    names = {t.name for t in tools}
    if "AgentSpawn" not in names or "BackgroundTask" in names:
        return tools
    # Local import sidesteps the ``tools/__init__.py`` cycle that
    # imports ``agent_spawn`` early.
    from sagent.tools.background_task import (  # noqa: PLC0415
        BackgroundTask,
    )

    return [*tools, BackgroundTask()]


# Verbosity -> set of RuntimeEvent subclasses forwarded to the parent observer.
# verbosity 0: nothing; 1: tool labels + tool results + assistant text;
# 2: also thinking blocks. Errors are always forwarded.
_VERBOSITY: dict[int, frozenset[type]] = {
    0: frozenset(),
    1: frozenset(
        {
            ToolLabel,
            ToolResult,
            ModelResponsePartial,
        }
    ),
    2: frozenset(
        {
            ToolLabel,
            ToolResult,
            ModelResponsePartial,
            ModelResponseThinking,
        }
    ),
}


@dataclasses.dataclass(slots=True, kw_only=True)
class ChildStats:
    """Parent-scoped status-pane stats for one child agent."""

    label: str
    """Child agent's display label."""

    start: float
    """Monotonic clock seconds when the child run began."""

    model_response_tokens: int = 0
    """Approximate response tokens streamed so far."""

    model_response_chars: int = 0
    """Response characters streamed so far (drives the token estimate)."""

    cost_usd: float = 0.0
    """Running cost in USD attributed to the child."""

    done: bool = False
    """True after ``emit_done`` publishes the final ``ChildDoneEvent``."""


class _ChildForwarder:
    """Observer adapter: wraps child events as ``ChildEvent`` on parent.publish.

    Tracks per-child stats so the parent's status pane can render running-child
    summaries. Errors and tool results always forward; other events follow
    the verbosity table.

    When ``notify_on_asleep`` is True, the child's ``AgentIdle`` event is
    additionally rendered as a ``UserMessage`` pushed into the parent's
    inbox so the parent's model sees "child is idle" in its conversation
    history (not just its observer pipeline). Distinct from rendering --
    the inbox push is how persistent-child status reaches the parent's
    decision layer.
    """

    __slots__ = (
        "_forward_set",
        "_label",
        "_notify_on_asleep",
        "_parent_agent",
        "_stats",
    )

    def __init__(
        self,
        *,
        parent_agent: _Agent,
        forward_set: frozenset[type],
        stats: ChildStats,
        label: str,
        notify_on_asleep: bool = False,
    ) -> None:
        self._parent_agent = parent_agent
        self._forward_set = forward_set
        self._stats = stats
        self._label = label
        self._notify_on_asleep = notify_on_asleep

    def __call__(self, event: RuntimeEvent) -> None:
        if isinstance(event, ChildEvent):
            self._parent_agent.publish(
                ChildEvent(label=f"{self._label}/{event.label}", inner=event.inner)
            )
            return
        if isinstance(event, ChildDoneEvent):
            self._parent_agent.publish(
                ChildDoneEvent(
                    label=f"{self._label}/{event.label}",
                    elapsed=event.elapsed,
                    tokens=event.tokens,
                    cost=event.cost,
                )
            )
            return
        if isinstance(event, AgentIdle) and self._notify_on_asleep:
            # Inbox push -- not parent.publish. The parent's model sees
            # this in its conversation history; the runtime event bus
            # (observers) is the rendering layer, not the decision
            # layer. Edge-triggered upstream: AgentRuntime publishes
            # AgentIdle at most once per idle transition, so we get one
            # push per idle, not per round.
            self._parent_agent.runtime.inbox.push_back(
                UserMessage(text=f"[{self._label} is idle]"),
            )
            return
        if isinstance(event, ModelResponsePartial):
            self._stats.model_response_chars += len(event.text)
            self._stats.model_response_tokens = self._stats.model_response_chars // 4
        always_forward = isinstance(event, ModelResponseError) or (
            isinstance(event, ToolResult) and event.is_error
        )
        if not always_forward and type(event) not in self._forward_set:
            return
        self._parent_agent.publish(ChildEvent(label=self._label, inner=event))

    def emit_done(self) -> None:
        """Publish the final ``ChildDoneEvent`` summary for this child run."""
        self._stats.done = True
        elapsed = time.monotonic() - self._stats.start
        self._parent_agent.publish(
            ChildDoneEvent(
                label=self._label,
                elapsed=elapsed,
                tokens=self._stats.model_response_tokens,
                cost=self._stats.cost_usd,
            ),
        )


def _build_forwarder(
    label: str,
    verbosity: int,
    parent_agent: _Agent | None,
    *,
    notify_on_asleep: bool = False,
) -> _ChildForwarder | None:
    """Construct a forwarder bound to ``parent_agent`` (or None when at root).

    ``notify_on_asleep`` only takes effect when this forwarder is attached
    to a persistent child; non-persistent children complete inside one
    ``child.run()`` call and never publish AgentIdle while the parent is
    waiting on the result.
    """
    if parent_agent is None:
        return None
    forward_set = _VERBOSITY.get(verbosity, _VERBOSITY[1])
    stats = ChildStats(label=label, start=time.monotonic())
    return _ChildForwarder(
        parent_agent=parent_agent,
        forward_set=forward_set,
        stats=stats,
        label=label,
        notify_on_asleep=notify_on_asleep,
    )


def _last_assistant_result(
    history: list[HistoryEntry],
) -> ToolResult:
    """Return the child's last assistant message as a ``ToolResult``."""
    for m in reversed(history):
        if isinstance(m, AssistantMessage):
            return ToolResult(call_id="", content=m.text)
    return ToolResult(call_id="", content="")


_logger = logging.getLogger(__name__)
