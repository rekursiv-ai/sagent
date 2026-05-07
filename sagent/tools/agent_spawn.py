"""Agent factory tool: let an agent spawn a configured sub-agent.

This is ``tools.AgentSpawn`` (the tool), distinct from
:class:`sagent.agent.Agent` (the runtime class).
Python namespaces disambiguate.

The factory's job is to convert an LLM-emitted tool call into a
fresh ``AgentSpawn`` instance with the right knobs resolved, run it to
completion, and return its final output. Every knob follows one
rule: ``LLM arg → factory arg → parent → hard default``. ``None``
at any layer means "fall through."

The factory also holds Python objects (``Model`` instances, tool
instances) that the LLM addresses by string name - the factory
itself *is* the registry for those objects.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import asyncio
import dataclasses
import logging
import time
import uuid

from sagent.custom_types import (
    JsonMessage,
    Message,
    ModelSpec,
    MultipartMessage,
    TextMessage,
    Tool,
    is_message,
)
from sagent.lib.descriptors import has_error
from sagent.lib.json import JSON, bool_val, json_freeze
from sagent.lib.lazy_import import lazy_import
from sagent.lib.message import get_directive
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


agent_lib = lazy_import("sagent.agent")


if TYPE_CHECKING:
    from sagent.agent import (
        Agent as _Agent,
        SystemPrompt,
    )
    from sagent.custom_types import Compactor, Model


# Prevent GC of persistent agent tasks. Keyed by label; cleaned
# up in the wrapper's ``finally`` block.
_persistent_tasks: dict[str, asyncio.Task[None]] = {}


def _get_agent_class() -> type[_Agent]:
    """Resolve ``sagent.agent.Agent`` lazily.

    ``tools.agent`` is re-exported from ``tools/__init__.py`` for
    ``tools.AgentSpawn`` ergonomics. That re-export runs while
    ``sagent.agent`` is mid-initialization (``sagent.__init__`` ->
    ``sagent.agent`` -> ``tools.core`` -> ``tools/__init__.py`` ->
    ``tools.agent``). A top-level attribute import of ``Agent`` at
    this stage would hit the partially-initialized ``sagent.agent``
    and fail. Deferring the lookup until ``__call__`` time sidesteps
    the cycle cleanly; the class is guaranteed to be resolved by then.
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
                        " run_forever(). Returns immediately with"
                        " the child's label. Send messages via"
                        " AgentSend; manage via BackgroundTask."
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
        system: SystemPrompt | None = None,
        tools: list[Tool] | None = None,
        max_tool_call_rounds: int | None = None,
        max_depth: int | None = None,
        compactor: Compactor | None = None,
        max_attempts: int | None = None,
        thinking: str | None = None,
        effort: str | None = None,
        session_root_dir: str | Path | None = None,
        on_message: Callable[[Message], None] | None = None,
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
        # Streaming callback. When set, the factory gives the child an
        # ``asyncio.Queue[Message | None]`` and pumps events into
        # ``on_message`` concurrently with the child's run. ``None``
        # means buffered-only (the default) - the caller sees just the
        # final ``ToolResponse``.
        self._on_message = on_message
        self._verbosity = verbosity
        self.on_persistent_spawn: (
            Callable[[str, asyncio.Queue[Message | None]], None] | None
        ) = None
        self.on_persistent_stop: Callable[[str], None] | None = None

    def summary(self, msg: Message) -> str:
        """Return a short label summarizing this spawn call.

        Args:
          msg: Tool call message.

        Returns:
          label: Human-readable summary with prompt preview and model.

        """
        directive = get_directive(msg)
        prompt_arg = directive.get("prompt")
        prompt = prompt_arg if isinstance(prompt_arg, str) else ""
        preview = prompt.replace("\n", " ").strip()
        if len(preview) > 60:
            preview = preview[:57] + "..."
        model_arg = directive.get("model_id")
        suffix = f" [{model_arg}]" if isinstance(model_arg, str) and model_arg else ""
        label = f"{self.name} {preview}{suffix}" if preview else self.name
        return label

    def prompt(self) -> str:
        """Return spawn-budget info for the system prompt.

        Returns:
          prompt: Budget summary, or empty string if uncapped.

        """
        # Surface spawn budget so agents plan inline instead of attempting
        # AgentSpawn calls that will immediately fail at the depth limit.
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
        return f"Spawn budget: depth {depth}/{cap} -- {remaining} generation{s} of sub-spawning available."

    async def run(self, msg: Message) -> Message:
        """Spawn and run a child agent per the directive.

        Args:
          msg: Tool call message with prompt, model, tools, etc.

        Returns:
          result: Child agent's final output message.

        """
        directive = get_directive(msg)
        prompt = str(directive.get("prompt", ""))
        system = opt_str(directive, "system")
        provider = opt_str(directive, "provider")
        auth = opt_str(directive, "auth")
        model_id = opt_str(directive, "model_id")
        account = opt_str(directive, "account")
        tools_raw = directive.get("tools")
        tools: list[str] | None = (
            [str(t) for t in tools_raw]
            if isinstance(tools_raw, (list, tuple))
            else None
        )
        max_rounds = opt_int(directive, "max_tool_call_rounds")
        max_depth = opt_int(directive, "max_depth")
        persistent = bool_val(directive.get("persistent"), False)
        custom_label = opt_str(directive, "label")
        parent_agent = current_agent_var.get()
        parent_depth = get_tool_state().depth

        # Effective max_depth = ``min`` over every non-None cap in
        # play: LLM-supplied, factory construction, and the ambient
        # ``max_depth_var`` set by an ancestor. Semantics is "tighten
        # only" - neither the LLM nor the factory can *raise* a cap an
        # ancestor already put in place. ``None`` still propagates a
        # no-cap signal, but any concrete int on any layer is binding.
        caps = [
            c
            for c in (max_depth, self._max_depth, max_depth_var.get())
            if c is not None
        ]
        eff_max_depth: int | None = min(caps) if caps else None

        if eff_max_depth is not None and parent_depth > eff_max_depth:
            return TextMessage(
                f"max_depth {eff_max_depth} exceeded (current depth {parent_depth})",
                "text/x-error",
            )

        resolved = self._resolve_model(
            provider=provider,
            auth=auth,
            model_id=model_id,
            account=account,
            parent_agent=parent_agent,
        )
        if is_message(resolved):
            return resolved
        child_model, child_spec = resolved
        child_tools = self._resolve_tools(tools, parent_agent)
        if is_message(child_tools):
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
            return self._spawn_persistent(child, label, prompt, msg)

        child_directive = json_freeze({"prompt": prompt})
        return await self._execute_child(
            child,
            child_directive=child_directive,
            label=label,
            child_path=child_path,
            eff_max_depth=eff_max_depth,
            msg=msg,
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
        child_directive: JSON,
        label: str,
        child_path: str,
        eff_max_depth: int | None,
        msg: Message,
    ) -> Message:
        """Run a non-persistent child with contextvar isolation."""
        parent_agent = current_agent_var.get()
        assert parent_agent is not None
        depth_token = max_depth_var.set(eff_max_depth)
        path_token = agent_path_var.set(child_path)
        label_token = agent_label_var.set(label)
        try:
            on_msg = self._on_message or _make_parent_forwarder(
                label, self._verbosity, parent_agent
            )
            handle = child.run(child_directive)
            async for event in handle:
                if on_msg is not None:
                    try:
                        on_msg(event)
                    except Exception:
                        _logger.exception("on_message callback raised")
            result = await handle
            return dataclasses.replace(result, parent_id=msg.id)
        finally:
            parent_agent.active_children.pop(label, None)
            agent_label_var.reset(label_token)
            agent_path_var.reset(path_token)
            max_depth_var.reset(depth_token)

    def _spawn_persistent(
        self,
        child: _Agent,
        label: str,
        prompt: str,
        msg: Message,
    ) -> Message:
        """Start a persistent child agent via ``run_forever()``.

        Registers the child in ``agent_registry`` and seeds its inbox
        with the initial prompt. Returns immediately with the label.
        """
        child._persistent = True  # noqa: SLF001 -- cross-layer flag
        child.name = label
        if self._session_root_dir is not None:
            child._session_dir = self._session_root_dir / label  # noqa: SLF001 -- cross-layer private attr
        agent_registry[label] = child
        agent_label_var_token = agent_label_var.set(label)
        parent_agent = current_agent_var.get()
        on_msg = self._on_message or _make_parent_forwarder(
            label, self._verbosity, parent_agent
        )
        external_queue: asyncio.Queue[Message | None] | None = None
        if self.on_persistent_spawn is not None:
            external_queue = asyncio.Queue()

        async def _run() -> None:
            try:
                async for event in child.run_forever():
                    if on_msg is not None and event is not None:
                        try:
                            on_msg(event)
                        except Exception:
                            _logger.exception("on_message callback raised")
                    if external_queue is not None:
                        external_queue.put_nowait(event)
            finally:
                agent_registry.pop(label, None)
                _persistent_tasks.pop(label, None)
                if self.on_persistent_stop is not None:
                    self.on_persistent_stop(label)

        child.inbox.put(TextMessage(prompt, "text/x-user-message"))
        task = asyncio.create_task(_run())
        _persistent_tasks[label] = task
        agent_label_var.reset(agent_label_var_token)
        if self.on_persistent_spawn is not None and external_queue is not None:
            self.on_persistent_spawn(label, external_queue)
        return TextMessage(
            f"Persistent agent started: {label}",
            "text/plain",
            parent_id=msg.id,
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
    ) -> SystemPrompt:
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
    ) -> tuple[Model, ModelSpec | None] | Message:
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
            return TextMessage(
                (
                    "Cannot build a model: need provider, auth, and"
                    f" model_id; got provider={p!r}, auth={a!r},"
                    f" model_id={m!r}. The parent agent has no model_spec"
                    " to inherit from."
                ),
                "text/x-error",
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
    ) -> list[Tool] | Message:
        """Resolve LLM-supplied tool names to tool instances.

        - ``names is None``: inherit. Use factory's ``self._tools``
          if set, else parent's full toolset (including AgentSpawn,
          so children can spawn further subagents).
        - ``names == []``: explicit empty; child runs with no tools.
          Honored - don't silently upgrade to inherit.
        - ``names == ["Read", ...]``: subset by ``.name``; error on
          unknown.
        """
        available: list[Tool]
        if self._tools is not None:
            available = list(self._tools)
        elif parent_agent is not None:
            available = list(parent_agent.tools)
        else:
            available = []

        if names is None:
            return available

        by_name = {t.name: t for t in available}
        missing = [n for n in names if n not in by_name]
        if missing:
            return TextMessage(
                f"Unknown tools: {missing}. Available: {list(by_name)}",
                "text/x-error",
            )
        return [by_name[n] for n in names]

    def _child_session_dir(
        self,
        parent_agent: _Agent | None,
    ) -> Path | None:
        """Per-child subdir under ``session_root_dir``, or None.

        Shape: ``<session_root_dir>/<parent_session_id>/<child_uuid>/``.
        When no root is configured, the child runs ephemerally (no
        transcript).
        """
        if self._session_root_dir is None:
            return None
        parent_id = parent_agent.session_id if parent_agent is not None else "root"
        return self._session_root_dir / parent_id / str(uuid.uuid4())


def _pick_field(
    llm_val: str | None, fac_val: str | None, spec_val: str | None
) -> str | None:
    """Return the first non-None value among LLM, factory, and spec."""
    if llm_val is not None:
        return llm_val
    if fac_val is not None:
        return fac_val
    return spec_val


# Verbosity → forwarded descriptors.
_VERBOSITY: dict[int, frozenset[str]] = {
    0: frozenset(),
    1: frozenset({"text/x-tool-label", "multipart/x-tool-result", "text/plain"}),
    2: frozenset(
        {
            "text/x-tool-label",
            "multipart/x-tool-result",
            "text/plain",
            "text/x-thinking",
        }
    ),
}
# Always forwarded regardless of verbosity.
_ALWAYS_FORWARD = frozenset({"multipart/x-child-event"})


@dataclasses.dataclass(slots=True, kw_only=True)
class ChildStats:
    """Parent-scoped toolbar stats for one child agent."""

    label: str
    start: float
    model_response_tokens: int = 0
    model_response_chars: int = 0
    cost_usd: float = 0.0
    done: bool = False


class _ChildForwarder:
    """Callable that wraps child events with a label and forwards them.

    Holds a reference to the parent agent and reads ``._events`` lazily
    at call time. Between parent ``run()`` calls ``._events`` is None
    and events are silently dropped.
    """

    __slots__ = ("_forward_set", "_label", "_parent_agent", "_stats")

    def __init__(
        self,
        *,
        parent_agent: _Agent,
        forward_set: frozenset[str],
        stats: ChildStats,
        label: str,
    ) -> None:
        self._parent_agent = parent_agent
        self._forward_set = forward_set
        self._stats = stats
        self._label = label

    def _put(self, event: Message) -> None:
        """Write to the parent's live queue, or no-op if between requests."""
        q = self._parent_agent._events  # noqa: SLF001 -- cross-layer queue access
        if q is not None:
            q.put_nowait(event)

    def __call__(self, event: Message) -> None:
        if isinstance(event, JsonMessage) and event.descriptor == "application/x-done":
            # Child done events are private; the parent stream gets a labeled summary.
            self._stats.model_response_tokens = (
                opt_int(event.content, "output_tokens") or 0
            )
            self._stats.done = True
            elapsed = time.monotonic() - self._stats.start
            done_inner = JsonMessage(
                json_freeze(
                    {
                        "elapsed": elapsed,
                        "model_response_tokens": self._stats.model_response_tokens,
                        "cost_usd": self._stats.cost_usd,
                    }
                ),
                "application/x-child-done",
            )
            self._put(
                MultipartMessage(
                    (
                        TextMessage(
                            self._label,
                            "text/x-agent-label",
                        ),
                        done_inner,
                    ),
                    "multipart/x-child-event",
                )
            )
            self._parent_agent.active_children.pop(self._label, None)
            return
        if event.descriptor == "text/plain" and isinstance(event, TextMessage):
            self._stats.model_response_chars += len(event.content)
            self._stats.model_response_tokens = self._stats.model_response_chars // 4
        if event.descriptor in _ALWAYS_FORWARD:
            self._put(event)
            return
        if event.descriptor not in self._forward_set:
            if event.descriptor == "multipart/x-tool-result" and has_error(event):
                pass  # Errors always forwarded.
            else:
                return
        self._put(
            MultipartMessage(
                (
                    TextMessage(self._label, "text/x-agent-label"),
                    event,
                ),
                "multipart/x-child-event",
            )
        )


def _make_parent_forwarder(
    label: str,
    verbosity: int,
    parent_agent: _Agent | None = None,
) -> _ChildForwarder | None:
    """Build a callback that wraps child events with a label and forwards them."""
    if parent_agent is None:
        return None
    forward_set = _VERBOSITY.get(verbosity, _VERBOSITY[1])
    stats = ChildStats(label=label, start=time.monotonic())
    parent_agent.active_children[label] = stats
    return _ChildForwarder(
        parent_agent=parent_agent,
        forward_set=forward_set,
        stats=stats,
        label=label,
    )


_logger = logging.getLogger(__name__)
