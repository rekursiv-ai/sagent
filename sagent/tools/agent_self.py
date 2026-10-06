"""AgentSelf tool: mutate the current agent's state.

Unifies all self-mutation operations (status, context management,
model swap) under one tool. Spawning a *new* agent is the
separate ``AgentSpawn`` tool.

Context verbs (``clear`` / ``compact`` / ``recompact``) are
dispatched by pushing them directly into ``agent.runtime.inbox`` as
first-class runtime events; the agent's loop handles them in turn.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextvars import ContextVar
from functools import partial
from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol, cast

import asyncio
import dataclasses

from sagent.agent.agent import ModelChange
from sagent.agent.state import (
    AgentLike,
    current_agent_var,
)
from sagent.catalog.table import (
    CONTEXT_TAGS,
    UnknownModelError,
    UnsupportedTagError,
    base_model_id,
)
from sagent.lib.custom_json import JSON, json_freeze
from sagent.providers.providers import (
    PROVIDER_NAMES,
    infer_provider,
    provider_class,
)
from sagent.thinking import apply_thinking_command
from sagent.tools.core import (
    load_tool_description,
    provider_not_allowed_result,
)
from sagent.types.capability import ModelSettings, ThinkingEffort
from sagent.types.cost import ServiceTier
from sagent.types.providers import ModelResolver, Provider
from sagent.types.runtime import (
    Clear,
    Compact,
    Recompact,
    ToolResult,
)


if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from sagent.agent.cost_tracker import CostTracker
    from sagent.agent.state import ToolState
    from sagent.types.model import Model, ModelRecipe


class AgentSelfAgent(AgentLike, Protocol):
    """Agent surface mutated by ``AgentSelf``."""

    model: Model
    model_recipe: ModelRecipe | None
    status: str
    max_request_tokens: int
    max_response_tokens: int
    cost_tracker: CostTracker
    num_tool_call_rounds: int
    session_dir: Path | None
    session_id: str
    tool_state: ToolState

    def prepare_model_change(  # noqa: D102 -- Protocol member declaration.
        self,
        *,
        provider: str | None = None,
        auth: str | None = None,
        model_id: str | None = None,
        account: str | None = None,
    ) -> ModelChange: ...

    def commit_model_change(  # noqa: D102 -- Protocol member declaration.
        self,
        change: ModelChange,
        *,
        then: Callable[[], None] | None = None,
    ) -> None: ...


CACHE_TTL_SEC: Mapping[str, float] = MappingProxyType({"5m": 300.0, "1h": 3600.0})
"""The two prompt-cache lifetimes the wire spells, in seconds."""

_discard_tasks: set[asyncio.Task[None]] = set()


# Per-call provider allow-list, set by ``AgentSelf.run`` so the module-level catalog
# helpers (``_allowed_providers``, ``_model_catalog_lines``) can filter without
# threading the list through every call.
_allow_providers_var: ContextVar[tuple[str, ...]] = ContextVar(
    "_allow_providers",
    default=(),
)


class AgentSelf:
    """Tool: patch the current agent state."""

    name: str = "AgentSelf"
    tool_id: str = "application/x-tool-agentself"
    clearable_results: bool = False
    description: str = load_tool_description("agentself")
    directive_schema: JSON = json_freeze(
        {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "description": (
                        "Optional status text; omit to keep the current status."
                    ),
                },
                "context": {
                    "type": "string",
                    "enum": ["clear", "compact", "recompact"],
                    "description": (
                        "Optional context action. Omit to preserve context;"
                        " use 'clear', 'compact', or 'recompact' to queue"
                        " that context mutation."
                    ),
                },
                "context_prompt": {
                    "type": "string",
                    "description": (
                        "Optional reason or compaction guidance."
                        " Only valid when context is set."
                    ),
                },
                "model_id": {
                    "type": "string",
                    "description": (
                        "Optional model ID. Provider/auth are inferred from known"
                        " model prefixes when possible."
                    ),
                },
                "provider": {
                    "type": "string",
                    "description": "Optional provider class name override.",
                },
                "auth": {
                    "type": "string",
                    "description": (
                        "Optional auth method suffix override (e.g. 'env' or"
                        " 'credentials')."
                    ),
                },
                "account": {
                    "type": "string",
                    "description": "Optional credential account name.",
                },
                "max_request_tokens": {
                    "type": "integer",
                    "minimum": 1,
                    "description": (
                        "Optional request-token limit. Omit to keep the current limit."
                    ),
                },
                "max_response_tokens": {
                    "type": "integer",
                    "minimum": 1,
                    "description": (
                        "Optional response-token limit. Omit to keep the current limit."
                    ),
                },
                "model_options": {
                    "type": "object",
                    "description": (
                        "Optional provider/model-specific settings:"
                        " 'thinking', 'effort', 'cache_ttl', 'service_tier'."
                        " Request fast serving with"
                        " service_tier='priority' on supported models."
                        " Supported keys per model are"
                        " reported by diagnostics."
                    ),
                    "additionalProperties": True,
                },
                "diagnostics": {
                    "type": "boolean",
                    "description": "Include current diagnostics in the result.",
                },
                "catalog": {
                    "type": "string",
                    "enum": ["providers", "models"],
                    "description": (
                        "Read-only catalog query. Use 'providers' to list known"
                        " providers, or 'models' with catalog_provider to list"
                        " known models."
                    ),
                },
                "catalog_provider": {
                    "type": "string",
                    "description": (
                        "Provider name for catalog='models'."
                        " Omit to use the active provider."
                    ),
                },
            },
            "required": [],
            "additionalProperties": False,
        },
    )

    def __init__(
        self,
        *,
        allow_providers: tuple[str, ...] | None = None,
    ) -> None:
        self._allow_providers: tuple[str, ...] = (
            tuple(allow_providers) if allow_providers is not None else PROVIDER_NAMES
        )

    def summary(self, args: Mapping[str, object]) -> str:
        """Return a short label summarizing this self-mutation call.

        Args:
          args: Parsed tool directive mapping.

        Returns:
          label: Compact one-line label for renderer display.

        """
        parts = _summary_parts(args)
        return "AgentSelf " + " ".join(parts) if parts else "AgentSelf"

    def prompt(self) -> str:
        """Return dynamic system-prompt guidance.

        Returns:
          text: Supplemental prompt text; empty for AgentSelf.

        """
        return ""

    def serialize_key(self, args: Mapping[str, object]) -> str | None:
        """Run in parallel: self-patching has no shared file resource."""
        del args
        return None

    async def run(self, args: Mapping[str, object]) -> ToolResult:
        """Apply an AgentSelf patch object.

        Args:
          args: Parsed tool directive mapping.

        Returns:
          result: Outcome of the patch (summary text or error).

        """
        token = _allow_providers_var.set(self._allow_providers)
        try:
            return _apply_patch(args)
        finally:
            _allow_providers_var.reset(token)


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _PatchPlan:
    """Validated AgentSelf patch ready to commit."""

    status: str | None = None
    """New status string (rendered in the status pane), or ``None`` to keep."""

    model: ModelChange | None = None
    """Prepared model swap, or ``None`` to keep."""

    model_options: Mapping[str, object] = dataclasses.field(
        default_factory=dict[str, object],
    )
    """Normalized concrete model settings; absent keys remain unchanged."""

    max_request_tokens: int | None = None
    """New per-request input budget; ``None`` to keep."""

    max_response_tokens: int | None = None
    """New per-request response budget; ``None`` to keep."""

    context: str | None = None
    """Context verb (``clear`` / ``compact`` / ``recompact``); ``None`` to keep."""

    context_prompt: str = ""
    """Free-form guidance forwarded to the context verb."""


def plan_model_options(
    model: Model,
    d: Mapping[str, object],
) -> dict[str, object] | ToolResult:
    """Validate provider/model-specific options against the target model.

    Shared with :class:`AgentSpawn`, which applies the validated options
    to a freshly built child agent.

    Args:
      model: Target model whose capabilities constrain the options.
      d: AgentSelf directive containing optional model options.

    Returns:
      options_or_error: Validated options or an error result.

    """
    raw = d.get("model_options")
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        return ToolResult(
            call_id="",
            content="model_options must be an object.",
            is_error=True,
        )
    untyped_options = cast(Mapping[object, object], raw)
    options: dict[str, object] = {}
    for key, value in untyped_options.items():
        if not isinstance(key, str):
            return ToolResult(
                call_id="",
                content="model_options keys must be strings.",
                is_error=True,
            )
        options[key] = value
    if "latency" in options:
        # Redirect beats the generic unsupported-key error: this knob was
        # renamed rather than removed.
        return ToolResult(
            call_id="",
            content=(
                "model_options.latency was replaced by service_tier"
                " (e.g. model_options={'service_tier': 'priority'})."
            ),
            is_error=True,
        )
    supported = _supported_model_options(model)
    # A key is unsupported only when it carries a *non-null* value the model
    # can't honor. Clearing an option to ``null`` (e.g. service_tier) is a
    # valid request the per-field validation handles, so a null must never
    # trip this capability gate.
    unknown = sorted(
        k for k in options if k not in supported and options[k] is not None
    )
    if unknown:
        return ToolResult(
            call_id="",
            content=f"Unsupported model_options for {model.tagged_model_id}: {', '.join(unknown)}.",
            is_error=True,
        )
    planned: dict[str, object] = {}
    if "thinking" in options:
        value = options["thinking"]
        if not isinstance(value, bool):
            return ToolResult(
                call_id="",
                content="model_options.thinking must be boolean.",
                is_error=True,
            )
        planned["thinking"] = value
    if "effort" in options:
        raw_value = options["effort"]
        value = "none" if raw_value is None else raw_value
        if not isinstance(value, str):
            return ToolResult(
                call_id="",
                content="model_options.effort must be a string or null.",
                is_error=True,
            )
        valid = model.capability.thinking.effort
        if value not in valid:
            quoted = ", ".join(repr(e) for e in valid) or "(none)"
            return ToolResult(
                call_id="",
                content=(
                    f"model_options.effort for {model.tagged_model_id} must"
                    f" be one of {quoted} or null, got {value!r}."
                ),
                is_error=True,
            )
        planned["effort"] = value
    if "cache_ttl" in options:
        raw_value = options["cache_ttl"]
        if raw_value is None:
            value = 0.0
        elif isinstance(raw_value, str) and raw_value in CACHE_TTL_SEC:
            value = CACHE_TTL_SEC[raw_value]
        else:
            quoted = ", ".join(repr(k) for k in CACHE_TTL_SEC)
            return ToolResult(
                call_id="",
                content=f"model_options.cache_ttl must be one of {quoted} or null.",
                is_error=True,
            )
        if value not in model.capability.cache_ttl_sec:
            return ToolResult(
                call_id="",
                content=(
                    f"model_options.cache_ttl for {model.tagged_model_id}"
                    f" does not support {raw_value!r}."
                ),
                is_error=True,
            )
        planned["cache_ttl_sec"] = value
    if "service_tier" in options:
        raw_value = options["service_tier"]
        value = "auto" if raw_value is None else raw_value
        if not isinstance(value, str):
            return ToolResult(
                call_id="",
                content="model_options.service_tier must be a string or null.",
                is_error=True,
            )
        valid = model.capability.service_tier
        if value not in valid:
            quoted = ", ".join(repr(t) for t in sorted(valid))
            return ToolResult(
                call_id="",
                content=(
                    f"model_options.service_tier for {model.tagged_model_id} must"
                    f" be one of {quoted} or null, got {value!r}."
                ),
                is_error=True,
            )
        planned["service_tier"] = value
    return planned


# A tool builds a provider before it knows the request will be honored; when it is
# refused, the model and provider belong to no agent and nothing else would close them.
# The task is held until done so it is not garbage-collected mid-close.
def discard_unadopted(model: Model | None, provider: Provider) -> None:
    """Schedule teardown of a model and provider no agent adopted.

    Shared with :class:`AgentSpawn`, which builds a child's provider before
    validating the rest of the spawn.

    Args:
      model: The model built from ``provider``, or ``None`` if building failed.
      provider: The provider to ``close_sdk``.

    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # Outside a loop no client has been opened yet, so nothing leaks.
        return
    task = loop.create_task(_close_built(model, provider))
    _discard_tasks.add(task)
    task.add_done_callback(_discard_tasks.discard)


# Every axis is total, so a non-empty set proves nothing: an axis is selectable only
# when it offers something BESIDES its unset value.
def _supported_model_options(model: Model) -> dict[str, str]:
    """Return supported model option names with compact descriptions."""
    supported: dict[str, str] = {}
    capability = model.capability
    if capability.thinking.budget != frozenset({"none"}):
        supported["thinking"] = "boolean"
    efforts = capability.thinking.effort - {"none"}
    if efforts:
        supported["effort"] = " | ".join(repr(e) for e in sorted(efforts))
    if capability.cache_ttl_sec != frozenset({0.0}):
        supported["cache_ttl"] = " | ".join(
            repr(k) for k, v in CACHE_TTL_SEC.items() if v in capability.cache_ttl_sec
        )
    tiers = capability.service_tier - {"auto"}
    if tiers:
        supported["service_tier"] = " | ".join(
            repr(t) for t in sorted(capability.service_tier)
        )
    return supported


def _plan_limits(
    agent: AgentSelfAgent,
    model: Model,
    d: Mapping[str, object],
) -> dict[str, int] | ToolResult:
    """Validate explicit token-limit updates against the target model."""
    limits: dict[str, int] = {}
    max_request_tokens = _plan_one_limit(
        d.get("max_request_tokens"),
        "max_request_tokens",
    )
    if isinstance(max_request_tokens, ToolResult):
        return max_request_tokens
    max_response_tokens = _plan_one_limit(
        d.get("max_response_tokens"),
        "max_response_tokens",
    )
    if isinstance(max_response_tokens, ToolResult):
        return max_response_tokens
    # A ceiling of ``0`` is "unknown", as ``Agent.max_request_tokens``'s setter reads
    # it: there is no cap to exceed.
    if max_request_tokens is not None:
        if 0 < model.limits.max_request_tokens < max_request_tokens:
            return ToolResult(
                call_id="",
                content=(
                    "Invalid AgentSelf limit override: "
                    f"max_request_tokens={max_request_tokens:,} exceeds model's"
                    f" {model.limits.max_request_tokens:,}"
                    + _window_variant_hint(agent, model, max_request_tokens)
                ),
                is_error=True,
            )
        limits["max_request_tokens"] = max_request_tokens
    if max_response_tokens is not None:
        if 0 < model.limits.max_response_tokens < max_response_tokens:
            return ToolResult(
                call_id="",
                content=(
                    "Invalid AgentSelf limit override: "
                    f"max_response_tokens={max_response_tokens:,} exceeds model's"
                    f" {model.limits.max_response_tokens:,}"
                ),
                is_error=True,
            )
        limits["max_response_tokens"] = max_response_tokens
    return limits


def _plan_one_limit(raw: object, attr: str) -> int | ToolResult | None:
    """Validate a single token limit without applying it."""
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, (int, float, str)):
        return ToolResult(
            call_id="",
            content=f"Invalid AgentSelf limit override: {attr} must be a number.",
            is_error=True,
        )
    # Schema declares token limits as ``type: integer``; ``int(1.9)``
    # would silently round to 1 and accept the request. A float here
    # is a directive error -- reject so the LLM doesn't see its
    # ``1.9`` request quietly become ``1``.
    if isinstance(raw, float) and not raw.is_integer():
        return ToolResult(
            call_id="",
            content=(
                f"Invalid AgentSelf limit override: {attr} must be an"
                f" integer, got {raw!r}."
            ),
            is_error=True,
        )
    try:
        val = int(raw)
    except ValueError as exc:
        return ToolResult(
            call_id="",
            content=f"Invalid AgentSelf limit override: {exc}",
            is_error=True,
        )
    if val < 1:
        return ToolResult(
            call_id="",
            content=(
                f"Invalid AgentSelf limit override: {attr}={val}. Must be at least 1."
            ),
            is_error=True,
        )
    return val


# Context-window size is selected by the model id, not by raising
# ``max_request_tokens``. Bare IDs use the largest profile; a tag may select a
# smaller one. Point an over-limit request at the first sibling profile that fits.
def _window_variant_hint(agent: AgentSelfAgent, model: Model, requested: int) -> str:
    """Suggest a larger-window model id when one would satisfy ``requested``."""
    spec = agent.model_recipe
    if spec is None:
        return ""
    provider_cls = provider_class(spec.provider)
    if not isinstance(provider_cls, ModelResolver):
        return ""
    base = base_model_id(model.tagged_model_id)
    try:
        cap, _ = provider_cls.catalog.resolve(base)
    except (UnknownModelError, UnsupportedTagError):
        return ""
    candidates = (
        (base, cap.context.get("")),
        *((base + tag, cap.context.get(tag)) for tag in CONTEXT_TAGS),
    )
    for candidate, limits in candidates:
        window = getattr(limits, "max_request_tokens", 0)
        if candidate != model.tagged_model_id and window >= requested:
            return (
                f". The window is part of the model id: switch to"
                f" model_id={candidate} (a {window:,}-token window) rather"
                f" than raising max_request_tokens"
            )
    return ""


# Context verbs are first-class ``RuntimeEvent``s on the inbox; the runtime loop
# dispatches them in arrival order.
def _commit_context(agent: AgentSelfAgent, context: str, prompt: str) -> None:
    """Push a validated context mutation directly to the runtime inbox."""
    if context == "clear":
        agent.runtime.inbox.push_back(Clear())
    elif context == "compact":
        agent.runtime.inbox.push_back(Compact(args=prompt))
    elif context == "recompact":
        agent.runtime.inbox.push_back(Recompact(args=prompt))


def _do_diagnostics(
    changes: list[str] | None = None,
    d: Mapping[str, object] | None = None,
) -> ToolResult:
    """Return current agent diagnostics."""
    agent = cast(AgentSelfAgent | None, current_agent_var.get(None))
    spec = agent.model_recipe if agent is not None else None
    lines: list[str] = []
    if changes:
        lines.append("Changes: " + ", ".join(changes))
    if d is not None:
        lines.extend(_catalog_lines(d, agent))
    if agent is not None:
        lines.extend(_format_stats(agent))
    else:
        lines.append("No agent context; stats unavailable.")
    lines.extend(_spec_lines(spec))
    if agent is not None:
        lines.extend(_agent_option_lines(agent))
        lines.extend(_session_lines(agent))
    return ToolResult(call_id="", content="\n".join(lines))


def _catalog_lines(d: Mapping[str, object], agent: AgentSelfAgent | None) -> list[str]:
    """Return read-only provider/model catalog diagnostics."""
    catalog = d.get("catalog")
    if catalog is None:
        return []
    if catalog == "providers":
        return ["Known providers: " + ", ".join(_allowed_providers())]
    if catalog == "models":
        provider = str(d.get("catalog_provider") or "").strip()
        if not provider and agent is not None and agent.model_recipe is not None:
            provider = agent.model_recipe.provider
        return _model_catalog_lines(provider)
    return []


def _model_catalog_lines(provider_name: str) -> list[str]:
    """List statically known models for one provider, gated by allow-list."""
    if not provider_name:
        return ["Known models: set catalog_provider or configure an active provider."]
    provider_cls = provider_class(provider_name)
    if provider_cls is None:
        return [f"Known models: unknown provider {provider_name!r}."]
    if provider_name not in _allowed_providers():
        return [
            (
                f"Known models: {provider_name!r} is not in the allowed list"
                f" {list(_allowed_providers())}."
            ),
        ]
    lines = [f"Provider catalog: {provider_name}"]
    if isinstance(provider_cls, ModelResolver):
        table = provider_cls.catalog.models
        roles = ", ".join(f"{role}={table[role].model_id}" for role in table.roles)
        lines.append(f"Roles: {roles or 'none'}")
        lines.append(f"Known models: {', '.join(table) or 'none'}")
    else:
        lines.append("Known models: unavailable for this provider.")
    return lines


# Reads directly from the single cost store (``agent.cost_tracker``) plus the live
# budget/round counters on the agent. No separate per-request publisher exists;
# diagnostics is a pull, not a push.
def _format_stats(agent: AgentSelfAgent) -> list[str]:
    """Format live cost/budget counters into display lines."""
    tracker = agent.cost_tracker
    max_req = agent.max_request_tokens
    max_resp = agent.max_response_tokens
    input_tokens = tracker.last_request.request
    pct = (input_tokens / max_req * 100) if max_req else 0.0
    return [
        f"Tool call rounds:   {agent.num_tool_call_rounds}",
        f"Max request tokens:   {max_req:,}",
        f"Max response tokens:  {max_resp:,}",
        f"Input tokens:       {input_tokens:,} ({pct:.1f}% of max request)",
        f"Total input tokens: {tracker.total.request:,}",
        f"Total output tokens:{tracker.total.response:,}",
        f"Cache creation 5m:  {tracker.total.cache_write:,}",
        f"Cache creation 1h:  {tracker.total.cache_write_1h:,}",
        f"Cache read:         {tracker.total.cache_read:,}",
        f"Total cost (USD):   ${tracker.spend.total:.2f}",
    ]


def _spec_lines(spec: ModelRecipe | None) -> list[str]:
    """Format model spec into display lines."""
    if spec is None:
        return []
    return [
        f"Provider:           {spec.provider}",
        f"Auth:               {spec.auth}",
        f"Model:              {spec.model_id}",
        f"Account:            {spec.account or 'default'}",
    ]


def _agent_option_lines(agent: AgentSelfAgent) -> list[str]:
    """Format current and supported model options."""
    # Each axis is total, so "unsupported" is the SINGLETON unset value, not
    # an empty set: ``not bool(frozenset)`` was never true and every model
    # reported every knob as supported.
    capability = agent.model.capability
    settings = agent.model.settings
    budget = settings.thinking_budget
    thinking = "off" if budget == "none" else budget
    if capability.thinking.budget == frozenset({"none"}):
        thinking = "unsupported"
    effort = settings.thinking_effort
    if capability.thinking.effort == frozenset({"none"}):
        effort = "unsupported"
    service_tier = settings.service_tier
    if capability.service_tier == frozenset({"auto"}):
        service_tier = "unsupported"
    cache_ttl = (
        "unsupported"
        if capability.cache_ttl_sec == frozenset({0.0})
        else f"{settings.cache_ttl_sec:g}s"
    )
    supported = _supported_model_options(agent.model)
    supported_text = ", ".join(f"{k}: {v}" for k, v in supported.items()) or "none"
    return [
        f"Cache TTL:          {cache_ttl}",
        f"Thinking:           {thinking}",
        f"Effort:             {effort}",
        f"Service tier:       {service_tier}",
        f"Supported model_options: {supported_text}",
    ]


def _session_lines(agent: AgentSelfAgent) -> list[str]:
    """Format session identity, path, and cwd."""
    if agent.session_dir is not None:
        lines = [f"Session:            {agent.session_dir.name}"]
        lines.append(f"Session dir:        {agent.session_dir}")
    else:
        lines = [f"Session:            {agent.session_id} (ephemeral)"]
    lines.append(f"Bash cwd:           {agent.tool_state.bash_cwd}")
    return lines


def _summary_parts(d: Mapping[str, object]) -> list[str]:
    """Return compact summary fragments for an AgentSelf patch."""
    parts: list[str] = []
    if d.get("status") is not None:
        parts.append(f"status={d.get('status')}")
    if d.get("context") is not None:
        parts.append(f"context={d.get('context')}")
    model = d.get("model_id")
    provider = d.get("provider")
    if model or provider:
        parts.append(f"model={provider or ''}/{model or ''}".strip("/"))
    if d.get("max_request_tokens") is not None:
        parts.append(f"max_request_tokens={d.get('max_request_tokens')}")
    if d.get("max_response_tokens") is not None:
        parts.append(f"max_response_tokens={d.get('max_response_tokens')}")
    if d.get("model_options") is not None:
        parts.append("model_options")
    if d.get("diagnostics"):
        parts.append("diagnostics")
    if d.get("catalog") is not None:
        parts.append(f"catalog={d.get('catalog')}")
    return parts


def _apply_patch(d: Mapping[str, object]) -> ToolResult:
    """Validate and apply an AgentSelf patch object."""
    active = current_agent_var.get(None)
    if active is None:
        return ToolResult(call_id="", content="No active agent.", is_error=True)
    agent = cast(AgentSelfAgent, active)
    plan_or_err = _build_patch_plan(agent, d)
    if isinstance(plan_or_err, ToolResult):
        return plan_or_err
    parts = _commit_patch_plan(agent, plan_or_err)
    if d.get("diagnostics") is True:
        return _do_diagnostics(parts, d)
    if d.get("catalog") is not None:
        # Read-only catalog query without diagnostics: render only the
        # catalog lines (plus any patch confirmation prefix).
        lines: list[str] = []
        if parts:
            lines.append("AgentSelf updated: " + ", ".join(parts))
        lines.extend(_catalog_lines(d, agent))
        return ToolResult(call_id="", content="\n".join(lines))
    return ToolResult(
        call_id="",
        content="AgentSelf updated: " + ", ".join(parts) if parts else "No changes.",
    )


def _build_patch_plan(
    agent: AgentSelfAgent,
    d: Mapping[str, object],
) -> _PatchPlan | ToolResult:
    """Validate an AgentSelf patch without mutating state."""
    err = _validate_patch(d)
    if err is not None:
        return err
    status = _plan_status(d)
    if isinstance(status, ToolResult):
        return status
    model_plan = _plan_model(agent, d)
    if isinstance(model_plan, ToolResult):
        return model_plan
    target_model = model_plan.model if model_plan is not None else agent.model
    options = plan_model_options(target_model, d)
    limits: dict[str, int] | ToolResult = {}
    if not isinstance(options, ToolResult) and (
        "max_request_tokens" in d or "max_response_tokens" in d
    ):
        limits = _plan_limits(agent, target_model, d)
    refused = isinstance(options, ToolResult) or isinstance(limits, ToolResult)
    if refused and model_plan is not None:
        # The built model was never adopted, so nothing else would close it.
        model_plan.discard()
    if isinstance(options, ToolResult):
        return options
    if isinstance(limits, ToolResult):
        return limits
    context = cast(str | None, d.get("context"))
    return _PatchPlan(
        status=status,
        model=model_plan,
        model_options=options,
        max_request_tokens=limits.get("max_request_tokens"),
        max_response_tokens=limits.get("max_response_tokens"),
        context=context,
        context_prompt=cast(str, d.get("context_prompt", "")),
    )


# Without a model change everything applies now. With one, the swap is queued like
# ``/model`` and the settings ride it, applying to the NEW model once the swap lands;
# the returned parts describe that queued state.
def _commit_patch_plan(agent: AgentSelfAgent, plan: _PatchPlan) -> list[str]:
    """Apply a fully validated AgentSelf patch plan."""
    parts: list[str] = []
    if plan.status is not None:
        agent.status = plan.status
        parts.append(f"status={plan.status}")
    if plan.model is None:
        _apply_settings(agent, plan)
        parts.extend(_describe_settings(plan))
    else:
        parts.append(f"model={plan.model.label} (queued)")
        # ``swap_model`` carries the selections across and drops the ones the
        # new model rejects. Report that drop from the same ``adopt`` rule,
        # previewed on the model in hand: each axis is total, so a capability
        # check would call every knob supported.
        incoming = plan.model.model.settings
        preview = ModelSettings.narrowest(
            incoming.capability,
            context=incoming.context,
        )
        preview.adopt(agent.model.settings)
        for name in ("thinking_effort", "thinking_budget", "service_tier"):
            was, now = getattr(agent.model.settings, name), getattr(preview, name)
            if was != now:
                parts.append(f"{name}={now} (unsupported)")
        parts.extend(_describe_settings(plan))
        agent.commit_model_change(
            plan.model,
            then=partial(_apply_settings, agent, plan),
        )
    if plan.context is not None:
        _commit_context(agent, plan.context, plan.context_prompt)
        parts.append(f"context={plan.context}")
    return parts


def _apply_settings(agent: AgentSelfAgent, plan: _PatchPlan) -> None:
    """Apply the plan's options and limits to the agent's current model."""
    settings = agent.model.settings
    options = plan.model_options
    if "thinking" in options:
        _ = apply_thinking_command(
            "adaptive" if options["thinking"] else "off",
            settings,
            show=False,
        )
    if "effort" in options:
        settings.thinking_effort = cast(ThinkingEffort, options["effort"])
    if "cache_ttl_sec" in options:
        settings.cache_ttl_sec = cast(float, options["cache_ttl_sec"])
    if "service_tier" in options:
        settings.service_tier = cast(ServiceTier, options["service_tier"])
    if plan.max_request_tokens is not None:
        agent.max_request_tokens = plan.max_request_tokens
    if plan.max_response_tokens is not None:
        agent.max_response_tokens = plan.max_response_tokens


def _describe_settings(plan: _PatchPlan) -> list[str]:
    """Render the plan's options and limits as ``name=value`` parts."""
    options = plan.model_options
    parts: list[str] = []
    if "thinking" in options:
        parts.append(f"thinking={'on' if options['thinking'] else 'off'}")
    if "effort" in options:
        parts.append(f"effort={options['effort']}")
    if "cache_ttl_sec" in options:
        parts.append(f"cache_ttl={cast(float, options['cache_ttl_sec']):g}s")
    if "service_tier" in options:
        parts.append(f"service_tier={options['service_tier']}")
    if plan.max_request_tokens is not None:
        parts.append(f"max_request_tokens={plan.max_request_tokens:,}")
    if plan.max_response_tokens is not None:
        parts.append(f"max_response_tokens={plan.max_response_tokens:,}")
    return parts


def _validate_patch(d: Mapping[str, object]) -> ToolResult | None:
    """Validate cross-field AgentSelf patch constraints."""
    if "context_prompt" in d and not isinstance(d["context_prompt"], str):
        return ToolResult(
            call_id="",
            content="context_prompt must be a string.",
            is_error=True,
        )
    if "context_prompt" in d and "context" not in d:
        return ToolResult(
            call_id="",
            content="context_prompt is only valid when context is set.",
            is_error=True,
        )
    context = d.get("context")
    if context is not None and context not in ("clear", "compact", "recompact"):
        return ToolResult(
            call_id="",
            content=f"Invalid context: {context!r}.",
            is_error=True,
        )
    options = d.get("model_options")
    if options is not None and not isinstance(options, Mapping):
        return ToolResult(
            call_id="",
            content="model_options must be an object.",
            is_error=True,
        )
    catalog = d.get("catalog")
    if catalog is not None and catalog not in ("providers", "models"):
        return ToolResult(
            call_id="",
            content=f"Invalid catalog: {catalog!r}.",
            is_error=True,
        )
    if "catalog_provider" in d and catalog != "models":
        return ToolResult(
            call_id="",
            content="catalog_provider is only valid with catalog='models'.",
            is_error=True,
        )
    return None


def _plan_status(d: Mapping[str, object]) -> str | ToolResult | None:
    """Validate an optional status update."""
    raw = d.get("status")
    if raw is None:
        return None
    if not isinstance(raw, str):
        return ToolResult(
            call_id="",
            content="status must be a string.",
            is_error=True,
        )
    status = raw.strip()
    if not status:
        return ToolResult(
            call_id="",
            content="status cannot be empty when provided.",
            is_error=True,
        )
    return status


# Resolution and building are ``Agent.prepare_model_change``'s; this adds only what
# the tool surface owns: the provider allow-list, inferring a provider from a bare
# model id, and the empty-account error.
def _plan_model(
    agent: AgentSelfAgent,
    d: Mapping[str, object],
) -> ModelChange | ToolResult | None:
    """Build an optional model/provider/account update without applying it."""
    if not any(k in d for k in ("model_id", "provider", "auth", "account")):
        return None
    spec = agent.model_recipe
    if spec is None:
        return ToolResult(
            call_id="",
            content="Agent has no model spec; cannot swap.",
            is_error=True,
        )
    model_id = str(d.get("model_id", "")).strip() or None
    provider = str(d.get("provider", "")).strip() or None
    auth = str(d["auth"]).strip() if "auth" in d else None
    account = str(d["account"]).strip() if "account" in d else None
    if account == "":
        return ToolResult(call_id="", content="account cannot be empty.", is_error=True)
    # Infer only when the caller named no provider: an explicit provider, even
    # the current one, is a choice inference must not override.
    if model_id and provider is None:
        inferred = infer_provider(model_id, spec.provider)
        if inferred is not None:
            provider, inferred_auth = inferred
            auth = auth if auth is not None else inferred_auth
    allow = _allowed_providers()
    if provider not in (None, spec.provider) and provider not in allow:
        return provider_not_allowed_result(provider, allow, spec.provider)
    try:
        return agent.prepare_model_change(
            provider=provider,
            auth=auth,
            model_id=model_id,
            account=account,
        )
    except (AttributeError, FileNotFoundError, RuntimeError, ValueError) as exc:
        return ToolResult(
            call_id="",
            content=f"Failed to build model {model_id or spec.model_id!r}: {exc}",
            is_error=True,
        )


async def _close_built(model: Model | None, provider: Provider) -> None:
    try:
        if model is not None:
            await model.close()
    finally:
        await provider.close_sdk()


def _allowed_providers() -> tuple[str, ...]:
    """Return the active allow-list, falling back to known providers."""
    return _allow_providers_var.get() or PROVIDER_NAMES
