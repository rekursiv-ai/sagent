"""AgentSelf tool: mutate the current agent's state.

Unifies all self-mutation operations (status, context management,
model swap) under one tool. Spawning a *new* agent is the
separate ``AgentSpawn`` tool.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, ClassVar, cast

import dataclasses

from sagent import providers
from sagent.custom_types import Message, Model, ModelSpec, TextMessage
from sagent.lib.json import JSON, int_val, json_freeze, json_unfreeze
from sagent.lib.lazy_import import lazy_import
from sagent.lib.message import get_directive
from sagent.providers import (
    PROVIDER_NAMES,
    build_provider,
    infer_provider,
)
from sagent.tools.core import (
    current_agent_var,
    get_tool_state,
    load_tool_description,
)


# Cycle break: ``tools/__init__.py`` imports ``AgentSelf`` and is imported
# by ``agent.agent``; agent_self needs ``PendingOp`` from agent.agent at
# call time. Module proxy defers the load to first attribute access.
_agent_module = lazy_import("sagent.agent.agent")


if TYPE_CHECKING:
    from sagent.agent import Agent


class AgentSelf:
    """Tool: patch the current agent state."""

    name: str = "AgentSelf"
    tool_id: str = "application/x-tool-agentself"
    description: str = load_tool_description("agentself")
    supports_microcompaction: bool = True
    directive_schema: ClassVar[JSON] = json_freeze(
        {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "description": "Optional status text; omit to keep the current status.",
                },
                "context": {
                    "type": "string",
                    "enum": ["clear", "compact", "recompact"],
                    "description": (
                        "Optional context action. Omit to preserve context; use 'clear',"
                        " 'compact', or 'recompact' to queue that context mutation."
                    ),
                },
                "context_prompt": {
                    "type": "string",
                    "description": (
                        "Optional reason or compaction guidance. Only valid when context"
                        " is set."
                    ),
                },
                "model_id": {
                    "type": "string",
                    "description": (
                        "Optional model ID. Provider/auth are inferred from known model"
                        " prefixes when possible."
                    ),
                },
                "provider": {
                    "type": "string",
                    "description": "Optional provider class name override.",
                },
                "auth": {
                    "type": "string",
                    "description": "Optional auth method suffix override.",
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
                        "Optional provider/model-specific settings. Supported keys are"
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
                        " providers, or 'models' with catalog_provider to list known"
                        " models."
                    ),
                },
                "catalog_provider": {
                    "type": "string",
                    "description": (
                        "Provider name for catalog='models'. Omit to use the active"
                        " provider."
                    ),
                },
            },
            "required": [],
            "additionalProperties": False,
        }
    )

    def summary(self, msg: Message) -> str:
        """Return a short label summarizing this self-mutation call.

        Args:
          msg: Tool call message.

        Returns:
          label: Human-readable summary of requested state changes.

        """
        d = get_directive(msg)
        parts = _summary_parts(d)
        return "AgentSelf " + " ".join(parts) if parts else "AgentSelf"

    def summary_result(self, result: Message) -> str | None:
        del result
        return None

    def prompt(self) -> str:
        """Return dynamic system-prompt guidance.

        Returns:
          prompt: Empty; stable AgentSelf guidance lives in tools_agentself.md.

        """
        return ""

    async def run(self, msg: Message) -> Message:
        """Apply an AgentSelf patch object.

        Args:
          msg: Tool call message with optional agent-state fields.

        Returns:
          result: Mutation summary, diagnostics, or error message.

        """
        return _apply_patch(get_directive(msg))


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _ModelPlan:
    model: Model
    spec: ModelSpec
    label: str


_UNSET = object()


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _PatchPlan:
    status: str | None = None
    model: _ModelPlan | None = None
    thinking: bool | None = None
    effort: str | None | object = _UNSET
    cache_ttl: str | None = None
    max_request_tokens: int | None = None
    max_response_tokens: int | None = None
    context: str | None = None
    context_prompt: str = ""


def _summary_parts(d: JSON) -> list[str]:
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


def _apply_patch(d: JSON) -> Message:
    """Validate and apply an AgentSelf patch object."""
    active = current_agent_var.get(None)
    if active is None:
        return TextMessage("No active agent.", "text/x-error")
    patch = json_unfreeze(d)
    if not isinstance(patch, dict):
        return TextMessage("AgentSelf patch must be an object.", "text/x-error")
    agent = active
    plan_or_err = _build_patch_plan(agent, patch)
    if isinstance(plan_or_err, TextMessage):
        return plan_or_err
    parts = _commit_patch_plan(agent, plan_or_err)
    if patch.get("diagnostics") is True:
        return _do_diagnostics(parts, patch)
    return TextMessage(
        "AgentSelf updated: " + ", ".join(parts) if parts else "No changes.",
        "text/plain",
    )


def _build_patch_plan(agent: Agent, d: JSON) -> _PatchPlan | TextMessage:
    """Validate an AgentSelf patch without mutating state."""
    err = _validate_patch(d)
    if err is not None:
        return err
    model_plan = _plan_model(agent, d)
    if isinstance(model_plan, TextMessage):
        return model_plan
    target_model = model_plan.model if model_plan is not None else agent.model
    status = _plan_status(d)
    if isinstance(status, TextMessage):
        return status
    options_or_err = _plan_model_options(target_model, d)
    if isinstance(options_or_err, TextMessage):
        return options_or_err
    options = options_or_err
    thinking = cast(bool | None, options.get("thinking"))
    cache_ttl = cast(str | None, options.get("cache_ttl"))
    has_explicit_limits = "max_request_tokens" in d or "max_response_tokens" in d
    if has_explicit_limits:
        limits = _plan_limits(agent, target_model, d)
        if isinstance(limits, TextMessage):
            return limits
    else:
        limits = {}
    context = cast(str | None, d.get("context"))
    return _PatchPlan(
        status=status,
        model=model_plan,
        thinking=thinking,
        effort=options.get("effort", _UNSET),
        cache_ttl=cache_ttl,
        max_request_tokens=limits.get("max_request_tokens"),
        max_response_tokens=limits.get("max_response_tokens"),
        context=context,
        context_prompt=str(d.get("context_prompt", "")),
    )


def _commit_patch_plan(agent: Agent, plan: _PatchPlan) -> list[str]:
    """Apply a fully validated AgentSelf patch plan."""
    parts: list[str] = []
    if plan.status is not None:
        agent.status = plan.status
        parts.append(f"status={plan.status}")
    if plan.model is not None:
        new_model = plan.model.model
        budget = agent.budget
        if (
            budget.max_request_tokens > new_model.max_request_tokens
            or budget.max_response_tokens > new_model.max_response_tokens
        ):
            agent.reset_budget()
        agent.swap_model(plan.model.model, spec=plan.model.spec)
        parts.append(f"model={plan.model.label}")
        if not plan.model.model.supports_effort and agent.effort is not None:
            agent.effort = None
            parts.append("effort=unset (unsupported)")
        if not plan.model.model.supports_thinking and agent.thinking is not None:
            agent.thinking = None
            parts.append("thinking=off (unsupported)")
    if plan.thinking is not None:
        agent.thinking = "adaptive" if plan.thinking else None
        parts.append(f"thinking={'on' if plan.thinking else 'off'}")
    if plan.effort is not _UNSET:
        effort = cast(str | None, plan.effort)
        agent.effort = effort
        parts.append(f"effort={effort or 'unset'}")
    if plan.cache_ttl is not None:
        agent.cache_ttl = plan.cache_ttl
        parts.append(f"cache_ttl={agent.cache_ttl}")
    if plan.max_request_tokens is not None:
        agent.max_request_tokens = plan.max_request_tokens
        parts.append(f"max_request_tokens={agent.max_request_tokens:,}")
    if plan.max_response_tokens is not None:
        agent.max_response_tokens = plan.max_response_tokens
        parts.append(f"max_response_tokens={agent.max_response_tokens:,}")
    if plan.context is not None:
        _commit_context(plan.context, plan.context_prompt)
        parts.append(f"context={plan.context}")
    return parts


def _validate_patch(d: JSON) -> TextMessage | None:
    """Validate cross-field AgentSelf patch constraints."""
    if "context_prompt" in d and "context" not in d:
        return TextMessage(
            "context_prompt is only valid when context is set.", "text/x-error"
        )
    context = d.get("context")
    if context is not None and context not in ("clear", "compact", "recompact"):
        return TextMessage(f"Invalid context: {context!r}.", "text/x-error")
    options = d.get("model_options")
    if options is not None and not isinstance(options, Mapping):
        return TextMessage("model_options must be an object.", "text/x-error")
    catalog = d.get("catalog")
    if catalog is not None and catalog not in ("providers", "models"):
        return TextMessage(f"Invalid catalog: {catalog!r}.", "text/x-error")
    if "catalog_provider" in d and catalog != "models":
        return TextMessage(
            "catalog_provider is only valid with catalog='models'.", "text/x-error"
        )
    return None


def _plan_status(d: JSON) -> str | TextMessage | None:
    """Validate an optional status update."""
    raw = d.get("status")
    if raw is None:
        return None
    status = str(raw).strip()
    if not status:
        return TextMessage("status cannot be empty when provided.", "text/x-error")
    return status


def _plan_model(agent: Agent, d: JSON) -> _ModelPlan | TextMessage | None:
    """Build an optional model/provider/account update without applying it."""
    if not any(k in d for k in ("model_id", "provider", "auth", "account")):
        return None
    spec = agent.model_spec
    if spec is None:
        return TextMessage("Agent has no model spec; cannot swap.", "text/x-error")
    model_id = str(d.get("model_id", "")).strip() or None
    prov_name = str(d.get("provider", "")).strip() or spec.provider
    auth = str(d.get("auth", "")).strip() or spec.auth
    account = str(d.get("account", "")).strip() or spec.account
    if not model_id and prov_name == spec.provider:
        return TextMessage(
            "model_id is required when changing auth/account without provider.",
            "text/x-error",
        )
    if model_id and prov_name == spec.provider:
        inferred = infer_provider(model_id, prov_name)
        if inferred is not None:
            prov_name, auth = inferred
    try:
        prov = build_provider(prov_name, auth, account=account)
        new_model = prov.model(model_id)
    except (AttributeError, RuntimeError, ValueError) as exc:
        return TextMessage(f"Failed to build model {model_id!r}: {exc}", "text/x-error")
    old_id = agent.model.model_id
    label = f"{old_id} → {new_model.model_id}"
    if prov_name != spec.provider:
        label = f"{spec.provider}/{old_id} → {prov_name}/{new_model.model_id}"
    return _ModelPlan(
        model=new_model,
        spec=dataclasses.replace(
            spec,
            provider=prov_name,
            auth=auth,
            model_id=new_model.model_id,
            account=account,
        ),
        label=label,
    )


def _plan_model_options(model: Model, d: JSON) -> dict[str, object] | TextMessage:
    """Validate provider/model-specific options against the target model."""
    raw = d.get("model_options")
    if raw is None:
        return {}
    options = cast(Mapping[str, object], raw)
    supported = _supported_model_options(model)
    unknown = sorted(set(options) - set(supported))
    if unknown:
        return TextMessage(
            f"Unsupported model_options for {model.model_id}: {', '.join(unknown)}.",
            "text/x-error",
        )
    planned: dict[str, object] = {}
    if "thinking" in options:
        value = options["thinking"]
        if not isinstance(value, bool):
            return TextMessage(
                "model_options.thinking must be boolean.", "text/x-error"
            )
        planned["thinking"] = value
    if "effort" in options:
        value = options["effort"]
        if value is not None and not isinstance(value, str):
            return TextMessage(
                "model_options.effort must be a string or null.", "text/x-error"
            )
        planned["effort"] = value
    if "cache_ttl" in options:
        value = options["cache_ttl"]
        if value not in ("5m", "1h"):
            return TextMessage(
                "model_options.cache_ttl must be '5m' or '1h'.", "text/x-error"
            )
        planned["cache_ttl"] = value
    return planned


def _supported_model_options(model: Model) -> dict[str, str]:
    """Return supported model option names with compact descriptions."""
    supported: dict[str, str] = {}
    if model.supports_thinking:
        supported["thinking"] = "boolean"
    if model.supports_effort:
        supported["effort"] = "string"
    if model.supports_cache_control:
        supported["cache_ttl"] = "'5m' | '1h'"
    return supported


def _plan_limits(agent: Agent, model: Model, d: JSON) -> dict[str, int] | TextMessage:
    """Validate explicit token-limit updates against the target model."""
    limits: dict[str, int] = {}
    max_request_tokens = _plan_one_limit(
        d.get("max_request_tokens"), "max_request_tokens"
    )
    if isinstance(max_request_tokens, TextMessage):
        return max_request_tokens
    max_response_tokens = _plan_one_limit(
        d.get("max_response_tokens"), "max_response_tokens"
    )
    if isinstance(max_response_tokens, TextMessage):
        return max_response_tokens
    if max_request_tokens is not None:
        if max_request_tokens > model.max_request_tokens:
            return TextMessage(
                "Invalid AgentSelf limit override: "
                f"max_request_tokens={max_request_tokens:,} exceeds model's"
                f" {model.max_request_tokens:,}",
                "text/x-error",
            )
        limits["max_request_tokens"] = max_request_tokens
    if max_response_tokens is not None:
        if max_response_tokens > model.max_response_tokens:
            return TextMessage(
                "Invalid AgentSelf limit override: "
                f"max_response_tokens={max_response_tokens:,} exceeds model's"
                f" {model.max_response_tokens:,}",
                "text/x-error",
            )
        limits["max_response_tokens"] = max_response_tokens
    try:
        budget = agent.budget
        if "max_request_tokens" in limits:
            budget = dataclasses.replace(
                budget, max_request_tokens=limits["max_request_tokens"]
            )
        if "max_response_tokens" in limits:
            budget = dataclasses.replace(
                budget, max_response_tokens=limits["max_response_tokens"]
            )
    except (ValueError, TypeError) as exc:
        return TextMessage(f"Invalid AgentSelf limit override: {exc}", "text/x-error")
    return limits


def _plan_one_limit(raw: object, attr: str) -> int | TextMessage | None:
    """Validate a single token limit without applying it."""
    if raw is None:
        return None
    if not isinstance(raw, (int, float, str)):
        return TextMessage(
            f"Invalid AgentSelf limit override: {attr} must be a number.",
            "text/x-error",
        )
    try:
        val = int(raw)
    except ValueError as exc:
        return TextMessage(f"Invalid AgentSelf limit override: {exc}", "text/x-error")
    if val < 1:
        return TextMessage(
            f"Invalid AgentSelf limit override: {attr}={val}. Must be at least 1.",
            "text/x-error",
        )
    return val


def _commit_context(context: str, prompt: str) -> None:
    """Queue a validated context mutation as ``agent._next_op``."""
    agent = current_agent_var.get(None)
    if agent is None:
        return
    if context not in ("clear", "compact", "recompact"):
        return
    if context == "clear":
        agent._next_op = _agent_module.PendingOp(kind="clear", args=prompt)  # noqa: SLF001
    elif context == "compact":
        agent._next_op = _agent_module.PendingOp(kind="compact", args=prompt)  # noqa: SLF001
    elif context == "recompact":
        agent._next_op = _agent_module.PendingOp(kind="recompact", args=prompt)  # noqa: SLF001


def _do_diagnostics(
    changes: list[str] | None = None, d: Mapping[str, object] | None = None
) -> Message:
    """Return current agent diagnostics."""
    agent = current_agent_var.get(None)
    spec = agent.model_spec if agent is not None else None
    stats = dict(get_tool_state().stats)
    lines: list[str] = []
    if changes:
        lines.append("Changes: " + ", ".join(changes))
    if d is not None:
        lines.extend(_catalog_lines(d, agent))
    if stats:
        lines.extend(_format_stats(stats))
    else:
        lines.append(
            "No stats yet - the Agent publishes stats after the first completed model request."
        )
    lines.extend(_spec_lines(spec))
    if agent is not None:
        lines.extend(_agent_option_lines(agent))
        lines.extend(_session_lines(agent))
    return TextMessage("\n".join(lines), "text/plain")


def _catalog_lines(d: Mapping[str, object], agent: Agent | None) -> list[str]:
    """Return read-only provider/model catalog diagnostics."""
    catalog = d.get("catalog")
    if catalog is None:
        return []
    if catalog == "providers":
        return _provider_catalog_lines()
    if catalog == "models":
        provider = str(d.get("catalog_provider") or "").strip()
        if not provider and agent is not None and agent.model_spec is not None:
            provider = agent.model_spec.provider
        return _model_catalog_lines(provider)
    return []


def _provider_catalog_lines() -> list[str]:
    """List providers known to the local build."""
    return ["Known providers: " + ", ".join(PROVIDER_NAMES)]


def _model_catalog_lines(provider_name: str) -> list[str]:
    """List statically known models for one provider."""
    if not provider_name:
        return ["Known models: set catalog_provider or configure an active provider."]
    provider_cls = _provider_class(provider_name)
    if provider_cls is None:
        return [f"Known models: unknown provider {provider_name!r}."]
    default = getattr(provider_cls, "DEFAULT_MODEL", None)
    known = getattr(provider_cls, "KNOWN_MODELS", None)
    lines = [f"Provider catalog: {provider_name}"]
    if isinstance(default, str):
        lines.append(f"Default model: {default}")
    if isinstance(known, Mapping):
        known_models = cast(Mapping[object, object], known)
        model_ids = [str(k) for k in known_models]
        models = ", ".join(sorted(model_ids))
        lines.append(f"Known models: {models or 'none'}")
    else:
        lines.append("Known models: unavailable for this provider.")
    return lines


def _provider_class(provider_name: str) -> type[object] | None:
    """Return the provider class object by public provider name."""
    cls = getattr(providers, provider_name, None)
    return cls if isinstance(cls, type) else None


def _format_stats(stats: dict[str, float | int]) -> list[str]:
    """Format session stats into display lines."""
    max_req = int_val(stats.get("max_request_tokens"), 0)
    max_resp = int_val(stats.get("max_response_tokens"), 0)
    input_tokens = int_val(stats.get("input_tokens"), 0)
    pct = (input_tokens / max_req * 100) if max_req else 0.0
    total_in = int_val(stats.get("total_input_tokens"), 0)
    total_out = int_val(stats.get("total_output_tokens"), 0)
    return [
        f"Tool call rounds:   {int_val(stats.get('num_tool_call_rounds'), 0)}",
        f"Max request tokens:   {max_req:,}",
        f"Max response tokens:  {max_resp:,}",
        f"Input tokens:       {input_tokens:,} ({pct:.1f}% of max request)",
        f"Total input tokens: {total_in:,}",
        f"Total output tokens:{total_out:,}",
        f"Cache creation:     {int_val(stats.get('cache_creation_tokens'), 0):,}",
        f"Cache read:         {int_val(stats.get('cache_read_tokens'), 0):,}",
        f"Total cost (USD):   ${cast(float, stats.get('total_cost_usd') or 0.0):.2f}",
    ]


def _spec_lines(spec: ModelSpec | None) -> list[str]:
    """Format model spec into display lines."""
    if spec is None:
        return []
    return [
        f"Provider:           {spec.provider}",
        f"Auth:               {spec.auth}",
        f"Model:              {spec.model_id}",
        f"Account:            {spec.account or 'default'}",
    ]


def _agent_option_lines(agent: Agent) -> list[str]:
    """Format current and supported model options."""
    thinking = "on" if agent.thinking is not None else "off"
    if not agent.model.supports_thinking:
        thinking = "unsupported"
    effort = agent.effort or "unset"
    if not agent.model.supports_effort:
        effort = "unsupported"
    supported = _supported_model_options(agent.model)
    supported_text = ", ".join(f"{k}: {v}" for k, v in supported.items()) or "none"
    return [
        f"Cache TTL:          {agent.cache_ttl}",
        f"Thinking:           {thinking}",
        f"Effort:             {effort}",
        f"Supported model_options: {supported_text}",
    ]


def _session_lines(agent: Agent) -> list[str]:
    """Format session identity, path, and cwd."""
    if agent.session_dir is not None:
        lines = [f"Session:            {agent.session_dir.name}"]
        lines.append(f"Session dir:        {agent.session_dir}")
    else:
        lines = [f"Session:            {agent.session_id} (ephemeral)"]
    lines.append(f"Bash cwd:           {agent.tool_state.bash_cwd}")
    return lines
