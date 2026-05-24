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
from typing import TYPE_CHECKING, ClassVar, cast

import dataclasses

from sagent import (
    providers as providers_module,
    types,
)
from sagent.lib.json import JSON, int_val, json_freeze
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


if TYPE_CHECKING:
    from sagent.agent.agent import Agent


class AgentSelf:
    """Tool: patch the current agent state."""

    name: str = "AgentSelf"
    tool_id: str = "application/x-tool-agentself"
    description: str = load_tool_description("agentself")
    directive_schema: ClassVar[JSON] = json_freeze(
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
                        "Optional provider/model-specific settings."
                        " Supported keys are reported by diagnostics."
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
        }
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

    def summary_result(self, result: types.history.ToolResult) -> str | None:
        """Return no per-result receipt for AgentSelf.

        Args:
          result: The tool's completed ``ToolResult``.

        Returns:
          receipt: Always ``None``; AgentSelf has no compact receipt.

        """
        del result
        return None

    def prompt(self) -> str:
        """Return dynamic system-prompt guidance.

        Returns:
          text: Supplemental prompt text; empty for AgentSelf.

        """
        return ""

    async def run(self, args: Mapping[str, object]) -> types.history.ToolResult:
        """Apply an AgentSelf patch object.

        Args:
          args: Parsed tool directive mapping.

        Returns:
          result: Outcome of the patch (summary text or error).

        """
        return _apply_patch(args)


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _ModelPlan:
    """Pending model swap (model + spec + display label)."""

    model: types.model.Model
    """New rich provider model to install."""

    spec: types.model.ModelSpec
    """Recipe describing how the model was built (for re-resume)."""

    label: str
    """Human-readable change label (``old → new``)."""


_UNSET = object()


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _PatchPlan:
    """Validated AgentSelf patch ready to commit."""

    status: str | None = None
    """New status string (rendered in the status pane), or ``None`` to keep."""

    model: _ModelPlan | None = None
    """Pending model swap, or ``None`` to keep."""

    thinking: bool | None = None
    """Toggle for extended-thinking; ``None`` to keep."""

    effort: str | None | object = _UNSET
    """Effort hint; ``_UNSET`` means leave unchanged."""

    cache_ttl: str | None = None
    """Cache TTL (``"5m"`` / ``"1h"``); ``None`` to keep."""

    service_tier: str | None | object = _UNSET
    """OpenAI service-tier hint; ``_UNSET`` means leave unchanged."""

    max_request_tokens: int | None = None
    """New per-request input budget; ``None`` to keep."""

    max_response_tokens: int | None = None
    """New per-request response budget; ``None`` to keep."""

    context: str | None = None
    """Context verb (``clear`` / ``compact`` / ``recompact``); ``None`` to keep."""

    context_prompt: str = ""
    """Free-form guidance forwarded to the context verb."""


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


def _apply_patch(d: Mapping[str, object]) -> types.history.ToolResult:
    """Validate and apply an AgentSelf patch object."""
    active = current_agent_var.get(None)
    if active is None:
        return types.history.ToolResult(
            call_id="", content="No active agent.", is_error=True
        )
    agent = cast("Agent", active)
    plan_or_err = _build_patch_plan(agent, d)
    if isinstance(plan_or_err, types.history.ToolResult):
        return plan_or_err
    parts = _commit_patch_plan(agent, plan_or_err)
    if d.get("diagnostics") is True:
        return _do_diagnostics(parts, d)
    return types.history.ToolResult(
        call_id="",
        content="AgentSelf updated: " + ", ".join(parts) if parts else "No changes.",
    )


def _build_patch_plan(
    agent: Agent, d: Mapping[str, object]
) -> _PatchPlan | types.history.ToolResult:
    """Validate an AgentSelf patch without mutating state."""
    err = _validate_patch(d)
    if err is not None:
        return err
    model_plan = _plan_model(agent, d)
    if isinstance(model_plan, types.history.ToolResult):
        return model_plan
    target_model = model_plan.model if model_plan is not None else agent.model
    status = _plan_status(d)
    if isinstance(status, types.history.ToolResult):
        return status
    options_or_err = _plan_model_options(target_model, d)
    if isinstance(options_or_err, types.history.ToolResult):
        return options_or_err
    options = options_or_err
    thinking = cast(bool | None, options.get("thinking"))
    cache_ttl = cast(str | None, options.get("cache_ttl"))
    service_tier = options.get("service_tier", _UNSET)
    has_explicit_limits = "max_request_tokens" in d or "max_response_tokens" in d
    if has_explicit_limits:
        limits = _plan_limits(agent, target_model, d)
        if isinstance(limits, types.history.ToolResult):
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
        service_tier=service_tier,
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
        if not plan.model.model.valid_service_tiers and agent.service_tier is not None:
            agent.service_tier = None
            parts.append("service_tier=unset (unsupported)")
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
    if plan.service_tier is not _UNSET:
        service_tier = cast(str | None, plan.service_tier)
        agent.service_tier = service_tier
        parts.append(f"service_tier={service_tier or 'unset'}")
    if plan.max_request_tokens is not None:
        agent.max_request_tokens = plan.max_request_tokens
        parts.append(f"max_request_tokens={agent.max_request_tokens:,}")
    if plan.max_response_tokens is not None:
        agent.max_response_tokens = plan.max_response_tokens
        parts.append(f"max_response_tokens={agent.max_response_tokens:,}")
    if plan.context is not None:
        _commit_context(agent, plan.context, plan.context_prompt)
        parts.append(f"context={plan.context}")
    return parts


def _validate_patch(d: Mapping[str, object]) -> types.history.ToolResult | None:
    """Validate cross-field AgentSelf patch constraints."""
    if "context_prompt" in d and "context" not in d:
        return types.history.ToolResult(
            call_id="",
            content="context_prompt is only valid when context is set.",
            is_error=True,
        )
    context = d.get("context")
    if context is not None and context not in ("clear", "compact", "recompact"):
        return types.history.ToolResult(
            call_id="", content=f"Invalid context: {context!r}.", is_error=True
        )
    options = d.get("model_options")
    if options is not None and not isinstance(options, Mapping):
        return types.history.ToolResult(
            call_id="", content="model_options must be an object.", is_error=True
        )
    catalog = d.get("catalog")
    if catalog is not None and catalog not in ("providers", "models"):
        return types.history.ToolResult(
            call_id="", content=f"Invalid catalog: {catalog!r}.", is_error=True
        )
    if "catalog_provider" in d and catalog != "models":
        return types.history.ToolResult(
            call_id="",
            content="catalog_provider is only valid with catalog='models'.",
            is_error=True,
        )
    return None


def _plan_status(d: Mapping[str, object]) -> str | types.history.ToolResult | None:
    """Validate an optional status update."""
    raw = d.get("status")
    if raw is None:
        return None
    status = str(raw).strip()
    if not status:
        return types.history.ToolResult(
            call_id="", content="status cannot be empty when provided.", is_error=True
        )
    return status


def _plan_model(
    agent: Agent, d: Mapping[str, object]
) -> _ModelPlan | types.history.ToolResult | None:
    """Build an optional model/provider/account update without applying it."""
    if not any(k in d for k in ("model_id", "provider", "auth", "account")):
        return None
    spec = agent.model_spec
    if spec is None:
        return types.history.ToolResult(
            call_id="", content="Agent has no model spec; cannot swap.", is_error=True
        )
    model_id = str(d.get("model_id", "")).strip() or None
    prov_name = str(d.get("provider", "")).strip() or spec.provider
    auth = str(d.get("auth", "")).strip() or spec.auth
    account = str(d.get("account", "")).strip() or spec.account
    if not model_id and prov_name == spec.provider:
        return types.history.ToolResult(
            call_id="",
            content="model_id is required when changing auth/account without provider.",
            is_error=True,
        )
    if model_id and prov_name == spec.provider:
        inferred = infer_provider(model_id, prov_name)
        if inferred is not None:
            prov_name, auth = inferred
    try:
        prov = build_provider(prov_name, auth, account=account)
        new_model = prov.model(model_id)
    except (AttributeError, RuntimeError, ValueError) as exc:
        return types.history.ToolResult(
            call_id="",
            content=f"Failed to build model {model_id!r}: {exc}",
            is_error=True,
        )
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


def _plan_model_options(
    model: types.model.Model, d: Mapping[str, object]
) -> dict[str, object] | types.history.ToolResult:
    """Validate provider/model-specific options against the target model."""
    raw = d.get("model_options")
    if raw is None:
        return {}
    options = cast(Mapping[str, object], raw)
    supported = _supported_model_options(model)
    unknown = sorted(set(options) - set(supported))
    if unknown:
        return types.history.ToolResult(
            call_id="",
            content=f"Unsupported model_options for {model.model_id}: {', '.join(unknown)}.",
            is_error=True,
        )
    planned: dict[str, object] = {}
    if "thinking" in options:
        value = options["thinking"]
        if not isinstance(value, bool):
            return types.history.ToolResult(
                call_id="",
                content="model_options.thinking must be boolean.",
                is_error=True,
            )
        planned["thinking"] = value
    if "effort" in options:
        value = options["effort"]
        if value is not None and not isinstance(value, str):
            return types.history.ToolResult(
                call_id="",
                content="model_options.effort must be a string or null.",
                is_error=True,
            )
        planned["effort"] = value
    if "cache_ttl" in options:
        value = options["cache_ttl"]
        if value not in ("5m", "1h"):
            return types.history.ToolResult(
                call_id="",
                content="model_options.cache_ttl must be '5m' or '1h'.",
                is_error=True,
            )
        planned["cache_ttl"] = value
    if "service_tier" in options:
        value = options["service_tier"]
        valid = model.valid_service_tiers
        if value is not None and value not in valid:
            quoted = ", ".join(repr(t) for t in valid) or "(none)"
            return types.history.ToolResult(
                call_id="",
                content=(
                    f"model_options.service_tier for {model.model_id} must"
                    f" be one of {quoted} or null, got {value!r}."
                ),
                is_error=True,
            )
        planned["service_tier"] = value
    return planned


def _supported_model_options(model: types.model.Model) -> dict[str, str]:
    """Return supported model option names with compact descriptions."""
    supported: dict[str, str] = {}
    if model.supports_thinking:
        supported["thinking"] = "boolean"
    if model.supports_effort:
        supported["effort"] = "string"
    if model.supports_cache_control:
        supported["cache_ttl"] = "'5m' | '1h'"
    tiers = model.valid_service_tiers
    if tiers:
        supported["service_tier"] = " | ".join(repr(t) for t in tiers)
    return supported


def _plan_limits(
    agent: Agent, model: types.model.Model, d: Mapping[str, object]
) -> dict[str, int] | types.history.ToolResult:
    """Validate explicit token-limit updates against the target model."""
    limits: dict[str, int] = {}
    max_request_tokens = _plan_one_limit(
        d.get("max_request_tokens"), "max_request_tokens"
    )
    if isinstance(max_request_tokens, types.history.ToolResult):
        return max_request_tokens
    max_response_tokens = _plan_one_limit(
        d.get("max_response_tokens"), "max_response_tokens"
    )
    if isinstance(max_response_tokens, types.history.ToolResult):
        return max_response_tokens
    if max_request_tokens is not None:
        if max_request_tokens > model.max_request_tokens:
            return types.history.ToolResult(
                call_id="",
                content=(
                    "Invalid AgentSelf limit override: "
                    f"max_request_tokens={max_request_tokens:,} exceeds model's"
                    f" {model.max_request_tokens:,}"
                ),
                is_error=True,
            )
        limits["max_request_tokens"] = max_request_tokens
    if max_response_tokens is not None:
        if max_response_tokens > model.max_response_tokens:
            return types.history.ToolResult(
                call_id="",
                content=(
                    "Invalid AgentSelf limit override: "
                    f"max_response_tokens={max_response_tokens:,} exceeds model's"
                    f" {model.max_response_tokens:,}"
                ),
                is_error=True,
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
        return types.history.ToolResult(
            call_id="",
            content=f"Invalid AgentSelf limit override: {exc}",
            is_error=True,
        )
    return limits


def _plan_one_limit(raw: object, attr: str) -> int | types.history.ToolResult | None:
    """Validate a single token limit without applying it."""
    if raw is None:
        return None
    if not isinstance(raw, (int, float, str)):
        return types.history.ToolResult(
            call_id="",
            content=f"Invalid AgentSelf limit override: {attr} must be a number.",
            is_error=True,
        )
    try:
        val = int(raw)
    except ValueError as exc:
        return types.history.ToolResult(
            call_id="",
            content=f"Invalid AgentSelf limit override: {exc}",
            is_error=True,
        )
    if val < 1:
        return types.history.ToolResult(
            call_id="",
            content=(
                f"Invalid AgentSelf limit override: {attr}={val}. Must be at least 1."
            ),
            is_error=True,
        )
    return val


def _commit_context(agent: Agent, context: str, prompt: str) -> None:
    """Push a validated context mutation directly to the runtime inbox.

    Context verbs are first-class ``RuntimeEvent``s on the inbox; the
    runtime loop dispatches them in arrival order.
    """
    if context == "clear":
        agent.runtime.inbox.push_back(types.runtime.Clear())
    elif context == "compact":
        agent.runtime.inbox.push_back(types.runtime.Compact(args=prompt))
    elif context == "recompact":
        agent.runtime.inbox.push_back(types.runtime.Recompact(args=prompt))


def _do_diagnostics(
    changes: list[str] | None = None,
    d: Mapping[str, object] | None = None,
) -> types.history.ToolResult:
    """Return current agent diagnostics."""
    agent = cast("Agent | None", current_agent_var.get(None))
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
    return types.history.ToolResult(call_id="", content="\n".join(lines))


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
    provider_cls = getattr(providers_module, provider_name, None)
    if not isinstance(provider_cls, type):
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


def _spec_lines(spec: types.model.ModelSpec | None) -> list[str]:
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
    service_tier = agent.service_tier or "unset"
    if not agent.model.valid_service_tiers:
        service_tier = "unsupported"
    supported = _supported_model_options(agent.model)
    supported_text = ", ".join(f"{k}: {v}" for k, v in supported.items()) or "none"
    return [
        f"Cache TTL:          {agent.cache_ttl}",
        f"Thinking:           {thinking}",
        f"Effort:             {effort}",
        f"Service tier:       {service_tier}",
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
