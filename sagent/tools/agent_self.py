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
from types import MappingProxyType
from typing import cast

import dataclasses

from sagent import (
    providers as providers_module,
)
from sagent.agent.agent import Agent
from sagent.agent.state import current_agent_var
from sagent.lib.custom_json import JSON, json_freeze
from sagent.providers import (
    PROVIDER_NAMES,
    build_provider,
    default_auth_for_provider,
    infer_provider,
)
from sagent.thinking import apply_thinking_command
from sagent.tools.core import (
    load_tool_description,
    provider_not_allowed_result,
)
from sagent.types.capability import (
    ModelCapability,
    ServiceTier,
    ThinkingEffort,
)
from sagent.types.model import (
    CONTEXT_TAGS,
    Model,
    ModelRecipe,
    base_model_id,
)
from sagent.types.runtime import (
    Clear,
    Compact,
    Recompact,
    ToolResult,
)


CACHE_TTL_SEC: Mapping[str, float] = MappingProxyType({"5m": 300.0, "1h": 3600.0})
"""The two prompt-cache lifetimes the wire spells, in seconds."""

_allow_providers_var: ContextVar[tuple[str, ...]] = ContextVar(
    "_allow_providers", default=tuple(PROVIDER_NAMES)
)
"""Per-call provider allow-list. Set by :meth:`AgentSelf.run` so that
module-level catalog helpers (``_provider_catalog_lines``,
``_model_catalog_lines``) can filter their output without threading the
list through every helper."""


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
                        "Optional auth method suffix override (e.g. 'env',"
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
                        " Fast serving is a model-id option tag, not an"
                        " option: request it via model='...+fast' on"
                        " supported models. Supported keys per model are"
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
        }
    )

    def __init__(
        self,
        *,
        allow_providers: tuple[str, ...] | None = None,
    ) -> None:
        self._allow_providers: tuple[str, ...] = (
            tuple(allow_providers)
            if allow_providers is not None
            else tuple(PROVIDER_NAMES)
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
class _ModelPlan:
    """Pending model swap (model + spec + display label)."""

    model: Model
    """New rich provider model to install."""

    spec: ModelRecipe
    """Recipe describing how the model was built (for re-resume)."""

    label: str
    """Human-readable change label (``old → new``)."""


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _PatchPlan:
    """Validated AgentSelf patch ready to commit."""

    status: str | None = None
    """New status string (rendered in the status pane), or ``None`` to keep."""

    model: _ModelPlan | None = None
    """Pending model swap, or ``None`` to keep."""

    thinking: bool | None = None
    """Toggle for extended-thinking; ``None`` to keep."""

    effort: ThinkingEffort | None = None
    """Effort level; ``None`` means leave unchanged."""

    cache_ttl: str | None = None
    """Cache TTL (``"5m"`` / ``"1h"``); ``None`` to keep."""

    service_tier: ServiceTier | None = None
    """Speed/price tier; ``None`` means leave unchanged."""

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


def _apply_patch(d: Mapping[str, object]) -> ToolResult:
    """Validate and apply an AgentSelf patch object."""
    active = current_agent_var.get(None)
    if active is None:
        return ToolResult(call_id="", content="No active agent.", is_error=True)
    agent = cast(Agent, active)
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


def _build_patch_plan(agent: Agent, d: Mapping[str, object]) -> _PatchPlan | ToolResult:
    """Validate an AgentSelf patch without mutating state."""
    err = _validate_patch(d)
    if err is not None:
        return err
    model_plan = _plan_model(agent, d)
    if isinstance(model_plan, ToolResult):
        return model_plan
    target_model = model_plan.model if model_plan is not None else agent.model
    status = _plan_status(d)
    if isinstance(status, ToolResult):
        return status
    options_or_err = plan_model_options(target_model, d)
    if isinstance(options_or_err, ToolResult):
        return options_or_err
    options = options_or_err
    thinking = cast(bool | None, options.get("thinking"))
    cache_ttl = cast(str | None, options.get("cache_ttl"))
    service_tier = cast(ServiceTier | None, options.get("service_tier"))
    has_explicit_limits = "max_request_tokens" in d or "max_response_tokens" in d
    if has_explicit_limits:
        limits = _plan_limits(agent, target_model, d)
        if isinstance(limits, ToolResult):
            return limits
    else:
        limits = {}
    context = cast(str | None, d.get("context"))
    return _PatchPlan(
        status=status,
        model=model_plan,
        thinking=thinking,
        effort=cast(ThinkingEffort | None, options.get("effort")),
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
        # ``swap_model`` carries the selections across and drops the ones the
        # new model rejects, so report what it DID rather than re-deriving
        # the rule here: two copies of it disagreed, and this one answered
        # "supported" for every model because each axis is total and so is
        # never empty.
        names = ("thinking_effort", "thinking_budget", "service_tier")
        before = [getattr(agent.model.settings, name) for name in names]
        agent.swap_model(plan.model.model, spec=plan.model.spec)
        parts.append(f"model={plan.model.label}")
        for name, was in zip(names, before, strict=True):
            now = getattr(agent.model.settings, name)
            if was != now:
                parts.append(f"{name}={now} (unsupported)")
    settings = agent.model.settings
    if plan.thinking is not None:
        _ = apply_thinking_command(
            "adaptive" if plan.thinking else "off", settings, show=False
        )
        parts.append(f"thinking={'on' if plan.thinking else 'off'}")
    if plan.effort is not None:
        settings.thinking_effort = plan.effort
        parts.append(f"effort={plan.effort}")
    if plan.cache_ttl is not None:
        settings.cache_ttl_sec = CACHE_TTL_SEC[plan.cache_ttl]
        parts.append(f"cache_ttl={plan.cache_ttl}")
    if plan.service_tier is not None:
        settings.service_tier = plan.service_tier
        parts.append(f"service_tier={plan.service_tier}")
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


def _validate_patch(d: Mapping[str, object]) -> ToolResult | None:
    """Validate cross-field AgentSelf patch constraints."""
    if "context_prompt" in d and "context" not in d:
        return ToolResult(
            call_id="",
            content="context_prompt is only valid when context is set.",
            is_error=True,
        )
    context = d.get("context")
    if context is not None and context not in ("clear", "compact", "recompact"):
        return ToolResult(
            call_id="", content=f"Invalid context: {context!r}.", is_error=True
        )
    options = d.get("model_options")
    if options is not None and not isinstance(options, Mapping):
        return ToolResult(
            call_id="", content="model_options must be an object.", is_error=True
        )
    catalog = d.get("catalog")
    if catalog is not None and catalog not in ("providers", "models"):
        return ToolResult(
            call_id="", content=f"Invalid catalog: {catalog!r}.", is_error=True
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
    status = str(raw).strip()
    if not status:
        return ToolResult(
            call_id="", content="status cannot be empty when provided.", is_error=True
        )
    return status


def _plan_model(
    agent: Agent, d: Mapping[str, object]
) -> _ModelPlan | ToolResult | None:
    """Build an optional model/provider/account update without applying it."""
    if not any(k in d for k in ("model_id", "provider", "auth", "account")):
        return None
    spec = agent.model_recipe
    if spec is None:
        return ToolResult(
            call_id="", content="Agent has no model spec; cannot swap.", is_error=True
        )
    model_id = str(d.get("model_id", "")).strip() or None
    prov_name = str(d.get("provider", "")).strip() or spec.provider
    if "auth" in d:
        auth = str(d["auth"]).strip()
    elif prov_name == spec.provider:
        auth = spec.auth
    else:
        auth = default_auth_for_provider(prov_name)
    if "account" in d:
        account = str(d["account"]).strip()
        if not account:
            return ToolResult(
                call_id="", content="account cannot be empty.", is_error=True
            )
    else:
        account = spec.account
    # An auth/account-only swap (no model_id, same provider) keeps the
    # current model -- matches REPL ``/model --auth sub`` semantics so
    # the tool surface and slash command behave the same way.
    if not model_id and prov_name == spec.provider:
        model_id = spec.model_id
    if model_id and prov_name == spec.provider:
        inferred = infer_provider(model_id, prov_name)
        if inferred is not None:
            prov_name, auth = inferred
    allow = _allow_providers_var.get()
    if prov_name != spec.provider and prov_name not in allow:
        return provider_not_allowed_result(prov_name, allow, spec.provider)
    try:
        prov = build_provider(
            prov_name,
            auth,
            account=account,
        )
        new_model = prov.model(model_id)
    except (AttributeError, RuntimeError, ValueError) as exc:
        return ToolResult(
            call_id="",
            content=f"Failed to build model {model_id!r}: {exc}",
            is_error=True,
        )
    old_id = agent.model.tagged_model_id
    label = f"{old_id} → {new_model.tagged_model_id}"
    if prov_name != spec.provider:
        label = f"{spec.provider}/{old_id} → {prov_name}/{new_model.tagged_model_id}"
    return _ModelPlan(
        model=new_model,
        spec=dataclasses.replace(
            spec,
            provider=prov_name,
            auth=auth,
            model_id=new_model.tagged_model_id,
            account=account,
        ),
        label=label,
    )


def plan_model_options(
    model: Model, d: Mapping[str, object]
) -> dict[str, object] | ToolResult:
    """Validate provider/model-specific options against the target model.

    Shared with :class:`AgentSpawn`, which applies the validated options
    to a freshly built child agent.
    """
    raw = d.get("model_options")
    if raw is None:
        return {}
    options = cast(Mapping[str, object], raw)
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
        value = options["effort"]
        if value is not None and not isinstance(value, str):
            return ToolResult(
                call_id="",
                content="model_options.effort must be a string or null.",
                is_error=True,
            )
        valid = model.capability.thinking_effort
        if value is not None and value not in valid:
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
        value = options["cache_ttl"]
        if value is not None and value not in CACHE_TTL_SEC:
            quoted = ", ".join(repr(k) for k in CACHE_TTL_SEC)
            return ToolResult(
                call_id="",
                content=f"model_options.cache_ttl must be one of {quoted} or null.",
                is_error=True,
            )
        planned["cache_ttl"] = value
    if "service_tier" in options:
        # ``null`` means "let the vendor pick", which is the axis's own unset
        # value rather than an absent one.
        value = (
            options["service_tier"] if options["service_tier"] is not None else ("auto")
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


def _supported_model_options(model: Model) -> dict[str, str]:
    """Return supported model option names with compact descriptions.

    Every axis is total, so a non-empty set proves nothing: an axis is
    selectable only when it offers something BESIDES its unset value.
    """
    supported: dict[str, str] = {}
    capability = model.capability
    if capability.thinking_budget != frozenset({"none"}):
        supported["thinking"] = "boolean"
    efforts = capability.thinking_effort - {"none"}
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
    agent: Agent, model: Model, d: Mapping[str, object]
) -> dict[str, int] | ToolResult:
    """Validate explicit token-limit updates against the target model."""
    limits: dict[str, int] = {}
    max_request_tokens = _plan_one_limit(
        d.get("max_request_tokens"), "max_request_tokens"
    )
    if isinstance(max_request_tokens, ToolResult):
        return max_request_tokens
    max_response_tokens = _plan_one_limit(
        d.get("max_response_tokens"), "max_response_tokens"
    )
    if isinstance(max_response_tokens, ToolResult):
        return max_response_tokens
    if max_request_tokens is not None:
        if max_request_tokens > model.limits.max_request_tokens:
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
        if max_response_tokens > model.limits.max_response_tokens:
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
        return ToolResult(
            call_id="",
            content=f"Invalid AgentSelf limit override: {exc}",
            is_error=True,
        )
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


def _window_variant_hint(agent: Agent, model: Model, requested: int) -> str:
    """Suggest a larger-window model id when one would satisfy ``requested``.

    Context-window size is encoded in the model id via a ``+1m`` / ``+200k``
    suffix, not the ``max_request_tokens`` limit. When an agent over-raises
    the limit to reach a bigger window, point it at the sibling variant
    (same base id, a window tag, enough room) so the rejection self-corrects.

    Args:
      agent: The active agent (source of provider + current model id).
      model: The target model whose window the request exceeded.
      requested: The rejected ``max_request_tokens`` value.

    Returns:
      hint: ``". Did you mean model_id=<variant>?"`` when a fitting variant
          exists, else the empty string.

    """
    spec = agent.model_recipe
    if spec is None:
        return ""
    provider_cls = getattr(providers_module, spec.provider, None)
    known = getattr(provider_cls, "CAPABILITIES", None)
    if not isinstance(known, Mapping):
        return ""
    catalog = cast(Mapping[str, ModelCapability], known)
    base = base_model_id(model.tagged_model_id)
    cap = catalog.get(base)
    if cap is None:
        return ""
    for tag in CONTEXT_TAGS:
        candidate = base + tag
        window = getattr(cap.context.get(tag), "max_request_tokens", 0)
        if candidate != model.tagged_model_id and window >= requested:
            return (
                f". The window is part of the model id: switch to"
                f" model_id={candidate} (a {window:,}-token window) rather"
                f" than raising max_request_tokens"
            )
    return ""


def _commit_context(agent: Agent, context: str, prompt: str) -> None:
    """Push a validated context mutation directly to the runtime inbox.

    Context verbs are first-class ``RuntimeEvent``s on the inbox; the
    runtime loop dispatches them in arrival order.
    """
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
    agent = cast("Agent | None", current_agent_var.get(None))
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


def _catalog_lines(d: Mapping[str, object], agent: Agent | None) -> list[str]:
    """Return read-only provider/model catalog diagnostics."""
    catalog = d.get("catalog")
    if catalog is None:
        return []
    if catalog == "providers":
        return _provider_catalog_lines()
    if catalog == "models":
        provider = str(d.get("catalog_provider") or "").strip()
        if not provider and agent is not None and agent.model_recipe is not None:
            provider = agent.model_recipe.provider
        return _model_catalog_lines(provider)
    return []


def _provider_catalog_lines() -> list[str]:
    """List providers known to the local build, filtered by allow-list."""
    return ["Known providers: " + ", ".join(_allow_providers_var.get())]


def _model_catalog_lines(provider_name: str) -> list[str]:
    """List statically known models for one provider, gated by allow-list."""
    if not provider_name:
        return ["Known models: set catalog_provider or configure an active provider."]
    provider_cls = getattr(providers_module, provider_name, None)
    if not isinstance(provider_cls, type):
        return [f"Known models: unknown provider {provider_name!r}."]
    if provider_name not in _allow_providers_var.get():
        return [
            (
                f"Known models: {provider_name!r} is not in the allowed list"
                f" {list(_allow_providers_var.get())}."
            )
        ]
    default = getattr(provider_cls, "DEFAULT_MODEL", None)
    known = getattr(provider_cls, "CAPABILITIES", None)
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


def _format_stats(agent: Agent) -> list[str]:
    """Format live cost/budget counters into display lines.

    Reads directly from the single cost store (``agent.cost_tracker``)
    plus the live budget/round counters on the agent. No separate
    per-request publisher exists; diagnostics is a pull, not a push.
    """
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
        f"Cache creation:     {tracker.total.cache_write:,}",
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


def _agent_option_lines(agent: Agent) -> list[str]:
    """Format current and supported model options."""
    # Each axis is total, so "unsupported" is the SINGLETON unset value, not
    # an empty set: ``not bool(frozenset)`` was never true and every model
    # reported every knob as supported.
    capability = agent.model.capability
    settings = agent.model.settings
    budget = settings.thinking_budget
    thinking = "off" if budget == "none" else budget
    if capability.thinking_budget == frozenset({"none"}):
        thinking = "unsupported"
    effort = settings.thinking_effort
    if capability.thinking_effort == frozenset({"none"}):
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


def _session_lines(agent: Agent) -> list[str]:
    """Format session identity, path, and cwd."""
    if agent.session_dir is not None:
        lines = [f"Session:            {agent.session_dir.name}"]
        lines.append(f"Session dir:        {agent.session_dir}")
    else:
        lines = [f"Session:            {agent.session_id} (ephemeral)"]
    lines.append(f"Bash cwd:           {agent.tool_state.bash_cwd}")
    return lines
