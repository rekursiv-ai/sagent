"""AgentSelf tool: mutate the current agent's state.

Unifies all self-mutation operations (status, context management,
model swap) under one tool. Spawning a *new* agent is the
separate ``AgentSpawn`` tool.
"""

from __future__ import annotations

from typing import cast

import dataclasses

from sagent.custom_types import Message, ModelSpec, TextMessage
from sagent.lib.json import JSON, int_val, json_freeze
from sagent.lib.message import get_directive
from sagent.providers import build_provider, infer_provider
from sagent.tools.core import (
    current_agent_var,
    get_tool_state,
    load_tool_description,
)


class AgentSelf:
    """Tool: mutate the current agent (status, context, model)."""

    name: str = "AgentSelf"
    tool_id: str = "application/x-tool-agentself"
    description: str = load_tool_description("agentself")
    supports_microcompaction: bool = True
    directive_schema: JSON = json_freeze(
        {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": [
                        "status",
                        "clear",
                        "compact",
                        "recompact",
                        "diagnostics",
                        "model",
                        "limits",
                    ],
                    "description": "Which mutation to perform.",
                },
                "status": {
                    "type": "string",
                    "description": "Status text (required for 'status').",
                },
                "reason": {
                    "type": "string",
                    "description": "Optional note (for 'clear').",
                },
                "custom_instructions": {
                    "type": "string",
                    "description": (
                        "Guidance for the compactor (for 'compact' and 'recompact')."
                    ),
                },
                "model_id": {
                    "type": "string",
                    "description": (
                        "Model ID to switch to. Provider and auth are"
                        " inferred from known model prefixes; usually"
                        " only model_id is needed. Omit to use the new"
                        " provider's default."
                    ),
                },
                "provider": {
                    "type": "string",
                    "description": (
                        "Provider class name. Usually omit unless"
                        " overriding model_id prefix inference."
                    ),
                },
                "auth": {
                    "type": "string",
                    "description": (
                        "Auth method suffix dispatched as"
                        " <Provider>.from_<auth>(). Usually omit unless"
                        " overriding inferred auth."
                    ),
                },
                "account": {
                    "type": "string",
                    "description": (
                        "Credential account name. Defaults to inheriting"
                        " the parent's account."
                    ),
                },
                "max_request_tokens": {
                    "type": "integer",
                    "minimum": 1,
                    "description": (
                        "Token-count limit for one model request. Only valid"
                        " with operation='limits'. The current value is the"
                        " active agent's current max_request_tokens, initialized"
                        " from the active model's context budget and possibly"
                        " changed by an earlier limits operation. Must be at"
                        " least 1, no larger than the active model's intrinsic"
                        " max_request_tokens, and larger than the current"
                        " context buffer."
                    ),
                },
                "max_response_tokens": {
                    "type": "integer",
                    "minimum": 1,
                    "description": (
                        "Token-count limit reserved for one model response."
                        " Only valid with operation='limits'. The current value"
                        " is the active agent's current max_response_tokens,"
                        " initialized from the active model's response limit"
                        " and possibly changed by an earlier limits operation."
                        " Must be at least 1 and no larger than the active"
                        " model's intrinsic max_response_tokens."
                    ),
                },
            },
            "required": ["operation"],
            "additionalProperties": False,
        }
    )

    def summary(self, msg: Message) -> str:
        """Return a short label summarizing this self-mutation call.

        Args:
          msg: Tool call message.

        Returns:
          label: Human-readable summary including operation.

        """
        d = get_directive(msg)
        op = str(d.get("operation", ""))
        if op == "status":
            return f"AgentSelf status={d.get('status', '')}"
        if op in ("compact", "recompact"):
            return _compact_summary(op, d)
        if op == "clear":
            return "AgentSelf clear"
        if op == "diagnostics":
            return "AgentSelf diagnostics"
        if op == "model":
            parts = [f"model={d.get('model_id', '')}" if d.get("model_id") else ""]
            if d.get("provider"):
                parts.insert(0, f"provider={d.get('provider')}")
            return f"AgentSelf {' '.join(p for p in parts if p)}".strip()
        if op == "limits":
            parts: list[str] = []
            if "max_request_tokens" in d:
                parts.append(f"max_request_tokens={d.get('max_request_tokens')}")
            if "max_response_tokens" in d:
                parts.append(f"max_response_tokens={d.get('max_response_tokens')}")
            return f"AgentSelf limits {' '.join(parts)}".strip()
        return f"AgentSelf {op}"

    def prompt(self) -> str:
        """Return dynamic system-prompt guidance.

        Returns:
          prompt: Empty; stable AgentSelf guidance lives in tools_agentself.md.

        """
        return ""

    async def run(self, msg: Message) -> Message:
        """Dispatch the requested self-mutation operation.

        Args:
          msg: Tool call message with ``operation`` and operation-specific fields.

        Returns:
          result: Operation result or error message.

        """
        d = get_directive(msg)
        op = str(d.get("operation", ""))
        limit_error = _validate_limit_operation(d, op)
        if limit_error is not None:
            return limit_error
        if op == "status":
            result = _do_status(d)
        elif op == "clear":
            result = _do_clear(d)
        elif op == "compact":
            result = _do_compact(d)
        elif op == "recompact":
            result = _do_recompact(d)
        elif op == "diagnostics":
            result = _do_diagnostics()
        elif op == "model":
            result = _do_model(d)
        elif op == "limits":
            result = _apply_limits(d)
        else:
            return TextMessage(f"Unknown operation: {op!r}", "text/x-error")
        return result


def _compact_summary(op: str, d: JSON) -> str:
    """Shared summary formatter for compact/recompact."""
    instr = str(d.get("custom_instructions", "") or "").strip()
    if instr:
        return f"AgentSelf {op}: {instr[:40]}{'…' if len(instr) > 40 else ''}"
    return f"AgentSelf {op}"


def _validate_limit_operation(d: JSON, op: str) -> Message | None:
    """Reject context-limit fields outside the dedicated limits operation."""
    has_req = d.get("max_request_tokens") is not None
    has_resp = d.get("max_response_tokens") is not None
    if not has_req and not has_resp:
        return None
    if op != "limits":
        return TextMessage(
            (
                "Invalid AgentSelf limit override: max_request_tokens and"
                ' max_response_tokens are only valid with operation="limits".'
                ' Use operation="limits" to change the agent\'s token limits;'
                " otherwise leave these fields unset."
            ),
            "text/x-error",
        )
    return None


def _do_status(d: JSON) -> Message:
    status = str(d.get("status", "")).strip()
    if not status:
        return TextMessage("Status cannot be empty.", "text/x-error")
    agent = current_agent_var.get(None)
    if agent is not None:
        agent.set_status(status)
    return TextMessage(status, "text/x-status")


def _do_clear(d: JSON) -> Message:
    reason = str(d.get("reason", ""))
    get_tool_state().clear_requested = reason or ""
    note = f" ({reason})" if reason.strip() else ""
    return TextMessage(
        f"Conversation clear queued{note}; wipe happens between model requests.",
        "text/plain",
    )


def _do_compact(d: JSON) -> Message:
    custom = str(d.get("custom_instructions", ""))
    get_tool_state().compact_requested = custom or ""
    note = f" Instructions: {custom.strip()}." if custom.strip() else ""
    return TextMessage(
        f"Compaction queued; summary will replace history between model requests.{note}",
        "text/plain",
    )


def _do_recompact(d: JSON) -> Message:
    custom = str(d.get("custom_instructions", ""))
    get_tool_state().recompact_requested = custom or ""
    note = f" Instructions: {custom.strip()}." if custom.strip() else ""
    return TextMessage(
        (
            "Recompaction queued; the pre-compact transcript will be"
            f" reloaded and re-summarized between model requests.{note}"
        ),
        "text/plain",
    )


def _do_diagnostics() -> Message:
    agent = current_agent_var.get(None)
    spec = agent.model_spec if agent is not None else None
    stats = dict(get_tool_state().stats)
    if not stats:
        lines = [
            "No stats yet - the Agent publishes stats after the first"
            " completed model request.",
        ]
        lines.extend(_spec_lines(spec))
        return TextMessage("\n".join(lines), "text/plain")
    lines = _format_stats(stats)
    lines.extend(_spec_lines(spec))
    return TextMessage("\n".join(lines), "text/plain")


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


def _do_model(d: JSON) -> Message:
    model_id = str(d.get("model_id", "")).strip() or None
    agent = current_agent_var.get(None)
    if agent is None:
        return TextMessage("No active agent.", "text/x-error")
    spec = agent.model_spec
    if spec is None:
        return TextMessage("Agent has no model spec; cannot swap.", "text/x-error")
    prov_name = str(d.get("provider", "")).strip() or spec.provider
    auth = str(d.get("auth", "")).strip() or spec.auth
    account = str(d.get("account", "")).strip() or spec.account
    if not model_id and prov_name == spec.provider:
        return TextMessage(
            "model_id is required when provider is unchanged.", "text/x-error"
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
    agent.swap_model(
        new_model,
        spec=dataclasses.replace(
            spec,
            provider=prov_name,
            auth=auth,
            model_id=new_model.model_id,
            account=account,
        ),
    )
    label = f"{old_id} → {new_model.model_id}"
    if prov_name != spec.provider:
        label = f"{spec.provider}/{old_id} → {prov_name}/{new_model.model_id}"
    return TextMessage(f"Model swapped: {label}", "text/plain")


def _apply_limits(d: JSON) -> Message:
    agent = current_agent_var.get(None)
    if agent is None:
        return TextMessage("No active agent.", "text/x-error")
    raw_req = d.get("max_request_tokens")
    raw_resp = d.get("max_response_tokens")
    if raw_req is None and raw_resp is None:
        return TextMessage(
            'operation="limits" requires max_request_tokens or max_response_tokens.',
            "text/x-error",
        )
    parts: list[str] = []
    try:
        if raw_req is not None:
            err = _apply_one_limit(
                agent, raw_req, attr="max_request_tokens", parts=parts
            )
            if err is not None:
                return err
        if raw_resp is not None:
            err = _apply_one_limit(
                agent, raw_resp, attr="max_response_tokens", parts=parts
            )
            if err is not None:
                return err
    except (ValueError, TypeError) as exc:
        return TextMessage(f"Invalid AgentSelf limit override: {exc}", "text/x-error")
    return TextMessage("Limits updated: " + ", ".join(parts), "text/plain")


def _apply_one_limit(
    agent: object,
    raw: object,
    *,
    attr: str,
    parts: list[str],
) -> Message | None:
    """Validate and apply a single token limit. Returns error or None."""
    if not isinstance(raw, (int, float, str)):
        return TextMessage(
            f"Invalid AgentSelf limit override: {attr} must be a number.",
            "text/x-error",
        )
    val = int(raw)
    if val < 1:
        return TextMessage(
            f"Invalid AgentSelf limit override: {attr}={val}. Must be at least 1.",
            "text/x-error",
        )
    setattr(agent, attr, val)
    parts.append(f"{attr}={getattr(agent, attr):,}")
    return None
