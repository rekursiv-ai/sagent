"""Adapt Sagent's native model settings to the daemon protocol.

The catalog and providers remain authoritative for provider state.  This module
only translates that native state into the VS Code response shape and turns
frontend actions into existing ``AgentSelf`` or provider operations.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, cast

import asyncio
import contextlib

from sagent.providers.types import HasConfiguration, HasLogin
from sagent.tools.agent_self import AgentSelf


if TYPE_CHECKING:
    from treekle import PlainTree

    from sagent.agent.agent import Agent
    from sagent.agent.team import Team
    from sagent.model.capability import ModelCapability, ModelSettingName
    from sagent.providers.usage import ProviderUsage


async def snapshot(agent: Agent, team: Team) -> dict[str, object]:
    """Project native provider/model settings for the frontend."""
    model = agent.model
    capability = model.capability
    settings = model.settings
    provider = model.provider
    provider_name = provider.name
    accounts = ["default"]
    if isinstance(provider, HasLogin):
        accounts.extend(
            account
            for account in provider.logins.accounts()
            if account != "default"
        )
    acp_servers = list(getattr(provider, "acp_servers", ()))
    current_acp_server = getattr(model, "acp_server", "")
    context_sizes = tuple(
        _context_size_payload(
            size, is_default=size == capability.default.context_size
        )
        for size in capability.prices.sizes(settings.service_tier)
    )
    usage = provider.native_usage()
    if (
        usage is None
        or not usage.details
        or usage.details == ("Antigravity account authenticated",)
    ):
        with contextlib.suppress(Exception):
            usage = (
                await asyncio.wait_for(provider.refresh_usage(), timeout=3.0)
                or usage
            )
    configured = list(team.catalog.configured_providers())
    if provider_name not in configured:
        configured.append(provider_name)

    return {
        "status": "ready",
        "agentLabel": agent.runtime.tool_context.label,
        "availableScopes": ["current-agent"],
        "selectedScope": "current-agent",
        "provider": {
            "providers": configured,
            "configuredProviders": configured,
            "allProviders": team.catalog.provider_registry(),
            "currentProvider": provider_name,
            "accounts": accounts,
            "currentAccount": agent.account or team.catalog.account or "default",
            "acpServers": acp_servers,
            "currentAcpServer": current_acp_server,
        },
        "model": {
            "models": list(provider),
            "currentModel": capability.model_id,
        },
        "capability": {
            "modelId": capability.model_id,
            "contextSizes": list(context_sizes),
            "thinkingEfforts": _setting_values(capability, "thinking_effort"),
            "thinkingBudgets": _setting_values(capability, "thinking_budget"),
            "thinkingOutputs": _setting_values(capability, "thinking_output"),
            "serviceTiers": _setting_values(capability, "service_tier"),
            "cacheTtlSeconds": _setting_values(capability, "cache_ttl_sec"),
            "manageContextServerSide": _setting_values(
                capability,
                "manage_context_server_side",
            ),
            "defaults": {
                "context": _context_size_payload(
                    capability.default.context_size,
                    is_default=True,
                ),
                "thinkingEffort": capability.default.thinking_effort,
                "thinkingBudget": capability.default.thinking_budget,
                "thinkingOutput": capability.default.thinking_output,
                "serviceTier": capability.default.service_tier,
                "cacheTtlSeconds": capability.default.cache_ttl_sec,
                "manageContextServerSide": capability.default.manage_context_server_side,
            },
            "limits": {
                "maxRequestTokens": capability.limits.max_request_tokens,
                "maxResponseTokens": capability.limits.max_response_tokens,
                "maxContextTokens": capability.limits.context_tokens,
            },
        },
        "settings": {
            "context": _context_size_payload(settings.context_size),
            "thinkingEffort": settings.thinking_effort,
            "thinkingBudget": settings.thinking_budget,
            "thinkingOutput": settings.thinking_output,
            "serviceTier": settings.service_tier,
            "cacheTtlSeconds": settings.cache_ttl_sec,
            "manageContextServerSide": settings.manage_context_server_side,
        },
        "writableSettings": [
            "thinking.effort",
            "thinking.budget",
            "thinking.output",
            "service_tier",
        ],
        "usage": _usage_payload(usage),
        "diagnostics": {
            "label": agent.runtime.tool_context.label,
            "model": f"{provider_name}/{capability.model_id}",
            "context": settings.context_size.display_name,
            "thinking": {
                "budget": settings.thinking_budget,
                "output": settings.thinking_output,
                "effort": settings.thinking_effort,
            },
            "serviceTier": settings.service_tier,
            "cacheTtlSeconds": settings.cache_ttl_sec,
            "limits": {
                "maxRequestTokens": capability.limits.max_request_tokens,
                "maxResponseTokens": capability.limits.max_response_tokens,
                "maxContextTokens": capability.limits.context_tokens,
            },
            "cwd": str(agent.runtime.tool_context.cwd),
            "tools": list(agent.tools),
        },
    }


async def usage(agent: Agent) -> dict[str, object]:
    """Fetch and return the freshest provider usage."""
    current = await agent.model.provider.refresh_usage()
    return {"usage": _usage_payload(current)}


async def apply(agent: Agent, team: Team, params: Mapping[str, object]) -> dict[str, object]:
    """Apply a frontend settings action through native Sagent owners."""
    if params.get("scope", "current-agent") != "current-agent":
        raise ValueError("only current-agent settings are supported")
    action = _required_object(params.get("action"), "action")
    _configure_custom_provider(team, action)
    arguments = _model_settings_arguments(action)
    if arguments:
        tool = agent.tools.get("AgentSelf")
        if not isinstance(tool, AgentSelf):
            raise RuntimeError("AgentSelf is not enabled for this Sagent agent")
        await tool.call(
            agent.runtime.tool_context,
            cast("Mapping[str, PlainTree]", arguments),
        )
    return await snapshot(agent, team)


def _setting_values(
    capability: ModelCapability, name: ModelSettingName
) -> list[object]:
    """Return native JSON values without formatting them as terminal text."""
    return list(capability.values(name))


def _usage_payload(usage: ProviderUsage | None) -> dict[str, object] | None:
    if usage is None:
        return None
    return {
        "provider": usage.provider,
        "account": usage.account,
        "summary": usage.summary,
        "details": list(usage.details),
        "raw": dict(usage.raw),
    }


def _context_size_payload(
    size: object,
    *,
    is_default: bool = False,
) -> dict[str, object]:
    context_name = getattr(size, "context_name", None)
    tokens = getattr(size, "tokens", None)
    display_name = getattr(size, "display_name", None)
    if (
        not isinstance(context_name, str)
        or not isinstance(tokens, int)
        or not isinstance(display_name, str)
    ):
        raise TypeError("invalid Sagent context size")
    return {
        "name": context_name,
        "tokens": tokens,
        "displayName": display_name,
        "isDefault": is_default,
    }


def _model_settings_arguments(action: Mapping[str, object]) -> dict[str, object]:
    action_type = action.get("type")
    if action_type == "select-provider":
        return {
            "provider": _required_action_string(action, "provider"),
            "model": "default",
        }
    if action_type == "configure-custom-provider":
        _, _, model = _custom_provider_settings(action)
        return {"provider": "custom", "model": model}
    if action_type == "select-acp-server":
        return {"provider": "acp", "model": "default"}
    if action_type == "select-model":
        return {"model": _required_action_string(action, "model")}
    if action_type == "set-setting":
        name = action.get("name")
        value = action.get("value")
        if not isinstance(name, str):
            raise ValueError("settings action name must be a string")
        if name not in {
            "thinking.effort",
            "thinking.budget",
            "thinking.output",
            "service_tier",
        }:
            raise ValueError(f"setting {name!r} is not writable through AgentSelf")
        if not isinstance(value, str):
            raise ValueError(f"setting {name!r} requires a string value")
        return {
            name.rsplit(".", 1)[-1] if name.startswith("thinking.") else name: value
        }
    if action_type in {"select-scope", "refresh"}:
        return {}
    raise ValueError(f"unknown model settings action {action_type!r}")


def _configure_custom_provider(team: Team, action: Mapping[str, object]) -> None:
    """Pass custom-provider configuration to the provider that owns it."""
    if action.get("type") != "configure-custom-provider":
        return
    base_url, api_key, model = _custom_provider_settings(action)
    provider = team.catalog["custom"]
    if not isinstance(provider, HasConfiguration):
        raise RuntimeError("custom provider does not support configuration")
    provider.configure(base_url=base_url, api_key=api_key, model=model)


def _custom_provider_settings(
    action: Mapping[str, object],
) -> tuple[str, str | None, str]:
    base_url = str(action.get("baseUrl") or action.get("base_url") or "").strip()
    api_key = action.get("apiKey")
    api_key_str = str(api_key).strip() if api_key is not None else None
    model = str(action.get("model") or "").strip() or "default"
    if not base_url:
        raise ValueError("Base URL is required to configure custom provider")
    return base_url, api_key_str, model


def _required_action_string(action: Mapping[str, object], key: str) -> str:
    value = action.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"settings action {key} must be a non-empty string")
    return value


def _required_object(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    return cast("Mapping[str, object]", value)
