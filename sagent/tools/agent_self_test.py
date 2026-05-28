"""Tests for ``tools.agent_self``: agent self-mutation directives."""

from __future__ import annotations

from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import override
from unittest.mock import MagicMock, patch

import pytest

from sagent.agent.agent import Agent
from sagent.testing import MockModelCaps
from sagent.tools.agent_self import AgentSelf
from sagent.tools.core import current_agent_var, tool_state_var
from sagent.types.model import (
    ModelRequest,
    ModelResponse,
    ModelSpec,
    Pricing,
)
from sagent.types.runtime import (
    AssistantMessage,
    Clear,
    Compact,
    Recompact,
)


@dataclass(slots=True, kw_only=True)
class StubProviderModel(MockModelCaps):
    """Provider model that returns scripted responses on call."""

    model_id: str = "stub-1"
    max_request_tokens: int = 100_000
    responses: list[AssistantMessage] = field(default_factory=list)
    _idx: int = field(default=0, init=False)

    @property
    @override
    def pricing(self) -> Pricing:
        return Pricing()

    async def buffer(self, request: ModelRequest) -> ModelResponse:
        return await self.stream(request)

    async def stream(
        self,
        request: ModelRequest,
        on_text: Callable[[str], None] | None = None,
        on_thinking: Callable[[str], None] | None = None,
    ) -> ModelResponse:
        del request, on_text, on_thinking
        idx = self._idx
        self._idx += 1
        msg = (
            self.responses[idx]
            if idx < len(self.responses)
            else AssistantMessage(text="ok")
        )
        return ModelResponse(message=msg)


def _make_agent(*, spec: ModelSpec | None = None) -> Agent:
    return Agent(
        model=StubProviderModel(),
        tools=[],
        model_spec=spec,
    )


@contextmanager
def _active(agent: Agent) -> Generator[Agent]:
    token = current_agent_var.set(agent)
    state_token = tool_state_var.set(agent.tool_state)
    try:
        yield agent
    finally:
        tool_state_var.reset(state_token)
        current_agent_var.reset(token)


def test_metadata_basics() -> None:
    t = AgentSelf()
    assert t.name == "AgentSelf"
    assert t.tool_id == "application/x-tool-agentself"
    assert t.prompt() == ""


def test_summary_renders_parts() -> None:
    t = AgentSelf()
    out = t.summary(
        {
            "status": "thinking",
            "context": "compact",
            "model_id": "m1",
            "provider": "P",
            "max_request_tokens": 100,
            "max_response_tokens": 10,
            "model_options": {"thinking": True},
            "diagnostics": True,
            "catalog": "models",
        },
    )
    assert "status=thinking" in out
    assert "context=compact" in out
    assert "diagnostics" in out
    assert "catalog=models" in out


def test_summary_empty_just_name() -> None:
    t = AgentSelf()
    assert t.summary({}) == "AgentSelf"


@pytest.mark.asyncio
async def test_run_no_active_agent() -> None:
    token = current_agent_var.set(None)
    try:
        t = AgentSelf()
        result = await t.run({})
    finally:
        current_agent_var.reset(token)
    assert result.is_error
    assert "No active agent" in result.content


@pytest.mark.asyncio
async def test_run_no_changes_message() -> None:
    agent = _make_agent()
    t = AgentSelf()
    with _active(agent):
        result = await t.run({})
    assert not result.is_error
    assert "No changes" in result.content


@pytest.mark.asyncio
async def test_run_status_updates_agent() -> None:
    agent = _make_agent()
    t = AgentSelf()
    with _active(agent):
        result = await t.run({"status": "working"})
    assert not result.is_error
    assert agent.status == "working"
    assert "status=working" in result.content


@pytest.mark.asyncio
async def test_run_empty_status_is_error() -> None:
    agent = _make_agent()
    t = AgentSelf()
    with _active(agent):
        result = await t.run({"status": "   "})
    assert result.is_error


@pytest.mark.asyncio
async def test_context_prompt_without_context_is_error() -> None:
    agent = _make_agent()
    t = AgentSelf()
    with _active(agent):
        result = await t.run({"context_prompt": "x"})
    assert result.is_error
    assert "context_prompt is only valid" in result.content


@pytest.mark.asyncio
async def test_invalid_context_is_error() -> None:
    agent = _make_agent()
    t = AgentSelf()
    with _active(agent):
        result = await t.run({"context": "wipe"})
    assert result.is_error
    assert "Invalid context" in result.content


@pytest.mark.asyncio
async def test_invalid_model_options_object_errors() -> None:
    agent = _make_agent()
    t = AgentSelf()
    with _active(agent):
        result = await t.run({"model_options": "not-an-object"})
    assert result.is_error
    assert "model_options must be an object" in result.content


@pytest.mark.asyncio
async def test_invalid_catalog_errors() -> None:
    agent = _make_agent()
    t = AgentSelf()
    with _active(agent):
        result = await t.run({"catalog": "garbage"})
    assert result.is_error
    assert "Invalid catalog" in result.content


@pytest.mark.asyncio
async def test_catalog_provider_without_models_catalog_errors() -> None:
    agent = _make_agent()
    t = AgentSelf()
    with _active(agent):
        result = await t.run({"catalog_provider": "P"})
    assert result.is_error
    assert "catalog_provider is only valid" in result.content


@pytest.mark.asyncio
async def test_context_clear_pushes_clear_event() -> None:
    agent = _make_agent()
    t = AgentSelf()
    with _active(agent):
        result = await t.run({"context": "clear"})
    assert not result.is_error
    items = await agent.runtime.inbox.drain()
    assert any(isinstance(i, Clear) for i in items)


@pytest.mark.asyncio
async def test_context_compact_pushes_compact_event() -> None:
    agent = _make_agent()
    t = AgentSelf()
    with _active(agent):
        result = await t.run({"context": "compact", "context_prompt": "trim it"})
    assert not result.is_error
    items = await agent.runtime.inbox.drain()
    compacts = [i for i in items if isinstance(i, Compact)]
    assert len(compacts) == 1
    assert compacts[0].args == "trim it"


@pytest.mark.asyncio
async def test_context_recompact_pushes_recompact_event() -> None:
    agent = _make_agent()
    t = AgentSelf()
    with _active(agent):
        result = await t.run({"context": "recompact"})
    assert not result.is_error
    items = await agent.runtime.inbox.drain()
    assert any(isinstance(i, Recompact) for i in items)


@pytest.mark.asyncio
async def test_max_request_tokens_within_model_cap() -> None:
    agent = _make_agent()
    t = AgentSelf()
    with _active(agent):
        result = await t.run({"max_request_tokens": 10_000})
    assert not result.is_error
    assert agent.max_request_tokens == 10_000


@pytest.mark.asyncio
async def test_max_request_tokens_exceeds_model_cap() -> None:
    agent = _make_agent()
    t = AgentSelf()
    with _active(agent):
        result = await t.run({"max_request_tokens": 999_999_999})
    assert result.is_error
    assert "exceeds model" in result.content


@pytest.mark.asyncio
async def test_max_request_tokens_below_one_errors() -> None:
    agent = _make_agent()
    t = AgentSelf()
    with _active(agent):
        result = await t.run({"max_request_tokens": 0})
    assert result.is_error
    assert "Must be at least 1" in result.content


@pytest.mark.asyncio
async def test_max_request_tokens_wrong_type_errors() -> None:
    agent = _make_agent()
    t = AgentSelf()
    with _active(agent):
        # Lists are not coercible to int.
        result = await t.run({"max_request_tokens": [10]})
    assert result.is_error
    assert "must be a number" in result.content


@pytest.mark.asyncio
async def test_model_swap_without_spec_errors() -> None:
    agent = _make_agent(spec=None)
    t = AgentSelf()
    with _active(agent):
        result = await t.run({"model_id": "claude"})
    assert result.is_error
    assert "no model spec" in result.content


@pytest.mark.asyncio
async def test_model_change_auth_without_model_id_errors() -> None:
    agent = _make_agent(
        spec=ModelSpec(provider="StubP", auth="env", model_id="stub-1", account=""),
    )
    t = AgentSelf()
    with _active(agent):
        result = await t.run({"auth": "new"})
    assert result.is_error
    assert "model_id is required" in result.content


@pytest.mark.asyncio
async def test_diagnostics_returns_lines() -> None:
    agent = _make_agent(
        spec=ModelSpec(provider="StubP", auth="env", model_id="stub-1", account=""),
    )
    t = AgentSelf()
    with _active(agent):
        result = await t.run({"diagnostics": True})
    assert not result.is_error
    assert "Provider:" in result.content
    assert "Bash cwd:" in result.content


@pytest.mark.asyncio
async def test_diagnostics_catalog_providers() -> None:
    agent = _make_agent()
    t = AgentSelf()
    with _active(agent):
        result = await t.run({"diagnostics": True, "catalog": "providers"})
    assert "Known providers:" in result.content


@pytest.mark.asyncio
async def test_diagnostics_catalog_models_unknown_provider() -> None:
    agent = _make_agent()
    t = AgentSelf()
    with _active(agent):
        result = await t.run(
            {
                "diagnostics": True,
                "catalog": "models",
                "catalog_provider": "Nope",
            },
        )
    assert "unknown provider" in result.content


@pytest.mark.asyncio
async def test_diagnostics_catalog_models_no_provider() -> None:
    agent = _make_agent()
    t = AgentSelf()
    with _active(agent):
        result = await t.run({"diagnostics": True, "catalog": "models"})
    assert "set catalog_provider" in result.content


@pytest.mark.asyncio
async def test_diagnostics_includes_stats_lines_when_present() -> None:
    agent = _make_agent()
    agent.tool_state.stats = {
        "num_tool_call_rounds": 2,
        "max_request_tokens": 50,
        "max_response_tokens": 5,
        "input_tokens": 10,
        "total_input_tokens": 20,
        "total_output_tokens": 30,
        "cache_creation_tokens": 0,
        "cache_read_tokens": 0,
        "total_cost_usd": 0.42,
    }
    t = AgentSelf()
    with _active(agent):
        result = await t.run({"diagnostics": True})
    assert "Tool call rounds:" in result.content
    assert "Total cost (USD):   $0.42" in result.content


@pytest.mark.asyncio
async def test_model_options_unsupported_key_errors() -> None:
    # StubProviderModel doesn't support thinking; passing the key errors.
    agent = _make_agent()
    t = AgentSelf()
    with _active(agent):
        result = await t.run({"model_options": {"thinking": True}})
    assert result.is_error
    assert "Unsupported model_options" in result.content


@dataclass(slots=True, kw_only=True)
class TierStubModel(StubProviderModel):
    """``StubProviderModel`` advertising a non-empty service-tier set."""

    valid_service_tiers: tuple[str, ...] = ("auto", "default", "flex", "priority")


@pytest.mark.asyncio
async def test_service_tier_unsupported_when_model_lacks_capability() -> None:
    agent = _make_agent()  # StubProviderModel.valid_service_tiers = ()
    t = AgentSelf()
    with _active(agent):
        result = await t.run({"model_options": {"service_tier": "priority"}})
    assert result.is_error
    assert "Unsupported model_options" in result.content


@pytest.mark.asyncio
async def test_service_tier_rejects_unknown_value() -> None:
    agent = Agent(model=TierStubModel(), tools=[])
    t = AgentSelf()
    with _active(agent):
        result = await t.run({"model_options": {"service_tier": "turbo"}})
    assert result.is_error
    assert "service_tier" in result.content
    assert "must be one of" in result.content


@pytest.mark.asyncio
async def test_service_tier_priority_applied() -> None:
    agent = Agent(model=TierStubModel(), tools=[])
    t = AgentSelf()
    with _active(agent):
        result = await t.run({"model_options": {"service_tier": "priority"}})
    assert not result.is_error
    assert agent.service_tier == "priority"
    assert "service_tier=priority" in result.content


@pytest.mark.asyncio
async def test_service_tier_null_clears() -> None:
    agent = Agent(model=TierStubModel(), tools=[])
    agent.service_tier = "priority"
    t = AgentSelf()
    with _active(agent):
        result = await t.run({"model_options": {"service_tier": None}})
    assert not result.is_error
    assert agent.service_tier is None
    assert "service_tier=unset" in result.content


@pytest.mark.asyncio
async def test_service_tier_listed_in_supported_diagnostics() -> None:
    agent = Agent(model=TierStubModel(), tools=[])
    t = AgentSelf()
    with _active(agent):
        result = await t.run({"diagnostics": True})
    assert "service_tier" in result.content
    assert "Service tier:" in result.content


@pytest.mark.asyncio
async def test_max_response_tokens_exceeds_model_cap() -> None:
    agent = _make_agent()
    t = AgentSelf()
    with _active(agent):
        result = await t.run({"max_response_tokens": 1_000_000})
    assert result.is_error
    assert "max_response_tokens" in result.content


@pytest.mark.asyncio
async def test_max_response_tokens_within_cap_applied() -> None:
    agent = _make_agent()
    t = AgentSelf()
    with _active(agent):
        result = await t.run({"max_response_tokens": 1_024})
    assert not result.is_error
    assert agent.max_response_tokens == 1_024


@pytest.mark.asyncio
async def test_account_empty_string_errors() -> None:
    agent = _make_agent(
        spec=ModelSpec(
            provider="OpenAISubscription",
            auth="credentials",
            model_id="gpt-5.5",
            account="work",
        ),
    )
    t = AgentSelf()
    with _active(agent):
        result = await t.run({"account": ""})
    assert result.is_error
    assert "account cannot be empty" in result.content


@pytest.mark.asyncio
async def test_provider_change_without_auth_uses_target_default() -> None:
    agent = _make_agent(
        spec=ModelSpec(
            provider="OpenAISubscription",
            auth="credentials",
            model_id="gpt-5.5",
            account="work",
        ),
    )
    fake_provider = MagicMock()
    fake_provider.model.return_value = StubProviderModel(model_id="gemini-3-pro")
    with patch(
        "sagent.tools.agent_self.build_provider",
        return_value=fake_provider,
    ) as build:
        t = AgentSelf()
        with _active(agent):
            result = await t.run({"provider": "Google", "model_id": "gemini-3-pro"})
    assert not result.is_error
    build.assert_called_once_with("Google", "env", account="work")


@pytest.mark.asyncio
async def test_account_default_string_is_preserved() -> None:
    agent = _make_agent(
        spec=ModelSpec(
            provider="OpenAISubscription",
            auth="credentials",
            model_id="gpt-5.5",
            account="work",
        ),
    )
    fake_provider = MagicMock()
    fake_provider.model.return_value = StubProviderModel(model_id="gpt-5")
    with patch(
        "sagent.tools.agent_self.build_provider",
        return_value=fake_provider,
    ) as build:
        t = AgentSelf()
        with _active(agent):
            result = await t.run({"model_id": "gpt-5", "account": "default"})
    assert not result.is_error
    build.assert_called_once_with("OpenAI", "env", account="default")


@pytest.mark.asyncio
async def test_model_id_change_unknown_provider_errors() -> None:
    agent = _make_agent(
        spec=ModelSpec(
            provider="UnknownProvider",
            auth="env",
            model_id="x",
            account="",
        ),
    )
    t = AgentSelf()
    with _active(agent):
        result = await t.run({"model_id": "newmodel"})
    assert result.is_error
    assert "Failed to build model" in result.content


@pytest.mark.asyncio
async def test_provider_catalog_respects_allow_list() -> None:
    """``catalog=providers`` lists only providers in the allow list."""
    agent = _make_agent()
    t = AgentSelf(allow_providers=("OpenAISubscription",))
    with _active(agent):
        result = await t.run({"catalog": "providers"})
    assert not result.is_error
    assert "OpenAISubscription" in result.content
    assert "DashScope" not in result.content
    assert "Moonshot" not in result.content


@pytest.mark.asyncio
async def test_model_catalog_rejects_provider_outside_allow_list() -> None:
    """``catalog=models, catalog_provider=...`` returns a clear message
    when the named provider isn't in the allow list.
    """
    agent = _make_agent()
    t = AgentSelf(allow_providers=("OpenAISubscription",))
    with _active(agent):
        result = await t.run({"catalog": "models", "catalog_provider": "DashScope"})
    assert not result.is_error
    assert "not in the allowed list" in result.content


@pytest.mark.asyncio
async def test_provider_switch_rejected_when_outside_allow_list() -> None:
    """An ``AgentSelf`` patch that asks to switch ``provider`` to a
    name not in the allow list returns an error before any
    ``build_provider`` call. Regression guard for the write surface
    of the allow-list contract (the read surface is the catalog).
    """
    spec = ModelSpec(
        provider="OpenAISubscription",
        auth="credentials",
        model_id="gpt-stub",
        account="",
    )
    agent = _make_agent(spec=spec)
    t = AgentSelf(allow_providers=("OpenAISubscription",))
    with _active(agent):
        result = await t.run({"provider": "Anthropic", "auth": "env"})
    assert result.is_error
    assert "not in the allowed list" in result.content
    assert "Anthropic" in result.content


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
