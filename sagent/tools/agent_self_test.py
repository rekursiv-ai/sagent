"""Tests for ``tools.agent_self``: agent self-mutation directives."""

from __future__ import annotations

from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from types import MappingProxyType
from unittest.mock import MagicMock, patch

import pytest

from sagent import providers as providers_module
from sagent.agent.agent import Agent
from sagent.agent.state import current_agent_var, tool_state_var
from sagent.testing import MockModelCaps
from sagent.tools.agent_self import AgentSelf
from sagent.types.cost import TokenCost
from sagent.types.model import (
    ModelCapability,
    ModelLimits,
    ModelRecipe,
    ModelRequest,
    ModelResponse,
    TokenCount,
)
from sagent.types.providers import ProviderOptions
from sagent.types.runtime import (
    AssistantMessage,
    Clear,
    Compact,
    Recompact,
    RuntimeEvent,
)


@dataclass(slots=True, kw_only=True)
class StubProviderModel(MockModelCaps):
    """Provider model that returns scripted responses on call."""

    model_id: str = "stub-1"
    max_request_tokens: int = 100_000
    responses: list[AssistantMessage] = field(default_factory=list)
    _idx: int = field(default=0, init=False)

    async def buffer(self, request: ModelRequest) -> ModelResponse:
        return await self.stream(request)

    async def stream(
        self,
        request: ModelRequest,
        publish: Callable[[RuntimeEvent], None] | None = None,
    ) -> ModelResponse:
        del request, publish
        idx = self._idx
        self._idx += 1
        msg = (
            self.responses[idx]
            if idx < len(self.responses)
            else AssistantMessage(text="ok")
        )
        return ModelResponse(message=msg)


@dataclass(frozen=True, slots=True, kw_only=True)
class _CatalogStub:
    """Stand-in provider class exposing only a ``CAPABILITIES`` catalog."""

    CAPABILITIES: Mapping[str, ModelCapability]


def _make_agent(*, spec: ModelRecipe | None = None) -> Agent:
    return Agent(
        model=StubProviderModel(),
        tools=[],
        model_recipe=spec,
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
async def test_exceeds_cap_suggests_window_variant_when_one_exists() -> None:
    """Over-raising the limit to reach a bigger window points at the +1m id.

    Regression for the AgentSelf footgun: asked to switch to a 1M-context
    model, agents kept the base 200k model id and forced
    ``max_request_tokens`` up -- rejected at the cap. The rejection now
    names the ``+1m`` sibling so the wrong path self-corrects.
    """
    catalog = _CatalogStub(
        CAPABILITIES={
            "big-base": ModelCapability(
                model_id="big-base",
                context_limits=MappingProxyType(
                    {
                        "": ModelLimits(
                            max_request_tokens=200_000, max_response_tokens=8_000
                        ),
                        "+1m": ModelLimits(
                            max_request_tokens=1_000_000, max_response_tokens=8_000
                        ),
                    }
                ),
            ),
        }
    )
    agent = Agent(
        model=StubProviderModel(model_id="big-base"),
        tools=[],
        model_recipe=ModelRecipe(
            provider="StubCat", auth="env", model_id="big-base", account=""
        ),
    )
    t = AgentSelf()
    with (
        _active(agent),
        patch.object(providers_module, "StubCat", catalog, create=True),
    ):
        result = await t.run({"max_request_tokens": 1_000_000})
    assert result.is_error
    assert "exceeds model" in result.content
    assert "model_id=big-base+1m" in result.content


@pytest.mark.asyncio
async def test_exceeds_cap_no_variant_hint_when_none_fits() -> None:
    """No window-variant sibling -> plain rejection, no spurious suggestion."""
    agent = _make_agent()  # StubProviderModel has no provider catalog.
    t = AgentSelf()
    with _active(agent):
        result = await t.run({"max_request_tokens": 999_999_999})
    assert result.is_error
    assert "exceeds model" in result.content
    assert "model_id=" not in result.content


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
async def test_model_change_auth_without_model_id_preserves_current_model() -> None:
    """Auth-only swap reuses the current model_id rather than erroring.

    Mirrors REPL ``/model --auth sub`` semantics (see
    ``repl/run_repl_test.py::test_parse_model_args_flag_auth``) so the
    tool surface and the slash command behave the same way. Without
    this, an agent that wants to bounce between subscription and API
    auth has to re-spell its model_id on every swap -- an LLM
    foot-gun.
    """
    agent = _make_agent(
        spec=ModelRecipe(provider="StubP", auth="env", model_id="stub-1", account=""),
    )
    t = AgentSelf()
    with (
        _active(agent),
        patch(
            "sagent.tools.agent_self.build_provider",
        ) as bp,
    ):
        bp.return_value = MagicMock(model=MagicMock(return_value=StubProviderModel()))
        result = await t.run({"auth": "new"})
    assert not result.is_error, result.content
    assert agent.model_recipe is not None
    assert agent.model_recipe.auth == "new"
    assert agent.model_recipe.model_id == "stub-1", (
        f"auth-only swap must preserve model_id; got {agent.model_recipe.model_id!r}"
    )


@pytest.mark.asyncio
async def test_diagnostics_returns_lines() -> None:
    agent = _make_agent(
        spec=ModelRecipe(provider="StubP", auth="env", model_id="stub-1", account=""),
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
async def test_diagnostics_reports_live_cost_tracker_state() -> None:
    """``diagnostics`` reads ``cost_tracker`` + ``num_tool_call_rounds``.

    Regression guard for the pre-existing dead ``ToolState.stats``
    contract (``docs/private/agent_v4_review.md`` P1): diagnostics now
    pulls directly from the single cost store, so a real model response
    surfaces in the output without any separate publisher step.
    """
    agent = _make_agent()
    agent.max_request_tokens = 50_000
    agent.max_response_tokens = 5_000
    response = ModelResponse(
        message=AssistantMessage(text="ok"),
        tokens=TokenCount(
            request=10_000,
            response=30,
            cache_write=7,
            cache_read=3,
        ),
        spend=TokenCost(request=0.42),
    )
    agent.cost_tracker.record_tokens(response, model_id="stub-1")
    agent.cost_tracker.record_cost(response)
    agent.activity.num_tool_call_rounds = 2
    t = AgentSelf()
    with _active(agent):
        result = await t.run({"diagnostics": True})
    assert "No stats yet" not in result.content
    assert "Tool call rounds:   2" in result.content
    assert "Max request tokens:   50,000" in result.content
    assert "Max response tokens:  5,000" in result.content
    assert "Input tokens:       10,000 (20.0% of max request)" in result.content
    assert "Total input tokens: 10,000" in result.content
    assert "Total output tokens:30" in result.content
    assert "Cache creation:     7" in result.content
    assert "Cache read:         3" in result.content
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

    service_tiers: tuple[str, ...] = ("auto", "default", "flex", "priority")


@pytest.mark.asyncio
async def test_service_tier_unsupported_when_model_lacks_capability() -> None:
    agent = _make_agent()  # StubProviderModel.spec.valid_service_tiers = ()
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


@dataclass(slots=True, kw_only=True)
class LatencyStubModel(StubProviderModel):
    """``StubProviderModel`` advertising a fast-latency mode."""

    latency_modes: tuple[str, ...] = ("fast",)


@pytest.mark.asyncio
@pytest.mark.parametrize("value", ["fast", "turbo", None])
async def test_latency_option_redirects_to_fast_model_tag(
    value: str | None,
) -> None:
    """``model_options.latency`` was replaced by the ``+fast`` model-id tag."""
    agent = Agent(model=LatencyStubModel(), tools=[])
    t = AgentSelf()
    with _active(agent):
        result = await t.run({"model_options": {"latency": value}})
    assert result.is_error
    assert "+fast" in result.content


@pytest.mark.asyncio
async def test_latency_derives_from_fast_model_tag() -> None:
    agent = Agent(
        model=LatencyStubModel(model_id="stub-1+fast"),
        tools=[],
    )
    t = AgentSelf()
    with _active(agent):
        result = await t.run({"diagnostics": True})
    assert not result.is_error
    assert agent.latency == "fast"
    assert "Latency:            fast" in result.content


@pytest.mark.asyncio
async def test_latency_shown_in_diagnostics() -> None:
    agent = Agent(model=LatencyStubModel(), tools=[])
    t = AgentSelf()
    with _active(agent):
        result = await t.run({"diagnostics": True})
    assert "Latency:" in result.content


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
        spec=ModelRecipe(
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
        spec=ModelRecipe(
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
    build.assert_called_once_with(
        "Google", "env", account="work", options=ProviderOptions()
    )


@pytest.mark.asyncio
async def test_account_default_string_is_preserved() -> None:
    agent = _make_agent(
        spec=ModelRecipe(
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
    build.assert_called_once_with(
        "OpenAISubscription",
        "credentials",
        account="default",
        options=ProviderOptions(),
    )


@pytest.mark.asyncio
async def test_model_swap_shrinks_budget_to_new_model_window() -> None:
    """Swapping to a smaller-window model must narrow the budget to fit.

    Regression: the swap once rejected an oversized budget instead of
    rescaling it down to the new model's window.
    """
    agent = _make_agent(
        spec=ModelRecipe(provider="StubP", auth="env", model_id="big", account=""),
    )
    # Agent's default budget tracks its current 100k-window model.
    assert agent.budget.max_request_tokens == 100_000
    fake_provider = MagicMock()
    fake_provider.model.return_value = StubProviderModel(
        model_id="small", max_request_tokens=50_000
    )
    with patch(
        "sagent.tools.agent_self.build_provider",
        return_value=fake_provider,
    ):
        t = AgentSelf()
        with _active(agent):
            result = await t.run({"model_id": "small"})
    assert not result.is_error, result.content
    assert agent.model.spec.tagged_model_id == "small"
    assert agent.budget.max_request_tokens <= 50_000


@pytest.mark.asyncio
async def test_model_swap_with_explicit_budget_lands_in_one_step() -> None:
    """Combined ``model_id`` + ``max_request_tokens`` must succeed.

    The patch lands both the swap and the explicit window in a single
    call: ``swap_model`` rescales the budget to the new model first, then
    the explicit cap is clamped on top, so no intermediate state exceeds
    the new model's window.
    """
    agent = _make_agent(
        spec=ModelRecipe(provider="StubP", auth="env", model_id="big", account=""),
    )
    fake_provider = MagicMock()
    fake_provider.model.return_value = StubProviderModel(
        model_id="small", max_request_tokens=50_000
    )
    with patch(
        "sagent.tools.agent_self.build_provider",
        return_value=fake_provider,
    ):
        t = AgentSelf()
        with _active(agent):
            result = await t.run({"model_id": "small", "max_request_tokens": 20_000})
    assert not result.is_error, result.content
    assert agent.model.spec.tagged_model_id == "small"
    assert agent.max_request_tokens == 20_000


@pytest.mark.asyncio
async def test_model_swap_clears_effort_and_reports_unset() -> None:
    """Swap to a model that lacks effort support; report the auto-clear.

    ``Agent.swap_model`` zeroes ``effort`` / ``thinking`` / ``service_tier``
    when the new model lacks the capability. ``_commit_patch_plan``'s
    "(unsupported)" line then has to fire so the LLM knows *why* its
    setting disappeared. The bug: ``_commit_patch_plan`` inspected the
    fields *after* swap, by which point ``swap_model`` had already
    cleared them -- silencing every report. Snapshot before swap.
    """

    @dataclass(slots=True, kw_only=True)
    class EffortStubModel(StubProviderModel):
        supports_effort: bool = True
        valid_efforts: tuple[str, ...] = ("low", "medium", "high")

    agent = Agent(
        model=EffortStubModel(model_id="rich-stub"),
        tools=[],
        model_recipe=ModelRecipe(
            provider="OpenAISubscription",
            auth="credentials",
            model_id="rich-stub",
            account="",
        ),
    )
    agent.effort = "high"
    assert agent.effort == "high"
    # Target model lacks effort/thinking/service-tier (``MockModelCaps``
    # defaults). Swap must clear ``effort`` AND mention it in the result.
    fake_provider = MagicMock()
    fake_provider.model.return_value = StubProviderModel(model_id="plain-stub")
    with patch(
        "sagent.tools.agent_self.build_provider",
        return_value=fake_provider,
    ):
        t = AgentSelf()
        with _active(agent):
            result = await t.run({"model_id": "plain-stub"})
    assert not result.is_error, result.content
    assert agent.effort is None
    assert "effort=unset" in result.content


@pytest.mark.asyncio
async def test_model_id_change_unknown_provider_errors() -> None:
    agent = _make_agent(
        spec=ModelRecipe(
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
    spec = ModelRecipe(
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


@pytest.mark.asyncio
@pytest.mark.parametrize("attr", ["max_request_tokens", "max_response_tokens"])
async def test_token_limit_rejects_non_integer_float(attr: str) -> None:
    """Schema declares ``type: integer``; runtime must reject 1.9.

    Pre-fix ``int(1.9) → 1`` and the spawn silently accepted the
    rounded request. A float here is a directive error.
    """
    agent = _make_agent()
    t = AgentSelf()
    with _active(agent):
        result = await t.run({attr: 1.9})
    assert result.is_error
    assert "integer" in result.content


@pytest.mark.asyncio
async def test_model_swap_clears_all_capabilities_and_reports_each_unset() -> None:
    # Swap from a fully-capable model to a plain one: thinking and
    # service_tier must each be cleared AND reported "(unsupported)". The
    # snapshot is read before swap, so a post-swap read would silence every
    # report -- this guards the thinking/service_tier axes the effort test
    # does not. (Latency needs no clearing: it derives from the model id.)
    @dataclass(slots=True, kw_only=True)
    class RichStubModel(StubProviderModel):
        supports_thinking: bool = True
        service_tiers: tuple[str, ...] = ("priority",)

    agent = Agent(
        model=RichStubModel(model_id="rich-stub"),
        tools=[],
        model_recipe=ModelRecipe(
            provider="OpenAISubscription",
            auth="credentials",
            model_id="rich-stub",
            account="",
        ),
    )
    agent.thinking = "adaptive"
    agent.service_tier = "priority"
    fake_provider = MagicMock()
    fake_provider.model.return_value = StubProviderModel(model_id="plain-stub")
    with patch(
        "sagent.tools.agent_self.build_provider",
        return_value=fake_provider,
    ):
        t = AgentSelf()
        with _active(agent):
            result = await t.run({"model_id": "plain-stub"})
    assert not result.is_error, result.content
    assert agent.thinking is None
    assert agent.service_tier is None
    assert "thinking=off (unsupported)" in result.content
    assert "service_tier=unset (unsupported)" in result.content


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
