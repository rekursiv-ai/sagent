"""Tests for ``agent.agent``: Agent composition class."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast, override
from unittest.mock import MagicMock, Mock, patch

import asyncio
import contextlib
import json
import logging

import pytest

from sagent import (
    providers as providers_module,
    types,
)
from sagent.agent.agent import (
    ActivityTracker,
    Agent,
    SystemPromptArg,
    _resolve_target_spec,
    _validate_input,
)
from sagent.agent.background import (
    BackgroundAwareTool,
    BackgroundTaskEntry,
    split_bg_args,
)
from sagent.agent.session_io import load_session
from sagent.agent.state import ToolState, agent_registry
from sagent.lib import last_models, token_count
from sagent.lib.json import JSON, json_freeze
from sagent.providers import Google
from sagent.types.tape import ContextSplice, TapeRecord, TapeRef


def _summary_override(
    summary: list[types.history.HistoryEntry],
    mint_ref: Callable[[], TapeRef],
    *,
    tape: Sequence[TapeRecord] | None = None,
    strategy: str = "summary",
    fallback_reason: str = "",
    preserved_tail_count: int = 0,
) -> ContextSplice:
    """Build a barrier splice carrying ``summary`` as its payload.

    When ``tape`` is supplied, the mask covers every existing record so
    every alive splice is absorbed and every HR is hidden. Without
    ``tape``, the splice has an empty mask (used by tests that only
    care about the payload and don't need barrier semantics).
    """
    if tape:
        mask: tuple[tuple[TapeRef, TapeRef], ...] = ((tape[0].ref, tape[-1].ref),)
    else:
        mask = ()
    return ContextSplice(
        ref=mint_ref(),
        mask=mask,
        insert_after=None,
        payload=tuple(summary),
        strategy=strategy,
        fallback_reason=fallback_reason,
        preserved_tail_count=preserved_tail_count,
    )


@dataclass(slots=True, kw_only=True)
class StubModel:
    """Configurable model that yields scripted responses."""

    model_id: str = "stub-1"
    max_request_tokens: int = 100_000
    max_response_tokens: int = 1_024
    supports_streaming: bool = True
    supports_thinking: bool = False
    supports_effort: bool = False
    supports_cache_control: bool = False
    valid_service_tiers: tuple[str, ...] = ()
    supports_context_management: bool = False
    supports_persistent_retry: bool = False
    supports_account_auth: bool = False
    max_image_dim: int = 8_000
    max_image_bytes: int = 5 * 1024 * 1024
    responses: list[types.history.AssistantMessage] = field(default_factory=list)
    received: list[types.model.ModelRequest] = field(default_factory=list)

    @property
    def pricing(self) -> types.model.Pricing:
        return types.model.Pricing()

    def approx_text_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    def approx_image_tokens(self, data: bytes) -> int:
        del data
        return 256

    def approx_request_tokens(self, request: types.model.ModelRequest) -> int:
        return token_count.approx_request_tokens(request, self)

    async def actual_text_tokens(self, text: str) -> int:
        return self.approx_text_tokens(text)

    async def actual_image_tokens(self, data: bytes) -> int:
        return self.approx_image_tokens(data)

    async def actual_request_tokens(self, request: types.model.ModelRequest) -> int:
        return self.approx_request_tokens(request)

    def is_context_overflow(self, error: Exception) -> bool:
        del error
        return False

    def is_retryable_provider_error(self, error: Exception) -> bool:
        del error
        return False

    async def buffer(
        self, request: types.model.ModelRequest
    ) -> types.model.ModelResponse:
        return await self.stream(request)

    async def stream(
        self,
        request: types.model.ModelRequest,
        on_text: object = None,
        on_thinking: object = None,
    ) -> types.model.ModelResponse:
        del on_text, on_thinking
        self.received.append(request)
        msg = (
            self.responses.pop(0)
            if self.responses
            else types.history.AssistantMessage(text="ok")
        )
        return types.model.ModelResponse(message=msg)


_STUB_SCHEMA: JSON = json_freeze({"type": "object"})


@dataclass(slots=True, kw_only=True)
class StubTool:
    """Minimal tool that records calls."""

    name: str = "Echo"
    tool_id: str = "application/x-tool-echo"
    description: str = "Echo tool."
    directive_schema: JSON = _STUB_SCHEMA
    clearable_results: bool = False
    calls: list[Mapping[str, object]] = field(default_factory=list)

    def summary(self, args: Mapping[str, object]) -> str:
        del args
        return "echo"

    def summary_result(self, result: types.history.ToolResult) -> str | None:
        del result
        return None

    def prompt(self) -> str:
        return ""

    async def run(self, args: Mapping[str, object]) -> types.history.ToolResult:
        self.calls.append(args)
        return types.history.ToolResult(call_id="", content=str(args.get("msg", "")))


def _build_agent(
    *,
    model: types.model.Model | None = None,
    tools: list[types.tools.Tool] | None = None,
    system: SystemPromptArg = "",
    budget: types.model.ContextBudget | None = None,
    max_budget_usd: float | None = None,
    session_dir: Path | None = None,
) -> Agent:
    return Agent(
        model=model or StubModel(),
        tools=tools or [],
        system=system,
        budget=budget,
        max_budget_usd=max_budget_usd,
        session_dir=session_dir,
    )


def test_agent_init_sets_basics() -> None:
    a = _build_agent()
    assert a.model is not None
    assert a.tools == []
    assert isinstance(a.activity, ActivityTracker)
    assert a.history == []
    assert a.cost_tracker.total_cost_usd == 0.0


def test_agent_budget_defaults_from_model() -> None:
    a = _build_agent()
    assert isinstance(a.budget, types.model.ContextBudget)
    assert a.budget.max_request_tokens == 100_000


def test_agent_budget_override_respected() -> None:
    b = types.model.ContextBudget.from_model(StubModel())
    a = _build_agent(budget=b)
    assert a.budget is b


def test_agent_register_and_cancel_background() -> None:
    a = _build_agent()
    loop = asyncio.new_event_loop()
    try:
        task = loop.create_task(asyncio.sleep(0))
        entry = BackgroundTaskEntry(
            task=task,
            tool_name="bg",
            queue_id="job-1",
            started=0.0,
        )
        a.register_background("job-1", entry)
        assert "job-1" in a.background
        a.cancel_background("job-1")
        assert "job-1" not in a.background
        _ = task.cancel()
        # Drive the loop so the cancellation propagates; otherwise the
        # ``Task`` is GC'd in pending state and CPython logs the warning.
        loop.run_until_complete(asyncio.gather(task, return_exceptions=True))
    finally:
        loop.close()


def test_agent_system_string_to_factory() -> None:
    a = _build_agent(system="hi there")
    assert a.system_prompt() == "hi there"


def test_agent_system_callable_passthrough() -> None:
    calls: list[int] = []

    def make_sys() -> str:
        calls.append(1)
        return f"call {len(calls)}"

    a = _build_agent(system=make_sys)
    n_before = len(calls)
    p1 = a.system_prompt()
    p2 = a.system_prompt()
    # Each ``system_prompt`` call re-invokes the factory; counters
    # therefore advance by exactly two between the two calls.
    assert p2 != p1
    assert len(calls) == n_before + 2


def test_agent_tools_map_stores_raw_tools() -> None:
    """H2: ``tools_map`` stores raw rich tools so isinstance / Protocol
    checks at consumer sites (CompactRestorable, Slack identity swap,
    etc.) pass through. Wrapping happens per-request in
    ``_AgentModel.stream``.
    """
    tool: types.tools.Tool = StubTool()
    a = _build_agent(tools=[tool])
    assert "Echo" in a.tools_map
    stored = a.tools_map["Echo"]
    assert stored is tool
    assert not isinstance(stored, BackgroundAwareTool)


@pytest.mark.asyncio
async def test_agent_request_tools_wrapped_in_background_aware() -> None:
    """H2: per-request wrapping advertises ``background`` / ``delay`` to
    the model while keeping ``tools_map`` storage raw.
    """
    model = StubModel()
    tool = StubTool(
        directive_schema=json_freeze(
            {"type": "object", "properties": {"msg": {"type": "string"}}}
        ),
    )
    a = _build_agent(model=model, tools=[tool])
    async for _ in a.run(types.history.UserMessage(text="hi")):
        pass
    assert model.received
    req = model.received[-1]
    assert req.tools is not None
    seen = req.tools[0]
    props = dict(cast(Mapping[str, object], seen.directive_schema["properties"]))
    assert "background" in props
    assert "delay" in props


@pytest.mark.asyncio
async def test_agent_run_yields_idle_at_end() -> None:
    a = _build_agent()
    events: list[str] = [
        type(ev).__name__ async for ev in a.run(types.history.UserMessage(text="ping"))
    ]
    assert "ModelIdle" in events
    assert len(a.history) >= 2
    assert isinstance(a.history[0], types.history.UserMessage)


@pytest.mark.asyncio
async def test_agent_run_passes_rich_tools_to_model() -> None:
    """Regression: provider iterates ``request.tools`` reading ``.description``
    etc.; the runtime hands the model layer ``_AgentTool`` wrappers, which
    expose only ``name``/``run``. The Agent must translate them back to rich
    tools before constructing ``types.model.ModelRequest`` so providers see the full
    Tool surface.
    """
    model = StubModel()
    tool: types.tools.Tool = StubTool()
    a = _build_agent(model=model, tools=[tool])
    async for _ in a.run(types.history.UserMessage(text="hi")):
        pass
    assert model.received, "model.stream must have been invoked"
    req = model.received[-1]
    assert req.tools is not None
    assert len(req.tools) == 1
    seen = req.tools[0]
    # Every Tool-protocol attribute the providers read must resolve.
    assert seen.name == "Echo"
    assert seen.description == "Echo tool."
    assert seen.tool_id == "application/x-tool-echo"
    assert dict(seen.directive_schema) == {"type": "object"}
    assert seen.summary({}) == "echo"
    assert seen.prompt() == ""


@pytest.mark.asyncio
async def test_agent_records_response_into_cost_tracker() -> None:
    model = StubModel()
    a = _build_agent(model=model)
    async for _ in a.run(types.history.UserMessage(text="ping")):
        pass
    # StubModel emits empty types.model.TokenCount, but ``calls_by_model`` records
    # one entry per model invocation.
    assert a.cost_tracker.calls_by_model.get("stub-1", 0) >= 1


def test_agent_record_response_budget_exhaustion_raises() -> None:
    a = _build_agent(max_budget_usd=1.0)
    # First response below the cap: clean.
    a.record_response(
        types.model.ModelResponse(message=types.history.AssistantMessage(text="x"))
    )
    # Force an over-budget total and verify the next call raises.
    a.cost_tracker.total_cost_usd = 2.0
    with pytest.raises(RuntimeError, match="Budget exhausted"):
        a.record_response(
            types.model.ModelResponse(message=types.history.AssistantMessage(text="x"))
        )


def test_token_count_addable() -> None:
    """``types.model.TokenCount`` supports ``+`` so ``CostTracker.record`` can fold."""
    a = types.model.TokenCount()
    b = types.model.TokenCount()
    c = a + b
    assert isinstance(c, types.model.TokenCount)


def test_agent_shutdown_idempotent() -> None:
    a = _build_agent()
    a.shutdown()
    a.shutdown()  # Second call must not raise.


@pytest.mark.asyncio
async def test_agent_shutdown_closes_active_model_once() -> None:
    @dataclass(slots=True, kw_only=True)
    class ClosableStubModel(StubModel):
        close_count: int = 0
        closed_event: asyncio.Event = field(default_factory=asyncio.Event)

        async def close(self) -> None:
            self.close_count += 1
            self.closed_event.set()

    model = ClosableStubModel()
    a = _build_agent(model=model)
    a.shutdown()
    a.shutdown()
    await asyncio.wait_for(model.closed_event.wait(), timeout=1.0)
    assert model.close_count == 1


def test_system_prompt_arg_type_alias_str_or_callable() -> None:
    # ``SystemPromptArg`` is exposed as a type alias for the constructor.
    arg: SystemPromptArg = "hi"
    assert isinstance(arg, str)

    def factory() -> str:
        return "x"

    arg2: SystemPromptArg = factory
    assert callable(arg2)


def test_max_request_tokens_setter_rejects_over_model_limit() -> None:
    a = _build_agent()
    with pytest.raises(ValueError, match="exceeds model's"):
        a.max_request_tokens = a.model.max_request_tokens + 1


def test_max_request_tokens_setter_accepts_within_limit() -> None:
    a = _build_agent()
    a.max_request_tokens = 50_000
    assert a.budget.max_request_tokens == 50_000


def test_max_response_tokens_setter_rejects_over_model_limit() -> None:
    a = _build_agent()
    with pytest.raises(ValueError, match="exceeds model's"):
        a.max_response_tokens = a.model.max_response_tokens + 1


def test_max_response_tokens_setter_accepts_within_limit() -> None:
    a = _build_agent()
    a.max_response_tokens = 256
    assert a.budget.max_response_tokens == 256


def test_reset_budget_restores_model_defaults() -> None:
    a = _build_agent()
    a.max_request_tokens = 50_000  # under buffer+chars-per-token constraints
    assert a.budget.max_request_tokens == 50_000
    a.reset_budget()
    assert a.budget.max_request_tokens == a.model.max_request_tokens


def test_thinking_setter() -> None:
    a = _build_agent()
    a.thinking = "extended"
    assert a.thinking == "extended"
    assert a.thinking_state is None
    a.thinking = None
    assert a.thinking is None


def test_thinking_state_sets_request_and_display_without_provider_arg() -> None:
    a = _build_agent()
    a.set_thinking_state("on-show")
    assert a.thinking_state == "on-show"
    assert a.thinking == "enabled"
    assert a.show_thinking is True
    assert "redact_thinking" not in a.provider_args
    a.set_thinking_state("redact-hide")
    assert a.thinking == "adaptive"
    assert a.show_thinking is False
    assert "redact_thinking" not in a.provider_args


def test_change_model_derives_redact_thinking_from_state(
    patched_build_provider: Mapping[str, object],
) -> None:
    del patched_build_provider
    a = _build_agent_with_spec()
    a.set_thinking_state("redact-hide")
    _ = a.change_model(model_id="claude-sonnet-4-6")
    build_provider = cast(Mock, providers_module.build_provider)
    build_provider.assert_called_with(
        "Anthropic",
        "api",
        account=None,
        redact_thinking=True,
    )


@pytest.mark.asyncio
async def test_agent_default_omits_thinking_from_request() -> None:
    model = StubModel(supports_thinking=True)
    a = _build_agent(model=model)
    async for _ in a.run(types.history.UserMessage(text="hi")):
        pass
    assert model.received[-1].thinking is None


@pytest.mark.asyncio
async def test_agent_thinking_state_sets_request_thinking() -> None:
    model = StubModel(supports_thinking=True)
    a = Agent(model=model, tools=[], thinking_state="adaptive-hide")
    async for _ in a.run(types.history.UserMessage(text="hi")):
        pass
    assert model.received[-1].thinking == "adaptive"


def test_effort_setter_rejects_when_model_lacks_support() -> None:
    a = _build_agent()  # StubModel.supports_effort = False
    with pytest.raises(ValueError, match="does not support effort"):
        a.effort = "high"


def test_effort_setter_accepts_when_model_supports() -> None:
    model = StubModel(supports_effort=True)
    a = _build_agent(model=model)
    a.effort = "medium"
    assert a.effort == "medium"


def test_effort_setter_accepts_none_unconditionally() -> None:
    a = _build_agent()
    a.effort = None
    assert a.effort is None


def test_cache_ttl_setter_rejects_invalid() -> None:
    a = _build_agent()
    with pytest.raises(ValueError, match="cache_ttl must be"):
        a.cache_ttl = "10m"


def test_cache_ttl_setter_accepts_valid() -> None:
    a = _build_agent()
    a.cache_ttl = "1h"
    assert a.cache_ttl == "1h"


def test_service_tier_setter_rejects_when_model_lacks_support() -> None:
    a = _build_agent()  # StubModel.valid_service_tiers = ()
    with pytest.raises(ValueError, match="does not support service_tier"):
        a.service_tier = "priority"


def test_service_tier_setter_rejects_unknown_value() -> None:
    model = StubModel(valid_service_tiers=("auto", "priority"))
    a = _build_agent(model=model)
    with pytest.raises(ValueError, match="service_tier must be one of"):
        a.service_tier = "turbo"


def test_service_tier_setter_accepts_when_model_supports() -> None:
    model = StubModel(valid_service_tiers=("auto", "priority"))
    a = _build_agent(model=model)
    a.service_tier = "priority"
    assert a.service_tier == "priority"
    a.service_tier = None
    assert a.service_tier is None


def test_status_setter_round_trip() -> None:
    a = _build_agent()
    a.status = "busy"
    assert a.status == "busy"


def test_session_id_is_hex_string() -> None:
    a = _build_agent()
    assert len(a.session_id) == 8


def test_inbox_and_work_properties_reflect_runtime() -> None:
    a = _build_agent()
    assert a.inbox is a.runtime.inbox
    # No model call or compact_task is active right after construction.
    assert a.work is None


def test_tools_property_lists_wrapped_tools() -> None:
    tool: types.tools.Tool = StubTool()
    a = _build_agent(tools=[tool])
    assert len(a.tools) == 1


def test_total_cost_total_tokens_num_rounds_initially_zero() -> None:
    a = _build_agent()
    assert a.total_cost_usd == 0.0
    assert a.total_tokens.input_tokens == 0
    assert a.num_tool_call_rounds == 0


def test_system_property_rebuilds_each_access() -> None:
    """``Agent.system`` property re-runs ``_build_system``."""
    a = _build_agent(system="root")
    assert a.system == "root"


def test_background_merges_detached_and_explicit() -> None:
    a = _build_agent()
    loop = asyncio.new_event_loop()
    try:
        # Stash a detached task on the runtime; expect it surfaces.
        det_task = loop.create_task(asyncio.sleep(0))
        a.runtime.detached["det-1"] = det_task
        # Register an explicit job too.
        ex_task = loop.create_task(asyncio.sleep(0))
        a.register_background(
            "job-1",
            BackgroundTaskEntry(
                task=ex_task, tool_name="X", queue_id="job-1", started=0.0
            ),
        )
        merged = a.background
        assert "det-1" in merged
        assert "job-1" in merged
        assert merged["det-1"].kind == "detached"
        _ = det_task.cancel()
        _ = ex_task.cancel()
        loop.run_until_complete(
            asyncio.gather(det_task, ex_task, return_exceptions=True),
        )
    finally:
        loop.close()


def test_swap_model_rejects_smaller_request_window() -> None:
    a = _build_agent()
    smaller = StubModel(max_request_tokens=50)
    with pytest.raises(ValueError, match="max_request_tokens"):
        a.swap_model(smaller)


def test_swap_model_rejects_smaller_response_window() -> None:
    a = _build_agent()
    smaller = StubModel(max_response_tokens=10)
    with pytest.raises(ValueError, match="max_response_tokens"):
        a.swap_model(smaller)


def test_swap_model_replaces_model_and_inner_wrapper() -> None:
    a = _build_agent()
    new = StubModel(model_id="stub-2")
    a.swap_model(new)
    assert a.model is new
    # The wrapper's inner reference was updated too.
    assert a._agent_model._inner is new


def test_swap_model_clears_unsupported_service_tier() -> None:
    a = _build_agent(model=StubModel(valid_service_tiers=("priority",)))
    a.service_tier = "priority"
    a.swap_model(StubModel(model_id="stub-2"))
    assert a.service_tier is None


def test_swap_model_normalizes_unsupported_cache_ttl() -> None:
    a = _build_agent(model=StubModel(supports_cache_control=True))
    a.cache_ttl = "1h"
    a.swap_model(StubModel(model_id="stub-2"))
    assert a.cache_ttl == "5m"


@pytest.mark.asyncio
async def test_swap_model_schedules_close_on_old_cli_model() -> None:
    """CLI providers own subprocess pools via ``HotSpare`` and define
    ``async def close()``. ``Agent.swap_model`` must schedule that
    teardown for the swapped-out model so the prior ``claude`` /
    ``gemini`` subprocess and warming-spare task don't leak past the
    swap. API-key models (no ``close``) are skipped silently.
    """

    @dataclass(slots=True, kw_only=True)
    class ClosableStubModel(StubModel):
        closed_event: asyncio.Event = field(default_factory=asyncio.Event)

        async def close(self) -> None:
            self.closed_event.set()

    old = ClosableStubModel()
    a = _build_agent(model=old)
    a.swap_model(StubModel(model_id="stub-2"))
    # The scheduled close task needs the loop to step once before its
    # body runs; ``wait_for`` with a short timeout is more robust than
    # ``sleep(0)`` against scheduler quirks (the close coroutine itself
    # only sets an Event, so this resolves immediately under correct
    # behavior and times out under regression).
    await asyncio.wait_for(old.closed_event.wait(), timeout=1.0)
    assert old.closed_event.is_set()


@pytest.mark.real_sleep
@pytest.mark.asyncio
async def test_swap_model_logs_close_failure_via_log_task_exception(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A crash inside ``close()`` on the swapped-out model lands at ERROR.

    The teardown task is fire-and-forget. Before the unification, the
    site used a hand-rolled done-callback (``_log_close_errors``) that
    called ``log_exception_or_warning`` outside any ``except`` block --
    so ``logger.exception`` read an empty ``sys.exc_info()`` and the
    traceback never made it into the log record. Replacing it with the
    shared ``log_task_exception`` helper (which passes ``exc_info=exc``
    explicitly) preserves the traceback. Pin both contracts here:
    ERROR level, AND traceback present.
    """
    import logging  # noqa: PLC0415 -- test-local

    @dataclass(slots=True, kw_only=True)
    class CrashingCloseModel(StubModel):
        closed_event: asyncio.Event = field(default_factory=asyncio.Event)

        async def close(self) -> None:
            try:
                raise RuntimeError("simulated close failure")
            finally:
                self.closed_event.set()

    old = CrashingCloseModel()
    a = _build_agent(model=old)
    logger_name = "sagent.agent.agent"
    caplog.set_level(logging.DEBUG, logger=logger_name)
    a.swap_model(StubModel(model_id="stub-2"))
    # Wait for the close coroutine to start its raise.
    await asyncio.wait_for(old.closed_event.wait(), timeout=1.0)
    # The done-callback fires on a later loop tick after the task
    # transitions to done. Poll a handful of yields rather than
    # guess the exact tick count.
    for _ in range(20):
        await asyncio.sleep(0.001)
        if any(
            r.name == logger_name and r.levelname == "ERROR" for r in caplog.records
        ):
            break
    errs = [
        r for r in caplog.records if r.name == logger_name and r.levelname == "ERROR"
    ]
    assert errs, (
        "swapped-out model close failure must surface at ERROR; "
        f"records={[(r.levelname, r.getMessage()) for r in caplog.records]!r}"
    )
    assert errs[0].exc_info is not None, "close ERROR must carry a traceback"


# --- Agent.change_model -----------------------------------------------------


def _build_agent_with_spec(model_id: str = "claude-opus-4-7") -> Agent:
    """Build an agent with a real ``types.model.ModelSpec`` so ``change_model`` works.

    ``_build_agent`` uses ``StubModel`` without a spec. ``change_model``
    consults ``self.model_spec`` to inherit fields and to detect the
    cross-provider case; this helper attaches an Anthropic spec.
    """
    a = _build_agent()
    a.model_spec = types.model.ModelSpec(
        provider="Anthropic", auth="api", model_id=model_id, account=None
    )
    return a


@pytest.fixture
def patched_build_provider() -> Iterator[Mapping[str, object]]:
    """Stub ``build_provider`` so ``change_model`` can run without real creds.

    The provider's ``model(model_id)`` returns a minimal stub model that
    satisfies the budget guards in ``swap_model``.
    """

    def fake_model(model_id: str) -> StubModel:
        return StubModel(model_id=model_id)

    fake_provider = MagicMock()
    fake_provider.model.side_effect = fake_model

    # ``change_model`` calls ``providers.build_provider(...)`` via the
    # package module binding -- patch the function on the providers
    # package and all consumers see the stub.
    with patch(
        "sagent.providers.build_provider",
        return_value=fake_provider,
    ):
        yield {"provider": fake_provider}


def test_change_model_same_provider_new_model_queues_swap(
    patched_build_provider: Mapping[str, object],
) -> None:
    """Naming a new model on the current provider queues the swap."""
    del patched_build_provider
    a = _build_agent_with_spec()
    target = a.change_model(model_id="claude-sonnet-4-6")
    assert target.provider == "Anthropic"
    assert target.model_id == "claude-sonnet-4-6"
    items = asyncio.new_event_loop().run_until_complete(a.runtime.inbox.drain())
    switches = [m for m in items if isinstance(m, types.runtime.ModelSwitch)]
    assert len(switches) == 1


def test_change_model_cross_provider_no_model_preserves_current(
    patched_build_provider: Mapping[str, object],
) -> None:
    """Cross-provider with no model_id keeps the current model when known.

    ``Anthropic`` / ``AnthropicCLI`` share ``KNOWN_MODELS`` via
    inheritance so ``claude-opus-4-7`` is valid on both transports.
    """
    del patched_build_provider
    a = _build_agent_with_spec(model_id="claude-opus-4-7")
    target = a.change_model(provider="AnthropicCLI")
    assert target.provider == "AnthropicCLI"
    assert target.model_id == "claude-opus-4-7"


def test_resolve_target_spec_provider_change_uses_target_default_auth() -> None:
    spec = types.model.ModelSpec(
        provider="OpenAISubscription",
        auth="credentials",
        model_id="gpt-5.5",
        account="work",
    )
    target = _resolve_target_spec(
        spec,
        provider="Google",
        auth=None,
        model_id="gemini-3-pro",
        account=None,
    )
    assert target.auth == "env"
    assert target.account == "work"


def test_resolve_target_spec_explicit_default_account_is_preserved() -> None:
    spec = types.model.ModelSpec(
        provider="OpenAISubscription",
        auth="credentials",
        model_id="gpt-5.5",
        account="work",
    )
    target = _resolve_target_spec(
        spec,
        provider=None,
        auth=None,
        model_id="gpt-5",
        account="default",
    )
    assert target.account == "default"


def test_change_model_provider_change_without_auth_uses_target_default(
    patched_build_provider: Mapping[str, object],
) -> None:
    del patched_build_provider
    a = _build_agent_with_spec(model_id="claude-opus-4-7")
    target = a.change_model(provider="Google", model_id="gemini-3-pro")
    assert target.auth == "env"


def test_change_model_cross_provider_falls_back_to_default(
    monkeypatch: pytest.MonkeyPatch,
    patched_build_provider: Mapping[str, object],
) -> None:
    """When current model isn't in new provider's catalog, use DEFAULT_MODEL.

    ``claude-opus-4-7`` is not in ``Google.KNOWN_MODELS``; with no
    ``last_models`` entry recorded, resolution falls through to
    ``Google.DEFAULT_MODEL``.
    """
    del patched_build_provider

    def _empty_last_models() -> dict[str, str]:
        return {}

    monkeypatch.setattr(last_models, "load", _empty_last_models)
    a = _build_agent_with_spec(model_id="claude-opus-4-7")
    target = a.change_model(provider="Google", auth="env")
    assert target.provider == "Google"
    assert target.model_id == Google.DEFAULT_MODEL


def test_change_model_cross_provider_falls_back_to_last_models(
    monkeypatch: pytest.MonkeyPatch,
    patched_build_provider: Mapping[str, object],
) -> None:
    """``last_models`` takes precedence over ``DEFAULT_MODEL`` on cross-provider."""
    del patched_build_provider

    def _remembered() -> dict[str, str]:
        return {"Google": "gemini-2.5-flash-lite"}

    monkeypatch.setattr(last_models, "load", _remembered)
    a = _build_agent_with_spec(model_id="claude-opus-4-7")
    target = a.change_model(provider="Google", auth="env")
    assert target.model_id == "gemini-2.5-flash-lite"


def test_change_model_unknown_provider_raises(
    patched_build_provider: Mapping[str, object],
) -> None:
    """Naming a non-existent provider class surfaces as ``ValueError``."""
    del patched_build_provider
    a = _build_agent_with_spec()
    with pytest.raises((AttributeError, ValueError), match="unknown provider"):
        _ = a.change_model(provider="NotAProvider")


def test_change_model_returns_resolved_spec(
    patched_build_provider: Mapping[str, object],
) -> None:
    """Return value is the resolved target spec, not the old one."""
    del patched_build_provider
    a = _build_agent_with_spec()
    target = a.change_model(model_id="claude-sonnet-4-6")
    assert target.model_id == "claude-sonnet-4-6"
    assert target.provider == "Anthropic"


def test_change_model_apply_resets_oversized_budget(
    patched_build_provider: Mapping[str, object],
) -> None:
    """Queued ``/model`` swaps reset stale budgets before applying the model."""
    del patched_build_provider
    a = _build_agent(
        model=StubModel(max_request_tokens=1_000_000, max_response_tokens=32_000)
    )
    a.model_spec = types.model.ModelSpec(
        provider="Anthropic", auth="api", model_id="claude-opus-4-7+1m"
    )
    _ = a.change_model(model_id="claude-sonnet-4-6")
    items = asyncio.new_event_loop().run_until_complete(a.runtime.inbox.drain())
    switch = next(m for m in items if isinstance(m, types.runtime.ModelSwitch))
    switch.apply()
    assert a.model.model_id == "claude-sonnet-4-6"
    assert a.max_request_tokens == a.model.max_request_tokens


# --- Agent.relogin ----------------------------------------------------------


@pytest.mark.asyncio
async def test_relogin_calls_login_classmethod() -> None:
    """``Agent.relogin`` drives the provider class's ``login`` classmethod."""
    a = _build_agent_with_spec()
    login_mock = MagicMock()
    fake_provider_cls = MagicMock()
    fake_provider_cls.login = login_mock
    with patch.object(providers_module, "Anthropic", fake_provider_cls, create=True):
        await a.relogin()
    login_mock.assert_called_once()


@pytest.mark.asyncio
async def test_relogin_reloads_auth_when_provider_supports_protocol() -> None:
    """When the running provider satisfies ``AuthReloadable``, hot-reload it.

    After ``login`` writes fresh credentials to disk, the in-memory
    refresh token is still the old one. ``handle_auth_error`` is the
    provider's "re-read disk creds" entry point.
    """

    @dataclass(slots=True, kw_only=True)
    class ReloadableProvider:
        handle_auth_error_count: int = 0

        async def handle_auth_error(self) -> None:
            self.handle_auth_error_count += 1

    @dataclass(slots=True, kw_only=True)
    class _ModelWithProvider(StubModel):
        _provider: object | None = None

    live_provider = ReloadableProvider()
    model = _ModelWithProvider(_provider=live_provider)

    a = _build_agent(model=model)
    a.model_spec = types.model.ModelSpec(
        provider="Anthropic", auth="api", model_id="claude-opus-4-7"
    )

    fake_cls = MagicMock()
    fake_cls.login = MagicMock()
    with patch.object(providers_module, "Anthropic", fake_cls, create=True):
        await a.relogin()
    assert live_provider.handle_auth_error_count == 1


@pytest.mark.asyncio
async def test_relogin_raises_when_provider_has_no_login() -> None:
    """Providers without a ``login`` classmethod surface as ``ValueError``."""
    a = _build_agent_with_spec()
    fake_cls = MagicMock(spec=[])  # no ``login`` attribute
    with patch.object(providers_module, "Anthropic", fake_cls, create=True):  # noqa: SIM117 -- pytest.raises is a separate context manager and reads more clearly when nested with the patch
        with pytest.raises(ValueError, match="no login method"):
            await a.relogin()


@pytest.mark.asyncio
async def test_relogin_unknown_provider_raises() -> None:
    """``relogin`` for an unrecognized provider class raises ``ValueError``."""
    a = _build_agent()
    a.model_spec = types.model.ModelSpec(
        provider="NotAProvider", auth="api", model_id="x"
    )
    with pytest.raises(ValueError, match="unknown provider"):
        await a.relogin()


def test_halt_pushes_halt_event() -> None:
    a = _build_agent()
    a.halt()
    items = asyncio.new_event_loop().run_until_complete(a.runtime.inbox.drain())
    assert any(isinstance(i, types.runtime.Halt) for i in items)


def test_kill_tool_pushes_kill_event() -> None:
    a = _build_agent()
    a.kill_tool("call-7")
    items = asyncio.new_event_loop().run_until_complete(a.runtime.inbox.drain())
    kills = [i for i in items if isinstance(i, types.runtime.Kill)]
    assert len(kills) == 1
    assert kills[0].call_id == "call-7"


def test_kill_all_tools_pushes_kill_with_none_id() -> None:
    a = _build_agent()
    a.kill_all_tools()
    items = asyncio.new_event_loop().run_until_complete(a.runtime.inbox.drain())
    kills = [i for i in items if isinstance(i, types.runtime.Kill)]
    assert kills[0].call_id is None


@pytest.mark.asyncio
async def test_shutdown_force_cancels_explicit_jobs() -> None:
    a = _build_agent()

    async def hang() -> None:
        await asyncio.sleep(10.0)

    task = asyncio.create_task(hang())
    a.register_background(
        "job-x",
        BackgroundTaskEntry(
            task=task,
            tool_name="bg",
            queue_id="job-x",
            started=0.0,
            hidden=False,
        ),
    )
    a.shutdown(force=True)
    # Yield so the cancellation propagates.
    with contextlib.suppress(asyncio.CancelledError):
        await task
    assert task.cancelled()


@pytest.mark.asyncio
async def test_shutdown_force_uses_persistent_subagent_lifecycle() -> None:
    @dataclass(slots=True, kw_only=True)
    class _Child:
        shutdown_calls: list[bool] = field(default_factory=list)

        def shutdown(self, *, force: bool = False) -> None:
            self.shutdown_calls.append(force)

    a = _build_agent()
    child = _Child()

    async def hang() -> None:
        await asyncio.sleep(10.0)

    task = asyncio.create_task(hang())
    a.register_background(
        "child",
        BackgroundTaskEntry(
            task=task,
            tool_name="Agent",
            queue_id="child",
            started=0.0,
            hidden=False,
            kind="persistent_subagent",
        ),
    )
    agent_registry["child"] = cast(Agent, child)
    try:
        a.shutdown(force=True)
        await asyncio.sleep(0)
        assert child.shutdown_calls == [True]
        assert not task.cancelled()
    finally:
        _ = agent_registry.pop("child", None)
        _ = task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_compact_awaits_compact_complete_event() -> None:
    @dataclass(slots=True, kw_only=True)
    class _StubCompactor:
        async def should_compact(
            self,
            input_tokens: int,
            max_request_tokens: int,
            max_response_tokens: int = 0,
        ) -> bool:
            del input_tokens, max_request_tokens, max_response_tokens
            return False

        async def compact(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[types.history.HistoryEntry],
            model: object,
            mint_ref: Callable[[], TapeRef],
            custom_instructions: str | None = None,
        ) -> ContextSplice:
            del context, model
            del custom_instructions
            return _summary_override(
                [types.history.UserMessage(text="[summary]")], mint_ref, tape=tape
            )

        def maintain(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[types.history.HistoryEntry],
            tools: object,
            mint_ref: Callable[[], TapeRef],
        ) -> tuple[ContextSplice, ...]:
            del tape, context, tools, mint_ref
            return ()

    a = Agent(model=StubModel(), tools=[], compactor=_StubCompactor())

    async def drive() -> None:
        await a.serve_forever()

    drive_task = asyncio.create_task(drive())
    try:
        await a.compact("")
    finally:
        a.shutdown()
        with contextlib.suppress(asyncio.CancelledError):
            await drive_task

    assert any(
        isinstance(e, types.history.UserMessage) and e.text == "[summary]"
        for e in a.history
    )


@pytest.mark.asyncio
async def test_public_compact_returns_when_halt_cancels_compaction() -> None:
    started = asyncio.Event()

    @dataclass(slots=True, kw_only=True)
    class _HangingCompactor:
        async def should_compact(
            self,
            input_tokens: int,
            max_request_tokens: int,
            max_response_tokens: int = 0,
        ) -> bool:
            del input_tokens, max_request_tokens, max_response_tokens
            return False

        async def compact(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[types.history.HistoryEntry],
            model: object,
            mint_ref: Callable[[], TapeRef],
            custom_instructions: str | None = None,
        ) -> ContextSplice:
            del tape, context, model, mint_ref, custom_instructions
            started.set()
            await asyncio.Future[None]()
            raise AssertionError("unreachable")

    a = Agent(model=StubModel(), tools=[], compactor=_HangingCompactor())
    drive_task = asyncio.create_task(a.serve_forever())
    compact_task = asyncio.create_task(a.compact(""))
    try:
        await asyncio.wait_for(started.wait(), timeout=1.0)
        a.halt()
        await asyncio.wait_for(compact_task, timeout=1.0)
    finally:
        a.shutdown()
        with contextlib.suppress(asyncio.CancelledError):
            await drive_task


@pytest.mark.asyncio
async def test_public_recompact_returns_when_clear_cancels_compaction() -> None:
    started = asyncio.Event()

    @dataclass(slots=True, kw_only=True)
    class _HangingCompactor:
        async def should_compact(
            self,
            input_tokens: int,
            max_request_tokens: int,
            max_response_tokens: int = 0,
        ) -> bool:
            del input_tokens, max_request_tokens, max_response_tokens
            return False

        async def compact(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[types.history.HistoryEntry],
            model: object,
            mint_ref: Callable[[], TapeRef],
            custom_instructions: str | None = None,
        ) -> ContextSplice:
            del tape, context, model, mint_ref, custom_instructions
            started.set()
            await asyncio.Future[None]()
            raise AssertionError("unreachable")

    a = Agent(model=StubModel(), tools=[], compactor=_HangingCompactor())
    drive_task = asyncio.create_task(a.serve_forever())
    recompact_task = asyncio.create_task(a.recompact("retry"))
    try:
        await asyncio.wait_for(started.wait(), timeout=1.0)
        await a.clear()
        await asyncio.wait_for(recompact_task, timeout=1.0)
    finally:
        a.shutdown()
        with contextlib.suppress(asyncio.CancelledError):
            await drive_task


@pytest.mark.asyncio
async def test_recompact_awaits_compact_complete_event() -> None:
    @dataclass(slots=True, kw_only=True)
    class _StubCompactor:
        async def should_compact(
            self,
            input_tokens: int,
            max_request_tokens: int,
            max_response_tokens: int = 0,
        ) -> bool:
            del input_tokens, max_request_tokens, max_response_tokens
            return False

        async def compact(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[types.history.HistoryEntry],
            model: object,
            mint_ref: Callable[[], TapeRef],
            custom_instructions: str | None = None,
        ) -> ContextSplice:
            del context, model
            del custom_instructions
            return _summary_override(
                [types.history.UserMessage(text="[recompacted]")], mint_ref, tape=tape
            )

        def maintain(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[types.history.HistoryEntry],
            tools: object,
            mint_ref: Callable[[], TapeRef],
        ) -> tuple[ContextSplice, ...]:
            del tape, context, tools, mint_ref
            return ()

    a = Agent(model=StubModel(), tools=[], compactor=_StubCompactor())
    drive_task = asyncio.create_task(a.serve_forever())
    try:
        await a.recompact("instr")
    finally:
        a.shutdown()
        with contextlib.suppress(asyncio.CancelledError):
            await drive_task

    assert any(
        isinstance(e, types.history.UserMessage) and e.text == "[recompacted]"
        for e in a.history
    )


@pytest.mark.asyncio
async def test_clear_pushes_clear_event_and_resets_file_tracking() -> None:
    a = _build_agent()
    # Stage a tracked file.
    a.tool_state.mark_read("/tmp/x.txt")  # noqa: S108 -- placeholder
    assert a.tool_state.has_been_read("/tmp/x.txt")  # noqa: S108

    await a.clear()
    assert not a.tool_state.has_been_read("/tmp/x.txt")  # noqa: S108

    # Clear event is sitting on the inbox.
    items = await a.runtime.inbox.drain()
    assert any(isinstance(i, types.runtime.Clear) for i in items)


def test_build_system_appends_tool_contributions() -> None:
    @dataclass(slots=True, kw_only=True)
    class _PromptingTool(StubTool):
        @override
        def prompt(self) -> str:
            return "(extra-tool-prompt)"

    a = _build_agent(system="root", tools=[_PromptingTool()])
    out = a.system_prompt()
    assert "root" in out
    assert "(extra-tool-prompt)" in out


@pytest.mark.asyncio
async def test_activity_active_spans_tool_execution() -> None:
    """``activity.active`` stays True from first types.runtime.ModelCallStarted through types.runtime.ModelIdle.

    Before fix: ``types.runtime.ModelResponseComplete`` always cleared ``active``,
    so the status-pane spinner went dark during tool execution. The
    user sees a long-running Bash with no visible progress indicator.

    After fix: when ``types.runtime.ModelResponseComplete`` carries ``tool_calls``,
    ``active`` stays True so the spinner keeps ticking through the
    cohort window. ``active`` only clears on true terminal events
    (``types.runtime.ModelIdle`` / cancel / error).
    """
    a = _build_agent()
    tc = types.history.ToolCall(id="c1", name="Echo", args={})
    msg_with_tools = types.history.AssistantMessage(text="", tool_calls=(tc,))

    a.publish(types.runtime.ModelCallStarted())
    assert a.activity.active is True

    a.publish(types.runtime.ModelResponseComplete(message=msg_with_tools))
    assert a.activity.active is True, (
        "spinner should keep ticking while tools run; tool_calls in the "
        "response mean the cohort is about to fire"
    )

    # Tool result arrives.
    a.publish(types.history.ToolResult(call_id="c1", content="ok"))
    assert a.activity.active is True

    # Round 2 model call fires.
    a.publish(types.runtime.ModelCallStarted())
    assert a.activity.active is True

    # Round 2 has no tool calls; model truly idles.
    a.publish(
        types.runtime.ModelResponseComplete(
            message=types.history.AssistantMessage(text="done")
        )
    )
    a.publish(types.runtime.ModelIdle())
    assert a.activity.active is False, "ModelIdle marks the end of the round chain"


@pytest.mark.asyncio
async def test_activity_active_clears_on_model_response_error() -> None:
    """``types.runtime.ModelResponseError`` must clear ``activity.active`` (stop spinner).

    Bug repro: model call fails (e.g. ``AuthRefreshError`` on expired
    OAuth). The runtime catches the exception and pushes
    ``types.runtime.ModelResponseError`` to the inbox. Before fix: ``_record_activity``
    only resets ``active`` on ``types.runtime.ModelResponseComplete`` / ``types.runtime.ModelIdle``
    / ``ModelResponseCancelled``, so the status-pane spinner keeps
    ticking forever even though no model call is in flight. After fix:
    ``types.runtime.ModelResponseError`` joins the terminal-event set and clears
    ``active``.
    """
    a = _build_agent()
    a.publish(types.runtime.ModelCallStarted())
    assert a.activity.active is True

    a.publish(types.runtime.ModelResponseError(RuntimeError("boom")))
    assert a.activity.active is False, (
        "ModelResponseError is terminal -- spinner must stop"
    )
    assert a.activity.current_call_start == 0.0, (
        f"current_call_start must reset; got {a.activity.current_call_start}"
    )


@pytest.mark.asyncio
async def test_activity_current_compact_start_resets_on_compact_complete() -> None:
    """``CompactComplete`` clears ``activity.current_compact_start``."""
    a = _build_agent()
    a.publish(types.runtime.CompactStarted())
    assert a.activity.current_compact_start > 0.0
    a.publish(types.runtime.CompactComplete(records=()))
    assert a.activity.current_compact_start == 0.0


@pytest.mark.asyncio
async def test_streaming_chars_recorded_in_activity() -> None:
    """``_track_activity`` accumulates streamed chars on types.runtime.ModelResponsePartial."""
    a = _build_agent()
    # Push a partial event through the publish path.
    a.publish(types.runtime.ModelResponsePartial(text="abc"))
    # The handler only acts when ``active`` is True; bracket via
    # types.runtime.ModelCallStarted first.
    a.publish(types.runtime.ModelCallStarted())
    a.publish(types.runtime.ModelResponsePartial(text="defg"))
    assert a.activity.live_response_chars == 4


def test_tool_registry_recorded_on_response_with_tool_calls() -> None:
    """``_track_tool_registry`` records cohort id → tool name and bumps rounds."""
    a = _build_agent()
    tc = types.history.ToolCall(id="c1", name="Echo", args={})
    msg = types.history.AssistantMessage(text="", tool_calls=(tc,))
    a.publish(types.runtime.ModelResponseComplete(message=msg))
    assert a._tool_registry["c1"][0] == "Echo"
    assert a.activity.num_tool_call_rounds == 1


def test_enforce_caps_pushes_error_when_limit_reached() -> None:
    """``_enforce_caps`` posts a types.runtime.ModelResponseError when rounds cap is hit."""
    a = Agent(model=StubModel(), tools=[], max_tool_call_rounds=1)
    tc = types.history.ToolCall(id="c1", name="Echo", args={})
    msg = types.history.AssistantMessage(text="", tool_calls=(tc,))
    a.publish(types.runtime.ModelResponseComplete(message=msg))

    # Round count is now 1 == cap; the next observation triggers the
    # error push.
    a.publish(types.runtime.ModelResponseComplete(message=msg))
    items = asyncio.new_event_loop().run_until_complete(a.runtime.inbox.drain())
    assert any(isinstance(i, types.runtime.ModelResponseError) for i in items)


@pytest.mark.asyncio
async def test_compact_now_no_compactor_is_noop() -> None:
    a = _build_agent()
    a.runtime.append_history(types.history.UserMessage(text="x"))
    await a.compact_now()
    # History untouched: no compactor wired.
    assert len(a.runtime.context().messages) == 1


@pytest.mark.asyncio
async def test_compact_now_replaces_history_in_place() -> None:
    @dataclass(slots=True, kw_only=True)
    class _ReplaceCompactor:
        async def should_compact(
            self,
            input_tokens: int,
            max_request_tokens: int,
            max_response_tokens: int = 0,
        ) -> bool:
            del input_tokens, max_request_tokens, max_response_tokens
            return False

        async def compact(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[types.history.HistoryEntry],
            model: object,
            mint_ref: Callable[[], TapeRef],
            custom_instructions: str | None = None,
        ) -> ContextSplice:
            del context, model
            del custom_instructions
            return _summary_override(
                [types.history.UserMessage(text="[summary]")], mint_ref, tape=tape
            )

        def maintain(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[types.history.HistoryEntry],
            tools: object,
            mint_ref: Callable[[], TapeRef],
        ) -> tuple[ContextSplice, ...]:
            del tape, context, tools, mint_ref
            return ()

    a = Agent(model=StubModel(), tools=[], compactor=_ReplaceCompactor())
    a.runtime.append_history(types.history.UserMessage(text="old"))
    await a.compact_now()
    assert len(a.runtime.context().messages) == 1
    entry = a.runtime.context().messages[0]
    assert isinstance(entry, types.history.UserMessage)
    assert entry.text == "[summary]"


@pytest.mark.asyncio
async def test_compact_now_absorbs_detached_splice_landing_during_compact() -> None:
    """The compact barrier covers detached splices appended during compact."""
    compact_started = asyncio.Event()
    release_compact = asyncio.Event()

    @dataclass(slots=True, kw_only=True)
    class _BlockingCompactor:
        async def should_compact(
            self,
            input_tokens: int,
            max_request_tokens: int,
            max_response_tokens: int = 0,
        ) -> bool:
            del input_tokens, max_request_tokens, max_response_tokens
            return False

        async def compact(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[types.history.HistoryEntry],
            model: object,
            mint_ref: Callable[[], TapeRef],
            custom_instructions: str | None = None,
        ) -> ContextSplice:
            del context, model, custom_instructions
            mask: tuple[tuple[TapeRef, TapeRef], ...] = (
                ((tape[0].ref, tape[-1].ref),) if tape else ()
            )
            compact_started.set()
            await release_compact.wait()
            return ContextSplice(
                ref=mint_ref(),
                mask=mask,
                insert_after=None,
                payload=(types.history.UserMessage(text="[summary]"),),
                strategy="summary",
            )

        def maintain(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[types.history.HistoryEntry],
            tools: object,
            mint_ref: Callable[[], TapeRef],
        ) -> tuple[ContextSplice, ...]:
            del tape, context, tools, mint_ref
            return ()

    a = Agent(model=StubModel(), tools=[], compactor=_BlockingCompactor())
    a.runtime.append_history(types.history.UserMessage(text="please run a tool"))
    tc = types.history.ToolCall(id="tc-1", name="echo", args={})
    a.runtime.append_history(types.history.AssistantMessage(text="", tool_calls=(tc,)))
    a.runtime.append_history(
        types.history.ToolResult(call_id="tc-1", content="[Running in background]"),
    )
    a.runtime.append_history(types.history.UserMessage(text="[worker is idle] ping"))

    compact_task = asyncio.create_task(a.compact_now())
    try:
        await asyncio.wait_for(compact_started.wait(), timeout=1.0)
        spliced = a.runtime._splice_detached_result("tc-1", "real output", False)
        assert spliced is not None
        release_compact.set()
        await asyncio.wait_for(compact_task, timeout=1.0)
    finally:
        if not compact_task.done():
            compact_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await compact_task

    messages = a.runtime.context().messages
    assert len(messages) == 1
    assert isinstance(messages[0], types.history.UserMessage)
    assert messages[0].text == "[summary]"


@pytest.mark.asyncio
async def test_compact_now_clears_tool_recall(tmp_path: Path) -> None:
    """After the barrier summarized prior context away, per-tool recall
    caches that assume the original tool results are still visible must
    be cleared. Otherwise ``Read.check_unchanged`` returns ``"[unchanged]"``
    stubs pointing at content the model can no longer see, and
    ``Skill.run`` short-circuits without re-emitting the body.
    """

    @dataclass(slots=True, kw_only=True)
    class _NoopCompactor:
        async def should_compact(
            self,
            input_tokens: int,
            max_request_tokens: int,
            max_response_tokens: int = 0,
        ) -> bool:
            del input_tokens, max_request_tokens, max_response_tokens
            return False

        async def compact(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[types.history.HistoryEntry],
            model: object,
            mint_ref: Callable[[], TapeRef],
            custom_instructions: str | None = None,
        ) -> ContextSplice:
            del context, model, custom_instructions
            return _summary_override(
                [types.history.UserMessage(text="[summary]")], mint_ref, tape=tape
            )

    a = Agent(model=StubModel(), tools=[], compactor=_NoopCompactor())
    f = tmp_path / "foo.py"
    f.write_text("x")
    a.tool_state.mark_read(str(f), content="x")
    a.tool_state.invoked_skills.add("alpha")
    a.tool_state.invoked_skills.add("beta")
    assert a.tool_state.read_cache, "read_cache should be populated pre-compact"
    assert a.tool_state.invoked_skills == {"alpha", "beta"}

    a.runtime.append_history(types.history.UserMessage(text="old"))
    await a.compact_now()

    assert a.tool_state.read_cache == {}, (
        "compact_now must clear read_cache so Read.check_unchanged stops"
        " returning [unchanged] stubs for content the model can no longer see"
    )
    assert a.tool_state.invoked_skills == set(), (
        "compact_now must clear invoked_skills so Skill.run re-emits the body"
        " on next invocation, since the prior <skill> block is no longer in"
        " context after the barrier"
    )


@pytest.mark.asyncio
async def test_compact_now_returns_true_on_success() -> None:
    """``compact_now`` reports whether compaction made progress.

    Recovery logic in ``_AgentModel.stream`` needs to distinguish a
    successful compaction (history actually shrank; retry the model)
    from a swallowed failure (compactor raised; history unchanged or
    longer). Returning bool gives that signal without the caller
    having to grovel through history mutation.
    """

    @dataclass(slots=True, kw_only=True)
    class _OkCompactor:
        async def should_compact(
            self,
            input_tokens: int,
            max_request_tokens: int,
            max_response_tokens: int = 0,
        ) -> bool:
            del input_tokens, max_request_tokens, max_response_tokens
            return False

        async def compact(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[types.history.HistoryEntry],
            model: object,
            mint_ref: Callable[[], TapeRef],
            custom_instructions: str | None = None,
        ) -> ContextSplice:
            del context, model
            del custom_instructions
            return _summary_override(
                [types.history.UserMessage(text="[summary]")], mint_ref, tape=tape
            )

        def maintain(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[types.history.HistoryEntry],
            tools: object,
            mint_ref: Callable[[], TapeRef],
        ) -> tuple[ContextSplice, ...]:
            del tape, context, tools, mint_ref
            return ()

    a = Agent(model=StubModel(), tools=[], compactor=_OkCompactor())
    a.runtime.append_history(types.history.UserMessage(text="old"))
    ok = await a.compact_now()
    assert ok is True


@pytest.mark.asyncio
async def test_compact_now_returns_false_on_compactor_failure() -> None:
    """``compact_now`` returns False when the inner compactor raises.

    Without this signal, ``_AgentModel.stream``'s overflow recovery
    would re-fire the model with stale (or slightly longer) history
    and burn ``MAX_OVERFLOW_RECOVERY`` retries on identical failures
    -- exactly the BUGS34 regression that produced cryptic ``RuntimeError:
    context overflow recovery failed after 3 compactions`` messages.
    """

    @dataclass(slots=True, kw_only=True)
    class _BrokenCompactor:
        async def should_compact(
            self,
            input_tokens: int,
            max_request_tokens: int,
            max_response_tokens: int = 0,
        ) -> bool:
            del input_tokens, max_request_tokens, max_response_tokens
            return False

        async def compact(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[types.history.HistoryEntry],
            model: object,
            mint_ref: Callable[[], TapeRef],
            custom_instructions: str | None = None,
        ) -> ContextSplice:
            del tape, context, model, mint_ref
            del custom_instructions
            raise RuntimeError("compaction blew up")

        def maintain(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[types.history.HistoryEntry],
            tools: object,
            mint_ref: Callable[[], TapeRef],
        ) -> tuple[ContextSplice, ...]:
            del tape, context, tools, mint_ref
            return ()

    a = Agent(model=StubModel(), tools=[], compactor=_BrokenCompactor())
    a.runtime.append_history(types.history.UserMessage(text="x"))
    ok = await a.compact_now()
    assert ok is False


@pytest.mark.asyncio
async def test_compact_now_returns_true_when_no_compactor_wired() -> None:
    """No compactor = nothing to do; report success so callers don't loop.

    Returning ``False`` from a no-op would mislead ``_AgentModel.stream``
    into raising ``ContextOverflowError`` for agents that never asked
    to manage their own context (subscription providers, tests). The
    "no compactor" path is intentional configuration, not a failure.
    """
    a = _build_agent()
    a.runtime.append_history(types.history.UserMessage(text="x"))
    ok = await a.compact_now()
    assert ok is True


@pytest.mark.asyncio
async def test_compact_if_needed_returns_true_when_no_compactor_wired() -> None:
    """``compact_if_needed`` matches ``compact_now``'s no-compactor semantics.

    Before unification, ``compact_if_needed`` returned ``None`` while
    ``compact_now`` returned ``True`` for the same no-compactor path.
    The divergence made every caller branch on the bool/None distinction.
    Unifying on bool gives one type signature, one contract: ``True``
    means "nothing further to do (success or no-op)", ``False`` means
    "tried to compact and the inner compactor raised".
    """
    a = _build_agent()
    history: list[types.history.HistoryEntry] = [types.history.UserMessage(text="x")]
    progressed = await a.compact_if_needed(history, a.model)
    assert progressed is True


@pytest.mark.asyncio
async def test_compact_if_needed_returns_true_when_should_compact_false() -> None:
    """When the compactor decides headroom is fine, ``compact_if_needed`` succeeds.

    Pre-unification, the no-trigger path returned ``None`` implicitly.
    Same contract as the no-compactor path: ``True`` = "no further work
    needed", with the gate result rolled up into the bool.
    """

    @dataclass(slots=True, kw_only=True)
    class _NeverCompactor:
        async def should_compact(
            self,
            input_tokens: int,
            max_request_tokens: int,
            max_response_tokens: int = 0,
        ) -> bool:
            del input_tokens, max_request_tokens, max_response_tokens
            return False

        async def compact(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[types.history.HistoryEntry],
            model: object,
            mint_ref: Callable[[], TapeRef],
            custom_instructions: str | None = None,
        ) -> ContextSplice:
            del tape, context, model, mint_ref
            del custom_instructions
            raise AssertionError("compact must not run when should_compact False")

        def maintain(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[types.history.HistoryEntry],
            tools: object,
            mint_ref: Callable[[], TapeRef],
        ) -> tuple[ContextSplice, ...]:
            del tape, context, tools, mint_ref
            return ()

    a = Agent(model=StubModel(), tools=[], compactor=_NeverCompactor())
    history: list[types.history.HistoryEntry] = [types.history.UserMessage(text="x")]
    progressed = await a.compact_if_needed(history, a.model)
    assert progressed is True


@pytest.mark.asyncio
async def test_compact_if_needed_returns_false_on_compaction_failure() -> None:
    """When ``compact_now`` reports False, ``compact_if_needed`` propagates it.

    The bool flows up so future callers (proactive sites beyond the
    ``_AgentModel.stream`` overflow loop) can short-circuit on the same
    "history did not shrink" signal that bool was introduced for.
    """

    @dataclass(slots=True, kw_only=True)
    class _CompactBrokenButGatedTrue:
        async def should_compact(
            self,
            input_tokens: int,
            max_request_tokens: int,
            max_response_tokens: int = 0,
        ) -> bool:
            del input_tokens, max_request_tokens, max_response_tokens
            return True

        async def compact(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[types.history.HistoryEntry],
            model: object,
            mint_ref: Callable[[], TapeRef],
            custom_instructions: str | None = None,
        ) -> ContextSplice:
            del tape, context, model, mint_ref
            del custom_instructions
            raise RuntimeError("compaction blew up")

        def maintain(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[types.history.HistoryEntry],
            tools: object,
            mint_ref: Callable[[], TapeRef],
        ) -> tuple[ContextSplice, ...]:
            del tape, context, tools, mint_ref
            return ()

    a = Agent(model=StubModel(), tools=[], compactor=_CompactBrokenButGatedTrue())
    history: list[types.history.HistoryEntry] = [types.history.UserMessage(text="x")]
    progressed = await a.compact_if_needed(history, a.model)
    assert progressed is False


@pytest.mark.asyncio
async def test_circuit_breaker_short_circuits_after_consecutive_failures() -> None:
    """``compact_if_needed`` returns False without invoking compactor after N failures."""

    @dataclass(slots=True, kw_only=True)
    class _AlwaysBroken:
        call_count: int = 0

        async def should_compact(
            self,
            input_tokens: int,
            max_request_tokens: int,
            max_response_tokens: int = 0,
        ) -> bool:
            del input_tokens, max_request_tokens, max_response_tokens
            return True

        async def compact(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[types.history.HistoryEntry],
            model: object,
            mint_ref: Callable[[], TapeRef],
            custom_instructions: str | None = None,
        ) -> ContextSplice:
            del tape, context, model, mint_ref, custom_instructions
            self.call_count += 1
            raise RuntimeError("always broken")

        def maintain(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[types.history.HistoryEntry],
            tools: object,
            mint_ref: Callable[[], TapeRef],
        ) -> tuple[ContextSplice, ...]:
            del tape, context, tools, mint_ref
            return ()

    compactor = _AlwaysBroken()
    a = Agent(model=StubModel(), tools=[], compactor=compactor)
    history: list[types.history.HistoryEntry] = [types.history.UserMessage(text="x")]

    # First 3 calls invoke the compactor and fail.
    for _ in range(3):
        progressed = await a.compact_if_needed(history, a.model)
        assert progressed is False
    assert compactor.call_count == 3
    assert a.compaction_state.compact_failures == 3

    # 4th call hits the circuit breaker -- compactor is NOT invoked.
    progressed = await a.compact_if_needed(history, a.model)
    assert progressed is False
    assert compactor.call_count == 3, (
        "circuit breaker should have short-circuited the 4th call"
    )


@pytest.mark.asyncio
async def test_circuit_breaker_resets_on_successful_compaction() -> None:
    """A successful compaction zeroes ``compact_failures``."""

    @dataclass(slots=True, kw_only=True)
    class _SuccessfulCompactor:
        async def should_compact(
            self,
            input_tokens: int,
            max_request_tokens: int,
            max_response_tokens: int = 0,
        ) -> bool:
            del input_tokens, max_request_tokens, max_response_tokens
            return True

        async def compact(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[types.history.HistoryEntry],
            model: object,
            mint_ref: Callable[[], TapeRef],
            custom_instructions: str | None = None,
        ) -> ContextSplice:
            del context, model, custom_instructions
            return _summary_override(
                [types.history.UserMessage(text="[summary]")], mint_ref, tape=tape
            )

        def maintain(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[types.history.HistoryEntry],
            tools: object,
            mint_ref: Callable[[], TapeRef],
        ) -> tuple[ContextSplice, ...]:
            del tape, context, tools, mint_ref
            return ()

    a = Agent(model=StubModel(), tools=[], compactor=_SuccessfulCompactor())
    # Pre-populate failure count -- simulating prior auto-failures.
    a.compaction_state.compact_failures = 2
    history: list[types.history.HistoryEntry] = [types.history.UserMessage(text="x")]
    progressed = await a.compact_if_needed(history, a.model)
    assert progressed is True
    assert a.compaction_state.compact_failures == 0


@pytest.mark.asyncio
async def test_compact_now_failure_appends_error_user_message() -> None:
    @dataclass(slots=True, kw_only=True)
    class _BrokenCompactor:
        async def should_compact(
            self,
            input_tokens: int,
            max_request_tokens: int,
            max_response_tokens: int = 0,
        ) -> bool:
            del input_tokens, max_request_tokens, max_response_tokens
            return False

        async def compact(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[types.history.HistoryEntry],
            model: object,
            mint_ref: Callable[[], TapeRef],
            custom_instructions: str | None = None,
        ) -> ContextSplice:
            del tape, context, model, mint_ref
            del custom_instructions
            raise RuntimeError("compaction failed")

        def maintain(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[types.history.HistoryEntry],
            tools: object,
            mint_ref: Callable[[], TapeRef],
        ) -> tuple[ContextSplice, ...]:
            del tape, context, tools, mint_ref
            return ()

    a = Agent(model=StubModel(), tools=[], compactor=_BrokenCompactor())
    a.runtime.append_history(types.history.UserMessage(text="x"))
    await a.compact_now()
    err = [
        e
        for e in a.runtime.context().messages
        if isinstance(e, types.history.UserMessage) and "[Compaction error:" in e.text
    ]
    assert len(err) == 1


@dataclass(slots=True, kw_only=True)
class _OverflowModel:
    """Model that raises types.exceptions.PromptTooLongError on the first N calls."""

    model_id: str = "ovf"
    max_request_tokens: int = 100_000
    max_response_tokens: int = 1_024
    supports_streaming: bool = True
    supports_thinking: bool = False
    supports_effort: bool = False
    supports_cache_control: bool = False
    valid_service_tiers: tuple[str, ...] = ()
    supports_context_management: bool = False
    supports_persistent_retry: bool = False
    supports_account_auth: bool = False
    max_image_dim: int = 8_000
    max_image_bytes: int = 5 * 1024 * 1024
    overflow_count: int = 0
    call_index: int = 0

    @property
    def pricing(self) -> types.model.Pricing:
        return types.model.Pricing()

    def approx_text_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    def approx_image_tokens(self, data: bytes) -> int:
        del data
        return 256

    def approx_request_tokens(self, request: types.model.ModelRequest) -> int:
        return token_count.approx_request_tokens(request, self)

    async def actual_text_tokens(self, text: str) -> int:
        return self.approx_text_tokens(text)

    async def actual_image_tokens(self, data: bytes) -> int:
        return self.approx_image_tokens(data)

    async def actual_request_tokens(self, request: types.model.ModelRequest) -> int:
        return self.approx_request_tokens(request)

    def is_context_overflow(self, error: Exception) -> bool:
        return isinstance(error, types.exceptions.PromptTooLongError)

    def is_retryable_provider_error(self, error: Exception) -> bool:
        del error
        return False

    async def buffer(
        self, request: types.model.ModelRequest
    ) -> types.model.ModelResponse:
        return await self.stream(request)

    async def stream(
        self,
        request: types.model.ModelRequest,
        on_text: object = None,
        on_thinking: object = None,
    ) -> types.model.ModelResponse:
        del request, on_text, on_thinking
        idx = self.call_index
        self.call_index += 1
        if idx < self.overflow_count:
            raise types.exceptions.PromptTooLongError("too long")
        return types.model.ModelResponse(
            message=types.history.AssistantMessage(text="recovered")
        )


@dataclass(slots=True, kw_only=True)
class _RawOverflowModel:
    """Model that raises a non-types.exceptions.PromptTooLongError but classifies it as overflow.

    Mirrors the production failure where ``anthropic.APIStatusError``
    propagated up un-normalized: the recovery loop's catch must rely
    on ``is_context_overflow``, not on ``isinstance(exc,
    types.exceptions.PromptTooLongError)``, or compaction never engages.
    """

    model_id: str = "raw"
    max_request_tokens: int = 100_000
    max_response_tokens: int = 1_024
    supports_streaming: bool = True
    supports_thinking: bool = False
    supports_effort: bool = False
    supports_cache_control: bool = False
    valid_service_tiers: tuple[str, ...] = ()
    supports_context_management: bool = False
    supports_persistent_retry: bool = False
    supports_account_auth: bool = False
    max_image_dim: int = 8_000
    max_image_bytes: int = 5 * 1024 * 1024
    overflow_count: int = 0
    call_index: int = 0

    @property
    def pricing(self) -> types.model.Pricing:
        return types.model.Pricing()

    def approx_text_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    def approx_image_tokens(self, data: bytes) -> int:
        del data
        return 256

    def approx_request_tokens(self, request: types.model.ModelRequest) -> int:
        return token_count.approx_request_tokens(request, self)

    async def actual_text_tokens(self, text: str) -> int:
        return self.approx_text_tokens(text)

    async def actual_image_tokens(self, data: bytes) -> int:
        return self.approx_image_tokens(data)

    async def actual_request_tokens(self, request: types.model.ModelRequest) -> int:
        return self.approx_request_tokens(request)

    def is_context_overflow(self, error: Exception) -> bool:
        return isinstance(error, RuntimeError) and "context window" in str(error)

    def is_retryable_provider_error(self, error: Exception) -> bool:
        del error
        return False

    async def buffer(
        self, request: types.model.ModelRequest
    ) -> types.model.ModelResponse:
        return await self.stream(request)

    async def stream(
        self,
        request: types.model.ModelRequest,
        on_text: object = None,
        on_thinking: object = None,
    ) -> types.model.ModelResponse:
        del request, on_text, on_thinking
        idx = self.call_index
        self.call_index += 1
        if idx < self.overflow_count:
            raise RuntimeError("Request size exceeds model context window")
        return types.model.ModelResponse(
            message=types.history.AssistantMessage(text="recovered")
        )


@pytest.mark.asyncio
async def test_agent_model_overflow_triggers_compact_now() -> None:
    """One overflow followed by success: compact_now runs once, response returned."""
    compact_calls: list[int] = []

    @dataclass(slots=True, kw_only=True)
    class _CountingCompactor:
        async def should_compact(
            self,
            input_tokens: int,
            max_request_tokens: int,
            max_response_tokens: int = 0,
        ) -> bool:
            del input_tokens, max_request_tokens, max_response_tokens
            return False

        async def compact(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[types.history.HistoryEntry],
            model: object,
            mint_ref: Callable[[], TapeRef],
            custom_instructions: str | None = None,
        ) -> ContextSplice:
            del context, model
            del custom_instructions
            compact_calls.append(1)
            return _summary_override(
                [types.history.UserMessage(text="[compact]")], mint_ref, tape=tape
            )

        def maintain(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[types.history.HistoryEntry],
            tools: object,
            mint_ref: Callable[[], TapeRef],
        ) -> tuple[ContextSplice, ...]:
            del tape, context, tools, mint_ref
            return ()

    model = _OverflowModel(overflow_count=1)
    a = Agent(model=model, tools=[], compactor=_CountingCompactor())
    async for _ in a.run(types.history.UserMessage(text="hi")):
        pass
    assert len(compact_calls) == 1
    # _OverflowModel emitted "recovered" on the second call.
    assert any(
        isinstance(e, types.history.AssistantMessage) and e.text == "recovered"
        for e in a.history
    )


@pytest.mark.asyncio
async def test_agent_model_proactive_compaction_runs_before_stream() -> None:
    """``should_compact`` -> True forces compaction BEFORE the provider call.

    The reactive path (overflow recovery on 400) is necessary but not
    sufficient: some providers happily accept oversized prompts up to
    a hard ceiling and only 400 at that ceiling, by which point the
    cost is already paid. The proactive path consults
    ``Compactor.should_compact`` ahead of each stream and runs
    ``compact_now`` when it returns True -- so the headroom buffer
    actually buys headroom.

    Failure mode without the wiring: a session at ~1.07M tokens with
    zero ``Compact`` events ever fired (see
    ``~/.sagent/.../session.jsonl`` from the wedged run).
    """
    order: list[str] = []

    @dataclass(slots=True, kw_only=True)
    class _OneShotCompactor:
        triggered: bool = False

        async def should_compact(
            self,
            input_tokens: int,
            max_request_tokens: int,
            max_response_tokens: int = 0,
        ) -> bool:
            del input_tokens, max_request_tokens, max_response_tokens
            if self.triggered:
                return False
            self.triggered = True
            return True

        async def compact(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[types.history.HistoryEntry],
            model: object,
            mint_ref: Callable[[], TapeRef],
            custom_instructions: str | None = None,
        ) -> ContextSplice:
            del context, model
            del custom_instructions
            order.append("compact")
            return _summary_override(
                [types.history.UserMessage(text="[compact]")], mint_ref, tape=tape
            )

        def maintain(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[types.history.HistoryEntry],
            tools: object,
            mint_ref: Callable[[], TapeRef],
        ) -> tuple[ContextSplice, ...]:
            del tape, context, tools, mint_ref
            return ()

    @dataclass(slots=True, kw_only=True)
    class _RecordingModel:
        order_log: list[str]
        model_id: str = "rec"
        max_request_tokens: int = 100_000
        max_response_tokens: int = 1_024
        supports_streaming: bool = True
        supports_thinking: bool = False
        supports_effort: bool = False
        supports_cache_control: bool = False
        valid_service_tiers: tuple[str, ...] = ()
        supports_context_management: bool = False
        supports_persistent_retry: bool = False
        supports_account_auth: bool = False
        max_image_dim: int = 8_000
        max_image_bytes: int = 5 * 1024 * 1024

        @property
        def pricing(self) -> types.model.Pricing:
            return types.model.Pricing()

        def approx_text_tokens(self, text: str) -> int:
            return max(1, len(text) // 4)

        def approx_image_tokens(self, data: bytes) -> int:
            del data
            return 256

        def approx_request_tokens(self, request: types.model.ModelRequest) -> int:
            return token_count.approx_request_tokens(request, self)

        async def actual_text_tokens(self, text: str) -> int:
            return self.approx_text_tokens(text)

        async def actual_image_tokens(self, data: bytes) -> int:
            return self.approx_image_tokens(data)

        async def actual_request_tokens(self, request: types.model.ModelRequest) -> int:
            return self.approx_request_tokens(request)

        def is_context_overflow(self, error: Exception) -> bool:
            del error
            return False

        def is_retryable_provider_error(self, error: Exception) -> bool:
            del error
            return False

        async def buffer(
            self, request: types.model.ModelRequest
        ) -> types.model.ModelResponse:
            return await self.stream(request)

        async def stream(
            self,
            request: types.model.ModelRequest,
            on_text: object = None,
            on_thinking: object = None,
        ) -> types.model.ModelResponse:
            del request, on_text, on_thinking
            self.order_log.append("stream")
            return types.model.ModelResponse(
                message=types.history.AssistantMessage(text="ok"),
            )

    model = _RecordingModel(order_log=order)
    a = Agent(model=model, tools=[], compactor=_OneShotCompactor())
    async for _ in a.run(types.history.UserMessage(text="hi")):
        pass
    assert order == ["compact", "stream"], order


@pytest.mark.asyncio
async def test_compact_now_publishes_compaction_progress_events() -> None:
    """Direct compaction path emits renderable observer events."""

    @dataclass(slots=True, kw_only=True)
    class _OkCompactor:
        async def should_compact(
            self,
            input_tokens: int,
            max_request_tokens: int,
            max_response_tokens: int = 0,
        ) -> bool:
            del input_tokens, max_request_tokens, max_response_tokens
            return True

        async def compact(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[types.history.HistoryEntry],
            model: object,
            mint_ref: Callable[[], TapeRef],
            custom_instructions: str | None = None,
        ) -> ContextSplice:
            del context, model
            del custom_instructions
            return _summary_override(
                [types.history.UserMessage(text="[compact]")], mint_ref, tape=tape
            )

        def maintain(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[types.history.HistoryEntry],
            tools: object,
            mint_ref: Callable[[], TapeRef],
        ) -> tuple[ContextSplice, ...]:
            del tape, context, tools, mint_ref
            return ()

    events: list[types.runtime.RuntimeEvent] = []
    a = Agent(model=StubModel(), tools=[], compactor=_OkCompactor())
    a.runtime.append_history(types.history.UserMessage(text="hi"))
    a.runtime.observers.append(events.append)

    assert await a.compact_now() is True

    assert [type(event) for event in events] == [
        types.runtime.CompactStarted,
        types.runtime.CompactComplete,
    ]
    complete = events[-1]
    assert isinstance(complete, types.runtime.CompactComplete)
    assert len(complete.records) == 1
    assert a.history == list(complete.records[0].payload)
    assert a.activity.current_compact_start == 0.0


@pytest.mark.asyncio
async def test_agent_model_proactive_compaction_failure_short_circuits() -> None:
    """Failed proactive compaction must not fall through to the provider.

    The same ``compact_if_needed`` bool used by reactive overflow
    recovery applies before the first provider call too. Ignoring it
    lets an unchanged oversized history hit the provider even though
    compaction already reported that no progress was made.

    When the compactor's underlying failure is NOT context overflow
    (transport drop, auth blip, generic ``RuntimeError``), the agent
    surfaces that error verbatim instead of the polished
    ``ContextOverflowError`` -- the original session ``bc528d70`` lost
    the real ``httpx.RemoteProtocolError`` behind a misleading
    "context window exhausted" message because the proactive path
    raised the polished error unconditionally.
    """

    @dataclass(slots=True, kw_only=True)
    class _BrokenProactiveCompactor:
        calls: int = 0

        async def should_compact(
            self,
            input_tokens: int,
            max_request_tokens: int,
            max_response_tokens: int = 0,
        ) -> bool:
            del input_tokens, max_request_tokens, max_response_tokens
            return True

        async def compact(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[types.history.HistoryEntry],
            model: object,
            mint_ref: Callable[[], TapeRef],
            custom_instructions: str | None = None,
        ) -> ContextSplice:
            del tape, context, model, mint_ref
            del custom_instructions
            self.calls += 1
            raise RuntimeError("compaction disconnected")

        def maintain(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[types.history.HistoryEntry],
            tools: object,
            mint_ref: Callable[[], TapeRef],
        ) -> tuple[ContextSplice, ...]:
            del tape, context, tools, mint_ref
            return ()

    model = StubModel()
    compactor = _BrokenProactiveCompactor()
    a = Agent(model=model, tools=[], compactor=compactor)

    with pytest.raises(RuntimeError, match="compaction disconnected"):
        await a._agent_model.stream(
            history=[types.history.UserMessage(text="hi")],
            system="",
            tools=[],
            on_text=lambda _t: None,
            on_thinking=lambda _t: None,
        )

    assert compactor.calls == 1
    assert model.received == []


@pytest.mark.asyncio
async def test_agent_model_proactive_compaction_overflow_surfaces_polished() -> None:
    """Compaction failure that IS overflow still surfaces the polished message.

    The agent distinguishes ``compact_now`` failures by whether the
    underlying error is classified as context overflow by the active
    model. When it is (``PromptTooLongError``, provider-specific
    overflow exceptions), the polished ``/clear`` / ``/compact`` /
    ``/model`` remediation message fires. When it isn't, the
    underlying error propagates verbatim (covered by
    ``test_agent_model_proactive_compaction_failure_short_circuits``).
    """

    @dataclass(slots=True, kw_only=True)
    class _OverflowingCompactor:
        async def should_compact(
            self,
            input_tokens: int,
            max_request_tokens: int,
            max_response_tokens: int = 0,
        ) -> bool:
            del input_tokens, max_request_tokens, max_response_tokens
            return True

        async def compact(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[types.history.HistoryEntry],
            model: object,
            mint_ref: Callable[[], TapeRef],
            custom_instructions: str | None = None,
        ) -> ContextSplice:
            del tape, context, model, mint_ref
            del custom_instructions
            raise types.exceptions.PromptTooLongError("compactor saw overflow")

        def maintain(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[types.history.HistoryEntry],
            tools: object,
            mint_ref: Callable[[], TapeRef],
        ) -> tuple[ContextSplice, ...]:
            del tape, context, tools, mint_ref
            return ()

    a = Agent(
        model=_OverflowModel(overflow_count=0),
        tools=[],
        compactor=_OverflowingCompactor(),
    )
    with pytest.raises(types.exceptions.ContextOverflowError) as ei:
        await a._agent_model.stream(
            history=[types.history.UserMessage(text="hi")],
            system="",
            tools=[],
            on_text=lambda _t: None,
            on_thinking=lambda _t: None,
        )
    msg = str(ei.value)
    assert "/clear" in msg
    assert "/compact" in msg
    assert isinstance(ei.value.__cause__, types.exceptions.PromptTooLongError)


@pytest.mark.asyncio
async def test_agent_model_overflow_exhausts_recovery_raises() -> None:
    """Exhaust MAX_OVERFLOW_RECOVERY: surface a user-facing context-overflow.

    The bare ``PromptTooLongError`` text ("too long") is not actionable
    for the user; the renderer prefixes it with ``ClassName:`` which is
    pure noise. After all overflow retries fail, the agent surfaces a
    :class:`types.exceptions.ContextOverflowError` (a ``UserFacingError``
    subclass) whose message names the remediations the user can take
    -- ``/clear``, ``/compact``, ``/model`` with a larger window.
    """

    @dataclass(slots=True, kw_only=True)
    class _NoOpCompactor:
        async def should_compact(
            self,
            input_tokens: int,
            max_request_tokens: int,
            max_response_tokens: int = 0,
        ) -> bool:
            del input_tokens, max_request_tokens, max_response_tokens
            return False

        async def compact(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[types.history.HistoryEntry],
            model: object,
            mint_ref: Callable[[], TapeRef],
            custom_instructions: str | None = None,
        ) -> ContextSplice:
            del context, model
            del custom_instructions
            # Returns short summary; model keeps overflowing.
            return _summary_override(
                [types.history.UserMessage(text="[compact]")], mint_ref, tape=tape
            )

        def maintain(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[types.history.HistoryEntry],
            tools: object,
            mint_ref: Callable[[], TapeRef],
        ) -> tuple[ContextSplice, ...]:
            del tape, context, tools, mint_ref
            return ()

    model = _OverflowModel(overflow_count=10)  # always overflow
    a = Agent(model=model, tools=[], compactor=_NoOpCompactor())
    with pytest.raises(types.exceptions.ContextOverflowError) as ei:
        await a._agent_model.stream(
            history=[types.history.UserMessage(text="x")],
            system="",
            tools=[],
            on_text=lambda _t: None,
            on_thinking=lambda _t: None,
        )
    msg = str(ei.value)
    assert "/clear" in msg
    assert "/compact" in msg
    assert isinstance(ei.value.__cause__, types.exceptions.PromptTooLongError)
    # The polished message should not embed the underlying exception
    # again -- ``__cause__`` already carries it. Duplicating the cause
    # in the message dilutes the actionable text with Python internals.
    assert "Underlying error" not in msg, (
        f"message should not duplicate __cause__; got {msg!r}"
    )
    assert "PromptTooLongError" not in msg, (
        f"message should not leak ClassName: prefix; got {msg!r}"
    )


@pytest.mark.asyncio
async def test_agent_model_overflow_short_circuits_on_compaction_failure() -> None:
    """When ``compact_now`` returns False, recovery stops immediately.

    BUGS34 regression coverage: previously the recovery loop called the
    model again with effectively unchanged history after every failed
    compaction, burning ``MAX_OVERFLOW_RECOVERY`` retries on identical
    400s. The agent halts after the FIRST failed compaction so the user
    gets a halt while we still know what went wrong. When the
    compactor's failure is NOT context overflow, the underlying error
    surfaces verbatim rather than being masked by a polished
    "context window exhausted" message.
    """

    @dataclass(slots=True, kw_only=True)
    class _BrokenCompactor:
        calls: int = 0

        async def should_compact(
            self,
            input_tokens: int,
            max_request_tokens: int,
            max_response_tokens: int = 0,
        ) -> bool:
            del input_tokens, max_request_tokens, max_response_tokens
            return False

        async def compact(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[types.history.HistoryEntry],
            model: object,
            mint_ref: Callable[[], TapeRef],
            custom_instructions: str | None = None,
        ) -> ContextSplice:
            del tape, context, model, mint_ref
            del custom_instructions
            self.calls += 1
            raise RuntimeError("compaction blew up")

        def maintain(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[types.history.HistoryEntry],
            tools: object,
            mint_ref: Callable[[], TapeRef],
        ) -> tuple[ContextSplice, ...]:
            del tape, context, tools, mint_ref
            return ()

    model = _OverflowModel(overflow_count=10)
    compactor = _BrokenCompactor()
    a = Agent(model=model, tools=[], compactor=compactor)
    with pytest.raises(RuntimeError, match="compaction blew up"):
        await a._agent_model.stream(
            history=[types.history.UserMessage(text="x")],
            system="",
            tools=[],
            on_text=lambda _t: None,
            on_thinking=lambda _t: None,
        )
    # Model was hit once (first attempt), compactor once (the failed
    # recovery). No further retries.
    assert model.call_index == 1, (
        f"model must not retry after failed compaction; got {model.call_index} calls"
    )
    assert compactor.calls == 1, (
        f"compactor must not retry on itself; got {compactor.calls} calls"
    )


@pytest.mark.asyncio
async def test_agent_model_overflow_recovery_via_classifier_not_isinstance() -> None:
    """Recovery engages on any exception classified as overflow, not just ``types.exceptions.PromptTooLongError``.

    When a provider's normalization slips and a raw provider exception
    propagates with the canonical ``is_context_overflow(exc)`` returning
    True, the recovery loop must still fire ``compact_now``. This is
    the bug that produced the production death-spiral: the recovery
    catch was narrowed to ``types.exceptions.PromptTooLongError`` while the classifier
    knew the exception was overflow.
    """
    compact_calls: list[int] = []

    @dataclass(slots=True, kw_only=True)
    class _CountingCompactor:
        async def should_compact(
            self,
            input_tokens: int,
            max_request_tokens: int,
            max_response_tokens: int = 0,
        ) -> bool:
            del input_tokens, max_request_tokens, max_response_tokens
            return False

        async def compact(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[types.history.HistoryEntry],
            model: object,
            mint_ref: Callable[[], TapeRef],
            custom_instructions: str | None = None,
        ) -> ContextSplice:
            del context, model
            del custom_instructions
            compact_calls.append(1)
            return _summary_override(
                [types.history.UserMessage(text="[compact]")], mint_ref, tape=tape
            )

        def maintain(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[types.history.HistoryEntry],
            tools: object,
            mint_ref: Callable[[], TapeRef],
        ) -> tuple[ContextSplice, ...]:
            del tape, context, tools, mint_ref
            return ()

    model = _RawOverflowModel(overflow_count=1)
    a = Agent(model=model, tools=[], compactor=_CountingCompactor())
    async for _ in a.run(types.history.UserMessage(text="hi")):
        pass
    assert len(compact_calls) == 1
    assert any(
        isinstance(e, types.history.AssistantMessage) and e.text == "recovered"
        for e in a.history
    )


@pytest.mark.asyncio
async def test_agent_tool_emits_label_and_delegates() -> None:
    """The wrapper publishes a types.runtime.ToolLabel and forwards to the inner tool."""
    inner = StubTool()
    a = _build_agent(tools=[inner])

    labels: list[types.runtime.ToolLabel] = []

    def _watch(ev: object) -> None:
        if isinstance(ev, types.runtime.ToolLabel):
            labels.append(ev)

    a.runtime.observers.append(_watch)

    wrapper = next(t for t in a.runtime.tools_map.values() if t.name == "Echo")
    result = await wrapper.run({"msg": "hi"})
    assert result.content == "hi"
    assert len(labels) == 1
    assert labels[0].text == "echo"


@pytest.mark.asyncio
async def test_agent_tool_invalid_input_emits_label_without_running() -> None:
    inner = StubTool(
        directive_schema=json_freeze(
            {
                "type": "object",
                "properties": {"msg": {"type": "string"}},
                "required": ["msg"],
                "additionalProperties": False,
            }
        )
    )
    a = _build_agent(tools=[inner])
    labels: list[types.runtime.ToolLabel] = []

    def _watch(ev: object) -> None:
        if isinstance(ev, types.runtime.ToolLabel):
            labels.append(ev)

    a.runtime.observers.append(_watch)
    wrapper = next(t for t in a.runtime.tools_map.values() if t.name == "Echo")

    result = await wrapper.run({})

    assert result.is_error
    assert "InputValidationError" in result.content
    assert inner.calls == []
    assert len(labels) == 1
    assert labels[0].text == "echo"


@pytest.mark.asyncio
async def test_clear_cancels_explicit_background_jobs() -> None:
    started = asyncio.Event()

    @dataclass(slots=True, kw_only=True)
    class SlowTool(StubTool):
        @override
        async def run(self, args: Mapping[str, object]) -> types.history.ToolResult:
            del args
            started.set()
            await asyncio.get_running_loop().create_future()
            return types.history.ToolResult(call_id="", content="done")

    a = _build_agent(tools=[SlowTool()])
    wrapper = next(t for t in a.runtime.tools_map.values() if t.name == "Echo")
    _ = await wrapper.run({"background": True})
    task = next(iter(a.background.values())).task
    await asyncio.wait_for(started.wait(), timeout=1.0)

    await a.clear()

    await asyncio.wait_for(task, timeout=1.0)
    assert a.background == {}


@pytest.mark.asyncio
async def test_kill_tool_cancels_explicit_background_job_by_id() -> None:
    started = asyncio.Event()

    @dataclass(slots=True, kw_only=True)
    class SlowTool(StubTool):
        @override
        async def run(self, args: Mapping[str, object]) -> types.history.ToolResult:
            del args
            started.set()
            await asyncio.get_running_loop().create_future()
            return types.history.ToolResult(call_id="", content="done")

    a = _build_agent(tools=[SlowTool()])
    wrapper = next(t for t in a.runtime.tools_map.values() if t.name == "Echo")
    _ = await wrapper.run({"background": True})
    call_id = next(iter(a.background))
    task = a.background[call_id].task
    await asyncio.wait_for(started.wait(), timeout=1.0)

    a.kill_tool(call_id)

    await asyncio.wait_for(task, timeout=1.0)
    assert a.background == {}


@pytest.mark.asyncio
async def test_kill_all_tools_cancels_explicit_background_jobs() -> None:
    started = asyncio.Event()

    @dataclass(slots=True, kw_only=True)
    class SlowTool(StubTool):
        @override
        async def run(self, args: Mapping[str, object]) -> types.history.ToolResult:
            del args
            started.set()
            await asyncio.get_running_loop().create_future()
            return types.history.ToolResult(call_id="", content="done")

    a = _build_agent(tools=[SlowTool()])
    wrapper = next(t for t in a.runtime.tools_map.values() if t.name == "Echo")
    _ = await wrapper.run({"background": True})
    task = next(iter(a.background.values())).task
    await asyncio.wait_for(started.wait(), timeout=1.0)

    a.kill_all_tools()

    await asyncio.wait_for(task, timeout=1.0)
    assert a.background == {}


@pytest.mark.asyncio
async def test_tool_call_round_cap_blocks_tool_spawn() -> None:
    started = asyncio.Event()

    @dataclass(slots=True, kw_only=True)
    class SideEffectTool(StubTool):
        @override
        async def run(self, args: Mapping[str, object]) -> types.history.ToolResult:
            del args
            started.set()
            return types.history.ToolResult(call_id="", content="side effect")

    model = StubModel(
        responses=[
            types.history.AssistantMessage(
                tool_calls=(types.history.ToolCall(id="c1", name="Echo", args={}),)
            )
        ]
    )
    a = Agent(model=model, tools=[SideEffectTool()], max_tool_call_rounds=1)
    events = [type(ev) async for ev in a.run(types.history.UserMessage(text="go"))]

    assert types.runtime.ModelResponseError in events
    assert not started.is_set()
    assert a.runtime.running_tools == {}
    assert a.runtime.cohort == set()


@pytest.mark.asyncio
async def test_clear_drops_cancelled_background_result_after_fresh_turn() -> None:
    started = asyncio.Event()
    idle = asyncio.Event()

    @dataclass(slots=True, kw_only=True)
    class SlowTool(StubTool):
        @override
        async def run(self, args: Mapping[str, object]) -> types.history.ToolResult:
            del args
            started.set()
            await asyncio.get_running_loop().create_future()
            return types.history.ToolResult(call_id="", content="done")

    a = _build_agent(
        model=StubModel(
            responses=[types.history.AssistantMessage(text="fresh response")]
        ),
        tools=[SlowTool()],
    )

    def _watch(event: types.runtime.RuntimeEvent) -> None:
        if isinstance(event, types.runtime.ModelIdle):
            idle.set()

    a.runtime.observers.append(_watch)
    wrapper = next(t for t in a.runtime.tools_map.values() if t.name == "Echo")
    _ = await wrapper.run({"background": True})
    await asyncio.wait_for(started.wait(), timeout=1.0)
    drive = asyncio.create_task(a.serve_forever())
    try:
        await a.clear()
        await asyncio.sleep(0)
        a.runtime.inbox.push_back(types.history.UserMessage(text="fresh"))
        await asyncio.wait_for(idle.wait(), timeout=1.0)
    finally:
        a.shutdown(force=True)
        with contextlib.suppress(asyncio.CancelledError):
            await drive
        a.runtime.observers.remove(_watch)

    user_texts = [
        entry.text
        for entry in a.history
        if isinstance(entry, types.history.UserMessage)
    ]
    assert not any("[cancelled]" in text for text in user_texts)
    assert not any(text.startswith("[Tool ") for text in user_texts)


@pytest.mark.asyncio
async def test_agent_tool_background_exception_is_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    @dataclass(slots=True, kw_only=True)
    class FailingTool(StubTool):
        @override
        async def run(self, args: Mapping[str, object]) -> types.history.ToolResult:
            del args
            raise RuntimeError("boom")

    a = _build_agent(tools=[FailingTool()])
    wrapper = next(t for t in a.runtime.tools_map.values() if t.name == "Echo")
    with caplog.at_level(logging.ERROR, logger="sagent.agent.agent"):
        result = await wrapper.run({"background": True})
        assert not result.is_error
        task = next(iter(a.background.values())).task
        await task
    assert any(
        "background tool 'Echo' failed" in r.getMessage() for r in caplog.records
    )
    items = await a.runtime.inbox.drain()
    detached = [i for i in items if isinstance(i, types.runtime.DetachedResult)]
    assert len(detached) == 1
    assert detached[0].is_error


@pytest.mark.asyncio
async def test_agent_compactor_appends_continuation_when_summary_ends_assistant(
    tmp_path: Path,
) -> None:
    """If summary ends with types.history.AssistantMessage, an inert types.history.UserMessage is appended."""

    @dataclass(slots=True, kw_only=True)
    class _AssistantTerminatedCompactor:
        async def should_compact(
            self,
            input_tokens: int,
            max_request_tokens: int,
            max_response_tokens: int = 0,
        ) -> bool:
            del input_tokens, max_request_tokens, max_response_tokens
            return False

        async def compact(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[types.history.HistoryEntry],
            model: object,
            mint_ref: Callable[[], TapeRef],
            custom_instructions: str | None = None,
        ) -> ContextSplice:
            del context, model
            del custom_instructions
            return _summary_override(
                [types.history.AssistantMessage(text="model said")], mint_ref, tape=tape
            )

        def maintain(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[types.history.HistoryEntry],
            tools: object,
            mint_ref: Callable[[], TapeRef],
        ) -> tuple[ContextSplice, ...]:
            del tape, context, tools, mint_ref
            return ()

    a = Agent(
        model=StubModel(),
        tools=[],
        compactor=_AssistantTerminatedCompactor(),
        session_dir=tmp_path,
    )
    a.runtime.append_history(types.history.UserMessage(text="x"))
    await a.compact_now()
    # The continuation user-message terminator was appended.
    last = a.runtime.context().messages[-1]
    assert isinstance(last, types.history.UserMessage)
    assert last.text == "[continuation]"


@pytest.mark.asyncio
async def test_agent_compactor_post_enrich_failure_swallowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Errors inside ``post_compact_enrich`` are logged and don't propagate."""

    @dataclass(slots=True, kw_only=True)
    class _OkCompactor:
        async def should_compact(
            self,
            input_tokens: int,
            max_request_tokens: int,
            max_response_tokens: int = 0,
        ) -> bool:
            del input_tokens, max_request_tokens, max_response_tokens
            return False

        async def compact(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[types.history.HistoryEntry],
            model: object,
            mint_ref: Callable[[], TapeRef],
            custom_instructions: str | None = None,
        ) -> ContextSplice:
            del context, model
            del custom_instructions
            return _summary_override(
                [types.history.UserMessage(text="[summary]")], mint_ref, tape=tape
            )

        def maintain(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[types.history.HistoryEntry],
            tools: object,
            mint_ref: Callable[[], TapeRef],
        ) -> tuple[ContextSplice, ...]:
            del tape, context, tools, mint_ref
            return ()

    async def _boom(**kwargs: object) -> None:
        del kwargs
        raise RuntimeError("enrich failed")

    a = Agent(
        model=StubModel(),
        tools=[],
        compactor=_OkCompactor(),
        session_dir=tmp_path,
    )
    a.runtime.append_history(types.history.UserMessage(text="x"))

    monkeypatch.setattr("sagent.agent.agent.post_compact_enrich", _boom)
    await a.compact_now()

    # Summary still survived; the enrich failure was swallowed.
    assert any(
        isinstance(e, types.history.UserMessage) and e.text == "[summary]"
        for e in a.runtime.context().messages
    )


def test_callable_system_composes_sections() -> None:
    """Callable system can compose multiple named sections internally."""

    def factory() -> str:
        return "\n\n".join(["static", _env()])

    def _env() -> str:
        return "dynamic"

    a = _build_agent(system=factory)
    assembled = a.system_prompt()
    assert "static" in assembled
    assert "dynamic" in assembled


def test_callable_system_re_evaluates_per_call() -> None:
    """C4: callable system is re-evaluated on every ``system_prompt`` call."""
    counter = {"n": 0}

    def factory() -> str:
        counter["n"] += 1
        return f"call#{counter['n']}"

    a = _build_agent(system=factory)
    p1 = a.system_prompt()
    p2 = a.system_prompt()
    assert p1 != p2


@pytest.mark.asyncio
async def test_stream_rebuilds_system_per_request() -> None:
    """C4: ``_AgentModel.stream`` must rebuild the system prompt each
    call so cwd-aware sections stay live after ``cd``.
    """
    call_count = 0

    def factory() -> str:
        nonlocal call_count
        call_count += 1
        return f"sys-v{call_count}"

    model = StubModel()
    a = _build_agent(model=model, system=factory)
    pre_run_count = call_count
    async for _ in a.run(types.history.UserMessage(text="hi")):
        pass
    assert call_count > pre_run_count
    assert model.received[-1].system == f"sys-v{call_count}"


def test_subagent_inherits_root_cost_tracker() -> None:
    """C5: non-persistent subagent's cost folds into the root tracker."""
    root = _build_agent()
    child = _build_agent()
    response = types.model.ModelResponse(
        message=types.history.AssistantMessage(text="ok"),
        tokens=types.model.TokenCount(input_tokens=10, output_tokens=5),
        total_cost=0.02,
    )
    with root._install_contextvars(), child._install_contextvars():
        child.record_response(response)
    assert root.cost_tracker.total_cost_usd == pytest.approx(0.02)
    assert child.cost_tracker.total_cost_usd == 0.0


def test_persistent_subagent_shadows_root_cost_tracker() -> None:
    """C5: persistent subagent gets its own cost tracker (parent unaffected)."""
    root = _build_agent()
    child = _build_agent()
    child._persistent = True
    response = types.model.ModelResponse(
        message=types.history.AssistantMessage(text="ok"),
        tokens=types.model.TokenCount(),
        total_cost=0.02,
    )
    with root._install_contextvars(), child._install_contextvars():
        child.record_response(response)
    assert root.cost_tracker.total_cost_usd == 0.0
    assert child.cost_tracker.total_cost_usd == pytest.approx(0.02)


def test_subagent_tool_state_depth_increments() -> None:
    """C6: nested subagent's ``ToolState.depth`` reflects spawn depth."""
    root = _build_agent()
    child = _build_agent()
    with root._install_contextvars():
        assert root.tool_state.depth == 0
        with child._install_contextvars():
            assert child.tool_state.depth == 1


def test_default_named_agents_get_unique_registry_label() -> None:
    """H11/M7: two agents with the same name don't overwrite each other."""
    a1 = _build_agent()
    a2 = _build_agent()
    assert a1.name == a2.name == "Agent"
    with a1._install_contextvars(), a2._install_contextvars():
        labels = [k for k, v in agent_registry.items() if v in (a1, a2)]
        assert len(set(labels)) == 2


def test_status_setter_publishes_status_changed() -> None:
    """H5: changing ``status`` publishes ``types.runtime.StatusChanged`` to observers."""
    a = _build_agent()
    events: list[str] = []

    def watch(event: object) -> None:
        if isinstance(event, types.runtime.StatusChanged):
            events.append(event.text)

    a.runtime.observers.append(watch)
    a.status = "working"
    a.status = "working"  # No-op: same value.
    a.status = "idle"
    assert events == ["working", "idle"]


def test_validate_input_missing_required() -> None:
    """C8: missing required field surfaces a structured error."""
    schema = json_freeze(
        {
            "type": "object",
            "properties": {"file_path": {"type": "string"}},
            "required": ["file_path"],
            "additionalProperties": False,
        }
    )
    err = _validate_input("Read", schema, {})
    assert err is not None
    assert "file_path" in err
    assert "InputValidationError" in err


def test_validate_input_unexpected_field() -> None:
    """C8: extra field with ``additionalProperties: false`` is reported."""
    schema = json_freeze(
        {
            "type": "object",
            "properties": {"msg": {"type": "string"}},
            "additionalProperties": False,
        }
    )
    err = _validate_input("Echo", schema, {"bogus": 1})
    assert err is not None
    assert "Unexpected parameter `bogus`" in err


def test_validate_input_nested_required() -> None:
    schema = json_freeze(
        {
            "type": "object",
            "properties": {
                "payload": {
                    "type": "object",
                    "properties": {"file_path": {"type": "string"}},
                    "required": ["file_path"],
                }
            },
        }
    )
    err = _validate_input("Nested", schema, {"payload": {}})
    assert err is not None
    assert "The required parameter `payload.file_path` is missing." in err


def test_validate_input_nested_unexpected_field() -> None:
    schema = json_freeze(
        {
            "type": "object",
            "properties": {
                "payload": {
                    "type": "object",
                    "properties": {"file_path": {"type": "string"}},
                    "additionalProperties": False,
                }
            },
        }
    )
    err = _validate_input(
        "Nested", schema, {"payload": {"file_path": "x", "extra": True}}
    )
    assert err is not None
    assert "Unexpected parameter `payload.extra`." in err


def test_validate_input_array_items_nested_required() -> None:
    schema = json_freeze(
        {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"id": {"type": "string"}},
                        "required": ["id"],
                    },
                }
            },
        }
    )
    err = _validate_input("Nested", schema, {"items": [dict[str, object]()]})
    assert err is not None
    assert "The required parameter `items[0].id` is missing." in err


def test_validate_input_valid_passes() -> None:
    """C8: well-formed args return ``None``."""
    schema = json_freeze(
        {
            "type": "object",
            "properties": {"msg": {"type": "string"}},
            "required": ["msg"],
            "additionalProperties": False,
        }
    )
    assert _validate_input("Echo", schema, {"msg": "hi"}) is None


def test_split_bg_args_strips_background_and_delay() -> None:
    """C7: ``background`` / ``delay`` removed from forwarded args."""
    bg, delay, clean = split_bg_args({"msg": "hi", "background": True, "delay": 3})
    assert bg is True
    assert delay == 3.0
    assert clean == {"msg": "hi"}


def test_split_bg_args_delay_implies_background() -> None:
    """C7: positive ``delay`` implies ``background=True`` even when omitted."""
    bg, delay, _ = split_bg_args({"delay": 5})
    assert bg is True
    assert delay == 5.0


@pytest.mark.asyncio
async def test_model_switch_event_queues_swap_until_call_drains() -> None:
    """Write-mode ``/model`` (via ``runtime.ModelSwitch``) defers the
    swap until the in-flight model call finishes, so cost is recorded
    against the OLD model and only the NEXT call uses the NEW one.

    Calling ``agent.swap_model`` directly mid-call mis-attributes cost
    (the bug); routing through the inbox sequences the swap correctly.
    """
    stream_entered = asyncio.Event()
    gate = asyncio.Event()

    @dataclass(slots=True, kw_only=True)
    class GatedModel(StubModel):
        @override
        async def stream(
            self,
            request: types.model.ModelRequest,
            on_text: object = None,
            on_thinking: object = None,
        ) -> types.model.ModelResponse:
            del on_text, on_thinking
            self.received.append(request)
            stream_entered.set()
            await gate.wait()
            return types.model.ModelResponse(
                message=types.history.AssistantMessage(text="from A"), total_cost=0.10
            )

    model_a = GatedModel(model_id="model-A")
    model_b = StubModel(model_id="model-B")
    agent = _build_agent(model=model_a)

    async def consume() -> None:
        async for _ in agent.run(types.history.UserMessage(text="hi")):
            pass

    drive = asyncio.create_task(consume())
    await asyncio.wait_for(stream_entered.wait(), timeout=1.0)

    # User types ``/model model-B``. Slash handler pushes a
    # ``ModelSwitch`` event into the inbox; runtime defers it.
    agent.runtime.inbox.push_back(
        types.runtime.ModelSwitch(
            apply=lambda: agent.swap_model(model_b), label="A -> B"
        ),
    )

    # Release the in-flight call.
    gate.set()
    await drive

    # Cost lands on model-A: the runtime didn't apply the swap until
    # AFTER ``types.runtime.ModelResponseComplete`` set ``model_call`` to None, which
    # is after ``_AgentModel.stream`` called ``record_response``.
    assert len(model_a.received) == 1
    assert len(model_b.received) == 0
    assert agent.cost_tracker.calls_by_model == {"model-A": 1}
    # And the deferred swap did fire: the NEXT call would go to B.
    assert agent.model is model_b


def test_swap_model_clears_unsupported_thinking() -> None:
    """L4: swapping to a model without thinking support clears ``_thinking``."""

    @dataclass(slots=True, kw_only=True)
    class _NoThinkingModel(StubModel):
        supports_thinking: bool = False

    a = _build_agent()
    a.thinking = "adaptive"
    a.swap_model(_NoThinkingModel())
    assert a.thinking is None


def test_recompact_docstring_describes_alias_semantics() -> None:
    assert Agent.recompact.__doc__ is not None
    assert "alias" in Agent.recompact.__doc__.lower()


@pytest.mark.asyncio
async def test_compact_if_needed_uses_agent_request_cap() -> None:
    @dataclass(slots=True, kw_only=True)
    class _RecorderCompactor:
        seen_max_request_tokens: list[int] = field(default_factory=list)

        async def should_compact(
            self,
            input_tokens: int,
            max_request_tokens: int,
            max_response_tokens: int = 0,
        ) -> bool:
            del input_tokens, max_response_tokens
            self.seen_max_request_tokens.append(max_request_tokens)
            return False

        async def compact(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[types.history.HistoryEntry],
            model: object,
            mint_ref: Callable[[], TapeRef],
            custom_instructions: str | None = None,
        ) -> ContextSplice:
            del tape, context, model, mint_ref, custom_instructions
            raise AssertionError("compact should not run")

    compactor = _RecorderCompactor()
    a = Agent(model=StubModel(max_request_tokens=100_000), compactor=compactor)
    a.max_request_tokens = 10_000

    progressed = await a.compact_if_needed(
        [types.history.UserMessage(text="x")], a.model
    )

    assert progressed is True
    assert compactor.seen_max_request_tokens == [10_000]


@pytest.mark.asyncio
async def test_compact_if_needed_resets_failure_breaker_when_healthy() -> None:
    @dataclass(slots=True, kw_only=True)
    class _HealthyCompactor:
        async def should_compact(
            self,
            input_tokens: int,
            max_request_tokens: int,
            max_response_tokens: int = 0,
        ) -> bool:
            del input_tokens, max_request_tokens, max_response_tokens
            return False

        async def compact(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[types.history.HistoryEntry],
            model: object,
            mint_ref: Callable[[], TapeRef],
            custom_instructions: str | None = None,
        ) -> ContextSplice:
            del tape, context, model, mint_ref, custom_instructions
            raise AssertionError("compact should not run")

    a = Agent(model=StubModel(), compactor=_HealthyCompactor())
    a.compaction_state.compact_failures = 3

    progressed = await a.compact_if_needed(
        [types.history.UserMessage(text="x")], a.model
    )

    assert progressed is True
    assert a.compaction_state.compact_failures == 0


@pytest.mark.asyncio
async def test_compactor_estimates_use_live_background_aware_tools() -> None:
    @dataclass(slots=True, kw_only=True)
    class _RecordingModel(StubModel):
        estimated_tools: list[Sequence[types.tools.Tool] | None] = field(
            default_factory=list
        )

        @override
        def approx_request_tokens(self, request: types.model.ModelRequest) -> int:
            self.estimated_tools.append(request.tools)
            return StubModel.approx_request_tokens(self, request)

    @dataclass(slots=True, kw_only=True)
    class _NoopCompactor:
        async def should_compact(
            self,
            input_tokens: int,
            max_request_tokens: int,
            max_response_tokens: int = 0,
        ) -> bool:
            del input_tokens, max_request_tokens, max_response_tokens
            return False

        async def compact(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[types.history.HistoryEntry],
            model: object,
            mint_ref: Callable[[], TapeRef],
            custom_instructions: str | None = None,
        ) -> ContextSplice:
            del tape, context, model, mint_ref, custom_instructions
            raise AssertionError("compact should not run")

    model = _RecordingModel()
    tool = StubTool(
        directive_schema=json_freeze(
            {"type": "object", "properties": {"msg": {"type": "string"}}}
        )
    )
    a = Agent(model=model, tools=[tool], compactor=_NoopCompactor())

    progressed = await a.compact_if_needed([types.history.UserMessage(text="x")], model)

    assert progressed is True
    tools_seen = model.estimated_tools[-1]
    assert tools_seen is not None
    assert isinstance(tools_seen[0], BackgroundAwareTool)


@pytest.mark.asyncio
async def test_post_compact_estimates_use_live_background_aware_tools() -> None:
    @dataclass(slots=True, kw_only=True)
    class _RecordingModel(StubModel):
        estimated_tools: list[Sequence[types.tools.Tool] | None] = field(
            default_factory=list
        )

        @override
        def approx_request_tokens(self, request: types.model.ModelRequest) -> int:
            self.estimated_tools.append(request.tools)
            return super().approx_request_tokens(request)

    @dataclass(slots=True, kw_only=True)
    class _OkCompactor:
        async def should_compact(
            self,
            input_tokens: int,
            max_request_tokens: int,
            max_response_tokens: int = 0,
        ) -> bool:
            del input_tokens, max_request_tokens, max_response_tokens
            return True

        async def compact(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[types.history.HistoryEntry],
            model: object,
            mint_ref: Callable[[], TapeRef],
            custom_instructions: str | None = None,
        ) -> ContextSplice:
            del context, model, custom_instructions
            return _summary_override(
                [types.history.UserMessage(text="[summary]")], mint_ref, tape=tape
            )

    model = _RecordingModel()
    tool = StubTool(
        directive_schema=json_freeze(
            {"type": "object", "properties": {"msg": {"type": "string"}}}
        )
    )
    a = Agent(model=model, tools=[tool], compactor=_OkCompactor())
    a.runtime.append_history(types.history.UserMessage(text="x"))

    assert await a.compact_now() is True

    for tools_seen in model.estimated_tools:
        assert tools_seen is not None
        assert isinstance(tools_seen[0], BackgroundAwareTool)


@pytest.mark.asyncio
async def test_post_compact_hook_budget_uses_active_model_ratio() -> None:
    calls: list[int] = []

    @dataclass(slots=True, kw_only=True)
    class _RatioModel(StubModel):
        @override
        def approx_text_tokens(self, text: str) -> int:
            return len(text) // 2

        @override
        def approx_request_tokens(self, request: types.model.ModelRequest) -> int:
            del request
            return self.max_request_tokens - 100

    @dataclass(slots=True, kw_only=True)
    class _RestorableTool(StubTool):
        async def post_compact_restore(
            self,
            history: list[types.history.HistoryEntry],
            tool_state: ToolState,
            *,
            budget_chars: int = 100_000,
        ) -> None:
            del history, tool_state
            calls.append(budget_chars)

    @dataclass(slots=True, kw_only=True)
    class _OkCompactor:
        async def should_compact(
            self,
            input_tokens: int,
            max_request_tokens: int,
            max_response_tokens: int = 0,
        ) -> bool:
            del input_tokens, max_request_tokens, max_response_tokens
            return True

        async def compact(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[types.history.HistoryEntry],
            model: object,
            mint_ref: Callable[[], TapeRef],
            custom_instructions: str | None = None,
        ) -> ContextSplice:
            del context, model, custom_instructions
            return _summary_override(
                [types.history.UserMessage(text="[summary]")], mint_ref, tape=tape
            )

    budget = types.model.ContextBudget(
        max_request_tokens=1_000,
        max_response_tokens=5,
        chars_per_token=4,
        buffer_tokens=5,
    )
    a = Agent(
        model=_RatioModel(max_request_tokens=1_000, max_response_tokens=5),
        tools=[_RestorableTool()],
        compactor=_OkCompactor(),
        budget=budget,
    )
    a.runtime.append_history(types.history.UserMessage(text="x"))

    assert await a.compact_now() is True

    assert calls == [180]


def test_agent_with_session_dir_auto_persists(tmp_path: Path) -> None:
    """Constructing an Agent with ``session_dir`` self-installs the persistence
    observer; no external wiring needed.

    Regression test for a bug surfaced by session 8588644a: AgentSpawn handed
    each child a ``session_dir`` but the persistence observer was only wired
    by the CLI for the root agent, so subagent work vanished without trace.
    Now Agent.__init__ auto-installs.
    """
    a = _build_agent(session_dir=tmp_path)
    a.status = "working"  # triggers StatusChanged → persistence observer fires
    session_file = tmp_path / "session.jsonl"
    assert session_file.exists()
    lines = session_file.read_text(encoding="utf-8").splitlines()
    assert lines, "expected at least one record"
    first = json.loads(lines[0])
    assert first["kind"] == "meta"
    assert first["status"] == "working"


def test_agent_without_session_dir_does_not_persist(tmp_path: Path) -> None:
    """No ``session_dir`` -> no observer attached -> no file written."""
    a = _build_agent()
    a.status = "working"
    assert not (tmp_path / "session.jsonl").exists()


def test_agent_resume_rebaselines_persistence(tmp_path: Path) -> None:
    """``resume()`` must rebaseline the persistence observer.

    Without rebaselining, the replayed tape records would be written
    back to ``session.jsonl`` on the next ``SaveSession``, duplicating
    every persisted record. Tests this by populating session.jsonl,
    constructing a fresh Agent with the same session_dir, resuming
    from the file, then triggering a SaveSession and asserting the
    file size hasn't grown.
    """
    # Phase 1: write some records via a first agent.
    a1 = _build_agent(session_dir=tmp_path)
    a1.status = "first"
    a1.status = "second"  # ensure at least one tape record gets written
    a1.runtime.publish(types.runtime.SaveSession())
    session_file = tmp_path / "session.jsonl"
    assert session_file.exists()
    size_after_phase1 = session_file.stat().st_size
    assert size_after_phase1 > 0

    # Phase 2: fresh agent, same session_dir. Load, resume, save.
    loaded = load_session(tmp_path, {})
    assert loaded is not None
    a2 = _build_agent(session_dir=tmp_path)
    a2.resume(*loaded)
    a2.runtime.publish(types.runtime.SaveSession())

    # File should not have grown materially -- the resumed tape was
    # already on disk, so the SaveSession after resume must be a no-op
    # on tape (only the meta line gets updated, if anything).
    size_after_resume = session_file.stat().st_size
    # The post-resume save may rewrite meta; allow some growth, but not
    # the full tape duplication that would result from a missing rebaseline.
    assert size_after_resume < 2 * size_after_phase1, (
        f"resume rebaseline broken: file grew from {size_after_phase1}"
        f" to {size_after_resume} after resume+save"
    )


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
