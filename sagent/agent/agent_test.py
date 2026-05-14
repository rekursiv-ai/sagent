"""Tests for ``agent.agent``: Agent composition class."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast, override

import asyncio
import contextlib

import pytest

from sagent.agent.agent import (
    ActivityTracker,
    Agent,
    SystemPromptArg,
    _validate_input,
)
from sagent.agent.background import (
    BackgroundAwareTool,
    BackgroundTaskEntry,
    split_bg_args,
)
from sagent.agent.runtime import (
    AssistantMessage,
    Clear as RuntimeClear,
    Halt as RuntimeHalt,
    HistoryEntry,
    Kill as RuntimeKill,
    ModelCallStarted,
    ModelIdle,
    ModelResponseComplete,
    ModelResponseError,
    ModelResponsePartial,
    ModelSwitch as RuntimeModelSwitch,
    StatusChanged,
    ToolCall as RuntimeToolCall,
    ToolLabel,
    ToolResult,
    UserMessage,
)
from sagent.agent.state import agent_registry
from sagent.custom_exceptions import PromptTooLongError
from sagent.custom_types import (
    ContextBudget,
    Model as RichModel,
    ModelRequest,
    ModelResponse,
    Pricing,
    TokenCount,
    Tool as RichTool,
)
from sagent.lib.json import JSON, json_freeze


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
    supports_context_management: bool = False
    supports_persistent_retry: bool = False
    supports_account_auth: bool = False
    max_image_dim: int = 8_000
    max_image_bytes: int = 5 * 1024 * 1024
    responses: list[AssistantMessage] = field(default_factory=list)
    received: list[ModelRequest] = field(default_factory=list)

    @property
    def pricing(self) -> Pricing:
        return Pricing()

    def estimate_text_token_count(self, text: str) -> int:
        return max(1, len(text) // 4)

    def estimate_image_token_count(self, data: bytes) -> int:
        del data
        return 256

    def is_context_overflow(self, error: Exception) -> bool:
        del error
        return False

    def is_retryable_provider_error(self, error: Exception) -> bool:
        del error
        return False

    async def buffer(self, request: ModelRequest) -> ModelResponse:
        return await self.stream(request)

    async def stream(
        self,
        request: ModelRequest,
        on_text: object = None,
        on_thinking: object = None,
    ) -> ModelResponse:
        del on_text, on_thinking
        self.received.append(request)
        msg = self.responses.pop(0) if self.responses else AssistantMessage(text="ok")
        return ModelResponse(message=msg)


_STUB_SCHEMA: JSON = json_freeze({"type": "object"})


@dataclass(slots=True, kw_only=True)
class StubTool:
    """Minimal tool that records calls."""

    name: str = "Echo"
    tool_id: str = "application/x-tool-echo"
    description: str = "Echo tool."
    supports_microcompaction: bool = False
    directive_schema: JSON = _STUB_SCHEMA
    calls: list[Mapping[str, object]] = field(default_factory=list)

    def summary(self, args: Mapping[str, object]) -> str:
        del args
        return "echo"

    def summary_result(self, result: ToolResult) -> str | None:
        del result
        return None

    def prompt(self) -> str:
        return ""

    async def run(self, args: Mapping[str, object]) -> ToolResult:
        self.calls.append(args)
        return ToolResult(call_id="", content=str(args.get("msg", "")))


def _build_agent(
    *,
    model: RichModel | None = None,
    tools: list[RichTool] | None = None,
    system: SystemPromptArg = "",
    budget: ContextBudget | None = None,
    max_budget_usd: float | None = None,
) -> Agent:
    return Agent(
        model=model or StubModel(),
        tools=tools or [],
        system=system,
        budget=budget,
        max_budget_usd=max_budget_usd,
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
    assert isinstance(a.budget, ContextBudget)
    assert a.budget.max_request_tokens == 100_000


def test_agent_budget_override_respected() -> None:
    b = ContextBudget.from_model(StubModel())
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
    finally:
        loop.close()


def test_agent_microcompact_history_noop_without_compactor() -> None:
    a = _build_agent()
    msg = UserMessage(text="hi")
    history: list[HistoryEntry] = [msg]
    a.microcompact_history(history)
    # No compactor wired -- history unchanged identity.
    assert history == [msg]
    assert history[0] is msg


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
    tool: RichTool = StubTool()
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
    async for _ in a.run(UserMessage(text="hi")):
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
        type(ev).__name__ async for ev in a.run(UserMessage(text="ping"))
    ]
    assert "ModelIdle" in events
    assert len(a.history) >= 2
    assert isinstance(a.history[0], UserMessage)


@pytest.mark.asyncio
async def test_agent_run_passes_rich_tools_to_model() -> None:
    """Regression: provider iterates ``request.tools`` reading ``.description``
    etc.; the runtime hands the model layer ``_AgentTool`` wrappers, which
    expose only ``name``/``run``. The Agent must translate them back to rich
    tools before constructing ``ModelRequest`` so providers see the full
    Tool surface.
    """
    model = StubModel()
    tool: RichTool = StubTool()
    a = _build_agent(model=model, tools=[tool])
    async for _ in a.run(UserMessage(text="hi")):
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
    assert seen.supports_microcompaction is False
    assert seen.summary({}) == "echo"
    assert seen.prompt() == ""


@pytest.mark.asyncio
async def test_agent_records_response_into_cost_tracker() -> None:
    model = StubModel()
    a = _build_agent(model=model)
    async for _ in a.run(UserMessage(text="ping")):
        pass
    # StubModel emits empty TokenCount, but ``calls_by_model`` records
    # one entry per model invocation.
    assert a.cost_tracker.calls_by_model.get("stub-1", 0) >= 1


def test_agent_record_response_budget_exhaustion_raises() -> None:
    a = _build_agent(max_budget_usd=1.0)
    # First response below the cap: clean.
    a.record_response(ModelResponse(message=AssistantMessage(text="x")))
    # Force an over-budget total and verify the next call raises.
    a.cost_tracker.total_cost_usd = 2.0
    with pytest.raises(RuntimeError, match="Budget exhausted"):
        a.record_response(ModelResponse(message=AssistantMessage(text="x")))


def test_token_count_addable() -> None:
    """``TokenCount`` supports ``+`` so ``CostTracker.record`` can fold."""
    a = TokenCount()
    b = TokenCount()
    c = a + b
    assert isinstance(c, TokenCount)


def test_agent_shutdown_idempotent() -> None:
    a = _build_agent()
    a.shutdown()
    a.shutdown()  # Second call must not raise.


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
    a.thinking = None
    assert a.thinking is None


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
    tool: RichTool = StubTool()
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


def test_halt_pushes_halt_event() -> None:
    a = _build_agent()
    a.halt()
    items = asyncio.new_event_loop().run_until_complete(a.runtime.inbox.drain())
    assert any(isinstance(i, RuntimeHalt) for i in items)


def test_kill_tool_pushes_kill_event() -> None:
    a = _build_agent()
    a.kill_tool("call-7")
    items = asyncio.new_event_loop().run_until_complete(a.runtime.inbox.drain())
    kills = [i for i in items if isinstance(i, RuntimeKill)]
    assert len(kills) == 1
    assert kills[0].call_id == "call-7"


def test_kill_all_tools_pushes_kill_with_none_id() -> None:
    a = _build_agent()
    a.kill_all_tools()
    items = asyncio.new_event_loop().run_until_complete(a.runtime.inbox.drain())
    kills = [i for i in items if isinstance(i, RuntimeKill)]
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
            history: list[HistoryEntry],
            model: object,
            transcript_path: object = None,
            direction: str = "from",
            keep_recent: int | None = None,
            custom_instructions: str | None = None,
            summary_pointers: object = None,
        ) -> list[HistoryEntry]:
            del history, model, transcript_path, direction, keep_recent
            del custom_instructions, summary_pointers
            return [UserMessage(text="[summary]")]

        def maintain(
            self, history: list[HistoryEntry], tools: object, **kwargs: object
        ) -> None:
            del history, tools, kwargs

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

    assert any(isinstance(e, UserMessage) and e.text == "[summary]" for e in a.history)


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
            history: list[HistoryEntry],
            model: object,
            transcript_path: object = None,
            direction: str = "from",
            keep_recent: int | None = None,
            custom_instructions: str | None = None,
            summary_pointers: object = None,
        ) -> list[HistoryEntry]:
            del history, model, transcript_path, direction, keep_recent
            del custom_instructions, summary_pointers
            return [UserMessage(text="[recompacted]")]

        def maintain(
            self, history: list[HistoryEntry], tools: object, **kwargs: object
        ) -> None:
            del history, tools, kwargs

    a = Agent(model=StubModel(), tools=[], compactor=_StubCompactor())
    drive_task = asyncio.create_task(a.serve_forever())
    try:
        await a.recompact("instr")
    finally:
        a.shutdown()
        with contextlib.suppress(asyncio.CancelledError):
            await drive_task

    assert any(
        isinstance(e, UserMessage) and e.text == "[recompacted]" for e in a.history
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
    assert any(isinstance(i, RuntimeClear) for i in items)


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
    """``activity.active`` stays True from first ModelCallStarted through ModelIdle.

    Before fix: ``ModelResponseComplete`` always cleared ``active``,
    so the status-pane spinner went dark during tool execution. The
    user sees a long-running Bash with no visible progress indicator.

    After fix: when ``ModelResponseComplete`` carries ``tool_calls``,
    ``active`` stays True so the spinner keeps ticking through the
    cohort window. ``active`` only clears on true terminal events
    (``ModelIdle`` / cancel / error).
    """
    a = _build_agent()
    tc = RuntimeToolCall(id="c1", name="Echo", args={})
    msg_with_tools = AssistantMessage(text="", tool_calls=(tc,))

    a.publish(ModelCallStarted())
    assert a.activity.active is True

    a.publish(ModelResponseComplete(message=msg_with_tools))
    assert a.activity.active is True, (
        "spinner should keep ticking while tools run; tool_calls in the "
        "response mean the cohort is about to fire"
    )

    # Tool result arrives.
    a.publish(ToolResult(call_id="c1", content="ok"))
    assert a.activity.active is True

    # Round 2 model call fires.
    a.publish(ModelCallStarted())
    assert a.activity.active is True

    # Round 2 has no tool calls; model truly idles.
    a.publish(ModelResponseComplete(message=AssistantMessage(text="done")))
    a.publish(ModelIdle())
    assert a.activity.active is False, "ModelIdle marks the end of the round chain"


@pytest.mark.asyncio
async def test_activity_active_clears_on_model_response_error() -> None:
    """``ModelResponseError`` must clear ``activity.active`` (stop spinner).

    Bug repro: model call fails (e.g. ``AuthRefreshError`` on expired
    OAuth). The runtime catches the exception and pushes
    ``ModelResponseError`` to the inbox. Before fix: ``_record_activity``
    only resets ``active`` on ``ModelResponseComplete`` / ``ModelIdle``
    / ``ModelResponseCancelled``, so the status-pane spinner keeps
    ticking forever even though no model call is in flight. After fix:
    ``ModelResponseError`` joins the terminal-event set and clears
    ``active``.
    """
    a = _build_agent()
    a.publish(ModelCallStarted())
    assert a.activity.active is True

    a.publish(ModelResponseError(RuntimeError("boom")))
    assert a.activity.active is False, (
        "ModelResponseError is terminal -- spinner must stop"
    )
    assert a.activity.current_call_start == 0.0, (
        f"current_call_start must reset; got {a.activity.current_call_start}"
    )


@pytest.mark.asyncio
async def test_streaming_chars_recorded_in_activity() -> None:
    """``_track_activity`` accumulates streamed chars on ModelResponsePartial."""
    a = _build_agent()
    # Push a partial event through the publish path.
    a.publish(ModelResponsePartial(text="abc"))
    # The handler only acts when ``active`` is True; bracket via
    # ModelCallStarted first.
    a.publish(ModelCallStarted())
    a.publish(ModelResponsePartial(text="defg"))
    assert a.activity.live_response_chars == 4


def test_tool_registry_recorded_on_response_with_tool_calls() -> None:
    """``_track_tool_registry`` records cohort id → tool name and bumps rounds."""
    a = _build_agent()
    tc = RuntimeToolCall(id="c1", name="Echo", args={})
    msg = AssistantMessage(text="", tool_calls=(tc,))
    a.publish(ModelResponseComplete(message=msg))
    assert a._tool_registry["c1"][0] == "Echo"
    assert a.activity.num_tool_call_rounds == 1


def test_enforce_caps_pushes_error_when_limit_reached() -> None:
    """``_enforce_caps`` posts a ModelResponseError when rounds cap is hit."""
    a = Agent(model=StubModel(), tools=[], max_tool_call_rounds=1)
    tc = RuntimeToolCall(id="c1", name="Echo", args={})
    msg = AssistantMessage(text="", tool_calls=(tc,))
    a.publish(ModelResponseComplete(message=msg))

    # Round count is now 1 == cap; the next observation triggers the
    # error push.
    a.publish(ModelResponseComplete(message=msg))
    items = asyncio.new_event_loop().run_until_complete(a.runtime.inbox.drain())
    assert any(isinstance(i, ModelResponseError) for i in items)


@dataclass(slots=True, kw_only=True)
class _MaintainStubCompactor:
    """Compactor whose ``maintain`` records calls."""

    maintained: list[list[HistoryEntry]] = field(default_factory=list)

    async def should_compact(
        self, input_tokens: int, max_request_tokens: int, max_response_tokens: int = 0
    ) -> bool:
        del input_tokens, max_request_tokens, max_response_tokens
        return False

    async def compact(
        self,
        history: list[HistoryEntry],
        model: object,
        transcript_path: object = None,
        direction: str = "from",
        keep_recent: int | None = None,
        custom_instructions: str | None = None,
        summary_pointers: object = None,
    ) -> list[HistoryEntry]:
        del model, transcript_path, direction, keep_recent
        del custom_instructions, summary_pointers
        return list(history)

    def maintain(
        self, history: list[HistoryEntry], tools: object, **kwargs: object
    ) -> None:
        del tools, kwargs
        self.maintained.append(list(history))


def test_microcompact_history_forwards_to_compactor() -> None:
    compactor = _MaintainStubCompactor()
    a = Agent(model=StubModel(), tools=[], compactor=compactor)
    history: list[HistoryEntry] = [UserMessage(text="x")]
    a.microcompact_history(history)
    assert compactor.maintained, "maintain() should have been called"


@pytest.mark.asyncio
async def test_compact_now_no_compactor_is_noop() -> None:
    a = _build_agent()
    a.runtime.history.append(UserMessage(text="x"))
    await a.compact_now()
    # History untouched: no compactor wired.
    assert len(a.runtime.history) == 1


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
            history: list[HistoryEntry],
            model: object,
            transcript_path: object = None,
            direction: str = "from",
            keep_recent: int | None = None,
            custom_instructions: str | None = None,
            summary_pointers: object = None,
        ) -> list[HistoryEntry]:
            del history, model, transcript_path, direction, keep_recent
            del custom_instructions, summary_pointers
            return [UserMessage(text="[summary]")]

        def maintain(
            self, history: list[HistoryEntry], tools: object, **kwargs: object
        ) -> None:
            del history, tools, kwargs

    a = Agent(model=StubModel(), tools=[], compactor=_ReplaceCompactor())
    a.runtime.history.append(UserMessage(text="old"))
    await a.compact_now()
    assert len(a.runtime.history) == 1
    entry = a.runtime.history[0]
    assert isinstance(entry, UserMessage)
    assert entry.text == "[summary]"


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
            history: list[HistoryEntry],
            model: object,
            transcript_path: object = None,
            direction: str = "from",
            keep_recent: int | None = None,
            custom_instructions: str | None = None,
            summary_pointers: object = None,
        ) -> list[HistoryEntry]:
            del history, model, transcript_path, direction, keep_recent
            del custom_instructions, summary_pointers
            raise RuntimeError("compaction failed")

        def maintain(
            self, history: list[HistoryEntry], tools: object, **kwargs: object
        ) -> None:
            del history, tools, kwargs

    a = Agent(model=StubModel(), tools=[], compactor=_BrokenCompactor())
    a.runtime.history.append(UserMessage(text="x"))
    await a.compact_now()
    err = [
        e
        for e in a.runtime.history
        if isinstance(e, UserMessage) and "[Compaction error:" in e.text
    ]
    assert len(err) == 1


@dataclass(slots=True, kw_only=True)
class _OverflowModel:
    """Model that raises PromptTooLongError on the first N calls."""

    model_id: str = "ovf"
    max_request_tokens: int = 100_000
    max_response_tokens: int = 1_024
    supports_streaming: bool = True
    supports_thinking: bool = False
    supports_effort: bool = False
    supports_cache_control: bool = False
    supports_context_management: bool = False
    supports_persistent_retry: bool = False
    supports_account_auth: bool = False
    max_image_dim: int = 8_000
    max_image_bytes: int = 5 * 1024 * 1024
    overflow_count: int = 0
    call_index: int = 0

    @property
    def pricing(self) -> Pricing:
        return Pricing()

    def estimate_text_token_count(self, text: str) -> int:
        return max(1, len(text) // 4)

    def estimate_image_token_count(self, data: bytes) -> int:
        del data
        return 256

    def is_context_overflow(self, error: Exception) -> bool:
        return isinstance(error, PromptTooLongError)

    def is_retryable_provider_error(self, error: Exception) -> bool:
        del error
        return False

    async def buffer(self, request: ModelRequest) -> ModelResponse:
        return await self.stream(request)

    async def stream(
        self,
        request: ModelRequest,
        on_text: object = None,
        on_thinking: object = None,
    ) -> ModelResponse:
        del request, on_text, on_thinking
        idx = self.call_index
        self.call_index += 1
        if idx < self.overflow_count:
            raise PromptTooLongError("too long")
        return ModelResponse(message=AssistantMessage(text="recovered"))


@dataclass(slots=True, kw_only=True)
class _RawOverflowModel:
    """Model that raises a non-PromptTooLongError but classifies it as overflow.

    Mirrors the production failure where ``anthropic.APIStatusError``
    propagated up un-normalized: the recovery loop's catch must rely
    on ``is_context_overflow``, not on ``isinstance(exc,
    PromptTooLongError)``, or compaction never engages.
    """

    model_id: str = "raw"
    max_request_tokens: int = 100_000
    max_response_tokens: int = 1_024
    supports_streaming: bool = True
    supports_thinking: bool = False
    supports_effort: bool = False
    supports_cache_control: bool = False
    supports_context_management: bool = False
    supports_persistent_retry: bool = False
    supports_account_auth: bool = False
    max_image_dim: int = 8_000
    max_image_bytes: int = 5 * 1024 * 1024
    overflow_count: int = 0
    call_index: int = 0

    @property
    def pricing(self) -> Pricing:
        return Pricing()

    def estimate_text_token_count(self, text: str) -> int:
        return max(1, len(text) // 4)

    def estimate_image_token_count(self, data: bytes) -> int:
        del data
        return 256

    def is_context_overflow(self, error: Exception) -> bool:
        return isinstance(error, RuntimeError) and "context window" in str(error)

    def is_retryable_provider_error(self, error: Exception) -> bool:
        del error
        return False

    async def buffer(self, request: ModelRequest) -> ModelResponse:
        return await self.stream(request)

    async def stream(
        self,
        request: ModelRequest,
        on_text: object = None,
        on_thinking: object = None,
    ) -> ModelResponse:
        del request, on_text, on_thinking
        idx = self.call_index
        self.call_index += 1
        if idx < self.overflow_count:
            raise RuntimeError("Request size exceeds model context window")
        return ModelResponse(message=AssistantMessage(text="recovered"))


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
            history: list[HistoryEntry],
            model: object,
            transcript_path: object = None,
            direction: str = "from",
            keep_recent: int | None = None,
            custom_instructions: str | None = None,
            summary_pointers: object = None,
        ) -> list[HistoryEntry]:
            del history, model, transcript_path, direction, keep_recent
            del custom_instructions, summary_pointers
            compact_calls.append(1)
            return [UserMessage(text="[compact]")]

        def maintain(
            self, history: list[HistoryEntry], tools: object, **kwargs: object
        ) -> None:
            del history, tools, kwargs

    model = _OverflowModel(overflow_count=1)
    a = Agent(model=model, tools=[], compactor=_CountingCompactor())
    async for _ in a.run(UserMessage(text="hi")):
        pass
    assert len(compact_calls) == 1
    # _OverflowModel emitted "recovered" on the second call.
    assert any(
        isinstance(e, AssistantMessage) and e.text == "recovered" for e in a.history
    )


@pytest.mark.asyncio
async def test_agent_model_overflow_exhausts_recovery_raises() -> None:
    """Exhaust MAX_OVERFLOW_RECOVERY: PromptTooLongError surfaces."""

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
            history: list[HistoryEntry],
            model: object,
            transcript_path: object = None,
            direction: str = "from",
            keep_recent: int | None = None,
            custom_instructions: str | None = None,
            summary_pointers: object = None,
        ) -> list[HistoryEntry]:
            del history, model, transcript_path, direction, keep_recent
            del custom_instructions, summary_pointers
            # Returns short summary; model keeps overflowing.
            return [UserMessage(text="[compact]")]

        def maintain(
            self, history: list[HistoryEntry], tools: object, **kwargs: object
        ) -> None:
            del history, tools, kwargs

    model = _OverflowModel(overflow_count=10)  # always overflow
    a = Agent(model=model, tools=[], compactor=_NoOpCompactor())
    with pytest.raises(PromptTooLongError):
        await a._agent_model.stream(
            history=[UserMessage(text="x")],
            system="",
            tools=[],
            on_text=lambda _t: None,
            on_thinking=lambda _t: None,
        )


@pytest.mark.asyncio
async def test_agent_model_overflow_recovery_via_classifier_not_isinstance() -> None:
    """Recovery engages on any exception classified as overflow, not just ``PromptTooLongError``.

    When a provider's normalization slips and a raw provider exception
    propagates with the canonical ``is_context_overflow(exc)`` returning
    True, the recovery loop must still fire ``compact_now``. This is
    the bug that produced the production death-spiral: the recovery
    catch was narrowed to ``PromptTooLongError`` while the classifier
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
            history: list[HistoryEntry],
            model: object,
            transcript_path: object = None,
            direction: str = "from",
            keep_recent: int | None = None,
            custom_instructions: str | None = None,
            summary_pointers: object = None,
        ) -> list[HistoryEntry]:
            del history, model, transcript_path, direction, keep_recent
            del custom_instructions, summary_pointers
            compact_calls.append(1)
            return [UserMessage(text="[compact]")]

        def maintain(
            self, history: list[HistoryEntry], tools: object, **kwargs: object
        ) -> None:
            del history, tools, kwargs

    model = _RawOverflowModel(overflow_count=1)
    a = Agent(model=model, tools=[], compactor=_CountingCompactor())
    async for _ in a.run(UserMessage(text="hi")):
        pass
    assert len(compact_calls) == 1
    assert any(
        isinstance(e, AssistantMessage) and e.text == "recovered" for e in a.history
    )


@pytest.mark.asyncio
async def test_agent_tool_emits_label_and_delegates() -> None:
    """The wrapper publishes a ToolLabel and forwards to the inner tool."""
    inner = StubTool()
    a = _build_agent(tools=[inner])

    labels: list[ToolLabel] = []

    def _watch(ev: object) -> None:
        if isinstance(ev, ToolLabel):
            labels.append(ev)

    a.runtime.observers.append(_watch)

    wrapper = next(t for t in a.runtime.tools_map.values() if t.name == "Echo")
    result = await wrapper.run({"msg": "hi"})
    assert result.content == "hi"
    assert len(labels) == 1
    assert labels[0].text == "echo"


@pytest.mark.asyncio
async def test_agent_compactor_appends_continuation_when_summary_ends_assistant(
    tmp_path: Path,
) -> None:
    """If summary ends with AssistantMessage, an inert UserMessage is appended."""

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
            history: list[HistoryEntry],
            model: object,
            transcript_path: object = None,
            direction: str = "from",
            keep_recent: int | None = None,
            custom_instructions: str | None = None,
            summary_pointers: object = None,
        ) -> list[HistoryEntry]:
            del history, model, transcript_path, direction, keep_recent
            del custom_instructions, summary_pointers
            return [AssistantMessage(text="model said")]

        def maintain(
            self, history: list[HistoryEntry], tools: object, **kwargs: object
        ) -> None:
            del history, tools, kwargs

    a = Agent(
        model=StubModel(),
        tools=[],
        compactor=_AssistantTerminatedCompactor(),
        session_dir=tmp_path,
    )
    a.runtime.history.append(UserMessage(text="x"))
    await a.compact_now()
    # The continuation user-message terminator was appended.
    last = a.runtime.history[-1]
    assert isinstance(last, UserMessage)
    assert last.text == "[continuation]"
    # Pre-compact transcript was written.
    assert (tmp_path / "pre_compact_0.jsonl").exists()


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
            history: list[HistoryEntry],
            model: object,
            transcript_path: object = None,
            direction: str = "from",
            keep_recent: int | None = None,
            custom_instructions: str | None = None,
            summary_pointers: object = None,
        ) -> list[HistoryEntry]:
            del history, model, transcript_path, direction, keep_recent
            del custom_instructions, summary_pointers
            return [UserMessage(text="[summary]")]

        def maintain(
            self, history: list[HistoryEntry], tools: object, **kwargs: object
        ) -> None:
            del history, tools, kwargs

    async def _boom(**kwargs: object) -> None:
        del kwargs
        raise RuntimeError("enrich failed")

    a = Agent(
        model=StubModel(),
        tools=[],
        compactor=_OkCompactor(),
        session_dir=tmp_path,
    )
    a.runtime.history.append(UserMessage(text="x"))

    monkeypatch.setattr("sagent.agent.agent.post_compact_enrich", _boom)
    await a.compact_now()

    # Summary still survived; the enrich failure was swallowed.
    assert any(
        isinstance(e, UserMessage) and e.text == "[summary]" for e in a.runtime.history
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
    async for _ in a.run(UserMessage(text="hi")):
        pass
    assert call_count > pre_run_count
    assert model.received[-1].system == f"sys-v{call_count}"


def test_subagent_inherits_root_cost_tracker() -> None:
    """C5: non-persistent subagent's cost folds into the root tracker."""
    root = _build_agent()
    child = _build_agent()
    response = ModelResponse(
        message=AssistantMessage(text="ok"),
        tokens=TokenCount(input_tokens=10, output_tokens=5),
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
    response = ModelResponse(
        message=AssistantMessage(text="ok"),
        tokens=TokenCount(),
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
    """H5: changing ``status`` publishes ``StatusChanged`` to observers."""
    a = _build_agent()
    events: list[str] = []

    def watch(event: object) -> None:
        if isinstance(event, StatusChanged):
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
            request: ModelRequest,
            on_text: object = None,
            on_thinking: object = None,
        ) -> ModelResponse:
            del on_text, on_thinking
            self.received.append(request)
            stream_entered.set()
            await gate.wait()
            return ModelResponse(
                message=AssistantMessage(text="from A"), total_cost=0.10
            )

    model_a = GatedModel(model_id="model-A")
    model_b = StubModel(model_id="model-B")
    agent = _build_agent(model=model_a)

    async def consume() -> None:
        async for _ in agent.run(UserMessage(text="hi")):
            pass

    drive = asyncio.create_task(consume())
    await asyncio.wait_for(stream_entered.wait(), timeout=1.0)

    # User types ``/model model-B``. Slash handler pushes a
    # ``ModelSwitch`` event into the inbox; runtime defers it.
    agent.runtime.inbox.push_back(
        RuntimeModelSwitch(apply=lambda: agent.swap_model(model_b), label="A -> B"),
    )

    # Release the in-flight call.
    gate.set()
    await drive

    # Cost lands on model-A: the runtime didn't apply the swap until
    # AFTER ``ModelResponseComplete`` set ``model_call`` to None, which
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


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
