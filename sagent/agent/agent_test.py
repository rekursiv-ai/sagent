"""Tests for ``agent.agent``: Agent composition class."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import override

import asyncio
import contextlib

import pytest

from sagent.agent.agent import (
    ActivityTracker,
    Agent,
    SystemPromptArg,
)
from sagent.agent.background import (
    BackgroundAwareTool,
    BackgroundTaskEntry,
)
from sagent.agent.runtime import (
    AssistantMessage,
    Clear as RuntimeClear,
    Halt as RuntimeHalt,
    HistoryEntry,
    Kill as RuntimeKill,
    ModelCallStarted,
    ModelResponseComplete,
    ModelResponseError,
    ModelResponsePartial,
    ToolCall as RuntimeToolCall,
    ToolLabel,
    ToolResult,
    UserMessage,
)
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


def test_agent_tools_map_wraps_in_background_aware() -> None:
    tool: RichTool = StubTool()
    a = _build_agent(tools=[tool])
    assert "Echo" in a.tools_map
    wrapped = a.tools_map["Echo"]
    assert isinstance(wrapped, BackgroundAwareTool)
    assert wrapped.name == "Echo"


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
        del request, on_text, on_thinking
        idx = self.call_index
        self.call_index += 1
        if idx < self.overflow_count:
            raise PromptTooLongError("too long")
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


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
