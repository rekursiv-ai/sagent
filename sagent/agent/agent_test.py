"""Tests for ``agent.agent``: Agent composition class."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal, cast, override
from unittest.mock import MagicMock, Mock, patch

import asyncio
import contextlib
import json
import logging
import re
import threading
import time

import pytest

from sagent import (
    providers as providers_module,
    types,
)
from sagent.agent import runtime as agent_runtime
from sagent.agent.agent import (
    ActivityTracker,
    Agent,
    SystemPromptArg,
    _AgentCompactor,
    _AgentTool,
    _repair_compact_payload,
    _resolve_target_spec,
    _should_cancel_background,
)
from sagent.agent.background import (
    BackgroundAwareTool,
    BackgroundTaskEntry,
    split_bg_args,
)
from sagent.agent.context import validate_context
from sagent.agent.session_io import append_session, load_session
from sagent.agent.state import (
    AgentLike,
    ToolState,
    agent_registry,
    tool_state_context,
)
from sagent.lib import last_models, token_count
from sagent.lib.json import JSON, json_freeze
from sagent.providers import Google
from sagent.tools.read import Read
from sagent.types.compactor import CompactRestorable
from sagent.types.tape import (
    ContextSplice,
    TapeRecord,
    TapeRef,
    full_tape_mask,
)


def _summary_override(
    summary: list[types.runtime.ModelContextEvent],
    mint_ref: Callable[[], TapeRef],
    *,
    tape: Sequence[TapeRecord] | None = None,
    strategy: str = "summary",
    fallback_reason: str = "",
    preserved_tail_count: int = 0,
    token_before: int = 0,
    token_after: int = 0,
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
        token_before=token_before,
        token_after=token_after,
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
    valid_efforts: tuple[str, ...] = ()
    supports_cache_control: bool = False
    valid_service_tiers: tuple[str, ...] = ()
    valid_latency_modes: tuple[str, ...] = ()
    supports_context_management: bool = False
    supports_persistent_retry: bool = False
    supports_account_auth: bool = False
    max_image_dim: int = 8_000
    max_image_bytes: int = 5 * 1024 * 1024
    responses: list[types.runtime.AssistantMessage] = field(default_factory=list)
    received: list[types.model.ModelRequest] = field(default_factory=list)

    @property
    def valid_thinking_states(self) -> tuple[str, ...]:
        return (
            ("adaptive-hide", "on-hide", "off-hide")
            if self.supports_thinking
            else ("off-hide",)
        )

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
            else types.runtime.AssistantMessage(text="ok")
        )
        return types.model.ModelResponse(message=msg)


_STUB_SCHEMA: JSON = json_freeze({"type": "object"})
_STRING_SCHEMA: JSON = json_freeze({"type": "string"})
_TYPELESS_SCHEMA: JSON = json_freeze({})


@dataclass(slots=True, kw_only=True)
class StubTool:
    """Minimal tool that records calls."""

    name: str = "Echo"
    tool_id: str = "application/x-tool-echo"
    description: str = "Echo tool."
    directive_schema: JSON = _STUB_SCHEMA
    clearable_results: bool = False
    response: str | None = None
    calls: list[Mapping[str, object]] = field(default_factory=list)

    def summary(self, args: Mapping[str, object]) -> str:
        del args
        return "echo"

    def summary_result(self, result: types.runtime.ToolResult) -> str | None:
        del result
        return None

    def prompt(self) -> str:
        return ""

    def serialize_key(self, args: Mapping[str, object]) -> str | None:
        del args
        return None

    async def run(self, args: Mapping[str, object]) -> types.runtime.ToolResult:
        self.calls.append(args)
        content = (
            self.response if self.response is not None else str(args.get("msg", ""))
        )
        return types.runtime.ToolResult(call_id="", content=content)


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


@pytest.mark.asyncio
async def test_agent_tool_injects_conditional_agents_md_rule(tmp_path: Path) -> None:
    rules = tmp_path / ".sagent" / "rules"
    rules.mkdir(parents=True)
    _ = (rules / "python.md").write_text(
        "---\npaths: ['**/*.py']\n---\nUse Python rule.\n"
    )
    target = tmp_path / "main.py"
    target.write_text("print('hi')\n")
    agent = _build_agent(tools=[Read()])
    agent.tool_state.bash_cwd = str(tmp_path)
    wrapped = _AgentTool(Read(), agent)
    with tool_state_context(agent.tool_state):
        result = await wrapped.run({"file_path": str(target)})
        repeated = await wrapped.run({"file_path": str(target)})
    assert "<system-reminder>" in result.content
    assert "Use Python rule." in result.content
    assert "Use Python rule." not in repeated.content


@pytest.mark.asyncio
async def test_agent_model_stream_materializes_request() -> None:
    call = types.runtime.ToolCall(id="call_1", name="Bash", args={})
    model = StubModel()
    budget = types.model.ContextBudget(
        max_request_tokens=100_000,
        max_response_tokens=1_024,
        message_budget_chars=10,
    )
    agent = Agent(model=model, budget=budget)

    agent.runtime.append_history(types.runtime.UserMessage(text="start"))
    agent.runtime.append_history(types.runtime.AssistantMessage(tool_calls=(call,)))
    agent.runtime.append_history(
        types.runtime.ToolResult(call_id="call_1", content="x" * 1_000)
    )
    _ = await agent._agent_model.stream(
        agent.runtime.context().messages,
        lambda _: None,
        lambda _: None,
    )
    result = model.received[-1].messages[2]
    assert isinstance(result, types.runtime.ToolResult)
    assert "<elided>" in result.content


def test_agent_budget_override_respected() -> None:
    b = types.model.ContextBudget.from_model(StubModel())
    a = _build_agent(budget=b)
    assert a.budget is b


def test_live_tool_result_chars_counts_read_results() -> None:
    """Read tool-result chars belong in the wire-budget tally.

    ``live_tool_result_chars`` previously skipped Read results (likely
    copy-pasted from ``PERSIST_EXEMPT_TOOLS``), but Read content still
    crosses the wire and consumes the message budget. Skipping it lets
    Read-heavy contexts evade message-budget compaction.
    """
    a = _build_agent()
    read_call = types.runtime.ToolCall(id="read-1", name="Read", args={})
    bash_call = types.runtime.ToolCall(id="bash-1", name="Bash", args={})
    a.runtime.append_history(types.runtime.UserMessage(text="go"))
    a.runtime.append_history(
        types.runtime.AssistantMessage(tool_calls=(read_call, bash_call))
    )
    a.runtime.append_history(
        types.runtime.ToolResult(call_id="read-1", content="r" * 200)
    )
    a.runtime.append_history(
        types.runtime.ToolResult(call_id="bash-1", content="b" * 50)
    )
    assert a.live_tool_result_chars() == 250


def test_agent_like_protocol_exposes_kill_all_tools() -> None:
    """Regression for H10: ``AgentLike`` advertises ``kill_all_tools``.

    Both the real ``Agent`` (``agent.py``) and the test ``FakeAgent``
    (``testing.py``) implement ``kill_all_tools``; callers route the
    verb via the Protocol (e.g. ``repl/input_pane.py``). The Protocol
    must include the method so static analysis matches reality.
    """
    a = _build_agent()
    via_protocol: AgentLike = a
    # The Protocol method is present on the structural type; callers
    # depend on this attribute being typed.
    assert callable(via_protocol.kill_all_tools)


def test_persist_budget_used_chars_excludes_persist_exempt_tools() -> None:
    """Persist-exempt tool results don't inflate the persist budget.

    Regression for AGENT-REVIEW-005: feeding ``post_process_result``
    with the live-wire tally inflated ``used_message_chars`` by Read's
    bytes, prematurely forcing unrelated (Bash) results to disk.
    """
    a = _build_agent()
    read_call = types.runtime.ToolCall(id="read-1", name="Read", args={})
    bash_call = types.runtime.ToolCall(id="bash-1", name="Bash", args={})
    err_call = types.runtime.ToolCall(id="bash-err", name="Bash", args={})
    a.runtime.append_history(types.runtime.UserMessage(text="go"))
    a.runtime.append_history(
        types.runtime.AssistantMessage(
            tool_calls=(read_call, bash_call, err_call),
        )
    )
    a.runtime.append_history(
        types.runtime.ToolResult(call_id="read-1", content="r" * 200)
    )
    a.runtime.append_history(
        types.runtime.ToolResult(call_id="bash-1", content="b" * 50)
    )
    a.runtime.append_history(
        types.runtime.ToolResult(call_id="bash-err", content="e" * 30, is_error=True)
    )
    # Only the non-error Bash result occupies the persist budget.
    assert a.persist_budget_used_chars() == 50


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


def test_agent_assigns_stable_job_ids_to_detached_call_ids() -> None:
    a = _build_agent()
    loop = asyncio.new_event_loop()
    task = loop.create_task(asyncio.sleep(0))
    try:
        a.runtime.detached["call-provider"] = task
        a._tool_registry["call-provider"] = ("Bash", 123.0)
        first = a.background
        second = a.background
        assert list(first) == ["job-1"]
        assert list(second) == ["job-1"]
        assert first["job-1"].queue_id == "job-1"
        assert first["job-1"].call_id == "call-provider"
    finally:
        _ = task.cancel()
        loop.run_until_complete(asyncio.gather(task, return_exceptions=True))
        loop.close()


@pytest.mark.asyncio
async def test_public_cancel_background_accepts_job_id_for_detached_task() -> None:
    a = _build_agent()
    task = asyncio.create_task(asyncio.sleep(60))
    a.runtime.detached["call-provider"] = task
    a._tool_registry["call-provider"] = ("Bash", 0.0)
    job_id = next(iter(a.background))
    assert job_id == "job-1"
    a.cancel_background(job_id)
    assert "call-provider" not in a.runtime.detached
    with contextlib.suppress(asyncio.CancelledError):
        await task
    assert task.cancelled()


@pytest.mark.asyncio
async def test_public_cancel_background_actually_cancels_task() -> None:
    """The public ``cancel_background`` must cancel the underlying task.

    Pre-fix, ``cancel_background`` only popped the registry entry while
    the private ``_cancel_background`` carried the actual ``task.cancel()``.
    Two near-identical names with opposite behavior; callers reading the
    public API expected cancellation and got registry removal.
    """
    a = _build_agent()
    task = asyncio.create_task(asyncio.sleep(10))
    a.register_background(
        "j",
        BackgroundTaskEntry(task=task, tool_name="x", queue_id="j", started=0.0),
    )
    a.cancel_background("j")
    assert task.cancelling() > 0
    assert "j" not in a.background


@pytest.mark.asyncio
async def test_cancel_background_tool_kind_forgets_job_id_mapping() -> None:
    """A tool-kind cancel must clear the call_id<->job_id mapping.

    BUG-1: ``cancel_background`` only invoked ``_forget_job_id`` on the
    detached branch. A ``kind="tool"`` entry cancelled this way left a
    permanent entry in ``_job_ids_by_call_id`` / ``_call_ids_by_job_id``,
    poisoning later job-id minting for fresh calls reusing the same
    provider call id. The symmetric ``forget_background`` already did
    this for the tool kind, exposing the asymmetry.
    """
    a = _build_agent()
    task = asyncio.create_task(asyncio.sleep(10))
    a.register_background(
        "j",
        BackgroundTaskEntry(
            task=task,
            tool_name="x",
            queue_id="j",
            started=0.0,
            kind="tool",
            call_id="call-x",
        ),
    )
    # Seed the job-id<->call-id map as the live path does at registration.
    _ = a.job_id_for_call("call-x")
    assert "call-x" in a._job_ids_by_call_id

    a.cancel_background("j")

    assert "call-x" not in a._job_ids_by_call_id
    assert task.cancelling() > 0


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


def test_agent_tools_map_preserves_compact_restorable_identity() -> None:
    """D1/CV2: a ``CompactRestorable`` tool keeps its Protocol identity.

    ``BackgroundAwareTool`` proxies attribute reads but cannot satisfy
    Protocol ``isinstance`` checks for methods it doesn't redeclare
    (Python's protocol-isinstance walks the class dict, not
    ``__getattr__``). The agent's structural fix keeps the *raw* tool
    in ``tools_map`` so consumer sites' ``isinstance(t,
    CompactRestorable)`` lookups still hit; per-request wrapping in
    ``_AgentModel.stream`` injects the BG schema without touching the
    stored tool. Regression-guard the contract here.
    """

    @dataclass(slots=True, kw_only=True)
    class _RestorableStub(StubTool):
        async def post_compact_restore(
            self,
            history: list[types.runtime.ModelContextEvent],
            tool_state: object,
            *,
            budget_chars: int = 100_000,
        ) -> None:
            del history, tool_state, budget_chars

    tool = _RestorableStub()
    assert isinstance(tool, CompactRestorable)
    a = _build_agent(tools=[tool])
    stored = a.tools_map["Echo"]
    assert isinstance(stored, CompactRestorable), (
        "raw rich tool must satisfy CompactRestorable; the wrapper would not."
    )


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
    async for _ in a.run(types.runtime.UserMessage(text="hi")):
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
        type(ev).__name__ async for ev in a.run(types.runtime.UserMessage(text="ping"))
    ]
    assert "ModelIdle" in events
    assert len(a.history) >= 2
    assert isinstance(a.history[0], types.runtime.UserMessage)


@pytest.mark.asyncio
async def test_agent_tool_persists_with_runtime_call_id(tmp_path: Path) -> None:
    call = types.runtime.ToolCall(id="tool_call_1", name="Echo", args={})
    model = StubModel(
        responses=[
            types.runtime.AssistantMessage(tool_calls=(call,)),
            types.runtime.AssistantMessage(text="done"),
        ]
    )
    budget = replace(
        types.model.ContextBudget.from_model(model),
        persist_threshold=1_000,
    )
    tool = StubTool(response="X" * 5_000)
    a = _build_agent(model=model, tools=[tool], budget=budget, session_dir=tmp_path)

    async for _ in a.run(types.runtime.UserMessage(text="hi")):
        pass

    assert (tmp_path / "tool-results" / "tool_call_1.txt").read_text() == "X" * 5_000
    assert not (tmp_path / "tool-results" / "id_e3b0c44298fc1c14.txt").exists()


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
    async for _ in a.run(types.runtime.UserMessage(text="hi")):
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
    # B1: the wrapper injects ``background`` / ``delay`` even into
    # schemaless tools; check the underlying ``type`` is preserved and
    # the injected fields are present.
    schema = dict(seen.directive_schema)
    assert schema["type"] == "object"
    props = cast(Mapping[str, object], schema["properties"])
    assert "background" in props
    assert "delay" in props
    assert seen.summary({}) == "echo"
    assert seen.prompt() == ""


@pytest.mark.asyncio
async def test_agent_records_response_into_cost_tracker() -> None:
    model = StubModel()
    a = _build_agent(model=model)
    async for _ in a.run(types.runtime.UserMessage(text="ping")):
        pass
    # StubModel emits empty types.model.TokenCount, but ``calls_by_model`` records
    # one entry per model invocation.
    assert a.cost_tracker.calls_by_model.get("stub-1", 0) >= 1


def test_agent_record_response_budget_exhaustion_raises() -> None:
    a = _build_agent(max_budget_usd=1.0)
    # First response below the cap: clean.
    a.record_response(
        types.model.ModelResponse(message=types.runtime.AssistantMessage(text="x"))
    )
    # Force an over-budget total and verify the next call raises a
    # ``UserFacingError`` subclass (polished remediation, not raw
    # ``RuntimeError``) so the REPL renderer surfaces it cleanly.
    a.cost_tracker.total_cost_usd = 2.0
    with pytest.raises(types.exceptions.BudgetExhaustedError) as exc_info:
        a.record_response(
            types.model.ModelResponse(message=types.runtime.AssistantMessage(text="x"))
        )
    assert exc_info.value.max_budget_usd == 1.0
    assert "Budget exhausted" in str(exc_info.value)


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


def test_system_prompt_rejects_dict() -> None:
    with pytest.raises(TypeError, match=r"system.*str or Callable"):
        _build_agent(system=cast(SystemPromptArg, {"static": "literal"}))


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


def test_restore_thinking_state_writes_all_three_fields() -> None:
    """Public rollback API for a failed thinking-mode rebuild."""
    a = _build_agent()
    a.set_thinking_state("adaptive-show")
    a.restore_thinking_state("off-hide", None, False)
    assert a.thinking_state == "off-hide"
    assert a.thinking is None
    assert a.show_thinking is False
    a.restore_thinking_state("on-show", "enabled", True)
    assert a.thinking_state == "on-show"
    assert a.thinking == "enabled"
    assert a.show_thinking is True


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
    async for _ in a.run(types.runtime.UserMessage(text="hi")):
        pass
    assert model.received[-1].thinking is None


@pytest.mark.asyncio
async def test_agent_thinking_state_sets_request_thinking() -> None:
    model = StubModel(supports_thinking=True)
    a = Agent(model=model, tools=[], thinking_state="adaptive-hide")
    async for _ in a.run(types.runtime.UserMessage(text="hi")):
        pass
    assert model.received[-1].thinking == "adaptive"


def test_effort_setter_rejects_when_model_lacks_support() -> None:
    a = _build_agent()  # StubModel.supports_effort = False
    with pytest.raises(ValueError, match="does not support effort"):
        a.effort = "high"


def test_effort_setter_accepts_when_model_supports() -> None:
    model = StubModel(supports_effort=True, valid_efforts=("low", "medium", "high"))
    a = _build_agent(model=model)
    a.effort = "medium"
    assert a.effort == "medium"


def test_effort_setter_rejects_value_outside_valid_efforts() -> None:
    model = StubModel(supports_effort=True, valid_efforts=("low", "high"))
    a = _build_agent(model=model)
    with pytest.raises(ValueError, match="effort must be one of"):
        a.effort = "medium"


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


def test_latency_setter_rejects_when_model_lacks_support() -> None:
    a = _build_agent()  # StubModel.valid_latency_modes = ()
    with pytest.raises(ValueError, match="does not support latency"):
        a.latency = "fast"


def test_latency_setter_rejects_unknown_value() -> None:
    model = StubModel(valid_latency_modes=("fast",))
    a = _build_agent(model=model)
    with pytest.raises(ValueError, match="latency must be one of"):
        a.latency = "turbo"


def test_latency_setter_accepts_when_model_supports() -> None:
    model = StubModel(valid_latency_modes=("fast",))
    a = _build_agent(model=model)
    a.latency = "fast"
    assert a.latency == "fast"
    a.latency = None
    assert a.latency is None


def test_swap_model_clears_unsupported_latency() -> None:
    a = _build_agent(model=StubModel(valid_latency_modes=("fast",)))
    a.latency = "fast"
    a.swap_model(StubModel(model_id="stub-2"))
    assert a.latency is None


def test_status_setter_round_trip() -> None:
    a = _build_agent()
    a.status = "busy"
    assert a.status == "busy"


def test_ephemeral_session_id_is_hex_string() -> None:
    a = _build_agent()
    assert len(a.session_id) == 8
    assert a.runtime.session_id == a.session_id


def test_persisted_session_uses_session_directory_name(tmp_path: Path) -> None:
    session_dir = tmp_path / "d940b751c9fe"
    a = _build_agent(session_dir=session_dir)
    a.runtime.append_history(types.runtime.UserMessage(text="hello"))

    assert a.session_id == "d940b751c9fe"
    assert a.runtime.session_id == "d940b751c9fe"
    assert a.runtime.tape[0].ref.session_id == "d940b751c9fe"


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


def test_system_prompt_adds_detached_arrived_note_only_when_in_history() -> None:
    """The ``DetachedArrived`` note appears once such a turn is in context."""
    a = _build_agent(system="root")
    # No detached activity yet -> lean prompt, no note.
    assert types.runtime.DETACHED_ARRIVED_TOOL not in a.system
    # A synthesized DetachedArrived turn in history -> the note is added.
    a.runtime.append_history(
        types.runtime.AssistantMessage(
            tool_calls=(
                types.runtime.ToolCall(
                    id="x:detached", name=types.runtime.DETACHED_ARRIVED_TOOL, args={}
                ),
            ),
        ),
    )
    assert types.runtime.DETACHED_ARRIVED_SYSTEM_NOTE in a.system


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
            "job-2",
            BackgroundTaskEntry(
                task=ex_task, tool_name="X", queue_id="job-2", started=0.0
            ),
        )
        merged = a.background
        assert "job-1" in merged
        assert "job-2" in merged
        assert merged["job-1"].kind == "detached"
        assert merged["job-1"].call_id == "det-1"
        _ = det_task.cancel()
        _ = ex_task.cancel()
        loop.run_until_complete(
            asyncio.gather(det_task, ex_task, return_exceptions=True),
        )
    finally:
        loop.close()


def test_swap_model_clamps_whole_window_budget_to_smaller_model() -> None:
    # Default budget == old model's max (the "whole window" case). Swapping
    # to a smaller model clamps the budget down instead of raising.
    a = _build_agent()
    assert a.budget.max_request_tokens == a.model.max_request_tokens
    a.swap_model(StubModel(model_id="small", max_request_tokens=50_000))
    assert a.budget.max_request_tokens == 50_000


def test_swap_model_grows_whole_window_budget_to_larger_model() -> None:
    # A budget pinned at the old model's max follows the new model up.
    a = _build_agent()
    assert a.budget.max_request_tokens == a.model.max_request_tokens
    a.swap_model(StubModel(model_id="big", max_request_tokens=1_000_000))
    assert a.budget.max_request_tokens == 1_000_000


def test_swap_model_preserves_pinned_budget_under_new_max() -> None:
    # An explicitly lowered budget (below the old model's max) is preserved
    # when it still fits the new model.
    a = _build_agent()
    a.max_request_tokens = 50_000
    a.swap_model(StubModel(model_id="big", max_request_tokens=1_000_000))
    assert a.budget.max_request_tokens == 50_000


def test_swap_model_clamps_pinned_budget_over_new_max() -> None:
    # An explicit budget that exceeds the new model's max is clamped down.
    a = _build_agent()
    a.max_request_tokens = 80_000
    a.swap_model(StubModel(model_id="small", max_request_tokens=50_000))
    assert a.budget.max_request_tokens == 50_000


def test_swap_model_rescales_response_window_to_smaller_model() -> None:
    a = _build_agent()
    assert a.budget.max_response_tokens == a.model.max_response_tokens
    a.swap_model(StubModel(model_id="small", max_response_tokens=256))
    assert a.budget.max_response_tokens == 256


@dataclass(slots=True, kw_only=True)
class _NoopCompactor:
    def should_compact(
        self,
        current_tokens: int,
        max_request_tokens: int,
        system_tokens: int = 0,
    ) -> bool:
        del current_tokens, max_request_tokens, system_tokens
        return False

    async def compact(
        self,
        tape: Sequence[TapeRecord],
        context: Sequence[types.runtime.ModelContextEvent],
        model: types.model.Model,
        mint_ref: Callable[[], TapeRef],
        custom_instructions: str | None = None,
    ) -> ContextSplice:
        del context, model, custom_instructions
        return _summary_override(
            [types.runtime.UserMessage(text="ok")],
            mint_ref,
            tape=tape or None,
        )


def test_swap_model_pushes_compact_when_history_exceeds_new_budget() -> None:
    """Swapping to a smaller-window model with oversized history triggers compact.

    The /model verb's whole purpose is rescuing a session whose
    current model wedged (rate limit, oversized history). Just
    rescaling the budget without compacting leaves the next provider
    call to overflow against the new (smaller) model -- the user is
    no better off. After a swap whose rescaled budget cannot hold the
    current resolved view, push a ``Compact()`` so the agent layer's
    bridge can fit history before the next stream call.
    """

    @dataclass(slots=True, kw_only=True)
    class _LyingTokenModel(StubModel):
        """Returns 500k tokens regardless of actual request size."""

        @override
        def approx_request_tokens(self, request: types.model.ModelRequest) -> int:
            del request
            return 500_000

    a = Agent(
        model=_LyingTokenModel(model_id="big", max_request_tokens=1_000_000),
        tools=[],
        compactor=_NoopCompactor(),
    )
    a.runtime.inbox.drain_nowait()  # clear any startup events
    a.runtime.append_history(types.runtime.UserMessage(text="payload"))

    # Swap to a smaller model that also reports 500k tokens for the
    # current request -- 500k > (100k - 1024 - small buffer) so the
    # post-swap budget cannot hold this history.
    a.swap_model(_LyingTokenModel(model_id="small", max_request_tokens=100_000))

    items = a.runtime.inbox.drain_nowait()
    compacts = [ev for ev in items if isinstance(ev, types.runtime.Compact)]
    assert compacts, (
        f"swap_model to smaller model did not push Compact(); inbox was {items!r}"
    )


def test_swap_model_does_not_push_compact_when_history_fits() -> None:
    """Swap to a model whose budget comfortably holds current history: no compact."""
    a = Agent(model=StubModel(), tools=[], compactor=_NoopCompactor())
    a.runtime.inbox.drain_nowait()

    a.swap_model(StubModel(model_id="same-size"))

    items = a.runtime.inbox.drain_nowait()
    compacts = [ev for ev in items if isinstance(ev, types.runtime.Compact)]
    assert not compacts, (
        f"swap_model to fitting model spuriously pushed Compact(); inbox was {items!r}"
    )


def test_swap_model_no_compact_when_no_compactor_configured() -> None:
    """No compactor wired: swap_model must not push Compact()."""

    @dataclass(slots=True, kw_only=True)
    class _LyingTokenModel(StubModel):
        @override
        def approx_request_tokens(self, request: types.model.ModelRequest) -> int:
            del request
            return 500_000

    a = Agent(
        model=_LyingTokenModel(model_id="big", max_request_tokens=1_000_000),
        tools=[],
    )
    a.runtime.inbox.drain_nowait()
    a.runtime.append_history(types.runtime.UserMessage(text="payload"))

    a.swap_model(_LyingTokenModel(model_id="small", max_request_tokens=100_000))

    items = a.runtime.inbox.drain_nowait()
    compacts = [ev for ev in items if isinstance(ev, types.runtime.Compact)]
    assert not compacts


def test_swap_model_replaces_model_and_inner_wrapper() -> None:
    a = _build_agent()
    new = StubModel(model_id="stub-2")
    a.swap_model(new)
    assert a.model is new
    # The wrapper's inner reference was updated too.
    assert a._agent_model._inner is new


@pytest.mark.asyncio
async def test_swap_model_noop_does_not_close_active_model() -> None:
    """Swapping in the *current* model must not tear it down.

    Pre-fix, ``swap_model(self.model)`` captured ``old = self.model`` then
    assigned ``self.model = model`` (same object), then unconditionally
    scheduled ``close(old)`` -- so the now-active model got torn down,
    leaving subsequent requests to fail against a dead SDK client.
    """

    @dataclass(slots=True, kw_only=True)
    class ClosableStubModel(StubModel):
        closed_event: asyncio.Event = field(default_factory=asyncio.Event)

        async def close(self) -> None:
            self.closed_event.set()

    m = ClosableStubModel()
    a = _build_agent(model=m)
    a.swap_model(m)
    # Give any scheduled close a chance to run; under the bug the
    # Event would be set after the next loop step.
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert not m.closed_event.is_set()
    assert a.model is m


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
async def test_relogin_runs_blocking_login_off_event_loop() -> None:
    """``login`` blocks (browser/``input()`` wait); it must not freeze the loop.

    Running it inline would wedge the single-threaded REPL: the input
    pump could not drain and a stuck auth would freeze the session.
    Assert the event loop keeps making progress while ``login`` blocks.
    """
    a = _build_agent_with_spec()
    login_entered = threading.Event()
    release = threading.Event()
    loop_progressed = False

    def _blocking_login() -> None:
        login_entered.set()
        # Hold the worker thread until the event loop proves it advanced.
        assert release.wait(timeout=5.0)

    fake_provider_cls = MagicMock()
    fake_provider_cls.login = _blocking_login

    async def _drive() -> None:
        nonlocal loop_progressed
        await asyncio.to_thread(login_entered.wait, 5.0)
        # The loop scheduled and ran this coroutine while login blocks.
        loop_progressed = True
        release.set()

    with patch.object(providers_module, "Anthropic", fake_provider_cls, create=True):
        await asyncio.gather(a.relogin(), _drive())

    assert loop_progressed


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
async def test_relogin_clears_suspension_and_halts_inflight_call() -> None:
    """``relogin`` recovers a call wedged in a service-suspension sleep.

    The retry loop sleeps uninterruptibly until ``retry_at``; without this,
    fresh credentials sit unused for the rest of that wait. ``relogin`` must
    clear the stale suspension timestamps and ``Halt`` the live call so the
    user regains control immediately.
    """
    a = _build_agent_with_spec()
    a.runtime.service_suspended_until = time.time() + 15_000.0
    a.runtime.resume_retry_at = time.time() + 15_000.0

    async def _never() -> None:
        await asyncio.sleep(3600)

    a.runtime.model_call = asyncio.ensure_future(_never())
    try:
        fake_cls = MagicMock()
        fake_cls.login = MagicMock()
        with patch.object(providers_module, "Anthropic", fake_cls, create=True):
            await a.relogin()
        assert a.runtime.service_suspended_until is None
        assert a.runtime.resume_retry_at is None
        items = a.runtime.inbox.drain_nowait()
        assert any(isinstance(it, types.runtime.Halt) for it in items)
    finally:
        a.runtime.model_call.cancel()


@pytest.mark.asyncio
async def test_relogin_no_halt_when_idle() -> None:
    """With no in-flight call, ``relogin`` clears suspension but queues no Halt."""
    a = _build_agent_with_spec()
    a.runtime.service_suspended_until = time.time() + 100.0
    fake_cls = MagicMock()
    fake_cls.login = MagicMock()
    with patch.object(providers_module, "Anthropic", fake_cls, create=True):
        await a.relogin()
    assert a.runtime.service_suspended_until is None
    items = a.runtime.inbox.drain_nowait()
    assert not any(isinstance(it, types.runtime.Halt) for it in items)


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
async def test_shutdown_force_preserves_persistent_subagent() -> None:
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
            persistent_run_id="run-child",
        ),
    )
    agent_registry["child"] = cast(Agent, child)
    try:
        a.shutdown(force=True)
        await asyncio.sleep(0)
        assert child.shutdown_calls == []
        assert not task.cancelled()
        assert "child" in a.background
    finally:
        _ = agent_registry.pop("child", None)
        _ = task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_shutdown_force_preserves_missing_registry_persistent_subagent() -> None:
    a = _build_agent()

    async def hang() -> None:
        await asyncio.sleep(10.0)

    task = asyncio.create_task(hang())
    a.register_background(
        "child",
        BackgroundTaskEntry(
            task=task,
            tool_name="Agent",
            queue_id="missing-child",
            started=0.0,
            hidden=False,
            kind="persistent_subagent",
            persistent_run_id="run-missing",
        ),
    )
    try:
        a.shutdown(force=True)
        await asyncio.sleep(0)
        assert not task.cancelled()
        assert "child" in a.background
    finally:
        _ = task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_compact_awaits_compact_complete_event() -> None:
    @dataclass(slots=True, kw_only=True)
    class _StubCompactor:
        def should_compact(
            self,
            current_tokens: int,
            max_request_tokens: int,
            system_tokens: int = 0,
        ) -> bool:
            del current_tokens, max_request_tokens, system_tokens
            return False

        async def compact(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[types.runtime.ModelContextEvent],
            model: object,
            mint_ref: Callable[[], TapeRef],
            custom_instructions: str | None = None,
        ) -> ContextSplice:
            del context, model
            del custom_instructions
            return _summary_override(
                [types.runtime.UserMessage(text="[summary]")], mint_ref, tape=tape
            )

        def maintain(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[types.runtime.ModelContextEvent],
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
        isinstance(e, types.runtime.UserMessage) and e.text == "[summary]"
        for e in a.history
    )


@pytest.mark.asyncio
async def test_public_compact_returns_when_halt_cancels_compaction() -> None:
    started = asyncio.Event()

    @dataclass(slots=True, kw_only=True)
    class _HangingCompactor:
        def should_compact(
            self,
            current_tokens: int,
            max_request_tokens: int,
            system_tokens: int = 0,
        ) -> bool:
            del current_tokens, max_request_tokens, system_tokens
            return False

        async def compact(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[types.runtime.ModelContextEvent],
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
        def should_compact(
            self,
            current_tokens: int,
            max_request_tokens: int,
            system_tokens: int = 0,
        ) -> bool:
            del current_tokens, max_request_tokens, system_tokens
            return False

        async def compact(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[types.runtime.ModelContextEvent],
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
        def should_compact(
            self,
            current_tokens: int,
            max_request_tokens: int,
            system_tokens: int = 0,
        ) -> bool:
            del current_tokens, max_request_tokens, system_tokens
            return False

        async def compact(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[types.runtime.ModelContextEvent],
            model: object,
            mint_ref: Callable[[], TapeRef],
            custom_instructions: str | None = None,
        ) -> ContextSplice:
            del context, model
            del custom_instructions
            return _summary_override(
                [types.runtime.UserMessage(text="[recompacted]")], mint_ref, tape=tape
            )

        def maintain(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[types.runtime.ModelContextEvent],
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
        isinstance(e, types.runtime.UserMessage) and e.text == "[recompacted]"
        for e in a.history
    )


@pytest.mark.asyncio
async def test_clear_resets_file_tracking_and_awaits_completion() -> None:
    """``Agent.clear`` resolves after the runtime publishes ``ClearComplete``.

    Regression for AGENT-REVIEW-003: ``clear`` used to push the
    ``Clear`` event and return immediately, so callers could observe
    history that had not yet been wiped.
    """
    a = _build_agent()
    # Stage state that ``clear`` should reset.
    a.tool_state.mark_read("/tmp/x.txt")  # noqa: S108 -- placeholder
    assert a.tool_state.has_been_read("/tmp/x.txt")  # noqa: S108
    a.runtime.append_history(types.runtime.UserMessage(text="hello"))
    assert a.history

    drive_task = asyncio.create_task(a.serve_forever())
    try:
        await a.clear()
        assert not a.tool_state.has_been_read("/tmp/x.txt")  # noqa: S108
        assert a.history == []
    finally:
        a.shutdown()
        await drive_task


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
    tc = types.runtime.ToolCall(id="c1", name="Echo", args={})
    msg_with_tools = types.runtime.AssistantMessage(text="", tool_calls=(tc,))

    a.publish(types.runtime.ModelCallStarted())
    assert a.activity.active is True

    a.publish(types.runtime.ModelResponseComplete(message=msg_with_tools))
    assert a.activity.active is True, (
        "spinner should keep ticking while tools run; tool_calls in the "
        "response mean the cohort is about to fire"
    )

    # Tool result arrives.
    a.publish(types.runtime.ToolResult(call_id="c1", content="ok"))
    assert a.activity.active is True

    # Round 2 model call fires.
    a.publish(types.runtime.ModelCallStarted())
    assert a.activity.active is True

    # Round 2 has no tool calls; model truly idles.
    a.publish(
        types.runtime.ModelResponseComplete(
            message=types.runtime.AssistantMessage(text="done")
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
async def test_activity_pauses_during_model_service_suspended() -> None:
    """Suspension banks elapsed-so-far and pauses the timer until the next chunk."""
    a = _build_agent()
    loop = asyncio.get_running_loop()

    a.publish(types.runtime.ModelCallStarted())
    start = a.activity.current_call_start
    assert start > 0
    # Tick the loop's notion of time forward to accrue some elapsed.
    await asyncio.sleep(0.01)
    a.publish(
        types.runtime.ModelServiceSuspended(
            provider="anthropic",
            auth="key",
            account="default",
            model_id="claude-test",
            retry_at=loop.time() + 0.05,
            delay_sec=0.05,
            server_supplied=True,
            error=types.runtime.ServiceErrorSnapshot(
                type_name="RateLimitError", message="429", status=429
            ),
        )
    )
    banked = a.activity.elapsed_seconds
    assert banked > 0
    assert a.activity.current_call_start == 0.0
    assert a.activity.active is True

    await asyncio.sleep(0.05)
    a.publish(types.runtime.ModelResponsePartial(text="x"))
    assert a.activity.current_call_start > 0
    # No further banking yet; live timer has just resumed.
    assert a.activity.elapsed_seconds == banked

    await asyncio.sleep(0.01)
    a.publish(
        types.runtime.ModelResponseComplete(
            message=types.runtime.AssistantMessage(text="done")
        )
    )
    # Total elapsed should exclude the ~0.05s suspension sleep.
    assert a.activity.elapsed_seconds < 0.05


@pytest.mark.asyncio
@pytest.mark.real_sleep
async def test_run_bg_propagates_external_cancellation() -> None:
    """``_AgentTool._run_bg`` must re-raise ``CancelledError`` on outer cancel.

    Today the handler swallows ``CancelledError`` unconditionally: if
    ``job_id not in agent.background`` it returns; otherwise it posts a
    "[cancelled]" ``ToolResult`` and returns. Neither path re-raises.
    An asyncio task that catches ``CancelledError`` without re-raising
    breaks the cancel chain -- the parent (e.g. event-loop shutdown)
    thinks the child finished normally.

    Test: cancel the ``_run_bg`` task from outside while the job IS
    registered. The task must not exit normally -- it must propagate
    the cancellation.
    """
    tool_started = asyncio.Event()
    release = asyncio.Event()

    @dataclass(kw_only=True, slots=True)
    class _BlockingTool:
        name: str = "blocker"
        tool_id: str = "application/x-tool-blocker"
        description: str = ""
        directive_schema: JSON = _STUB_SCHEMA
        clearable_results: bool = False

        def summary(self, args: Mapping[str, object]) -> str:
            del args
            return ""

        def summary_result(self, result: types.runtime.ToolResult) -> str | None:
            del result
            return None

        def prompt(self) -> str:
            return ""

        def serialize_key(self, args: Mapping[str, object]) -> str | None:
            del args
            return None

        async def run(self, args: Mapping[str, object]) -> types.runtime.ToolResult:
            del args
            tool_started.set()
            await release.wait()  # never released
            return types.runtime.ToolResult(call_id="", content="unreached")

    a = _build_agent()
    wrapped = _AgentTool(_BlockingTool(), a)
    job_id = "job-bg-1"
    task = asyncio.create_task(wrapped._run_bg("call-1", job_id, {}, 0.0))
    a.register_background(
        job_id,
        BackgroundTaskEntry(
            task=task,
            tool_name="blocker",
            queue_id=job_id,
            call_id="call-1",
            started=0.0,
            kind="tool",
        ),
    )
    await tool_started.wait()
    # External cancel while job_id is still registered.
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
@pytest.mark.real_sleep
async def test_activity_current_call_start_resets_on_each_model_call() -> None:
    """Mid-chain ``ModelCallStarted`` must reset ``current_call_start``.

    Round chain: ``ModelCallStarted`` (call 1) → tools → ``ModelCallStarted``
    (call 2). The status pane reads ``now - current_call_start`` to decide
    whether to surface "waiting on model.". If ``current_call_start`` is
    not reset on call 2, the pane shows the entire chain's elapsed time,
    not the current call's -- "waiting on model." appears the instant
    call 2 starts even if it just began.
    """
    a = _build_agent()
    a.publish(types.runtime.ModelCallStarted())
    t1 = a.activity.current_call_start
    assert t1 > 0
    await asyncio.sleep(0.02)
    # Mid-chain tool-bearing response: spinner keeps ticking; the next
    # ``ModelCallStarted`` is the start of a new call.
    a.publish(
        types.runtime.ModelResponseComplete(
            message=types.runtime.AssistantMessage(
                text="",
                tool_calls=(types.runtime.ToolCall(id="t1", name="Bash", args={}),),
            )
        )
    )
    await asyncio.sleep(0.02)
    a.publish(types.runtime.ModelCallStarted())
    t2 = a.activity.current_call_start
    assert t2 > t1, (
        "second ModelCallStarted must restamp current_call_start so the"
        " status pane measures the current call's age, not the whole"
        f" round chain; got t1={t1!r} t2={t2!r}"
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
    tc = types.runtime.ToolCall(id="c1", name="Echo", args={})
    msg = types.runtime.AssistantMessage(text="", tool_calls=(tc,))
    a.publish(types.runtime.ModelResponseComplete(message=msg))
    assert a._tool_registry["c1"][0] == "Echo"
    assert a.activity.num_tool_call_rounds == 1


def test_enforce_caps_pushes_error_when_limit_reached() -> None:
    """``_enforce_caps`` posts a types.runtime.ModelResponseError when rounds cap is hit."""
    a = Agent(model=StubModel(), tools=[], max_tool_call_rounds=1)
    tc = types.runtime.ToolCall(id="c1", name="Echo", args={})
    msg = types.runtime.AssistantMessage(text="", tool_calls=(tc,))
    a.publish(types.runtime.ModelResponseComplete(message=msg))

    # Round count is now 1 == cap; the next observation triggers the
    # error push.
    a.publish(types.runtime.ModelResponseComplete(message=msg))
    items = asyncio.new_event_loop().run_until_complete(a.runtime.inbox.drain())
    assert any(isinstance(i, types.runtime.ModelResponseError) for i in items)


@pytest.mark.asyncio
async def test_compact_now_no_compactor_is_noop() -> None:
    a = _build_agent()
    a.runtime.append_history(types.runtime.UserMessage(text="x"))
    await a.compact_now()
    # History untouched: no compactor wired.
    assert len(a.runtime.context().messages) == 1


@pytest.mark.asyncio
async def test_compact_now_replaces_history_in_place() -> None:
    @dataclass(slots=True, kw_only=True)
    class _ReplaceCompactor:
        def should_compact(
            self,
            current_tokens: int,
            max_request_tokens: int,
            system_tokens: int = 0,
        ) -> bool:
            del current_tokens, max_request_tokens, system_tokens
            return False

        async def compact(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[types.runtime.ModelContextEvent],
            model: object,
            mint_ref: Callable[[], TapeRef],
            custom_instructions: str | None = None,
        ) -> ContextSplice:
            del context, model
            del custom_instructions
            return _summary_override(
                [types.runtime.UserMessage(text="[summary]")], mint_ref, tape=tape
            )

        def maintain(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[types.runtime.ModelContextEvent],
            tools: object,
            mint_ref: Callable[[], TapeRef],
        ) -> tuple[ContextSplice, ...]:
            del tape, context, tools, mint_ref
            return ()

    a = Agent(model=StubModel(), tools=[], compactor=_ReplaceCompactor())
    a.runtime.append_history(types.runtime.UserMessage(text="old"))
    await a.compact_now()
    assert len(a.runtime.context().messages) == 1
    entry = a.runtime.context().messages[0]
    assert isinstance(entry, types.runtime.UserMessage)
    assert entry.text == "[summary]"


@pytest.mark.asyncio
async def test_compact_recall_reset_waits_for_barrier_adoption(tmp_path: Path) -> None:
    @dataclass(slots=True, kw_only=True)
    class _NoopCompactor:
        def should_compact(
            self,
            current_tokens: int,
            max_request_tokens: int,
            system_tokens: int = 0,
        ) -> bool:
            del current_tokens, max_request_tokens, system_tokens
            return False

        async def compact(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[types.runtime.ModelContextEvent],
            model: object,
            mint_ref: Callable[[], TapeRef],
            custom_instructions: str | None = None,
        ) -> ContextSplice:
            del context, model, custom_instructions
            return _summary_override(
                [types.runtime.UserMessage(text="[summary]")], mint_ref, tape=tape
            )

    a = Agent(model=StubModel(), tools=[], compactor=_NoopCompactor())
    agent_compactor = a._agent_compactor
    assert agent_compactor is not None
    f = tmp_path / "foo.py"
    f.write_text("x")
    a.tool_state.mark_read(str(f), content="x")
    a.tool_state.invoked_skills.update({"alpha", "beta"})
    a.runtime.append_history(types.runtime.UserMessage(text="old"))

    override = await agent_compactor.compact(
        a.runtime.tape,
        a.runtime.context().messages,
        a._agent_model,
        a.runtime.mint_ref,
        "",
    )

    assert a.tool_state.read_cache
    assert a.tool_state.invoked_skills == {"alpha", "beta"}

    a.runtime.adopt_record(override)
    a.publish(types.runtime.CompactComplete(records=(override,)))

    assert a.tool_state.read_cache == {}
    assert a.tool_state.invoked_skills == set()


@pytest.mark.asyncio
async def test_compact_now_clears_tool_recall(tmp_path: Path) -> None:
    @dataclass(slots=True, kw_only=True)
    class _NoopCompactor:
        def should_compact(
            self,
            current_tokens: int,
            max_request_tokens: int,
            system_tokens: int = 0,
        ) -> bool:
            del current_tokens, max_request_tokens, system_tokens
            return False

        async def compact(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[types.runtime.ModelContextEvent],
            model: object,
            mint_ref: Callable[[], TapeRef],
            custom_instructions: str | None = None,
        ) -> ContextSplice:
            del context, model, custom_instructions
            return _summary_override(
                [types.runtime.UserMessage(text="[summary]")], mint_ref, tape=tape
            )

    a = Agent(model=StubModel(), tools=[], compactor=_NoopCompactor())
    f = tmp_path / "foo.py"
    f.write_text("x")
    a.tool_state.mark_read(str(f), content="x")
    a.tool_state.invoked_skills.update({"alpha", "beta"})
    assert a.tool_state.read_cache
    assert a.tool_state.invoked_skills == {"alpha", "beta"}

    a.runtime.append_history(types.runtime.UserMessage(text="old"))
    await a.compact_now()

    assert a.tool_state.read_cache == {}
    assert a.tool_state.invoked_skills == set()


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
        def should_compact(
            self,
            current_tokens: int,
            max_request_tokens: int,
            system_tokens: int = 0,
        ) -> bool:
            del current_tokens, max_request_tokens, system_tokens
            return False

        async def compact(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[types.runtime.ModelContextEvent],
            model: object,
            mint_ref: Callable[[], TapeRef],
            custom_instructions: str | None = None,
        ) -> ContextSplice:
            del context, model
            del custom_instructions
            return _summary_override(
                [types.runtime.UserMessage(text="[summary]")], mint_ref, tape=tape
            )

        def maintain(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[types.runtime.ModelContextEvent],
            tools: object,
            mint_ref: Callable[[], TapeRef],
        ) -> tuple[ContextSplice, ...]:
            del tape, context, tools, mint_ref
            return ()

    a = Agent(model=StubModel(), tools=[], compactor=_OkCompactor())
    a.runtime.append_history(types.runtime.UserMessage(text="old"))
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
        def should_compact(
            self,
            current_tokens: int,
            max_request_tokens: int,
            system_tokens: int = 0,
        ) -> bool:
            del current_tokens, max_request_tokens, system_tokens
            return False

        async def compact(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[types.runtime.ModelContextEvent],
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
            context: Sequence[types.runtime.ModelContextEvent],
            tools: object,
            mint_ref: Callable[[], TapeRef],
        ) -> tuple[ContextSplice, ...]:
            del tape, context, tools, mint_ref
            return ()

    a = Agent(model=StubModel(), tools=[], compactor=_BrokenCompactor())
    a.runtime.append_history(types.runtime.UserMessage(text="x"))
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
    a.runtime.append_history(types.runtime.UserMessage(text="x"))
    ok = await a.compact_now()
    assert ok is True


@dataclass(slots=True, kw_only=True)
class _ThresholdCompactor:
    """Recording compactor that compacts exactly when ``should_compact`` fires."""

    compacted: bool = False

    def should_compact(
        self,
        current_tokens: int,
        max_request_tokens: int,
        system_tokens: int = 0,
    ) -> bool:
        # Mirror SummaryCompactor (u=0.95, c=0.075):
        #   body >= u * (window - system) / (1 + c*u)
        u, c = 0.95, 0.075
        body = max(0, current_tokens - system_tokens)
        max_window = max(0, max_request_tokens - system_tokens)
        threshold = u * max_window / (1.0 + c * u)
        return body >= threshold

    async def compact(
        self,
        tape: Sequence[TapeRecord],
        context: Sequence[types.runtime.ModelContextEvent],
        model: object,
        mint_ref: Callable[[], TapeRef],
        custom_instructions: str | None = None,
    ) -> ContextSplice:
        del context, model, custom_instructions
        self.compacted = True
        return _summary_override(
            [types.runtime.UserMessage(text="[s]")], mint_ref, tape=tape
        )

    def maintain(
        self,
        tape: Sequence[TapeRecord],
        context: Sequence[types.runtime.ModelContextEvent],
        tools: object,
        mint_ref: Callable[[], TapeRef],
    ) -> tuple[ContextSplice, ...]:
        del tape, context, tools, mint_ref
        return ()


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
    history: list[types.runtime.ModelContextEvent] = [
        types.runtime.UserMessage(text="x")
    ]
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
        def should_compact(
            self,
            current_tokens: int,
            max_request_tokens: int,
            system_tokens: int = 0,
        ) -> bool:
            del current_tokens, max_request_tokens, system_tokens
            return False

        async def compact(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[types.runtime.ModelContextEvent],
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
            context: Sequence[types.runtime.ModelContextEvent],
            tools: object,
            mint_ref: Callable[[], TapeRef],
        ) -> tuple[ContextSplice, ...]:
            del tape, context, tools, mint_ref
            return ()

    a = Agent(model=StubModel(), tools=[], compactor=_NeverCompactor())
    history: list[types.runtime.ModelContextEvent] = [
        types.runtime.UserMessage(text="x")
    ]
    progressed = await a.compact_if_needed(history, a.model)
    assert progressed is True


@pytest.mark.asyncio
async def test_compact_if_needed_triggers_on_last_response_total() -> None:
    """The gate keys on the last response's real token count, not an estimate.

    A crude local estimator can under-count the true context (opus packs
    ~2.83 chars/token, not 4), so a 990k-token context can estimate to 100k
    and never cross the threshold. Anchoring on the provider's reported total
    fixes it: once a response reports 990k, the next gate compacts regardless
    of the estimate. The first turn (no response yet) falls back to the
    estimate.
    """

    class _UndercountModel(StubModel):
        @override
        def approx_request_tokens(self, request: types.model.ModelRequest) -> int:
            del request
            return 100_000

    history: list[types.runtime.ModelContextEvent] = [
        types.runtime.UserMessage(text="x")
    ]

    # Turn 1: no response recorded yet -> fall back to the estimate (100k <
    # threshold), so no compaction.
    rec = _ThresholdCompactor()
    a = Agent(
        model=_UndercountModel(
            max_request_tokens=1_000_000, max_response_tokens=128_000
        ),
        tools=[],
        compactor=rec,
    )
    assert await a.compact_if_needed(history, a.model) is True
    assert rec.compacted is False

    # Provider reports the sent request at 990k. Rule: (1 + 0.1) * input >=
    # window, i.e. 1.1 * 990_000 = 1_089_000 >= 1_000_000. The estimate still
    # under-counts at 100k, but the gate must now use the real 990k.
    a.record_response(
        types.model.ModelResponse(
            message=types.runtime.AssistantMessage(text=""),
            tokens=types.model.TokenCount(input_tokens=990_000),
        )
    )
    await a.compact_if_needed(history, a.model)
    assert rec.compacted is True


@pytest.mark.asyncio
async def test_compact_if_needed_adds_tokens_appended_since_last_response() -> None:
    """The gate adds entries appended after the last response to the anchor.

    Rule: ``X + min(0.075*X, max_response) >= window``. Anchor 800k alone
    does not fire (800k + 0.075*800k = 860k < 1M), but a tool-result appended
    since (~150k est tokens) does: X=950k, 950k + 0.075*950k = 1_021_250 >=
    1M. Without the since-term the gate would lag a turn and skip compaction.
    """
    rec = _ThresholdCompactor()
    a = Agent(
        model=StubModel(max_request_tokens=1_000_000, max_response_tokens=128_000),
        tools=[],
        compactor=rec,
    )
    # Anchor: provider counted the last request at 800k.
    a.record_response(
        types.model.ModelResponse(
            message=types.runtime.AssistantMessage(text=""),
            tokens=types.model.TokenCount(input_tokens=800_000),
        )
    )
    # History ends with the response's AssistantMessage, then a fresh
    # ToolResult appended this turn: 600_000 chars / 4 = ~150_000 est tokens
    # (StubModel: len // 4). X=950k; 950k + 0.075*950k = 1_021_250 >= 1M.
    history: list[types.runtime.ModelContextEvent] = [
        types.runtime.AssistantMessage(text=""),
        types.runtime.ToolResult(call_id="c", content="x" * 600_000),
    ]
    await a.compact_if_needed(history, a.model)
    assert rec.compacted is True


@pytest.mark.asyncio
async def test_compact_if_needed_no_since_term_when_response_is_tail() -> None:
    """When history ends at the last response, the since-term is zero.

    Anchor 800k: 800k + 0.075*800k = 860k < 1M window, and nothing has been
    appended since the last ``AssistantMessage``, so the gate must NOT compact.
    """
    rec = _ThresholdCompactor()
    a = Agent(
        model=StubModel(max_request_tokens=1_000_000, max_response_tokens=128_000),
        tools=[],
        compactor=rec,
    )
    a.record_response(
        types.model.ModelResponse(
            message=types.runtime.AssistantMessage(text=""),
            tokens=types.model.TokenCount(input_tokens=800_000),
        )
    )
    history: list[types.runtime.ModelContextEvent] = [
        types.runtime.UserMessage(text="q"),
        types.runtime.AssistantMessage(text=""),
    ]
    assert await a.compact_if_needed(history, a.model) is True
    assert rec.compacted is False


@pytest.mark.asyncio
async def test_compact_if_needed_counts_cached_input_tokens() -> None:
    """Prompt-cached context still triggers compaction.

    Under prompt caching the provider bills most of the prompt as
    ``cache_read``; ``input_tokens`` alone stays tiny. The gate must sum the
    cache pools (input + cache_creation + cache_read), or a 990k cached
    context -- billed as 5k input + 985k cache_read -- would never compact.
    This is the exact shape that wedged session 27a70970 (compact_count=0).
    """
    rec = _ThresholdCompactor()
    a = Agent(
        model=StubModel(max_request_tokens=1_000_000, max_response_tokens=128_000),
        tools=[],
        compactor=rec,
    )
    history: list[types.runtime.ModelContextEvent] = [
        types.runtime.UserMessage(text="x")
    ]
    a.record_response(
        types.model.ModelResponse(
            message=types.runtime.AssistantMessage(text=""),
            tokens=types.model.TokenCount(
                input_tokens=5_000, cache_read_tokens=985_000
            ),
        )
    )
    # 5_000 + 985_000 = 990_000; 1.1 * 990_000 = 1_089_000 >= 1M -> fires.
    await a.compact_if_needed(history, a.model)
    assert rec.compacted is True


@pytest.mark.asyncio
async def test_compact_if_needed_returns_false_on_compaction_failure() -> None:
    """When ``compact_now`` reports False, ``compact_if_needed`` propagates it.

    The bool flows up so future callers (proactive sites beyond the
    ``_AgentModel.stream`` overflow loop) can short-circuit on the same
    "history did not shrink" signal that bool was introduced for.
    """

    @dataclass(slots=True, kw_only=True)
    class _CompactBrokenButGatedTrue:
        def should_compact(
            self,
            current_tokens: int,
            max_request_tokens: int,
            system_tokens: int = 0,
        ) -> bool:
            del current_tokens, max_request_tokens, system_tokens
            return True

        async def compact(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[types.runtime.ModelContextEvent],
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
            context: Sequence[types.runtime.ModelContextEvent],
            tools: object,
            mint_ref: Callable[[], TapeRef],
        ) -> tuple[ContextSplice, ...]:
            del tape, context, tools, mint_ref
            return ()

    a = Agent(model=StubModel(), tools=[], compactor=_CompactBrokenButGatedTrue())
    history: list[types.runtime.ModelContextEvent] = [
        types.runtime.UserMessage(text="x")
    ]
    progressed = await a.compact_if_needed(history, a.model)
    assert progressed is False


@pytest.mark.asyncio
async def test_circuit_breaker_short_circuits_after_consecutive_failures() -> None:
    """``compact_if_needed`` returns False without invoking compactor after N failures."""

    @dataclass(slots=True, kw_only=True)
    class _AlwaysBroken:
        call_count: int = 0

        def should_compact(
            self,
            current_tokens: int,
            max_request_tokens: int,
            system_tokens: int = 0,
        ) -> bool:
            del current_tokens, max_request_tokens, system_tokens
            return True

        async def compact(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[types.runtime.ModelContextEvent],
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
            context: Sequence[types.runtime.ModelContextEvent],
            tools: object,
            mint_ref: Callable[[], TapeRef],
        ) -> tuple[ContextSplice, ...]:
            del tape, context, tools, mint_ref
            return ()

    compactor = _AlwaysBroken()
    a = Agent(model=StubModel(), tools=[], compactor=compactor)
    history: list[types.runtime.ModelContextEvent] = [
        types.runtime.UserMessage(text="x")
    ]

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
        def should_compact(
            self,
            current_tokens: int,
            max_request_tokens: int,
            system_tokens: int = 0,
        ) -> bool:
            del current_tokens, max_request_tokens, system_tokens
            return True

        async def compact(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[types.runtime.ModelContextEvent],
            model: object,
            mint_ref: Callable[[], TapeRef],
            custom_instructions: str | None = None,
        ) -> ContextSplice:
            del context, model, custom_instructions
            return _summary_override(
                [types.runtime.UserMessage(text="[summary]")], mint_ref, tape=tape
            )

        def maintain(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[types.runtime.ModelContextEvent],
            tools: object,
            mint_ref: Callable[[], TapeRef],
        ) -> tuple[ContextSplice, ...]:
            del tape, context, tools, mint_ref
            return ()

    a = Agent(model=StubModel(), tools=[], compactor=_SuccessfulCompactor())
    # Pre-populate failure count -- simulating prior auto-failures.
    a.compaction_state.compact_failures = 2
    history: list[types.runtime.ModelContextEvent] = [
        types.runtime.UserMessage(text="x")
    ]
    progressed = await a.compact_if_needed(history, a.model)
    assert progressed is True
    assert a.compaction_state.compact_failures == 0


def test_validate_payload_pending_check_subtracts_paired_externally() -> None:
    """Mid-payload ``if pending:`` check must subtract ``paired_externally``.

    The splice payload validator's end-of-payload check already subtracts
    ``paired_externally``; the mid-payload check fired on encountering
    a ``UserMessage`` did not, so the validator rejected a perfectly
    well-formed splice whose AM tool_calls were externally paired and
    whose payload then continued with a user-side entry.

    Without the fix, ``ContextSplice.__post_init__`` raises
    ``InvalidPayloadError`` here.
    """
    splice = ContextSplice(
        ref=TapeRef(session_id="s", ordinal=0),
        mask=(),
        insert_after=None,
        payload=(
            types.runtime.AssistantMessage(
                tool_calls=(types.runtime.ToolCall(id="c1", name="Bash", args={}),),
            ),
            types.runtime.UserMessage(text="[continuation]"),
        ),
        strategy="external_am_then_user",
        paired_externally=frozenset({"c1"}),
    )
    assert splice.payload[0]


@pytest.mark.asyncio
async def test_agent_compactor_recomputes_paired_externally_from_final_payload() -> (
    None
):
    """``_AgentCompactor.compact`` derives ``paired_externally`` from the
    post-rewrite payload, not the producer's declaration.

    The inner compactor returned a (validator-bypassed) override whose
    payload has an externally paired AM tool_call followed by a UM.
    The agent layer's repair pass (``_repair_compact_payload``) sees the
    AM with unmatched tool_calls and synthesizes a local
    ``[interrupted]`` ``ToolResult`` -- which means the call_id now has
    a *local* pair and is no longer "external". Inheriting the
    producer's ``paired_externally`` blindly would lie to the strict
    validator. The bridge must recompute via :func:`unpaired_call_ids`
    so the declaration stays honest. Here the synthetic TR pairs
    locally, so ``external_c1`` is correctly dropped.
    """

    @dataclass(slots=True, kw_only=True)
    class _ExternalAmCompactor:
        def should_compact(
            self,
            current_tokens: int,
            max_request_tokens: int,
            system_tokens: int = 0,
        ) -> bool:
            del current_tokens, max_request_tokens, system_tokens
            return False

        async def compact(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[types.runtime.ModelContextEvent],
            model: object,
            mint_ref: Callable[[], TapeRef],
            custom_instructions: str | None = None,
        ) -> ContextSplice:
            del context, model, custom_instructions
            # Use replay() so we can land [AM, UM] with paired_externally
            # without the constructor rejecting it -- mirrors a legacy
            # on-disk splice the runtime resumes.
            return ContextSplice.replay(
                ref=mint_ref(),
                mask=full_tape_mask(tape) if tape else (),
                insert_after=None,
                payload=(
                    types.runtime.AssistantMessage(
                        text="resuming",
                        tool_calls=(
                            types.runtime.ToolCall(
                                id="external_c1",
                                name="Bash",
                                args={},
                            ),
                        ),
                    ),
                    types.runtime.UserMessage(text="postscript"),
                ),
                strategy="external_am_then_user",
                paired_externally=frozenset({"external_c1"}),
            )

        def maintain(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[types.runtime.ModelContextEvent],
            tools: object,
            mint_ref: Callable[[], TapeRef],
        ) -> tuple[ContextSplice, ...]:
            del tape, context, tools, mint_ref
            return ()

    a = Agent(model=StubModel(), tools=[], compactor=_ExternalAmCompactor())
    a.runtime.append_history(types.runtime.UserMessage(text="x"))
    bridge = _AgentCompactor(_ExternalAmCompactor(), a)
    # If the validator regression returns, ContextSplice.__post_init__
    # raises InvalidPayloadError inside dataclasses.replace and the
    # await propagates the error here.
    override = await bridge.compact(
        tape=a.runtime.tape,
        context=a.runtime.context().messages,
        model=a.runtime.model,
        mint_ref=a.runtime.mint_ref,
        custom_instructions=None,
    )
    # ``_repair_compact_payload`` synthesized a local TR for external_c1,
    # so the call_id is no longer external and must be dropped from
    # ``paired_externally``. The local pair is present and well-formed.
    assert "external_c1" not in override.paired_externally
    am_call_ids: set[str] = set()
    tr_call_ids: set[str] = set()
    for entry in override.payload:
        if isinstance(entry, types.runtime.AssistantMessage):
            am_call_ids.update(tc.id for tc in entry.tool_calls)
        elif isinstance(entry, types.runtime.ToolResult):
            tr_call_ids.add(entry.call_id)
    assert "external_c1" in am_call_ids
    assert "external_c1" in tr_call_ids


@pytest.mark.asyncio
async def test_agent_compactor_scrunches_when_inner_output_still_oversized() -> None:
    """``_AgentCompactor.compact`` runs scrunch when the inner output won't fit.

    Producer returns a splice whose payload, after enrich+repair, still
    exceeds the model's input budget (``max_request_tokens -
    max_response_tokens - buffer_tokens``). The bridge must rescue via
    the scrunch maneuver -- partitioning oldest-first and re-running
    the inner producer per partition until the resolved view fits --
    instead of returning the oversized splice and letting the next
    provider call wedge.
    """

    @dataclass(slots=True, kw_only=True)
    class _OversizedCompactor:
        compact_calls: int = 0

        def should_compact(
            self,
            current_tokens: int,
            max_request_tokens: int,
            system_tokens: int = 0,
        ) -> bool:
            effective = max_request_tokens - system_tokens
            return current_tokens >= max(0, effective - 10)

        async def compact(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[types.runtime.ModelContextEvent],
            model: types.model.Model,
            mint_ref: Callable[[], TapeRef],
            custom_instructions: str | None = None,
        ) -> ContextSplice:
            del context, model, custom_instructions
            self.compact_calls += 1
            # First call returns a payload exceeding budget target so
            # the bridge triggers scrunch. Subsequent calls (scrunch
            # passes) return small payloads so the maneuver makes
            # progress and stops within a few iterations.
            payload_text = "X" * 5_000 if self.compact_calls == 1 else "ok"
            return _summary_override(
                [types.runtime.UserMessage(text=payload_text)],
                mint_ref,
                tape=tape or None,
            )

    @dataclass(slots=True, kw_only=True)
    class _OverflowModel(StubModel):
        max_request_tokens: int = 1_000
        max_response_tokens: int = 100

        @override
        def approx_request_tokens(self, request: types.model.ModelRequest) -> int:
            # Each entry's text length / 4 = tokens. Mirrors the chars
            # heuristic the scrunch planner uses.
            return sum(
                len(m.text) // 4
                for m in request.messages
                if isinstance(
                    m, (types.runtime.UserMessage, types.runtime.AgentSendMessage)
                )
            )

    budget = types.model.ContextBudget(
        max_request_tokens=1_000,
        max_response_tokens=100,
        buffer_tokens=100,
    )
    compactor = _OversizedCompactor()
    a = Agent(
        model=_OverflowModel(),
        compactor=compactor,
        budget=budget,
    )
    # Seed history just so the runtime has something to compact.
    a.runtime.append_history(types.runtime.UserMessage(text="x" * 4_000))

    assert await a.compact_now() is True

    # The bridge invoked the producer multiple times: once for the
    # normal compact, then again for each scrunch pass. At least 2.
    assert compactor.compact_calls >= 2, (
        f"expected scrunch to invoke producer multiple times, got"
        f" {compactor.compact_calls}"
    )
    # Final resolved view must fit (input + buffer below max).
    resolved = a.runtime.context().messages
    visible_chars = sum(
        len(m.text)
        for m in resolved
        if isinstance(m, (types.runtime.UserMessage, types.runtime.AgentSendMessage))
    )
    target = a.model.max_request_tokens - a.max_response_tokens - a.budget.buffer_tokens
    assert visible_chars // 4 <= target, (
        f"post-scrunch view ({visible_chars // 4} tok) still exceeds target {target}"
    )


@pytest.mark.asyncio
async def test_agent_compactor_retrigger_estimate_failure_is_nonfatal() -> None:
    """A raising token estimator in the willRetriggerNextTurn probe must not
    abort an otherwise-successful compaction.

    The retrigger probe formerly caught only ``(TypeError, ValueError)``; a
    provider-backed estimator can raise arbitrary errors (e.g. a CLI
    subprocess ``RuntimeError``). It must degrade like the sibling estimate
    blocks, not propagate.
    """

    @dataclass(slots=True, kw_only=True)
    class _FlakyEstimateModel(StubModel):
        n: int = 0

        @override
        def approx_request_tokens(self, request: types.model.ModelRequest) -> int:
            del request
            self.n += 1
            # The third estimate call is the willRetriggerNextTurn probe;
            # blow up only there so enrich/pre-scrunch estimates succeed.
            if self.n >= 3:
                raise RuntimeError("tokenizer subprocess gone")
            return 1

    @dataclass(slots=True, kw_only=True)
    class _OkCompactor:
        def should_compact(
            self,
            current_tokens: int,
            max_request_tokens: int,
            system_tokens: int = 0,
        ) -> bool:
            del current_tokens, max_request_tokens, system_tokens
            return False

        async def compact(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[types.runtime.ModelContextEvent],
            model: object,
            mint_ref: Callable[[], TapeRef],
            custom_instructions: str | None = None,
        ) -> ContextSplice:
            del context, model, custom_instructions
            return _summary_override(
                [types.runtime.UserMessage(text="[summary]")], mint_ref, tape=tape
            )

        def maintain(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[types.runtime.ModelContextEvent],
            tools: object,
            mint_ref: Callable[[], TapeRef],
        ) -> tuple[ContextSplice, ...]:
            del tape, context, tools, mint_ref
            return ()

    a = Agent(model=_FlakyEstimateModel(), tools=[], compactor=_OkCompactor())
    a.runtime.append_history(types.runtime.UserMessage(text="x"))
    # Must not raise -- the RuntimeError inside the retrigger probe is caught.
    assert await a.compact_now() is True


@pytest.mark.asyncio
async def test_agent_compactor_repairs_payload_after_scrunch() -> None:
    """Scrunch output is re-repaired so no AM tool_call is left unpaired.

    Scrunch re-runs the producer per partition and can emit a fresh
    ``AssistantMessage`` whose ``tool_calls`` have no local ``ToolResult``.
    Without a post-scrunch repair, ``unpaired_call_ids`` would declare those
    ids ``paired_externally`` even though no external partner exists, and the
    next provider call would 400. The bridge must synthesize an
    ``[interrupted]`` result so the splice declares nothing externally paired.
    """

    @dataclass(slots=True, kw_only=True)
    class _UnpairedScrunchCompactor:
        compact_calls: int = 0

        def should_compact(
            self,
            current_tokens: int,
            max_request_tokens: int,
            system_tokens: int = 0,
        ) -> bool:
            del current_tokens, max_request_tokens, system_tokens
            return False

        async def compact(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[types.runtime.ModelContextEvent],
            model: object,
            mint_ref: Callable[[], TapeRef],
            custom_instructions: str | None = None,
        ) -> ContextSplice:
            del context, model, custom_instructions
            self.compact_calls += 1
            if self.compact_calls == 1:
                # First pass: oversized so the bridge triggers scrunch.
                return _summary_override(
                    [types.runtime.UserMessage(text="X" * 5_000)],
                    mint_ref,
                    tape=tape or None,
                )
            # Scrunch passes: emit an AM with a tool_call but NO ToolResult.
            return _summary_override(
                [
                    types.runtime.AssistantMessage(
                        text="partial",
                        tool_calls=(
                            types.runtime.ToolCall(id="orphan", name="x", args={}),
                        ),
                    )
                ],
                mint_ref,
                tape=tape or None,
            )

        def maintain(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[types.runtime.ModelContextEvent],
            tools: object,
            mint_ref: Callable[[], TapeRef],
        ) -> tuple[ContextSplice, ...]:
            del tape, context, tools, mint_ref
            return ()

    @dataclass(slots=True, kw_only=True)
    class _OverflowModel(StubModel):
        max_request_tokens: int = 1_000
        max_response_tokens: int = 100

        @override
        def approx_request_tokens(self, request: types.model.ModelRequest) -> int:
            return sum(
                len(m.text) // 4
                for m in request.messages
                if isinstance(
                    m, (types.runtime.UserMessage, types.runtime.AgentSendMessage)
                )
            )

    budget = types.model.ContextBudget(
        max_request_tokens=1_000, max_response_tokens=100, buffer_tokens=100
    )
    a = Agent(
        model=_OverflowModel(),
        compactor=_UnpairedScrunchCompactor(),
        budget=budget,
    )
    a.runtime.append_history(types.runtime.UserMessage(text="x" * 4_000))

    assert await a.compact_now() is True

    splice = next(r for r in reversed(a.runtime.tape) if isinstance(r, ContextSplice))
    # No call_id may be declared externally paired without a real partner;
    # the post-scrunch repair must have synthesized a local TR for "orphan".
    assert "orphan" not in splice.paired_externally
    am_ids = {
        tc.id
        for e in splice.payload
        if isinstance(e, types.runtime.AssistantMessage)
        for tc in e.tool_calls
    }
    tr_ids = {
        e.call_id for e in splice.payload if isinstance(e, types.runtime.ToolResult)
    }
    # Every AM tool_call in the payload has a local TR pair.
    assert am_ids <= tr_ids, f"unpaired AM tool_calls in payload: {am_ids - tr_ids}"


@pytest.mark.asyncio
async def test_agent_compactor_scrunch_uses_agent_budget_not_model_cap() -> None:
    """Scrunch target uses the agent's (possibly lowered) ``max_request_tokens``.

    A user may set ``max_request_tokens`` below the model cap. Scrunch must fit
    that budget, not the raw model window, or a payload skips scrunch yet the
    willRetriggerNextTurn check (which uses the agent budget) flags it -- an
    inconsistent state that re-compacts every turn.
    """
    seen_targets: list[int] = []

    @dataclass(slots=True, kw_only=True)
    class _RecordingScrunchCompactor:
        compact_calls: int = 0

        def should_compact(
            self,
            current_tokens: int,
            max_request_tokens: int,
            system_tokens: int = 0,
        ) -> bool:
            del current_tokens, max_request_tokens, system_tokens
            return False

        async def compact(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[types.runtime.ModelContextEvent],
            model: object,
            mint_ref: Callable[[], TapeRef],
            custom_instructions: str | None = None,
        ) -> ContextSplice:
            del context, model, custom_instructions
            self.compact_calls += 1
            payload_text = "X" * 5_000 if self.compact_calls == 1 else "ok"
            return _summary_override(
                [types.runtime.UserMessage(text=payload_text)],
                mint_ref,
                tape=tape or None,
            )

        def maintain(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[types.runtime.ModelContextEvent],
            tools: object,
            mint_ref: Callable[[], TapeRef],
        ) -> tuple[ContextSplice, ...]:
            del tape, context, tools, mint_ref
            return ()

    @dataclass(slots=True, kw_only=True)
    class _OverflowModel(StubModel):
        max_request_tokens: int = 1_000_000  # model cap is huge
        max_response_tokens: int = 100

        @override
        def approx_request_tokens(self, request: types.model.ModelRequest) -> int:
            return sum(
                len(m.text) // 4
                for m in request.messages
                if isinstance(
                    m, (types.runtime.UserMessage, types.runtime.AgentSendMessage)
                )
            )

    # Agent budget far below the model cap.
    budget = types.model.ContextBudget(
        max_request_tokens=1_000, max_response_tokens=100, buffer_tokens=100
    )
    a = Agent(
        model=_OverflowModel(),
        compactor=_RecordingScrunchCompactor(),
        budget=budget,
    )

    async def _spy(
        self: _AgentCompactor,
        *,
        payload: list[types.runtime.ModelContextEvent],
        mint_ref: Callable[[], TapeRef],
        target_input_tokens: int,
    ) -> list[types.runtime.ModelContextEvent]:
        del self, mint_ref
        seen_targets.append(target_input_tokens)
        return list(payload)

    with patch.object(_AgentCompactor, "_scrunch_payload", _spy):
        a.runtime.append_history(types.runtime.UserMessage(text="x" * 4_000))
        await a.compact_now()

    assert seen_targets, "scrunch was never invoked"
    expected = a.max_request_tokens - a.max_response_tokens - a.budget.buffer_tokens
    assert seen_targets[0] == expected, (
        f"scrunch target {seen_targets[0]} != agent budget {expected}"
        f" (model cap is {a.model.max_request_tokens})"
    )


@pytest.mark.asyncio
async def test_agent_compactor_skips_scrunch_when_inner_output_fits() -> None:
    """Bridge does NOT run scrunch when the inner producer's output fits.

    Sanity check on the gate: scrunch is a rescue, not a default path.
    When the producer's normal compact() produces a payload that fits
    the budget, the bridge returns it as-is with one producer call.
    """

    @dataclass(slots=True, kw_only=True)
    class _NormalCompactor:
        compact_calls: int = 0

        def should_compact(
            self,
            current_tokens: int,
            max_request_tokens: int,
            system_tokens: int = 0,
        ) -> bool:
            del current_tokens, max_request_tokens, system_tokens
            return True

        async def compact(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[types.runtime.ModelContextEvent],
            model: types.model.Model,
            mint_ref: Callable[[], TapeRef],
            custom_instructions: str | None = None,
        ) -> ContextSplice:
            del context, model, custom_instructions
            self.compact_calls += 1
            return _summary_override(
                [types.runtime.UserMessage(text="ok")],
                mint_ref,
                tape=tape or None,
            )

    compactor = _NormalCompactor()
    a = Agent(model=StubModel(), compactor=compactor)
    a.runtime.append_history(types.runtime.UserMessage(text="x"))

    assert await a.compact_now() is True

    assert compactor.compact_calls == 1, (
        f"bridge should invoke producer exactly once when output fits;"
        f" got {compactor.compact_calls}"
    )


@pytest.mark.asyncio
async def test_compact_now_failure_appends_error_user_message() -> None:
    @dataclass(slots=True, kw_only=True)
    class _BrokenCompactor:
        def should_compact(
            self,
            current_tokens: int,
            max_request_tokens: int,
            system_tokens: int = 0,
        ) -> bool:
            del current_tokens, max_request_tokens, system_tokens
            return False

        async def compact(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[types.runtime.ModelContextEvent],
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
            context: Sequence[types.runtime.ModelContextEvent],
            tools: object,
            mint_ref: Callable[[], TapeRef],
        ) -> tuple[ContextSplice, ...]:
            del tape, context, tools, mint_ref
            return ()

    a = Agent(model=StubModel(), tools=[], compactor=_BrokenCompactor())
    a.runtime.append_history(types.runtime.UserMessage(text="x"))
    await a.compact_now()
    err = [
        e
        for e in a.runtime.context().messages
        if isinstance(e, types.runtime.UserMessage) and "[Compaction error:" in e.text
    ]
    assert len(err) == 1


@dataclass(slots=True, kw_only=True)
class _OverflowModel:
    """Model that raises types.model.PromptTooLongError on the first N calls."""

    model_id: str = "ovf"
    max_request_tokens: int = 100_000
    max_response_tokens: int = 1_024
    supports_streaming: bool = True
    supports_thinking: bool = False
    valid_thinking_states: tuple[str, ...] = ("off-hide",)
    supports_effort: bool = False
    valid_efforts: tuple[str, ...] = ()
    supports_cache_control: bool = False
    valid_service_tiers: tuple[str, ...] = ()
    valid_latency_modes: tuple[str, ...] = ()
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
        return isinstance(error, types.model.PromptTooLongError)

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
            raise types.model.PromptTooLongError("too long")
        return types.model.ModelResponse(
            message=types.runtime.AssistantMessage(text="recovered")
        )


@dataclass(slots=True, kw_only=True)
class _RawOverflowModel:
    """Model that raises a non-types.model.PromptTooLongError but classifies it as overflow.

    Mirrors the production failure where ``anthropic.APIStatusError``
    propagated up un-normalized: the recovery loop's catch must rely
    on ``is_context_overflow``, not on ``isinstance(exc,
    types.model.PromptTooLongError)``, or compaction never engages.
    """

    model_id: str = "raw"
    max_request_tokens: int = 100_000
    max_response_tokens: int = 1_024
    supports_streaming: bool = True
    supports_thinking: bool = False
    valid_thinking_states: tuple[str, ...] = ("off-hide",)
    supports_effort: bool = False
    valid_efforts: tuple[str, ...] = ()
    supports_cache_control: bool = False
    valid_service_tiers: tuple[str, ...] = ()
    valid_latency_modes: tuple[str, ...] = ()
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
            message=types.runtime.AssistantMessage(text="recovered")
        )


@pytest.mark.asyncio
async def test_agent_model_overflow_triggers_compact_now() -> None:
    """One overflow followed by success: compact_now runs once, response returned."""
    compact_calls: list[int] = []

    @dataclass(slots=True, kw_only=True)
    class _CountingCompactor:
        def should_compact(
            self,
            current_tokens: int,
            max_request_tokens: int,
            system_tokens: int = 0,
        ) -> bool:
            del current_tokens, max_request_tokens, system_tokens
            return False

        async def compact(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[types.runtime.ModelContextEvent],
            model: object,
            mint_ref: Callable[[], TapeRef],
            custom_instructions: str | None = None,
        ) -> ContextSplice:
            del context, model
            del custom_instructions
            compact_calls.append(1)
            return _summary_override(
                [types.runtime.UserMessage(text="[compact]")], mint_ref, tape=tape
            )

        def maintain(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[types.runtime.ModelContextEvent],
            tools: object,
            mint_ref: Callable[[], TapeRef],
        ) -> tuple[ContextSplice, ...]:
            del tape, context, tools, mint_ref
            return ()

    model = _OverflowModel(overflow_count=1)
    a = Agent(model=model, tools=[], compactor=_CountingCompactor())
    async for _ in a.run(types.runtime.UserMessage(text="hi")):
        pass
    assert len(compact_calls) == 1
    # _OverflowModel emitted "recovered" on the second call.
    assert any(
        isinstance(e, types.runtime.AssistantMessage) and e.text == "recovered"
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

        def should_compact(
            self,
            current_tokens: int,
            max_request_tokens: int,
            system_tokens: int = 0,
        ) -> bool:
            del current_tokens, max_request_tokens, system_tokens
            if self.triggered:
                return False
            self.triggered = True
            return True

        async def compact(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[types.runtime.ModelContextEvent],
            model: object,
            mint_ref: Callable[[], TapeRef],
            custom_instructions: str | None = None,
        ) -> ContextSplice:
            del context, model
            del custom_instructions
            order.append("compact")
            return _summary_override(
                [types.runtime.UserMessage(text="[compact]")], mint_ref, tape=tape
            )

        def maintain(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[types.runtime.ModelContextEvent],
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
        valid_thinking_states: tuple[str, ...] = ("off-hide",)
        supports_effort: bool = False
        valid_efforts: tuple[str, ...] = ()
        supports_cache_control: bool = False
        valid_service_tiers: tuple[str, ...] = ()
        valid_latency_modes: tuple[str, ...] = ()
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
                message=types.runtime.AssistantMessage(text="ok"),
            )

    model = _RecordingModel(order_log=order)
    a = Agent(model=model, tools=[], compactor=_OneShotCompactor())
    async for _ in a.run(types.runtime.UserMessage(text="hi")):
        pass
    assert order == ["compact", "stream"], order


@pytest.mark.asyncio
async def test_compact_now_publishes_compaction_progress_events() -> None:
    """Direct compaction path emits renderable observer events."""

    @dataclass(slots=True, kw_only=True)
    class _OkCompactor:
        def should_compact(
            self,
            current_tokens: int,
            max_request_tokens: int,
            system_tokens: int = 0,
        ) -> bool:
            del current_tokens, max_request_tokens, system_tokens
            return True

        async def compact(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[types.runtime.ModelContextEvent],
            model: object,
            mint_ref: Callable[[], TapeRef],
            custom_instructions: str | None = None,
        ) -> ContextSplice:
            del context, model
            del custom_instructions
            return _summary_override(
                [types.runtime.UserMessage(text="[compact]")], mint_ref, tape=tape
            )

        def maintain(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[types.runtime.ModelContextEvent],
            tools: object,
            mint_ref: Callable[[], TapeRef],
        ) -> tuple[ContextSplice, ...]:
            del tape, context, tools, mint_ref
            return ()

    events: list[types.runtime.RuntimeEvent] = []
    a = Agent(model=StubModel(), tools=[], compactor=_OkCompactor())
    a.runtime.append_history(types.runtime.UserMessage(text="hi"))
    a.runtime.observers.append(events.append)

    assert await a.compact_now() is True

    assert [type(event) for event in events] == [
        types.runtime.CompactStarted,
        types.runtime.CompactComplete,
    ]
    complete = events[-1]
    assert isinstance(complete, types.runtime.CompactComplete)
    assert len(complete.records) == 1
    record = complete.records[0]
    assert isinstance(record, ContextSplice)
    assert a.history == list(record.payload)
    assert a.activity.current_compact_start == 0.0


@pytest.mark.asyncio
async def test_sync_compact_now_reports_token_counts_from_override() -> None:
    """The sync ``compact_now`` path must report the override's token counts.

    Regression: the synchronous emit site built ``CompactComplete``
    without ``token_before`` / ``token_after`` / ``payload_entries``,
    so its fields defaulted to 0. The REPL then rendered
    ``~0 → ~0 tokens, 0 entries`` for every overflow-recovery
    compaction even though the override carried real counts. The async
    ``_compact_and_post`` path forwarded them; the two diverged. Both
    must derive the counts from the override.
    """

    @dataclass(slots=True, kw_only=True)
    class _OkCompactor:
        def should_compact(
            self,
            current_tokens: int,
            max_request_tokens: int,
            system_tokens: int = 0,
        ) -> bool:
            del current_tokens, max_request_tokens, system_tokens
            return True

        async def compact(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[types.runtime.ModelContextEvent],
            model: object,
            mint_ref: Callable[[], TapeRef],
            custom_instructions: str | None = None,
        ) -> ContextSplice:
            del context, model, custom_instructions
            return _summary_override(
                [types.runtime.UserMessage(text="[compact]")],
                mint_ref,
                tape=tape,
                token_before=120,
                token_after=10,
            )

        def maintain(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[types.runtime.ModelContextEvent],
            tools: object,
            mint_ref: Callable[[], TapeRef],
        ) -> tuple[ContextSplice, ...]:
            del tape, context, tools, mint_ref
            return ()

    events: list[types.runtime.RuntimeEvent] = []
    a = Agent(model=StubModel(), tools=[], compactor=_OkCompactor())
    a.runtime.append_history(types.runtime.UserMessage(text="hi"))
    a.runtime.observers.append(events.append)

    assert await a.compact_now() is True

    complete = events[-1]
    assert isinstance(complete, types.runtime.CompactComplete)
    assert complete.token_before == 120
    assert complete.token_after == 10
    assert complete.payload_entries == 1


@pytest.mark.asyncio
async def test_sync_compact_now_appends_lifecycle_markers_to_tape() -> None:
    """Sync compaction path appends CompactStarted + CompactComplete to tape.

    The inbox-arm handler at ``runtime.py: case Compact():`` calls
    ``append_history`` for the lifecycle markers. ``compact_now``
    bypasses the inbox (the runtime would cancel its own task if it
    pushed ``Compact``), and previously only published the markers --
    leaving them off the tape. Resume could not see that a synchronous
    overflow-recovery compaction had run, only the resulting splice.
    """

    @dataclass(slots=True, kw_only=True)
    class _OkCompactor:
        def should_compact(
            self,
            current_tokens: int,
            max_request_tokens: int,
            system_tokens: int = 0,
        ) -> bool:
            del current_tokens, max_request_tokens, system_tokens
            return True

        async def compact(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[types.runtime.ModelContextEvent],
            model: object,
            mint_ref: Callable[[], TapeRef],
            custom_instructions: str | None = None,
        ) -> ContextSplice:
            del context, model, custom_instructions
            return _summary_override(
                [types.runtime.UserMessage(text="[compact]")], mint_ref, tape=tape
            )

        def maintain(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[types.runtime.ModelContextEvent],
            tools: object,
            mint_ref: Callable[[], TapeRef],
        ) -> tuple[ContextSplice, ...]:
            del tape, context, tools, mint_ref
            return ()

    a = Agent(model=StubModel(), tools=[], compactor=_OkCompactor())
    a.runtime.append_history(types.runtime.UserMessage(text="hi"))

    assert await a.compact_now() is True

    started_seen = any(
        isinstance(record, types.tape.ReferrableTapeEvent)
        and isinstance(record.event, types.runtime.CompactStarted)
        for record in a.runtime.tape
    )
    complete_seen = any(
        isinstance(record, types.tape.ReferrableTapeEvent)
        and isinstance(record.event, types.runtime.CompactComplete)
        for record in a.runtime.tape
    )
    assert started_seen
    assert complete_seen


@pytest.mark.asyncio
async def test_sync_compact_now_failure_appends_compact_failed_to_tape() -> None:
    """Sync compaction failure also appends CompactFailed to the tape."""

    @dataclass(slots=True, kw_only=True)
    class _BrokenCompactor:
        def should_compact(
            self,
            current_tokens: int,
            max_request_tokens: int,
            system_tokens: int = 0,
        ) -> bool:
            del current_tokens, max_request_tokens, system_tokens
            return True

        async def compact(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[types.runtime.ModelContextEvent],
            model: object,
            mint_ref: Callable[[], TapeRef],
            custom_instructions: str | None = None,
        ) -> ContextSplice:
            del tape, context, model, mint_ref, custom_instructions
            raise RuntimeError("boom")

        def maintain(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[types.runtime.ModelContextEvent],
            tools: object,
            mint_ref: Callable[[], TapeRef],
        ) -> tuple[ContextSplice, ...]:
            del tape, context, tools, mint_ref
            return ()

    a = Agent(model=StubModel(), tools=[], compactor=_BrokenCompactor())
    a.runtime.append_history(types.runtime.UserMessage(text="hi"))

    assert await a.compact_now() is False

    started_seen = any(
        isinstance(record, types.tape.ReferrableTapeEvent)
        and isinstance(record.event, types.runtime.CompactStarted)
        for record in a.runtime.tape
    )
    failed_seen = any(
        isinstance(record, types.tape.ReferrableTapeEvent)
        and isinstance(record.event, types.runtime.CompactFailed)
        for record in a.runtime.tape
    )
    assert started_seen
    assert failed_seen


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

        def should_compact(
            self,
            current_tokens: int,
            max_request_tokens: int,
            system_tokens: int = 0,
        ) -> bool:
            del current_tokens, max_request_tokens, system_tokens
            return True

        async def compact(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[types.runtime.ModelContextEvent],
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
            context: Sequence[types.runtime.ModelContextEvent],
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
            history=[types.runtime.UserMessage(text="hi")],
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
        def should_compact(
            self,
            current_tokens: int,
            max_request_tokens: int,
            system_tokens: int = 0,
        ) -> bool:
            del current_tokens, max_request_tokens, system_tokens
            return True

        async def compact(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[types.runtime.ModelContextEvent],
            model: object,
            mint_ref: Callable[[], TapeRef],
            custom_instructions: str | None = None,
        ) -> ContextSplice:
            del tape, context, model, mint_ref
            del custom_instructions
            raise types.model.PromptTooLongError("compactor saw overflow")

        def maintain(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[types.runtime.ModelContextEvent],
            tools: object,
            mint_ref: Callable[[], TapeRef],
        ) -> tuple[ContextSplice, ...]:
            del tape, context, tools, mint_ref
            return ()

    a = Agent(
        model=_OverflowModel(overflow_count=0),
        compactor=_OverflowingCompactor(),
    )
    with pytest.raises(types.exceptions.ContextOverflowError) as ei:
        await a._agent_model.stream(
            history=[types.runtime.UserMessage(text="hi")],
            on_text=lambda _t: None,
            on_thinking=lambda _t: None,
        )
    msg = str(ei.value)
    assert "/clear" in msg
    assert "/compact" in msg
    assert isinstance(ei.value.__cause__, types.model.PromptTooLongError)


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
        def should_compact(
            self,
            current_tokens: int,
            max_request_tokens: int,
            system_tokens: int = 0,
        ) -> bool:
            del current_tokens, max_request_tokens, system_tokens
            return False

        async def compact(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[types.runtime.ModelContextEvent],
            model: object,
            mint_ref: Callable[[], TapeRef],
            custom_instructions: str | None = None,
        ) -> ContextSplice:
            del context, model
            del custom_instructions
            # Returns short summary; model keeps overflowing.
            return _summary_override(
                [types.runtime.UserMessage(text="[compact]")], mint_ref, tape=tape
            )

        def maintain(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[types.runtime.ModelContextEvent],
            tools: object,
            mint_ref: Callable[[], TapeRef],
        ) -> tuple[ContextSplice, ...]:
            del tape, context, tools, mint_ref
            return ()

    model = _OverflowModel(overflow_count=10)  # always overflow
    a = Agent(model=model, tools=[], compactor=_NoOpCompactor())
    with pytest.raises(types.exceptions.ContextOverflowError) as ei:
        await a._agent_model.stream(
            history=[types.runtime.UserMessage(text="x")],
            on_text=lambda _t: None,
            on_thinking=lambda _t: None,
        )
    msg = str(ei.value)
    assert "/clear" in msg
    assert "/compact" in msg
    assert isinstance(ei.value.__cause__, types.model.PromptTooLongError)
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

        def should_compact(
            self,
            current_tokens: int,
            max_request_tokens: int,
            system_tokens: int = 0,
        ) -> bool:
            del current_tokens, max_request_tokens, system_tokens
            return False

        async def compact(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[types.runtime.ModelContextEvent],
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
            context: Sequence[types.runtime.ModelContextEvent],
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
            history=[types.runtime.UserMessage(text="x")],
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
    """Recovery engages on any exception classified as overflow, not just ``types.model.PromptTooLongError``.

    When a provider's normalization slips and a raw provider exception
    propagates with the canonical ``is_context_overflow(exc)`` returning
    True, the recovery loop must still fire ``compact_now``. This is
    the bug that produced the production death-spiral: the recovery
    catch was narrowed to ``types.model.PromptTooLongError`` while the classifier
    knew the exception was overflow.
    """
    compact_calls: list[int] = []

    @dataclass(slots=True, kw_only=True)
    class _CountingCompactor:
        def should_compact(
            self,
            current_tokens: int,
            max_request_tokens: int,
            system_tokens: int = 0,
        ) -> bool:
            del current_tokens, max_request_tokens, system_tokens
            return False

        async def compact(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[types.runtime.ModelContextEvent],
            model: object,
            mint_ref: Callable[[], TapeRef],
            custom_instructions: str | None = None,
        ) -> ContextSplice:
            del context, model
            del custom_instructions
            compact_calls.append(1)
            return _summary_override(
                [types.runtime.UserMessage(text="[compact]")], mint_ref, tape=tape
            )

        def maintain(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[types.runtime.ModelContextEvent],
            tools: object,
            mint_ref: Callable[[], TapeRef],
        ) -> tuple[ContextSplice, ...]:
            del tape, context, tools, mint_ref
            return ()

    model = _RawOverflowModel(overflow_count=1)
    a = Agent(model=model, tools=[], compactor=_CountingCompactor())
    async for _ in a.run(types.runtime.UserMessage(text="hi")):
        pass
    assert len(compact_calls) == 1
    assert any(
        isinstance(e, types.runtime.AssistantMessage) and e.text == "recovered"
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
async def test_agent_tool_invalid_input_labels_then_errors() -> None:
    """Invalid input short-circuits dispatch but still publishes a label.

    The label renders as a plain dim tool-call line (not a "running"
    indicator), so emitting it keeps scrollback consistent with every other
    tool outcome -- each is preceded by its tool-call line. The inner tool is
    not invoked; the result is an ``InputValidationError`` naming the tool.
    """
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
    # A label IS published (tool name), before the error result.
    assert [label.text for label in labels] == ["Echo"]


@pytest.mark.asyncio
async def test_clear_cancels_explicit_background_jobs() -> None:
    started = asyncio.Event()

    @dataclass(slots=True, kw_only=True)
    class SlowTool(StubTool):
        @override
        def serialize_key(self, args: Mapping[str, object]) -> str | None:
            del args
            return None

        @override
        async def run(self, args: Mapping[str, object]) -> types.runtime.ToolResult:
            del args
            started.set()
            await asyncio.get_running_loop().create_future()
            return types.runtime.ToolResult(call_id="", content="done")

    a = _build_agent(tools=[SlowTool()])
    wrapper = next(t for t in a.runtime.tools_map.values() if t.name == "Echo")
    _ = await wrapper.run({"background": True})
    task = next(iter(a.background.values())).task
    await asyncio.wait_for(started.wait(), timeout=1.0)

    drive_task = asyncio.create_task(a.serve_forever())
    try:
        await a.clear()
        await asyncio.wait_for(task, timeout=1.0)
        assert a.background == {}
    finally:
        a.shutdown()
        await drive_task


@pytest.mark.asyncio
async def test_cancelled_background_tool_splices_placeholder() -> None:
    started = asyncio.Event()

    @dataclass(slots=True, kw_only=True)
    class SlowTool(StubTool):
        @override
        def serialize_key(self, args: Mapping[str, object]) -> str | None:
            del args
            return None

        @override
        async def run(self, args: Mapping[str, object]) -> types.runtime.ToolResult:
            del args
            started.set()
            await asyncio.get_running_loop().create_future()
            return types.runtime.ToolResult(call_id="", content="done")

    a = _build_agent(tools=[SlowTool()])
    wrapper = next(t for t in a.runtime.tools_map.values() if t.name == "Echo")
    token = agent_runtime.current_call_id_var.set("bg-1")
    try:
        placeholder = await wrapper.run({"background": True})
    finally:
        agent_runtime.current_call_id_var.reset(token)
    a.runtime.append_history(
        types.runtime.AssistantMessage(
            tool_calls=(
                types.runtime.ToolCall(
                    id="bg-1", name="Echo", args={"background": True}
                ),
            )
        )
    )
    a.runtime.append_history(placeholder)
    task = a.background["job-1"].task
    await asyncio.wait_for(started.wait(), timeout=1.0)

    task.cancel()
    # The job is still in the registry (no ``kill_tool``-style pop),
    # so the cancellation is treated as an external cascade: ``_run_bg``
    # posts a ``[cancelled]`` ``DetachedResult`` AND re-raises so the
    # cancel chain reaches the scheduler.
    with contextlib.suppress(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=1.0)
    items = await asyncio.wait_for(a.runtime.inbox.drain(), timeout=1.0)
    detached = [
        item for item in items if isinstance(item, types.runtime.DetachedResult)
    ]
    assert len(detached) == 1

    # The cancelled background job posts a ``[cancelled]`` result. Its
    # delivery into history (foreground-splice) is the BackgroundTask
    # foreground path, tracked separately under
    # ``docs/private/design_detached_tool_results.md`` scope 2; here we
    # assert the posted result itself.
    result = detached[0].result
    assert result.content == types.runtime.CANCELLED_PLACEHOLDER
    assert result.is_error


@pytest.mark.asyncio
async def test_kill_tool_cancels_explicit_background_job_by_id() -> None:
    started = asyncio.Event()

    @dataclass(slots=True, kw_only=True)
    class SlowTool(StubTool):
        @override
        def serialize_key(self, args: Mapping[str, object]) -> str | None:
            del args
            return None

        @override
        async def run(self, args: Mapping[str, object]) -> types.runtime.ToolResult:
            del args
            started.set()
            await asyncio.get_running_loop().create_future()
            return types.runtime.ToolResult(call_id="", content="done")

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
        def serialize_key(self, args: Mapping[str, object]) -> str | None:
            del args
            return None

        @override
        async def run(self, args: Mapping[str, object]) -> types.runtime.ToolResult:
            del args
            started.set()
            await asyncio.get_running_loop().create_future()
            return types.runtime.ToolResult(call_id="", content="done")

    a = _build_agent(tools=[SlowTool()])
    wrapper = next(t for t in a.runtime.tools_map.values() if t.name == "Echo")
    _ = await wrapper.run({"background": True})
    task = next(iter(a.background.values())).task
    await asyncio.wait_for(started.wait(), timeout=1.0)

    a.kill_all_tools()

    await asyncio.wait_for(task, timeout=1.0)
    assert a.background == {}


@pytest.mark.asyncio
async def test_tool_call_round_cap_allows_first_round_when_cap_is_one() -> None:
    """``max_tool_call_rounds=1`` permits the first tool round.

    Regression for AGENT-REVIEW-001: ``_before_tool_spawn`` used to
    block when ``num + 1 >= cap``, rejecting the first round whenever
    ``cap=1``. The docs ("Must be >= 1") and call sites such as
    ``examples/multi_agent_reviewer.py`` expect exactly one round.
    """
    started = asyncio.Event()

    @dataclass(slots=True, kw_only=True)
    class SideEffectTool(StubTool):
        @override
        def serialize_key(self, args: Mapping[str, object]) -> str | None:
            del args
            return None

        @override
        async def run(self, args: Mapping[str, object]) -> types.runtime.ToolResult:
            del args
            started.set()
            return types.runtime.ToolResult(call_id="", content="side effect")

    model = StubModel(
        responses=[
            types.runtime.AssistantMessage(
                tool_calls=(types.runtime.ToolCall(id="c1", name="Echo", args={}),)
            ),
            types.runtime.AssistantMessage(text="done"),
        ]
    )
    a = Agent(model=model, tools=[SideEffectTool()], max_tool_call_rounds=1)
    events = [type(ev) async for ev in a.run(types.runtime.UserMessage(text="go"))]

    assert started.is_set()
    assert types.runtime.ModelIdle in events
    assert types.runtime.ModelResponseError not in events


@pytest.mark.asyncio
async def test_tool_call_round_cap_blocks_second_round_when_cap_is_one() -> None:
    """``max_tool_call_rounds=1`` rejects the second tool-bearing response."""
    started_count = 0

    @dataclass(slots=True, kw_only=True)
    class CountingTool(StubTool):
        @override
        def serialize_key(self, args: Mapping[str, object]) -> str | None:
            del args
            return None

        @override
        async def run(self, args: Mapping[str, object]) -> types.runtime.ToolResult:
            del args
            nonlocal started_count
            started_count += 1
            return types.runtime.ToolResult(call_id="", content="ran")

    model = StubModel(
        responses=[
            types.runtime.AssistantMessage(
                tool_calls=(types.runtime.ToolCall(id="c1", name="Echo", args={}),)
            ),
            types.runtime.AssistantMessage(
                tool_calls=(types.runtime.ToolCall(id="c2", name="Echo", args={}),)
            ),
        ]
    )
    a = Agent(model=model, tools=[CountingTool()], max_tool_call_rounds=1)
    events = [type(ev) async for ev in a.run(types.runtime.UserMessage(text="go"))]

    assert started_count == 1
    assert types.runtime.ModelResponseError in events


@pytest.mark.asyncio
async def test_clear_drops_cancelled_background_result_after_fresh_turn() -> None:
    started = asyncio.Event()
    idle = asyncio.Event()

    @dataclass(slots=True, kw_only=True)
    class SlowTool(StubTool):
        @override
        def serialize_key(self, args: Mapping[str, object]) -> str | None:
            del args
            return None

        @override
        async def run(self, args: Mapping[str, object]) -> types.runtime.ToolResult:
            del args
            started.set()
            await asyncio.get_running_loop().create_future()
            return types.runtime.ToolResult(call_id="", content="done")

    a = _build_agent(
        model=StubModel(
            responses=[types.runtime.AssistantMessage(text="fresh response")]
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
        a.runtime.inbox.push_back(types.runtime.UserMessage(text="fresh"))
        await asyncio.wait_for(idle.wait(), timeout=1.0)
    finally:
        a.shutdown(force=True)
        with contextlib.suppress(asyncio.CancelledError):
            await drive
        a.runtime.observers.remove(_watch)

    user_texts = [
        entry.text
        for entry in a.history
        if isinstance(entry, types.runtime.UserMessage)
    ]
    assert not any(types.runtime.CANCELLED_PLACEHOLDER in text for text in user_texts)
    assert not any(text.startswith("[Tool ") for text in user_texts)


@pytest.mark.asyncio
async def test_agent_tool_background_exception_is_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    @dataclass(slots=True, kw_only=True)
    class FailingTool(StubTool):
        @override
        def serialize_key(self, args: Mapping[str, object]) -> str | None:
            del args
            return None

        @override
        async def run(self, args: Mapping[str, object]) -> types.runtime.ToolResult:
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
    items = a.runtime.inbox.drain_nowait()
    detached = [i for i in items if isinstance(i, types.runtime.DetachedResult)]
    assert len(detached) == 1
    assert detached[0].is_error


@pytest.mark.asyncio
async def test_agent_compactor_appends_continuation_when_summary_ends_assistant(
    tmp_path: Path,
) -> None:
    """If summary ends with types.runtime.AssistantMessage, an inert types.runtime.UserMessage is appended."""

    @dataclass(slots=True, kw_only=True)
    class _AssistantTerminatedCompactor:
        def should_compact(
            self,
            current_tokens: int,
            max_request_tokens: int,
            system_tokens: int = 0,
        ) -> bool:
            del current_tokens, max_request_tokens, system_tokens
            return False

        async def compact(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[types.runtime.ModelContextEvent],
            model: object,
            mint_ref: Callable[[], TapeRef],
            custom_instructions: str | None = None,
        ) -> ContextSplice:
            del context, model
            del custom_instructions
            return _summary_override(
                [types.runtime.AssistantMessage(text="model said")], mint_ref, tape=tape
            )

        def maintain(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[types.runtime.ModelContextEvent],
            tools: object,
            mint_ref: Callable[[], TapeRef],
        ) -> tuple[ContextSplice, ...]:
            del tape, context, tools, mint_ref
            return ()

    a = Agent(
        model=StubModel(),
        compactor=_AssistantTerminatedCompactor(),
        session_dir=tmp_path,
    )
    a.runtime.append_history(types.runtime.UserMessage(text="x"))
    await a.compact_now()
    # The continuation user-message terminator was appended.
    last = a.runtime.context().messages[-1]
    assert isinstance(last, types.runtime.UserMessage)
    assert last.text == "[continuation]"


@pytest.mark.asyncio
async def test_agent_compactor_post_enrich_failure_swallowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Errors inside ``post_compact_enrich`` are logged and don't propagate."""

    @dataclass(slots=True, kw_only=True)
    class _OkCompactor:
        def should_compact(
            self,
            current_tokens: int,
            max_request_tokens: int,
            system_tokens: int = 0,
        ) -> bool:
            del current_tokens, max_request_tokens, system_tokens
            return False

        async def compact(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[types.runtime.ModelContextEvent],
            model: object,
            mint_ref: Callable[[], TapeRef],
            custom_instructions: str | None = None,
        ) -> ContextSplice:
            del context, model
            del custom_instructions
            return _summary_override(
                [types.runtime.UserMessage(text="[summary]")], mint_ref, tape=tape
            )

        def maintain(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[types.runtime.ModelContextEvent],
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
        compactor=_OkCompactor(),
        session_dir=tmp_path,
    )
    a.runtime.append_history(types.runtime.UserMessage(text="x"))

    monkeypatch.setattr("sagent.agent.agent.post_compact_enrich", _boom)
    await a.compact_now()

    # Summary still survived; the enrich failure was swallowed.
    assert any(
        isinstance(e, types.runtime.UserMessage) and e.text == "[summary]"
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
    async for _ in a.run(types.runtime.UserMessage(text="hi")):
        pass
    assert call_count > pre_run_count
    assert model.received[-1].system == f"sys-v{call_count}"


@pytest.mark.asyncio
async def test_stream_sets_cli_publish_var_to_runtime_publish() -> None:
    """``_AgentModel.stream`` exposes ``runtime.publish`` via ``cli_publish_var``.

    The CLI MCP bridge reads this var to surface ``ToolLabel`` for
    subprocess-driven tool calls. The agent layer is the seam that
    wires the publisher; without the ``ContextVar.set`` around
    ``send_with_retry``, the bridge silently no-ops and the REPL never
    announces CLI tool calls.

    Asserts the visible behavior: invoking the captured publisher
    delivers the event to the runtime's observer list, just as
    calling ``runtime.publish`` would. ``is``-comparison fails for
    bound methods (each ``.publish`` access mints a fresh wrapper),
    so we verify via fan-out instead.
    """

    @dataclass(slots=True, kw_only=True)
    class _RecordingModel(StubModel):
        seen: list[Callable[[types.runtime.RuntimeEvent], None] | None] = field(
            default_factory=list
        )

        @override
        async def stream(
            self,
            request: types.model.ModelRequest,
            on_text: object = None,
            on_thinking: object = None,
        ) -> types.model.ModelResponse:
            self.seen.append(agent_runtime.cli_publish_var.get())
            return await super().stream(request, on_text, on_thinking)

    model = _RecordingModel()
    a = _build_agent(model=model)
    observed: list[types.runtime.RuntimeEvent] = []
    a.runtime.observers.append(observed.append)
    async for _ in a.run(types.runtime.UserMessage(text="hi")):
        pass

    assert model.seen, "model.stream was never invoked"
    publish = model.seen[0]
    assert publish is not None, (
        "cli_publish_var was unset during the model call; the agent layer"
        " must ``set`` it before invoking ``send_with_retry``"
    )
    sentinel = types.runtime.ToolLabel(call_id="probe", text="probe")
    publish(sentinel)
    assert sentinel in observed, (
        "captured publisher did not fan out to the runtime's observers;"
        " the var must hold ``runtime.publish`` (or an equivalent that"
        " reaches the same observer list), not an arbitrary callable"
    )


@pytest.mark.asyncio
async def test_stream_resets_cli_publish_var_after_request() -> None:
    """``_AgentModel.stream`` restores ``cli_publish_var`` to its prior value.

    Without the ``finally`` reset, the publisher leaks to whatever
    coroutine the agent layer hands control to next (another model
    call, a compaction, a subagent), routing its bridge-driven tool
    labels to the wrong runtime.
    """

    def sentinel(_ev: types.runtime.RuntimeEvent) -> None:
        return None

    token = agent_runtime.cli_publish_var.set(sentinel)
    try:
        model = StubModel()
        a = _build_agent(model=model)
        async for _ in a.run(types.runtime.UserMessage(text="hi")):
            pass
        assert agent_runtime.cli_publish_var.get() is sentinel, (
            "cli_publish_var leaked past the model call; the agent layer"
            " must reset to the pre-call value in a finally block"
        )
    finally:
        agent_runtime.cli_publish_var.reset(token)


def test_subagent_inherits_root_cost_tracker() -> None:
    """C5: non-persistent subagent's cost folds into the root tracker."""
    root = _build_agent()
    child = _build_agent()
    response = types.model.ModelResponse(
        message=types.runtime.AssistantMessage(text="ok"),
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
        message=types.runtime.AssistantMessage(text="ok"),
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
                message=types.runtime.AssistantMessage(text="from A"), total_cost=0.10
            )

    model_a = GatedModel(model_id="model-A")
    model_b = StubModel(model_id="model-B")
    agent = _build_agent(model=model_a)

    async def consume() -> None:
        async for _ in agent.run(types.runtime.UserMessage(text="hi")):
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


def test_swap_model_resets_thinking_state_invalid_for_new_model() -> None:
    """Swapping to a model that supports thinking but rejects the current
    state's wire mode must reset the state, not leave it to 400.

    Mirrors the live generation split: an ``adaptive-*`` state is valid on
    opus-4-8 but the 4-5 generation rejects ``adaptive`` (only ``on-*``).
    The swap must drop the now-invalid state to ``off-hide`` rather than
    send ``thinking.type=adaptive`` and 400 on the next turn.
    """

    @dataclass(slots=True, kw_only=True)
    class _EnabledOnlyModel(StubModel):
        # Supports thinking, but only the ``on-*`` (enabled) states.
        @property
        @override
        def valid_thinking_states(self) -> tuple[str, ...]:
            return ("on-show", "on-hide", "off-hide")

    a = Agent(model=StubModel(), tools=[], thinking_state="adaptive-hide")
    assert a.thinking_state == "adaptive-hide"
    a.swap_model(_EnabledOnlyModel())
    assert a.thinking_state == "off-hide"
    assert a.thinking is None
    assert a.show_thinking is False


def test_swap_model_keeps_thinking_state_valid_for_new_model() -> None:
    """A state still valid for the new model survives the swap unchanged."""
    a = Agent(
        model=StubModel(supports_thinking=True), tools=[], thinking_state="on-hide"
    )
    a.swap_model(StubModel(model_id="other", supports_thinking=True))
    assert a.thinking_state == "on-hide"


def test_swap_model_resets_legacy_thinking_mode_invalid_for_new_model() -> None:
    """The legacy ``agent.thinking = "adaptive"`` path (state ``None``, set by
    ``AgentSelf`` and direct assignment) must also reset on an incompatible
    swap, or the materialized request sends a rejected wire mode and 400s.
    """

    @dataclass(slots=True, kw_only=True)
    class _EnabledOnlyModel(StubModel):
        supports_thinking: bool = True  # supports thinking, but enabled-only

        @property
        @override
        def valid_thinking_states(self) -> tuple[str, ...]:
            return ("on-show", "on-hide", "off-hide")

    a = _build_agent(model=StubModel(supports_thinking=True))
    a.thinking = "adaptive"  # state=None, wire mode "adaptive"
    assert a.thinking_state is None
    a.swap_model(_EnabledOnlyModel())
    # "adaptive" is unsatisfiable on an enabled-only model; must be cleared.
    assert a.thinking is None


def test_swap_model_resets_effort_invalid_for_new_model() -> None:
    """Effort outside the new model's ``valid_efforts`` resets to unset."""
    a = _build_agent(
        model=StubModel(supports_effort=True, valid_efforts=("low", "high"))
    )
    a.effort = "high"
    a.swap_model(StubModel(supports_effort=True, valid_efforts=("low", "medium")))
    assert a.effort is None


def test_recompact_docstring_describes_alias_semantics() -> None:
    assert Agent.recompact.__doc__ is not None
    assert "alias" in Agent.recompact.__doc__.lower()


@pytest.mark.asyncio
async def test_compact_if_needed_uses_agent_request_cap() -> None:
    @dataclass(slots=True, kw_only=True)
    class _RecorderCompactor:
        seen_max_request_tokens: list[int] = field(default_factory=list)

        def should_compact(
            self,
            current_tokens: int,
            max_request_tokens: int,
            system_tokens: int = 0,
        ) -> bool:
            del current_tokens, system_tokens
            self.seen_max_request_tokens.append(max_request_tokens)
            return False

        async def compact(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[types.runtime.ModelContextEvent],
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
        [types.runtime.UserMessage(text="x")], a.model
    )

    assert progressed is True
    assert compactor.seen_max_request_tokens == [10_000]


@pytest.mark.asyncio
async def test_compact_if_needed_resets_failure_breaker_when_healthy() -> None:
    @dataclass(slots=True, kw_only=True)
    class _HealthyCompactor:
        def should_compact(
            self,
            current_tokens: int,
            max_request_tokens: int,
            system_tokens: int = 0,
        ) -> bool:
            del current_tokens, max_request_tokens, system_tokens
            return False

        async def compact(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[types.runtime.ModelContextEvent],
            model: object,
            mint_ref: Callable[[], TapeRef],
            custom_instructions: str | None = None,
        ) -> ContextSplice:
            del tape, context, model, mint_ref, custom_instructions
            raise AssertionError("compact should not run")

    a = Agent(model=StubModel(), compactor=_HealthyCompactor())
    a.compaction_state.compact_failures = 3

    progressed = await a.compact_if_needed(
        [types.runtime.UserMessage(text="x")], a.model
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
        def should_compact(
            self,
            current_tokens: int,
            max_request_tokens: int,
            system_tokens: int = 0,
        ) -> bool:
            del current_tokens, max_request_tokens, system_tokens
            return False

        async def compact(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[types.runtime.ModelContextEvent],
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

    progressed = await a.compact_if_needed([types.runtime.UserMessage(text="x")], model)

    assert progressed is True
    tools_seen = model.estimated_tools[-1]
    assert tools_seen is not None
    assert isinstance(tools_seen[0], BackgroundAwareTool)


@pytest.mark.asyncio
async def test_agent_compactor_receives_canonical_context() -> None:
    seen_context: list[Sequence[types.runtime.ModelContextEvent]] = []

    @dataclass(slots=True, kw_only=True)
    class _RecordingCompactor:
        def should_compact(
            self,
            current_tokens: int,
            max_request_tokens: int,
            system_tokens: int = 0,
        ) -> bool:
            del current_tokens, max_request_tokens, system_tokens
            return True

        async def compact(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[types.runtime.ModelContextEvent],
            model: object,
            mint_ref: Callable[[], TapeRef],
            custom_instructions: str | None = None,
        ) -> ContextSplice:
            del model, custom_instructions
            seen_context.append(context)
            return _summary_override(
                [types.runtime.UserMessage(text="[summary]")], mint_ref, tape=tape
            )

    call = types.runtime.ToolCall(id="call_1", name="Bash", args={})
    budget = types.model.ContextBudget(
        max_request_tokens=100_000,
        max_response_tokens=1_024,
        message_budget_chars=10,
    )
    a = Agent(model=StubModel(), compactor=_RecordingCompactor(), budget=budget)
    a.runtime.append_history(types.runtime.UserMessage(text="start"))
    a.runtime.append_history(types.runtime.AssistantMessage(tool_calls=(call,)))
    a.runtime.append_history(
        types.runtime.ToolResult(call_id="call_1", content="x" * 1_000)
    )

    assert await a.compact_now() is True
    result = seen_context[-1][2]
    assert isinstance(result, types.runtime.ToolResult)
    assert result.content == "x" * 1_000


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
        def should_compact(
            self,
            current_tokens: int,
            max_request_tokens: int,
            system_tokens: int = 0,
        ) -> bool:
            del current_tokens, max_request_tokens, system_tokens
            return True

        async def compact(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[types.runtime.ModelContextEvent],
            model: object,
            mint_ref: Callable[[], TapeRef],
            custom_instructions: str | None = None,
        ) -> ContextSplice:
            del context, model, custom_instructions
            return _summary_override(
                [types.runtime.UserMessage(text="[summary]")], mint_ref, tape=tape
            )

    model = _RecordingModel()
    tool = StubTool(
        directive_schema=json_freeze(
            {"type": "object", "properties": {"msg": {"type": "string"}}}
        )
    )
    a = Agent(model=model, tools=[tool], compactor=_OkCompactor())
    a.runtime.append_history(types.runtime.UserMessage(text="x"))

    assert await a.compact_now() is True

    for tools_seen in model.estimated_tools:
        assert tools_seen is not None
        assert isinstance(tools_seen[0], BackgroundAwareTool)


@pytest.mark.asyncio
async def test_compact_payload_ending_with_tool_calls_gets_synthetic_results() -> None:
    """Post-compact tail repair must preserve tool-call pairing."""

    class _ToolCallCompactor:
        def should_compact(
            self,
            current_tokens: int,
            max_request_tokens: int,
            system_tokens: int = 0,
        ) -> bool:
            del current_tokens, max_request_tokens, system_tokens
            return True

        async def compact(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[types.runtime.ModelContextEvent],
            model: types.model.Model,
            mint_ref: Callable[[], TapeRef],
            custom_instructions: str | None = None,
        ) -> ContextSplice:
            del context, model, custom_instructions
            call = types.runtime.ToolCall(id="call-1", name="Echo", args={})
            del tape
            return ContextSplice.replay(
                ref=mint_ref(),
                mask=(),
                insert_after=None,
                payload=(types.runtime.AssistantMessage(tool_calls=(call,)),),
                strategy="legacy",
                paired_externally=frozenset({"call-1"}),
            )

    a = Agent(model=StubModel(), compactor=_ToolCallCompactor())
    a.runtime.append_history(types.runtime.UserMessage(text="x"))

    assert await a.compact_now() is True

    payload = a.runtime.context().messages
    validate_context(payload)
    assert isinstance(payload[-1], types.runtime.ToolResult)
    assert payload[-1].call_id == "call-1"
    assert payload[-1].is_error is True


@pytest.mark.asyncio
@pytest.mark.real_sleep
async def test_compact_now_awaits_active_runtime_compact_task() -> None:
    """Overflow compaction must not race an inbox-driven compaction."""
    started = asyncio.Event()
    release = asyncio.Event()
    call_count = 0

    class _SlowCompactor:
        def should_compact(
            self,
            current_tokens: int,
            max_request_tokens: int,
            system_tokens: int = 0,
        ) -> bool:
            del current_tokens, max_request_tokens, system_tokens
            return True

        async def compact(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[types.runtime.ModelContextEvent],
            model: types.model.Model,
            mint_ref: Callable[[], TapeRef],
            custom_instructions: str | None = None,
        ) -> ContextSplice:
            nonlocal call_count
            del context, model, custom_instructions
            call_count += 1
            started.set()
            await release.wait()
            return _summary_override(
                [types.runtime.UserMessage(text="[summary]")],
                mint_ref,
                tape=tape,
            )

    a = Agent(model=StubModel(), compactor=_SlowCompactor())
    a.runtime.append_history(types.runtime.UserMessage(text="x"))
    a.runtime.compact_task = asyncio.create_task(a.runtime._compact_and_post(""))
    await started.wait()

    waiting = asyncio.create_task(a.compact_now())
    await asyncio.sleep(0.05)
    assert call_count == 1
    assert not waiting.done()
    release.set()

    assert await waiting is True
    await a.runtime.compact_task
    assert call_count == 1


@pytest.mark.asyncio
async def test_will_retrigger_uses_agent_budget_not_model_cap() -> None:
    """Post-compact retry prediction must call compactor with agent budget."""

    @dataclass(slots=True, kw_only=True)
    class _BudgetModel(StubModel):
        @override
        def approx_request_tokens(self, request: types.model.ModelRequest) -> int:
            del request
            return 90

    @dataclass(slots=True, kw_only=True)
    class _RecordingCompactor:
        should_compact_calls: list[tuple[int, int, int]] = field(default_factory=list)

        def should_compact(
            self,
            current_tokens: int,
            max_request_tokens: int,
            system_tokens: int = 0,
        ) -> bool:
            self.should_compact_calls.append(
                (current_tokens, max_request_tokens, system_tokens),
            )
            return True

        async def compact(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[types.runtime.ModelContextEvent],
            model: types.model.Model,
            mint_ref: Callable[[], TapeRef],
            custom_instructions: str | None = None,
        ) -> ContextSplice:
            del context, model, custom_instructions
            return _summary_override(
                [types.runtime.UserMessage(text="[summary]")],
                mint_ref,
                tape=tape,
            )

    budget = types.model.ContextBudget(
        max_request_tokens=100,
        max_response_tokens=10,
        buffer_tokens=10,
    )
    compactor = _RecordingCompactor()
    a = Agent(
        model=_BudgetModel(max_request_tokens=1_000, max_response_tokens=10),
        compactor=compactor,
        budget=budget,
    )
    a.runtime.append_history(types.runtime.UserMessage(text="x"))

    assert await a.compact_now() is True

    splices = [record for record in a.runtime.tape if isinstance(record, ContextSplice)]
    assert len(splices) == 1
    assert "re-trigger compaction" in splices[0].fallback_reason
    # The retry-prediction call must use the agent budget (100), not the
    # raw model cap (1_000). The leading should_compact call is the
    # proactive gate; the trailing one is the willRetriggerNextTurn
    # prediction -- both run against agent.max_request_tokens.
    assert compactor.should_compact_calls, "should_compact never called"
    for _input, max_req, _max_resp in compactor.should_compact_calls:
        assert max_req == 100, (
            f"willRetriggerNextTurn used max_request_tokens={max_req}, expected 100"
        )


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
            history: list[types.runtime.ModelContextEvent],
            tool_state: ToolState,
            *,
            budget_chars: int = 100_000,
        ) -> None:
            del history, tool_state
            calls.append(budget_chars)

    @dataclass(slots=True, kw_only=True)
    class _OkCompactor:
        def should_compact(
            self,
            current_tokens: int,
            max_request_tokens: int,
            system_tokens: int = 0,
        ) -> bool:
            del current_tokens, max_request_tokens, system_tokens
            return True

        async def compact(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[types.runtime.ModelContextEvent],
            model: object,
            mint_ref: Callable[[], TapeRef],
            custom_instructions: str | None = None,
        ) -> ContextSplice:
            del context, model, custom_instructions
            return _summary_override(
                [types.runtime.UserMessage(text="[summary]")], mint_ref, tape=tape
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
    a.runtime.append_history(types.runtime.UserMessage(text="x"))

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


def test_agent_resume_loads_service_suspended_retry_at(tmp_path: Path) -> None:
    """``resume()`` populates ``runtime.resume_retry_at`` from the latest event."""
    a1 = _build_agent(session_dir=tmp_path)
    later = types.runtime.ModelServiceSuspended(
        provider="anthropic",
        auth="key",
        account="default",
        model_id="claude-test",
        retry_at=2_000.0,
        delay_sec=120.0,
        server_supplied=True,
        error=types.runtime.ServiceErrorSnapshot(
            type_name="RateLimitError", message="429"
        ),
    )
    earlier = replace(later, retry_at=1_000.0)
    a1.runtime.publish(types.runtime.SaveSession())
    append_session(tmp_path / "session.jsonl", runtime_events=[earlier, later])

    loaded = load_session(tmp_path, {})
    assert loaded is not None
    a2 = _build_agent(session_dir=tmp_path)
    a2.resume(*loaded)
    assert a2.runtime.resume_retry_at == 2_000.0


@pytest.mark.asyncio
async def test_shutdown_force_false_cancels_detached_and_explicit_bg() -> None:
    """``shutdown(force=False)`` must cancel detached + explicit_bg tasks.

    Without this, detached / explicit-bg tasks outlive ``serve_forever``
    and post to a dead inbox after the runtime has Quit. Only persistent
    subagents are spared -- they own their own ``serve_forever``.
    """
    a = _build_agent()
    detached_task = asyncio.create_task(asyncio.sleep(60))
    explicit_task = asyncio.create_task(asyncio.sleep(60))
    persistent_task = asyncio.create_task(asyncio.sleep(60))
    a.runtime.detached["call-d"] = detached_task
    a.register_background(
        "job-explicit",
        BackgroundTaskEntry(
            task=explicit_task,
            tool_name="X",
            queue_id="job-explicit",
            started=0.0,
            kind="tool",
        ),
    )
    a.register_background(
        "job-persistent",
        BackgroundTaskEntry(
            task=persistent_task,
            tool_name="Y",
            queue_id="job-persistent",
            started=0.0,
            kind="persistent_subagent",
            persistent_run_id="run-persistent",
        ),
    )
    try:
        a.shutdown(force=False)
        with contextlib.suppress(asyncio.CancelledError):
            await detached_task
        with contextlib.suppress(asyncio.CancelledError):
            await explicit_task
        assert detached_task.cancelled()
        assert explicit_task.cancelled()
        assert not persistent_task.cancelled(), (
            "persistent_subagent owns its own serve_forever; must NOT be cancelled"
        )
    finally:
        _ = persistent_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await persistent_task


# --- A40: _repair_compact_payload pairs tool calls and fills missing TRs --


def test_repair_compact_payload_fills_missing_tool_results_with_interrupted() -> None:
    """Each ``ToolCall`` id declared by an ``AssistantMessage`` must
    receive a matching ``ToolResult`` after repair. Missing pairs get
    a synthetic ``[interrupted]`` placeholder so the payload is a
    wire-valid compaction input.
    """
    am = types.runtime.AssistantMessage(
        tool_calls=tuple(
            types.runtime.ToolCall(id=f"c{idx}", name="x", args={}) for idx in range(20)
        ),
    )
    # Only every other id has a real result; the rest must be filled.
    results = [
        types.runtime.ToolResult(call_id=f"c{idx}", content="ok")
        for idx in range(0, 20, 2)
    ]
    out = _repair_compact_payload(
        [am, *results, types.runtime.UserMessage(text="next")]
    )
    seen_ids = {e.call_id for e in out if isinstance(e, types.runtime.ToolResult)}
    assert seen_ids == {f"c{idx}" for idx in range(20)}
    interrupted = [
        e
        for e in out
        if isinstance(e, types.runtime.ToolResult) and e.content == "[interrupted]"
    ]
    assert len(interrupted) == 10


@pytest.mark.asyncio
async def test_agent_run_rejects_concurrent_drivers() -> None:
    """Two concurrent ``Agent.run`` calls on the same agent are forbidden.

    The single-driver contract owns ``shutdown`` in ``finally``;
    overlapping callers would push ``Quit()`` into the foreign driver
    and silently corrupt it. Fail loudly instead.
    """
    a = _build_agent()
    a._run_active = True
    try:
        with pytest.raises(RuntimeError, match="not reentrant"):
            async for _event in a.run(types.runtime.UserMessage(text="second")):
                pass
    finally:
        a._run_active = False


# --- A18: lock-in the "no await between check and set" invariant -----------


def test_agent_run_has_no_await_between_run_active_check_and_set() -> None:
    """A18: source-level guard on ``Agent.run``'s check-and-set atomicity.

    The ``if self._run_active`` / ``self._run_active = True`` pair runs
    atomically only because no ``await`` interleaves them under asyncio's
    cooperative scheduling. A future maintainer dropping an ``await`` (a
    log flush, a metric push) between the two would silently let two
    concurrent ``run`` callers slip past the guard. Scan the source so
    the invariant fails CI rather than failing in production.
    """
    source = Path(Agent.run.__code__.co_filename).read_text(encoding="utf-8")
    lines = source.splitlines()
    check_line = next(
        i for i, line in enumerate(lines) if "if self._run_active:" in line
    )
    set_line = next(
        i
        for i, line in enumerate(lines[check_line:], start=check_line)
        if "self._run_active = True" in line
    )
    # Strip strings/comments before scanning for the await keyword so the
    # lock-down comment + docstring that *describes* the no-await rule
    # doesn't trip its own test.
    between = lines[check_line + 1 : set_line]
    code_only = [
        re.sub(r"#.*$", "", re.sub(r"(\".*?\"|'.*?')", "", line)) for line in between
    ]
    blob = "\n".join(code_only)
    assert not re.search(r"\bawait\b", blob), (
        f"`await` snuck between ``Agent.run`` check and set (lines "
        f"{check_line + 1}-{set_line}); see test docstring for why this is "
        f"unsafe"
    )


# --- A31: unified background-cancel predicate ------------------------------


_BgKind = Literal["tool", "persistent_subagent", "detached"]


def _bg_entry(
    task: asyncio.Task[object],
    *,
    kind: _BgKind,
    hidden: bool = False,
) -> BackgroundTaskEntry:
    # ``persistent_run_id`` is mandatory when ``kind="persistent_subagent"``
    # and ignored otherwise; supplying a stub keeps the helper general.
    return BackgroundTaskEntry(
        task=task,
        tool_name="t",
        queue_id="q",
        started=0.0,
        hidden=hidden,
        kind=kind,
        persistent_run_id="run-test" if kind == "persistent_subagent" else "",
    )


@pytest.mark.asyncio
async def test_should_cancel_background_tools_only_mode() -> None:
    """tools_only mode cancels only ``kind == 'tool'`` non-hidden jobs."""
    loop = asyncio.get_running_loop()
    live = cast(asyncio.Task[object], loop.create_future())
    try:
        assert (
            _should_cancel_background(_bg_entry(live, kind="tool"), mode="tools_only")
            is True
        )
        assert (
            _should_cancel_background(
                _bg_entry(live, kind="detached"), mode="tools_only"
            )
            is False
        )
        assert (
            _should_cancel_background(
                _bg_entry(live, kind="persistent_subagent"), mode="tools_only"
            )
            is False
        )
        assert (
            _should_cancel_background(
                _bg_entry(live, kind="tool", hidden=True), mode="tools_only"
            )
            is False
        )
    finally:
        _ = cast(asyncio.Future[object], live).cancel()


@pytest.mark.asyncio
async def test_should_cancel_background_all_mode() -> None:
    """All mode cancels every non-hidden non-persistent live job."""
    loop = asyncio.get_running_loop()
    live = cast(asyncio.Task[object], loop.create_future())
    done_future = loop.create_future()
    done_future.set_result(None)
    done = cast(asyncio.Task[object], done_future)
    try:
        assert (
            _should_cancel_background(_bg_entry(live, kind="tool"), mode="all") is True
        )
        assert (
            _should_cancel_background(_bg_entry(live, kind="detached"), mode="all")
            is True
        )
        assert (
            _should_cancel_background(
                _bg_entry(live, kind="persistent_subagent"), mode="all"
            )
            is False
        )
        assert (
            _should_cancel_background(
                _bg_entry(live, kind="tool", hidden=True), mode="all"
            )
            is False
        )
        assert (
            _should_cancel_background(_bg_entry(done, kind="tool"), mode="all") is False
        )
    finally:
        _ = cast(asyncio.Future[object], live).cancel()


# --- A6: ``job_id_for_call`` is public (no SLF001 noqa needed) -------------


def test_job_id_for_call_is_public_on_agent() -> None:
    """The cross-class helper must not require an SLF001 suppression."""
    assert hasattr(Agent, "job_id_for_call")
    assert not hasattr(Agent, "_job_id_for_call"), (
        "rename leftover: remove the private alias"
    )
    source = Path(Agent.run.__code__.co_filename).read_text(encoding="utf-8")
    assert "SLF001" not in source or "_job_id_for_call" not in source, (
        "SLF001 noqa should be gone with the rename"
    )


# --- W7-α regression tests --------------------------------------------------


@pytest.mark.asyncio
async def test_background_aware_tool_strips_bg_keys_before_inner() -> None:
    """REV7-020: direct ``BackgroundAwareTool.run`` calls must strip bg keys.

    The wrapper advertises ``background`` / ``delay`` in the schema, so
    it -- not just the runtime-path ``_AgentTool`` wrapper -- owns
    stripping them before the inner tool's schema validation sees them.
    """
    inner_calls: list[Mapping[str, object]] = []

    @dataclass(slots=True, kw_only=True)
    class CapturingTool(StubTool):
        @override
        def serialize_key(self, args: Mapping[str, object]) -> str | None:
            del args
            return None

        @override
        async def run(self, args: Mapping[str, object]) -> types.runtime.ToolResult:
            inner_calls.append(dict(args))
            return types.runtime.ToolResult(call_id="", content="ok")

    wrap = BackgroundAwareTool(CapturingTool())
    _ = await wrap.run({"background": True, "msg": "hi"})
    assert inner_calls, "inner tool was not invoked"
    last = inner_calls[-1]
    assert "background" not in last
    assert "delay" not in last
    assert last["msg"] == "hi"


def test_background_aware_tool_rejects_non_object_schema() -> None:
    """REV7-019: ``BackgroundAwareTool`` requires object-typed schemas.

    Injecting ``properties`` into a ``type: "string"`` / ``"array"`` /
    etc. schema silently produces a schema that strict validators reject
    and misrepresents the tool to the model. The wrapper must refuse
    construction up front.
    """

    @dataclass(slots=True, kw_only=True)
    class StringSchemaTool(StubTool):
        directive_schema: JSON = _STRING_SCHEMA

    with pytest.raises(ValueError, match="object-typed"):
        _ = BackgroundAwareTool(StringSchemaTool())


def test_background_aware_tool_accepts_typeless_schema() -> None:
    """REV7-019: schemaless inner schemas still wrap (JSON Schema's "any")."""

    @dataclass(slots=True, kw_only=True)
    class TypelessTool(StubTool):
        directive_schema: JSON = _TYPELESS_SCHEMA

    wrap = BackgroundAwareTool(TypelessTool())
    # Injection still happens; the wrapper's whole job depends on it.
    props = cast(Mapping[str, object], wrap.directive_schema["properties"])
    assert "background" in props
    assert "delay" in props


@pytest.mark.asyncio
async def test_compact_if_needed_circuit_breaker_stashes_synthetic_error(
    tmp_path: Path,
) -> None:
    """REV7-004: the breaker branch sets ``last_compact_error`` so the
    ``_AgentModel.stream`` ``assert last_err is not None`` invariant holds
    even after a coupling slip clears the prior error.
    """

    @dataclass(slots=True, kw_only=True)
    class _AlwaysCompactCompactor:
        def should_compact(
            self,
            current_tokens: int,
            max_request_tokens: int,
            system_tokens: int = 0,
        ) -> bool:
            del current_tokens, max_request_tokens, system_tokens
            return True

        async def compact(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[types.runtime.ModelContextEvent],
            model: object,
            mint_ref: Callable[[], TapeRef],
            custom_instructions: str | None = None,
        ) -> ContextSplice:
            del tape, context, model, mint_ref, custom_instructions
            raise RuntimeError("ignored: breaker is open")

        def maintain(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[types.runtime.ModelContextEvent],
            tools: object,
            mint_ref: Callable[[], TapeRef],
        ) -> tuple[ContextSplice, ...]:
            del tape, context, tools, mint_ref
            return ()

    a = Agent(
        model=StubModel(),
        compactor=_AlwaysCompactCompactor(),
        session_dir=tmp_path,
    )
    # Open the breaker AND clear last_compact_error to simulate the
    # coupling slip the assert is supposed to defend against.
    a.compaction_state.compact_failures = 3
    a.last_compact_error = None
    a.runtime.append_history(types.runtime.UserMessage(text="x"))
    progressed = await a.compact_if_needed(a.runtime.context().messages, a.model)
    assert progressed is False
    assert a.last_compact_error is not None


def test_agent_rejects_zero_max_attempts() -> None:
    """REV7-015: ``max_attempts < 1`` would yield
    ``RetriesExhaustedError("Failed after 0 attempts: None")`` from the
    first send. Validate up front instead.
    """
    with pytest.raises(ValueError, match="max_attempts"):
        _ = Agent(model=StubModel(), max_attempts=0)


def test_agent_summary_failure_falls_back_to_tool_name(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """REV7-006: any ``summary()`` exception falls back to the tool name."""

    @dataclass(slots=True, kw_only=True)
    class BrokenSummaryTool(StubTool):
        @override
        def summary(self, args: Mapping[str, object]) -> str:
            del args
            raise TypeError("author bug")

    a = _build_agent(tools=[BrokenSummaryTool()])
    wrapper = next(t for t in a.runtime.tools_map.values() if t.name == "Echo")
    with caplog.at_level(logging.ERROR, logger="sagent.agent.agent"):
        result = asyncio.new_event_loop().run_until_complete(wrapper.run({}))
    # Tool ran to completion; no ``is_error`` leak from the label failure.
    assert not result.is_error
    assert any("summary" in r.getMessage() for r in caplog.records)


def test_tool_round_cap_pushes_single_error_when_before_spawn_blocks() -> None:
    """REV7-044 verification: when ``_before_tool_spawn`` returns the
    round-cap error, the runtime suppresses ``publish(item)`` for that
    ``ModelResponseComplete``. ``_enforce_caps`` therefore does not see
    that complete event, so no second ``ModelResponseError`` is pushed.

    Documents the observed runtime behavior to guard against a future
    refactor that re-orders publish vs before-spawn and accidentally
    introduces the double-push the original review feared.
    """

    @dataclass(slots=True, kw_only=True)
    class CountingTool(StubTool):
        @override
        def serialize_key(self, args: Mapping[str, object]) -> str | None:
            del args
            return None

        @override
        async def run(self, args: Mapping[str, object]) -> types.runtime.ToolResult:
            del args
            return types.runtime.ToolResult(call_id="", content="ran")

    model = StubModel(
        responses=[
            types.runtime.AssistantMessage(
                tool_calls=(types.runtime.ToolCall(id="c1", name="Echo", args={}),)
            ),
            types.runtime.AssistantMessage(
                tool_calls=(types.runtime.ToolCall(id="c2", name="Echo", args={}),)
            ),
        ]
    )
    a = Agent(model=model, tools=[CountingTool()], max_tool_call_rounds=1)

    async def _drive() -> int:
        count = 0
        async for ev in a.run(types.runtime.UserMessage(text="go")):
            if isinstance(ev, types.runtime.ModelResponseError):
                count += 1
        return count

    error_count = asyncio.new_event_loop().run_until_complete(_drive())
    assert error_count == 1, f"expected 1 ModelResponseError, got {error_count}"


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
