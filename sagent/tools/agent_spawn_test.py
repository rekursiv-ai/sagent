"""Tests for ``tools.agent_spawn``: child-agent factory & dispatch."""

from __future__ import annotations

from collections.abc import Callable, Generator
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import override
from unittest.mock import MagicMock, patch

import asyncio
import itertools
import json
import logging
import time

import pytest

from sagent.agent.agent import Agent
from sagent.agent.state import (
    agent_counter_var,
    agent_label_var,
    agent_path_var,
    agent_registry,
    current_agent_var,
    max_depth_var,
    tool_state_var,
)
from sagent.providers import PROVIDER_NAMES
from sagent.testing import MockModelCaps
from sagent.tools import agent_spawn as _agent_spawn_mod
from sagent.tools.agent_spawn import (
    AgentSpawn,
    ChildStats,
    _ChildForwarder,
    _last_assistant_result,
    _persistent_tasks,
    _pick_field,
)
from sagent.tools.background_task import BackgroundTask
from sagent.types.model import (
    ModelRecipe,
    ModelRequest,
    ModelResponse,
)
from sagent.types.runtime import (
    AgentIdle,
    AgentSendMessage,
    AssistantMessage,
    ChildDoneEvent,
    ChildEvent,
    ModelIdle,
    ModelResponsePartial,
    ModelServiceSuspended,
    NoticeMessage,
    RuntimeEvent,
    SaveSession,
    ServiceErrorSnapshot,
    ToolCall,
    ToolResult,
    UserMessage,
)


_AGENT_SPAWN_LOGGER = _agent_spawn_mod.__name__


@dataclass(slots=True, kw_only=True)
class StubProviderModel(MockModelCaps):
    """Scripted provider model. Always returns one response then `done`."""

    model_id: str = "stub"
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


def _make_parent(model: StubProviderModel | None = None) -> Agent:
    """Build a minimal parent ``Agent`` with no tools."""
    m = model or StubProviderModel(responses=[AssistantMessage(text="root")])
    return Agent(model=m, tools=[])


def _stub_provider_model(model_id: str) -> StubProviderModel:
    """``provider.model`` side_effect: a stub model for any requested id."""
    del model_id
    return StubProviderModel(model_id="stub")


def test_pick_field_priority() -> None:
    assert _pick_field("a", "b", "c") == "a"
    assert _pick_field(None, "b", "c") == "b"
    assert _pick_field(None, None, "c") == "c"
    assert _pick_field(None, None, None) is None


def test_last_assistant_result_finds_last() -> None:
    history = [
        AssistantMessage(text="first"),
        AssistantMessage(text="second"),
    ]
    r = _last_assistant_result(list(history))
    assert r.content == "second"


def test_last_assistant_result_empty_history() -> None:
    r = _last_assistant_result([])
    assert r.content == ""


def test_last_assistant_result_two_agent_sends_returns_last() -> None:
    """Two AgentSends in the same turn -- return the most recent one.

    The child emitted two sends in parallel; the last one in tool-call
    order is the most recent message the child wanted to deliver.
    Returning the first one buries the more current content.
    """
    history = [
        UserMessage(text="kick"),
        AssistantMessage(
            tool_calls=(
                ToolCall(
                    id="t1",
                    name="AgentSend",
                    args={"to": "parent", "content": "first send"},
                ),
                ToolCall(
                    id="t2",
                    name="AgentSend",
                    args={"to": "parent", "content": "second send"},
                ),
            ),
        ),
        ToolResult(call_id="t1", content="ok"),
        ToolResult(call_id="t2", content="ok"),
        AssistantMessage(text="Done."),
    ]
    r = _last_assistant_result(list(history))
    assert r.content == "second send", (
        f"two-AgentSends-in-one-turn must return the last send (most"
        f" recent intent), not the first; got {r.content!r}"
    )


def test_last_assistant_result_does_not_return_stale_agent_send() -> None:
    """An ancient ``AgentSend`` content must NOT shadow a newer text turn.

    The child made an AgentSend in turn 1, then moved on and produced
    a new text-only assistant reply in turn N. The "most recent reply"
    is turn N, not turn 1's AgentSend. The walk must stop at the most
    recent assistant turn, not scan the whole history for any
    AgentSend ever.
    """
    history = [
        UserMessage(text="kick"),
        AssistantMessage(
            tool_calls=(
                ToolCall(
                    id="t1",
                    name="AgentSend",
                    args={"to": "parent", "content": "OLD AGENTSEND"},
                ),
            ),
        ),
        ToolResult(call_id="t1", content="ok"),
        AssistantMessage(text="Done."),
        UserMessage(text="another kick"),
        AssistantMessage(text="real recent reply"),
    ]
    r = _last_assistant_result(list(history))
    assert r.content == "real recent reply", (
        f"stale AgentSend shadowed the actual most recent reply; got {r.content!r}"
    )


def test_last_assistant_result_returns_empty_for_empty_last_text() -> None:
    """If the most recent assistant turn has empty text, return empty --
    don't skip back to an older non-empty turn.

    The blocking-spawn caller uses this as the child's reply. If the
    child finished with an empty turn (e.g. only thinking blocks or
    only non-AgentSend tool calls), the right return is empty -- not
    ancient content from a much earlier turn that the child has since
    moved past.
    """
    history = [
        AssistantMessage(text="old answer the user has since moved past"),
        UserMessage(text="follow-up"),
        AssistantMessage(text=""),
    ]
    r = _last_assistant_result(list(history))
    assert r.content == "", (
        "fallback must respect the actual most-recent assistant turn;"
        f" returning {r.content!r} surfaces stale content"
    )


def test_metadata_basics() -> None:
    t = AgentSpawn()
    assert t.name == "AgentSpawn"
    assert t.tool_id == "application/x-tool-agentspawn"


def test_summary_short_and_long() -> None:
    t = AgentSpawn()
    assert t.summary({"prompt": "go"}) == "AgentSpawn go"
    long = "x" * 100
    s = t.summary({"prompt": long})
    assert s.endswith("...")
    assert t.summary({"prompt": "go", "model_id": "m1"}) == "AgentSpawn go [m1]"
    assert t.summary({}) == "AgentSpawn"


def test_summary_result_off_by_default() -> None:
    t = AgentSpawn()
    assert t.summary_result(ToolResult(call_id="", content="x\ny")) is None
    t.emit_tool_summary = True
    assert t.summary_result(ToolResult(call_id="", content="x\ny")) == "2L"
    assert (
        t.summary_result(ToolResult(call_id="", content=""))
        == "completed with no output"
    )


def test_prompt_no_cap() -> None:
    t = AgentSpawn()
    token = max_depth_var.set(None)
    try:
        assert t.prompt() == ""
    finally:
        max_depth_var.reset(token)


def test_prompt_with_remaining_budget() -> None:
    t = AgentSpawn()
    token = max_depth_var.set(3)
    try:
        p = t.prompt()
    finally:
        max_depth_var.reset(token)
    assert "depth 0/3" in p
    # 3 generations -> plural "generations".
    assert "3 generations" in p


def test_prompt_leaf_at_cap() -> None:
    t = AgentSpawn()
    token = max_depth_var.set(0)
    try:
        p = t.prompt()
    finally:
        max_depth_var.reset(token)
    assert "leaf agent" in p


@contextmanager
def _parent_context(parent: Agent, *, label: str = "Agent") -> Generator[None]:
    """Install minimal parent contextvars for the block."""
    agent_t = current_agent_var.set(parent)
    path_t = agent_path_var.set("")
    label_t = agent_label_var.set(label)
    counter_t = agent_counter_var.set(itertools.count())
    state_t = tool_state_var.set(parent.tool_state)
    try:
        yield
    finally:
        tool_state_var.reset(state_t)
        agent_counter_var.reset(counter_t)
        agent_label_var.reset(label_t)
        agent_path_var.reset(path_t)
        current_agent_var.reset(agent_t)


@pytest.mark.asyncio
async def test_run_depth_cap_exceeded() -> None:
    parent = _make_parent()
    parent.tool_state.depth = 5
    token = max_depth_var.set(2)
    try:
        with _parent_context(parent):
            t = AgentSpawn()
            result = await t.run({"prompt": "task"})
    finally:
        max_depth_var.reset(token)
    assert result.is_error
    assert "max_depth" in result.content


@pytest.mark.asyncio
async def test_run_without_current_agent_returns_error() -> None:
    """Root use (no ``current_agent_var``) must not assertion-crash.

    ``AgentSpawn._run_oneshot_child`` later asserts ``parent_agent is not
    None``; if the run path admits a ``None`` parent we hit a bare
    ``AssertionError`` that escapes the tool envelope. Reject early
    with a clean ``ToolResult(is_error=True)`` so a tool-call from a
    root context surfaces a structured message instead of an unhandled
    exception.
    """
    parent = _make_parent()
    # Install the minimal contextvars so the depth-cap check passes,
    # but explicitly clear ``current_agent_var`` (root use) to model
    # the "tool fired without an active agent" path.
    path_t = agent_path_var.set("")
    label_t = agent_label_var.set("Agent")
    counter_t = agent_counter_var.set(itertools.count())
    state_t = tool_state_var.set(parent.tool_state)
    agent_t = current_agent_var.set(None)
    try:
        t = AgentSpawn()
        result = await t.run({"prompt": "task"})
    finally:
        current_agent_var.reset(agent_t)
        tool_state_var.reset(state_t)
        agent_counter_var.reset(counter_t)
        agent_label_var.reset(label_t)
        agent_path_var.reset(path_t)
    assert result.is_error
    assert "no active agent" in result.content.lower()


@pytest.mark.asyncio
async def test_run_basic_child_returns_last_assistant() -> None:
    """Default-everything spawn inherits parent's model + tools."""
    # StubProviderModel always returns "child-said" so when the child
    # is run it will record "child-said" as its last assistant message.
    parent_model = StubProviderModel(responses=[AssistantMessage(text="child-said")])
    parent = _make_parent(parent_model)
    with _parent_context(parent):
        t = AgentSpawn()
        result = await t.run({"prompt": "do it"})
    assert not result.is_error
    assert result.content == "child-said"


@pytest.mark.asyncio
async def test_non_persistent_child_has_single_registry_label() -> None:
    """Non-persistent AgentSpawn leaves registry ownership to Agent.run."""

    @dataclass(slots=True, kw_only=True)
    class _RegistryInspectingModel(StubProviderModel):
        label_count: int | None = None

        @override
        async def stream(
            self,
            request: ModelRequest,
            publish: Callable[[RuntimeEvent], None] | None = None,
        ) -> ModelResponse:
            child = current_agent_var.get()
            self.label_count = sum(
                1 for agent in agent_registry.values() if agent is child
            )
            return await StubProviderModel.stream(self, request, publish)

    parent_model = _RegistryInspectingModel(
        responses=[AssistantMessage(text="child-said")]
    )
    parent = _make_parent(parent_model)
    with _parent_context(parent):
        result = await AgentSpawn().run({"prompt": "do it"})

    assert not result.is_error
    assert parent_model.label_count == 1


@pytest.mark.asyncio
async def test_oneshot_child_stays_registered_until_explicit_stop() -> None:
    """T1: a oneshot child's registry entry survives its first result.

    ``drive_until_first_idle`` returns the first reply but leaves the
    serve loop live -- so the label is still addressable in
    ``agent_registry`` (by ``/send`` / AgentSend) until an explicit
    shutdown. The prior bug popped the entry the instant the driver task's
    contextvar CM unwound.
    """
    child = Agent(
        model=StubProviderModel(responses=[AssistantMessage(text="answer")]),
        tools=[],
    )
    child.name = "oneshot-child"
    try:
        result = await child.drive_until_first_idle(UserMessage(text="go"))
        assert result.content == "answer"
        assert "oneshot-child" in agent_registry, (
            "oneshot child must remain registered until explicit stop"
        )
        assert agent_registry["oneshot-child"] is child
    finally:
        child.shutdown(force=True)
        drive = child._drive_task
        if drive is not None:
            with suppress(asyncio.CancelledError):
                await drive
        agent_registry.pop("oneshot-child", None)
    assert "oneshot-child" not in agent_registry


@pytest.mark.asyncio
async def test_oneshot_spawn_writes_completed_lifecycle(tmp_path: Path) -> None:
    """T6: a finished oneshot child writes a terminal ``completed`` record.

    The terminal record lets ``load_persistent_agents`` (which keeps only
    ``running`` entries) omit the finished oneshot, so a resume never
    resurrects it.
    """
    parent = Agent(
        model=StubProviderModel(responses=[AssistantMessage(text="root")]),
        tools=[],
        session_dir=tmp_path,
    )
    with _parent_context(parent):
        result = await AgentSpawn().run({"prompt": "do it", "label": "oneshot-w"})
    assert not result.is_error
    lifecycle = [
        json.loads(line)
        for line in (tmp_path / "session.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if "persistent_agent" in line
    ]
    assert lifecycle, "oneshot child wrote no lifecycle record"
    assert lifecycle[-1]["label"] == "oneshot-w"
    assert lifecycle[-1]["state"] == "completed"


@pytest.mark.asyncio
async def test_serviced_forwarder_delivers_first_reply_exactly_once() -> None:
    """T4: the serviced forwarder delivers a child's first reply once, not twice.

    The forwarder pushes exactly one ``AgentSendMessage`` for the child's
    first post-work idle -- the boot idle (empty history) is suppressed
    and the ``skip_first_work_idle`` latch (used when a caller consumes
    the first idle via ``drive_until_first_idle``) prevents a duplicate.
    """
    parent = _make_parent()
    child = Agent(
        model=StubProviderModel(responses=[AssistantMessage(text="first reply")]),
        tools=[],
    )
    # Latched forwarder: the first work idle is consumed elsewhere, so the
    # forwarder must NOT also push it.
    latched = _ChildForwarder(
        parent_agent=parent,
        child=child,
        forward_set=frozenset(),
        stats=ChildStats(label="c", start=time.monotonic()),
        label="c",
        notify_on_asleep=True,
        skip_first_work_idle=True,
    )
    child.runtime.append_history(UserMessage(text="go"))
    child.runtime.append_history(AssistantMessage(text="first reply"))
    latched(AgentIdle())  # first work idle -- latched out
    assert parent.runtime.inbox.empty(), "latched first work idle must not be pushed"
    latched(AgentIdle())  # second work idle -- delivered
    queue = parent.runtime.inbox._queue
    assert queue.qsize() == 1, f"expected exactly one delivery, got {queue.qsize()}"
    msg = queue.get_nowait()
    assert isinstance(msg, AgentSendMessage)
    assert msg.text == "[c is idle] first reply"


@pytest.mark.asyncio
async def test_run_child_model_error_returns_tool_error() -> None:
    """One-shot child model failures must resolve the AgentSpawn call."""

    @dataclass(slots=True, kw_only=True)
    class _FailingModel(StubProviderModel):
        @override
        async def stream(
            self,
            request: ModelRequest,
            publish: Callable[[RuntimeEvent], None] | None = None,
        ) -> ModelResponse:
            del request, publish
            raise RuntimeError("invalid child credentials")

    parent = _make_parent(_FailingModel())
    with _parent_context(parent):
        result = await asyncio.wait_for(
            AgentSpawn().run({"prompt": "do it", "label": "reviewer"}),
            timeout=0.5,
        )

    assert result.is_error
    assert "reviewer" in result.content
    assert "invalid child credentials" in result.content


@pytest.mark.asyncio
async def test_run_unknown_tool_name_errors() -> None:
    parent = _make_parent()
    with _parent_context(parent):
        t = AgentSpawn()
        result = await t.run({"prompt": "x", "tools": ["DoesNotExist"]})
    assert result.is_error
    assert "Unknown tools" in result.content


@pytest.mark.asyncio
async def test_run_missing_model_recipe_when_asking_for_change() -> None:
    parent = _make_parent()
    parent.model_recipe = None
    with _parent_context(parent):
        t = AgentSpawn()
        # LLM asks for a partial model change, parent has no spec to
        # fill in the rest.
        result = await t.run({"prompt": "x", "model_id": "claude"})
    assert result.is_error
    assert "Cannot build a model" in result.content


def test_resolve_tools_inherits_when_names_none() -> None:
    parent = _make_parent()
    t = AgentSpawn()
    out = t._resolve_tools(None, parent)
    # Parent has no tools, so child gets no tools.
    assert out == []


def test_resolve_tools_explicit_empty_honored() -> None:
    parent = _make_parent()
    t = AgentSpawn()
    out = t._resolve_tools([], parent)
    assert out == []


def test_resolve_tools_auto_bundles_background_task_with_agent_spawn() -> None:
    """Granting AgentSpawn must auto-grant BackgroundTask.

    The principle: any agent that can create persistent / background
    work must also be able to list, cancel, and foreground that work.
    Bundling avoids the "spawned a runaway child, no way to kill it"
    failure mode.
    """
    spawn = AgentSpawn()
    parent = Agent(
        model=StubProviderModel(responses=[AssistantMessage(text="root")]),
        tools=[spawn, BackgroundTask()],
    )
    out = spawn._resolve_tools(["AgentSpawn"], parent)
    assert isinstance(out, list)
    assert {t.name for t in out} == {"AgentSpawn", "BackgroundTask"}


def test_resolve_tools_no_bundle_when_agent_spawn_absent() -> None:
    """Without AgentSpawn, no need to bundle BackgroundTask."""
    spawn = AgentSpawn()
    parent = Agent(
        model=StubProviderModel(responses=[AssistantMessage(text="root")]),
        tools=[spawn, BackgroundTask()],
    )
    out = spawn._resolve_tools(["BackgroundTask"], parent)
    assert isinstance(out, list)
    assert {t.name for t in out} == {"BackgroundTask"}


def test_resolve_tools_bundle_respects_explicit_empty() -> None:
    """``names=[]`` returns empty regardless of bundle rule."""
    spawn = AgentSpawn()
    parent = Agent(
        model=StubProviderModel(responses=[AssistantMessage(text="root")]),
        tools=[spawn, BackgroundTask()],
    )
    assert spawn._resolve_tools([], parent) == []


def test_resolve_tools_bundle_when_parent_lacks_background_task() -> None:
    """If parent has AgentSpawn but no BackgroundTask, factory adds a
    fresh BackgroundTask -- the cancel capability is non-negotiable.
    """
    spawn = AgentSpawn()
    parent = Agent(
        model=StubProviderModel(responses=[AssistantMessage(text="root")]),
        tools=[spawn],
    )
    out = spawn._resolve_tools(["AgentSpawn"], parent)
    assert isinstance(out, list)
    assert {t.name for t in out} == {"AgentSpawn", "BackgroundTask"}


def test_resolve_system_llm_wins() -> None:
    parent = _make_parent()
    t = AgentSpawn(system="factory-default")
    assert t._resolve_system("llm-arg", parent) == "llm-arg"


def test_resolve_system_factory_then_parent() -> None:
    parent = _make_parent()
    t = AgentSpawn(system="factory")
    assert t._resolve_system(None, parent) == "factory"
    t2 = AgentSpawn()
    # parent has system="" by default; fallthrough returns "".
    assert t2._resolve_system(None, parent) == ""


def test_resolve_system_no_parent() -> None:
    t = AgentSpawn()
    assert t._resolve_system(None, None) == ""


def test_inherit_factory_wins() -> None:
    parent = _make_parent()
    parent.thinking = "adaptive"
    t = AgentSpawn(thinking="off")
    assert t._inherit("thinking", parent) == "off"


def test_inherit_falls_through_to_parent() -> None:
    parent = _make_parent()
    parent.thinking = "adaptive"
    t = AgentSpawn()
    assert t._inherit("thinking", parent) == "adaptive"


def test_inherit_no_parent() -> None:
    t = AgentSpawn()
    assert t._inherit("thinking", None) is None


def test_resolve_model_rebuilds_fresh_transport_when_spec_matches() -> None:
    """A child inheriting the parent's spec gets its OWN transport, not an alias.

    Regression guard for the shared-subprocess bug: ``_resolve_model`` used
    to return ``parent.model`` verbatim when the resolved provider/auth/
    model_id/account matched the parent. On a subprocess-backed provider
    (AnthropicCLI/GoogleCLI) that aliased every same-model child onto the
    parent's single ``claude`` process and pipe -- serializing (and
    corrupting) N spawns through one transport. A spec that can be rebuilt
    MUST rebuild via ``build_provider(...).model(...)`` so each child owns
    an independent transport.
    """
    spec = ModelRecipe(provider="StubP", auth="env", model_id="stub", account=None)
    parent = Agent(
        model=StubProviderModel(model_id="stub"),
        tools=[],
        model_recipe=spec,
    )
    fake_provider = MagicMock()
    fake_provider.model.side_effect = _stub_provider_model
    t = AgentSpawn()
    with patch(
        "sagent.tools.agent_spawn.build_provider",
        return_value=fake_provider,
    ) as build:
        resolved = t._resolve_model(
            provider="StubP",
            auth="env",
            model_id="stub",
            account=None,
            parent_agent=parent,
        )
    assert isinstance(resolved, tuple)
    model, returned_spec = resolved
    assert model is not parent.model  # fresh transport, not the shared alias
    assert returned_spec == spec
    build.assert_called_once_with("StubP", "env", account=None)


def test_resolve_model_each_child_gets_distinct_transport() -> None:
    """N inherit-spec children resolve to N distinct model objects.

    The load-bearing property for scaling: each spawn's transport must be
    independent so N children run concurrently on N processes rather than
    serializing through one.
    """
    spec = ModelRecipe(provider="StubP", auth="env", model_id="stub", account=None)
    parent = Agent(
        model=StubProviderModel(model_id="stub"),
        tools=[],
        model_recipe=spec,
    )
    fake_provider = MagicMock()
    fake_provider.model.side_effect = _stub_provider_model
    t = AgentSpawn()
    models: list[object] = []
    with patch(
        "sagent.tools.agent_spawn.build_provider",
        return_value=fake_provider,
    ):
        for _ in range(5):
            resolved = t._resolve_model(
                provider=None,
                auth=None,
                model_id=None,
                account=None,
                parent_agent=parent,
            )
            assert isinstance(resolved, tuple)
            models.append(resolved[0])
    assert len({id(m) for m in models}) == 5
    assert all(m is not parent.model for m in models)


def test_resolve_model_no_spec_falls_back_to_parent_model() -> None:
    """A spec-less parent (test harness / raw-Model inject) can't rebuild.

    Without a ``model_recipe`` there is nothing to hand ``build_provider``,
    so aliasing ``parent.model`` is the only option and is correct here --
    the reuse hazard only exists when a rebuildable spec is present.
    """
    parent = _make_parent()  # constructed with no model_recipe
    assert parent.model_recipe is None
    t = AgentSpawn()
    resolved = t._resolve_model(
        provider=None,
        auth=None,
        model_id=None,
        account=None,
        parent_agent=parent,
    )
    assert isinstance(resolved, tuple)
    model, _ = resolved
    assert model is parent.model


@dataclass(slots=True, kw_only=True)
class _FastModel(StubProviderModel):
    """Stub model advertising a fast latency path."""

    latency_modes: tuple[str, ...] = ("fast",)


@dataclass(slots=True, kw_only=True)
class _ThinkingEffortModel(StubProviderModel):
    """Stub model advertising thinking and graded effort."""

    supports_thinking: bool = True
    supports_effort: bool = True
    valid_efforts: tuple[str, ...] = ("low", "high")


def test_build_child_latency_derives_from_fast_model_tag() -> None:
    """A ``+fast`` model-id tag on the child model sets its latency."""
    parent = _make_parent()
    child = AgentSpawn()._build_child(
        system=None,
        child_model=_FastModel(model_id="stub-1+fast"),
        child_spec=None,
        child_tools=[],
        max_rounds=None,
        model_options={},
        parent_agent=parent,
    )
    assert child.latency == "fast"


def test_build_child_applies_thinking_and_effort_from_model_options() -> None:
    """``thinking``/``effort`` options feed the child constructor."""
    parent = _make_parent()
    child = AgentSpawn()._build_child(
        system=None,
        child_model=_ThinkingEffortModel(),
        child_spec=None,
        child_tools=[],
        max_rounds=None,
        model_options={"thinking": True, "effort": "high"},
        parent_agent=parent,
    )
    assert child.thinking == "adaptive"
    assert child.effort == "high"


@pytest.mark.asyncio
async def test_run_redirects_latency_option_to_fast_model_tag() -> None:
    """``model_options.latency`` is gone; the error points at ``+fast``."""
    parent = _make_parent()
    with _parent_context(parent):
        result = await AgentSpawn().run(
            {"prompt": "p", "model_options": {"latency": "fast"}}
        )
    assert result.is_error
    assert "+fast" in result.content


def test_resolve_model_provider_change_without_auth_uses_target_default() -> None:
    parent = _make_parent()
    parent.model_recipe = ModelRecipe(
        provider="OpenAISubscription",
        auth="credentials",
        model_id="gpt-5.5",
        account="work",
    )
    fake_provider = MagicMock()
    fake_provider.model.return_value = StubProviderModel(model_id="gemini-3-pro")
    with patch(
        "sagent.tools.agent_spawn.build_provider",
        return_value=fake_provider,
    ) as build:
        resolved = AgentSpawn()._resolve_model(
            provider="Google",
            auth=None,
            model_id="gemini-3-pro",
            account=None,
            parent_agent=parent,
        )
    assert isinstance(resolved, tuple)
    _, spec = resolved
    assert spec == ModelRecipe(
        provider="Google",
        auth="env",
        model_id="gemini-3-pro",
        account="work",
    )
    build.assert_called_once_with("Google", "env", account="work")


def test_resolve_model_rejects_empty_account() -> None:
    parent = _make_parent()
    parent.model_recipe = ModelRecipe(
        provider="OpenAISubscription",
        auth="credentials",
        model_id="gpt-5.5",
        account="work",
    )
    result = AgentSpawn()._resolve_model(
        provider=None,
        auth=None,
        model_id="gpt-5",
        account="",
        parent_agent=parent,
    )
    assert isinstance(result, ToolResult)
    assert result.is_error
    assert "account cannot be empty" in result.content


@pytest.mark.asyncio
async def test_run_custom_label_used() -> None:
    """Caller-provided label flows through to the run path without errors."""
    parent = _make_parent(StubProviderModel(responses=[AssistantMessage(text="x")]))
    with _parent_context(parent):
        t = AgentSpawn()
        result = await t.run({"prompt": "p", "label": "Sub_007"})
    assert not result.is_error


class _BoomAgent(Agent):
    """Agent whose ``serve_forever`` raises immediately.

    Models a child that fails at startup (e.g. its provider rejects
    the very first model call). Subclassing keeps the override
    type-safe versus monkey-patching an instance attribute.
    """

    @override
    async def serve_forever(self) -> None:
        raise RuntimeError("simulated crash")


@pytest.mark.asyncio
async def test_persistent_run_writes_failed_lifecycle_record(
    tmp_path: Path,
) -> None:
    parent = Agent(
        model=StubProviderModel(responses=[AssistantMessage(text="root")]),
        tools=[],
        session_dir=tmp_path,
    )
    child = _BoomAgent(
        model=StubProviderModel(responses=[AssistantMessage(text="x")]),
        tools=[],
    )

    t = AgentSpawn()
    with _parent_context(parent):
        result = t._spawn_serviced(child, "doomed", "p")
        assert not result.is_error
        task = _persistent_tasks.get("doomed")
        assert task is not None
        await task

    lifecycle = [
        json.loads(line)
        for line in (tmp_path / "session.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if "persistent_agent" in line
    ]
    assert lifecycle[-1]["label"] == "doomed"
    assert lifecycle[-1]["state"] == "failed"


@pytest.mark.asyncio
async def test_persistent_run_logs_unhandled_exception(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A crash inside ``serve_forever`` must be logged with the agent label.

    The ``_run`` wrapper around ``serve_forever`` previously had only
    ``try/finally`` -- exceptions from the child were stored on the
    asyncio task and silently lost. Persistent agents that crashed at
    startup appeared "alive" in the registry but produced no work and
    no diagnostic. This test pins the contract: any exception out of
    ``serve_forever`` lands in the logger at ERROR or higher, with the
    failing child's label in the message.
    """
    parent = _make_parent()
    child = _BoomAgent(
        model=StubProviderModel(responses=[AssistantMessage(text="x")]),
        tools=[],
    )

    t = AgentSpawn()
    with (
        _parent_context(parent),
        caplog.at_level(logging.ERROR, logger=_AGENT_SPAWN_LOGGER),
    ):
        result = t._spawn_serviced(child, "doomed", "p")
        assert not result.is_error
        task = _persistent_tasks.get("doomed")
        assert task is not None
        # The whole point: ``_run`` must catch -- never re-raise -- so
        # the task completes cleanly with the exception logged. In
        # production, nothing awaits this task; an uncaught raise
        # would be lost to asyncio's "exception never retrieved"
        # GC warning.
        await task

    assert any(
        "doomed" in record.getMessage() and record.levelno >= logging.ERROR
        for record in caplog.records
    ), [r.getMessage() for r in caplog.records]


@pytest.mark.asyncio
async def test_persistent_child_does_not_overwrite_parent_registry_entry() -> None:
    """A persistent child must NOT clobber ``agent_registry['Agent']``.

    Repro for the AgentSend-doesn't-wake-the-parent bug: the persistent
    child's task inherits ``agent_label_var`` from the parent task
    (``asyncio.create_task`` copies the current context). In
    ``Agent._install_contextvars`` the line ``base_label =
    agent_label_var.get("") or self.name`` then returns the parent's
    label (``"Agent"``) rather than the child's own ``self.name``
    (``"child1"``). For persistent agents, ``label = base_label``, so
    ``agent_registry["Agent"] = child`` silently overwrites the parent's
    entry. After this, every ``AgentSend("Agent", ...)`` from any sub
    (including back from this same child) routes to the child, not to
    the running parent. The parent never wakes.

    The fix: serviced agents have a definite ``self.name`` set by
    ``_spawn_serviced``; the label must come from that, not from the
    inherited ``agent_label_var``.
    """
    parent = _make_parent()
    with _parent_context(parent):
        # Simulate the parent having registered itself under "Agent"
        # (which it does inside its own serve_forever's
        # _install_contextvars). The parent context manager already
        # sets agent_label_var="Agent" so the child task will inherit
        # it via asyncio.create_task.
        agent_registry["Agent"] = parent
        child = Agent(
            model=StubProviderModel(responses=[AssistantMessage(text="x")]),
            tools=[],
        )
        t = AgentSpawn()
        try:
            result = t._spawn_serviced(child, "child1", "p")
            assert not result.is_error
            # Yield to scheduler so the child's task enters
            # serve_forever -> _install_contextvars (the bug site).
            for _ in range(20):
                await asyncio.sleep(0.005)
                if agent_registry.get("Agent") is not parent:
                    break

            assert agent_registry.get("Agent") is parent, (
                "persistent child clobbered parent's 'Agent' registry"
                f" entry; got {agent_registry.get('Agent')!r}"
            )
            assert agent_registry.get("child1") is child, (
                "persistent child must remain reachable under its own"
                f" label; got {agent_registry.get('child1')!r}"
            )
        finally:
            child.shutdown(force=True)
            task = _persistent_tasks.get("child1")
            if task is not None:
                try:
                    await asyncio.wait_for(task, timeout=2.0)
                except (TimeoutError, Exception):
                    _ = task.cancel()
            agent_registry.pop("Agent", None)
            agent_registry.pop("child1", None)


@pytest.mark.asyncio
async def test_spawn_persistent_rejects_job_prefix_label() -> None:
    parent = _make_parent()
    child = _BoomAgent(
        model=StubProviderModel(responses=[AssistantMessage(text="x")]),
        tools=[],
    )

    t = AgentSpawn()
    with _parent_context(parent):
        result = t._spawn_serviced(child, "job-helper", "prompt")
    assert result.is_error
    assert "reserved" in result.content


@pytest.mark.asyncio
async def test_spawn_persistent_rejects_duplicate_label() -> None:
    """Duplicate label must error -- silent overwrite orphans the prior agent.

    A persistent agent's label is its addressable identity for AgentSend.
    Silently doing ``agent_registry[label] = child`` over an existing
    entry orphans the prior agent (its background task keeps running but
    is unreachable) and routes every subsequent ``AgentSend`` to the new
    instance only. Worse, when the prior agent eventually terminates,
    its ``finally`` block does ``agent_registry.pop(label, None)`` --
    popping the NEW entry, leaving the new agent unreachable too.

    Reject the duplicate spawn so the user kills the prior agent first.
    """
    parent = _make_parent()
    child1 = _BoomAgent(
        model=StubProviderModel(responses=[AssistantMessage(text="x")]),
        tools=[],
    )
    child2 = _BoomAgent(
        model=StubProviderModel(responses=[AssistantMessage(text="x")]),
        tools=[],
    )

    t = AgentSpawn()
    with _parent_context(parent):
        first = t._spawn_serviced(child1, "dup-label", "p1")
        assert not first.is_error
        second = t._spawn_serviced(child2, "dup-label", "p2")
        assert second.is_error, f"second spawn must error, got {second.content!r}"
        assert "dup-label" in second.content
        # The first agent's task is still scheduled; let it run to
        # completion so cleanup pops its own registry entry rather than
        # leaving litter for the next test.
        task1 = _persistent_tasks.get("dup-label")
        assert task1 is not None
        await task1


# -- notify_on_asleep --------------------------------------------------------
#
# AgentSpawn(persistent=True, notify_on_asleep=True) installs a forwarder
# that pushes a UserMessage("[<label> is idle]") into the parent's inbox
# every time the child's runtime publishes AgentIdle (edge-triggered: one
# per idle transition, suppressed until the next drain returns work).
#
# Two layers tested:
#   1. Forwarder unit: AgentIdle in -> inbox push out (or no push when
#      notify_on_asleep=False).
#   2. End-to-end: _spawn_serviced(..., notify_on_asleep=True) wires
#      the child's serve_forever such that the parent inbox observes
#      the notification after the child finishes its seeded prompt.


def _make_forwarder(
    parent: Agent,
    label: str,
    *,
    notify_on_asleep: bool,
    child: Agent | None = None,
) -> _ChildForwarder:
    """Construct a _ChildForwarder bound to ``parent``, no verbosity gating."""
    return _ChildForwarder(
        parent_agent=parent,
        child=child or _make_parent(),
        forward_set=frozenset(),
        stats=ChildStats(label=label, start=time.monotonic()),
        label=label,
        notify_on_asleep=notify_on_asleep,
    )


def test_forwarder_notify_on_asleep_false_skips_inbox_push() -> None:
    """Default: AgentIdle never reaches the parent inbox."""
    parent = _make_parent()
    fwd = _make_forwarder(parent, "child", notify_on_asleep=False)

    fwd(AgentIdle())

    assert parent.runtime.inbox.empty(), (
        "AgentIdle leaked into parent inbox with notify_on_asleep=False"
    )


def test_forwarder_notify_on_asleep_true_suppresses_boot_idle() -> None:
    """AgentIdle on a child with EMPTY history is the boot state, not a
    busy→idle transition. The forwarder must NOT push a notification
    until the child has done at least one turn -- otherwise every
    freshly-spawned persistent child spams the parent with a useless
    "[child is idle]" the moment it starts.
    """
    parent = _make_parent()
    fwd = _make_forwarder(parent, "child", notify_on_asleep=True)

    fwd(AgentIdle())

    assert parent.runtime.inbox.empty(), (
        "boot AgentIdle (empty child history) leaked into parent inbox"
    )


def test_forwarder_notify_on_asleep_true_pushes_after_first_turn() -> None:
    """notify_on_asleep=True: AgentIdle AFTER the child has produced
    history pushes an AgentSendMessage attributed to the child. This is
    the real busy→idle edge.

    Attribution: source must be the child's label so the renderer's
    AgentSendMessage bar surfaces the right "from" header and the
    parent's model context disambiguates the idle ping from anonymous
    human input.
    """
    parent = _make_parent()
    child = _make_parent()
    child.runtime.append_history(UserMessage(text="kick off"))
    child.runtime.append_history(AssistantMessage(text="reply"))
    fwd = _make_forwarder(parent, "child", notify_on_asleep=True, child=child)

    fwd(AgentIdle())

    queue = parent.runtime.inbox._queue
    assert queue.qsize() == 1
    msg = queue.get_nowait()
    assert isinstance(msg, AgentSendMessage)
    assert msg.source == "child"
    assert msg.text == "[child is idle] reply"


def test_forwarder_notify_on_asleep_includes_last_assistant_text() -> None:
    """When the child has assistant history, the idle payload carries it.

    This is the safety net for a persistent child that replies with
    plain assistant text instead of calling ``AgentSend`` back -- the
    forwarder ferries the last text into the parent's inbox so the
    parent's model still sees the content.
    """
    parent = _make_parent()
    child = _make_parent()
    child.runtime.append_history(AssistantMessage(text="hello parent"))
    fwd = _make_forwarder(parent, "child", notify_on_asleep=True, child=child)

    fwd(AgentIdle())

    queue = parent.runtime.inbox._queue
    assert queue.qsize() == 1
    msg = queue.get_nowait()
    assert isinstance(msg, AgentSendMessage)
    assert msg.source == "child"
    assert msg.text == "[child is idle] hello parent"


def test_forwarder_notify_on_asleep_prefers_agent_send_content() -> None:
    """When child's last reply was an ``AgentSend`` to the parent, the
    idle payload carries that content -- not the subsequent ack/"Done."
    assistant turn that fires after the mandatory follow-up round.

    Mechanism: ``AgentSend`` is a tool call. After the child's tool
    dispatches it, a ``ToolResult`` sits at history tail and triggers
    a second model call (per ``_should_call_model``); the child's
    answering text ("Done.") becomes the most recent assistant message.
    The idle payload's old behavior was to ferry that ack into the
    parent's inbox, dropping the actual report carried in
    ``AgentSend.args["content"]``.
    """
    parent = _make_parent()
    child = _make_parent()
    child.runtime.append_history(UserMessage(text="user kicked off"))
    child.runtime.append_history(
        AssistantMessage(
            tool_calls=(
                ToolCall(
                    id="t1",
                    name="AgentSend",
                    args={"to": "parent", "content": "report body"},
                ),
            ),
        )
    )
    child.runtime.append_history(ToolResult(call_id="t1", content="Delivered"))
    child.runtime.append_history(AssistantMessage(text="Done."))
    fwd = _make_forwarder(parent, "child", notify_on_asleep=True, child=child)

    fwd(AgentIdle())

    queue = parent.runtime.inbox._queue
    assert queue.qsize() == 1
    msg = queue.get_nowait()
    assert isinstance(msg, AgentSendMessage)
    assert msg.source == "child"
    assert msg.text == "[child is idle] report body", (
        f"forwarder must surface AgentSend content, not the ack 'Done.'; got {msg.text!r}"
    )


def test_forwarder_notify_on_asleep_one_push_per_event_after_first_turn() -> None:
    """Each AgentIdle after the child has history is independently
    translated; edge-trigger lives in the runtime, not the forwarder.

    The forwarder is stateless w.r.t. AgentIdle ONCE the boot
    suppression has passed (child history non-empty). Each runtime-
    delivered AgentIdle then produces one inbox push.
    """
    parent = _make_parent()
    child = _make_parent()
    child.runtime.append_history(AssistantMessage(text="real reply"))
    fwd = _make_forwarder(parent, "child", notify_on_asleep=True, child=child)

    fwd(AgentIdle())
    fwd(AgentIdle())
    fwd(AgentIdle())

    assert parent.runtime.inbox._queue.qsize() == 3


def test_forwarder_notify_on_asleep_ignores_other_events() -> None:
    """ModelIdle and other non-AgentIdle events do not push to the inbox
    even when notify_on_asleep=True.
    """
    parent = _make_parent()
    fwd = _make_forwarder(parent, "child", notify_on_asleep=True)

    fwd(ModelIdle())

    assert parent.runtime.inbox.empty()


def test_forwarder_always_forwards_model_service_suspended() -> None:
    parent = _make_parent()
    seen: list[ChildEvent] = []
    parent.runtime.observers.append(
        lambda event: seen.append(event) if isinstance(event, ChildEvent) else None
    )
    fwd = _make_forwarder(parent, "child", notify_on_asleep=False)
    suspended = ModelServiceSuspended(
        provider="OpenAISubscription",
        auth="credentials",
        account="default",
        model_id="gpt-5.5",
        retry_at=1_800_000_000.0,
        delay_sec=120.0,
        server_supplied=True,
        error=ServiceErrorSnapshot(
            type_name="RateLimitError",
            message="limited",
            status=429,
        ),
    )

    fwd(suspended)

    assert len(seen) == 1
    assert seen[0].label == "child"
    assert seen[0].inner is suspended


def test_forwarder_always_forwards_usage_notice() -> None:
    # A child burns the same provider quota as the root; its near-limit usage
    # advisory must reach the parent even at verbosity 0 (empty forward_set).
    parent = _make_parent()
    seen: list[ChildEvent] = []
    parent.runtime.observers.append(
        lambda event: seen.append(event) if isinstance(event, ChildEvent) else None
    )
    fwd = _make_forwarder(parent, "child", notify_on_asleep=False)
    notice = NoticeMessage(text="[usage: 7d window 89% used]", tier="advisory")

    fwd(notice)

    assert len(seen) == 1
    assert seen[0].label == "child"
    assert seen[0].inner is notice


@dataclass(slots=True, kw_only=True)
class _RatioStubModel(StubProviderModel):
    """Stub whose token estimator is ``len(text) // ratio_divisor``.

    Distinguishes a model-derived child response-token count from the
    forwarder's old hardcoded ``chars // 4``.
    """

    ratio_divisor: int = 2

    @override
    def approx_text_tokens(self, text: str) -> int:
        return len(text) // self.ratio_divisor


def _child_done_tokens(parent: Agent, fwd: _ChildForwarder) -> int:
    """Run ``emit_done`` and return the published ``ChildDoneEvent.tokens``."""
    seen: list[ChildDoneEvent] = []
    parent.runtime.observers.append(
        lambda e: seen.append(e) if isinstance(e, ChildDoneEvent) else None
    )
    fwd.emit_done()
    assert len(seen) == 1
    return seen[0].tokens


def test_forwarder_response_tokens_use_child_model_estimator_not_chars4() -> None:
    """``ChildDoneEvent.tokens`` must use the child model's tokenizer over
    the whole streamed text, not a hardcoded ``chars // 4``.

    A child model at 2 chars/token over 800 streamed chars is 400 tokens;
    ``// 4`` would report 200.
    """
    child = _make_parent(_RatioStubModel(ratio_divisor=2))
    parent = _make_parent()
    fwd = _make_forwarder(parent, "child", notify_on_asleep=False, child=child)

    fwd(ModelResponsePartial(text="x" * 500))
    fwd(ModelResponsePartial(text="y" * 300))

    assert _child_done_tokens(parent, fwd) == child.model.approx_text_tokens("z" * 800)


def test_forwarder_response_tokens_tokenize_whole_not_per_chunk() -> None:
    """Tiny chunks must not floor to zero.

    A truncating estimator (``int(len/3)``) over nine 1-char chunks reads
    0 per chunk but 3 for the joined text; the forwarder must tokenize the
    accumulated whole.
    """

    @dataclass(slots=True, kw_only=True)
    class _Floor3Model(StubProviderModel):
        model_id: str = "floor3"

        @override
        def approx_text_tokens(self, text: str) -> int:
            return int(len(text) / 3)

    child = _make_parent(_Floor3Model())
    parent = _make_parent()
    fwd = _make_forwarder(parent, "child", notify_on_asleep=False, child=child)
    for _ in range(9):
        fwd(ModelResponsePartial(text="x"))
    assert _child_done_tokens(parent, fwd) == 3


@pytest.mark.asyncio
async def test_persistent_spawn_session_root_dir_uses_label_path(
    tmp_path: Path,
) -> None:
    parent_dir = tmp_path / "parent"
    parent = Agent(
        model=StubProviderModel(responses=[AssistantMessage(text="root")]),
        tools=[],
        session_dir=parent_dir,
    )
    child = Agent(
        model=StubProviderModel(responses=[AssistantMessage(text="done")]),
        tools=[],
    )
    spawn = AgentSpawn(session_root_dir=tmp_path / "children")

    with _parent_context(parent):
        result = spawn._spawn_serviced(child, "fix-tools", "do work")

    task = _persistent_tasks.get("fix-tools")
    spawned = agent_registry.get("fix-tools")
    try:
        assert not result.is_error
        assert isinstance(spawned, Agent)
        child_session_dir = spawned.session_dir
        assert child_session_dir is not None
        assert child_session_dir == tmp_path / "children" / "fix-tools"
        spawned.runtime.append_history(UserMessage(text="persisted child message"))
        spawned.runtime.publish(SaveSession())
        assert (child_session_dir / "session.jsonl").exists()
        lifecycle = [
            json.loads(line)
            for line in (parent_dir / "session.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if "persistent_agent" in line
        ]
        assert lifecycle[-1]["session_dir"] == str(tmp_path / "children" / "fix-tools")
    finally:
        if spawned is not None:
            spawned.shutdown(force=True)
        if task is not None:
            _ = task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        agent_registry.pop("fix-tools", None)


@pytest.mark.asyncio
async def test_persistent_spawn_persists_base_system_without_ipc_rule(
    tmp_path: Path,
) -> None:
    parent = Agent(
        model=StubProviderModel(responses=[AssistantMessage(text="root")]),
        tools=[],
        session_dir=tmp_path,
    )
    child = Agent(
        model=StubProviderModel(responses=[AssistantMessage(text="done")]),
        tools=[],
        system="base system prompt",
    )
    spawn = AgentSpawn()

    with _parent_context(parent, label="parent-label"):
        result = spawn._spawn_serviced(child, "fix-tools", "do work")

    task = _persistent_tasks.get("fix-tools")
    try:
        assert not result.is_error
        lifecycle = [
            json.loads(line)
            for line in (tmp_path / "session.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if "persistent_agent" in line
        ]
        assert lifecycle[-1]["system"] == "base system prompt"
        assert "persistent agent" not in lifecycle[-1]["system"]
    finally:
        child.shutdown(force=True)
        if task is not None:
            _ = task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        agent_registry.pop("fix-tools", None)


@pytest.mark.asyncio
async def test_persistent_spawn_writes_parent_lifecycle_record(tmp_path: Path) -> None:
    parent = Agent(
        model=StubProviderModel(responses=[AssistantMessage(text="root")]),
        tools=[],
        session_dir=tmp_path,
    )
    child = Agent(
        model=StubProviderModel(
            model_id="gpt-5.5",
            responses=[AssistantMessage(text="done")],
        ),
        model_recipe=ModelRecipe(
            provider="OpenAISubscription",
            auth="credentials",
            model_id="gpt-5.5",
            account="default",
        ),
        tools=[],
        system="child system",
    )
    spawn = AgentSpawn()

    with _parent_context(parent):
        result = spawn._spawn_serviced(
            child, "fix-tools", "do work", notify_on_asleep=False
        )

    task = _persistent_tasks.get("fix-tools")
    try:
        assert not result.is_error
        lines = (tmp_path / "session.jsonl").read_text(encoding="utf-8").splitlines()
        lifecycle = [json.loads(line) for line in lines if "persistent_agent" in line]
        assert len(lifecycle) == 1
        record = lifecycle[0]
        assert record["label"] == "fix-tools"
        assert record["state"] == "running"
        assert record["provider"] == "OpenAISubscription"
        assert record["auth"] == "credentials"
        assert record["account"] == "default"
        assert record["model_id"] == "gpt-5.5"
        assert record["tools"] == []
        assert record["system"] == child.base_system_spec
        assert record["notify_on_asleep"] is False
        assert record["session_dir"] == str(child.session_dir or "")
    finally:
        child.shutdown(force=True)
        if task is not None:
            _ = task.cancel()
            with suppress(asyncio.CancelledError):
                await task


@pytest.mark.asyncio
async def test_persistent_spawn_model_error_reaches_parent_inbox() -> None:
    """A serviced child whose model call fails must reach the parent's inbox.

    A model-level failure (bad model id, revoked creds) leaves the child
    parked on ``AWAIT_RECOVERY`` -- it never publishes ``AgentIdle`` and its
    ``serve_forever`` does not crash -- so the ``notify_on_asleep`` idle ping
    never fires. Without a dedicated error edge the parent's model never learns
    the child died: the forwarder renders ``ModelResponseError`` to the pane
    only. This asserts the child-fatal error is delivered to the parent's inbox
    (its decision layer), the only channel the parent's model reads.
    """

    @dataclass(slots=True, kw_only=True)
    class _FailingModel(StubProviderModel):
        @override
        async def stream(
            self,
            request: ModelRequest,
            publish: Callable[[RuntimeEvent], None] | None = None,
        ) -> ModelResponse:
            del request, publish
            raise RuntimeError("unsupported model for this account")

    parent = _make_parent()
    child = Agent(model=_FailingModel(), tools=[])
    t = AgentSpawn()

    with _parent_context(parent):
        result = t._spawn_serviced(
            child, "doomed-child", "do work", notify_on_asleep=True
        )
        assert not result.is_error
        task = _persistent_tasks.get("doomed-child")
        assert task is not None

        deadline = asyncio.get_running_loop().time() + 2.0
        while parent.runtime.inbox.empty():
            if asyncio.get_running_loop().time() >= deadline:
                pytest.fail(
                    "parent inbox never received the child's model-error notification"
                )
            await asyncio.sleep(0.01)

        first = parent.runtime.inbox._queue.get_nowait()
        assert isinstance(first, AgentSendMessage)
        assert first.source == "doomed-child"
        assert "unsupported model for this account" in first.text, (
            f"unexpected payload: {first.text!r}"
        )

        child.shutdown(force=True)
        try:
            await asyncio.wait_for(task, timeout=2.0)
        except (TimeoutError, Exception):
            _ = task.cancel()


@pytest.mark.asyncio
async def test_persistent_spawn_with_notify_on_asleep_notifies_parent() -> None:
    """End-to-end: a persistent child seeded with a prompt completes one
    round, becomes idle, and the parent inbox receives the notification
    with the child's last assistant text embedded.
    """
    parent = _make_parent()
    child = Agent(
        model=StubProviderModel(responses=[AssistantMessage(text="done")]),
        tools=[],
    )
    t = AgentSpawn()

    with _parent_context(parent):
        result = t._spawn_serviced(
            child, "watcher-child", "do work", notify_on_asleep=True
        )
        assert not result.is_error
        task = _persistent_tasks.get("watcher-child")
        assert task is not None

        # Wait until the parent inbox has at least one notification.
        deadline = asyncio.get_running_loop().time() + 2.0
        while parent.runtime.inbox.empty():
            if asyncio.get_running_loop().time() >= deadline:
                pytest.fail("parent inbox never received the AgentIdle notification")
            await asyncio.sleep(0.01)

        # First message is the idle ping carrying the child's last
        # assistant text; subsequent pings are possible if the agent
        # cycles, but at least one must be present.
        first = parent.runtime.inbox._queue.get_nowait()
        assert isinstance(first, AgentSendMessage)
        assert first.source == "watcher-child"
        assert first.text == "[watcher-child is idle] done", (
            f"unexpected payload: {first.text!r}"
        )

        # Tear down: child has no further work, but the persistent loop
        # keeps running. Force shutdown so the task drains.
        child.shutdown(force=True)
        try:
            await asyncio.wait_for(task, timeout=2.0)
        except (TimeoutError, Exception):
            _ = task.cancel()


@pytest.mark.asyncio
async def test_persistent_spawn_notify_on_asleep_false_stays_silent() -> None:
    """Explicit ``notify_on_asleep=False`` suppresses idle pings.

    The default is now True (so a parent that forgets the flag still
    hears from its child); this test pins the opt-out path.
    """
    parent = _make_parent()
    child = Agent(
        model=StubProviderModel(responses=[AssistantMessage(text="done")]),
        tools=[],
    )
    t = AgentSpawn()

    with _parent_context(parent):
        result = t._spawn_serviced(
            child, "quiet-child", "do work", notify_on_asleep=False
        )
        assert not result.is_error
        task = _persistent_tasks.get("quiet-child")
        assert task is not None

        # Let the child run to idle. We can't easily detect idleness
        # from outside without an observer, so let the task complete
        # its initial prompt and then wait a small real period for any
        # spurious inbox push to surface.
        for _ in range(20):
            await asyncio.sleep(0.01)
            if not child.runtime.inbox.empty():
                continue
            # Heuristic: model_call drained and inbox empty -> idle
            # transition almost certainly fired by now.
            if child.runtime.model_call is None and not child.history == []:  # noqa: SIM201 -- explicit non-empty check
                break

        assert parent.runtime.inbox.empty(), (
            "parent inbox received an unexpected push with notify_on_asleep=False"
        )

        child.shutdown(force=True)
        try:
            await asyncio.wait_for(task, timeout=2.0)
        except (TimeoutError, Exception):
            _ = task.cancel()


@pytest.mark.asyncio
async def test_persistent_spawn_augments_child_system_prompt() -> None:
    """Persistent children get an IPC rule appended to their system prompt
    so the child's LLM knows raw assistant text is invisible to the parent
    and that ``AgentSend(to=<parent>)`` is the only reliable reply path.
    """
    parent = _make_parent()
    child = Agent(
        model=StubProviderModel(responses=[AssistantMessage(text="done")]),
        tools=[],
        system="base system prompt",
    )
    t = AgentSpawn()

    with _parent_context(parent, label="parent-label"):
        result = t._spawn_serviced(child, "augmented-child", "do work")
        assert not result.is_error
        task = _persistent_tasks.get("augmented-child")
        assert task is not None

        prompt = child.system
        assert "base system prompt" in prompt
        assert "persistent agent" in prompt
        assert "AgentSend(to='parent-label'" in prompt
        assert "invisible" in prompt

        child.shutdown(force=True)
        try:
            await asyncio.wait_for(task, timeout=2.0)
        except (TimeoutError, Exception):
            _ = task.cancel()


@pytest.mark.asyncio
async def test_persistent_spawn_return_value_names_reply_channel() -> None:
    """The tool result must tell the parent's LLM how replies arrive,
    so it doesn't assume the next assistant message will surface
    automatically.
    """
    parent = _make_parent()
    child = Agent(
        model=StubProviderModel(responses=[AssistantMessage(text="done")]),
        tools=[],
    )
    t = AgentSpawn()

    with _parent_context(parent):
        result = t._spawn_serviced(child, "channel-child", "do work")
        assert not result.is_error
        assert "AgentSend" in result.content
        assert "channel-child" in result.content
        assert "is idle" in result.content

        child.shutdown(force=True)
        task = _persistent_tasks.get("channel-child")
        if task is not None:
            try:
                await asyncio.wait_for(task, timeout=2.0)
            except (TimeoutError, Exception):
                _ = task.cancel()


def test_directive_schema_documents_notify_on_asleep() -> None:
    """The directive schema must advertise the parameter so the LLM
    can discover and emit it. We verify via string-level inspection
    because the schema's nested JSON value type (recursive union) makes
    structural narrowing fragile across type checkers.
    """
    rendered = repr(AgentSpawn().directive_schema)
    assert "'notify_on_asleep'" in rendered
    # Assert the actual description text, not a bare keyword: the param is
    # documented as persistent-only and as defaulting to true.
    lowered = rendered.lower()
    assert "persistent only" in lowered
    assert "the default" in lowered


def test_directive_schema_prefers_subscription_providers() -> None:
    """The provider/auth descriptions must steer the LLM toward
    subscription providers so it doesn't default to env-API-key auth
    on hosts where only a CLI subscription is configured. Regression
    guard: session b0e1ced6 spawned ``provider="OpenAI", auth="env"``
    on a host with no API key set, when a subscription variant was
    available in the schema enumeration.
    """
    rendered = repr(AgentSpawn().directive_schema)
    assert "Subscription" in rendered
    assert "credentials" in rendered


def test_allow_providers_filters_schema_enumeration() -> None:
    """Restricting ``allow_providers`` narrows the provider enum
    rendered in the schema description, hiding providers the host has
    no credentials for. Companion to ``--allow-providers`` CLI flag.
    """
    spawn = AgentSpawn(allow_providers=("OpenAISubscription",))
    rendered = repr(spawn.directive_schema)
    assert "``OpenAISubscription``" in rendered
    assert "DashScope" not in rendered
    assert "MiniMax" not in rendered
    assert "Moonshot" not in rendered


def test_allow_providers_default_includes_all() -> None:
    """Default construction (no ``allow_providers``) enumerates every
    provider known to ``sagent.providers``.
    """
    spawn = AgentSpawn()
    rendered = repr(spawn.directive_schema)
    for name in PROVIDER_NAMES:
        assert f"``{name}``" in rendered, f"missing {name}"


def test_resolve_model_rejects_provider_outside_allow_list() -> None:
    """An explicit ``provider`` not in the allow list is rejected with
    a ``ToolResult`` error rather than reaching ``build_provider``.
    """
    parent = _make_parent()
    t = AgentSpawn(allow_providers=("OpenAISubscription",))
    result = t._resolve_model(
        provider="Anthropic",
        auth="env",
        model_id="gpt-5.5",
        account=None,
        parent_agent=parent,
    )
    assert isinstance(result, ToolResult)
    assert result.is_error
    assert "not in the allowed list" in result.content
    assert "Anthropic" in result.content


def test_resolve_model_parent_provider_always_allowed() -> None:
    """The parent's own provider is always accepted, even when not
    in the allow list -- inheritance must keep working. The child gets a
    FRESH transport built from the inherited spec, never the parent's alias.
    """
    spec = ModelRecipe(provider="StubP", auth="env", model_id="stub", account=None)
    parent_model = StubProviderModel(model_id="stub")
    parent = Agent(
        model=parent_model,
        tools=[],
        model_recipe=spec,
    )
    fake_provider = MagicMock()
    fake_provider.model.side_effect = _stub_provider_model
    t = AgentSpawn(allow_providers=("OpenAISubscription",))
    with patch(
        "sagent.tools.agent_spawn.build_provider",
        return_value=fake_provider,
    ) as build:
        resolved = t._resolve_model(
            provider="StubP",
            auth="env",
            model_id="stub",
            account=None,
            parent_agent=parent,
        )
    assert isinstance(resolved, tuple)
    model, returned_spec = resolved
    assert model is not parent_model  # fresh transport, parent provider allowed
    assert returned_spec == spec
    build.assert_called_once_with("StubP", "env", account=None)


@pytest.mark.asyncio
async def test_spawned_child_writes_own_session_jsonl(tmp_path: Path) -> None:
    """Child agents persist their own ``session.jsonl`` under the parent's dir.

    Regression test for session 8588644a: subagents did all the real work
    (Read/Edit/Write) but their tape was never written to disk because the
    persistence observer was only installed by the CLI for the root agent.
    With ``Agent.__init__`` auto-installing the observer when ``session_dir``
    is set, every child constructed by ``AgentSpawn`` -- which already
    threads a derived ``session_dir`` through ``_child_session_dir`` --
    self-persists. This test proves the chain works end-to-end.
    """
    parent_model = StubProviderModel(responses=[AssistantMessage(text="child-said")])
    parent = Agent(model=parent_model, tools=[], session_dir=tmp_path)
    with _parent_context(parent):
        t = AgentSpawn()
        result = await t.run({"prompt": "do it"})
    assert not result.is_error
    # Inherited layout: ``<parent_session_dir>/<child_uuid>/session.jsonl``.
    # The parent's session_dir already encodes its identity, so no
    # redundant parent_id prepend.
    child_files = list(tmp_path.glob("*/session.jsonl"))  # noqa: ASYNC240 -- post-spawn test assertion; no live concurrency to protect
    assert child_files, (
        f"expected at least one child session.jsonl directly under"
        f" {tmp_path}; found nothing"
    )
    # The child's session.jsonl must carry real records, not be empty.
    content = child_files[0].read_text(encoding="utf-8")
    assert content.strip(), "child session.jsonl is empty"


@pytest.mark.asyncio
async def test_run_rejects_empty_account() -> None:
    """``account=""`` must be a hard error, not silent inheritance.

    Mirrors ``AgentSelf.run({"account": ""})`` which already errors at
    ``agent_self.py:485-490``. Pre-fix the local ``account`` had
    already been collapsed to ``None`` by ``opt_str`` before the
    reject branch ran, so the branch was unreachable and the spawn
    silently inherited the parent's account.
    """
    parent = _make_parent()
    with _parent_context(parent):
        t = AgentSpawn()
        result = await t.run(
            {"prompt": "p", "model_id": "gpt-5", "account": ""},
        )
    assert result.is_error
    assert "account cannot be empty" in result.content


@pytest.mark.asyncio
async def test_run_rejects_malformed_tools_string() -> None:
    """``tools="Read"`` is malformed -- not an array. Fail closed.

    Pre-fix this silently became ``tools=None`` → inherit the
    parent's full toolset. The LLM asked to restrict tools but got
    the unrestricted set: a permission gap.
    """
    parent = _make_parent()
    with _parent_context(parent):
        t = AgentSpawn()
        result = await t.run({"prompt": "p", "tools": "Read"})
    assert result.is_error
    assert "tools" in result.content


@pytest.mark.asyncio
async def test_run_rejects_max_tool_call_rounds_zero() -> None:
    """Schema declares ``minimum: 1``; runtime must enforce.

    Pre-fix ``max_tool_call_rounds=0`` slipped through and produced
    an agent that immediately hit its cap before any model round.
    """
    parent = _make_parent()
    with _parent_context(parent):
        t = AgentSpawn()
        result = await t.run({"prompt": "p", "max_tool_call_rounds": 0})
    assert result.is_error
    assert "max_tool_call_rounds" in result.content


@pytest.mark.asyncio
async def test_run_rejects_negative_max_depth() -> None:
    """Schema declares ``minimum: 0``; runtime must enforce."""
    parent = _make_parent()
    with _parent_context(parent):
        t = AgentSpawn()
        result = await t.run({"prompt": "p", "max_depth": -1})
    assert result.is_error
    assert "max_depth" in result.content


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
