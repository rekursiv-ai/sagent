"""Tests for ``tools.agent_spawn``: child-agent factory & dispatch."""

from __future__ import annotations

from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass, field

import itertools

import pytest

from sagent.agent.agent import Agent
from sagent.testing import MockModelCaps
from sagent.tools.agent_spawn import (
    AgentSpawn,
    _last_assistant_result,
    _pick_field,
)
from sagent.tools.core import (
    agent_counter_var,
    agent_label_var,
    agent_path_var,
    current_agent_var,
    max_depth_var,
    tool_state_var,
)
from sagent.types.history import AssistantMessage, ToolResult
from sagent.types.model import ModelRequest, ModelResponse, ModelSpec


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


def _make_parent(model: StubProviderModel | None = None) -> Agent:
    """Build a minimal parent ``Agent`` with no tools."""
    m = model or StubProviderModel(responses=[AssistantMessage(text="root")])
    return Agent(model=m, tools=[])


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


def test_metadata_basics() -> None:
    t = AgentSpawn()
    assert t.name == "AgentSpawn"
    assert t.tool_id == "application/x-tool-agentspawn"
    assert t.supports_microcompaction is False


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
def _parent_context(parent: Agent) -> Generator[None]:
    """Install minimal parent contextvars for the block."""
    agent_t = current_agent_var.set(parent)
    path_t = agent_path_var.set("")
    label_t = agent_label_var.set("Agent")
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
async def test_run_unknown_tool_name_errors() -> None:
    parent = _make_parent()
    with _parent_context(parent):
        t = AgentSpawn()
        result = await t.run({"prompt": "x", "tools": ["DoesNotExist"]})
    assert result.is_error
    assert "Unknown tools" in result.content


@pytest.mark.asyncio
async def test_run_missing_model_spec_when_asking_for_change() -> None:
    parent = _make_parent()
    parent.model_spec = None
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


def test_resolve_model_reuses_parent_when_spec_matches() -> None:
    spec = ModelSpec(provider="StubP", auth="env", model_id="stub", account="")
    parent = Agent(
        model=StubProviderModel(model_id="stub"),
        tools=[],
        model_spec=spec,
    )
    t = AgentSpawn()
    resolved = t._resolve_model(
        provider="StubP",
        auth="env",
        model_id="stub",
        account="",
        parent_agent=parent,
    )
    assert isinstance(resolved, tuple)
    model, returned_spec = resolved
    assert model is parent.model
    assert returned_spec == spec


def test_resolve_model_no_args_reuses_parent_model() -> None:
    parent = _make_parent()
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


@pytest.mark.asyncio
async def test_run_custom_label_used() -> None:
    """Caller-provided label flows through to the run path without errors."""
    parent = _make_parent(StubProviderModel(responses=[AssistantMessage(text="x")]))
    with _parent_context(parent):
        t = AgentSpawn()
        result = await t.run({"prompt": "p", "label": "Sub_007"})
    assert not result.is_error


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
