"""Tests for ``tools.agent_spawn``: child-agent factory & dispatch."""

from __future__ import annotations

from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import override

import asyncio
import itertools
import logging

import pytest

from sagent.agent.agent import Agent
from sagent.agent.state import agent_registry
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
from sagent.tools.core import (
    agent_counter_var,
    agent_label_var,
    agent_path_var,
    current_agent_var,
    max_depth_var,
    tool_state_var,
)
from sagent.types.history import (
    AssistantMessage,
    ToolResult,
    UserMessage,
)
from sagent.types.model import ModelRequest, ModelResponse, ModelSpec
from sagent.types.runtime import AgentIdle, ModelIdle


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
        result = t._spawn_persistent(child, "doomed", "p")
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
@pytest.mark.real_sleep
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

    The fix: persistent agents have a definite ``self.name`` set by
    ``_spawn_persistent``; the label must come from that, not from the
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
            result = t._spawn_persistent(child, "child1", "p")
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
                except (TimeoutError, Exception):  # noqa: BLE001
                    _ = task.cancel()
            agent_registry.pop("Agent", None)
            agent_registry.pop("child1", None)


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
        first = t._spawn_persistent(child1, "dup-label", "p1")
        assert not first.is_error
        second = t._spawn_persistent(child2, "dup-label", "p2")
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
#   2. End-to-end: _spawn_persistent(..., notify_on_asleep=True) wires
#      the child's serve_forever such that the parent inbox observes
#      the notification after the child finishes its seeded prompt.


def _make_forwarder(
    parent: Agent, label: str, *, notify_on_asleep: bool
) -> _ChildForwarder:
    """Construct a _ChildForwarder bound to ``parent``, no verbosity gating."""
    import time as _time  # noqa: PLC0415 -- isolated to test helper

    return _ChildForwarder(
        parent_agent=parent,
        forward_set=frozenset(),
        stats=ChildStats(label=label, start=_time.monotonic()),
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


def test_forwarder_notify_on_asleep_true_pushes_one_user_message() -> None:
    """notify_on_asleep=True: one AgentIdle -> one UserMessage on parent."""
    parent = _make_parent()
    fwd = _make_forwarder(parent, "child", notify_on_asleep=True)

    fwd(AgentIdle())

    queue = parent.runtime.inbox._queue
    assert queue.qsize() == 1
    msg = queue.get_nowait()
    assert isinstance(msg, UserMessage)
    assert msg.text == "[child is idle]"


def test_forwarder_notify_on_asleep_one_push_per_event() -> None:
    """Each AgentIdle is independently translated; edge-trigger lives in
    the runtime (which we trust), not the forwarder.

    Validates the forwarder is stateless w.r.t. AgentIdle -- it pushes on
    every event the runtime delivers, relying on the runtime's
    ``_was_idle`` flag to control cadence.
    """
    parent = _make_parent()
    fwd = _make_forwarder(parent, "child", notify_on_asleep=True)

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


@pytest.mark.asyncio
@pytest.mark.real_sleep
async def test_persistent_spawn_with_notify_on_asleep_notifies_parent() -> None:
    """End-to-end: a persistent child seeded with a prompt completes one
    round, becomes idle, and the parent inbox receives the notification.
    """
    parent = _make_parent()
    child = Agent(
        model=StubProviderModel(responses=[AssistantMessage(text="done")]),
        tools=[],
    )
    t = AgentSpawn()

    with _parent_context(parent):
        result = t._spawn_persistent(
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

        # First message is the idle ping; subsequent pings are possible
        # if the agent cycles, but at least one must be present.
        first = parent.runtime.inbox._queue.get_nowait()
        assert isinstance(first, UserMessage)
        assert first.text == "[watcher-child is idle]", (
            f"unexpected payload: {first.text!r}"
        )

        # Tear down: child has no further work, but the persistent loop
        # keeps running. Force shutdown so the task drains.
        child.shutdown(force=True)
        try:
            await asyncio.wait_for(task, timeout=2.0)
        except (TimeoutError, Exception):  # noqa: BLE001
            _ = task.cancel()


@pytest.mark.asyncio
@pytest.mark.real_sleep
async def test_persistent_spawn_without_notify_on_asleep_silent() -> None:
    """Without the flag, the parent inbox stays empty even after the
    child becomes idle.
    """
    parent = _make_parent()
    child = Agent(
        model=StubProviderModel(responses=[AssistantMessage(text="done")]),
        tools=[],
    )
    t = AgentSpawn()

    with _parent_context(parent):
        result = t._spawn_persistent(child, "quiet-child", "do work")
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
            "parent inbox received an unexpected push without notify_on_asleep"
        )

        child.shutdown(force=True)
        try:
            await asyncio.wait_for(task, timeout=2.0)
        except (TimeoutError, Exception):  # noqa: BLE001
            _ = task.cancel()


def test_directive_schema_documents_notify_on_asleep() -> None:
    """The directive schema must advertise the parameter so the LLM
    can discover and emit it. We verify via string-level inspection
    because the schema's nested JSON value type (recursive union) makes
    structural narrowing fragile across type checkers.
    """
    rendered = repr(AgentSpawn.directive_schema)
    assert "'notify_on_asleep'" in rendered
    assert "'persistent only'" in rendered.lower() or "persistent" in rendered.lower()


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
