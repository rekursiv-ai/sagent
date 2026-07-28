"""Tests for ``repl.input_pane``: pump dispatch + input-pane rendering."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock, patch

import asyncio

from prompt_toolkit.formatted_text import FormattedText

import pytest

from sagent.agent.agent import Agent as _RealAgent
from sagent.agent.background import BackgroundTaskEntry
from sagent.agent.state import (
    AgentLike,
    agent_label_var,
    agent_registry,
)
from sagent.repl import input_pane as repl_input_mod
from sagent.repl.input_pane import (
    REPL_PUMP_KEY,
    PromptToolkitInputSource,
    StubInputSource,
    _dispatch,
    _input_pump,
    _resolve_targets,
    render_input_pane,
    spawn_repl_pump,
)
from sagent.repl.input_queues import InputQueues, QueuedInputBlock
from sagent.repl.render import RecordingPrinter
from sagent.repl.slash import (
    Clear as SlashClear,
    Compact as SlashCompact,
    Defer as SlashDefer,
    Halt as SlashHalt,
    Help as SlashHelp,
    Kill as SlashKill,
    Login as SlashLogin,
    ModelSwitch as SlashModelSwitch,
    Quit as SlashQuit,
    Recompact as SlashRecompact,
    Send as SlashSend,
    Tasks as SlashTasks,
    Text as SlashText,
    Unknown as SlashUnknown,
)
from sagent.types.runtime import (
    AgentSendMessage,
    Clear,
    Compact,
    Recompact,
    RuntimeEvent,
    UserDeferredMessage,
    UserMessage,
)


if TYPE_CHECKING:
    from sagent.agent.agent import Agent


class _StubInbox:
    """Inbox stub that just records pushes."""

    def __init__(self) -> None:
        self.items: list[RuntimeEvent] = []

    def push_back(self, item: RuntimeEvent) -> None:
        self.items.append(item)


@dataclass(slots=True, kw_only=True)
class _StubRuntime:
    inbox: _StubInbox = field(default_factory=_StubInbox)


@dataclass(slots=True, kw_only=True)
class _StubAgent:
    """Minimal Agent surface for ``_dispatch``."""

    name: str = "AgentA"
    runtime: _StubRuntime = field(default_factory=_StubRuntime)
    halted: int = 0
    killed_all: int = 0
    killed: list[str] = field(default_factory=list)
    shutdown_calls: list[bool] = field(default_factory=list)
    background_registry: dict[str, BackgroundTaskEntry] = field(default_factory=dict)
    is_subagent: bool = False

    def halt(self) -> None:
        self.halted += 1

    def kill_all_tools(self) -> None:
        self.killed_all += 1

    def kill_tool(self, qid: str) -> None:
        self.killed.append(qid)

    def shutdown(self, *, force: bool = False) -> None:
        self.shutdown_calls.append(force)

    @property
    def background(self) -> dict[str, BackgroundTaskEntry]:
        return self.background_registry

    def register_background(self, job_id: str, entry: BackgroundTaskEntry) -> None:
        self.background_registry[job_id] = entry

    def cancel_background(self, job_id: str) -> None:
        self.background_registry.pop(job_id, None)


def _agent() -> Agent:
    return cast("Agent", _StubAgent())


def _persistent_agent() -> Agent:
    return cast("Agent", _StubAgent(is_subagent=True))


@pytest.mark.asyncio
async def test_stub_input_source_yields_then_none() -> None:
    src = StubInputSource(["one", "two", None])
    assert await src.next_line() == "one"
    assert await src.next_line() == "two"
    assert await src.next_line() is None
    # Exhausted queue keeps returning None.
    assert await src.next_line() is None


@pytest.mark.asyncio
async def test_dispatch_quit_shuts_down_and_exits() -> None:
    a = _agent()
    stub = cast(_StubAgent, a)
    exit_ = await _dispatch(a, SlashQuit(), None)
    assert exit_ is True
    assert stub.shutdown_calls == [False]


@pytest.mark.asyncio
async def test_dispatch_halt_self_calls_halt() -> None:
    a = _agent()
    stub = cast(_StubAgent, a)
    _ = await _dispatch(a, SlashHalt(target=""), None)
    assert stub.halted == 1


@pytest.mark.asyncio
async def test_dispatch_halt_bare_name_does_not_halt_when_registry_label_suffixed() -> (
    None
):
    a = _agent()
    stub = cast(_StubAgent, a)
    p = RecordingPrinter()
    with patch(
        "sagent.repl.input_pane.agent_registry",
        new={"AgentA_2": a},
    ):
        _ = await _dispatch(a, SlashHalt(target="AgentA"), p)
    assert stub.halted == 0
    assert any("no matching subagents" in e for e in p.tool_errors)


@pytest.mark.asyncio
async def test_dispatch_halt_by_registry_label() -> None:
    a = _agent()
    child = _persistent_agent()
    child_stub = cast(_StubAgent, child)
    p = RecordingPrinter()
    with patch(
        "sagent.repl.input_pane.agent_registry",
        new={"fix-tools": child},
    ):
        _ = await _dispatch(a, SlashHalt(target="fix-tools"), p)
    assert child_stub.halted == 1
    assert "[/halt fix-tools] halted" in p.slash_blocks


@pytest.mark.asyncio
async def test_dispatch_halt_unknown_agent_writes_error() -> None:
    a = _agent()
    p = RecordingPrinter()
    empty: dict[str, AgentLike] = {}
    with patch(
        "sagent.repl.input_pane.agent_registry",
        new=empty,
    ):
        _ = await _dispatch(a, SlashHalt(target="Other"), p)
    assert any("no matching subagents" in e for e in p.tool_errors)


@pytest.mark.asyncio
async def test_dispatch_kill_job_id() -> None:
    a = _agent()
    stub = cast(_StubAgent, a)
    p = RecordingPrinter()
    _ = await _dispatch(a, SlashKill(target="job-1"), p)
    assert stub.killed == ["job-1"]
    assert any("cancelled job-1" in line for line in p.slash_blocks)


@pytest.mark.asyncio
async def test_dispatch_kill_namespaced_subagent_job() -> None:
    a = _agent()
    child = _persistent_agent()
    child_stub = cast(_StubAgent, child)
    p = RecordingPrinter()
    with patch(
        "sagent.repl.input_pane.agent_registry",
        new={"fix-tools": child},
    ):
        _ = await _dispatch(a, SlashKill(target="fix-tools/job-1"), p)
    assert child_stub.killed == ["job-1"]
    assert "[/kill fix-tools/job-1] cancelled" in p.slash_blocks


@pytest.mark.asyncio
async def test_dispatch_kill_owner_slash_job_unknown_owner_surfaces_error() -> None:
    """F7: ``/kill bogus/qid`` with an unknown owner must surface an error.

    Previously the code fell through to ``_resolve_targets`` and then
    ``agent.kill_tool(target)``, silently no-op'ing while the user
    believed the kill landed.
    """
    a = _agent()
    stub = cast(_StubAgent, a)
    p = RecordingPrinter()
    empty: dict[str, AgentLike] = {}
    with patch(
        "sagent.repl.input_pane.agent_registry",
        new=empty,
    ):
        _ = await _dispatch(a, SlashKill(target="bogus/qid"), p)
    assert stub.killed == [], (
        f"unknown owner must not trigger self.kill_tool fallback; got {stub.killed!r}"
    )
    assert any("unknown owner" in error for error in p.tool_errors), (
        f"expected 'unknown owner' error; got {p.tool_errors!r}"
    )


@pytest.mark.asyncio
async def test_dispatch_kill_all() -> None:
    a = _agent()
    stub = cast(_StubAgent, a)
    p = RecordingPrinter()
    _ = await _dispatch(a, SlashKill(target="all"), p)
    assert stub.killed_all == 1
    assert any("cancelled all" in line for line in p.slash_blocks)


@pytest.mark.asyncio
async def test_dispatch_kill_all_clears_local_repl_queues() -> None:
    """F136: ``/kill all`` must also clear REPL-local urgent/deferred staging.

    Otherwise the user sees a "pending" preview that survives a wipe
    they explicitly requested, contradicting the kill-all gesture.
    """
    a = _agent()
    stub = cast(_StubAgent, a)
    p = RecordingPrinter()
    queues = InputQueues(
        queue=QueuedInputBlock(text="staged urgent"),
        deferred=QueuedInputBlock(text="staged later"),
    )
    _ = await _dispatch(a, SlashKill(target="all"), p, queues=queues)
    assert stub.killed_all == 1
    assert not queues.has_any(), (
        f"/kill all must clear REPL queues; got queue={queues.queue!r}"
        f" deferred={queues.deferred!r}"
    )


@pytest.mark.asyncio
async def test_dispatch_kill_persistent_subagent() -> None:
    a = _agent()
    child = _persistent_agent()
    child_stub = cast(_StubAgent, child)
    p = RecordingPrinter()
    registry = {"fix-tools": child}
    with (
        patch(
            "sagent.repl.input_pane.agent_registry",
            new=registry,
        ),
        patch(
            "sagent.tools.background_task.agent_registry",
            new=registry,
        ),
    ):
        _ = await _dispatch(a, SlashKill(target="fix-tools"), p)
    assert child_stub.shutdown_calls == [True]
    assert "[/kill fix-tools] cancelled" in p.slash_blocks


@pytest.mark.asyncio
async def test_dispatch_clear_pushes_clear_event() -> None:
    a = _agent()
    stub = cast(_StubAgent, a)
    p = RecordingPrinter()
    _ = await _dispatch(a, SlashClear(), p)
    assert any(isinstance(i, Clear) for i in stub.runtime.inbox.items)
    assert any("history cleared" in line for line in p.slash_blocks)


@pytest.mark.asyncio
async def test_dispatch_compact_pushes_compact_with_args() -> None:
    a = _agent()
    stub = cast(_StubAgent, a)
    p = RecordingPrinter()
    _ = await _dispatch(a, SlashCompact(args="hints"), p)
    pushed = stub.runtime.inbox.items
    assert any(isinstance(i, Compact) and i.args == "hints" for i in pushed)
    assert any("/compact" in line for line in p.slash_blocks)


@pytest.mark.asyncio
async def test_dispatch_compact_no_args_no_note() -> None:
    a = _agent()
    p = RecordingPrinter()
    _ = await _dispatch(a, SlashCompact(args=""), p)
    line = next(line for line in p.slash_blocks if "/compact" in line)
    assert "(" not in line


@pytest.mark.asyncio
async def test_dispatch_recompact_pushes_recompact() -> None:
    a = _agent()
    stub = cast(_StubAgent, a)
    p = RecordingPrinter()
    _ = await _dispatch(a, SlashRecompact(args="redo"), p)
    pushed = stub.runtime.inbox.items
    assert any(isinstance(i, Recompact) for i in pushed)


@pytest.mark.asyncio
async def test_dispatch_text_pushes_user_message() -> None:
    a = _agent()
    stub = cast(_StubAgent, a)
    _ = await _dispatch(a, SlashText(content="hi"), None)
    pushed = stub.runtime.inbox.items
    assert any(isinstance(i, UserMessage) and i.text == "hi" for i in pushed)


@pytest.mark.asyncio
async def test_dispatch_defer_pushes_user_deferred_message() -> None:
    """F31: ``/defer <text>`` pushes ``UserDeferredMessage``.

    Aligns the slash with Tab's ``_kb_defer`` (which also stages as
    deferred) and with ``Defer``'s docstring -- "drains at
    ``AgentIdle``". Previously the slash pushed ``UserQueuedMessage``,
    which drains at the chat-safe boundary; that divergence let
    headless ``/defer`` callers preempt mid-round while the equivalent
    Tab gesture did not.
    """
    a = _agent()
    stub = cast(_StubAgent, a)
    _ = await _dispatch(a, SlashDefer(content="for later"), None)
    pushed = stub.runtime.inbox.items
    assert any(
        isinstance(i, UserDeferredMessage) and i.text == "for later" for i in pushed
    ), f"expected UserDeferredMessage; got {pushed!r}"


def test_resolve_targets_includes_oneshot_subagent() -> None:
    """T2: a live oneshot subagent is a valid ``/send`` target.

    Under the unified lifecycle every spawned agent -- oneshot or
    serviced -- is a live, inbox-serviceable loop registered under its
    stable label. Target resolution gates on ``is_subagent``, not on the
    old serviced-only ``_persistent`` flag, so a oneshot child resolves.
    """
    oneshot = cast("Agent", _StubAgent(name="oneshot-child", is_subagent=True))
    agent_registry.update({"oneshot-child": oneshot})
    try:
        assert _resolve_targets("oneshot-child") == ["oneshot-child"]
    finally:
        agent_registry.clear()


def test_resolve_targets_star_excludes_caller_and_root() -> None:
    """T5: ``*`` never resolves the caller's own label or the root agent.

    ``_resolve_targets`` gates on ``_is_serviceable`` (a subagent that is
    not the caller and not the root), so a fan-out cannot route to self
    or to the root.
    """
    root = _agent()  # is_subagent=False -> the root
    me = cast("Agent", _StubAgent(name="me", is_subagent=True))
    peer = cast("Agent", _StubAgent(name="peer", is_subagent=True))
    agent_registry.update({"root": root, "me": me, "peer": peer})
    label_token = agent_label_var.set("me")
    try:
        resolved = _resolve_targets("*")
        assert "root" not in resolved, "root must not be a target"
        assert "me" not in resolved, "caller must not target itself"
        assert resolved == ["peer"]
    finally:
        agent_label_var.reset(label_token)
        agent_registry.clear()


def test_resolve_targets_supports_exact_glob_brace_and_regex() -> None:
    child1 = _persistent_agent()
    child2 = _persistent_agent()
    non_persistent = _agent()
    agent_registry.update(
        {
            "fix-tools": child1,
            "fix-compact": child2,
            "helper": non_persistent,
        }
    )
    try:
        assert _resolve_targets("fix-tools") == ["fix-tools"]
        assert _resolve_targets("fix-*") == ["fix-tools", "fix-compact"]
        assert _resolve_targets("{fix-compact,helper,fix-tools}") == [
            "fix-compact",
            "fix-tools",
        ]
        assert _resolve_targets("/compact$/") == ["fix-compact"]
    finally:
        agent_registry.clear()


def test_resolve_targets_empty_regex_does_not_match_everything() -> None:
    """REPL-020: ``/`` parses as an empty regex that matches every label.

    Empty-pattern regex search returns True on every string, silently
    fanning ``/halt /`` (or ``/send / ...``) out to every persistent
    subagent. Reject patterns shorter than ``/x/`` to keep the regex
    surface meaningfully filterable.
    """
    child1 = _persistent_agent()
    child2 = _persistent_agent()
    agent_registry.update({"fix-tools": child1, "fix-compact": child2})
    try:
        assert _resolve_targets("/") == [], (
            "bare '/' must not match every persistent subagent"
        )
        assert _resolve_targets("//") == [], (
            "empty-body '//' regex must not match every persistent subagent"
        )
    finally:
        agent_registry.clear()


@pytest.mark.asyncio
async def test_dispatch_halt_all_targets_every_persistent_subagent() -> None:
    """REPL-041: ``/halt all`` must halt every persistent subagent.

    Mirrors ``/kill all``. Without a special case, the literal target
    string "all" falls through to ``_resolve_targets`` which only
    matches a registry label called ``all`` -- almost never what the
    user meant.
    """
    a = _agent()
    child1 = _persistent_agent()
    child2 = _persistent_agent()
    child1_stub = cast(_StubAgent, child1)
    child2_stub = cast(_StubAgent, child2)
    p = RecordingPrinter()
    with patch(
        "sagent.repl.input_pane.agent_registry",
        new={"fix-tools": child1, "fix-compact": child2},
    ):
        _ = await _dispatch(a, SlashHalt(target="all"), p)
    assert child1_stub.halted == 1, (
        f"fix-tools must be halted by /halt all; halted={child1_stub.halted}"
    )
    assert child2_stub.halted == 1, (
        f"fix-compact must be halted by /halt all; halted={child2_stub.halted}"
    )


@pytest.mark.asyncio
async def test_dispatch_halt_all_never_halts_self() -> None:
    """T5: ``/halt all`` halts peers but never the caller itself.

    The caller is registered as a subagent under its own label; the
    ``_is_serviceable`` gate excludes it, so ``/halt all`` from that
    caller must not call ``halt`` on itself.
    """
    me = _persistent_agent()
    peer = _persistent_agent()
    me_stub = cast(_StubAgent, me)
    peer_stub = cast(_StubAgent, peer)
    label_token = agent_label_var.set("me")
    with patch(
        "sagent.repl.input_pane.agent_registry",
        new={"me": me, "peer": peer},
    ):
        try:
            _ = await _dispatch(me, SlashHalt(target="all"), None)
        finally:
            agent_label_var.reset(label_token)
    assert me_stub.halted == 0, "/halt all must never halt the caller itself"
    assert peer_stub.halted == 1, "/halt all must halt peer subagents"


@pytest.mark.asyncio
async def test_dispatch_send_pushes_agent_send_message_with_source() -> None:
    """``/send <child>`` routes a parent->child handoff, not anonymous user
    input.

    The child must see an ``AgentSendMessage`` attributed to the
    parent's name so the renderer (`repl/render.py`) and replay
    (`repl/replay.py`) draw the attributed bar, and so the child's
    model context distinguishes parent text from human input.
    """
    a = _agent()
    stub_parent = cast(_StubAgent, a)
    stub_parent.name = "Parent"
    child = _persistent_agent()
    child_stub = cast(_StubAgent, child)
    p = RecordingPrinter()
    agent_registry["fix-tools"] = child
    try:
        _ = await _dispatch(a, SlashSend(target="fix-tools", content="continue"), p)
    finally:
        agent_registry.clear()

    pushed = child_stub.runtime.inbox.items
    matches = [
        item
        for item in pushed
        if isinstance(item, AgentSendMessage)
        and item.source == "Parent"
        and item.text == "continue"
    ]
    assert matches, (
        f"expected one AgentSendMessage(source='Parent', text='continue'); got {pushed!r}"
    )
    assert "[/send fix-tools] sent" in p.slash_blocks


@pytest.mark.asyncio
async def test_dispatch_send_model_switch_routes_to_child() -> None:
    a = _agent()
    child = _persistent_agent()
    p = RecordingPrinter()
    agent_registry["fix-tools"] = child
    _ = repl_input_mod._run_repl.do_switch_model  # type: ignore[attr-defined] -- trigger proxy import
    try:
        with patch.object(
            repl_input_mod._run_repl,  # type: ignore[attr-defined] -- module-internal access by design
            "do_switch_model",
        ) as mock:
            _ = await _dispatch(
                a,
                SlashSend(target="fix-tools", content="/model claude-opus-4-7"),
                p,
            )
    finally:
        agent_registry.clear()

    args = mock.call_args.args
    assert args[0] is child
    assert args[1] == "claude-opus-4-7"


@pytest.mark.asyncio
async def test_dispatch_send_unknown_writes_error() -> None:
    a = _agent()
    p = RecordingPrinter()
    _ = await _dispatch(a, SlashSend(target="missing", content="hello"), p)
    assert any("no matching subagents" in error for error in p.tool_errors)


@pytest.mark.asyncio
async def test_dispatch_send_invalid_target_regex_writes_target_error() -> None:
    a = _agent()
    p = RecordingPrinter()
    exit_ = await _dispatch(a, SlashSend(target="/[/", content="hello"), p)
    assert exit_ is False
    assert any("invalid target regex" in error for error in p.tool_errors)


@pytest.mark.asyncio
async def test_dispatch_help_writes_help() -> None:
    a = _agent()
    p = RecordingPrinter()
    _ = await _dispatch(a, SlashHelp(), p)
    assert any("/help" in line for line in p.lines)


@pytest.mark.asyncio
async def test_dispatch_tasks_calls_run_repl_format_tasks() -> None:
    a = _agent()
    p = RecordingPrinter()
    _ = repl_input_mod._run_repl.format_tasks  # type: ignore[attr-defined] -- trigger proxy import
    with patch.object(
        repl_input_mod._run_repl,  # type: ignore[attr-defined] -- module-internal access by design
        "format_tasks",
        return_value="tasks listing",
    ) as mock:
        _ = await _dispatch(a, SlashTasks(), p)
    assert mock.called
    assert "tasks listing" in p.lines


@pytest.mark.asyncio
async def test_dispatch_unknown_writes_error() -> None:
    a = _agent()
    p = RecordingPrinter()
    _ = await _dispatch(a, SlashUnknown(text="oops bad cmd"), p)
    assert "oops bad cmd" in p.tool_errors


@pytest.mark.asyncio
async def test_dispatch_model_switch_calls_run_repl() -> None:
    a = _agent()
    p = RecordingPrinter()
    # Force resolution of the lazy module proxy so patching takes effect.
    _ = repl_input_mod._run_repl.do_switch_model  # type: ignore[attr-defined] -- proxy attr access triggers import
    with patch.object(
        repl_input_mod._run_repl,  # type: ignore[attr-defined] -- module-internal access by design
        "do_switch_model",
    ) as mock:
        _ = await _dispatch(a, SlashModelSwitch(args="claude-opus-4-7"), p)
    args = mock.call_args.args
    assert args[1] == "claude-opus-4-7"


@pytest.mark.asyncio
async def test_dispatch_login_calls_run_repl() -> None:
    a = _agent()
    p = RecordingPrinter()
    _ = repl_input_mod._run_repl.do_login  # type: ignore[attr-defined] -- trigger proxy import
    with patch.object(
        repl_input_mod._run_repl,  # type: ignore[attr-defined] -- module-internal access by design
        "do_login",
    ) as mock:
        _ = await _dispatch(a, SlashLogin(), p)
    assert mock.called


@pytest.mark.asyncio
async def test_dispatch_login_flushes_local_deferred_queue() -> None:
    a = _agent()
    stub = cast(_StubAgent, a)
    queues = InputQueues(deferred=QueuedInputBlock(text="retry after login"))
    _ = repl_input_mod._run_repl.do_login  # type: ignore[attr-defined] -- trigger proxy import
    with patch.object(
        repl_input_mod._run_repl,  # type: ignore[attr-defined] -- module-internal access by design
        "do_login",
    ):
        _ = await _dispatch(a, SlashLogin(), None, queues=queues)
    assert not queues.has_any()
    assert any(
        isinstance(item, UserDeferredMessage) and item.text == "retry after login"
        for item in stub.runtime.inbox.items
    )


def test_repl_pump_key_is_stable() -> None:
    # Used by ``spawn_repl_pump`` and the orchestrator to address the
    # hidden background entry; verify it doesn't drift.
    assert REPL_PUMP_KEY == "__repl_pump__"


def test_dispatch_module_exports() -> None:
    # Coverage for the public ``__all__`` keeping discovery clean.
    exports: list[str] = list(repl_input_mod.__all__)
    assert "REPL_PUMP_KEY" in exports
    assert "InputSource" in exports
    assert "StubInputSource" in exports
    assert "spawn_repl_pump" in exports


@pytest.mark.asyncio
async def test_input_pump_none_line_shuts_down() -> None:
    a = _agent()
    stub = cast(_StubAgent, a)
    src = StubInputSource([None])
    await _input_pump(a, src, None, None)
    assert stub.shutdown_calls == [False]


@pytest.mark.asyncio
async def test_input_pump_empty_line_skipped() -> None:
    a = _agent()
    stub = cast(_StubAgent, a)
    src = StubInputSource(["", None])
    await _input_pump(a, src, None, None)
    # Empty line returned None from parse_slash and was ignored;
    # second None triggers shutdown.
    assert stub.shutdown_calls == [False]


@pytest.mark.asyncio
async def test_input_pump_quit_exits_loop() -> None:
    a = _agent()
    stub = cast(_StubAgent, a)
    src = StubInputSource(["/quit"])
    await _input_pump(a, src, None, None)
    assert stub.shutdown_calls == [False]


@pytest.mark.asyncio
async def test_input_pump_text_lines_pushed() -> None:
    a = _agent()
    stub = cast(_StubAgent, a)
    src = StubInputSource(["first message", "second message", None])
    await _input_pump(a, src, None, None)
    texts = [
        item.text for item in stub.runtime.inbox.items if isinstance(item, UserMessage)
    ]
    assert texts == ["first message", "second message"]


@pytest.mark.asyncio
async def test_input_pump_handles_dispatch_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    a = _agent()
    p = RecordingPrinter()

    async def boom(
        _a: object,
        _action: object,
        _p: object,
        *,
        queues: object | None = None,
    ) -> bool:
        del queues
        raise RuntimeError("pump crashed")

    monkeypatch.setattr(
        "sagent.repl.input_pane._dispatch",
        boom,
    )
    # Send one line then None; pump should log the error and shut down.
    src = StubInputSource(["hello", None])
    await _input_pump(a, src, None, p)
    assert any("pump crashed" in e for e in p.tool_errors)


@pytest.mark.asyncio
async def test_spawn_repl_pump_registers_under_repl_pump_key() -> None:
    a = _agent()
    stub = cast(_StubAgent, a)
    src = StubInputSource([None])
    task = spawn_repl_pump(a, src, printer=None)
    try:
        await task
    finally:
        if not task.done():
            _ = task.cancel()
    entry = stub.background_registry[REPL_PUMP_KEY]
    assert entry.tool_name == "repl-input"
    assert entry.queue_id == REPL_PUMP_KEY
    assert entry.hidden is True
    assert entry.kind == "tool"
    # Pump exited cleanly via shutdown.
    assert stub.shutdown_calls == [False]


@pytest.mark.asyncio
async def test_input_pump_cancellation_propagates() -> None:
    a = _agent()

    class _BlockingSource:
        async def next_line(self) -> str | None:
            # Block until cancelled.
            await asyncio.Event().wait()
            return None

    task = asyncio.create_task(_input_pump(a, _BlockingSource(), None, None))
    # Let the pump enter ``source.next_line()``.
    await asyncio.sleep(0)
    _ = task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_dispatch_halt_routes_to_registered_persistent_agent() -> None:
    a = _agent()
    other = _StubAgent(name="Other", is_subagent=True)
    with patch(
        "sagent.repl.input_pane.agent_registry",
        new={"Other": other},
    ):
        _ = await _dispatch(a, SlashHalt(target="Other"), None)
    assert other.halted == 1


@pytest.mark.asyncio
async def test_dispatch_halt_no_printer_swallows_unknown_agent() -> None:
    a = _agent()
    empty: dict[str, AgentLike] = {}
    with patch(
        "sagent.repl.input_pane.agent_registry",
        new=empty,
    ):
        exit_ = await _dispatch(a, SlashHalt(target="ghost"), None)
    assert exit_ is False


# --- input-pane rendering + PromptToolkitInputSource tests ------------
# (merged from former repl/prompt_test.py)


@dataclass(slots=True, kw_only=True)
class _FakeRuntime:
    cohort: set[str] = field(default_factory=set)
    _mid_stream_queue: list[UserMessage | AgentSendMessage] = field(
        default_factory=list
    )

    def pending_mid_stream(self) -> tuple[UserMessage | AgentSendMessage, ...]:
        return tuple(self._mid_stream_queue)


@dataclass(slots=True, kw_only=True)
class _FakeAgent:
    work: object = None
    runtime: _FakeRuntime = field(default_factory=_FakeRuntime)


def _as_real_agent(a: _FakeAgent) -> _RealAgent:
    return cast(_RealAgent, a)


def test_render_input_pane_empty_queue_renders_only_sigil() -> None:
    """Empty queues render only the ``> `` prompt token."""
    fp = render_input_pane(_as_real_agent(_FakeAgent()), InputQueues())
    assert isinstance(fp, FormattedText)
    assert list(fp) == [("class:input_pane", "> ")]


def test_render_input_pane_shows_mid_stream_queue() -> None:
    """Mid-stream ``UserMessage`` buffer must surface in ``queue_pane``.

    Two buffers exist for pending user content: REPL-local queues
    (urgent/deferred) and ``_mid_stream_queue`` (external mid-stream,
    runtime-internal). Both hold messages waiting for the model to be
    ready. Both should render in the same dim queue preview so users can
    tell whether their message is waiting for the model boundary.
    """
    fake = _FakeAgent()
    fake.runtime._mid_stream_queue = [
        UserMessage(text="tuttle"),
        UserMessage(text="lawnmower"),
    ]
    fp = render_input_pane(_as_real_agent(fake), InputQueues())
    # ``FormattedText`` entries are 2- or 3-tuples (the 3rd is an
    # optional mouse handler). Index-access to stay shape-agnostic.
    rendered = "".join(t[1] for t in fp)
    assert "tuttle" in rendered, (
        f"mid-stream UserMessage 'tuttle' must appear in queue pane; got {rendered!r}"
    )
    assert "lawnmower" in rendered, (
        f"mid-stream UserMessage 'lawnmower' must appear in queue pane; "
        f"got {rendered!r}"
    )
    # The queue pane styling must apply (dim preview, not the input
    # sigil row).
    queue_parts = [t[1] for t in fp if t[0] == "class:queued_input_pane"]
    assert queue_parts, (
        f"expected at least one ``queued_input_pane``-styled segment; got {list(fp)!r}"
    )


def test_render_input_pane_hides_agent_sourced_mid_stream() -> None:
    """``AgentSendMessage`` in the mid-stream buffer is NOT rendered.

    Queue/pending state is for the human's own input -- the user only
    needs to see what they typed and might still edit. Agent-to-agent
    traffic (``AgentSendMessage`` arriving mid-stream and getting
    buffered onto ``_mid_stream_queue``) must not surface as
    ``pending: ...`` in the input pane.
    """
    fake = _FakeAgent()
    fake.runtime._mid_stream_queue = [
        UserMessage(text="user-pending"),
        AgentSendMessage(source="reviewer", text="agent-payload"),
    ]
    fp = render_input_pane(_as_real_agent(fake), InputQueues())
    rendered = "".join(t[1] for t in fp)
    assert "pending: user-pending" in rendered
    assert "agent-payload" not in rendered, (
        f"AgentSendMessage must not appear in input pane; got {rendered!r}"
    )


def test_render_input_pane_single_block_renders_full_text() -> None:
    """A deferred message renders with the ``[deferred]`` prefix."""
    fp = render_input_pane(
        _as_real_agent(_FakeAgent()),
        InputQueues(deferred=QueuedInputBlock(text="hello world")),
    )
    parts = list(fp)
    assert parts[0] == ("class:queued_input_pane", "[deferred] hello world")
    assert parts[1] == ("", "\n")
    assert parts[2] == ("class:input_pane", "> ")


def test_render_input_pane_labels_retractable_and_pending_queues() -> None:
    fake = _FakeAgent()
    fake.runtime._mid_stream_queue = [UserMessage(text="already sent")]
    fp = render_input_pane(
        _as_real_agent(fake),
        InputQueues(deferred=QueuedInputBlock(text="tab staged")),
    )
    rendered = "".join(t[1] for t in fp)
    assert "[deferred] tab staged" in rendered
    assert "pending: already sent" in rendered


def test_render_input_pane_deferred_above_queue() -> None:
    """Spec: deferred pane renders above the queue pane; queue has no prefix."""
    fp = render_input_pane(
        _as_real_agent(_FakeAgent()),
        InputQueues(
            queue=QueuedInputBlock(text="q-msg"),
            deferred=QueuedInputBlock(text="d-msg"),
        ),
    )
    parts = list(fp)
    assert parts[0] == ("class:queued_input_pane", "[deferred] d-msg\n\nq-msg")
    assert parts[1] == ("", "\n")
    assert parts[2] == ("class:input_pane", "> ")


def test_render_input_pane_preserves_multi_line_block_content() -> None:
    """Internal newlines in a block are preserved verbatim (no collapse)."""
    fp = render_input_pane(
        _as_real_agent(_FakeAgent()),
        InputQueues(deferred=QueuedInputBlock(text="line1\nline2")),
    )
    parts = list(fp)
    assert parts[0] == ("class:queued_input_pane", "[deferred] line1\nline2")


def test_next_line_returns_typed_text() -> None:
    session = MagicMock()

    async def _prompt_async(**kwargs: object) -> str:
        del kwargs
        return "hello"

    session.prompt_async = _prompt_async
    src = PromptToolkitInputSource(session, queues=InputQueues())
    line = asyncio.run(src.next_line())
    assert line == "hello"


def test_next_line_disables_prompt_toolkit_exception_pause() -> None:
    session = MagicMock()
    calls: list[dict[str, object]] = []

    async def _prompt_async(**kwargs: object) -> str:
        calls.append(kwargs)
        return "hello"

    session.prompt_async = _prompt_async
    src = PromptToolkitInputSource(session, queues=InputQueues())
    line = asyncio.run(src.next_line())

    assert line == "hello"
    assert calls == [{"set_exception_handler": False}]


def test_next_line_quit_returns_none() -> None:
    session = MagicMock()

    async def _prompt_async(**kwargs: object) -> str:
        del kwargs
        return "/quit"

    session.prompt_async = _prompt_async
    src = PromptToolkitInputSource(session, queues=InputQueues())
    line = asyncio.run(src.next_line())
    assert line is None


def test_next_line_eof_returns_none() -> None:
    session = MagicMock()

    async def _prompt_async(**kwargs: object) -> str:
        del kwargs
        raise EOFError

    session.prompt_async = _prompt_async
    src = PromptToolkitInputSource(session, queues=InputQueues())
    line = asyncio.run(src.next_line())
    assert line is None


def test_next_line_keyboard_interrupt_reprompts_never_quits() -> None:
    """A stray Ctrl+C must re-arm the prompt, not terminate the loop.

    Ctrl+C is owned by ``_kb_ctrl_c`` (halt/clear-line); it never exits
    the REPL. When a ``KeyboardInterrupt`` escapes that binding,
    ``next_line`` must swallow it and re-prompt -- returning ``None`` here
    would make ``_input_pump`` call ``agent.shutdown`` and wedge the REPL.
    """
    session = MagicMock()
    calls: list[int] = []

    async def _prompt_async(**kwargs: object) -> str:
        del kwargs
        calls.append(1)
        if len(calls) == 1:
            raise KeyboardInterrupt
        return "after interrupt"

    session.prompt_async = _prompt_async
    src = PromptToolkitInputSource(session, queues=InputQueues())
    line = asyncio.run(src.next_line())
    assert line == "after interrupt"
    assert len(calls) == 2


def test_next_line_keyboard_interrupt_preserves_queue() -> None:
    """Stray Ctrl+C must not discard staged input (no quit-surface path)."""
    session = MagicMock()
    calls: list[int] = []

    async def _prompt_async(**kwargs: object) -> str:
        del kwargs
        calls.append(1)
        if len(calls) == 1:
            raise KeyboardInterrupt
        return "resumed"

    session.prompt_async = _prompt_async
    console = MagicMock()
    queues = InputQueues(deferred=QueuedInputBlock(text="staged line"))
    src = PromptToolkitInputSource(session, queues=queues, console=console)
    line = asyncio.run(src.next_line())
    assert line == "resumed"
    assert queues.has_any()
    console.print.assert_not_called()


def test_quit_surfaces_queued_input_preview() -> None:
    session = MagicMock()

    async def _prompt_async(**kwargs: object) -> str:
        del kwargs
        return "/quit"

    session.prompt_async = _prompt_async
    console = MagicMock()
    queues = InputQueues(deferred=QueuedInputBlock(text="queued line"))
    src = PromptToolkitInputSource(session, queues=queues, console=console)
    line = asyncio.run(src.next_line())
    assert line is None
    console.print.assert_called_once()
    assert not queues.has_any()


def test_quit_discard_preview_includes_count_when_both_panes() -> None:
    """The discard message surfaces the count of populated panes.

    Both panes populated -> count 2, plural noun. The preview shows the
    queue pane's text (it outranks deferred).
    """
    session = MagicMock()

    async def _prompt_async(**kwargs: object) -> str:
        del kwargs
        return "/quit"

    session.prompt_async = _prompt_async
    console = MagicMock()
    queues = InputQueues(
        queue=QueuedInputBlock(text="queued one"),
        deferred=QueuedInputBlock(text="deferred one"),
    )
    src = PromptToolkitInputSource(session, queues=queues, console=console)
    line = asyncio.run(src.next_line())
    assert line is None
    console.print.assert_called_once()
    rendered = str(console.print.call_args.args[0])
    assert "2" in rendered, f"discard preview must include count; got {rendered!r}"
    assert "messages" in rendered, (
        f"discard preview must use plural for >1 pane; got {rendered!r}"
    )


def test_quit_discard_preview_marks_truncated_body_with_ellipsis() -> None:
    """A long block preview must be marked as truncated, not silently cut.

    Otherwise an 80-char clip looks like the entire message; the
    operator can't tell content was elided.
    """
    session = MagicMock()

    async def _prompt_async(**kwargs: object) -> str:
        del kwargs
        return "/quit"

    session.prompt_async = _prompt_async
    console = MagicMock()
    long_text = "x" * 200
    queues = InputQueues(deferred=QueuedInputBlock(text=long_text))
    src = PromptToolkitInputSource(session, queues=queues, console=console)
    line = asyncio.run(src.next_line())
    assert line is None
    rendered = str(console.print.call_args.args[0])
    assert "…" in rendered, (
        f"truncated preview must end with ellipsis marker; got {rendered!r}"
    )


def test_quit_without_console_swallows_preview() -> None:
    session = MagicMock()

    async def _prompt_async(**kwargs: object) -> str:
        del kwargs
        return "/quit"

    session.prompt_async = _prompt_async
    queues = InputQueues(deferred=QueuedInputBlock(text="queued"))
    src = PromptToolkitInputSource(session, queues=queues, console=None)
    line = asyncio.run(src.next_line())
    assert line is None
    # buffer left alone when there's no console to surface to.
    assert queues.deferred is not None
    assert queues.deferred.text == "queued"


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
