"""Checkpoint transitions, independent events, durable handoff and local inspection."""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING

import json

from rich.cells import cell_len

import pytest

from sagent.agent.agent import Agent
from sagent.agent.session_io import SessionMeta, load_session
from sagent.agent.state import current_agent_var
from sagent.repl.checkpoints import (
    checkpoint_entries,
    format_pending,
    render_pending_pane,
)
from sagent.repl.input_pane import _dispatch
from sagent.repl.render import RecordingPrinter, make_render_observer
from sagent.repl.replay import replay_messages
from sagent.repl.slash import Pending, parse_slash
from sagent.testing import MockModelCaps
from sagent.tools.agent_self import AgentSelf
from sagent.types.runtime import (
    AgentIdle,
    AgentSendMessage,
    AssistantMessage,
    Checkpoint,
    CheckpointChanged,
    ChildEvent,
    ModelResponseComplete,
    ModelResponsePartial,
    SaveSession,
    ToolResult,
)


if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from sagent.types.model import ModelRequest, ModelResponse
    from sagent.types.runtime import RuntimeEvent


class _Model(MockModelCaps):
    """No provider requests are permitted in checkpoint tests."""

    async def buffer(self, request: ModelRequest) -> ModelResponse:
        raise AssertionError(request)

    async def stream(
        self,
        request: ModelRequest,
        publish: Callable[[RuntimeEvent], None] | None = None,
    ) -> ModelResponse:
        raise AssertionError((request, publish))


def _agent(path: Path | None = None) -> Agent:
    return Agent(model=_Model(), tools=[AgentSelf()], name="lead", session_dir=path)


def _checkpoint(
    *,
    pending: str = "Official evaluator result",
    user_action: str = "",
) -> Checkpoint:
    return Checkpoint(
        progress="Two inputs verified",
        pending=pending,
        user_action=user_action,
    )


@pytest.mark.asyncio
async def test_202_unchanged_reports_keep_one_checkpoint_and_all_tool_receipts(
    tmp_path: Path,
) -> None:
    agent = _agent(tmp_path)
    printer = RecordingPrinter()
    agent.runtime.observers.append(make_render_observer(printer))
    token = current_agent_var.set(agent)
    try:
        results = [
            await AgentSelf().run({"checkpoint": asdict(_checkpoint())})
            for _ in range(202)
        ]
    finally:
        current_agent_var.reset(token)
    assert len(results) == 202
    assert all(not result.is_error for result in results)
    assert "updated" in results[0].content
    assert all("unchanged" in result.content for result in results[1:])
    assert len(printer.slash_blocks) == 1
    assert "Pending: Official evaluator result" in render_pending_pane(agent, width=80)
    records = [
        json.loads(line)
        for line in (tmp_path / "session.jsonl").read_text().splitlines()
    ]
    assert sum(record.get("type") == "checkpoint_changed" for record in records) == 1


def test_changed_pending_and_user_action_emit_complete_checkpoints() -> None:
    agent = _agent()
    printer = RecordingPrinter()
    agent.runtime.observers.append(make_render_observer(printer))
    agent.checkpoint = _checkpoint()
    agent.checkpoint = _checkpoint(
        pending="Approval",
        user_action="Approve the next input",
    )
    assert len(printer.slash_blocks) == 2
    assert "Progress: Two inputs verified" in printer.slash_blocks[-1]
    assert "User action: Approve the next input" in printer.slash_blocks[-1]
    agent.runtime.publish(AgentIdle())
    assert "Pending: Approval" in render_pending_pane(agent, width=80)
    assert agent.checkpoint == _checkpoint(
        pending="Approval",
        user_action="Approve the next input",
    )


def test_equal_text_in_independent_messages_and_senders_is_preserved() -> None:
    agent = _agent()
    printer = RecordingPrinter()
    render = make_render_observer(printer)
    agent.runtime.observers.append(render)
    agent.checkpoint = _checkpoint()
    text = "Awaiting official results."
    for _ in range(2):
        render(ModelResponsePartial(text=text))
        render(ModelResponseComplete(message=AssistantMessage(text=text)))
    for sender in ("worker-a", "worker-b", "worker-a"):
        render(AgentSendMessage(source=sender, text=text))
    render(ToolResult(call_id="error", content="Evaluator failed", is_error=True))
    assert printer.markdowns == [text, text]
    assert printer.agent_bars == [
        ("worker-a", text),
        ("worker-b", text),
        ("worker-a", text),
    ]
    assert printer.tool_errors == ["Evaluator failed"]


def test_child_reports_use_event_provenance_and_survive_deregistration(
    tmp_path: Path,
) -> None:
    agent = _agent(tmp_path)
    printer = RecordingPrinter()
    agent.runtime.observers.append(make_render_observer(printer))
    change = CheckpointChanged(checkpoint=_checkpoint(), observed_at=100.0)
    for sender in ("worker-a", "worker-b"):
        agent.runtime.publish(ChildEvent(label=sender, inner=change))
    agent.runtime.publish(
        ChildEvent(label="parent", inner=ChildEvent(label="worker-a", inner=change)),
    )
    assert [entry.label for entry in checkpoint_entries(agent)] == [
        "lead/worker-a",
        "lead/worker-b",
        "lead/parent/worker-a",
    ]
    assert [label for label, _ in printer.child_blocks] == [
        "worker-a",
        "worker-b",
        "parent/worker-a",
    ]
    assert len(printer.child_blocks) == 3
    loaded = load_session(tmp_path)
    assert loaded is not None
    resumed = _agent(tmp_path)
    resumed.resume(*loaded)
    assert len(checkpoint_entries(resumed)) == 3
    assert all(entry.restored for entry in checkpoint_entries(resumed))
    assert format_pending(resumed).count("not been rechecked") == 3
    replay = RecordingPrinter()
    replay_messages(resumed, replay)
    assert "worker-b" in replay.slash_blocks[0]


def test_restore_clear_and_reconfirm_have_durable_state(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    agent.checkpoint = _checkpoint()
    agent.runtime.publish(SaveSession())
    loaded = load_session(tmp_path)
    assert loaded is not None
    resumed = _agent(tmp_path)
    resumed.resume(*loaded)
    assert resumed.checkpoint_restored
    assert "not rechecked" in render_pending_pane(resumed, width=80)
    assert "not rechecked" in render_pending_pane(resumed, width=40)
    assert "not been rechecked" in format_pending(resumed)
    resumed.checkpoint = _checkpoint()
    assert not resumed.checkpoint_restored
    resumed.checkpoint = None
    loaded_after = load_session(tmp_path)
    assert loaded_after is not None
    assert loaded_after[0].checkpoint is None
    assert (
        len(
            [
                event
                for event in loaded_after[0].runtime_events
                if isinstance(event, CheckpointChanged)
            ],
        )
        == 3
    )
    assert render_pending_pane(resumed, width=80) == ""


def test_child_clear_removes_only_the_corresponding_report() -> None:
    agent = _agent()
    for source in ("a", "b"):
        agent.runtime.publish(
            ChildEvent(
                label=source,
                inner=CheckpointChanged(checkpoint=_checkpoint(), observed_at=1.0),
            ),
        )
    agent.runtime.publish(
        ChildEvent(
            label="a",
            inner=CheckpointChanged(checkpoint=None, observed_at=2.0),
        ),
    )
    assert [entry.label for entry in checkpoint_entries(agent)] == ["lead/b"]


@pytest.mark.parametrize("width", [0, 1, 10, 40, 80, 120])
def test_toolbar_limits_physical_rows_and_terminal_cell_width(width: int) -> None:
    agent = _agent()
    agent.checkpoint = Checkpoint(
        progress="Verified 界🙂 " * 60,
        pending="Long\nresult\x1b[31m" * 80,
        user_action="Review this result",
    )
    rendered = render_pending_pane(agent, width=width)
    assert len(rendered.split("\n")) == 4
    assert all(cell_len(line) <= width for line in rendered.split("\n"))
    assert "\x1b" not in rendered
    assert "Long\nresult" in format_pending(agent)


def test_newer_wait_cannot_hide_an_older_user_action() -> None:
    agent = _agent()
    agent.checkpoint = _checkpoint(user_action="Review the input")
    agent.runtime.publish(
        ChildEvent(
            label="worker",
            inner=CheckpointChanged(
                checkpoint=_checkpoint(),
                observed_at=agent.checkpoint_updated_at + 1,
            ),
        ),
    )
    assert "User action: Review the input" in render_pending_pane(agent, width=80)
    assert len(checkpoint_entries(agent)) == 2


@pytest.mark.asyncio
async def test_pending_is_local_and_does_not_enqueue_model_work() -> None:
    agent = _agent()
    agent.checkpoint = _checkpoint()
    printer = RecordingPrinter()
    action = parse_slash("/pending")
    assert isinstance(action, Pending)
    assert not await _dispatch(agent, action, printer)
    assert agent.runtime.inbox.empty()
    assert "Progress: Two inputs verified" in printer.slash_blocks[0]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid",
    [
        {},
        {"progress": "", "pending": "", "user_action": ""},
        {"progress": 42, "pending": "", "user_action": ""},
        {"progress": "ok", "pending": "", "user_action": "", "extra": True},
        "waiting",
    ],
)
async def test_invalid_checkpoint_rejects_entire_patch(invalid: object) -> None:
    agent = _agent()
    token = current_agent_var.set(agent)
    try:
        result = await AgentSelf().run(
            {"status": "Must not change", "checkpoint": invalid},
        )
    finally:
        current_agent_var.reset(token)
    assert result.is_error
    assert agent.status == ""
    assert agent.checkpoint is None


def test_old_and_malformed_metadata_do_not_create_pending_work() -> None:
    assert SessionMeta.deserialize({"status": "old"}).checkpoint is None
    assert (
        SessionMeta.deserialize(
            {"checkpoint": {"progress": "missing fields"}},
        ).checkpoint
        is None
    )
