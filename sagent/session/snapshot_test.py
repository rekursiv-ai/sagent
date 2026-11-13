"""Tests for the shared native session snapshot projector."""

from __future__ import annotations

from unittest.mock import MagicMock

from sagent.session.snapshot import SessionSnapshotProjector
from sagent.session.types import (
    AgentToAgentMessage,
    AssistantMessage,
    ShellCommandResult,
    Thinking,
    ToolCall,
    TurnContext,
    UserMessage,
)


def _agent(*records: object) -> MagicMock:
    agent = MagicMock()
    agent.session = records
    agent.tools = {}
    return agent


def test_snapshot_preserves_native_conversation_records() -> None:
    snapshot = SessionSnapshotProjector().project(
        _agent(
            TurnContext(model="test-model"),
            UserMessage(content="hello"),
            Thinking(content="thinking"),
            AssistantMessage(content="answer"),
            AgentToAgentMessage(content="delegate", sender="Agent", recipient="reviewer"),
        ),
    )

    assert snapshot == {
        "version": 1,
        "items": [
            {
                "id": "record-1",
                "type": "user",
                "turnId": "turn-1",
                "text": "hello",
                "model": "test-model",
            },
            {
                "id": "record-2",
                "type": "thinking",
                "turnId": "turn-1",
                "text": "thinking",
            },
            {
                "id": "record-3",
                "type": "assistant",
                "turnId": "turn-1",
                "text": "answer",
            },
            {
                "id": "record-4",
                "type": "agent",
                "turnId": "turn-2",
                "text": "delegate",
                "model": "test-model",
                "sender": "Agent",
                "recipient": "reviewer",
            },
        ],
    }


def test_snapshot_reconciles_tool_results_with_native_calls() -> None:
    snapshot = SessionSnapshotProjector().project(
        _agent(
            UserMessage(content="read the file"),
            ToolCall(call_id="call-1", name="Shell", arguments={"command": "pwd"}),
            ShellCommandResult(call_id="call-1", stdout="/work", exit_code=0),
        ),
    )

    assert snapshot["items"] == [
        {
            "id": "record-0",
            "type": "user",
            "turnId": "turn-1",
            "text": "read the file",
        },
        {
            "id": "tool-call-1",
            "type": "tool",
            "turnId": "turn-1",
            "callId": "call-1",
            "name": "Shell",
            "summary": "Shell",
                "arguments": {"command": "pwd"},
                "status": "completed",
                "resultCallId": "call-1",
                "text": "/work",
                "isError": None,
        },
    ]
