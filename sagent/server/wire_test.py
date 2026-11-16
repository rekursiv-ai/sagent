"""Contract tests for lossless native daemon envelopes."""

from __future__ import annotations

from typing import cast

from sagent.model.cost import TokenCost, TokenCount
from sagent.model.types import Response
from sagent.runtime import types
from sagent.server.wire import runtime_event_message, session_record_message
from sagent.session.types import (
    AssistantMessage,
    ShellCommandResult,
    Thinking,
    ToolCall,
    UserMessage,
)


def test_native_records_keep_the_value_and_add_only_native_semantics() -> None:
    user = UserMessage(content="hello")
    result = ShellCommandResult(
        call_id="call-1",
        stdout="out\n",
        stderr="err\n",
        exit_code=7,
    )

    assert session_record_message(user) == {
        "recordType": "UserMessage",
        "category": "user-message",
        "value": user,
    }
    assert session_record_message(result) == {
        "recordType": "ShellCommandResult",
        "category": "tool-result",
        "value": result,
        "native": {"callId": "call-1", "text": "out\nerr\n\n[exit 7]"},
    }


def test_tool_call_summary_comes_from_the_native_tool_resolver() -> None:
    call = ToolCall(call_id="call-1", name="Read", arguments={"file_path": "a.py"})

    message = session_record_message(
        call,
        tool_summary=lambda record: f"native {record.name}",
    )

    assert message["category"] == "tool-call"
    assert message["native"] == {"callId": "call-1", "summary": "native Read"}
    assert message["value"] is call


def test_detached_arrival_identity_is_native_metadata() -> None:
    call = ToolCall(
        call_id="call-1:detached",
        name="BackgroundTask",
        arguments={},
    )
    result = ShellCommandResult(
        call_id="call-1:detached",
        stdout="done\n",
        stderr="",
        exit_code=0,
    )

    assert session_record_message(call)["native"] == {
        "callId": "call-1",
        "arrival": True,
    }
    assert session_record_message(result)["native"] == {
        "callId": "call-1",
        "arrival": True,
        "text": "done\n",
    }


def test_completed_response_uses_the_same_record_contract_as_replay() -> None:
    call = ToolCall(call_id="call-1", name="Read", arguments={})
    event = types.ModelResponseComplete(
        response=Response(
            records=(
                Thinking(content="considering"),
                AssistantMessage(content="checking"),
                call,
            ),
            tokens=TokenCount(),
            spend=TokenCost(),
            stop_reason="tool_use",
        ),
    )

    message = runtime_event_message(
        event,
        agent_label="Agent",
        session_id="session-1",
        sequence=4,
        tool_summary=lambda _: "Read a file",
    )

    assert message["category"] == "model-response"
    records = cast("list[dict[str, object]]", message["records"])
    assert [record["category"] for record in records] == [
        "thinking",
        "assistant-message",
        "tool-call",
    ]
    assert records[2]["native"] == {"callId": "call-1", "summary": "Read a file"}
    assert message["value"] is event


def test_runtime_categories_do_not_invent_a_frontend_turn_lifecycle() -> None:
    cancelled = runtime_event_message(
        types.ModelResponseCancelled(),
        agent_label="Agent",
    )
    idle = runtime_event_message(types.AgentIdle(), agent_label="Agent")
    halted = runtime_event_message(types.Halt(), agent_label="Agent")

    assert cancelled["category"] == "lifecycle"
    assert idle["category"] == "lifecycle"
    assert halted["category"] == "runtime"
