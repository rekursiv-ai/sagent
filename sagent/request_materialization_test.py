"""Tests for provider-request materialization."""

from __future__ import annotations

from collections.abc import Sequence

from sagent.agent.context import validate_context
from sagent.request_materialization import (
    ELIDED_TOOL_RESULT_TAG,
    materialize_messages,
    materialize_request,
)
from sagent.types.history import (
    AssistantMessage,
    ToolCall,
    ToolResult,
    UserMessage,
)
from sagent.types.model import ModelRequest


def test_materialize_messages_bounds_tool_results_and_preserves_pairs() -> None:
    first = ToolCall(id="call_1", name="Bash", args={})
    second = ToolCall(id="call_2", name="Bash", args={})
    messages = [
        UserMessage(text="start"),
        AssistantMessage(tool_calls=(first,)),
        ToolResult(call_id="call_1", content="x" * 1_000),
        UserMessage(text="continue"),
        AssistantMessage(tool_calls=(second,)),
        ToolResult(call_id="call_2", content="ok"),
    ]

    materialized = materialize_messages(messages, tool_result_budget_chars=20)

    first_result = materialized[2]
    second_result = materialized[5]
    assert isinstance(first_result, ToolResult)
    assert isinstance(second_result, ToolResult)
    assert first_result.content == ELIDED_TOOL_RESULT_TAG
    assert second_result.content == "ok"
    assert _visible_tool_result_chars(materialized) <= 20
    validate_context(materialized)


def test_materialize_messages_elides_error_results() -> None:
    call = ToolCall(id="call_1", name="Bash", args={})
    messages = [
        AssistantMessage(tool_calls=(call,)),
        ToolResult(call_id="call_1", content="traceback" * 1_000, is_error=True),
    ]

    materialized = materialize_messages(messages, tool_result_budget_chars=10)

    result = materialized[1]
    assert isinstance(result, ToolResult)
    assert result.is_error is True
    assert result.content == ELIDED_TOOL_RESULT_TAG[:10]


def test_materialize_messages_drops_excess_turns_before_emptying_results() -> None:
    calls = [ToolCall(id=f"call_{idx}", name="Bash", args={}) for idx in range(100)]
    messages = [
        entry
        for call in calls
        for entry in (
            AssistantMessage(tool_calls=(call,)),
            ToolResult(call_id=call.id, content="x" * 1_000),
        )
    ]

    materialized = materialize_messages(messages, tool_result_budget_chars=10)

    assert _visible_tool_result_chars(materialized) <= 10
    for entry in materialized:
        if isinstance(entry, ToolResult):
            assert entry.content
    validate_context(materialized)


def test_materialize_messages_preserves_assistant_text_when_result_drops() -> None:
    call = ToolCall(id="call_1", name="Bash", args={})
    messages = [
        UserMessage(text="start"),
        AssistantMessage(text="I checked the file", tool_calls=(call,)),
        ToolResult(call_id="call_1", content="x" * 1_000),
    ]

    materialized = materialize_messages(messages, tool_result_budget_chars=1)

    assert len(materialized) == 2
    assistant = materialized[1]
    assert isinstance(assistant, AssistantMessage)
    assert assistant.text == "I checked the file"
    assert assistant.tool_calls == ()
    assert _visible_tool_result_chars(materialized) == 0
    validate_context(materialized)


def test_materialize_messages_preserves_thinking_when_result_drops() -> None:
    call = ToolCall(id="call_1", name="Bash", args={})
    thinking = ({"type": "thinking", "thinking": "checked the file"},)
    messages = [
        UserMessage(text="start"),
        AssistantMessage(thinking_blocks=thinking, tool_calls=(call,)),
        ToolResult(call_id="call_1", content="x" * 1_000),
    ]

    materialized = materialize_messages(messages, tool_result_budget_chars=1)

    assert len(materialized) == 2
    assistant = materialized[1]
    assert isinstance(assistant, AssistantMessage)
    assert assistant.text == ""
    assert assistant.thinking_blocks == thinking
    assert assistant.tool_calls == ()
    assert _visible_tool_result_chars(materialized) == 0
    validate_context(materialized)


def test_materialize_messages_counts_only_kept_results_against_budget() -> None:
    first = ToolCall(id="call_1", name="Bash", args={})
    second = ToolCall(id="call_2", name="Bash", args={})
    messages = [
        AssistantMessage(tool_calls=(first,)),
        ToolResult(call_id="call_1", content="x" * 1_000),
        AssistantMessage(tool_calls=(second,)),
        ToolResult(call_id="call_2", content="y" * 1_000),
    ]

    materialized = materialize_messages(messages, tool_result_budget_chars=10)

    assert len(materialized) == 2
    result = materialized[1]
    assert isinstance(result, ToolResult)
    assert result.call_id == "call_2"
    assert result.content == ELIDED_TOOL_RESULT_TAG
    assert _visible_tool_result_chars(materialized) <= 10
    validate_context(materialized)


def test_materialize_request_reuses_message_materialization() -> None:
    call = ToolCall(id="call_1", name="Bash", args={})
    request = ModelRequest(
        messages=[
            UserMessage(text="start"),
            AssistantMessage(tool_calls=(call,)),
            ToolResult(call_id="call_1", content="x" * 1_000),
        ],
        system="sys",
    )

    materialized = materialize_request(request, tool_result_budget_chars=10)

    result = materialized.messages[2]
    assert isinstance(result, ToolResult)
    assert ELIDED_TOOL_RESULT_TAG in result.content
    assert materialized.system == "sys"


def _visible_tool_result_chars(messages: Sequence[object]) -> int:
    return sum(
        len(entry.content) for entry in messages if isinstance(entry, ToolResult)
    )


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
