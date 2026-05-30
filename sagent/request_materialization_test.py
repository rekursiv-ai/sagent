"""Tests for provider-request materialization."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise

from sagent.agent.context import validate_context
from sagent.request_materialization import (
    ELIDED_TOOL_RESULT_TAG,
    materialize_messages,
    materialize_request,
)
from sagent.types.model import ModelRequest
from sagent.types.runtime import (
    AgentSendMessage,
    AssistantMessage,
    ToolCall,
    ToolResult,
    UserMessage,
)


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


def test_materialize_messages_drops_head_assistant_when_result_drops() -> None:
    call = ToolCall(id="call_1", name="Bash", args={})
    messages = [
        AssistantMessage(text="I checked", tool_calls=(call,)),
        ToolResult(call_id="call_1", content="x" * 1_000),
    ]

    materialized = materialize_messages(messages, tool_result_budget_chars=1)

    assert materialized == []
    validate_context(materialized)


def test_materialize_messages_drops_empty_assistant_when_result_drops() -> None:
    call = ToolCall(id="call_1", name="Bash", args={})
    messages = [
        AssistantMessage(tool_calls=(call,)),
        ToolResult(call_id="call_1", content="x" * 1_000),
    ]

    materialized = materialize_messages(messages, tool_result_budget_chars=1)

    assert materialized == []
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


def test_materialize_messages_drops_text_assistant_between_users_when_result_drops() -> (
    None
):
    call = ToolCall(id="call_1", name="Bash", args={})
    messages = [
        UserMessage(text="start"),
        AssistantMessage(text="I checked the file", tool_calls=(call,)),
        ToolResult(call_id="call_1", content="x" * 1_000),
        UserMessage(text="continue"),
    ]

    materialized = materialize_messages(messages, tool_result_budget_chars=1)

    assert len(materialized) == 1
    user = materialized[0]
    assert isinstance(user, UserMessage)
    assert user.text == "start\n\ncontinue"
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


def test_materialize_messages_drops_orphaned_tool_use_text_before_assistant() -> None:
    call = ToolCall(id="call_1", name="Bash", args={})
    messages = [
        UserMessage(text="start"),
        AssistantMessage(text="I checked", tool_calls=(call,)),
        ToolResult(call_id=call.id, content="x" * 1_000),
        AssistantMessage(text="next answer"),
    ]

    materialized = materialize_messages(messages, tool_result_budget_chars=1)

    assert materialized == [messages[0], messages[3]]
    assert all(
        not isinstance(prev, AssistantMessage) or not isinstance(curr, AssistantMessage)
        for prev, curr in pairwise(materialized)
    )
    validate_context(materialized)


def test_materialize_messages_drops_multi_tool_turn_instead_of_shrinking() -> None:
    first = ToolCall(id="call_1", name="Bash", args={})
    second = ToolCall(id="call_2", name="Bash", args={})
    messages = [
        AssistantMessage(tool_calls=(first, second)),
        ToolResult(call_id="call_1", content="x" * 1_000),
        ToolResult(call_id="call_2", content="ok"),
    ]

    materialized = materialize_messages(messages, tool_result_budget_chars=5)

    assert materialized == []
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


def test_agent_send_message_receives_from_prefix() -> None:
    """AgentSendMessage.text is prefixed with ``[from <source>]: `` in provider view."""
    messages = [AgentSendMessage(source="Alice", text="hello")]
    materialized = materialize_messages(messages)
    result = materialized[0]
    assert isinstance(result, AgentSendMessage)
    assert result.text == "[from Alice]: hello"


def test_agent_send_different_sources_must_not_coalesce_under_one_source() -> None:
    """Adjacent AgentSends from DIFFERENT sources must keep their identity.

    Current behavior: ``_coalesce_adjacent_users`` merges adjacent
    AgentSendMessages by type, taking the FIRST source. Bob's message
    becomes attributed to Alice in the structured ``source`` field.
    The textual prefix saves the model-visible attribution, but any
    downstream consumer that reads ``message.source`` (e.g. provider
    serializers that emit OpenAI's ``name`` field) attributes both
    sends to Alice. Either don't coalesce across sources, or surface
    a multi-source marker.
    """
    messages = [
        AgentSendMessage(source="alice", text="from alice"),
        AgentSendMessage(source="bob", text="from bob"),
    ]
    materialized = materialize_messages(messages)
    # Either keep them separate, or invent a multi-source marker --
    # whatever we do, the structured field must not silently claim
    # one sender owns the other's content.
    if len(materialized) == 1:
        merged = materialized[0]
        assert isinstance(merged, AgentSendMessage)
        # Bob is in the text, so the source field cannot be a bare
        # "alice" without misleading downstream consumers.
        assert merged.source != "alice" or "bob" not in merged.text, (
            f"merged AgentSend's source={merged.source!r} attributes"
            f" mixed content to alice alone; text={merged.text!r}"
        )
    else:
        # Acceptable alternative: each sender retains its own message.
        sources = [m.source for m in materialized if isinstance(m, AgentSendMessage)]
        assert sources == ["alice", "bob"]


def test_agent_send_adjacent_coalesce_preserves_per_sender_prefix() -> None:
    """Adjacent AgentSendMessages from the same source coalesce with prefixes intact."""
    messages = [
        AgentSendMessage(source="Alice", text="first"),
        AgentSendMessage(source="Alice", text="second"),
    ]
    materialized = materialize_messages(messages)
    assert len(materialized) == 1
    result = materialized[0]
    assert isinstance(result, AgentSendMessage)
    assert result.text == "[from Alice]: first\n\n[from Alice]: second"


def test_agent_send_prefix_applied_under_budget_path() -> None:
    """``[from <source>]: `` prefix is applied when tool-result budget is active."""
    call = ToolCall(id="c1", name="Bash", args={})
    messages = [
        AgentSendMessage(source="Bob", text="go"),
        AssistantMessage(tool_calls=(call,)),
        ToolResult(call_id="c1", content="ok"),
    ]
    materialized = materialize_messages(messages, tool_result_budget_chars=10)
    first = materialized[0]
    assert isinstance(first, AgentSendMessage)
    assert first.text == "[from Bob]: go"


def _visible_tool_result_chars(messages: Sequence[object]) -> int:
    return sum(
        len(entry.content) for entry in messages if isinstance(entry, ToolResult)
    )


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
