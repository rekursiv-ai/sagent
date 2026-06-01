"""Tests for provider-request materialization."""

from __future__ import annotations

from collections.abc import Sequence

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


def test_materialize_messages_keeps_head_assistant_text_when_result_drops() -> None:
    """A head ``AssistantMessage`` whose tool_calls are all elided keeps
    its text content; only the ``tool_calls`` are stripped. Dropping the
    text would silently elide the assistant's user-visible payload.
    """
    call = ToolCall(id="call_1", name="Bash", args={})
    messages = [
        AssistantMessage(text="I checked", tool_calls=(call,)),
        ToolResult(call_id="call_1", content="x" * 1_000),
    ]

    materialized = materialize_messages(messages, tool_result_budget_chars=1)

    assert len(materialized) == 1
    only = materialized[0]
    assert isinstance(only, AssistantMessage)
    assert only.text == "I checked"
    assert only.tool_calls == ()
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


def test_materialize_messages_keeps_text_assistant_between_users_when_result_drops() -> (
    None
):
    """An ``AssistantMessage`` with text content between two users keeps
    its text (tool_calls stripped) when its result is elided. The users
    no longer coalesce because the surviving assistant turn separates
    them.
    """
    call = ToolCall(id="call_1", name="Bash", args={})
    messages = [
        UserMessage(text="start"),
        AssistantMessage(text="I checked the file", tool_calls=(call,)),
        ToolResult(call_id="call_1", content="x" * 1_000),
        UserMessage(text="continue"),
    ]

    materialized = materialize_messages(messages, tool_result_budget_chars=1)

    assert len(materialized) == 3
    first, middle, last = materialized
    assert isinstance(first, UserMessage)
    assert first.text == "start"
    assert isinstance(middle, AssistantMessage)
    assert middle.text == "I checked the file"
    assert middle.tool_calls == ()
    assert isinstance(last, UserMessage)
    assert last.text == "continue"
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
    """An ``AssistantMessage`` with text whose tool_calls are elided is
    dropped when the next chronological entry is also an assistant turn:
    keeping the text would create two adjacent assistant messages,
    which violates provider role alternation.
    """
    call = ToolCall(id="call_1", name="Bash", args={})
    messages = [
        UserMessage(text="start"),
        AssistantMessage(text="I checked", tool_calls=(call,)),
        ToolResult(call_id=call.id, content="x" * 1_000),
        AssistantMessage(text="next answer"),
    ]

    materialized = materialize_messages(messages, tool_result_budget_chars=1)

    assert materialized == [messages[0], messages[3]]
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
    """Adjacent AgentSends from DIFFERENT sources must not be re-attributed.

    The fix: a cross-source merge demotes to ``UserMessage``. The
    structured ``source`` field cannot honestly represent two senders,
    so it is dropped; the in-text ``[from X]: `` labels carry the
    attribution. Provider serializers that key on ``source`` see no
    bare-string lie.
    """
    messages = [
        AgentSendMessage(source="alice", text="from alice"),
        AgentSendMessage(source="bob", text="from bob"),
    ]
    materialized = materialize_messages(messages)
    if len(materialized) == 1:
        merged = materialized[0]
        assert isinstance(merged, (UserMessage, AgentSendMessage))
        # Either it's a UserMessage (no source field), or any AgentSend
        # must not silently claim one sender owns the other's content.
        if isinstance(merged, AgentSendMessage):
            assert merged.source != "alice" or "bob" not in merged.text, (
                f"merged AgentSend's source={merged.source!r} attributes"
                f" mixed content to alice alone; text={merged.text!r}"
            )
        assert "[from alice]: from alice" in merged.text
        assert "[from bob]: from bob" in merged.text
    else:
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


def test_materialize_messages_preserves_tool_pair_around_interleaved_agent_send() -> (
    None
):
    """AgentSend interleaved between AM(tool_calls) and TR is deferred past the pair.

    Provider APIs reject any user-role turn between an
    ``AssistantMessage`` with ``tool_calls`` and the matching
    ``ToolResult``. Materialization must keep the tool-pair contiguous
    while preserving the user content.
    """
    call = ToolCall(id="c1", name="Bash", args={})
    messages = [
        AssistantMessage(tool_calls=(call,)),
        AgentSendMessage(source="peer", text="ping"),
        ToolResult(call_id="c1", content="ok"),
    ]
    materialized = materialize_messages(messages, tool_result_budget_chars=1000)
    validate_context(materialized)
    # The pair stays contiguous, with the AgentSend pushed past the TR.
    assert isinstance(materialized[0], AssistantMessage)
    assert isinstance(materialized[1], ToolResult)
    assert isinstance(materialized[2], AgentSendMessage)


def test_materialize_messages_coalesces_cross_type_user_role() -> None:
    """UserMessage + AgentSendMessage adjacent merge into a single user-role turn.

    The merged entry demotes to ``UserMessage``: the structured
    ``source`` field would otherwise silently re-attribute the human's
    content to the agent. Attribution is preserved by the in-text
    ``[from <source>]: `` label that :func:`_label_agent_sends`
    prepends upstream.
    """
    messages = [
        UserMessage(text="u1"),
        AgentSendMessage(source="A", text="a"),
    ]
    materialized = materialize_messages(messages)
    validate_context(materialized)
    assert len(materialized) == 1
    merged = materialized[0]
    assert isinstance(merged, UserMessage)
    assert "u1" in merged.text
    assert "[from A]: a" in merged.text


# --- A5: AM text must survive when budget elides all of its TRs ----------


def test_materialize_messages_keeps_assistant_text_when_all_tool_results_elided() -> (
    None
):
    """When budget admits zero ``ToolResult`` for an ``AssistantMessage``
    that has non-empty text, the text-only variant must survive even
    if a newer entry follows. Dropping it silently elides the assistant
    turn's user-visible payload.

    Budget = 1 char admits neither real content nor the ``<elided>``
    sentinel; the AM has no kept call ids and a newer ``UserMessage``,
    so the buggy branch (``next_newer is None``) drops it.
    """
    call = ToolCall(id="t1", name="x", args={})
    messages = [
        AssistantMessage(text="hello", tool_calls=(call,)),
        ToolResult(call_id="t1", content="x" * 100_000),
        UserMessage(text="ok"),
    ]
    out = materialize_messages(messages, tool_result_budget_chars=1)
    assistants = [e for e in out if isinstance(e, AssistantMessage)]
    assert any("hello" in a.text for a in assistants)


def _visible_tool_result_chars(messages: Sequence[object]) -> int:
    return sum(
        len(entry.content) for entry in messages if isinstance(entry, ToolResult)
    )


def test_agent_send_label_not_suppressed_by_body_mention() -> None:
    """SAGENT-REV-002: an unprefixed AgentSend whose body merely *mentions*
    its own ``[from <source>]: `` marker must still be labelled.

    Pre-fix, ``_label_agent_sends`` used ``prefix in entry.text`` (any
    substring match) to mean "already labelled". A legitimate body
    that happened to quote the marker (e.g. "please write [from bob]:
    literally") was passed through unlabelled, losing attribution.
    """
    out = materialize_messages(
        [AgentSendMessage(source="bob", text="please write [from bob]: literally")]
    )

    first = out[0]
    assert isinstance(first, AgentSendMessage)
    assert first.text.startswith("[from bob]: ")
    assert first.text == "[from bob]: please write [from bob]: literally"


def test_cross_source_agent_send_merge_keeps_attribution_in_text() -> None:
    """Adjacent different-source AgentSends merge without re-attributing one to the other.

    Pre-fix, two AgentSends ``alice``+``bob`` either stayed separate
    (wire-invalid consecutive user-role) or merged adopting one
    sender's structured ``source``. The fix: merge into a single
    user-role turn but preserve in-text ``[from X]: `` labels, since
    those convey the attribution unambiguously.
    """
    out = materialize_messages(
        [
            AgentSendMessage(source="alice", text="from alice"),
            AgentSendMessage(source="bob", text="from bob"),
        ]
    )
    validate_context(out)
    assert len(out) == 1
    merged = out[0]
    assert isinstance(merged, (UserMessage, AgentSendMessage))
    assert "[from alice]: from alice" in merged.text
    assert "[from bob]: from bob" in merged.text


def test_user_with_agent_send_merge_does_not_re_attribute_user_text() -> None:
    """UserMessage + AgentSendMessage merge must not silently attribute user text.

    Pre-fix, the merge adopted the AgentSendMessage type so the
    structured ``source`` field claimed both texts. The fix: cross-type
    merges demote to UserMessage with labels preserved in text.
    """
    out = materialize_messages(
        [
            UserMessage(text="user said"),
            AgentSendMessage(source="alice", text="alice said"),
        ]
    )
    validate_context(out)
    assert len(out) == 1
    merged = out[0]
    assert isinstance(merged, (UserMessage, AgentSendMessage))
    # No structured re-attribution: the merged entry must not claim
    # Alice authored the user text. Either it's a UserMessage (no
    # source), or its source is something other than a bare "alice".
    if isinstance(merged, AgentSendMessage):
        # If the implementation chooses to keep AgentSendMessage form,
        # the user content must not be attributed to alice alone.
        assert merged.source != "alice" or "user said" not in merged.text
    assert "user said" in merged.text
    assert "alice said" in merged.text


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
