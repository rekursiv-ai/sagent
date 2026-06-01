"""Property-based tests for pure data-flow functions in the runtime layer.

Hypothesis emits random message sequences and asserts structural
properties that should hold across every input shape. Targets pure
functions (no async, no state): ``_label_agent_sends``,
``_coalesce_adjacent_users``, ``materialize_messages``, and
``_last_assistant_result``. Async runtime state-machine fuzzing lives
elsewhere; the runtime's async coupling is a poor fit for hypothesis.

Each test below would have caught at least one bug we've fixed this
session:

* Bug H (cross-source coalesce) -- ``coalesce_preserves_each_source``.
* Bug F (two AgentSends in one turn) -- ``last_assistant_result_picks_most_recent_send``.
* ``_label_agent_sends`` not idempotent under double materialization
  (audit-flagged hardening) -- ``materialize_is_idempotent``.
"""

from __future__ import annotations

from typing import cast

from hypothesis import given, settings
from hypothesis.strategies import (
    DrawFn,
    booleans,
    composite,
    integers,
    lists,
    sampled_from,
    text,
)

from sagent.request_materialization import (
    _coalesce_adjacent_users,
    _label_agent_sends,
    materialize_messages,
)
from sagent.tools.agent_spawn import _last_assistant_result
from sagent.types.runtime import (
    AgentSendMessage,
    AssistantMessage,
    ModelContextEvent,
    ToolCall,
    ToolResult,
    UserMessage,
)


_TEXT = text(min_size=0, max_size=20)
_NONEMPTY_TEXT = text(min_size=1, max_size=20).filter(lambda s: bool(s.strip()))
_SOURCE = sampled_from(["alice", "bob", "carol", "dave"])


@composite
def _user_message(draw: DrawFn) -> UserMessage:
    return UserMessage(text=draw(_TEXT))


@composite
def _agent_send_message(draw: DrawFn) -> AgentSendMessage:
    return AgentSendMessage(source=draw(_SOURCE), text=draw(_TEXT))


@composite
def _assistant_message(draw: DrawFn) -> AssistantMessage:
    has_tools = draw(booleans())
    n_tools = draw(integers(min_value=0, max_value=3)) if has_tools else 0
    tool_calls = tuple(
        ToolCall(
            id=f"tc-{i}-{draw(integers(0, 9999))}",
            name="Bash",
            args={},
        )
        for i in range(n_tools)
    )
    return AssistantMessage(text=draw(_TEXT), tool_calls=tool_calls)


@composite
def _agent_send_with_tool_call(draw: DrawFn) -> AssistantMessage:
    """An assistant turn whose tool_calls includes one or more AgentSends."""
    n_sends = draw(integers(min_value=1, max_value=3))
    sends = tuple(
        ToolCall(
            id=f"send-{i}-{draw(integers(0, 9999))}",
            name="AgentSend",
            args={"to": draw(_SOURCE), "content": draw(_NONEMPTY_TEXT)},
        )
        for i in range(n_sends)
    )
    return AssistantMessage(text=draw(_TEXT), tool_calls=sends)


@composite
def _message_history(draw: DrawFn) -> list[ModelContextEvent]:
    """Random history mixing user/agent/assistant; tool_results paired by call_id."""
    n = draw(integers(min_value=0, max_value=12))
    history: list[ModelContextEvent] = []
    pending: list[str] = []  # tool_call ids awaiting results
    for _ in range(n):
        choice = draw(sampled_from(["user", "agent", "assistant", "tool_result"]))
        if choice == "user":
            history.append(draw(_user_message()))
        elif choice == "agent":
            history.append(draw(_agent_send_message()))
        elif choice == "assistant":
            am = draw(_assistant_message())
            history.append(am)
            pending.extend(tc.id for tc in am.tool_calls)
        elif choice == "tool_result" and pending:
            cid = pending.pop(0)
            history.append(ToolResult(call_id=cid, content=draw(_TEXT)))
    # Pair any remaining tool_calls so output is wire-format valid.
    history.extend(ToolResult(call_id=cid, content="[orphan]") for cid in pending)
    return history


# -- _label_agent_sends ------------------------------------------------------


@settings(max_examples=100, deadline=None)
@given(_message_history())
def test_label_agent_sends_only_touches_agent_send_messages(
    history: list[ModelContextEvent],
) -> None:
    """``_label_agent_sends`` must not mutate any non-AgentSend entry."""
    labelled = list(_label_agent_sends(history))
    assert len(labelled) == len(history)
    for original, output in zip(history, labelled, strict=True):
        if not isinstance(original, AgentSendMessage):
            assert output is original, (
                f"non-AgentSend entry mutated: {type(original).__name__}"
            )


@settings(max_examples=100, deadline=None)
@given(_message_history())
def test_label_agent_sends_prefixes_with_source(
    history: list[ModelContextEvent],
) -> None:
    """Every output AgentSend starts with its source's ``[from <source>]:`` tag."""
    labelled = list(_label_agent_sends(history))
    for entry in labelled:
        if isinstance(entry, AgentSendMessage):
            assert entry.text.startswith(f"[from {entry.source}]: "), (
                f"AgentSend text lacks source prefix: {entry.text!r}"
            )


# -- _coalesce_adjacent_users -----------------------------------------------


@settings(max_examples=200, deadline=None)
@given(_message_history())
def test_coalesce_preserves_each_source(history: list[ModelContextEvent]) -> None:
    """Every source that appears in the input also appears in the output.

    Catches bug H: pre-fix, adjacent AgentSends from different sources
    merged under the first's source, silently dropping the second.

    Post-fix the structured ``source`` field can be lost when a cross-
    source merge demotes to ``UserMessage`` (the only honest move when
    one structured field cannot represent two senders). The label
    upstream (:func:`_label_agent_sends`, prepended *before* coalesce
    in :func:`materialize_messages`) carries the attribution into the
    text of the merged turn; assert via labelled output that every
    source still appears.
    """
    labelled = list(_label_agent_sends(history))
    coalesced = _coalesce_adjacent_users(labelled)
    input_sources = {m.source for m in history if isinstance(m, AgentSendMessage)}
    merged_text = "".join(
        m.text for m in coalesced if isinstance(m, (UserMessage, AgentSendMessage))
    )
    missing = {src for src in input_sources if f"[from {src}]: " not in merged_text}
    assert not missing, (
        f"sources lost during coalesce: {sorted(missing)!r} output_text={merged_text!r}"
    )


@settings(max_examples=200, deadline=None)
@given(_message_history())
def test_coalesce_preserves_total_text_length(
    history: list[ModelContextEvent],
) -> None:
    r"""User-side text content is preserved (modulo ``\n\n`` join separators)."""
    coalesced = _coalesce_adjacent_users(history)
    input_user_text_len = sum(
        len(m.text) for m in history if isinstance(m, (UserMessage, AgentSendMessage))
    )
    output_user_text_len = sum(
        len(m.text) for m in coalesced if isinstance(m, (UserMessage, AgentSendMessage))
    )
    # Output length = input length + 2 (the ``\n\n``) per coalesce join.
    # Conservatively: output >= input length, output <= input + 2 * input_count.
    assert output_user_text_len >= input_user_text_len, (
        f"text lost during coalesce: in={input_user_text_len}"
        f" out={output_user_text_len}"
    )


@settings(max_examples=100, deadline=None)
@given(_message_history())
def test_coalesce_does_not_reorder_non_user_entries(
    history: list[ModelContextEvent],
) -> None:
    """AssistantMessages and ToolResults stay in their original order."""
    coalesced = _coalesce_adjacent_users(history)
    input_non_user = [
        m for m in history if isinstance(m, (AssistantMessage, ToolResult))
    ]
    output_non_user = [
        m for m in coalesced if isinstance(m, (AssistantMessage, ToolResult))
    ]
    assert input_non_user == output_non_user, (
        "AssistantMessage/ToolResult sequence changed during coalesce"
    )


# -- materialize_messages ----------------------------------------------------


@settings(max_examples=100, deadline=None)
@given(_message_history())
def test_materialize_is_idempotent(history: list[ModelContextEvent]) -> None:
    """Materializing twice yields the same result as once.

    Catches the audit's bug6 hardening note: ``_label_agent_sends``
    prepends ``[from X]: `` unconditionally, so re-materializing would
    produce ``[from X]: [from X]: ...``. Today the call site only
    materializes once, but a future caller re-feeding output back
    through would compound prefixes silently.
    """
    once = materialize_messages(history)
    twice = materialize_messages(once)
    assert once == twice, (
        f"materialize_messages is not idempotent; diff:\nonce={once!r}\ntwice={twice!r}"
    )


@settings(max_examples=100, deadline=None)
@given(_message_history())
def test_materialize_preserves_assistant_count_when_unbudgeted(
    history: list[ModelContextEvent],
) -> None:
    """Without a tool-result budget, materialization doesn't drop assistants."""
    materialized = materialize_messages(history)
    input_assistants = sum(1 for m in history if isinstance(m, AssistantMessage))
    output_assistants = sum(1 for m in materialized if isinstance(m, AssistantMessage))
    assert input_assistants == output_assistants, (
        f"AssistantMessage count changed: input={input_assistants}"
        f" output={output_assistants}"
    )


# -- _last_assistant_result --------------------------------------------------


@settings(max_examples=200, deadline=None)
@given(_message_history())
def test_last_assistant_result_returns_string(
    history: list[ModelContextEvent],
) -> None:
    """``_last_assistant_result`` always returns a ``ToolResult`` with a
    string ``content`` field; no crashes on any history shape.
    """
    r = _last_assistant_result(list(history))
    assert isinstance(r.content, str)


@settings(max_examples=100, deadline=None)
@given(_message_history())
def test_last_assistant_result_returns_empty_when_no_assistant(
    history: list[ModelContextEvent],
) -> None:
    """When history has no ``AssistantMessage``, the result content is ``""``."""
    if not any(isinstance(m, AssistantMessage) for m in history):
        r = _last_assistant_result(list(history))
        assert r.content == ""


@settings(max_examples=200, deadline=None)
@given(lists(_agent_send_with_tool_call(), min_size=1, max_size=3))
def test_last_assistant_result_picks_most_recent_send(
    history: list[AssistantMessage],
) -> None:
    """When the LAST assistant turn has AgentSend calls, return the
    LAST AgentSend's content (not the first).

    Catches bug F: the original two-AgentSend-in-one-turn case
    returned the first send's content, burying the more recent intent.
    """
    last = history[-1]
    # The last AgentSend in the last assistant's tool_calls is what
    # should win. Find it.
    sends = [tc for tc in last.tool_calls if tc.name == "AgentSend"]
    if not sends:
        return  # composite always produces at least one but defensive
    expected = sends[-1].args["content"]
    typed_history = cast("list[ModelContextEvent]", history)
    r = _last_assistant_result(typed_history)
    assert r.content == expected, (
        f"expected last AgentSend's content {expected!r}; got {r.content!r}"
    )


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
