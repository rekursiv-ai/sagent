"""Property-based tests for pure data-flow functions in the runtime layer.

Hypothesis emits random message sequences and asserts structural
properties that should hold across every input shape. Targets pure
functions (no async, no state): ``_label_agent_sends``,
``_coalesce_adjacent_users``, ``materialize_messages``,
``_last_assistant_result``, and ``_sanitize_for_send`` (the runtime's
tool-pairing rescue path -- the executable form of the ToolCall ->
ToolResult wire invariant). Async runtime state-machine fuzzing lives
elsewhere; the runtime's async coupling is a poor fit for hypothesis.

Each test below would have caught at least one bug we've fixed this
session:

* Bug H (cross-source coalesce) -- ``coalesce_preserves_each_source``.
* Bug F (two AgentSends in one turn) -- ``last_assistant_result_picks_most_recent_send``.
* ``_label_agent_sends`` not idempotent under double materialization
  (audit-flagged hardening) -- ``materialize_is_idempotent``.
"""

from __future__ import annotations

from collections.abc import Sequence
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

from sagent.agent.context import (
    InvalidContextError,
    validate_context,
    wire_role,
)
from sagent.agent.runtime import _sanitize_for_send
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


# -- _sanitize_for_send: the ToolCall -> ToolResult wire invariant ----------
#
# ``_sanitize_for_send`` is the runtime's rescue path: it takes any history
# and returns a tool-pairing-correct sequence. The invariant it enforces --
# "every ``tool_use`` gets exactly one ``tool_result``; no orphan results; no
# duplicates; no user turn while a call is pending" -- is the load-bearing
# wire contract behind every detached / mimicked / compacted edge case. These
# properties make that invariant executable across arbitrary input shapes,
# so a future change to the rescue path that silently strands a call is
# caught by a counterexample rather than a live HTTP 400.


def _tool_pairing_violation(entries: Sequence[ModelContextEvent]) -> str | None:
    """Return a description of the first tool-pairing violation, or ``None``.

    Independent re-implementation of the pairing half of
    :func:`validate_context` (no role-alternation check), so the property
    asserts against a second witness rather than the function under test's
    own logic.
    """
    pending: set[str] = set()
    seen: set[str] = set()
    for entry in entries:
        if isinstance(entry, AssistantMessage):
            if pending:
                return f"new assistant turn with unanswered calls: {sorted(pending)}"
            pending = {tc.id for tc in entry.tool_calls}
        elif isinstance(entry, ToolResult):
            if entry.call_id in seen:
                return f"duplicate ToolResult for {entry.call_id!r}"
            if entry.call_id not in pending:
                return f"orphan ToolResult for {entry.call_id!r}"
            pending.discard(entry.call_id)
            seen.add(entry.call_id)
        elif pending:
            return f"user turn while calls pending: {sorted(pending)}"
    if pending:
        return f"unanswered calls at end: {sorted(pending)}"
    return None


@composite
def _unique_id_history(draw: DrawFn) -> list[ModelContextEvent]:
    """Random history with GLOBALLY-UNIQUE tool-call ids, like real tape.

    The runtime mints a fresh id per ``ToolCall``; an id is never reused
    across two ``AssistantMessage`` turns. The generic ``_message_history``
    strategy can collide ids (``f"tc-{i}-{rand}"``), which would let the same
    id be "unpaired" twice and make ``_sanitize_for_send`` synthesize two
    ``[interrupted]`` results for it -- a duplicate that the real id scheme
    makes impossible. This strategy re-stamps every call id from a single
    counter so the pairing properties test the input shape the runtime
    actually produces. Tool results, orphans, and unpaired calls are all
    still emitted (the rescue path's job) -- only id-collision is excluded.
    """
    counter = 0
    history: list[ModelContextEvent] = []
    pending: list[str] = []
    n = draw(integers(min_value=0, max_value=12))
    for _ in range(n):
        choice = draw(sampled_from(["user", "agent", "assistant", "tool_result"]))
        if choice == "user":
            history.append(draw(_user_message()))
        elif choice == "agent":
            history.append(draw(_agent_send_message()))
        elif choice == "assistant":
            am = draw(_assistant_message())
            calls = tuple(
                ToolCall(id=f"u{counter + i}", name=tc.name, args=tc.args)
                for i, tc in enumerate(am.tool_calls)
            )
            counter += len(calls)
            history.append(AssistantMessage(text=am.text, tool_calls=calls))
            pending.extend(tc.id for tc in calls)
        elif choice == "tool_result" and pending:
            history.append(ToolResult(call_id=pending.pop(0), content=draw(_TEXT)))
    # Emit a few genuine orphan results (ids never requested) so the rescue
    # path's orphan-drop is exercised; these are distinct from id-collision.
    if draw(booleans()):
        history.append(ToolResult(call_id=f"orphan{counter}", content=draw(_TEXT)))
    return history


@composite
def _alternating_history(draw: DrawFn) -> list[ModelContextEvent]:
    """Random history that never emits two consecutive same-wire-role turns.

    Mirrors the real tape the runtime builds (it never appends
    assistant-after-assistant or user-after-user). Tool results always
    immediately follow their declaring assistant, before the next turn --
    the shape a wire-valid context requires -- so ``_sanitize_for_send`` on
    this input should produce fully :func:`validate_context`-clean output,
    not merely pairing-clean output.
    """
    n = draw(integers(min_value=0, max_value=10))
    history: list[ModelContextEvent] = []
    prev_role: str | None = None
    counter = 0
    for _ in range(n):
        want_user = draw(booleans())
        if prev_role != "user" and want_user:
            history.append(
                draw(_user_message())
                if draw(booleans())
                else draw(_agent_send_message())
            )
            prev_role = "user"
        elif prev_role != "assistant":
            am = draw(_assistant_message())
            # Re-stamp call ids unique across the whole history.
            calls = tuple(
                ToolCall(id=f"c{counter + i}", name=tc.name, args=tc.args)
                for i, tc in enumerate(am.tool_calls)
            )
            counter += len(calls)
            am = AssistantMessage(text=am.text, tool_calls=calls)
            history.append(am)
            # Answer every call immediately (wire-valid pairing); randomly
            # drop some so the rescue path has orphans/unpaired to repair.
            for tc in calls:
                # Not list.extend: each append is gated on a fresh per-call
                # ``draw`` (drop-or-keep) and a fresh ``draw`` for content.
                if draw(booleans()):
                    history.append(  # noqa: PERF401
                        ToolResult(call_id=tc.id, content=draw(_TEXT))
                    )
            prev_role = None if calls else "assistant"
    return history


@settings(max_examples=300, deadline=None)
@given(_unique_id_history())
def test_sanitize_output_satisfies_tool_pairing(
    history: list[ModelContextEvent],
) -> None:
    """``_sanitize_for_send`` output is tool-pairing-correct for ANY input.

    The core ToolCall -> ToolResult invariant: regardless of orphans,
    duplicates, or unpaired calls in the input, the sanitized output has
    every ``tool_use`` answered exactly once, no orphan results, and no
    user turn interleaved with pending calls.
    """
    out = _sanitize_for_send(history)
    violation = _tool_pairing_violation(out)
    assert violation is None, (
        f"sanitized output violates pairing: {violation}\nout={out!r}"
    )


@settings(max_examples=300, deadline=None)
@given(_unique_id_history())
def test_sanitize_is_idempotent(history: list[ModelContextEvent]) -> None:
    """Sanitizing twice equals sanitizing once.

    A repair pass that is not a fixed point would keep mutating already-valid
    context (e.g. re-synthesizing ``[interrupted]`` results), drifting history
    on every gate iteration.
    """
    once = _sanitize_for_send(history)
    twice = _sanitize_for_send(once)
    assert once == twice, f"not idempotent:\nonce={once!r}\ntwice={twice!r}"


@settings(max_examples=300, deadline=None)
@given(_unique_id_history())
def test_sanitize_never_invents_call_ids(
    history: list[ModelContextEvent],
) -> None:
    """Every ToolResult in the output answers a call id present in the input.

    The rescue path may synthesize ``[interrupted]`` results for unpaired
    *input* calls, but it must never fabricate a result for a call id that
    was never requested -- that would be an orphan the provider rejects.
    """
    input_call_ids = {
        tc.id for m in history if isinstance(m, AssistantMessage) for tc in m.tool_calls
    }
    out = _sanitize_for_send(history)
    for entry in out:
        if isinstance(entry, ToolResult):
            assert entry.call_id in input_call_ids, (
                f"sanitize invented a result for unknown call id {entry.call_id!r}"
            )


@settings(max_examples=300, deadline=None)
@given(_alternating_history())
def test_sanitize_of_alternating_history_fully_validates(
    history: list[ModelContextEvent],
) -> None:
    """On wire-alternation-respecting input, output passes ``validate_context``.

    ``_sanitize_for_send`` repairs tool pairing but NOT assistant/user role
    alternation (the runtime maintains that by construction, never appending
    two same-role turns). So full ``validate_context`` cleanliness is asserted
    only for input that already respects alternation -- the shape the runtime
    actually produces. This pins that the rescue path closes every pairing
    gap such input can contain.
    """
    out = _sanitize_for_send(history)
    try:
        validate_context(out)
    except InvalidContextError as exc:  # pragma: no cover - failure path
        roles = [wire_role(m) for m in out]
        raise AssertionError(
            f"sanitized alternating history failed validate_context: {exc}\n"
            f"roles={roles}\nout={out!r}"
        ) from exc


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
