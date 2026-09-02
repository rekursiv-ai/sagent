"""Tests for ``repl.render``: ``RuntimeEvent`` -> ``Printer`` dispatch."""

from __future__ import annotations

from typing import cast, override

import time

import pytest

from sagent.repl.render import (
    _STREAM_BUF_FLUSH_BYTES,
    HALT_MESSAGE,
    HELP_TEXT,
    RecordingPrinter,
    make_render_observer,
    render_tool_result,
    service_suspended_text,
    strict_observer,
)
from sagent.tools.display import OutputSpec
from sagent.types.exceptions import (
    AuthRefreshError,
    ContextOverflowError,
)
from sagent.types.runtime import (
    AgentSendDeferredMessage,
    AgentSendMessage,
    AgentSendQueuedMessage,
    AssistantMessage,
    BudgetReset,
    ChildDoneEvent,
    ChildEvent,
    CompactComplete,
    CompactFailed,
    CompactStarted,
    DetachedResult,
    ModelResponseComplete,
    ModelResponseError,
    ModelResponsePartial,
    ModelResponseThinking,
    ModelServiceSuspended,
    ModelSwitchRejected,
    NoticeMessage,
    RuntimeEvent,
    ServiceErrorSnapshot,
    ToolCall,
    ToolLabel,
    ToolResult,
    ToolResultPartial,
    UserMessage,
)


def test_render_tool_result_error_only_emits_error() -> None:
    p = RecordingPrinter()
    render_tool_result(
        p,
        ToolResult(call_id="c1", content="boom", is_error=True, hint="x", summary="y"),
    )
    assert p.tool_errors == ["boom"]
    assert p.hints == []
    assert p.tool_summaries == []


def test_render_tool_result_success_emits_diff_hint_summary() -> None:
    p = RecordingPrinter()
    render_tool_result(
        p,
        ToolResult(
            call_id="c1",
            content="ok",
            diff="--- a\n+++ b\n",
            diff_file_path="x.py",
            hint="be careful",
            summary="1 file",
        ),
    )
    assert p.diffs == [("--- a\n+++ b\n", "x.py")]
    assert p.hints == ["be careful"]
    assert p.tool_summaries == ["1 file"]


def test_render_tool_result_missing_fields_omitted() -> None:
    p = RecordingPrinter()
    render_tool_result(p, ToolResult(call_id="c1", content="ok"))
    assert p.diffs == []
    assert p.hints == []
    assert p.tool_summaries == []


def test_user_message_flushes_stream_then_writes_bar() -> None:
    p = RecordingPrinter()
    obs = make_render_observer(p)
    # Partial that has no stable boundary -> stays buffered.
    obs(ModelResponsePartial(text="incomplete"))
    obs(UserMessage(text="hello"))
    # Flush should emit the buffered text as markdown, then bar.
    assert p.markdowns == ["incomplete"]
    assert p.user_bars == ["hello"]


def test_hidden_user_message_is_not_rendered() -> None:
    """A ``hidden`` message is sent to the model but not shown to the human."""
    p = RecordingPrinter()
    obs = make_render_observer(p)
    obs(UserMessage(text="visible"))
    obs(UserMessage(text="<system-reminder>nudge</system-reminder>", hidden=True))
    # The visible one renders; the hidden one does not.
    assert p.user_bars == ["visible"]


def test_agent_send_message_routes_to_agent_bar_with_source() -> None:
    """Incoming ``AgentSendMessage`` renders attributed to its source.

    Conflating it with ``UserMessage`` (the prior behavior) hid the
    sender label from the human UI: the parent had no visual cue
    distinguishing "user typed this" from "child agent sent this".
    """
    p = RecordingPrinter()
    obs = make_render_observer(p)
    obs(AgentSendMessage(source="reviewer", text="report body"))
    assert p.agent_bars == [("reviewer", "report body")], (
        f"AgentSendMessage must route to write_agent_bar with the source"
        f" label; got agent_bars={p.agent_bars!r} user_bars={p.user_bars!r}"
    )
    assert p.user_bars == [], (
        "AgentSendMessage must NOT fall through to write_user_bar -- the"
        " user bar is reserved for the live human's input"
    )


def test_model_switch_rejected_emits_error_without_halt() -> None:
    p = RecordingPrinter()
    obs = make_render_observer(p)
    obs(ModelSwitchRejected(exception=ValueError("too small")))
    assert p.tool_errors == ["ValueError: too small"]
    assert p.halts == []


def test_budget_reset_emits_notification_line() -> None:
    """``BudgetReset`` surfaces as a ``[/model] budget reset ...`` line."""
    p = RecordingPrinter()
    obs = make_render_observer(p)
    obs(
        BudgetReset(
            model_id="claude-sonnet-4-6",
            prior_max_request_tokens=1_000_000,
            prior_max_response_tokens=32_000,
            new_max_request_tokens=200_000,
            new_max_response_tokens=16_000,
        )
    )
    assert len(p.lines) == 1
    assert "claude-sonnet-4-6" in p.lines[0]
    assert "1,000,000" in p.lines[0]
    assert "200,000" in p.lines[0]


def test_compact_started_emits_dim_line() -> None:
    p = RecordingPrinter()
    obs = make_render_observer(p)
    obs(CompactStarted())
    assert p.dim_lines == ["[compacting history…]"]


def test_compact_complete_emits_progress_dim_line() -> None:
    p = RecordingPrinter()
    obs = make_render_observer(p)
    obs(
        CompactComplete(
            token_before=42,
            token_after=8,
            payload_entries=2,
        )
    )
    assert p.dim_lines == [
        "[compaction complete: ~42 → ~8 tokens, 2 entries]",
    ]


def test_compact_failed_emits_error_dim_line() -> None:
    p = RecordingPrinter()
    obs = make_render_observer(p)
    obs(CompactFailed(exception=RuntimeError("ctx full"), tape_len=10))
    assert p.dim_lines == ["[compaction failed: RuntimeError: ctx full]"]


def test_compact_fallback_emits_progress_dim_line() -> None:
    p = RecordingPrinter()
    obs = make_render_observer(p)
    obs(
        CompactComplete(
            payload_entries=1,
            fallback_reason="summary failed after 3 attempts",
            preserved_tail_count=1,
        )
    )
    assert p.dim_lines == [
        "[compaction fallback: summary failed after 3 attempts; preserved 1 tail entry]",
    ]


def test_user_message_with_empty_buffer() -> None:
    p = RecordingPrinter()
    obs = make_render_observer(p)
    obs(UserMessage(text="hi"))
    assert p.user_bars == ["hi"]
    assert p.markdowns == []


def test_model_response_partial_buffers_until_boundary() -> None:
    p = RecordingPrinter()
    obs = make_render_observer(p)
    # Two paragraphs ⇒ stable boundary on the first.
    obs(ModelResponsePartial(text="first para\n\nsecond para"))
    # First paragraph committed; second still buffered.
    assert p.markdowns == ["first para"]


def test_model_response_complete_flushes_remaining() -> None:
    p = RecordingPrinter()
    obs = make_render_observer(p)
    obs(ModelResponsePartial(text="hanging"))
    obs(
        ModelResponseComplete(
            message=AssistantMessage(text="hanging"),
        )
    )
    assert p.markdowns == ["hanging"]


def test_model_response_thinking_routes_to_printer() -> None:
    p = RecordingPrinter()
    obs = make_render_observer(p)
    obs(ModelResponseThinking(text="hmm"))
    assert p.thinkings == ["hmm"]


def test_model_response_thinking_can_be_hidden() -> None:
    """The printer owns the flag, and the observer reads it per event."""
    p = RecordingPrinter()
    p.show_thinking = False
    obs = make_render_observer(p)
    obs(ModelResponseThinking(text="hmm"))
    assert p.thinkings == []


def test_hiding_thinking_takes_effect_mid_stream() -> None:
    """``/thinking hide`` must apply to the response already in flight."""
    p = RecordingPrinter()
    obs = make_render_observer(p)
    obs(ModelResponseThinking(text="shown"))
    p.show_thinking = False
    obs(ModelResponseThinking(text="hidden"))
    assert p.thinkings == ["shown"]


def test_model_service_suspended_flushes_stream_and_renders_dim_line() -> None:
    p = RecordingPrinter()
    obs = make_render_observer(p)
    obs(ModelResponsePartial(text="pending"))

    obs(
        ModelServiceSuspended(
            provider="OpenAISubscription",
            auth="credentials",
            account="default",
            model_id="gpt-5.5",
            retry_at=1_800_000_000.0,
            delay_sec=120.0,
            server_supplied=True,
            error=ServiceErrorSnapshot(
                type_name="RateLimitError",
                message="limited",
                status=429,
            ),
        )
    )

    assert p.markdowns == ["pending"]
    assert len(p.dim_lines) == 1
    assert "model service suspended" in p.dim_lines[0]
    assert "rate-limited" in p.dim_lines[0]
    assert "resumes at" in p.dim_lines[0]


def test_notice_message_advisory_renders_dim_line() -> None:
    p = RecordingPrinter()
    obs = make_render_observer(p)
    obs(ModelResponsePartial(text="pending"))
    obs(NoticeMessage(text="[rate-limit warning: 89% of weekly]", tier="advisory"))
    assert p.markdowns == ["pending"]
    assert p.dim_lines == ["[rate-limit warning: 89% of weekly]"]
    assert p.halts == []


def test_child_notice_message_renders_block() -> None:
    # Issue#316 RUNTIME-001: a NoticeMessage forwarded from a child
    # (agent_spawn always-forwards it) must render as a child block, not
    # silently vanish in the child renderer.
    p = RecordingPrinter()
    obs = make_render_observer(p)
    notice = NoticeMessage(text="[usage: 7d window 89% used]", tier="advisory")
    obs(ChildEvent(label="Agent_0", inner=notice))
    obs(ChildDoneEvent(label="Agent_0", elapsed=0.0, tokens=0, cost=0.0))
    assert p.child_blocks
    label, items = p.child_blocks[0]
    assert label == "Agent_0"
    assert items == [notice]


def test_child_model_response_error_renders_block() -> None:
    p = RecordingPrinter()
    obs = make_render_observer(p)
    err = ModelResponseError(RuntimeError("boom"))
    obs(ChildEvent(label="Agent_0", inner=err))
    obs(ChildDoneEvent(label="Agent_0", elapsed=0.0, tokens=0, cost=0.0))
    assert p.child_blocks
    label, items = p.child_blocks[0]
    assert label == "Agent_0"
    assert items == [err]


def test_every_always_forwarded_event_renders_a_child_block() -> None:
    # Binding test: the set the forwarder always crosses to the parent must be
    # a subset of what the child renderer materializes. Fails the instant a new
    # always-forwarded event type is added without a child render path.
    from sagent.tools.agent_spawn import (  # noqa: PLC0415 -- avoids heavy agent_spawn import at module load
        always_forwarded_sample,
    )

    for event in always_forwarded_sample():
        p = RecordingPrinter()
        obs = make_render_observer(p)
        obs(ChildEvent(label="L", inner=event))
        obs(ChildDoneEvent(label="L", elapsed=0.0, tokens=0, cost=0.0))
        assert p.child_blocks, (
            f"{type(event).__name__} is always-forwarded but renders no child block"
        )


def test_service_suspended_text_short_wait_renders_relative_seconds() -> None:
    """Short waits use ``resumes in Ns`` rather than wall-clock format."""
    event = ModelServiceSuspended(
        provider="anthropic",
        auth="key",
        account="default",
        model_id="claude-test",
        retry_at=time.time() + 12.0,
        delay_sec=12.0,
        server_supplied=False,
        error=ServiceErrorSnapshot(type_name="Boom", message="500", status=500),
    )
    text = service_suspended_text(event)
    assert "resumes in" in text
    assert "resumes at" not in text
    assert "temporarily blocked" in text


def test_tool_label_flushes_stream() -> None:
    p = RecordingPrinter()
    obs = make_render_observer(p)
    obs(ModelResponsePartial(text="incomplete"))
    obs(ToolLabel(call_id="c1", text="Bash"))
    assert p.markdowns == ["incomplete"]
    assert p.tool_labels == ["Bash"]


def test_tool_result_routed() -> None:
    p = RecordingPrinter()
    obs = make_render_observer(p)
    obs(ToolResult(call_id="c1", content="ok", summary="done"))
    assert p.tool_summaries == ["done"]


def test_tool_result_partial_routes_to_chunk() -> None:
    p = RecordingPrinter()
    obs = make_render_observer(p)
    obs(ToolResultPartial(call_id="c1", text="streaming"))
    assert p.chunks == ["streaming"]


def test_model_response_error_emits_error_and_halt() -> None:
    p = RecordingPrinter()
    obs = make_render_observer(p)
    obs(ModelResponseError(exception=RuntimeError("creds expired")))
    assert any("creds expired" in e for e in p.tool_errors)
    assert p.halts == [HALT_MESSAGE]


def test_auth_refresh_error_renders_without_class_name_prefix() -> None:
    """``AuthRefreshError`` (any ``UserFacingError``) renders ``str(exc)`` only.

    Plain exceptions get the ``ClassName: message`` shape so the
    operator can identify the failure type. ``UserFacingError`` carries
    a polished, user-actionable message; prefixing it with the class
    name ("AuthRefreshError: ...") just adds Python-internals noise to
    text the user is supposed to read and act on. Drop the prefix.
    """
    p = RecordingPrinter()
    obs = make_render_observer(p)
    msg = "OpenAI Codex subscription session expired. Run /login to re-authenticate."
    obs(ModelResponseError(exception=AuthRefreshError(msg)))

    rendered = " ".join(p.tool_errors)
    assert msg in rendered, (
        f"renderer must surface the polished message; got tool_errors={p.tool_errors!r}"
    )
    assert "AuthRefreshError" not in rendered, (
        f"UserFacingError messages must not be prefixed with their class "
        f"name; got tool_errors={p.tool_errors!r}"
    )
    assert "Traceback" not in rendered
    assert "HTTPStatusError" not in rendered


def test_auth_refresh_error_uses_auth_specific_halt_banner() -> None:
    """``AuthRefreshError`` halt banner mentions ``/login`` / ``/model``, not "retry".

    The generic ``HALT_MESSAGE`` suggests "type to retry" -- but for an
    expired refresh token, typing anything just re-fires the same auth
    failure. The banner should reflect that only ``/login`` (or
    switching provider via ``/model``) resolves the failure.
    """
    p = RecordingPrinter()
    obs = make_render_observer(p)
    obs(ModelResponseError(exception=AuthRefreshError("session expired")))

    assert len(p.halts) == 1
    banner = p.halts[0]
    assert banner != HALT_MESSAGE, (
        f"AuthRefreshError must use the auth-specific banner; "
        f"got the generic HALT_MESSAGE: {banner!r}"
    )
    assert "/login" in banner, (
        f"auth-specific banner must mention /login; got {banner!r}"
    )
    # "retry" implies typing more text fixes it; for auth, it doesn't.
    assert "retry" not in banner.lower(), (
        f"auth banner must not suggest retry (typing won't help an "
        f"expired refresh token); got {banner!r}"
    )


def test_context_overflow_error_uses_context_specific_halt_banner() -> None:
    """Context exhaustion needs a repair action, not a plain retry.

    Typing another message after auto-compaction failed just sends the
    same oversized transcript through the same failing path. The halt
    banner should point at the actions that can actually change prompt
    shape: clear, compact with guidance, or switch model.
    """
    p = RecordingPrinter()
    obs = make_render_observer(p)
    obs(ModelResponseError(exception=ContextOverflowError("context exhausted")))

    assert len(p.halts) == 1
    banner = p.halts[0]
    assert banner != HALT_MESSAGE, (
        f"ContextOverflowError must not use the generic retry banner; got {banner!r}"
    )
    assert "/clear" in banner
    assert "/compact" in banner
    assert "/model" in banner
    assert "retry" not in banner.lower()


def test_plain_exception_keeps_class_name_prefix_and_generic_banner() -> None:
    """Unexpected exceptions retain the ``ClassName: message`` shape + generic banner.

    Plain failures help the operator diagnose; the class name is
    diagnostic. The generic banner ("type to retry") is appropriate
    since retry might succeed.
    """
    p = RecordingPrinter()
    obs = make_render_observer(p)
    obs(ModelResponseError(exception=RuntimeError("boom")))

    rendered = " ".join(p.tool_errors)
    assert "RuntimeError: boom" in rendered, (
        f"plain exception should keep class-name prefix; got {p.tool_errors!r}"
    )
    assert p.halts == [HALT_MESSAGE]


def test_dispatch_handles_unknown_events_silently() -> None:
    # ``RecordingPrinter`` is not a RuntimeEvent; an unrelated event
    # type should be ignored without raising.
    p = RecordingPrinter()
    obs = make_render_observer(p)
    # ``ToolCall`` is part of an AssistantMessage but never a top-level event.
    # The dispatcher should hit the ``case _`` branch -- no observable effect.
    obs(
        cast(
            RuntimeEvent,
            ToolCall(id="x", name="y", args={}),
        )
    )
    assert p.markdowns == []
    assert p.tool_errors == []


def test_render_observer_error_logs_without_reentering_printer(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The safety net must not recurse through the failing printer.

    A printer that raises was previously sent ``write_tool_error`` from
    the catch-all -- which would re-raise from the same broken sink and
    bury the original failure. The safety net now logs only.
    """

    class BoomPrinter(RecordingPrinter):
        @override
        def write_user_bar(self, text: str) -> None:
            del text
            raise RuntimeError("printer down")

    p = BoomPrinter()
    obs = make_render_observer(p)
    with caplog.at_level("ERROR", logger="sagent.repl.render"):
        obs(UserMessage(text="oops"))
    assert p.tool_errors == [], (
        f"safety net must not re-enter the failing printer; "
        f"got tool_errors={p.tool_errors!r}"
    )
    assert any("render observer failed" in r.message for r in caplog.records)


def test_child_event_streaming_partial_buffers_until_boundary() -> None:
    p = RecordingPrinter()
    obs = make_render_observer(p)
    obs(
        ChildEvent(
            label="Agent_0",
            inner=ModelResponsePartial(text="first\n\nsecond"),
        )
    )
    assert p.child_blocks
    label, items = p.child_blocks[0]
    assert label == "Agent_0"
    assert any(isinstance(i, AssistantMessage) for i in items)


def test_child_event_atomic_tool_label_emits_block() -> None:
    p = RecordingPrinter()
    obs = make_render_observer(p)
    obs(ChildEvent(label="Agent_0", inner=ToolLabel(call_id="c1", text="Bash")))
    label, items = p.child_blocks[0]
    assert label == "Agent_0"
    assert any(isinstance(i, ToolLabel) for i in items)


def test_child_event_atomic_tool_result_emits_block() -> None:
    p = RecordingPrinter()
    obs = make_render_observer(p)
    obs(
        ChildEvent(
            label="Agent_0",
            inner=ToolResult(call_id="c1", content="done"),
        )
    )
    label, items = p.child_blocks[0]
    assert label == "Agent_0"
    assert any(isinstance(i, ToolResult) for i in items)


def test_child_event_thinking_emits_block() -> None:
    p = RecordingPrinter()
    obs = make_render_observer(p)
    obs(
        ChildEvent(
            label="Agent_0",
            inner=ModelResponseThinking(text="thinking"),
        )
    )
    label, items = p.child_blocks[0]
    assert label == "Agent_0"
    assert any(isinstance(i, ModelResponseThinking) for i in items)


def test_child_event_model_service_suspended_emits_block() -> None:
    p = RecordingPrinter()
    obs = make_render_observer(p)
    suspended = ModelServiceSuspended(
        provider="OpenAISubscription",
        auth="credentials",
        account="default",
        model_id="gpt-5.5",
        retry_at=1_800_000_000.0,
        delay_sec=120.0,
        server_supplied=True,
        error=ServiceErrorSnapshot(
            type_name="RateLimitError",
            message="limited",
            status=429,
        ),
    )

    obs(ChildEvent(label="Agent_0", inner=suspended))

    label, items = p.child_blocks[0]
    assert label == "Agent_0"
    assert items == [suspended]


def test_child_event_user_message_emits_block() -> None:
    p = RecordingPrinter()
    obs = make_render_observer(p)
    obs(ChildEvent(label="Agent_0", inner=UserMessage(text="hi")))
    label, items = p.child_blocks[0]
    assert label == "Agent_0"
    assert any(isinstance(i, UserMessage) for i in items)


def test_child_event_unknown_inner_ignored() -> None:
    p = RecordingPrinter()
    obs = make_render_observer(p)
    obs(
        ChildEvent(
            label="Agent_0",
            inner=ModelResponseComplete(message=AssistantMessage(text="x")),
        )
    )
    # No matching atomic translator -> no child block.
    assert p.child_blocks == []


def test_child_event_switches_label_flushes_previous() -> None:
    p = RecordingPrinter()
    obs = make_render_observer(p)
    obs(
        ChildEvent(
            label="Agent_0",
            inner=ModelResponsePartial(text="first\n\nsecond"),
        )
    )
    p.child_blocks.clear()
    obs(ChildEvent(label="Agent_1", inner=ToolLabel(call_id="c1", text="Read")))
    # Switching label should flush any pending Agent_0 (none here),
    # then emit Agent_1 atomic event.
    assert any(label == "Agent_1" for label, _ in p.child_blocks)


def test_child_done_event_flushes_pending() -> None:
    p = RecordingPrinter()
    obs = make_render_observer(p)
    # Buffer some streaming text under Agent_0 but never hit a boundary.
    obs(
        ChildEvent(
            label="Agent_0",
            inner=ModelResponsePartial(text="no boundary"),
        )
    )
    obs(ChildDoneEvent(label="Agent_0", elapsed=1.0, tokens=10, cost=0.0))
    # Flush picks up the buffered partial as an AssistantMessage.
    labels = [label for label, _ in p.child_blocks]
    assert labels == ["Agent_0"]


def test_nested_child_event_grandchild_reaches_printer() -> None:
    """``ChildEvent(ChildEvent(...))`` must unwrap to the innermost event.

    A grandchild forwards through its parent as
    ``ChildEvent(label, ChildEvent(label, inner))``. Without recursive
    unwrap, ``_child_atomic_item`` sees the outer ``ChildEvent`` (not in
    its dispatch table), returns ``None``, and the grandchild's output
    silently disappears.
    """
    p = RecordingPrinter()
    obs = make_render_observer(p)
    obs(
        ChildEvent(
            label="Agent_0",
            inner=ChildEvent(
                label="Agent_0",
                inner=ToolLabel(call_id="c1", text="x"),
            ),
        ),
    )
    assert p.child_blocks, (
        "nested ChildEvent must render to a child block; grandchild output"
        " was dropped instead"
    )
    label, items = p.child_blocks[0]
    assert label == "Agent_0"
    assert any(isinstance(i, ToolLabel) and i.text == "x" for i in items)


def test_cross_child_boundary_flushes_pending_stream_text() -> None:
    """Switching active child label flushes the previous label's stream.

    Without this, a slow child's buffered partial output lingers until
    its ``ChildDoneEvent`` and renders interleaved with later children
    -- the wrong slot from the user's point of view.
    """
    p = RecordingPrinter()
    obs = make_render_observer(p)
    # Buffer streaming text under Agent_0 (no boundary, so it stays).
    obs(
        ChildEvent(
            label="Agent_0",
            inner=ModelResponsePartial(text="incomplete from 0"),
        ),
    )
    # Switch to Agent_1; should flush Agent_0's buffered text first.
    obs(ChildEvent(label="Agent_1", inner=ToolLabel(call_id="c1", text="x")))
    labels = [label for label, _ in p.child_blocks]
    assert "Agent_0" in labels, (
        f"Agent_0's buffered stream text must flush at boundary change;"
        f" got labels={labels!r}"
    )
    agent0_items = next(items for lbl, items in p.child_blocks if lbl == "Agent_0")
    assert any(
        isinstance(i, AssistantMessage) and "incomplete from 0" in i.text
        for i in agent0_items
    )


def test_stream_buf_flushes_at_size_cap() -> None:
    """An in-progress fence without a boundary still flushes past the cap.

    Without the cap, a 100K+-char in-progress fenced block would let
    the buffer grow unbounded for the whole round.
    """
    p = RecordingPrinter()
    obs = make_render_observer(p)
    # Open a fence and stream a long single-line body so
    # ``find_stable_boundary`` returns 0 (no paragraph break).
    obs(ModelResponsePartial(text="```\n" + "x" * (_STREAM_BUF_FLUSH_BYTES + 1)))
    assert p.markdowns, "stream buffer must flush once past the size cap"


def test_strict_observer_reraises_dispatch_failures() -> None:
    """REPL-026: ``strict_observer()`` flips the safety-net swallow off.

    By default, a printer that raises inside the dispatch is logged and
    swallowed so a renderer bug never tears down the agent loop. Tests
    legitimately need the exception to surface; ``strict_observer``
    re-raises so the assertion sees the real failure.
    """

    class _BoomPrinter(RecordingPrinter):
        @override
        def write_user_bar(self, text: str) -> None:
            del text
            raise RuntimeError("boom")

    obs = make_render_observer(_BoomPrinter())
    # Default: swallowed; no exception escapes the observer call.
    obs(UserMessage(text="hi"))
    # Strict: re-raised.
    with strict_observer(), pytest.raises(RuntimeError, match="boom"):
        obs(UserMessage(text="hi"))


def test_child_stream_buf_flushes_at_size_cap() -> None:
    r"""REPL-017: per-child stream buffer must mirror the parent's size cap.

    Parent stream has a 64KB cap; the per-child buffer in ``_consume_child``
    had no equivalent. A child streaming a long fenced block (no ``\n\n``
    boundary) would let ``_child_text[label]`` grow without bound for the
    whole round.
    """
    p = RecordingPrinter()
    obs = make_render_observer(p)
    # Open a fence under one child and stream a long single-line body so
    # ``find_stable_boundary`` returns 0 (no paragraph break).
    obs(
        ChildEvent(
            label="Agent_0",
            inner=ModelResponsePartial(
                text="```\n" + "x" * (_STREAM_BUF_FLUSH_BYTES + 1),
            ),
        ),
    )
    # The buffer for Agent_0 must be bounded by the same cap as the
    # parent stream buffer.
    child_text = obs._child_text.get("Agent_0", "")
    assert len(child_text) <= _STREAM_BUF_FLUSH_BYTES, (
        f"child stream buffer must flush at size cap;"
        f" len={len(child_text)} cap={_STREAM_BUF_FLUSH_BYTES}"
    )


def test_detached_result_routes_through_tool_result_path() -> None:
    """``DetachedResult`` renders the inner ``ToolResult`` to the user.

    Without this case, a previously-detached tool's late completion
    would not have any visual cue -- the user wouldn't know it landed.
    """
    p = RecordingPrinter()
    obs = make_render_observer(p)
    result = ToolResult(call_id="c1", content="done", summary="ok")
    obs(DetachedResult(result=result))
    assert p.tool_summaries == ["ok"], (
        f"DetachedResult must render the inner ToolResult; got"
        f" tool_summaries={p.tool_summaries!r}"
    )


def test_agent_send_queued_and_deferred_are_silent() -> None:
    """``AgentSendQueuedMessage`` / ``AgentSendDeferredMessage`` are
    intentionally silent.

    Both are inbox-side placeholders; the visible event is the
    ``AgentSendMessage`` that lands once the gate drains. Rendering the
    queued/deferred wrapper would double-render the same payload.
    """
    p = RecordingPrinter()
    obs = make_render_observer(p)
    obs(AgentSendQueuedMessage(source="r", text="t"))
    obs(AgentSendDeferredMessage(source="r", text="t"))
    assert p.user_bars == [], (
        f"queued/deferred wrappers must not write user bar; got {p.user_bars!r}"
    )
    assert p.agent_bars == [], (
        f"queued/deferred wrappers must not write agent bar; got {p.agent_bars!r}"
    )
    assert p.tool_errors == [], (
        f"queued/deferred wrappers must not raise errors; got {p.tool_errors!r}"
    )


def test_recording_printer_rendered_text_concats() -> None:
    p = RecordingPrinter()
    p.write_markdown("a")
    p.write_markdown("b")
    assert p.rendered_text == "ab"


def test_help_text_contains_core_commands() -> None:
    assert "/help" in HELP_TEXT
    assert "/quit" in HELP_TEXT
    assert "/exit" in HELP_TEXT
    assert "/clear" in HELP_TEXT
    assert "/send" in HELP_TEXT
    assert "glob" in HELP_TEXT


def test_help_text_documents_recompact_as_compact_alias() -> None:
    line = next(line for line in HELP_TEXT.splitlines() if "/recompact" in line)
    assert "alias" in line
    assert "/compact" in line
    assert "reload" not in line.lower()


def test_tool_result_content_renders_when_requested() -> None:
    """A successful result must be able to show its body.

    ``render_tool_result`` handled error/diff/hint/summary only, so a
    plain ``ls`` printed nothing at all -- there was no display path for
    ``content`` to switch on.
    """
    printer = RecordingPrinter()
    render_tool_result(
        printer,
        ToolResult(call_id="c1", content="alpha\nbeta"),
        output=OutputSpec(show=True, unbounded=True),
    )
    assert "alpha" in "".join(printer.tool_outputs)
    assert "beta" in "".join(printer.tool_outputs)


def test_tool_result_content_hidden_by_default() -> None:
    printer = RecordingPrinter()
    render_tool_result(printer, ToolResult(call_id="c1", content="alpha"))
    assert not printer.tool_outputs


def test_tool_result_content_is_line_capped() -> None:
    printer = RecordingPrinter()
    render_tool_result(
        printer,
        ToolResult(call_id="c1", content="\n".join(str(i) for i in range(100))),
        output=OutputSpec(show=True, head_rows=2, tail_rows=2),
    )
    body = "".join(printer.tool_outputs)
    assert "96 lines" in body, body


def test_tool_result_content_is_width_chopped() -> None:
    """``output_wrap=chop`` keeps the head and marks the cut."""
    printer = RecordingPrinter()
    render_tool_result(
        printer,
        ToolResult(call_id="c1", content="abcdefgh"),
        output=OutputSpec(show=True, unbounded=True, max_width=4, wrap="chop"),
    )
    assert printer.tool_outputs == ["abc\u2026"]


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)


def test_service_suspended_surfaces_the_provider_message() -> None:
    """The provider's explanation must reach the banner.

    A bare "temporarily blocked" label hid an entitlement message across
    four escalating retries -- the user saw a wait with no way to learn
    it would never clear.
    """
    event = ModelServiceSuspended(
        provider="OpenAISubscription",
        auth="credentials",
        account=None,
        model_id="gpt-5.6-sol+fast",
        retry_at=time.time() + 7.0,
        delay_sec=7.0,
        server_supplied=False,
        error=ServiceErrorSnapshot(
            type_name="RateLimitError",
            message="Usage credits are required for fast mode.",
            status=429,
        ),
    )
    text = service_suspended_text(event)
    assert "Usage credits are required for fast mode." in text
    assert "resumes in 7s" in text


def test_service_suspended_truncates_a_long_provider_message() -> None:
    """A verbose provider message must not bury the resume time.

    The banner is a single line; an unbounded message wraps across the
    pane and pushes "resumes in Ns" out of view.
    """
    event = ModelServiceSuspended(
        provider="p",
        auth="a",
        account=None,
        model_id="m",
        retry_at=time.time() + 7.0,
        delay_sec=7.0,
        server_supplied=False,
        error=ServiceErrorSnapshot(type_name="E", message="x" * 400, status=429),
    )
    text = service_suspended_text(event)
    assert len(text) < 200
    assert "resumes in 7s" in text
    assert text.endswith("]")
