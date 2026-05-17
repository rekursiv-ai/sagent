"""Tests for ``repl.render``: ``RuntimeEvent`` -> ``Printer`` dispatch."""

from __future__ import annotations

from typing import cast, override

from sagent.repl.render import (
    HALT_MESSAGE,
    HELP_TEXT,
    RecordingPrinter,
    make_render_observer,
    render_tool_result,
)
from sagent.types.exceptions import AuthRefreshError
from sagent.types.history import (
    AssistantMessage,
    ToolCall,
    ToolResult,
    UserMessage,
)
from sagent.types.runtime import (
    BudgetReset,
    ChildDoneEvent,
    ChildEvent,
    ModelResponseComplete,
    ModelResponseError,
    ModelResponsePartial,
    ModelResponseThinking,
    ModelSwitchRejected,
    RuntimeEvent,
    ToolLabel,
    ToolResultPartial,
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
            "RuntimeEvent",
            ToolCall(id="x", name="y", args={}),
        )
    )
    assert p.markdowns == []
    assert p.tool_errors == []


def test_render_observer_error_emits_error_line() -> None:
    class BoomPrinter(RecordingPrinter):
        @override
        def write_user_bar(self, text: str) -> None:
            del text
            raise RuntimeError("printer down")

    p = BoomPrinter()
    obs = make_render_observer(p)
    obs(UserMessage(text="oops"))
    # Safety net: a write_tool_error must be recorded.
    assert any("render failed" in e for e in p.tool_errors)


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


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
