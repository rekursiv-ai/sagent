"""Smoke tests for the v3 REPL surface.

Pinpoints the parser/observer translation; the end-to-end agent harness
lives in ``agent/agent_test.py``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import override

from prompt_toolkit.formatted_text import to_plain_text

from sagent.agent.agent import Agent
from sagent.custom_types import (
    ChildEvent,
    ErrorEvent,
    InterruptedEvent,
    ModelRequest,
    ModelResponse,
    MultipartMessage,
    Pricing,
    StatusUpdateEvent,
    StreamEndEvent,
    TextChunkEvent,
    TextMessage,
    ThinkingEvent,
    ToolLabelEvent,
    ToolResultEvent,
    TurnCompleteEvent,
    UserBarEvent,
)
from sagent.lib.descriptors import QUEUED_USER_MESSAGE
from sagent.repl.prompt import dynamic_prompt
from sagent.repl.render import (
    RecordingPrinter,
    make_render_observer,
)
from sagent.repl.slash import (
    Abort,
    AbortAll,
    Break,
    BreakAll,
    Clear,
    Compact,
    Help,
    Login,
    ModelSwitch,
    Quit,
    Recompact,
    Tasks,
    Text,
    Unknown,
    parse_slash,
)


class TestParseSlash:
    def test_quit(self) -> None:
        assert isinstance(parse_slash("/quit"), Quit)

    def test_clear(self) -> None:
        assert isinstance(parse_slash("/clear"), Clear)

    def test_compact_with_args(self) -> None:
        result = parse_slash("/compact keep API spec")
        assert isinstance(result, Compact)
        assert result.args == "keep API spec"

    def test_compact_no_args(self) -> None:
        result = parse_slash("/compact")
        assert isinstance(result, Compact)
        assert result.args == ""

    def test_recompact(self) -> None:
        result = parse_slash("/recompact")
        assert isinstance(result, Recompact)

    def test_model_switch(self) -> None:
        result = parse_slash("/model gpt-5")
        assert isinstance(result, ModelSwitch)
        assert result.args == "gpt-5"

    def test_provider_sugar(self) -> None:
        result = parse_slash("/provider Google")
        assert isinstance(result, ModelSwitch)
        assert result.args == "--provider Google"

    def test_break_all(self) -> None:
        assert isinstance(parse_slash("/break all"), BreakAll)

    def test_break_target(self) -> None:
        result = parse_slash("/break Agent_0")
        assert isinstance(result, Break)
        assert result.target == "Agent_0"

    def test_abort_all(self) -> None:
        assert isinstance(parse_slash("/abort all"), AbortAll)

    def test_abort_target(self) -> None:
        result = parse_slash("/abort Agent_0")
        assert isinstance(result, Abort)
        assert result.target == "Agent_0"

    def test_login(self) -> None:
        assert isinstance(parse_slash("/login"), Login)

    def test_help(self) -> None:
        assert isinstance(parse_slash("/help"), Help)

    def test_tasks(self) -> None:
        assert isinstance(parse_slash("/tasks"), Tasks)

    def test_unknown(self) -> None:
        result = parse_slash("/foo")
        assert isinstance(result, Unknown)
        assert "/foo" in result.text

    def test_plain_text(self) -> None:
        result = parse_slash("hello world")
        assert isinstance(result, Text)
        assert result.content == "hello world"

    def test_empty_returns_none(self) -> None:
        assert parse_slash("") is None
        assert parse_slash("   \n") is None


class TestQueuedFollowUps:
    def test_agent_lists_and_pops_latest_tab_queued_message(self) -> None:
        agent = Agent(model=_NoopModel())
        agent.inbox.put(TextMessage("inline", "text/x-user-message"))
        agent.inbox.put(TextMessage("first", QUEUED_USER_MESSAGE))
        agent.inbox.put(TextMessage("control", "text/x-quit"))
        agent.inbox.put(TextMessage("second", QUEUED_USER_MESSAGE))

        assert [str(m.content) for m in agent.queued_user_messages()] == [
            "first",
            "second",
        ]
        popped = agent.pop_latest_queued_user_message()

        assert popped is not None
        assert str(popped.content) == "second"
        assert [(m.descriptor, str(m.content)) for m in agent.inbox] == [
            ("text/x-user-message", "inline"),
            (QUEUED_USER_MESSAGE, "first"),
            ("text/x-quit", "control"),
        ]

    def test_dynamic_prompt_renders_only_tab_queued_follow_ups(self) -> None:
        agent = Agent(model=_NoopModel())
        agent.inbox.put(TextMessage("inline", "text/x-user-message"))
        agent.inbox.put(TextMessage("control", "text/x-quit"))
        agent.inbox.put(TextMessage("second\nline", QUEUED_USER_MESSAGE))

        text = to_plain_text(dynamic_prompt(agent))

        assert "Queued follow-up inputs" in text
        assert "inline" not in text
        assert "second (+1 more line)" in text
        assert "Shift+Left edit last queued message" in text
        assert "control" not in text


class _NoopModel:
    model_id = "noop"
    max_request_tokens = 16_000
    max_response_tokens = 8_000
    supports_thinking = False
    supports_effort = False
    supports_cache_control = False
    supports_streaming = True
    supports_context_management = False
    supports_persistent_retry = False
    supports_account_auth = False
    max_image_dim = 0
    max_image_bytes = 0
    pricing = Pricing()

    def estimate_text_token_count(self, text: str) -> int:
        return len(text) // 4

    def estimate_image_token_count(self, data: bytes) -> int:
        del data
        return 0

    async def buffer(self, request: ModelRequest) -> ModelResponse:
        del request
        raise AssertionError("Noop model should not be called")

    async def stream(
        self,
        request: ModelRequest,
        on_text: Callable[[str], None] | None = None,
        on_thinking: Callable[[str], None] | None = None,
    ) -> ModelResponse:
        del request, on_text, on_thinking
        raise AssertionError("Noop model should not be called")

    def is_context_overflow(self, error: Exception) -> bool:
        del error
        return False


class TestRenderObserver:
    def test_user_bar(self) -> None:
        printer = RecordingPrinter()
        obs = make_render_observer(printer)
        obs(UserBarEvent("hello"))
        assert printer.user_bars == ["hello"]

    def test_text_chunk_buffers_until_paragraph(self) -> None:
        printer = RecordingPrinter()
        obs = make_render_observer(printer)
        obs(TextChunkEvent("partial"))
        # Mid-stream chunk doesn't render until a stable boundary.
        assert printer.markdowns == []
        obs(TextChunkEvent(" rest\n\nmore"))
        # First paragraph flushed at the blank-line boundary.
        assert printer.markdowns

    def test_stream_end_flushes(self) -> None:
        printer = RecordingPrinter()
        obs = make_render_observer(printer)
        obs(TextChunkEvent("hello"))
        obs(StreamEndEvent())
        assert printer.markdowns == ["hello"]

    def test_thinking(self) -> None:
        printer = RecordingPrinter()
        obs = make_render_observer(printer)
        obs(ThinkingEvent("planning..."))
        assert printer.thinkings == ["planning..."]

    def test_tool_label(self) -> None:
        printer = RecordingPrinter()
        obs = make_render_observer(printer)
        obs(ToolLabelEvent("Bash ls"))
        assert printer.tool_labels == ["Bash ls"]

    def test_tool_result_renders_diff_and_error(self) -> None:
        printer = RecordingPrinter()
        obs = make_render_observer(printer)
        msg = MultipartMessage(
            (
                TextMessage("qid_1", "text/x-queue-id"),
                TextMessage("boom", "text/x-error"),
            ),
            "multipart/x-tool-result",
        )
        obs(ToolResultEvent(msg))
        assert printer.tool_errors == ["boom"]

    def test_error(self) -> None:
        printer = RecordingPrinter()
        obs = make_render_observer(printer)
        obs(ErrorEvent("nope"))
        assert printer.tool_errors == ["nope"]

    def test_interrupted(self) -> None:
        printer = RecordingPrinter()
        obs = make_render_observer(printer)
        obs(InterruptedEvent())
        assert printer.interruptions == 1

    def test_status_update_sets_terminal_title(self) -> None:
        printer = RecordingPrinter()
        obs = make_render_observer(printer)
        obs(StatusUpdateEvent("debugging"))
        assert printer.titles == ["debugging"]

    def test_turn_complete_is_silent(self) -> None:
        printer = RecordingPrinter()
        obs = make_render_observer(printer)
        obs(TurnCompleteEvent())
        assert printer.markdowns == []

    def test_child_event_atomic_renders_in_block(self) -> None:
        printer = RecordingPrinter()
        obs = make_render_observer(printer)
        obs(ChildEvent(label="Agent_0", inner=ToolLabelEvent("Bash ls")))
        # Atomic events flush immediately.
        assert len(printer.child_blocks) == 1
        label, items = printer.child_blocks[0]
        assert label == "Agent_0"
        assert items[0].descriptor == "text/x-tool-label"


class _RaisingPrinter(RecordingPrinter):
    """``RecordingPrinter`` whose ``write_thinking`` raises, to exercise self-report."""

    @override
    def write_thinking(self, text: str) -> None:
        del text
        raise RuntimeError("boom")


class TestRenderObserverSelfReport:
    def test_dispatch_failure_routed_to_tool_error(self) -> None:
        printer = _RaisingPrinter()
        obs = make_render_observer(printer)
        obs(ThinkingEvent("planning..."))
        assert len(printer.tool_errors) == 1
        msg = printer.tool_errors[0]
        assert "ThinkingEvent" in msg
        assert "RuntimeError" in msg
        assert "boom" in msg
        assert printer.thinkings == []

    def test_dispatch_failure_does_not_propagate(self) -> None:
        printer = _RaisingPrinter()
        obs = make_render_observer(printer)
        obs(ThinkingEvent("x"))  # must not raise


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
