"""Tests for sagent.compactor."""

from __future__ import annotations

import pytest

from sagent.compactor import (
    _MAX_FALLBACK_SUMMARY_CHARS,
    SummaryCompactor,
    _format_summary,
    _safe_split,
    build_continuation,
    microcompact,
)
from sagent.custom_exceptions import PromptTooLongError
from sagent.custom_types import (
    Message,
    ModelRequest,
    ModelResponse,
    MultipartMessage,
    TextMessage,
    TokenCount,
    Tool,
)
from sagent.lib.compaction import CLEARED
from sagent.lib.json import json_freeze
from sagent.lib.message import get_queue_id, tool_call_message
from sagent.testing import MockModelCaps


def _user(text: str) -> Message:
    return TextMessage(text, "text/x-user-message")


def _assistant(
    text: str = "", tool_calls: list[Message] | None = None, message_id: str = ""
) -> Message:
    id_parts: list[Message] = []
    if message_id:
        id_parts.append(TextMessage(message_id, "text/x-queue-id"))
    text_parts = (TextMessage(text, "text/plain"),) if text else ()
    tc_parts = tuple(tool_calls) if tool_calls else ()
    return MultipartMessage(
        tuple(id_parts) + text_parts + tc_parts,
        "multipart/x-model-message",
    )


def _tool_result(queue_id: str, text: str) -> Message:
    return MultipartMessage(
        (
            TextMessage(queue_id, "text/x-queue-id"),
            TextMessage(text, "text/plain"),
        ),
        "multipart/x-tool-result",
    )


def _mock_response(text: str) -> ModelResponse:
    return ModelResponse(
        content=MultipartMessage(
            (TextMessage(text, "text/plain"),),
            "multipart/x-model-message",
        ),
        tokens=TokenCount(input_tokens=500, output_tokens=100),
    )


class _MockCaps(MockModelCaps):
    """Capability flags + max_response_tokens + overflow hook for mocks."""

    supports_streaming: bool = False
    max_image_dim: int = 2000


class _MockModel(_MockCaps):
    """Model that returns canned text responses."""

    def __init__(
        self,
        text: str = "<analysis>thinking</analysis><summary>Summary here.</summary>",
    ) -> None:
        self._text = text
        self.requests: list[ModelRequest] = []

    @property
    def max_request_tokens(self) -> int:
        return 100_000

    @property
    def model_id(self) -> str:
        return "mock"

    async def buffer(
        self,
        request: ModelRequest,
    ) -> ModelResponse:
        self.requests.append(request)
        return _mock_response(self._text)

    async def stream(
        self,
        request: ModelRequest,
        on_text: object = None,
        on_thinking: object = None,
    ) -> ModelResponse:
        del on_text, on_thinking
        return await self.buffer(request=request)


class _FailThenSucceedModel(_MockCaps):
    """Fails first N calls with prompt-too-long, then succeeds."""

    def __init__(self, fail_count: int = 1) -> None:
        self._fail_count = fail_count
        self._call_count = 0
        self.requests: list[ModelRequest] = []

    @property
    def max_request_tokens(self) -> int:
        return 100_000

    @property
    def model_id(self) -> str:
        return "mock"

    async def buffer(
        self,
        request: ModelRequest,
    ) -> ModelResponse:
        self.requests.append(request)
        self._call_count += 1
        if self._call_count <= self._fail_count:
            raise PromptTooLongError("prompt_too_long")
        return _mock_response("<summary>Recovered.</summary>")

    async def stream(
        self,
        request: ModelRequest,
        on_text: object = None,
        on_thinking: object = None,
    ) -> ModelResponse:
        del on_text, on_thinking
        return await self.buffer(request=request)


# -- Format summary ----------------------------------------------------


class TestFormatSummary:
    def test_strips_analysis(self) -> None:
        raw = "<analysis>My thoughts.</analysis><summary>The summary.</summary>"
        result = _format_summary(raw)
        assert "My thoughts" not in result
        assert "The summary." in result

    def test_no_tags(self) -> None:
        raw = "Just plain text."
        result = _format_summary(raw)
        assert result == "Just plain text."

    def test_normalizes_newlines(self) -> None:
        raw = "<summary>A\n\n\n\nB</summary>"
        result = _format_summary(raw)
        assert "\n\n\n" not in result
        assert "A\n\nB" in result


# -- Should compact ----------------------------------------------------


class TestShouldCompact:
    @pytest.mark.anyio
    async def test_below_threshold(self) -> None:
        c = SummaryCompactor(buffer_tokens=10_000)
        assert not await c.should_compact(
            input_tokens=50_000,
            max_request_tokens=100_000,
        )

    @pytest.mark.anyio
    async def test_above_threshold(self) -> None:
        c = SummaryCompactor(buffer_tokens=10_000)
        assert await c.should_compact(
            input_tokens=95_000,
            max_request_tokens=100_000,
        )


# -- Compact -----------------------------------------------------------


class TestCompact:
    @pytest.mark.anyio
    async def test_returns_single_message(self) -> None:
        c = SummaryCompactor()
        model = _MockModel()
        messages = [_user("hello"), _user("do something")]
        result = await c.compact(messages=messages, model=model)
        assert len(result) == 1
        assert result[0].descriptor.endswith("/x-user-message")
        assert "Summary here." in str(result[0].content)

    @pytest.mark.anyio
    async def test_strips_analysis(self) -> None:
        c = SummaryCompactor()
        model = _MockModel()
        result = await c.compact(messages=[_user("hello")], model=model)
        assert "thinking" not in str(result[0].content)

    @pytest.mark.anyio
    async def test_continuation_suppression(self) -> None:
        c = SummaryCompactor()
        model = _MockModel()
        result = await c.compact(messages=[_user("hello")], model=model)
        assert result[0].descriptor.endswith("/x-user-message")
        assert "without preamble" in str(result[0].content).lower()

    @pytest.mark.anyio
    async def test_uses_custom_model(self) -> None:
        cheap = _MockModel()
        expensive = _MockModel()
        c = SummaryCompactor(model=cheap)
        await c.compact(messages=[_user("hello")], model=expensive)
        assert len(cheap.requests) == 1
        assert len(expensive.requests) == 0

    @pytest.mark.anyio
    async def test_no_tools_in_prompt(self) -> None:
        c = SummaryCompactor()
        model = _MockModel()
        await c.compact(messages=[_user("hello")], model=model)
        sent = model.requests[0]
        assert sent.tools is None
        last_msg = sent.messages[-1]
        assert last_msg.descriptor.endswith("/x-user-message")
        assert "not invoke any tools" in str(last_msg.content).lower()

    @pytest.mark.anyio
    async def test_prompt_too_long_retry(self) -> None:
        model = _FailThenSucceedModel(fail_count=1)
        c = SummaryCompactor()
        messages = [_user(f"msg{i}") for i in range(20)]
        result = await c.compact(messages=messages, model=model)
        assert len(result) == 1
        assert "Recovered" in str(result[0].content)
        # Second request should have fewer messages.
        assert len(model.requests[1].messages) < len(model.requests[0].messages)

    @pytest.mark.anyio
    async def test_prompt_too_long_retry_preserves_user_first_invariant(self) -> None:
        """After dropping oldest groups, the request must still start
        with a user message - Anthropic's API rejects anything else.
        Groups built from round-boundaries begin with an assistant on
        every group after the first, so a naive drop leaves the wire
        payload starting with assistant. Compactor must bridge that.
        """
        model = _FailThenSucceedModel(fail_count=1)
        c = SummaryCompactor()
        # Shape: [user, assistant, user, assistant, ...] - each assistant
        # gets its own queue_id so every assistant starts a new group.
        msgs: list[Message] = []
        for i in range(10):
            msgs.append(_user(f"u{i}"))
            msgs.append(_assistant(f"a{i}", message_id=f"m{i}"))
        await c.compact(messages=msgs, model=model)
        # Second attempt dropped a group; check its wire messages start
        # with a user message (or our synthetic bridge).
        retry_wire_msgs = model.requests[1].messages
        assert retry_wire_msgs[0].descriptor.endswith("/x-user-message")

    @pytest.mark.anyio
    async def test_prompt_too_long_all_retries_fail(self) -> None:
        model = _FailThenSucceedModel(fail_count=10)
        c = SummaryCompactor(max_attempts=2)
        result = await c.compact(messages=[_user("hello")], model=model)
        assert result[0].descriptor.endswith("/x-user-message")
        assert "failed" in str(result[0].content).lower()


class TestValidation:
    def test_max_attempts_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="max_attempts"):
            SummaryCompactor(max_attempts=0)


class TestKeepRecent:
    @pytest.mark.anyio
    async def test_keeps_recent_messages(self) -> None:
        c = SummaryCompactor(keep_recent=2)
        model = _MockModel()
        messages = [
            _user("old1"),
            _user("old2"),
            _user("recent1"),
            _user("recent2"),
        ]
        result = await c.compact(messages=messages, model=model)
        assert len(result) == 3
        assert result[1].content == "recent1"
        assert result[2].content == "recent2"


# -- Format summary: long text without tags (line 84) -----------------


class TestFormatSummaryLongFallback:
    def test_long_text_no_summary_tag_truncated(self) -> None:
        raw = "x" * (_MAX_FALLBACK_SUMMARY_CHARS + 500)
        result = _format_summary(raw)
        assert result.endswith("...(truncated)")
        assert len(result) <= _MAX_FALLBACK_SUMMARY_CHARS + 20


class TestSafeSplit:
    """``_safe_split`` never leaves a tool_use without its tool_result."""

    def test_snaps_back_when_split_lands_after_tool_use(self) -> None:
        messages = [
            _user("q"),
            _assistant(tool_calls=[tool_call_message("t1", "X", json_freeze({}))]),
            _tool_result("t1", "ok"),
            _user("next"),
        ]
        # Naive slice at keep=2 ⇒ to_summarize=[user, assistant(tool_use)]
        # which would orphan t1. Safe split snaps back to include the
        # assistant+result pair in to_keep.
        left, right = _safe_split(messages, 2, direction="from")
        _assert_no_orphan_tool_use(left)
        # No messages lost across the split.
        assert [*left, *right] == messages

    def test_resolved_pair_no_snap_needed(self) -> None:
        messages = [_user("a"), _user("b"), _user("c")]
        _, to_keep = _safe_split(messages, 1, direction="from")
        assert len(to_keep) == 1
        assert to_keep[0].content == "c"

    def test_up_to_direction(self) -> None:
        # Preserve prefix; summarize the rest. Must not leave an
        # unresolved tool_use at the tail of the prefix either.
        messages = [
            _user("setup"),
            _assistant(tool_calls=[tool_call_message("t1", "X", json_freeze({}))]),
            _tool_result("t1", "ok"),
            _user("next"),
        ]
        to_keep, _ = _safe_split(messages, 2, direction="up_to")
        _assert_no_orphan_tool_use(to_keep)

    def test_catches_earlier_unresolved_when_latest_resolved(self) -> None:
        """Regression: scanning only the latest assistant-with-tool_calls
        missed an earlier orphan. ``_prefix_has_unresolved_tool_use`` scans
        *every* assistant in the prefix, so a pathological session with an
        older orphan is still caught.
        """
        messages = [
            # Earlier assistant with an unresolved tool_use (t_old).
            _assistant(tool_calls=[tool_call_message("t_old", "X", json_freeze({}))]),
            _user("user speaks"),
            # Later assistant whose call *is* resolved in prefix.
            _assistant(tool_calls=[tool_call_message("t_new", "X", json_freeze({}))]),
            _tool_result("t_new", "ok"),
            # keep_recent boundary would normally leave just this tail:
            _user("tail"),
        ]
        left, right = _safe_split(messages, 1, direction="from")
        # No safe split exists: falls back to full compaction.
        assert left == messages
        assert right == []


class TestMicrocompact:
    def test_cleared_results_preserve_queue_id(self) -> None:
        class _FakeTool:
            supports_microcompaction = True

        tools: dict[str, Tool] = {"application/x-tool-read": _FakeTool()}  # pyright: ignore[reportAssignmentType]  # ty: ignore[invalid-assignment] -- test fake; only supports_microcompaction needed
        msgs: list[Message] = [
            _user("hi"),
            _assistant(
                "reading", tool_calls=[tool_call_message("t1", "read", json_freeze({}))]
            ),
            _tool_result("t1", "file contents here"),
            _assistant("done"),
        ]
        microcompact(
            msgs,
            tools,
            read_cache={},
            last_response_time=0.0,
            gap_sec=0,
            keep_recent=0,
        )
        cleared = msgs[2]
        assert get_queue_id(cleared) == "t1"
        assert isinstance(cleared, MultipartMessage)
        assert any(
            p.descriptor == "text/plain" and p.content == CLEARED
            for p in cleared.content
        )


def _assert_no_orphan_tool_use(slice_: list[Message]) -> None:
    """No model response in ``slice_`` has an unresolved tool_call."""
    for i, m in enumerate(slice_):
        if m.descriptor != "multipart/x-model-message" or not isinstance(
            m, MultipartMessage
        ):
            continue
        tool_parts = [p for p in m.content if p.descriptor == "multipart/x-tool-call"]
        if not tool_parts:
            continue
        resolved = {
            get_queue_id(r)
            for r in slice_[i + 1 :]
            if r.descriptor == "multipart/x-tool-result"
        }
        for tc in tool_parts:
            qid = get_queue_id(tc)
            assert qid in resolved, f"orphan tool_use {qid} at index {i}"


# -- Prompt-too-long token-gap retry ---------------------------------------


class TestPromptTooLongTokenGap:
    def test_structured_gap(self) -> None:
        err = PromptTooLongError(actual_tokens=123_456, limit_tokens=100_000)
        assert err.token_gap == 23_456

    def test_none_when_no_fields(self) -> None:
        err = PromptTooLongError("something went wrong")
        assert err.token_gap is None

    def test_none_when_zero_gap(self) -> None:
        err = PromptTooLongError(actual_tokens=100, limit_tokens=100)
        assert err.token_gap is None


# -- Build continuation with pointers --------------------------------------


class TestBuildContinuationPointers:
    def test_includes_pointers(self) -> None:
        result = build_continuation(
            "The summary.",
            summary_pointers=[
                ("/s/summary_0.md", "Initial setup"),
                ("/s/summary_1.md", "Refactor"),
            ],
        )
        assert "summary_0.md: Initial setup" in result
        assert "summary_1.md: Refactor" in result
        assert "The summary." in result

    def test_no_pointers_no_header(self) -> None:
        result = build_continuation("The summary.")
        assert "Prior context" not in result


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
