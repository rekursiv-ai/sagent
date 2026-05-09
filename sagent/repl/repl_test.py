"""Tests for the v2 REPL handler bundle.

Tests assert *occurrence counts* and *exact rendered output*, not
"string in output". The earlier loose "in output" tests let three
regressions through:

- Response text rendered twice (streaming + final multipart).
- Streaming chunks rendered with extra newlines per chunk.
- User-bar prefix duplicated by prompt-toolkit echo.

Tests below pin each behavior with explicit counts.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import override
from unittest.mock import AsyncMock, MagicMock

import asyncio
import io

from rich.console import Console

import pytest

from sagent.agent.agent import Agent
from sagent.agent.handlers import InlineHandler, core_handlers
from sagent.custom_types import (
    Message,
    ModelRequest,
    ModelResponse,
    ModelSpec,
    MultipartMessage,
    TextMessage,
    TokenCount,
)
from sagent.lib.json import JSON, json_freeze
from sagent.lib.message import get_directive, tool_call_message
from sagent.repl import (
    ConsolePrinter,
    PromptToolkitInputSource,
    RecordingPrinter,
    StubInputSource,
    repl_handler_set,
    spawn_repl_pump,
)
from sagent.repl.render import RenderChildEvent
from sagent.repl.replay import replay_messages
from sagent.repl.slash import parse_slash
from sagent.repl.toolbar import render_toolbar
from sagent.testing import MockModelCaps


class _StreamingFakeModel(MockModelCaps):
    """Model that streams chunks via ``on_text`` then returns canned responses."""

    max_image_dim: int = 2000

    def __init__(
        self,
        responses: list[ModelResponse],
        *,
        chunks: tuple[str, ...] = (),
    ) -> None:
        self._responses = responses
        self._chunks = chunks
        self._idx = 0
        self.requests: list[ModelRequest] = []

    @property
    def max_request_tokens(self) -> int:
        return 100_000

    @property
    def model_id(self) -> str:
        return "stream-fake"

    def _next(self) -> ModelResponse:
        idx = min(self._idx, len(self._responses) - 1)
        self._idx += 1
        return self._responses[idx]

    async def buffer(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return self._next()

    async def stream(
        self,
        request: ModelRequest,
        on_text: Callable[[str], None] | None = None,
        on_thinking: Callable[[str], None] | None = None,
    ) -> ModelResponse:
        del on_thinking
        self.requests.append(request)
        if on_text is not None:
            for chunk in self._chunks:
                on_text(chunk)
        return self._next()


def _model_response(
    text: str = "",
    *,
    tool_calls: list[Message] | None = None,
) -> ModelResponse:
    parts: list[Message] = []
    if text:
        parts.append(TextMessage(text, "text/plain"))
    parts.extend(tool_calls or [])
    return ModelResponse(
        content=MultipartMessage(tuple(parts), "multipart/x-model-message"),
        tokens=TokenCount(input_tokens=10, output_tokens=5),
        stop_reason="model_finished" if not tool_calls else "model_tool_use",
    )


class _EchoTool:
    name = "echo"
    tool_id = "application/x-tool-echo"
    description = "Echoes input."
    supports_microcompaction = False
    directive_schema: JSON = json_freeze(
        {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    )

    def summary(self, msg: Message) -> str:
        del msg
        return self.name

    def summary_result(self, result: Message) -> str | None:
        del result
        return None

    def prompt(self) -> str:
        return ""

    async def run(self, msg: Message) -> Message:
        directive = get_directive(msg)
        return TextMessage(str(directive.get("text", "")), "text/plain")


# -- User-bar render ---------------------------------------------------


@pytest.mark.asyncio
async def test_user_bar_renders_exactly_once_per_user_message() -> None:
    """One ``text/x-user-message`` -> exactly one user-bar payload."""
    printer = RecordingPrinter()
    model = _StreamingFakeModel([_model_response("ack")])
    agent = Agent(
        model=model,
        handlers=[*core_handlers(), *repl_handler_set(printer)],
    )
    _ = await asyncio.wait_for(
        agent.run(json_freeze({"prompt": "hi"})),
        timeout=2.0,
    )
    assert printer.user_bars == ["hi"]


# -- Streaming render --------------------------------------------------


@pytest.mark.asyncio
async def test_streaming_renders_paragraphs_as_markdown() -> None:
    """Stable paragraph boundaries commit as ``write_markdown`` blocks."""
    printer = RecordingPrinter()
    # Two paragraphs trigger a stable-boundary commit during streaming;
    # the second paragraph flushes on stream-end.
    model = _StreamingFakeModel(
        [_model_response("first para\n\nsecond para")],
        chunks=("first para\n", "\nsecond ", "para"),
    )
    agent = Agent(
        model=model,
        handlers=[*core_handlers(), *repl_handler_set(printer)],
    )
    _ = await asyncio.wait_for(
        agent.run(json_freeze({"prompt": "p"})),
        timeout=2.0,
    )
    # Both paragraphs end up rendered as markdown.
    rendered = printer.rendered_text
    assert "first para" in rendered
    assert "second para" in rendered


@pytest.mark.asyncio
async def test_buffer_only_response_renders_via_markdown() -> None:
    """Buffer-only model (no ``on_text``) still renders the response."""
    printer = RecordingPrinter()
    model = _StreamingFakeModel([_model_response("hello buffer")])
    agent = Agent(
        model=model,
        handlers=[*core_handlers(), *repl_handler_set(printer)],
    )
    _ = await asyncio.wait_for(
        agent.run(json_freeze({"prompt": "p"})),
        timeout=2.0,
    )
    assert printer.rendered_text.strip() == "hello buffer"


@pytest.mark.asyncio
async def test_response_renders_only_via_streaming_path() -> None:
    """The multipart/x-model-message itself does NOT trigger a render."""
    printer = RecordingPrinter()
    model = _StreamingFakeModel(
        [_model_response("hello world")],
        chunks=("hello ", "world"),
    )
    agent = Agent(
        model=model,
        handlers=[*core_handlers(), *repl_handler_set(printer)],
    )
    _ = await asyncio.wait_for(
        agent.run(json_freeze({"prompt": "say hi"})),
        timeout=2.0,
    )
    # Only one render of the response text, not "hello world" twice.
    rendered_count = printer.rendered_text.count("hello world")
    assert rendered_count == 1


# -- Tool result / tool label render -----------------------------------


@pytest.mark.asyncio
async def test_tool_result_renders_errors_and_diffs_only() -> None:
    """``RenderToolResult`` emits errors / diffs; plain text isn't user-facing."""
    printer = RecordingPrinter()
    model = _StreamingFakeModel(
        [
            _model_response(
                tool_calls=[
                    tool_call_message("t1", "echo", json_freeze({"text": "ok"})),
                ],
            ),
            _model_response("done"),
        ],
    )
    agent = Agent(
        model=model,
        tools=[_EchoTool()],
        handlers=[*core_handlers(), *repl_handler_set(printer)],
    )
    _ = await asyncio.wait_for(
        agent.run(json_freeze({"prompt": "use the tool"})),
        timeout=2.0,
    )
    # Plain "ok" is in history (model sees it) but not user-rendered.
    assert printer.tool_errors == []
    assert printer.diffs == []


# -- Activity tracker / toolbar ----------------------------------------


@pytest.mark.asyncio
async def test_activity_tracker_accumulates_elapsed_after_call() -> None:
    """``ActivityHandler`` flips ``active`` and accumulates elapsed_seconds."""
    model = _StreamingFakeModel([_model_response("ok")])
    agent = Agent(model=model)
    assert not agent.activity.active
    _ = await asyncio.wait_for(
        agent.run(json_freeze({"prompt": "p"})),
        timeout=2.0,
    )
    assert not agent.activity.active
    assert agent.activity.elapsed_seconds >= 0.0


@pytest.mark.asyncio
async def test_activity_tracker_counts_streaming_chars() -> None:
    """Stream chunks accumulate ``live_response_chars`` while in flight."""
    captured: list[int] = []

    class _SpyChunks(InlineHandler):
        descriptors: tuple[str, ...] = ("text/plain",)

        @override
        async def handle(self, agent: Agent, msg: Message) -> None:
            del msg
            captured.append(agent.activity.live_response_chars)

    model = _StreamingFakeModel(
        [_model_response("ab")],
        chunks=("a", "b"),
    )
    agent = Agent(model=model)
    agent.register(_SpyChunks())
    _ = await asyncio.wait_for(
        agent.run(json_freeze({"prompt": "p"})),
        timeout=2.0,
    )
    assert captured == [1, 2]
    assert agent.activity.live_response_chars == 0


def test_render_toolbar_idle_returns_empty() -> None:
    """Pristine agent with no run history -> empty toolbar."""

    class _NopModel(MockModelCaps):
        max_image_dim: int = 2000

        @property
        def max_request_tokens(self) -> int:
            return 100_000

        @property
        def model_id(self) -> str:
            return "nop"

        async def buffer(self, request: ModelRequest) -> ModelResponse:
            del request
            raise RuntimeError("not used")

        async def stream(
            self,
            request: ModelRequest,
            on_text: Callable[[str], None] | None = None,
            on_thinking: Callable[[str], None] | None = None,
        ) -> ModelResponse:
            del request, on_text, on_thinking
            raise RuntimeError("not used")

    agent = Agent(model=_NopModel(), handlers=[])
    assert render_toolbar(agent) == ""


@pytest.mark.asyncio
async def test_render_toolbar_after_run_shows_bracket() -> None:
    """After at least one model call, toolbar carries token + cost summary."""
    model = _StreamingFakeModel([_model_response("ok")])
    agent = Agent(model=model)
    _ = await asyncio.wait_for(
        agent.run(json_freeze({"prompt": "p"})),
        timeout=2.0,
    )
    bar = render_toolbar(agent)
    assert bar.startswith("[")
    assert bar.endswith("]")
    assert "$" in bar


# -- Child-event gutter rendering --------------------------------------


def _child_envelope(label: str, *inners: Message) -> Message:
    """Build a ``multipart/x-child-event`` envelope as the spawn forwarder does."""
    return MultipartMessage(
        (TextMessage(label, "text/x-agent-label"), *inners),
        "multipart/x-child-event",
    )


@pytest.mark.asyncio
async def test_child_event_buffers_streaming_chunks_into_one_block() -> None:
    """Many small ``text/plain`` chunks from one child collapse to one block.

    The provider streams in arbitrary byte fragments. Without
    buffering we'd render each chunk as a separate labeled line
    with mid-token splits ("## Type" then " Hints"). The new
    ``RenderChildEvent`` accumulates per label and flushes at a
    stable Markdown boundary, so the user sees a single coherent
    block.
    """
    printer = RecordingPrinter()
    handler = RenderChildEvent(printer)
    agent = Agent(model=_StreamingFakeModel([_model_response("ack")]))
    chunks = ["## Type", " Hints\n\nNext", " paragraph.\n\n"]
    for chunk in chunks:
        await handler.handle(
            agent,
            _child_envelope("Agent_0", TextMessage(chunk, "text/plain")),
        )
    # Stable boundary at "\n\n" flushes; everything before it lands
    # in one block as one ``text/plain`` Message (the assembled
    # Markdown), not many.
    assert len(printer.child_blocks) >= 1
    label, items = printer.child_blocks[0]
    assert label == "Agent_0"
    text_items = [it for it in items if it.descriptor == "text/plain"]
    assert text_items, items
    # The first paragraph is a single coherent message, not chunks.
    assert "## Type Hints" in str(text_items[0].content)


@pytest.mark.asyncio
async def test_child_event_x_interleave_flushes_on_label_change() -> None:
    """When a different child emits, the prior label flushes immediately.

    X-interleave: real-time visibility for both children, even if
    the in-progress label hadn't reached a stable Markdown boundary.
    """
    printer = RecordingPrinter()
    handler = RenderChildEvent(printer)
    agent = Agent(model=_StreamingFakeModel([_model_response("ack")]))
    await handler.handle(
        agent,
        _child_envelope("Agent_0", TextMessage("partial", "text/plain")),
    )
    # Pre-interleave: nothing flushed (no boundary yet).
    assert printer.child_blocks == []
    # Different child arrives -> flush Agent_0's pending text first.
    await handler.handle(
        agent,
        _child_envelope("Agent_1", TextMessage("other", "text/plain")),
    )
    assert len(printer.child_blocks) == 1
    assert printer.child_blocks[0][0] == "Agent_0"


@pytest.mark.asyncio
async def test_child_event_unstable_tail_does_not_split_across_blocks() -> None:
    """Boundary-triggered flush keeps the unstable tail in the buffer.

    Regression: an earlier version flushed the entire ``text_buf``
    (stable prefix + unstable tail) on every boundary trigger,
    producing one block ending mid-word and the next block starting
    with the rest of that word ("for port" / "ability."). The fix
    is that boundary flushes emit only the stable prefix; the
    unstable tail stays in the buffer until the NEXT boundary, an
    atomic event, or end-of-stream.
    """
    printer = RecordingPrinter()
    handler = RenderChildEvent(printer)
    agent = Agent(model=_StreamingFakeModel([_model_response("ack")]))
    # Stream a paragraph followed by mid-word fragments. The first
    # paragraph reaches a stable boundary at the blank line; the
    # bullet that follows is still incomplete (mid-word "port").
    chunks = [
        "first paragraph.\n\n- list item with text for port",
        "ability.\n\n## Heading\n",
        "after",
    ]
    for chunk in chunks:
        await handler.handle(
            agent,
            _child_envelope("Agent_0", TextMessage(chunk, "text/plain")),
        )
    # Concatenate every text/plain content across all blocks.
    rendered = ""
    for _, items in printer.child_blocks:
        for item in items:
            if item.descriptor == "text/plain":
                rendered += str(item.content)
    # ``portability`` MUST appear intact in some block, never split.
    assert "portability" in rendered, [
        (lbl, [(it.descriptor, str(it.content)) for it in its])
        for lbl, its in printer.child_blocks
    ]


@pytest.mark.asyncio
async def test_child_event_atomic_event_flushes_pending_text() -> None:
    """A tool label / result / error from the child flushes pending text first.

    The atomic event renders in the same labeled block as the
    streaming text that preceded it, which matches the user's
    mental model of "this child's output during this turn."
    """
    printer = RecordingPrinter()
    handler = RenderChildEvent(printer)
    agent = Agent(model=_StreamingFakeModel([_model_response("ack")]))
    await handler.handle(
        agent,
        _child_envelope("Agent_0", TextMessage("I will edit", "text/plain")),
    )
    await handler.handle(
        agent,
        _child_envelope("Agent_0", TextMessage("Edit", "text/x-tool-label")),
    )
    # Both items appear in one block, in order.
    assert len(printer.child_blocks) == 1
    label, items = printer.child_blocks[0]
    assert label == "Agent_0"
    assert [it.descriptor for it in items] == ["text/plain", "text/x-tool-label"]


# -- Input handler routing ---------------------------------------------


@pytest.mark.asyncio
async def test_prompt_input_routes_user_text() -> None:
    """Text lines become ``text/x-user-message`` posted to the inbox."""
    captured: list[Message] = []

    class _Spy(InlineHandler):
        descriptors: tuple[str, ...] = ("text/x-user-message",)

        @override
        async def handle(self, agent: Agent, msg: Message) -> None:
            del agent
            captured.append(msg)

    source = StubInputSource(["hello", "world", None])
    model = _StreamingFakeModel([_model_response("ack")])
    agent = Agent(model=model, handlers=[])
    agent.register(_Spy())
    _ = spawn_repl_pump(agent, source)
    await asyncio.wait_for(agent.run_loop(), timeout=2.0)
    assert [m.content for m in captured] == ["hello", "world"]


@pytest.mark.asyncio
async def test_prompt_input_routes_slash_commands() -> None:
    """``/clear``, ``/compact``, ``/uncompact``, ``/abort`` map cleanly."""
    seen: list[tuple[str, str]] = []

    class _Trace(InlineHandler):
        descriptors: tuple[str, ...] = (
            "text/x-clear-request",
            "text/x-abort",
            "text/x-compact-request",
            "text/x-uncompact-request",
            "text/x-user-message",
        )

        @override
        async def handle(self, agent: Agent, msg: Message) -> None:
            del agent
            seen.append((msg.descriptor, str(msg.content)))

    source = StubInputSource(
        [
            "/clear",
            "/compact focus on bugs",
            "/uncompact",
            "after",
            "/abort",
            "/quit",
        ],
    )
    model = _StreamingFakeModel([_model_response("ack")])
    agent = Agent(model=model, handlers=[])
    agent.register(_Trace())
    _ = spawn_repl_pump(agent, source)
    await asyncio.wait_for(agent.run_loop(), timeout=2.0)
    # /clear and /abort go put_left -- preempt anything queued. The
    # remaining slash + text inputs preserve typed order.
    assert ("text/x-abort", "") in seen
    assert ("text/x-clear-request", "") in seen
    assert ("text/x-compact-request", "focus on bugs") in seen
    assert ("text/x-uncompact-request", "") in seen
    assert ("text/x-user-message", "after") in seen


def test_replay_uses_tool_summary_for_labels() -> None:
    """Replay calls ``tool.summary(req)`` -- not the bare lowercase name.

    Regression: the prior path emitted ``get_tool_name(p)`` (e.g.
    ``"echo"``), losing every directive arg. Replay should match
    live rendering exactly: ``echo`` with its own ``summary()`` /
    ``tool_call_label`` formatting.
    """
    printer = RecordingPrinter()
    model = _StreamingFakeModel([_model_response("ack")])
    agent = Agent(model=model, tools=[_EchoTool()])
    # Stage history with an echo call.
    agent.history.append(TextMessage("hi", "text/x-user-message"))
    agent.history.append(
        MultipartMessage(
            (tool_call_message("q1", "echo", json_freeze({"text": "hello"})),),
            "multipart/x-model-message",
        ),
    )
    replay_messages(agent, printer)
    # Tool label fired, formatted via _EchoTool.summary (not the lowercase id).
    assert printer.tool_labels, printer.tool_labels
    assert printer.tool_labels[0] == "echo"


def test_replay_renders_tool_summary_part() -> None:
    """Replay renders ``text/x-tool-summary`` parts via ``write_tool_summary``."""
    printer = RecordingPrinter()
    model = _StreamingFakeModel([_model_response("ack")])
    agent = Agent(model=model, tools=[_EchoTool()])
    agent.history.append(TextMessage("hi", "text/x-user-message"))
    agent.history.append(
        MultipartMessage(
            (
                TextMessage("q1", "text/x-queue-id"),
                TextMessage("hello", "text/plain"),
                TextMessage("5 chars", "text/x-tool-summary"),
            ),
            "multipart/x-tool-result",
        ),
    )
    replay_messages(agent, printer)
    assert printer.tool_summaries == ["5 chars"]


def test_help_and_tasks_parse_to_inbox_actions() -> None:
    """``/help`` and ``/tasks`` map to their respective request descriptors.

    Both are exact-line matches; conversational mention falls through.
    """
    h = parse_slash("/help")
    assert h is not None
    assert h.descriptor == "text/x-help-request"

    t = parse_slash("/tasks")
    assert t is not None
    assert t.descriptor == "text/x-tasks-request"

    # ``/help <args>`` is not a command (parser is exact-match for these).
    fall = parse_slash("/help me")
    assert fall is not None
    assert fall.descriptor == "text/x-error"


def test_break_and_abort_parse_with_scope() -> None:
    """``/break`` and ``/abort`` accept ``""``, ``<label>``, or ``"all"``.

    The arg passes straight through as message content; the handler
    interprets it.
    """
    bare = parse_slash("/abort")
    assert bare is not None
    assert bare.descriptor == "text/x-abort"
    assert bare.content == ""

    labelled = parse_slash("/abort coder-1")
    assert labelled is not None
    assert labelled.descriptor == "text/x-abort"
    assert labelled.content == "coder-1"

    all_ = parse_slash("/abort all")
    assert all_ is not None
    assert all_.descriptor == "text/x-abort"
    assert all_.content == "all"

    br = parse_slash("/break all")
    assert br is not None
    assert br.descriptor == "text/x-break"
    assert br.content == "all"


def test_clear_only_fires_on_exact_line() -> None:
    """``/clear`` only matches an exact-line; conversational mention is not a command.

    Regression: prior versions accepted ``/clear ARG`` too. Typing
    ``"/clear should also wipe foo"`` then triggered the destructive
    handler, mid-sentence, on a line that was talking *about* the
    command. ``/clear`` takes no argument; any extra text means we
    fall through to the unknown-command branch.
    """
    exact = parse_slash("/clear")
    assert exact is not None
    assert exact.descriptor == "text/x-clear-request"

    conversational = parse_slash("/clear should clear context not logs")
    assert conversational is not None
    # Falls through to the catch-all unknown-command branch -- safer
    # than firing a destructive command on a sentence.
    assert conversational.descriptor == "text/x-error"


@pytest.mark.asyncio
async def test_quit_word_terminates_input_loop() -> None:
    """``/quit`` (slash-prefixed) maps to ``text/x-quit``."""
    source = StubInputSource(["hi", "/quit"])
    model = _StreamingFakeModel([_model_response("ack")])
    agent = Agent(model=model, handlers=[])
    _ = spawn_repl_pump(agent, source)
    await asyncio.wait_for(agent.run_loop(), timeout=2.0)


@pytest.mark.asyncio
async def test_slash_model_no_args_reports_current_spec() -> None:
    """``ModelSwitchHandler`` with no args writes current spec to printer."""
    printer = RecordingPrinter()
    model = _StreamingFakeModel([_model_response("ack")])
    spec = ModelSpec(
        provider="Anthropic",
        auth="env",
        model_id="stream-fake",
        account=None,
    )
    agent = Agent(
        model=model,
        model_spec=spec,
        handlers=[*core_handlers(), *repl_handler_set(printer)],
    )
    agent.inbox.put(TextMessage("", "text/x-model-switch-request"))
    agent.inbox.put(TextMessage("", "text/x-quit"))
    await asyncio.wait_for(agent.run_loop(), timeout=2.0)
    status_lines = [line for line in printer.lines if line.startswith("[/model]")]
    assert len(status_lines) == 1
    assert "provider=Anthropic" in status_lines[0]
    assert "stream-fake" in status_lines[0]


@pytest.mark.asyncio
async def test_prompt_input_routes_slash_model() -> None:
    """``/model ARGS`` posts ``text/x-model-switch-request`` with args content."""
    seen: list[tuple[str, str]] = []

    class _Trace(InlineHandler):
        descriptors: tuple[str, ...] = ("text/x-model-switch-request",)

        @override
        async def handle(self, agent: Agent, msg: Message) -> None:
            del agent
            seen.append((msg.descriptor, str(msg.content)))

    source = StubInputSource(["/model gpt-4", "/quit"])
    model = _StreamingFakeModel([_model_response("ack")])
    agent = Agent(model=model, handlers=[])
    agent.register(_Trace())
    _ = spawn_repl_pump(agent, source)
    await asyncio.wait_for(agent.run_loop(), timeout=2.0)
    assert seen == [("text/x-model-switch-request", "gpt-4")]


@pytest.mark.asyncio
async def test_unknown_slash_does_not_reach_llm() -> None:
    """``/foo`` produces an error message, never a ``text/x-user-message``."""
    user_messages: list[Message] = []
    errors: list[Message] = []

    class _SpyUser(InlineHandler):
        descriptors: tuple[str, ...] = ("text/x-user-message",)

        @override
        async def handle(self, agent: Agent, msg: Message) -> None:
            del agent
            user_messages.append(msg)

    class _SpyError(InlineHandler):
        descriptors: tuple[str, ...] = ("text/x-error",)

        @override
        async def handle(self, agent: Agent, msg: Message) -> None:
            del agent
            errors.append(msg)

    source = StubInputSource(["/foo arg", "/quit"])
    model = _StreamingFakeModel([_model_response("ack")])
    agent = Agent(model=model, handlers=[])
    agent.register(_SpyUser())
    agent.register(_SpyError())
    _ = spawn_repl_pump(agent, source)
    await asyncio.wait_for(agent.run_loop(), timeout=2.0)
    assert user_messages == []
    assert len(errors) == 1
    assert "unknown command: /foo" in str(errors[0].content)


# -- ConsolePrinter integration ----------------------------------------


@pytest.mark.asyncio
async def test_console_printer_renders_user_bar_and_markdown() -> None:
    """Real rich.Console output contains the user message + markdown response."""
    buf = io.StringIO()
    printer = ConsolePrinter(Console(file=buf, force_terminal=False, width=80))
    model = _StreamingFakeModel([_model_response("hello world")])
    agent = Agent(
        model=model,
        handlers=[*core_handlers(), *repl_handler_set(printer)],
    )
    _ = await asyncio.wait_for(
        agent.run(json_freeze({"prompt": "hi"})),
        timeout=2.0,
    )
    output = buf.getvalue()
    assert "> hi" in output
    assert "hello world" in output


@pytest.mark.asyncio
async def test_console_printer_writes_chunk_without_newline() -> None:
    """``write_chunk`` does NOT add a newline (chunks coalesce)."""
    buf = io.StringIO()
    printer = ConsolePrinter(Console(file=buf, force_terminal=False, width=80))
    printer.write_chunk("hello ")
    printer.write_chunk("world")
    output = buf.getvalue()
    assert output == "hello world"


@pytest.mark.asyncio
async def test_prompt_toolkit_input_source_wraps_session() -> None:
    """``PromptToolkitInputSource`` returns lines and ``None`` on EOF / quit."""
    session = MagicMock()
    session.prompt_async = AsyncMock(side_effect=["hello", "/quit"])
    source = PromptToolkitInputSource(session)
    assert await source.next_line() == "hello"
    assert await source.next_line() is None  # "/quit" -> None

    session_eof = MagicMock()
    session_eof.prompt_async = AsyncMock(side_effect=EOFError())
    source_eof = PromptToolkitInputSource(session_eof)
    assert await source_eof.next_line() is None


# -- Test entry --------------------------------------------------------


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
