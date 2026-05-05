"""Tests for repl.repl."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from io import StringIO
from typing import Any
from unittest.mock import MagicMock, patch

import asyncio

from rich.console import Console

import pytest

from sagent.agent import QUIT_SENTINEL
from sagent.custom_types import (
    JsonMessage,
    Message,
    ModelSpec,
    MultipartMessage,
    TextMessage,
)
from sagent.lib.asyncio_collections import Deque
from sagent.lib.json import json_freeze
from sagent.repl.custom_types import RenderState
from sagent.repl.handlers import render_batch
from sagent.repl.repl import (
    _flush_render_frame,
    run_repl,
)
from sagent.repl.slash_commands import (
    handle_slash_clear,
    handle_slash_model,
)


def _make_agent(
    event_batches: list[list[Any]],
    raise_exc: BaseException | None = None,
) -> Any:
    call_idx = 0
    wrapper: Any = MagicMock()
    wrapper.tool_state = MagicMock()
    wrapper.inbox = Deque[str]()
    wrapper.active = False
    wrapper.inflight = None
    wrapper.title = ""
    wrapper.messages = []
    wrapper.total_cost_usd = 0.0

    async def _run_continuous() -> AsyncGenerator[Message | None, None]:
        nonlocal call_idx
        while True:
            prompt = await wrapper.inbox.get()
            if prompt == QUIT_SENTINEL:
                return
            if call_idx < len(event_batches):
                for ev in event_batches[call_idx]:
                    yield ev
                yield JsonMessage(
                    json_freeze({"input_tokens": 10, "output_tokens": 5}),
                    "application/x-done",
                )
                yield None
            call_idx += 1
            if raise_exc is not None:
                raise raise_exc

    wrapper.run_continuous = _run_continuous
    return wrapper


def _prompt_sequence(*inputs: str | type[BaseException]) -> Any:
    """prompt_async mock: yields inputs in order, then blocks forever."""
    items: tuple[str | type[BaseException], ...] = inputs
    idx = 0
    hang = asyncio.Event()

    async def _prompt(_prompt_text: str = "") -> str:
        nonlocal idx
        await asyncio.sleep(0)
        if idx < len(items):
            val = items[idx]
            idx += 1
            if isinstance(val, type):
                raise val()
            return val
        await hang.wait()
        return ""

    return _prompt


def _simple_agent() -> Any:
    a: Any = MagicMock()
    a.inbox = Deque[str]()
    a.active = False
    a.inflight = None
    a.title = ""
    a.messages = []
    a.tool_state = MagicMock()
    a.total_cost_usd = 0.0

    async def _run_continuous() -> AsyncGenerator[Message | None, None]:
        while True:
            prompt = await a.inbox.get()
            if prompt == QUIT_SENTINEL:
                return
            yield None

    a.run_continuous = _run_continuous
    return a


def _repl_ctx() -> Any:
    return (
        patch("sagent.repl.repl.PromptSession"),
        patch("sagent.repl.repl.Console"),
    )


def _render(events: list[Message]) -> tuple[RenderState, str, str]:
    """Run render_batch and return (render_frame, stdout_text, stderr_text)."""
    out_io = StringIO()
    err_io = StringIO()
    out = Console(file=out_io, width=80, force_terminal=False)
    console = Console(file=err_io, width=80, force_terminal=False)
    render_frame = RenderState(console=console, out=out)
    render_batch(events, render_frame)
    return render_frame, out_io.getvalue(), err_io.getvalue()


class TestRendering:
    """Exercise render_batch event handling."""

    def test_text_event_accumulated(self) -> None:
        render_frame, _, _ = _render([TextMessage("Hello\n\n", "text/plain")])
        assert "Hello" in render_frame.buf

    def test_thinking_event_rendered(self) -> None:
        _, out, _ = _render([TextMessage("Let me think...", "text/x-thinking")])
        assert "Thinking" in out

    def test_tool_call_rendered(self) -> None:
        # Tool-call display events carry a string label (from tool.summary()).
        tc = TextMessage("bash", "text/x-tool-label")
        _, _, err = _render([tc])
        assert "bash" in err

    def test_error_tool_result_rendered(self) -> None:
        tr = MultipartMessage(
            (TextMessage("command failed", "text/x-error"),),
            "multipart/x-tool-result",
        )
        _, _, err = _render([tr])
        assert "✗" in err

    def test_bash_lint_hint_surfaces_on_error(self) -> None:
        tr = MultipartMessage(
            (
                TextMessage("grep: no such file\n", "text/x-error"),
                TextMessage(
                    "grep via Bash is a bad UX. Use the Grep tool.",
                    "text/x-hint-tool-use-nudge",
                ),
            ),
            "multipart/x-tool-result",
        )
        _, _, err = _render([tr])
        assert "✗" in err
        assert "hint:" in err

    def test_bash_lint_hint_surfaces(self) -> None:
        tr = MultipartMessage(
            (
                TextMessage("stdout contents\n", "text/plain"),
                TextMessage(
                    "grep via Bash is a bad UX. Use the Grep tool.",
                    "text/x-hint-tool-use-nudge",
                ),
            ),
            "multipart/x-tool-result",
        )
        _, _, err = _render([tr])
        assert "hint:" in err
        assert "grep via Bash is a bad UX" in err

    def test_no_phantom_hint_from_literal_in_content(self) -> None:
        """Regression: literal ``[bash-lint]`` in content must not trigger hint."""
        tr = MultipartMessage(
            (
                TextMessage(
                    (
                        "so its ``[bash-lint]`` feature can suggest"
                        " dedicated replacements\n"
                        "<system-reminder>\n[bash-lint] phony\n"
                        "</system-reminder>\n"
                    ),
                    "text/plain",
                ),
            ),
            "multipart/x-tool-result",
        )
        _, _, err = _render([tr])
        assert "hint:" not in err
        assert "phony" not in err

    def test_text_tool_text_preserves_paragraph_break(self) -> None:
        """Text before a tool call must not concatenate with text after it."""
        events: list[Message] = [
            TextMessage("Did you mean a different file?", "text/plain"),
            TextMessage("Read", "text/x-tool-label"),
            TextMessage("Read repl.py.", "text/plain"),
        ]
        _, out, err = _render(events)
        rendered = out + err
        assert "?Read" not in rendered
        assert "Did you mean a different file?" in rendered


class TestChildEventRendering:
    """Exercise _handle_child_event via render_batch."""

    def test_child_tool_label_renders_with_prefix(self) -> None:
        event = MultipartMessage(
            (
                TextMessage("Agent_0", "text/plain"),
                TextMessage("Bash ls", "text/x-tool-label"),
            ),
            "multipart/x-child-event",
        )
        _, _, err = _render([event])
        assert "[Agent_0]" in err
        assert "Bash ls" in err

    def test_child_error_renders_red_with_prefix(self) -> None:
        event = MultipartMessage(
            (
                TextMessage("Agent_0", "text/plain"),
                MultipartMessage(
                    (TextMessage("something broke", "text/x-error"),),
                    "multipart/x-tool-result",
                ),
            ),
            "multipart/x-child-event",
        )
        _, _, err = _render([event])
        assert "[Agent_0]" in err
        assert "something broke" in err

    def test_child_done_renders_summary(self) -> None:
        event = MultipartMessage(
            (
                TextMessage("Agent_0", "text/plain"),
                JsonMessage(
                    json_freeze(
                        {"elapsed": 5.0, "model_response_tokens": 100, "cost_usd": 0.05}
                    ),
                    "application/x-child-done",
                ),
            ),
            "multipart/x-child-event",
        )
        _, _, err = _render([event])
        assert "[Agent_0]" in err
        assert "done" in err
        assert "5s" in err

    def test_child_text_deltas_buffer_until_line_boundary(self) -> None:
        events: list[Message] = [
            MultipartMessage(
                (
                    TextMessage("Agent_0", "text/plain"),
                    TextMessage("Depth ", "text/plain"),
                ),
                "multipart/x-child-event",
            ),
            MultipartMessage(
                (
                    TextMessage("Agent_0", "text/plain"),
                    TextMessage("is set\n", "text/plain"),
                ),
                "multipart/x-child-event",
            ),
        ]
        _, _, err = _render(events)
        assert "[Agent_0] Depth is set" in err
        assert "[Agent_0] Depth\n" not in err
        assert "[Agent_0] is set" not in err

    def test_parent_text_flushes_before_nested_child_event(self) -> None:
        nested = MultipartMessage(
            (
                TextMessage("Agent_0_0", "text/plain"),
                TextMessage("Bash ls", "text/x-tool-label"),
            ),
            "multipart/x-child-event",
        )
        events: list[Message] = [
            MultipartMessage(
                (
                    TextMessage("Agent_0", "text/plain"),
                    TextMessage("before nested", "text/plain"),
                ),
                "multipart/x-child-event",
            ),
            MultipartMessage(
                (
                    TextMessage("Agent_0", "text/plain"),
                    nested,
                ),
                "multipart/x-child-event",
            ),
        ]
        _, _, err = _render(events)
        assert err.index("[Agent_0] before nested") < err.index("[Agent_0_0] Bash ls")


class TestRunRepl:
    @pytest.mark.anyio
    async def test_quit_command(self) -> None:
        agent = _simple_agent()
        p1, p2 = _repl_ctx()
        with p1 as mock_cls, p2:
            mock_cls.return_value.prompt_async = _prompt_sequence("quit")
            await run_repl(agent, name="test")

    @pytest.mark.anyio
    async def test_exit_command(self) -> None:
        agent = _simple_agent()
        p1, p2 = _repl_ctx()
        with p1 as mock_cls, p2:
            mock_cls.return_value.prompt_async = _prompt_sequence("exit")
            await run_repl(agent, name="test")

    @pytest.mark.anyio
    async def test_empty_input_skipped(self) -> None:
        agent = _simple_agent()
        p1, p2 = _repl_ctx()
        with p1 as mock_cls, p2:
            mock_cls.return_value.prompt_async = _prompt_sequence("", "  ", "quit")
            await run_repl(agent, name="test")

    @pytest.mark.anyio
    async def test_eof_exits(self) -> None:
        agent = _simple_agent()
        p1, p2 = _repl_ctx()
        with p1 as mock_cls, p2:
            mock_cls.return_value.prompt_async = _prompt_sequence(EOFError)
            await run_repl(agent, name="test")

    @pytest.mark.anyio
    async def test_keyboard_interrupt_exits(self) -> None:
        agent = _simple_agent()
        p1, p2 = _repl_ctx()
        with p1 as mock_cls, p2:
            mock_cls.return_value.prompt_async = _prompt_sequence(KeyboardInterrupt)
            await run_repl(agent, name="test")

    @pytest.mark.anyio
    async def test_message_sent_to_agent(self) -> None:
        agent = _make_agent([[TextMessage("ok\n\n", "text/plain")]])
        p1, p2 = _repl_ctx()
        with p1 as mock_cls, p2:
            mock_cls.return_value.prompt_async = _prompt_sequence("hello", "quit")
            await run_repl(agent, name="test")

    @pytest.mark.anyio
    async def test_agent_exception_handled(self) -> None:
        agent = _make_agent([[]], raise_exc=RuntimeError("boom"))
        p1, p2 = _repl_ctx()
        with p1 as mock_cls, p2:
            mock_cls.return_value.prompt_async = _prompt_sequence("x", "quit")
            await run_repl(agent, name="test")


class TestRunReplQueueDiscard:
    @pytest.mark.anyio
    async def test_discards_queued_on_quit(self) -> None:
        agent = _simple_agent()
        p1, p2 = _repl_ctx()
        with p1 as mock_cls, p2:
            session_mock = mock_cls.return_value
            session_mock.prompt_async = _prompt_sequence("quit")
            await run_repl(agent, name="test")


class TestSlashModel:
    @pytest.mark.anyio
    async def test_slash_model_swaps_provider(self) -> None:
        agent = _simple_agent()
        agent.model_spec = ModelSpec(
            provider="Anthropic",
            auth="env",
            model_id="old-model",
        )
        agent.model = MagicMock(model_id="old-model")
        p1, p2 = _repl_ctx()
        with (
            p1 as mock_cls,
            p2,
            patch("sagent.repl.slash_commands.build_provider") as mock_bp,
        ):
            mock_model = MagicMock(model_id="new-model")
            mock_bp.return_value.model.return_value = mock_model
            mock_cls.return_value.prompt_async = _prompt_sequence(
                "/model --provider Google --auth env new-model",
                "quit",
            )
            await run_repl(agent, name="test")
        agent.swap_model.assert_called_once()
        call_args = agent.swap_model.call_args
        assert call_args.args[0] is mock_model
        assert call_args.kwargs["spec"].provider == "Google"
        assert call_args.kwargs["spec"].model_id == "new-model"

    @pytest.mark.anyio
    def test_slash_model_no_args_prints_current_settings(self) -> None:
        agent = _simple_agent()
        agent.model_spec = ModelSpec(
            provider="Anthropic",
            auth="env",
            model_id="old-model",
            account="work",
        )
        err_io = StringIO()
        console = Console(file=err_io, width=80, force_terminal=False)

        assert handle_slash_model(agent, console, "") is True

        text = err_io.getvalue()
        assert "provider=Anthropic" in text
        assert "auth=env" in text
        assert "model=old-model" in text
        assert "account=work" in text
        agent.swap_model.assert_not_called()

    @pytest.mark.anyio
    async def test_slash_model_no_args_not_forwarded(self) -> None:
        agent = _simple_agent()
        agent.model_spec = ModelSpec(
            provider="Anthropic",
            auth="env",
            model_id="old-model",
        )
        p1, p2 = _repl_ctx()
        with p1 as mock_cls, p2:
            mock_cls.return_value.prompt_async = _prompt_sequence("/model", "quit")
            await run_repl(agent, name="test")

    @pytest.mark.anyio
    async def test_slash_model_not_forwarded_to_agent(self) -> None:
        """A /model command must not enter the agent inbox."""
        agent = _simple_agent()
        agent.model_spec = ModelSpec(
            provider="Anthropic",
            auth="env",
            model_id="m",
        )
        received: list[str] = []

        async def _spy() -> AsyncGenerator[Message | None, None]:
            while True:
                prompt = await agent.inbox.get()
                if prompt == QUIT_SENTINEL:
                    return
                received.append(prompt)
                yield None

        agent.run_continuous = _spy
        p1, p2 = _repl_ctx()
        with p1 as mock_cls, p2:
            mock_cls.return_value.prompt_async = _prompt_sequence(
                "/model",
                "quit",
            )
            await run_repl(agent, name="test")
        assert not received


class TestSlashClear:
    def test_slash_clear_queues_front_for_drain(self) -> None:
        agent = _simple_agent()
        agent.inbox.put("later")
        err_io = StringIO()
        console = Console(file=err_io, width=80, force_terminal=False)

        assert handle_slash_clear(agent, console, " fresh start ") is True

        assert agent.inbox.drain() == ["/clear fresh start", "later"]
        assert "[/clear]" in err_io.getvalue()
        assert "fresh start" in err_io.getvalue()

    @pytest.mark.anyio
    async def test_slash_clear_goes_through_inbox(self) -> None:
        agent = _simple_agent()
        received: list[str] = []
        received_event = asyncio.Event()
        sent_clear = False

        async def _spy() -> AsyncGenerator[Message | None, None]:
            while True:
                prompt = await agent.inbox.get()
                if prompt == QUIT_SENTINEL:
                    return
                received.append(prompt)
                received_event.set()
                yield None

        async def _prompt(_prompt_text: str = "") -> str:
            nonlocal sent_clear
            del _prompt_text
            if not sent_clear:
                sent_clear = True
                return "/clear fresh start"
            await received_event.wait()
            return "quit"

        agent.run_continuous = _spy
        p1, p2 = _repl_ctx()
        with p1 as mock_cls, p2:
            mock_cls.return_value.prompt_async = _prompt
            await run_repl(agent, name="test")
        assert received == ["/clear fresh start"]


class TestUserBarRendering:
    """Verify user-input signal events produce a visible user bar."""

    def test_signal_event_renders_user_bar(self) -> None:
        """Basic: a text/x-signal-user-input event must render a user bar."""
        _, out, _ = _render([TextMessage("status?", "text/x-signal-user-input")])
        assert "status?" in out

    def test_injected_event_renders_user_bar(self) -> None:
        """Mid-request: a text/x-user-injected event must also render."""
        _, out, _ = _render([TextMessage("hey", "text/x-user-injected")])
        assert "hey" in out

    def test_signal_followed_by_agent_events(self) -> None:
        """Signal event batched with agent response events."""
        _, out, _ = _render(
            [
                TextMessage("status?", "text/x-signal-user-input"),
                TextMessage("AgentSelf status=Reviewing", "text/x-tool-label"),
                MultipartMessage(
                    (TextMessage("ok", "text/plain"),),
                    "multipart/x-tool-result",
                ),
                TextMessage("Here is the status.\n\n", "text/plain"),
            ]
        )
        assert "status?" in out

    def test_signal_followed_by_done(self) -> None:
        """Signal + done in same batch -- user bar must survive flush."""
        events: list[Message] = [
            TextMessage("status?", "text/x-signal-user-input"),
        ]
        out_io = StringIO()
        err_io = StringIO()
        out = Console(file=out_io, width=80, force_terminal=False)
        console = Console(file=err_io, width=80, force_terminal=False)
        render_frame = RenderState(console=console, out=out)
        render_batch(events, render_frame)
        _flush_render_frame(render_frame)
        assert "status?" in out_io.getvalue()

    @pytest.mark.anyio
    async def test_full_flow_user_bar_appears(self) -> None:
        """End-to-end: pump puts signal, agent responds, user bar rendered."""
        agent = _make_agent([[TextMessage("Done.\n\n", "text/plain")]])
        p1, p2 = _repl_ctx()
        with p1 as mock_cls, p2 as mock_console_cls:
            out_io = StringIO()
            mock_out = Console(file=out_io, width=80, force_terminal=False)
            err_io = StringIO()
            mock_console = Console(file=err_io, width=80, force_terminal=False)
            mock_console_cls.side_effect = [
                mock_console,  # console = Console(stderr=True)
                mock_out,  # out = Console(force_terminal=True)
            ]
            mock_cls.return_value.prompt_async = _prompt_sequence("status?", "quit")
            await run_repl(agent, name="test")
        rendered = out_io.getvalue()
        assert "status?" in rendered, (
            f"User bar missing from output. Got:\n{rendered!r}"
        )


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
