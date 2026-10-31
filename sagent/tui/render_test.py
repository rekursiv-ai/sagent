"""Tests for the renderer: runtime events, as calls on a printer."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast, override

import io
import logging
import time

from rich.console import Console
from treekle import PlainTree

import pytest

from sagent.model.exceptions import (
    AuthError,
    ContextOverflowError,
    DisconnectedError,
    OverloadedError,
    RateLimitedError,
    RefusedError,
    RequestTooLargeError,
    RetryableError,
)
from sagent.model.scripted import call, reply
from sagent.runtime import types
from sagent.runtime.exceptions import BudgetExhaustedError
from sagent.session.types import (
    AgentToAgentMessage,
    AnyToolResult,
    AssistantMessage,
    FileEditResult,
    FileWriteResult,
    ShellCommandResult,
    Splice,
    Thinking,
    ToolCall,
    UncategorizedToolResult,
    UserMessage,
)
from sagent.tools.display import DisplaySpec, ToolDisplay
from sagent.tools.edit import Edit
from sagent.tools.glob_tool import Glob
from sagent.tools.grep import Grep
from sagent.tools.read import Read
from sagent.tools.write import Write
from sagent.tui.console import ConsolePrinter
from sagent.tui.recording import RecordingPrinter
from sagent.tui.render import Scrollback, error_text, header_of, show_result
from sagent.tui.types import CommandOutput, Header


if TYPE_CHECKING:
    from sagent.tools.types import Tool, ToolContext
    from sagent.tui.types import Shown


_SHOWN = ToolDisplay(output=DisplaySpec(lines=None))
_ECHO = Header(key="1", label="Echo", display=_SHOWN)
"""A call whose label prints before ``show_result`` does, so none prints again."""


@dataclass(frozen=True, slots=True, kw_only=True)
class _Echo:
    name: str = "Echo"
    description: str = "Say the text back."
    schema: Mapping[str, PlainTree] = field(default_factory=dict[str, PlainTree])
    display: ToolDisplay = _SHOWN

    def summary(self, arguments: Mapping[str, PlainTree]) -> str:
        return f"Echo {arguments.get('text', '')}"

    def serialize_key(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> str | None:
        del context, arguments
        return None

    async def call(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> AnyToolResult:
        del context
        return UncategorizedToolResult(call_id="", content=str(arguments.get("text")))


def _tools() -> dict[str, Tool]:
    return {"Echo": _Echo()}


def _renderer(
    *,
    show_thinking: bool = True,
    flush_chars: int = 64 * 1024,
) -> tuple[Scrollback, RecordingPrinter]:
    printer = RecordingPrinter(show_thinking=show_thinking)
    return Scrollback(printer, tools=_tools(), flush_chars=flush_chars), printer


def _turn(*calls: ToolCall, text: str = "") -> types.ModelResponseComplete:
    return types.ModelResponseComplete(response=reply(text, calls=calls))


def _result(
    call_id: str,
    text: str = "out",
    *,
    is_error: bool | None = None,
    stub: Literal["pending", "cancelled"] | None = None,
) -> UncategorizedToolResult:
    return UncategorizedToolResult(
        call_id=call_id,
        content=text,
        is_error=is_error,
        stub=stub,
    )


def _kind(value: object) -> str:
    """Name a case by its value's class; a banner's text names nothing."""
    return "" if isinstance(value, str) else type(value).__name__


def test_a_block_renders_once_the_next_one_starts() -> None:
    render, printer = _renderer()
    render(types.ModelResponsePartial(text="first para"))
    assert printer.said == []
    render(types.ModelResponsePartial(text="\n\nsecond"))
    assert printer.said == [("markdown", "first para")]
    render(types.ModelResponsePartial(text=" para"))
    assert printer.said == [("markdown", "first para")]


@pytest.mark.parametrize("block", ["ends X  ", "ends X"])
def test_a_finished_block_keeps_all_but_its_final_newlines(block: str) -> None:
    render, printer = _renderer()
    render(types.ModelResponsePartial(text=f"{block}\n\n\nnext"))
    assert printer.said == [("markdown", block)]


@pytest.mark.parametrize(
    ("held", "shown"),
    [("tail X  \n\n", "tail X  "), ("tail X  ", "tail X  "), ("tail X\n", "tail X")],
)
def test_what_is_held_renders_when_anything_else_shows(held: str, shown: str) -> None:
    render, printer = _renderer()
    render(types.ModelResponsePartial(text=held))
    render(UserMessage(content="next"))
    assert printer.said == [("markdown", shown), ("user", "next")]


def test_nothing_held_renders_nothing() -> None:
    render, printer = _renderer()
    render(types.ModelResponsePartial(text="\n\n"))
    render(types.CompactStarted())
    assert printer.said == [("note", "[compacting history…]")]


def test_blank_lines_between_blocks_render_once() -> None:
    render, printer = _renderer()
    render(types.ModelResponsePartial(text="a\n\n- b"))
    render(types.ModelResponsePartial(text="\n\n- c\n\n\n\nd"))
    assert printer.said == [("markdown", "a"), ("markdown", "- b\n\n- c")]


def test_held_prose_past_the_cap_renders_only_its_whole_lines() -> None:
    render, printer = _renderer(flush_chars=10)
    render(types.ModelResponsePartial(text="line one\nline tw"))
    assert printer.said == [("markdown", "line one")]
    render(types.ModelResponsePartial(text="o\n\nnext"))
    assert printer.said == [("markdown", "line one"), ("markdown", "line two")]


def test_held_text_past_the_cap_renders_unfinished() -> None:
    render, printer = _renderer(flush_chars=10)
    render(types.ModelResponsePartial(text="```\n" + "x" * 6))
    assert printer.said == []
    render(types.ModelResponsePartial(text="y"))
    assert printer.said == [("markdown", "```\nxxxxxxy")]
    render(types.ModelResponsePartial(text="z"))
    assert printer.said == [("markdown", "```\nxxxxxxy")]


@pytest.mark.parametrize(
    "event",
    [
        types.ModelIdle(),
        types.AgentIdle(),
        types.CohortStarted(),
        types.ModelCallStarted(),
        types.UserQueuedMessage(text="q"),
        types.UserDeferredMessage(text="d"),
        types.AgentSendQueuedMessage(sender="a", text="q"),
        types.AgentSendDeferredMessage(sender="a", text="d"),
        types.SaveSession(),
    ],
    ids=_kind,
)
def test_events_with_nothing_to_show_neither_print_nor_flush(
    event: types.RuntimeEvent,
) -> None:
    render, printer = _renderer()
    render(types.ModelResponsePartial(text="held"))
    render(event)
    assert printer.said == []


def test_thinking_shows_only_while_the_printer_shows_it() -> None:
    render, printer = _renderer()
    printer.show_thinking = False
    render(types.ModelResponseThinking(text="hidden"))
    render(_turn(text="done"))
    assert printer.said == [("markdown", "done")]


def test_thinking_heads_once_then_streams_by_block_and_closes_once() -> None:
    render, printer = _renderer()
    for word in ("\n", "first ", "para", "\n\nsecond", " para"):
        render(types.ModelResponseThinking(text=word))
    assert printer.said == [("thinking", "["), ("thinking", "\nfirst para\n\n")]
    render(types.ModelResponsePartial(text="reply"))
    assert printer.said[2:] == [("thinking", "second para]")]


def test_thinking_a_notice_interrupts_reopens_under_a_new_heading() -> None:
    render, printer = _renderer()
    render(types.ModelResponseThinking(text="before\n\n"))
    render(types.CompactStarted())
    render(types.ModelResponseThinking(text="after"))
    render(types.ModelResponsePartial(text="reply"))
    thinking = [s for k, s in printer.said if k == "thinking"]
    assert thinking.count("[") == 2
    assert sum(s.endswith("]") for s in thinking) == 2


def test_word_sized_thinking_deltas_print_one_heading_and_the_joined_text() -> None:
    out = io.StringIO()
    printer = ConsolePrinter(
        Console(file=out, width=200, color_system=None, highlight=False),
        show_thinking=True,
    )
    render = Scrollback(printer, tools=_tools())
    words = [f"word{i} " for i in range(20)]
    for word in words:
        render(types.ModelResponseThinking(text=word))
    render(types.ModelResponsePartial(text="Answer.", record=1))
    response = reply("Answer.")
    render(
        types.ModelResponseComplete(
            response=type(response)(
                records=(Thinking(content="".join(words)), *response.records),
                tokens=response.tokens,
                spend=response.spend,
                stop_reason="end_turn",
            ),
        ),
    )
    rows = [row.rstrip() for row in out.getvalue().split("\n")]
    assert rows == ["∴ Thinking", f"  {''.join(words).rstrip()}", "", "Answer.", ""]


def test_hidden_thinking_does_not_hold_back_the_reply_after_it() -> None:
    render, printer = _renderer()
    printer.show_thinking = False
    render(types.ModelResponseThinking(text="hidden"))
    render(types.ModelResponsePartial(text="a\n\nb", record=1))
    assert printer.said == [("markdown", "a")]


def test_text_after_a_call_waits_to_print_after_the_call() -> None:
    render, printer = _renderer()
    render(types.ModelResponsePartial(text="Before.\n\nheld"))
    render(types.ModelResponsePartial(text="After.\n\nnext", record=2))
    assert printer.said == [("markdown", "Before.")]
    response = reply("Before.\n\nheld", calls=[call("Echo", "1", text="a")])
    render(
        types.ModelResponseComplete(
            response=type(response)(
                records=(*response.records, AssistantMessage(content="After.\n\nnext")),
                tokens=response.tokens,
                spend=response.spend,
                stop_reason="tool_use",
            ),
        ),
    )
    assert printer.said == [
        ("markdown", "Before."),
        ("markdown", "held"),
        ("call", "Echo a"),
        ("markdown", "After.\n\nnext"),
    ]


def test_a_blank_thinking_stream_prints_nothing() -> None:
    render, printer = _renderer()
    render(types.ModelResponseThinking(text=" \n"))
    render(_turn(text="done"))
    assert printer.said == [("markdown", "done")]


def test_streamed_thinking_is_not_printed_again_by_its_record() -> None:
    render, printer = _renderer()
    render(types.ModelResponseThinking(text="hm"))
    response = reply("x")
    render(
        types.ModelResponseComplete(
            response=type(response)(
                records=(Thinking(content="hm"), *response.records),
                tokens=response.tokens,
                spend=response.spend,
                stop_reason="end_turn",
            ),
        ),
    )
    assert printer.said == [("thinking", "["), ("thinking", "hm]"), ("markdown", "x")]


def test_thinking_ends_the_reply_being_streamed_and_titles_do_not() -> None:
    render, printer = _renderer()
    render(types.ModelResponsePartial(text="held"))
    render(types.StatusChanged(text="working"))
    assert printer.said == []
    render(types.ModelResponseThinking(text="hm"))
    assert printer.said == [("markdown", "held"), ("thinking", "[")]


def test_an_unstreamed_turn_prints_its_reasoning_and_reply_from_its_records() -> None:
    render, printer = _renderer()
    response = reply("**done**\n", calls=[call("Echo", "1", text="a")])
    render(
        types.ModelResponseComplete(
            response=type(response)(
                records=(Thinking(content="because"), *response.records),
                tokens=response.tokens,
                spend=response.spend,
                stop_reason="tool_use",
            ),
        ),
    )
    render(types.CohortStarted())
    assert printer.said == [
        ("thinking", "[because]"),
        ("markdown", "**done**"),
        ("call", "Echo a"),
    ]


def test_command_output_is_a_note_or_an_error() -> None:
    render, printer = _renderer()
    render.show(types.ModelResponsePartial(text="held"))
    render.show(CommandOutput(text="[/model] a", tone="note"))
    render.show(CommandOutput(text="no such", tone="error"))
    assert printer.said == [
        ("markdown", "held"),
        ("note", "[/model] a"),
        ("error", "no such"),
    ]


def test_a_skipped_file_is_an_error_the_user_sees() -> None:
    render, printer = _renderer()
    render.show(types.Skipped(text="Skipping a.md: line 3."))
    assert printer.said == [("error", "Skipping a.md: line 3.")]


def test_a_skipped_file_is_drawn_when_it_arrives() -> None:
    render, printer = _renderer()
    render(types.Skipped(text="Skipping a.md: line 3."))
    assert printer.said == [("error", "Skipping a.md: line 3.")]


class _Stranger:
    """No event or record the renderer knows, as a type added and not drawn is."""

    call_id: str = "1"
    stub: str | None = None


@pytest.mark.parametrize(
    "given",
    [
        cast("Shown", _Stranger()),
        # A record arrives by a known event, and fails where records are drawn.
        types.DetachedResult(result=cast("AnyToolResult", _Stranger())),
    ],
    ids=["event", "record"],
)
def test_what_the_renderer_does_not_know_fails_rather_than_draws(given: Shown) -> None:
    render, printer = _renderer()
    with pytest.raises(AssertionError, match="_Stranger"):
        render.show(given)
    assert printer.said == []


def test_a_child_says_its_text_and_calls_under_its_label() -> None:
    render, printer = _renderer()
    turn = _turn(call("Echo", text="hi"), call("Mystery"), text="looking")
    render(types.ChildEvent(label="Agent_0", inner=turn))
    assert printer.said == [
        ("note", "Agent_0: looking"),
        ("call", "Agent_0: Echo hi"),
        ("call", "Agent_0: Mystery"),
    ]
    assert printer.displays == [_SHOWN, ToolDisplay()]


def test_a_child_s_blank_text_says_nothing() -> None:
    render, printer = _renderer()
    response = reply("x")
    blank = types.ModelResponseComplete(
        response=type(response)(
            records=(
                AssistantMessage(content="  "),
                AssistantMessage(),
                Thinking(content="t"),
            ),
            tokens=response.tokens,
            spend=response.spend,
            stop_reason="end_turn",
        ),
    )
    render(types.ChildEvent(label="c", inner=blank))
    assert printer.said == []


def test_a_grandchild_is_shown_under_its_own_label() -> None:
    render, printer = _renderer()
    inner = types.ChildEvent(
        label="Agent_0_0",
        inner=types.ModelResponseError(exception=ValueError("x")),
    )
    render(types.ChildEvent(label="Agent_0", inner=inner))
    assert printer.said == [("error", "Agent_0_0: ValueError: x")]


def test_a_child_s_results_show_only_when_they_fail() -> None:
    render, printer = _renderer()
    turn = _turn(call("Echo", "1", text="a"), call("Echo", "2", text="b"))
    render(types.ChildEvent(label="c", inner=turn))
    render(types.ChildEvent(label="c", inner=_result("1", "fine", is_error=False)))
    failed = _result("2", "bad<system-reminder>r</system-reminder>\n", is_error=True)
    render(types.ChildEvent(label="c", inner=failed))
    assert printer.said == [
        ("call", "c: Echo a"),
        ("call", "c: Echo b"),
        ("failed", "c: bad"),
    ]


def _console() -> tuple[Scrollback, ConsolePrinter, io.StringIO]:
    """Return a scrollback over a real printer, and what the printer writes to."""
    out = io.StringIO()
    printer = ConsolePrinter(
        Console(file=out, width=60, color_system=None, highlight=False),
    )
    return Scrollback(printer, tools=_tools()), printer, out


def _rows(out: io.StringIO) -> list[str]:
    return [row.rstrip() for row in out.getvalue().splitlines() if row.strip()]


def test_a_child_s_failure_names_its_own_call_again() -> None:
    render, _, out = _console()
    turn = _turn(call("Echo", "1", text="a"), call("Echo", "2", text="b"))
    render(types.ChildEvent(label="c", inner=turn))
    render(types.ChildEvent(label="c", inner=_result("1", "broke", is_error=True)))
    assert _rows(out) == ["  c: Echo a", "  c: Echo b", "  c: Echo a", "    ✗ c: broke"]


def test_a_child_s_failure_right_under_its_call_does_not_name_it_again() -> None:
    render, _, out = _console()
    render(types.ChildEvent(label="c", inner=_turn(call("Echo", "1", text="a"))))
    render(types.ChildEvent(label="c", inner=_result("1", "broke", is_error=True)))
    assert _rows(out) == ["  c: Echo a", "    ✗ c: broke"]


def test_a_result_whose_call_is_unknown_sits_under_no_call() -> None:
    render, printer = _renderer()
    render(_turn(call("Echo", "1", text="a")))
    render(_result("1", "said a"))
    render(_result("9", "orphan"))
    assert render.shown == 1
    render(_result("9", "bad", is_error=True))
    assert render.shown == 2
    render(types.ChildEvent(label="c", inner=_result("8", "worse", is_error=True)))
    assert printer.said == [
        ("call", "Echo a"),
        ("output", "said a"),
        ("error", "bad"),
        ("error", "c: worse"),
    ]
    assert render.shown == 2


def test_a_result_that_shows_nothing_names_no_call_again() -> None:
    render, _, out = _console()
    render(_turn(call("Hidden", "1"), call("Hidden", "2"), call("Echo", "3", text="c")))
    render(types.CohortStarted())
    for call_id in ("1", "2"):
        render(_result(call_id, "x"))
    render(_result("3", "said c"))
    assert _rows(out) == ["  Hidden", "  Hidden", "  Echo c", "    said c"]


def test_a_hidden_result_between_a_call_and_its_failure_names_no_call_again() -> None:
    render, printer = _renderer()
    render(_turn(call("Other", "1"), call("Echo", "2", text="b")))
    render(types.CohortStarted())
    render(_result("1", "hidden"))
    render(_result("2", "bad", is_error=True))
    assert printer.said == [
        ("call", "Other"),
        ("call", "Echo b"),
        ("output", "hidden"),
        ("failed", "bad"),
    ]


def test_show_result_puts_every_row_under_its_call() -> None:
    printer = RecordingPrinter()
    printer.note("x")
    hidden = Header(key="1", label="Echo", display=ToolDisplay())
    show_result(printer, _result("1", "hidden"), header=hidden)
    show_result(printer, _result("1", "bad", is_error=True), header=_ECHO)
    assert printer.said == [
        ("note", "x"),
        ("output", "hidden"),
        ("call", "Echo"),
        ("failed", "bad"),
    ]


@pytest.mark.parametrize("writer", ["agent", "line", "note", "error"])
def test_a_result_below_a_row_printed_past_scrollback_names_its_call_again(
    writer: str,
) -> None:
    render, printer, out = _console()
    render(_turn(call("Echo", "1", text="a")))
    render(types.CohortStarted())
    match writer:
        case "agent":
            printer.agent("Agent_0", "user", "a question")
        case _:
            getattr(printer, writer)("x")
    render(_result("1", "said a"))
    assert _rows(out)[-2:] == ["  Echo a", "    said a"]


def test_a_child_s_failures_messages_and_suspensions_are_labelled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(time, "time", lambda: _NOON)
    render, printer = _renderer()
    render(
        types.ChildEvent(
            label="c",
            inner=types.ModelResponseError(exception=ValueError("x")),
        ),
    )
    render(
        types.ChildEvent(
            label="c",
            inner=AgentToAgentMessage(sender="me", content="hi"),
        ),
    )
    render(types.ChildEvent(label="c", inner=AgentToAgentMessage()))
    suspended = types.ModelCallSuspended(
        exception=RateLimitedError("spent", limit="weekly"),
        until=_NOON + 5.0,
        attempt=1,
    )
    render(types.ChildEvent(label="c", inner=suspended))
    render(types.ChildEvent(label="c", inner=UserMessage(content="ignored")))
    assert printer.said == [
        ("error", "c: ValueError: x"),
        ("agent", "me→c: hi"),
        ("agent", "agent→c: "),
        ("error", "c: Weekly limit reached. Resumes Wed Oct 28 12:00:05 (5s)."),
    ]


def test_a_child_s_work_does_not_end_the_reply_being_streamed() -> None:
    render, printer = _renderer()
    render(types.ModelResponsePartial(text="held"))
    render(
        types.ChildEvent(
            label="c",
            inner=types.ModelResponseError(exception=ValueError("x")),
        ),
    )
    render(types.ChildDoneEvent(label="c", elapsed=65.0, tokens=12_345, cost=0.426))
    assert printer.said == [
        ("error", "c: ValueError: x"),
        ("note", "[c done: 1m 5s, 12K tokens, $0.43]"),
    ]


def test_messages_show_as_bars() -> None:
    render, printer = _renderer()
    render(UserMessage(content="hi"))
    render(UserMessage())
    render(AgentToAgentMessage(sender="Agent_0", content="done"))
    render(AgentToAgentMessage(sender="watcher", recipient="Agent", content="ok"))
    render(AgentToAgentMessage())
    render(AgentToAgentMessage(sender="user", recipient="Agent", content="ping"))
    assert printer.said == [
        ("user", "hi"),
        ("user", ""),
        ("agent", "Agent_0→me: done"),
        ("agent", "watcher→me: ok"),
        ("agent", "agent→me: "),
        ("note", "[/send] user→me: ping"),
    ]


def test_a_turn_shows_its_calls_and_each_result_by_its_tool_s_display() -> None:
    render, printer = _renderer()
    render(_turn(call("Echo", "1", text="a"), call("Other", "2"), text="unstreamed"))
    render(_result("2", "said b"))
    assert printer.said == [
        ("markdown", "unstreamed"),
        ("call", "Echo a"),
        ("call", "Other"),
        ("output", "said b"),
    ]
    assert printer.displays == [_SHOWN, ToolDisplay(), ToolDisplay()]


def test_a_call_prints_when_it_starts_before_any_result() -> None:
    render, printer = _renderer()
    render(_turn(call("Echo", "1", text="a"), call("Other", "2")))
    assert printer.said == []
    render(types.CohortStarted())
    assert printer.said == [("call", "Echo a"), ("call", "Other")]


@pytest.mark.parametrize(
    ("between", "said"),
    [
        (types.CompactStarted(), ("note", "[compacting history…]")),
        (
            types.ChildEvent(
                label="kid",
                inner=types.ModelResponseError(exception=ValueError("x")),
            ),
            ("error", "kid: ValueError: x"),
        ),
    ],
    ids=["note", "child"],
)
def test_a_result_below_anything_else_names_its_call_again(
    between: types.RuntimeEvent,
    said: tuple[str, str],
) -> None:
    render, printer = _renderer()
    render(_turn(call("Echo", "1", text="a")))
    render(between)
    render(_result("1", "said a"))
    assert printer.said == [
        ("call", "Echo a"),
        said,
        ("call", "Echo a"),
        ("output", "said a"),
    ]


def test_a_result_after_its_turn_s_closing_text_names_its_call_again() -> None:
    render, printer = _renderer()
    response = reply("", calls=[call("Echo", "1", text="a")])
    render(
        types.ModelResponseComplete(
            response=type(response)(
                records=(*response.records, AssistantMessage(content="Running.")),
                tokens=response.tokens,
                spend=response.spend,
                stop_reason="tool_use",
            ),
        ),
    )
    render(_result("1", "said a"))
    assert printer.said == [
        ("call", "Echo a"),
        ("markdown", "Running."),
        ("call", "Echo a"),
        ("output", "said a"),
    ]


def test_a_detached_result_shows_like_any_other() -> None:
    render, printer = _renderer()
    render(_turn(call("Echo", "1")))
    render(types.DetachedResult(result=_result("1", "late")))
    assert printer.said[-1] == ("output", "late")
    assert printer.displays[-1] == _SHOWN


@pytest.mark.parametrize("delivery", ["BackgroundTask", "DetachedArrived"])
def test_a_taped_delivery_is_numbered_when_detached_and_again_when_it_lands(
    delivery: str,
) -> None:
    render, printer = _renderer()
    render.show(call("Echo", "1", text="a"))
    render.show(_result("1", "[detached]", stub="pending"))
    render.show(call("Echo", "2", text="b"))
    render.show(_result("2", "now"))
    render.show(UserMessage(content="[detached tool 1 completed]"))
    render.show(ToolCall(call_id="1:detached", name=delivery))
    render.show(_result("1:detached", "late"))
    assert printer.said == [
        ("call", "Echo a"),
        ("output", "[detached; #1]"),
        ("call", "Echo b"),
        ("output", "now"),
        ("call", "Echo a"),
        ("output", "[undetached; #1]"),
        ("output", "late"),
    ]
    assert render.shown == 3


def test_a_second_detached_call_takes_the_next_number() -> None:
    render, printer = _renderer()
    render(_turn(call("Echo", "1", text="a"), call("Echo", "2", text="b")))
    render(_result("1", "[detached]", stub="pending"))
    render(_result("2", "[detached]", stub="pending"))
    render(types.ModelCallStarted())
    assert printer.said == [
        ("call", "Echo a"),
        ("output", "[detached; #1]"),
        ("call", "Echo b"),
        ("output", "[detached; #2]"),
    ]


def test_a_backgrounded_call_beside_a_running_one_is_labelled_once() -> None:
    render, printer = _renderer()
    render(_turn(call("Echo", "1", text="slow"), call("Echo", "2", text="fast")))
    render(_result("1", "[Running in background: Echo as job 1]", stub="pending"))
    render(types.CohortStarted())
    render(_result("2", "fast done"))
    render(types.DetachedResult(result=_result("1", "slow done")))
    assert printer.said == [
        ("call", "Echo slow"),
        ("output", "[detached; #1]"),
        ("call", "Echo fast"),
        ("output", "fast done"),
        ("call", "Echo slow"),
        ("output", "[undetached; #1]"),
        ("output", "slow done"),
    ]
    assert printer.displays[1] == printer.displays[5] == _SHOWN


def test_a_call_detached_while_it_runs_is_numbered_under_its_label() -> None:
    render, printer = _renderer()
    render(_turn(call("Echo", "1", text="a")))
    render(types.CohortStarted())
    render(_result("1", "[detached]", stub="pending"))
    assert printer.said == [("call", "Echo a"), ("output", "[detached; #1]")]
    assert printer.displays[-1] == _SHOWN


def test_a_stub_for_an_earlier_call_labels_the_held_ones_first() -> None:
    render, printer = _renderer()
    render(UserMessage(content="go"))
    render(_turn(call("Echo", "1", text="a")))
    render(types.CohortStarted())
    render(_turn(call("Echo", "2", text="b")))
    render(_result("1", "[detached]", stub="pending"))
    assert render.shown == 2
    assert printer.said == [
        ("user", "go"),
        ("call", "Echo a"),
        ("call", "Echo b"),
        ("call", "Echo a"),
        ("output", "[detached; #1]"),
    ]


def test_scrollback_names_paths_below_where_sagent_started() -> None:
    printer = RecordingPrinter()
    render = Scrollback(printer, tools={"Read": Read()}, home=Path("/r"))
    render(_turn(call("Read", "1", file_path="/r/a.py")))
    render(_result("1", "[detached]", stub="pending"))
    render(types.CohortStarted())
    render(types.CompactStarted())
    render(types.DetachedResult(result=_result("1", "x")))
    render(
        types.ChildEvent(label="kid", inner=_turn(call("Read", file_path="/r/b.py"))),
    )
    assert [said for kind, said in printer.said if kind == "call"] == [
        "Read a.py",
        "Read a.py",
        "kid: Read b.py",
    ]


def test_a_user_message_that_merely_mentions_a_delivery_still_shows() -> None:
    render, printer = _renderer()
    render.show(UserMessage(content="see [detached tool 1 completed] above"))
    assert printer.said == [("user", "see [detached tool 1 completed] above")]


def test_a_write_shows_as_a_diff_cut_like_its_tool_s_output() -> None:
    printer = RecordingPrinter()
    printer.call(_ECHO)
    write = FileWriteResult(call_id="1", path="a.py", content="x\ny\n")
    cut = ToolDisplay(output=DisplaySpec(lines=(1, 1)))
    show_result(printer, write, header=replace(_ECHO, display=cut))
    show_result(printer, write, header=_ECHO)
    show_result(printer, write, header=replace(_ECHO, display=ToolDisplay()))
    show_result(printer, FileWriteResult(call_id="2", path="b.py"), header=_ECHO)
    assert printer.said == [
        ("call", "Echo"),
        ("diff", "a.py\n@@ -0,0 +1,2 @@\n+x\n+y"),
        ("diff", "a.py\n@@ -0,0 +1,2 @@\n+x\n+y"),
        ("output", "Wrote a.py."),
        ("output", "Wrote b.py."),
    ]
    assert printer.cuts == [(1, 1), None]


def test_cancellations_and_compactions_are_notes() -> None:
    render, printer = _renderer()
    render(types.ModelResponseCancelled())
    render(types.CompactStarted())
    render(types.CompactComplete(token_before=900, token_after=100, payload_entries=4))
    render(types.CompactComplete(fallback_reason="too big", preserved_tail_count=1))
    render(types.CompactComplete(fallback_reason="too big", preserved_tail_count=3))
    render(types.CompactFailed(exception=TimeoutError(), tape_len=1))
    render(
        types.CompactFailed(
            exception=BudgetExhaustedError(total_cost_usd=2, max_budget_usd=1),
            tape_len=1,
        ),
    )
    assert printer.said == [
        ("note", "[interrupted]"),
        ("note", "[compacting history…]"),
        ("note", "[compaction complete: ~900 → ~100 tokens, 4 entries]"),
        ("note", "[compaction fallback: too big; preserved 1 tail entry]"),
        ("note", "[compaction fallback: too big; preserved 3 tail entries]"),
        ("note", "[compaction failed: TimeoutError]"),
        ("note", "[compaction failed: Budget exhausted: $2.00 >= $1.00]"),
    ]


@pytest.mark.parametrize(
    ("error", "banner"),
    [
        (
            AuthError("expired"),
            "agent halted -- /login to sign in again, or /account or /model to switch",
        ),
        (
            RateLimitedError("slow down"),
            "agent halted -- /account or /model to switch, or type to retry",
        ),
        (
            ContextOverflowError("big"),
            "agent halted -- run /compact <hints>, /clear, or /model to reduce context",
        ),
        (
            RequestTooLargeError("bytes"),
            "agent halted -- reduce request data or run /clear or /quit",
        ),
        (
            RefusedError("no"),
            "refused -- input since the last answer is withheld; type to continue",
        ),
        (
            ValueError("other"),
            "agent halted -- type to retry, or /login, /model, /quit",
        ),
    ],
    ids=_kind,
)
def test_a_model_error_shows_and_says_what_the_user_can_do(
    error: Exception,
    banner: str,
) -> None:
    render, printer = _renderer()
    render(types.ModelResponseError(exception=error))
    assert printer.said == [
        ("error", f"{type(error).__name__}: {error}"),
        ("halt", banner),
    ]


def test_a_child_s_long_reply_hangs_under_its_label() -> None:
    """A child's words wrapped back to column 0, under its label, not its text."""
    out = io.StringIO()
    printer = ConsolePrinter(Console(file=out, width=30, color_system=None))
    render = Scrollback(printer, tools=_tools())
    render(types.ChildEvent(label="Agent_0", inner=_turn(text="word " * 12)))
    rows = [r.rstrip() for r in out.getvalue().split("\n") if r.strip()]
    assert len(rows) > 1
    assert rows[0].startswith("Agent_0: word")
    assert all(r.startswith(" " * 9) and r[9] != " " for r in rows[1:]), rows


def test_a_printer_failure_is_logged_and_never_raised(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class _Broken(RecordingPrinter):
        @override
        def note(self, text: str, *, lead: str = "", prose: bool = False) -> None:
            del prose
            raise RuntimeError(lead + text)

    render = Scrollback(_Broken(), tools=_tools())
    with caplog.at_level(logging.ERROR, logger="sagent.tui.render"):
        render(types.CompactStarted())
    assert caplog.messages == ["rendering CompactStarted failed"]


# Noon local, so a wait under 12 hours stays on the same day in any zone.
_NOON = time.mktime((2026, 10, 28, 12, 0, 0, 0, 0, -1))

_WAITING = "agent waiting -- /account or /model to switch now, or type to retry now"


@pytest.mark.parametrize(
    ("error", "wait_sec", "reason"),
    [
        (
            RateLimitedError("limit hit"),
            59.0,
            "Rate limited. Resumes Wed Oct 28 12:00:59 (59s).",
        ),
        (
            RateLimitedError("spent", limit="weekly"),
            86_400.0 + 22 * 3_600.0 + 6 * 60.0 + 9.0,
            "Weekly limit reached. Resumes Fri Oct 30 10:06:09 (1d 22h 6m 9s).",
        ),
        (
            RateLimitedError("spent", limit="5-hour"),
            10_683.0,
            "5-hour limit reached. Resumes Wed Oct 28 14:58:03 (2h 58m 3s).",
        ),
        (
            OverloadedError("busy"),
            65.0,
            "Service overloaded. Resumes Wed Oct 28 12:01:05 (1m 5s).",
        ),
        (
            DisconnectedError("dropped"),
            5.0,
            "Disconnected. Resumes Wed Oct 28 12:00:05 (5s).",
        ),
        (
            RetryableError("down"),
            5.0,
            "Service unavailable. Resumes Wed Oct 28 12:00:05 (5s).",
        ),
    ],
    ids=_kind,
)
def test_a_suspension_shows_why_when_it_resumes_and_what_to_do(
    monkeypatch: pytest.MonkeyPatch,
    error: RetryableError,
    wait_sec: float,
    reason: str,
) -> None:
    """The date as ``date`` prints it, and the wait as the status line does."""
    monkeypatch.setattr(time, "time", lambda: _NOON)
    render, printer = _renderer()
    render(types.ModelCallSuspended(exception=error, until=_NOON + wait_sec, attempt=1))
    assert printer.said == [("error", reason), ("halt", _WAITING)]


@pytest.mark.parametrize(
    ("switched", "note"),
    [
        (
            types.LoginSwitched(
                provider="acli",
                was="default",
                now="personal",
                window="5-hour",
                until=_NOON + 3_600.0,
            ),
            "[default spent (5-hour, resets Wed Oct 28 13:00:00 (1h 0m 0s)); now on personal]",
        ),
        (
            types.LoginSwitched(
                provider="acli",
                was="work",
                now="default",
                first="default",
            ),
            "[on default, first in order]",
        ),
        (
            types.LoginSwitched(
                provider="acli",
                was="work",
                now="personal",
                first="default",
            ),
            "[on personal, first with quota]",
        ),
        (
            types.LoginSwitched(
                provider="acli",
                was="work",
                now="default",
                window="signed out",
            ),
            "[work signed out; now on default]",
        ),
    ],
    ids=["spent", "first", "first-live", "signed-out"],
)
def test_a_login_switch_says_why_and_what_signs_in_now(
    monkeypatch: pytest.MonkeyPatch,
    switched: types.LoginSwitched,
    note: str,
) -> None:
    monkeypatch.setattr(time, "time", lambda: _NOON)
    render, printer = _renderer()
    render(switched)
    assert printer.said == [("note", note)]


def test_every_login_spent_says_when_each_resets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(time, "time", lambda: _NOON)
    render, printer = _renderer()
    render(
        types.LoginsSpent(
            provider="acli",
            resets=(("work", _NOON + 60.0), ("default", _NOON + 3_600.0)),
        ),
    )
    assert printer.said == [
        (
            "note",
            (
                "[no acli login has quota: work Wed Oct 28 12:01:00 (1m 0s),"
                " default Wed Oct 28 13:00:00 (1h 0m 0s)]"
            ),
        ),
    ]


def test_a_child_s_login_switch_is_noted_under_its_label() -> None:
    render, printer = _renderer()
    switched = types.LoginSwitched(
        provider="acli",
        was="work",
        now="default",
        first="default",
    )
    render(types.ChildEvent(label="Agent_0", inner=switched))
    spent = types.LoginsSpent(provider="acli", resets=(("work", _NOON),))
    render(types.ChildEvent(label="Agent_0", inner=spent))
    assert printer.said[0] == ("note", "Agent_0: [on default, first in order]")
    assert printer.said[1][1].startswith("Agent_0: [no acli login has quota: work ")


def test_a_suspension_failing_again_says_why_without_the_banner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A backoff of sixty tries drew sixty banners."""
    monkeypatch.setattr(time, "time", lambda: _NOON)
    render, printer = _renderer()
    render(
        types.ModelCallSuspended(
            exception=OverloadedError("busy"),
            until=_NOON + 10.0,
            attempt=2,
        ),
    )
    assert printer.said == [
        ("error", "Service overloaded. Resumes Wed Oct 28 12:00:10 (10s)."),
    ]


@pytest.mark.parametrize(
    ("error", "text"),
    [
        (
            RateLimitedError("spent", retry_after_sec=7_200.0, limit="weekly"),
            "Weekly limit reached. Resets Wed Oct 28 14:00:00 (2h 0m 0s).",
        ),
        (
            RateLimitedError("slow", retry_after_sec=30.0),
            "Rate limited. Resets Wed Oct 28 12:00:30 (30s).",
        ),
        (RateLimitedError("slow"), "RateLimitedError: slow"),
    ],
    ids=["window", "throttle", "no-wait"],
)
def test_a_rate_limit_that_ends_the_turn_says_when_it_resets(
    monkeypatch: pytest.MonkeyPatch,
    error: RateLimitedError,
    text: str,
) -> None:
    monkeypatch.setattr(time, "time", lambda: _NOON)
    assert error_text(error) == text


def _label_id(value: object) -> str:
    """Name a case by its label alone."""
    return value if isinstance(value, str) else ""


@pytest.mark.parametrize(
    ("tool", "arguments", "label"),
    [
        (Read(), {"file_path": "/home/u/repo/pkg/a.py"}, "Read pkg/a.py"),
        (
            Read(),
            {"file_path": "/home/u/repo/a.py", "offset": 1, "limit": 50},
            "Read a.py:1-50",
        ),
        (Write(), {"file_path": "/home/u/repo/pkg/a.py"}, "Write pkg/a.py"),
        (Edit(), {"file_path": "/home/u/repo/pkg/a.py"}, "Edit pkg/a.py"),
        (Grep(), {"pattern": "x", "path": "/home/u/repo/pkg"}, "Grep 'x' in pkg"),
        (Grep(), {"pattern": "x", "path": "/home/u/repo"}, "Grep 'x'"),
        (Glob(), {"pattern": "*.py", "path": "/home/u/repo/pkg"}, "Glob *.py in pkg"),
        (Read(), {"file_path": "/etc/hosts"}, "Read /etc/hosts"),
        (
            Read(),
            {"file_path": "/home/u/repository/a.py"},
            "Read /home/u/repository/a.py",
        ),
        (Read(), {"file_path": "pkg/a.py"}, "Read pkg/a.py"),
        (Grep(), {"pattern": "/home/u/repo/x"}, "Grep '/home/u/repo/x'"),
        (Glob(), {"pattern": "*", "path": 3}, "Glob * in 3"),
    ],
    ids=_label_id,
)
def test_a_call_names_a_path_below_where_sagent_started_relative_to_it(
    tool: Tool,
    arguments: dict[str, PlainTree],
    label: str,
) -> None:
    header = header_of(
        ToolCall(call_id="1", name=tool.name, arguments=arguments),
        tools={tool.name: tool},
        home=Path("/home/u/repo"),
    )
    assert header.label == label


def test_a_header_is_its_tool_s_label_and_display_or_the_call_s_name() -> None:
    echo = header_of(call("Echo", "1", text="hi"), tools=_tools())
    gone = header_of(call("Gone", "2"), tools=_tools(), agent="c")
    assert echo == Header(key="1", label="Echo hi", display=_SHOWN)
    assert gone == Header(key="c: 2", label="c: Gone", display=ToolDisplay())


def test_show_result_shows_a_failure_whatever_the_display() -> None:
    printer = RecordingPrinter()
    printer.call(_ECHO)
    hidden = replace(_ECHO, display=ToolDisplay())
    show_result(printer, _result("1", "bad", is_error=True), header=hidden)
    assert printer.said == [("call", "Echo"), ("failed", "bad")]


def test_show_result_shows_an_edit_as_a_diff() -> None:
    printer = RecordingPrinter()
    edit = FileEditResult(
        call_id="1",
        path="a.py",
        edits=(Splice(before="x", after="y", start=2),),
    )
    printer.call(_ECHO)
    show_result(printer, edit, header=_ECHO)
    show_result(printer, FileEditResult(call_id="2", path="b.py"), header=_ECHO)
    assert printer.said == [
        ("call", "Echo"),
        ("diff", "a.py\n@@ -2,1 +2,1 @@\n-x\n+y"),
        ("output", "Edited b.py."),
    ]
    show_result(printer, FileEditResult(call_id="3", edits=edit.edits), header=_ECHO)
    assert printer.said[-1] == ("diff", "\n@@ -2,1 +2,1 @@\n-x\n+y")


def test_show_result_skips_a_pending_stub_but_not_a_cancelled_one() -> None:
    printer = RecordingPrinter()
    printer.call(_ECHO)
    show_result(printer, _result("1", "[detached]", stub="pending"), header=_ECHO)
    show_result(printer, _result("1", "[cancelled]", stub="cancelled"), header=_ECHO)
    assert printer.said == [("call", "Echo"), ("output", "[cancelled]")]


@pytest.mark.parametrize(
    ("text", "shown"),
    [
        ("  a\n  b", "  a\n  b"),
        ("\n<system-reminder>x</system-reminder>\n  a\n", "  a"),
        (
            "out\n<system-reminder>\nmulti\nline\n</system-reminder>\n\n\ntail",
            "out\ntail",
        ),
        (
            "<system-reminder>a</system-reminder>keep<system-reminder>b</system-reminder>",
            "keep",
        ),
        ("X\nkeep X", "X\nkeep X"),
    ],
)
def test_show_result_drops_reminders_meant_for_the_model(text: str, shown: str) -> None:
    printer = RecordingPrinter()
    printer.call(_ECHO)
    show_result(printer, ShellCommandResult(call_id="1", stdout=text), header=_ECHO)
    assert printer.said == [("call", "Echo"), ("output", shown)]


def test_error_text_keeps_a_polished_message_and_names_any_other() -> None:
    assert error_text(BudgetExhaustedError(total_cost_usd=1, max_budget_usd=1)) == (
        "Budget exhausted: $1.00 >= $1.00"
    )
    assert error_text(ValueError("bad")) == "ValueError: bad"
    assert error_text(TimeoutError()) == "TimeoutError"


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
