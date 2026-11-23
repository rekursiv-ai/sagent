"""Tests for the renderer: runtime events, as calls on a printer."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast, override
from unittest.mock import (
    Mock,
    call as mock_call,
)

import logging
import time

from treekle import PlainTree

import pytest

from sagent.model.cost import TokenCost
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
    AgentIdle,
    AgentMessage,
    AgentSpawnResult,
    AssistantMessage,
    BashResult,
    Cancelled,
    EditResult,
    Failed,
    McpResult,
    ModelIdle,
    SessionContext,
    Splice,
    Succeeded,
    SystemMessage,
    Thinking,
    ToolCall,
    ToolMessage,
    UncategorizedToolResult,
    UserMessage,
    WriteResult,
)
from sagent.tools.display import DisplaySpec, ToolDisplay
from sagent.tools.edit import Edit
from sagent.tools.glob_tool import Glob
from sagent.tools.grep import Grep
from sagent.tools.read import Read
from sagent.tools.types import ToolOutput, ToolStreamingResult, whole_output
from sagent.tools.write import Write
from sagent.tui.recording import RecordingPrinter
from sagent.tui.render import (
    Scrollback,
    call_header,
    error_text,
    show_result,
    stamped,
)
from sagent.tui.types import CommandOutput, Header, Printer


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

    def call(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> ToolStreamingResult[UncategorizedToolResult, object, object]:
        return whole_output(self._run(context, arguments))

    async def _run(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> ToolOutput[UncategorizedToolResult, object, object]:
        del context
        return ToolOutput(
            stdout=(str(arguments.get("text")),),
            result=UncategorizedToolResult(),
        )


def _tools() -> dict[str, Tool]:
    return {"Echo": _Echo()}


def _renderer(
    *,
    show_thinking: bool = True,
    flush_chars: int = 64 * 1024,
) -> tuple[Scrollback, RecordingPrinter]:
    printer = RecordingPrinter(show_thinking=show_thinking)
    return Scrollback(printer, tools=_tools(), flush_chars=flush_chars), printer


def _turn(*calls: ToolCall, text: str = "") -> types.ModelResponseCompleted:
    return types.ModelResponseCompleted(
        outcome=Succeeded(value=reply(text, calls=calls)),
    )


def _result(
    call_id: str,
    text: str = "out",
    *,
    is_error: bool = False,
    stub: Literal["pending", "cancelled"] | None = None,
) -> ToolMessage:
    if is_error:
        return ToolMessage(
            call_id=call_id,
            state="done",
            stderr=(text,),
            result=UncategorizedToolResult(exit_code=1),
        )
    return ToolMessage(
        call_id=call_id,
        state=stub or "done",
        stdout=(text,),
        result=None if stub else UncategorizedToolResult(),
    )


def _kind(value: object) -> str:
    """Name a case by its value's class; a banner's text names nothing."""
    return "" if isinstance(value, str) else type(value).__name__


# Only the runtime knows whether a turn stopped: a compaction cancels a call, a
# failure may be got past, and a halt or a clear of an agent at rest stops nothing.
def test_a_turn_that_stopped_is_noted_and_a_call_that_ended_or_a_halt_asked_is_not() -> (
    None
):
    render, printer = _renderer()
    render(types.ModelHalt())
    render(types.SessionClearCompleted(outcome=Succeeded(value=None)))
    assert printer.printed == []
    render(types.ModelResponseStarted())
    render(types.ModelResponseTextStreamed(text="partial"))
    render(types.ModelResponseCompleted(outcome=Cancelled()))
    render(types.ModelResponseTextStreamed(text="more"))
    render(
        types.ModelResponseCompleted(outcome=Failed(exception=ValueError("x"))),
    )
    assert printer.printed == [("markdown", "partial"), ("markdown", "more")]
    render(ModelIdle(outcome=Cancelled()))
    assert printer.printed[2:] == [("note", "[interrupted]")]


def test_what_a_halted_reply_had_said_is_printed_before_the_note() -> None:
    render, printer = _renderer()
    render(types.ModelResponseStarted())
    render(types.ModelResponseTextStreamed(text="so far"))
    render(types.ModelHalt())
    render(types.ModelResponseCompleted(outcome=Cancelled()))
    render(ModelIdle(outcome=Cancelled()))
    assert printer.printed == [("markdown", "so far"), ("note", "[interrupted]")]


def test_the_calls_a_halted_turn_waited_on_are_labelled_before_the_note() -> None:
    render, printer = _renderer()
    render(_turn(call("Echo", "1", text="a")))
    render(ModelIdle(outcome=Cancelled()))
    assert printer.printed == [("call", "Echo a"), ("note", "[interrupted]")]


def test_a_block_renders_once_the_next_one_starts() -> None:
    render, printer = _renderer()
    render(types.ModelResponseTextStreamed(text="first para"))
    assert printer.printed == []
    render(types.ModelResponseTextStreamed(text="\n\nsecond"))
    assert printer.printed == [("markdown", "first para")]
    render(types.ModelResponseTextStreamed(text=" para"))
    assert printer.printed == [("markdown", "first para")]


@pytest.mark.parametrize("block", ["ends X  ", "ends X"])
def test_a_finished_block_keeps_all_but_its_final_newlines(block: str) -> None:
    render, printer = _renderer()
    render(types.ModelResponseTextStreamed(text=f"{block}\n\n\nnext"))
    assert printer.printed == [("markdown", block)]


@pytest.mark.parametrize(
    ("held", "shown"),
    [("tail X  \n\n", "tail X  "), ("tail X  ", "tail X  "), ("tail X\n", "tail X")],
)
def test_what_is_held_renders_when_anything_else_shows(held: str, shown: str) -> None:
    render, printer = _renderer()
    render(types.ModelResponseTextStreamed(text=held))
    render(UserMessage(content="next"))
    assert printer.printed == [("markdown", shown), ("user", "next")]


def test_nothing_held_renders_nothing() -> None:
    render, printer = _renderer()
    render(types.ModelResponseTextStreamed(text="\n\n"))
    render(types.SessionCompactStarted())
    assert printer.printed == [("note", "[compacting history…]")]


def test_blank_lines_between_blocks_render_once() -> None:
    render, printer = _renderer()
    render(types.ModelResponseTextStreamed(text="a\n\n- b"))
    render(types.ModelResponseTextStreamed(text="\n\n- c\n\n\n\nd"))
    assert printer.printed == [("markdown", "a"), ("markdown", "- b\n\n- c")]


def test_held_prose_past_the_cap_renders_only_its_whole_lines() -> None:
    render, printer = _renderer(flush_chars=10)
    render(types.ModelResponseTextStreamed(text="line one\nline tw"))
    assert printer.printed == [("markdown", "line one")]
    render(types.ModelResponseTextStreamed(text="o\n\nnext"))
    assert printer.printed == [("markdown", "line one"), ("markdown", "line two")]


@pytest.mark.parametrize("tail", ["", "z"], ids=["empty-tail", "held-tail"])
def test_held_text_past_the_cap_renders_unfinished(tail: str) -> None:
    markdown, note = Mock(), Mock()
    printer = Mock(spec=Printer, markdown=markdown, note=note, show_thinking=False)
    render = Scrollback(printer, tools={}, flush_chars=4)
    render(types.ModelResponseTextStreamed(text="```\n"))
    markdown.assert_not_called()
    render(types.ModelResponseTextStreamed(text="y"))
    markdown.assert_called_once_with("```\ny")
    render(types.ModelResponseTextStreamed(text=tail))
    markdown.assert_called_once_with("```\ny")
    render(types.SessionCompactStarted())
    assert markdown.call_args_list == [
        mock_call("```\ny"),
        *([mock_call(tail)] if tail else []),
    ]
    note.assert_called_once_with("[compacting history…]")


@pytest.mark.parametrize(
    "event",
    [
        ModelIdle(outcome=Succeeded(value=None)),
        types.ModelHalt(),
        types.SessionClearCompleted(outcome=Succeeded(value=None)),
        AgentIdle(),
        types.ToolCohortStarted(),
        types.ModelResponseStarted(),
        types.SessionSaveDue(),
    ],
    ids=_kind,
)
def test_events_with_nothing_to_show_neither_print_nor_flush(
    event: types.RuntimeEvent,
) -> None:
    render, printer = _renderer()
    render(types.ModelResponseTextStreamed(text="held"))
    render(event)
    assert printer.printed == []


def test_thinking_shows_only_while_the_printer_shows_it() -> None:
    render, printer = _renderer()
    printer.show_thinking = False
    render(types.ModelResponseThinkingStreamed(text="hidden"))
    render(_turn(text="done"))
    assert printer.printed == [("markdown", "done")]


def test_thinking_heads_once_then_streams_by_block_and_closes_once() -> None:
    render, printer = _renderer()
    for word in ("\n", "first ", "para", "\n\nsecond", " para"):
        render(types.ModelResponseThinkingStreamed(text=word))
    assert printer.printed == [("thinking", "["), ("thinking", "\nfirst para\n\n")]
    render(types.ModelResponseTextStreamed(text="reply"))
    assert printer.printed[2:] == [("thinking", "second para]")]


def test_thinking_a_notice_interrupts_reopens_under_a_new_heading() -> None:
    render, printer = _renderer()
    render(types.ModelResponseThinkingStreamed(text="before\n\n"))
    render(types.SessionCompactStarted())
    render(types.ModelResponseThinkingStreamed(text="after"))
    render(types.ModelResponseTextStreamed(text="reply"))
    thinking = [s for k, s in printer.printed if k == "thinking"]
    assert thinking.count("[") == 2
    assert sum(s.endswith("]") for s in thinking) == 2


def test_word_sized_thinking_deltas_print_one_heading_and_the_joined_text() -> None:
    printer = RecordingPrinter(show_thinking=True, width=200)
    render = Scrollback(printer, tools=_tools())
    words = [f"word{i} " for i in range(20)]
    for word in words:
        render(types.ModelResponseThinkingStreamed(text=word))
    render(types.ModelResponseTextStreamed(text="Answer.", response_record_index=1))
    response = reply("Answer.")
    render(
        types.ModelResponseCompleted(
            outcome=Succeeded(
                value=type(response)(
                    records=(Thinking(content="".join(words)), *response.records),
                    tokens=response.tokens,
                    spend=response.spend,
                    stop_reason="end_turn",
                ),
            ),
        ),
    )
    rows = [row.rstrip() for row in printer.text.split("\n")]
    assert rows == ["∴ Thinking", f"  {''.join(words).rstrip()}", "", "Answer.", ""]


def test_hidden_thinking_does_not_hold_back_the_reply_after_it() -> None:
    render, printer = _renderer()
    printer.show_thinking = False
    render(types.ModelResponseThinkingStreamed(text="hidden"))
    render(types.ModelResponseTextStreamed(text="a\n\nb", response_record_index=1))
    assert printer.printed == [("markdown", "a")]


def test_text_after_a_call_waits_to_print_after_the_call() -> None:
    render, printer = _renderer()
    render(types.ModelResponseTextStreamed(text="Before.\n\nheld"))
    render(
        types.ModelResponseTextStreamed(text="After.\n\nnext", response_record_index=2),
    )
    assert printer.printed == [("markdown", "Before.")]
    response = reply("Before.\n\nheld", calls=[call("Echo", "1", text="a")])
    render(
        types.ModelResponseCompleted(
            outcome=Succeeded(
                value=type(response)(
                    records=(
                        *response.records,
                        AssistantMessage(content="After.\n\nnext"),
                    ),
                    tokens=response.tokens,
                    spend=response.spend,
                    stop_reason="tool_use",
                ),
            ),
        ),
    )
    assert printer.printed == [
        ("markdown", "Before."),
        ("markdown", "held"),
        ("call", "Echo a"),
        ("markdown", "After.\n\nnext"),
    ]


def test_a_blank_thinking_stream_prints_nothing() -> None:
    render, printer = _renderer()
    render(types.ModelResponseThinkingStreamed(text=" \n"))
    render(_turn(text="done"))
    assert printer.printed == [("markdown", "done")]


def test_streamed_thinking_is_not_printed_again_by_its_record() -> None:
    render, printer = _renderer()
    render(types.ModelResponseThinkingStreamed(text="hm"))
    response = reply("x")
    render(
        types.ModelResponseCompleted(
            outcome=Succeeded(
                value=type(response)(
                    records=(Thinking(content="hm"), *response.records),
                    tokens=response.tokens,
                    spend=response.spend,
                    stop_reason="end_turn",
                ),
            ),
        ),
    )
    assert printer.printed == [
        ("thinking", "["),
        ("thinking", "hm]"),
        ("markdown", "x"),
    ]


def test_thinking_ends_the_reply_being_streamed_and_titles_do_not() -> None:
    render, printer = _renderer()
    render(types.ModelResponseTextStreamed(text="held"))
    render(types.SessionTitleChanged(title="working"))
    assert printer.printed == []
    render(types.ModelResponseThinkingStreamed(text="hm"))
    assert printer.printed == [("markdown", "held"), ("thinking", "[")]


def test_an_unstreamed_turn_prints_its_reasoning_and_reply_from_its_records() -> None:
    render, printer = _renderer()
    response = reply("**done**\n", calls=[call("Echo", "1", text="a")])
    render(
        types.ModelResponseCompleted(
            outcome=Succeeded(
                value=type(response)(
                    records=(Thinking(content="because"), *response.records),
                    tokens=response.tokens,
                    spend=response.spend,
                    stop_reason="tool_use",
                ),
            ),
        ),
    )
    render(types.ToolCohortStarted())
    assert printer.printed == [
        ("thinking", "[because]"),
        ("markdown", "**done**"),
        ("call", "Echo a"),
    ]


def test_command_output_is_a_note_or_an_error() -> None:
    render, printer = _renderer()
    render.show(types.ModelResponseTextStreamed(text="held"))
    render.show(CommandOutput(text="a", tone="note", command="/model"))
    render.show(CommandOutput(text="no such", tone="error"))
    assert printer.printed == [
        ("markdown", "held"),
        ("note", "[/model] a"),
        ("error", "no such"),
    ]


def _context_put_in_force(
    skipped: tuple[Path, ...],
) -> types.SessionContextUpdateCompleted:
    return types.SessionContextUpdateCompleted(
        outcome=Succeeded(
            value=SessionContext(skipped_instruction_files=skipped),
        ),
    )


def test_each_file_a_context_skips_is_an_error_the_user_sees() -> None:
    render, printer = _renderer()
    render.show(_context_put_in_force((Path("a.md"), Path("b.md"))))
    render.show(_context_put_in_force(()))
    render.show(types.SessionContextUpdateCompleted(outcome=Cancelled()))
    assert printer.printed == [
        ("error", "Skipping a.md: see the log for why."),
        ("error", "Skipping b.md: see the log for why."),
    ]


def test_a_note_that_drives_the_model_is_shown_as_the_model_read_it() -> None:
    render, printer = _renderer()
    render(types.ModelResponseTextStreamed(text="held"))
    render(
        SystemMessage(
            content="[talk t1 still running after 15s]\none",
            subtype="nudge",
        ),
    )
    assert printer.printed == [
        ("markdown", "held"),
        ("note", "[talk t1 still running after 15s]\none"),
    ]


@pytest.mark.parametrize("subtype", ["rules", "reminder", None])
def test_a_note_that_only_informs_the_model_is_not_shown(subtype: str | None) -> None:
    render, printer = _renderer()
    render.show(SystemMessage(content="cwd: /x", subtype=subtype))
    assert printer.printed == []


def test_a_skipped_file_is_drawn_when_its_context_is_put_in_force() -> None:
    render, printer = _renderer()
    render(_context_put_in_force((Path("a.md"),)))
    assert printer.printed == [("error", "Skipping a.md: see the log for why.")]


class _Stranger:
    """No event or record the renderer knows, as a type added and not drawn is."""

    call_id: str = "1"
    stub: str | None = None


@pytest.mark.parametrize(
    "given",
    [
        cast("Shown", _Stranger()),
        # A record arrives by a known event, and fails where records are drawn.
        types.ModelResponseCompleted(
            outcome=Succeeded(
                value=replace(reply(), records=(cast("ToolCall", _Stranger()),)),
            ),
        ),
        CommandOutput(text="x", tone=cast("Literal['line']", _Stranger.__name__)),
        types.SessionCompactCompleted(
            outcome=cast("Cancelled", _Stranger()),
            tape_len_before=1,
        ),
    ],
    ids=["event", "record", "tone", "compaction"],
)
def test_what_the_renderer_does_not_know_fails_rather_than_draws(given: Shown) -> None:
    render, printer = _renderer()
    with pytest.raises(AssertionError, match="_Stranger"):
        render.show(given)
    assert printer.printed == []


def test_a_child_says_its_text_and_calls_under_its_label() -> None:
    render, printer = _renderer()
    turn = _turn(call("Echo", text="hi"), call("Mystery"), text="looking")
    render(types.ChildEvent(label="Agent_0", event=turn))
    assert printer.printed == [
        ("note", "Agent_0: looking"),
        ("call", "Agent_0: Echo hi"),
        ("call", "Agent_0: Mystery"),
    ]
    assert printer.displays == [_SHOWN, ToolDisplay()]


def test_a_child_s_blank_text_says_nothing() -> None:
    render, printer = _renderer()
    response = reply("x")
    blank = types.ModelResponseCompleted(
        outcome=Succeeded(
            value=type(response)(
                records=(
                    AssistantMessage(content="  "),
                    AssistantMessage(),
                    Thinking(content="t"),
                ),
                tokens=response.tokens,
                spend=response.spend,
                stop_reason="end_turn",
            ),
        ),
    )
    render(types.ChildEvent(label="c", event=blank))
    assert printer.printed == []


def test_a_grandchild_is_shown_under_its_own_label() -> None:
    render, printer = _renderer()
    inner = types.ChildEvent(
        label="Agent_0_0",
        event=ModelIdle(outcome=Failed(exception=ValueError("x"))),
    )
    render(types.ChildEvent(label="Agent_0", event=inner))
    assert printer.printed == [("error", "Agent_0_0: ValueError: x")]


def test_a_child_s_results_show_only_when_they_fail() -> None:
    render, printer = _renderer()
    turn = _turn(call("Echo", "1", text="a"), call("Echo", "2", text="b"))
    render(types.ChildEvent(label="c", event=turn))
    render(types.ChildEvent(label="c", event=_result("1", "fine", is_error=False)))
    failed = _result("2", "bad<system-reminder>r</system-reminder>\n", is_error=True)
    render(types.ChildEvent(label="c", event=failed))
    assert printer.printed == [
        ("call", "c: Echo a"),
        ("call", "c: Echo b"),
        ("failed", "c: bad"),
    ]


def _rows(printer: RecordingPrinter) -> list[str]:
    return [row.rstrip() for row in printer.text.splitlines() if row.strip()]


def test_a_child_s_failure_names_its_own_call_again() -> None:
    render, printer = _renderer()
    turn = _turn(call("Echo", "1", text="a"), call("Echo", "2", text="b"))
    render(types.ChildEvent(label="c", event=turn))
    render(types.ChildEvent(label="c", event=_result("1", "broke", is_error=True)))
    assert _rows(printer) == [
        "  c: Echo a",
        "  c: Echo b",
        "  c: Echo a",
        "    ✗ c: broke",
    ]


def test_a_child_s_failure_right_under_its_call_does_not_name_it_again() -> None:
    render, printer = _renderer()
    render(types.ChildEvent(label="c", event=_turn(call("Echo", "1", text="a"))))
    render(types.ChildEvent(label="c", event=_result("1", "broke", is_error=True)))
    assert _rows(printer) == ["  c: Echo a", "    ✗ c: broke"]


def test_a_result_whose_call_is_unknown_sits_under_no_call() -> None:
    render, printer = _renderer()
    render(_turn(call("Echo", "1", text="a")))
    render(_result("1", "said a"))
    render(_result("9", "orphan"))
    assert render.shown == 1
    render(_result("9", "bad", is_error=True))
    assert render.shown == 2
    render(types.ChildEvent(label="c", event=_result("8", "worse", is_error=True)))
    assert printer.printed == [
        ("call", "Echo a"),
        ("output", "said a"),
        ("error", "bad"),
        ("error", "c: worse"),
    ]
    assert render.shown == 2


def test_a_result_that_shows_nothing_names_no_call_again() -> None:
    render, printer = _renderer()
    render(_turn(call("Hidden", "1"), call("Hidden", "2"), call("Echo", "3", text="c")))
    render(types.ToolCohortStarted())
    for call_id in ("1", "2"):
        render(_result(call_id, "x"))
    render(_result("3", "said c"))
    assert _rows(printer) == ["  Hidden", "  Hidden", "  Echo c", "    said c"]


def test_a_hidden_result_between_a_call_and_its_failure_names_no_call_again() -> None:
    render, printer = _renderer()
    render(_turn(call("Other", "1"), call("Echo", "2", text="b")))
    render(types.ToolCohortStarted())
    render(_result("1", "hidden"))
    render(_result("2", "bad", is_error=True))
    assert printer.printed == [
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
    assert printer.printed == [
        ("note", "x"),
        ("output", "hidden"),
        ("call", "Echo"),
        ("failed", "bad"),
    ]


@pytest.mark.parametrize("writer", ["agent", "line", "note", "error"])
def test_a_result_below_a_row_printed_past_scrollback_names_its_call_again(
    writer: str,
) -> None:
    render, printer = _renderer()
    render(_turn(call("Echo", "1", text="a")))
    render(types.ToolCohortStarted())
    match writer:
        case "agent":
            printer.agent("Agent_0", "user", "a question")
        case _:
            getattr(printer, writer)("x")
    render(_result("1", "said a"))
    assert _rows(printer)[-2:] == ["  Echo a", "    said a"]


def test_a_child_s_failures_messages_and_suspensions_are_labelled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(time, "time", lambda: _NOON)
    render, printer = _renderer()
    failed = Failed(exception=ValueError("x"))
    # A call of its that failed is its own business: its turn may go on.
    render(
        types.ChildEvent(
            label="c",
            event=types.ModelResponseCompleted(outcome=failed),
        ),
    )
    render(types.ChildEvent(label="c", event=ModelIdle(outcome=failed)))
    render(
        types.ChildEvent(
            label="c",
            event=AgentMessage(sender="me", content="hi"),
        ),
    )
    render(types.ChildEvent(label="c", event=AgentMessage()))
    suspended = types.ModelResponseSuspended(
        exception=RateLimitedError("spent", limit="weekly"),
        retry_at=datetime.fromtimestamp(_NOON + 5.0, UTC),
        failures_in_a_row=1,
    )
    render(types.ChildEvent(label="c", event=suspended))
    render(types.ChildEvent(label="c", event=UserMessage(content="ignored")))
    assert printer.printed == [
        ("error", "c: ValueError: x"),
        ("agent", "me→c: hi"),
        ("agent", "agent→c: "),
        ("error", "c: Weekly limit reached. Resumes Wed Oct 28 12:00:05 (5s)."),
    ]


def test_a_child_s_work_does_not_end_the_reply_being_streamed() -> None:
    render, printer = _renderer()
    render(types.ModelResponseTextStreamed(text="held"))
    render(
        types.ChildEvent(
            label="c",
            event=ModelIdle(outcome=Failed(exception=ValueError("x"))),
        ),
    )
    assert printer.printed == [("error", "c: ValueError: x")]


@pytest.mark.parametrize(
    ("spawned", "note"),
    [
        (
            AgentSpawnResult(
                agent_id="c",
                tokens=12_345,
                spend=TokenCost(request=0.426),
                duration_sec=65.0,
            ),
            "[c done: 1m 5s, 12K tokens, $0.43]",
        ),
        (AgentSpawnResult(agent_id="c"), "[c done: 0s, 0 tokens, $0.00]"),
    ],
    ids=["used", "unreported"],
)
def test_a_spawn_s_result_says_what_its_child_used(
    spawned: AgentSpawnResult,
    note: str,
) -> None:
    render, printer = _renderer()
    render(_turn(call("AgentSpawn", "1", prompt="go")))
    render(
        ToolMessage(call_id="1", state="done", stdout=("said",), result=spawned),
    )
    assert printer.printed[-1] == ("note", note)


def test_a_spawn_another_harness_logged_says_nothing_of_a_child() -> None:
    render, printer = _renderer()
    render(_turn(call("AgentSpawn", "1", prompt="go")))
    render(
        ToolMessage(
            call_id="1",
            state="done",
            stdout=("said",),
            result=AgentSpawnResult(prompt="go"),
        ),
    )
    assert [kind for kind, _ in printer.printed if kind == "note"] == []


def test_messages_show_as_bars() -> None:
    render, printer = _renderer()
    render(UserMessage(content="hi"))
    render(UserMessage())
    render(AgentMessage(sender="Agent_0", content="done"))
    render(AgentMessage(sender="watcher", recipient="Agent", content="ok"))
    render(AgentMessage())
    render(AgentMessage(sender="user", recipient="Agent", content="ping"))
    assert printer.printed == [
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
    assert printer.printed == [
        ("markdown", "unstreamed"),
        ("call", "Echo a"),
        ("call", "Other"),
        ("output", "said b"),
    ]
    assert printer.displays == [_SHOWN, ToolDisplay(), ToolDisplay()]


def test_a_call_prints_when_it_starts_before_any_result() -> None:
    render, printer = _renderer()
    render(_turn(call("Echo", "1", text="a"), call("Other", "2")))
    assert printer.printed == []
    render(types.ToolCohortStarted())
    assert printer.printed == [("call", "Echo a"), ("call", "Other")]


@pytest.mark.parametrize(
    ("between", "said"),
    [
        (types.SessionCompactStarted(), ("note", "[compacting history…]")),
        (
            types.ChildEvent(
                label="kid",
                event=ModelIdle(
                    outcome=Failed(exception=ValueError("x")),
                ),
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
    assert printer.printed == [
        ("call", "Echo a"),
        said,
        ("call", "Echo a"),
        ("output", "said a"),
    ]


def test_a_result_after_its_turn_s_closing_text_names_its_call_again() -> None:
    render, printer = _renderer()
    response = reply("", calls=[call("Echo", "1", text="a")])
    render(
        types.ModelResponseCompleted(
            outcome=Succeeded(
                value=type(response)(
                    records=(*response.records, AssistantMessage(content="Running.")),
                    tokens=response.tokens,
                    spend=response.spend,
                    stop_reason="tool_use",
                ),
            ),
        ),
    )
    render(_result("1", "said a"))
    assert printer.printed == [
        ("call", "Echo a"),
        ("markdown", "Running."),
        ("call", "Echo a"),
        ("output", "said a"),
    ]


def test_a_detached_result_shows_like_any_other() -> None:
    render, printer = _renderer()
    render(_turn(call("Echo", "1")))
    render(_result("1", "late"))
    assert printer.printed[-1] == ("output", "late")
    assert printer.displays[-1] == _SHOWN


def test_a_detached_call_is_numbered_when_detached_and_again_when_its_result_lands() -> (
    None
):
    render, printer = _renderer()
    render.show(call("Echo", "1", text="a"))
    render.show(_result("1", "[detached]", stub="pending"))
    render.show(call("Echo", "2", text="b"))
    render.show(_result("2", "now"))
    render.show(_result("1", "late"))
    assert printer.printed == [
        ("call", "Echo a"),
        ("output", "[detached; Echo#1]"),
        ("call", "Echo b"),
        ("output", "now"),
        ("call", "Echo a"),
        ("output", "[undetached; Echo#1]"),
        ("output", "late"),
    ]
    assert render.shown == 3


def test_a_wake_shows_as_its_call_s_output_named_by_its_job() -> None:
    render, printer = _renderer()
    render(_turn(call("Echo", "c9", text="a")))
    render(_result("c9", "[detached]", stub="pending"))
    render(types.ModelResponseStarted())
    render(UserMessage(content="meanwhile"))
    render(
        ToolMessage(
            call_id="c9",
            state="pending",
            stdout=("[Echo c9 still running after 30s]\nbeat 4\nbeat 5",),
        ),
    )
    assert printer.printed == [
        ("call", "Echo a"),
        ("output", "[detached; Echo#1]"),
        ("user", "meanwhile"),
        ("call", "Echo a"),
        ("output", "[Echo#1 still running after 30s]"),
        ("output", "beat 4\nbeat 5"),
    ]
    assert printer.displays[-2:] == [_SHOWN, _tools()["Echo"].display]


def test_a_wake_names_its_call_once_and_labels_what_it_wrote_when_it_has_no_head() -> (
    None
):
    render, printer = _renderer()
    render(_turn(call("Echo", "c9", text="a")))
    render(_result("c9", "[detached]", stub="pending"))
    render(types.ModelResponseStarted())
    render(UserMessage(content="one"))
    # Only the note's own naming of the call is its job's: a second is the call's words.
    render(
        ToolMessage(call_id="c9", state="pending", stdout=("[Echo c9 Echo c9 ]\nx",)),
    )
    render(UserMessage(content="two"))
    render(ToolMessage(call_id="c9", state="pending", stdout=("\nbeat 4",)))
    assert printer.printed[2:] == [
        ("user", "one"),
        ("call", "Echo a"),
        ("output", "[Echo#1 Echo c9 ]"),
        ("output", "x"),
        ("user", "two"),
        # An empty head draws no row, so what the call wrote is what labels it.
        ("output", ""),
        ("call", "Echo a"),
        ("output", "beat 4"),
    ]


def test_a_detached_call_that_ends_cancelled_is_not_said_to_be_undetached() -> None:
    render, printer = _renderer()
    render(_turn(call("Echo", "1", text="a")))
    render(types.ToolCohortStarted())
    render(_result("1", "[detached]", stub="pending"))
    render(_result("1", "stopped", stub="cancelled"))
    assert [text for _, text in printer.printed if "detached" in text] == [
        "[detached; Echo#1]",
    ]
    assert render.shown == 2


def test_a_wake_for_a_call_scrollback_never_labelled_is_a_note() -> None:
    render, printer = _renderer()
    note = "[Echo c9 still running after 30s]\nbeat 4"
    render.show(ToolMessage(call_id="c9", state="pending", stdout=(note,)))
    assert printer.printed == [("note", note)]


def test_a_second_detached_call_takes_the_next_number() -> None:
    render, printer = _renderer()
    render(_turn(call("Echo", "1", text="a"), call("Echo", "2", text="b")))
    render(_result("1", "[detached]", stub="pending"))
    render(_result("2", "[detached]", stub="pending"))
    render(types.ModelResponseStarted())
    assert printer.printed == [
        ("call", "Echo a"),
        ("output", "[detached; Echo#1]"),
        ("call", "Echo b"),
        ("output", "[detached; Echo#2]"),
    ]


def test_a_backgrounded_call_beside_a_running_one_is_labelled_once() -> None:
    render, printer = _renderer()
    render(_turn(call("Echo", "1", text="slow"), call("Echo", "2", text="fast")))
    render(_result("1", "[Running in background: Echo as job 1]", stub="pending"))
    render(types.ToolCohortStarted())
    render(_result("2", "fast done"))
    render(_result("1", "slow done"))
    assert printer.printed == [
        ("call", "Echo slow"),
        ("output", "[detached; Echo#1]"),
        ("call", "Echo fast"),
        ("output", "fast done"),
        ("call", "Echo slow"),
        ("output", "[undetached; Echo#1]"),
        ("output", "slow done"),
    ]
    assert printer.displays[1] == printer.displays[5] == _SHOWN


def test_a_call_detached_while_it_runs_is_numbered_under_its_label() -> None:
    render, printer = _renderer()
    render(_turn(call("Echo", "1", text="a")))
    render(types.ToolCohortStarted())
    render(_result("1", "[detached]", stub="pending"))
    assert printer.printed == [("call", "Echo a"), ("output", "[detached; Echo#1]")]
    assert printer.displays[-1] == _SHOWN


def test_a_person_who_is_not_the_session_s_user_is_shown_by_name() -> None:
    render, printer = _renderer()
    render(UserMessage(sender="bob", content="hi"))
    render(UserMessage(content="mine"))
    assert printer.printed == [("user", "[from user bob]: hi"), ("user", "mine")]


def test_a_reply_that_says_nothing_leaves_the_last_turn_s_calls_held() -> None:
    """A turn's records say for themselves whether they end the labels; none do."""
    render, printer = _renderer()
    render(_turn(call("Echo", "1", text="a")))
    render(_turn())
    assert printer.printed == []
    render(types.ToolCohortStarted())
    assert printer.printed == [("call", "Echo a")]


def test_a_stub_for_an_earlier_call_labels_the_held_ones_first() -> None:
    render, printer = _renderer()
    render(UserMessage(content="go"))
    render(_turn(call("Echo", "1", text="a")))
    render(types.ToolCohortStarted())
    render(_turn(call("Echo", "2", text="b")))
    render(_result("1", "[detached]", stub="pending"))
    assert render.shown == 2
    assert printer.printed == [
        ("user", "go"),
        ("call", "Echo a"),
        ("call", "Echo b"),
        ("call", "Echo a"),
        ("output", "[detached; Echo#1]"),
    ]


def test_scrollback_names_paths_below_where_sagent_started() -> None:
    printer = RecordingPrinter()
    render = Scrollback(printer, tools={"Read": Read()}, home=Path("/r"))
    render(_turn(call("Read", "1", file_path="/r/a.py")))
    render(_result("1", "[detached]", stub="pending"))
    render(types.ToolCohortStarted())
    render(types.SessionCompactStarted())
    render(_result("1", "x"))
    render(
        types.ChildEvent(label="kid", event=_turn(call("Read", file_path="/r/b.py"))),
    )
    assert [label for kind, label in printer.printed if kind == "call"] == [
        "Read a.py",
        "Read a.py",
        "kid: Read b.py",
    ]


def test_a_user_message_that_merely_mentions_a_delivery_still_shows() -> None:
    render, printer = _renderer()
    render.show(UserMessage(content="see [detached tool 1 completed] above"))
    assert printer.printed == [("user", "see [detached tool 1 completed] above")]


def test_a_write_shows_as_a_diff_cut_like_its_tool_s_output() -> None:
    printer = RecordingPrinter()
    printer.call(_ECHO)
    write = ToolMessage(
        call_id="1",
        state="done",
        stdout=("Wrote a.py.",),
        result=WriteResult(path="a.py", content="x\ny\n"),
    )
    cut = ToolDisplay(output=DisplaySpec(lines=(1, 1)))
    show_result(printer, write, header=replace(_ECHO, display=cut))
    show_result(printer, write, header=_ECHO)
    show_result(printer, write, header=replace(_ECHO, display=ToolDisplay()))
    show_result(
        printer,
        ToolMessage(
            call_id="2",
            state="done",
            stdout=("Wrote b.py.",),
            result=UncategorizedToolResult(),
        ),
        header=_ECHO,
    )
    assert printer.printed == [
        ("call", "Echo"),
        ("diff", "a.py\n@@ -0,0 +1,2 @@\n+x\n+y"),
        ("diff", "a.py\n@@ -0,0 +1,2 @@\n+x\n+y"),
        ("output", "Wrote a.py."),
        ("output", "Wrote b.py."),
    ]
    assert printer.cuts == [(1, 1), None]


def test_cancellations_and_compactions_are_notes() -> None:
    render, printer = _renderer()
    render(types.SessionCompactStarted())
    render(
        types.SessionCompactCompleted(
            outcome=Succeeded(
                value=types.SessionCompacted(
                    masked_tokens=900,
                    replacement_tokens=100,
                    replacement_entries=4,
                ),
            ),
        ),
    )
    render(
        types.SessionCompactCompleted(
            outcome=Succeeded(
                value=types.SessionCompacted(
                    summary_fallback_reason="too big",
                    entries_kept_verbatim=1,
                ),
            ),
        ),
    )
    render(
        types.SessionCompactCompleted(
            outcome=Succeeded(
                value=types.SessionCompacted(
                    summary_fallback_reason="too big",
                    entries_kept_verbatim=3,
                ),
            ),
        ),
    )
    render(
        types.SessionCompactCompleted(
            outcome=Failed(exception=TimeoutError()),
            tape_len_before=1,
        ),
    )
    render(
        types.SessionCompactCompleted(
            outcome=Failed(
                exception=BudgetExhaustedError(total_cost_usd=2, max_budget_usd=1),
            ),
            tape_len_before=1,
        ),
    )
    render(types.SessionCompactCompleted(outcome=Cancelled(), tape_len_before=1))
    assert printer.printed == [
        ("note", "[compacting history…]"),
        ("note", "[compaction complete: ~900 → ~100 tokens, 4 entries]"),
        ("note", "[compaction fallback: too big; preserved 1 tail entry]"),
        ("note", "[compaction fallback: too big; preserved 3 tail entries]"),
        ("note", "[compaction failed: TimeoutError]"),
        ("note", "[compaction failed: Budget exhausted: $2.00 >= $1.00]"),
        ("note", "[compaction cancelled]"),
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
def test_a_turn_that_failed_shows_why_and_says_what_the_user_can_do(
    error: Exception,
    banner: str,
) -> None:
    render, printer = _renderer()
    render(ModelIdle(outcome=Failed(exception=error)))
    assert printer.printed == [
        ("error", f"{type(error).__name__}: {error}"),
        ("halt", banner),
    ]


def test_a_child_s_long_reply_hangs_under_its_label() -> None:
    """A child's words wrapped back to column 0, under its label, not its text."""
    printer = RecordingPrinter(width=30)
    render = Scrollback(printer, tools=_tools())
    render(types.ChildEvent(label="Agent_0", event=_turn(text="word " * 12)))
    rows = _rows(printer)
    assert len(rows) > 1
    assert rows[0].startswith("Agent_0: word")
    assert all(r.startswith(" " * 9) and r[9] != " " for r in rows[1:]), rows
    # Rows break between words, as a model's do, never inside one.
    assert {w for r in rows for w in r.split()} == {"Agent_0:", "word"}, rows


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
        render(types.SessionCompactStarted())
    assert caplog.messages == ["rendering SessionCompactStarted failed"]


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
    render(
        types.ModelResponseSuspended(
            exception=error,
            retry_at=datetime.fromtimestamp(_NOON + wait_sec, UTC),
            failures_in_a_row=1,
        ),
    )
    assert printer.printed == [("error", reason), ("halt", _WAITING)]


@pytest.mark.parametrize(
    ("switched", "note"),
    [
        (
            types.LoginSwitched(
                provider="acli",
                from_login="default",
                to_login="personal",
                spent_window="5-hour",
                spent_window_resets_at=datetime.fromtimestamp(_NOON + 3_600.0, UTC),
            ),
            "[default spent (5-hour, resets Wed Oct 28 13:00:00 (1h 0m 0s)); now on personal]",
        ),
        (
            types.LoginSwitched(
                provider="acli",
                from_login="work",
                to_login="default",
                first_login_in_order="default",
            ),
            "[on default, first in order]",
        ),
        (
            types.LoginSwitched(
                provider="acli",
                from_login="work",
                to_login="personal",
                first_login_in_order="default",
            ),
            "[on personal, first with quota]",
        ),
        (
            types.LoginSwitched(
                provider="acli",
                from_login="work",
                to_login="default",
                spent_window="signed out",
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
    assert printer.printed == [("note", note)]


def test_every_login_spent_says_when_each_resets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(time, "time", lambda: _NOON)
    render, printer = _renderer()
    render(
        types.ModelResponseSuspended(
            exception=RateLimitedError("spent", limit="5-hour"),
            retry_at=datetime.fromtimestamp(_NOON + 60.0, UTC),
            failures_in_a_row=1,
            provider="acli",
            reset_at_by_login={
                "work": datetime.fromtimestamp(_NOON + 60.0, UTC),
                "default": datetime.fromtimestamp(_NOON + 3_600.0, UTC),
            },
        ),
    )
    assert printer.printed[0] == (
        "note",
        (
            "[no acli login has quota: work Wed Oct 28 12:01:00 (1m 0s),"
            " default Wed Oct 28 13:00:00 (1h 0m 0s)]"
        ),
    )


def test_a_child_s_login_switch_is_noted_under_its_label() -> None:
    render, printer = _renderer()
    switched = types.LoginSwitched(
        provider="acli",
        from_login="work",
        to_login="default",
        first_login_in_order="default",
    )
    render(types.ChildEvent(label="Agent_0", event=switched))
    spent = types.ModelResponseSuspended(
        exception=RateLimitedError("spent", limit="5-hour"),
        retry_at=datetime.fromtimestamp(_NOON, UTC),
        failures_in_a_row=1,
        provider="acli",
        reset_at_by_login={"work": datetime.fromtimestamp(_NOON, UTC)},
    )
    render(types.ChildEvent(label="Agent_0", event=spent))
    assert printer.printed[0] == ("note", "Agent_0: [on default, first in order]")
    assert printer.printed[1][1].startswith("Agent_0: [no acli login has quota: work ")


def test_a_suspension_failing_again_says_why_without_the_banner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A backoff of sixty tries drew sixty banners."""
    monkeypatch.setattr(time, "time", lambda: _NOON)
    render, printer = _renderer()
    render(
        types.ModelResponseSuspended(
            exception=OverloadedError("busy"),
            retry_at=datetime.fromtimestamp(_NOON + 10.0, UTC),
            failures_in_a_row=2,
        ),
    )
    assert printer.printed == [
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
    header = call_header(
        ToolCall(call_id="1", name=tool.name, arguments=arguments),
        tools={tool.name: tool},
        home=Path("/home/u/repo"),
    )
    assert header.label == label


def test_a_header_is_its_tool_s_label_and_display_or_the_call_s_name() -> None:
    echo = call_header(call("Echo", "1", text="hi"), tools=_tools())
    gone = call_header(call("Gone", "2"), tools=_tools(), agent="c")
    assert echo == Header(key="1", label="Echo hi", display=_SHOWN)
    assert gone == Header(key="c: 2", label="c: Gone", display=ToolDisplay())


def test_a_stamped_call_keeps_its_label_and_display_beside_what_it_had() -> None:
    echo = replace(call("Echo", "1", text="hi"), extra={"kept": 1})
    assert stamped(echo, tools=_tools()).extra == {
        "kept": 1,
        "$summary": "Echo hi",
        "$display": {"command": {"lines": None}, "output": {"lines": None}},
    }


def test_a_stamped_call_names_a_path_below_where_sagent_started_relative_to_it() -> (
    None
):
    read = ToolCall(call_id="1", name="Read", arguments={"file_path": "/h/r/a.py"})
    shown = stamped(read, tools={"Read": Read()}, home=Path("/h/r"))
    assert shown.extra["$summary"] == "Read a.py"


def test_a_call_of_an_unknown_tool_is_stamped_with_nothing() -> None:
    gone = call("Gone", "2")
    assert stamped(gone, tools=_tools()) is gone


def test_a_tool_that_printed_and_failed_shows_as_a_failure_and_a_command_as_output() -> (
    None
):
    """A shell that ran a failing command did its job: what it printed is output."""
    printer = RecordingPrinter()
    printer.call(_ECHO)
    failed = ToolMessage(
        call_id="1",
        state="done",
        stdout=("no such page",),
        result=McpResult(server="srv", tool="get", exit_code=1),
    )
    exited = ToolMessage(
        call_id="2",
        state="done",
        result=BashResult(command=("bash", "-c", "false"), exit_code=1),
    )
    show_result(printer, failed, header=_ECHO)
    show_result(printer, exited, header=_ECHO)
    assert printer.printed == [
        ("call", "Echo"),
        ("failed", "no such page"),
        ("output", "[exit 1]"),
    ]


def test_show_result_shows_a_failure_whatever_the_display() -> None:
    printer = RecordingPrinter()
    printer.call(_ECHO)
    hidden = replace(_ECHO, display=ToolDisplay())
    show_result(printer, _result("1", "bad", is_error=True), header=hidden)
    assert printer.printed == [("call", "Echo"), ("failed", "bad")]


def test_show_result_shows_an_edit_as_a_diff() -> None:
    printer = RecordingPrinter()
    edit = ToolMessage(
        call_id="1",
        state="done",
        stdout=("Edited a.py.", Splice(before="x", after="y", start=2)),
        result=EditResult(path="a.py"),
    )
    printer.call(_ECHO)
    show_result(printer, edit, header=_ECHO)
    show_result(
        printer,
        ToolMessage(
            call_id="2",
            state="done",
            stdout=("Edited b.py.",),
            result=EditResult(path="b.py"),
        ),
        header=_ECHO,
    )
    assert printer.printed == [
        ("call", "Echo"),
        ("diff", "a.py\n@@ -2,1 +2,1 @@\n-x\n+y"),
        ("output", "Edited b.py."),
    ]
    pathless = replace(edit, call_id="3", result=EditResult())
    show_result(printer, pathless, header=_ECHO)
    assert printer.printed[-1] == ("diff", "\n@@ -2,1 +2,1 @@\n-x\n+y")


def test_show_result_skips_a_pending_stub_but_not_a_cancelled_one() -> None:
    printer = RecordingPrinter()
    printer.call(_ECHO)
    show_result(printer, _result("1", "[detached]", stub="pending"), header=_ECHO)
    show_result(printer, _result("1", "[cancelled]", stub="cancelled"), header=_ECHO)
    assert printer.printed == [("call", "Echo"), ("failed", "[cancelled]")]


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
    show_result(
        printer,
        ToolMessage(
            call_id="1",
            state="done",
            stdout=(text,),
            result=UncategorizedToolResult(),
        ),
        header=_ECHO,
    )
    assert printer.printed == [("call", "Echo"), ("output", shown)]


def test_error_text_keeps_a_polished_message_and_names_any_other() -> None:
    assert error_text(BudgetExhaustedError(total_cost_usd=1, max_budget_usd=1)) == (
        "Budget exhausted: $1.00 >= $1.00"
    )
    assert error_text(ValueError("bad")) == "ValueError: bad"
    assert error_text(TimeoutError()) == "TimeoutError"


def test_a_wait_already_past_counts_down_from_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(time, "time", lambda: _NOON)
    render, printer = _renderer()
    render(
        types.ModelResponseSuspended(
            exception=OverloadedError("busy"),
            retry_at=datetime.fromtimestamp(_NOON - 5.0, UTC),
            failures_in_a_row=2,
        ),
    )
    assert printer.printed == [
        ("error", "Service overloaded. Resumes Wed Oct 28 11:59:55 (0s)."),
    ]


def test_a_one_second_rate_limit_says_when_it_resets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(time, "time", lambda: _NOON)
    assert error_text(RateLimitedError("slow", retry_after_sec=1.0)) == (
        "Rate limited. Resets Wed Oct 28 12:00:01 (1s)."
    )


def test_an_empty_write_says_it_wrote_and_a_pathless_one_still_diffs() -> None:
    printer = RecordingPrinter()
    printer.call(_ECHO)
    show_result(
        printer,
        ToolMessage(
            call_id="1",
            state="done",
            stdout=("Wrote c.py.",),
            result=WriteResult(path="c.py", content=""),
        ),
        header=_ECHO,
    )
    show_result(
        printer,
        ToolMessage(
            call_id="2",
            state="done",
            stdout=("Wrote the file.",),
            result=WriteResult(content="x\n"),
        ),
        header=_ECHO,
    )
    assert printer.printed == [
        ("call", "Echo"),
        ("output", "Wrote c.py."),
        ("diff", "\n@@ -0,0 +1,1 @@\n+x"),
    ]


def test_a_child_s_result_is_found_past_a_pending_stub() -> None:
    render, printer = _renderer()
    turn = _turn(call("Echo", "1", text="a"), call("Echo", "2", text="b"))
    render(types.ChildEvent(label="c", event=turn))
    render(types.ChildEvent(label="c", event=_result("1", "[x]", stub="pending")))
    render(types.ChildEvent(label="c", event=_result("1", "late", is_error=True)))
    render(types.ChildEvent(label="c", event=_result("2", "[y]", stub="pending")))
    render(types.ChildEvent(label="c", event=_result("2", "later", is_error=True)))
    assert [entry for entry in printer.printed if entry[0] != "call"] == [
        ("failed", "c: late"),
        ("failed", "c: later"),
    ]


def test_a_stream_closed_by_a_note_starts_afresh() -> None:
    render, printer = _renderer()
    render(types.ModelResponseTextStreamed(text="a"))
    render(types.SessionCompactStarted())
    render(types.ModelResponseTextStreamed(text="b\n\nc"))
    assert printer.printed == [
        ("markdown", "a"),
        ("note", "[compacting history…]"),
        ("markdown", "b"),
    ]


@pytest.mark.parametrize(
    ("held", "shown"),
    [
        ("  ```\n" + "x" * 6, "  ```\n" + "x" * 6),
        ("a ```b``` c\nmore text", "a ```b``` c"),
        ("- a\n  ```\n  xxxxxxx", "- a\n  ```\n  xxxxxxx"),
        ("- a\n  ```\n  x\n  ```\n  tail", "- a\n  ```\n  x\n  ```"),
        ("line one\nline two\nline th", "line one\nline two"),
    ],
    ids=["indented-fence", "inline-ticks", "fence-in-a-list", "closed-fence", "lines"],
)
def test_held_text_past_the_cap_keeps_an_open_fence_whole(
    held: str,
    shown: str,
) -> None:
    """Only a line that starts with a fence opens or closes one."""
    render, printer = _renderer(flush_chars=10)
    render(types.ModelResponseTextStreamed(text=held))
    assert printer.printed == [("markdown", shown)]


def test_each_message_and_reply_counts_toward_what_a_replay_shows() -> None:
    render, printer = _renderer()
    render(UserMessage(content="a"))
    render(UserMessage(content="b"))
    render(AgentMessage(sender="user"))
    render(AgentMessage(sender="Agent_0", content="c"))
    render(_turn(text="reply"))
    assert render.shown == 5
    assert ("note", "[/send] user→me: ") in printer.printed


@pytest.mark.parametrize(
    ("show_thinking", "content"),
    [(False, "because"), (True, "  "), (True, None)],
    ids=["hidden", "blank", "none"],
)
def test_a_thinking_record_prints_only_when_shown_and_said(
    show_thinking: bool,
    content: str | None,
) -> None:
    render, printer = _renderer(show_thinking=show_thinking)
    response = reply("x")
    render(
        types.ModelResponseCompleted(
            outcome=Succeeded(
                value=type(response)(
                    records=(Thinking(content=content), *response.records),
                    tokens=response.tokens,
                    spend=response.spend,
                    stop_reason="end_turn",
                ),
            ),
        ),
    )
    assert printer.printed == [("markdown", "x")]


def test_a_reply_keeps_its_trailing_spaces_and_a_wordless_one_prints_nothing() -> None:
    render, printer = _renderer()
    response = reply("x")
    render(
        types.ModelResponseCompleted(
            outcome=Succeeded(
                value=type(response)(
                    records=(
                        AssistantMessage(),
                        AssistantMessage(content="tail X  \n"),
                        AssistantMessage(content="ends X\n"),
                    ),
                    tokens=response.tokens,
                    spend=response.spend,
                    stop_reason="end_turn",
                ),
            ),
        ),
    )
    assert printer.printed == [("markdown", "tail X  "), ("markdown", "ends X")]
    assert render.shown == 2


def test_a_command_s_note_breaks_between_words() -> None:
    """Help is prose: cut at the edge, ``serves`` read as ``se`` / ``rves``."""
    printer = RecordingPrinter(width=24)
    render = Scrollback(printer, tools=_tools())
    render.show(
        CommandOutput(
            text="the current one serves the model",
            tone="note",
            command="/m",
        ),
    )
    assert printer.text.splitlines() == ["[/m] the current one", "serves the model"]


_TABLE = "col_a      col_b      col_c      col_d      col_e      col_f"


@pytest.mark.parametrize("tone", ["note", "error"])
def test_a_command_s_table_keeps_its_spacing_where_it_wraps(
    tone: Literal["note", "error"],
) -> None:
    """Prose is the output's to say: a table's spaces are its columns."""
    printer = RecordingPrinter(width=40)
    render = Scrollback(printer, tools=_tools())
    render.show(CommandOutput(text=_TABLE, tone=tone, prose=False))
    first, second = (row.rstrip() for row in printer.text.splitlines()[:2])
    lead = len("  ✗ ") if tone == "error" else 0
    assert first[lead:] == _TABLE[: 40 - lead].rstrip()
    assert second[lead:] == _TABLE[40 - lead :][: 40 - lead].rstrip()


def test_a_tool_s_failure_with_no_call_keeps_its_spacing() -> None:
    """What a tool wrote is no sentence, though no call of it is on screen."""
    printer = RecordingPrinter(width=40)
    render = Scrollback(printer, tools=_tools())
    render(_result("9", _TABLE, is_error=True))
    first = printer.text.splitlines()[0]
    assert first[len("  ✗ ") :] == _TABLE[:36]


def test_command_output_may_be_a_plain_line() -> None:
    render, printer = _renderer()
    render.show(CommandOutput(text="a", tone="line"))
    assert printer.printed == [("line", "a")]


def test_a_result_right_after_its_detached_label_names_its_call_once() -> None:
    render, printer = _renderer()
    render(_turn(call("Echo", "1", text="a")))
    render(_result("1", "[detached]", stub="pending"))
    render(_result("1", "late"))
    assert printer.printed == [
        ("call", "Echo a"),
        ("output", "[detached; Echo#1]"),
        ("output", "[undetached; Echo#1]"),
        ("output", "late"),
    ]


_SUSPENDED = types.ModelResponseSuspended(
    exception=OverloadedError("busy"),
    retry_at=datetime.fromtimestamp(_NOON, UTC),
    failures_in_a_row=1,
)
_FAILED = ModelIdle(outcome=Failed(exception=ValueError("the model said no more")))


@pytest.mark.parametrize(
    "given",
    [
        _FAILED,
        _SUSPENDED,
        _context_put_in_force((Path("a.md"),)),
        CommandOutput(text="no model named sonnet-9 here", tone="error"),
        types.ChildEvent(label="c", event=_FAILED),
        types.ChildEvent(label="c", event=_SUSPENDED),
    ],
    ids=["failed", "suspended", "skipped", "command", "child-failed", "child-waits"],
)
def test_an_error_that_is_a_sentence_breaks_between_words(given: Shown) -> None:
    printer = RecordingPrinter(width=24)
    Scrollback(printer, tools=_tools()).show(given)
    (said,) = [text for kind, text in printer.printed if kind == "error"]
    words = ["✗", *said.split()]
    assert printer.text.split()[: len(words)] == words


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
