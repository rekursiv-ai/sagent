"""Tests for ``repl.console_pane``: rich-backed ``Printer`` implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import io
import re

from rich.cells import cell_len
from rich.console import Console

import pytest

from sagent.repl.console_pane import (
    _LABEL_MAX_LINES,
    ConsolePrinter,
    _wrap_label,
)
from sagent.repl.render import make_render_observer, strict_observer
from sagent.tools.display import OutputSpec, ToolDisplay
from sagent.types.exceptions import AuthRefreshError
from sagent.types.runtime import (
    AssistantMessage,
    ChildDoneEvent,
    ChildEvent,
    ModelResponseCancelled,
    ModelResponseComplete,
    ModelResponseError,
    ModelResponsePartial,
    ModelResponseThinking,
    StatusChanged,
    ToolLabel,
    ToolResult,
    UserMessage,
)


if TYPE_CHECKING:
    from collections.abc import Callable

    from sagent.repl.render import ChildItem
    from sagent.types.runtime import RuntimeEvent


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _printer(width: int = 80) -> tuple[ConsolePrinter, io.StringIO]:
    buf = io.StringIO()
    con = Console(
        file=buf,
        width=width,
        force_terminal=False,
        color_system=None,
        highlight=False,
    )
    return ConsolePrinter(con), buf


def test_write_line_emits_payload() -> None:
    printer, buf = _printer()
    printer.write_line("hello")
    assert "hello" in buf.getvalue()


def test_write_line_markup_disabled() -> None:
    # ``[/clear]`` would otherwise be parsed as a closing rich tag.
    printer, buf = _printer()
    printer.write_line("[/clear] history cleared")
    assert "[/clear] history cleared" in buf.getvalue()


def test_write_chunk_no_newline() -> None:
    printer, buf = _printer()
    printer.write_chunk("partial")
    out = buf.getvalue()
    assert "partial" in out
    # No trailing newline appended.
    assert not out.endswith("\n")


def test_write_markdown_emits_blank_then_content() -> None:
    printer, buf = _printer()
    printer.write_markdown("# Heading")
    out = buf.getvalue()
    assert "Heading" in out


def test_write_user_bar_includes_payload() -> None:
    printer, buf = _printer()
    printer.write_user_bar("hello there")
    assert "hello there" in buf.getvalue()


def test_write_agent_bar_body_not_dim() -> None:
    """``write_agent_bar`` body must render at normal brightness.

    The ``[from <source>]:`` prefix is dim cyan to mark the
    attribution as machinery. The body itself should NOT be dim --
    it's the substantive message and must read with the same weight
    as a normal user/agent reply. Today the body inherits the dim
    attribute from the prefix's Text style merge, so the entire line
    reads as background output.
    """
    buf = io.StringIO()
    con = Console(
        file=buf,
        width=80,
        force_terminal=True,
        color_system="truecolor",
        highlight=False,
    )
    printer = ConsolePrinter(con)
    printer.write_agent_bar("reviewer", "important payload")
    out = buf.getvalue()
    # ANSI ``\x1b[2`` enables dim; the body span must not carry it.
    # Split on the prefix's closing ``[0m`` reset; the body span follows.
    _prefix, _, after = out.partition("[from reviewer]: ")
    # The body characters should NOT be wrapped in a dim-on span.
    # Look for the dim attribute on the body's opening ANSI block.
    body_open = after.split("important")[0]
    assert "\x1b[2" not in body_open.split("\x1b[0m")[-1], (
        f"agent-bar body inherits dim from prefix style; body opening"
        f" sequence: {body_open!r}"
    )


def test_write_tool_label_marks_the_input_row() -> None:
    """Line 0 is the header; further lines are the call's INPUT.

    The input glyph matches the diff receipt, so the one row that is not
    output looks the same across every tool.
    """
    printer, buf = _printer()
    printer.write_tool_label(
        "Bash List files\nls -la",
        command=OutputSpec(show=True, unbounded=True),
    )
    out = buf.getvalue()
    assert "  Bash List files" in out
    assert "\u23bf  ls -la" in out


def test_write_tool_output_is_indented_without_a_glyph() -> None:
    """Output carries no glyph: a 20-line body must not be 20 glyphs."""
    printer, buf = _printer()
    printer.write_tool_output("alpha\nbeta")
    out = buf.getvalue()
    assert "   alpha" in out
    assert "   beta" in out
    assert "\u23bf" not in out


def test_wrapped_input_continues_at_the_output_indent() -> None:
    """The input glyph marks the row once, not on every wrapped line."""
    printer, buf = _printer(width=20)
    printer.write_tool_label(
        f"Bash\n{'x' * 40}",
        command=OutputSpec(show=True, unbounded=True),
    )
    lines = [ln for ln in buf.getvalue().split("\n") if ln.strip()]
    assert lines[1].startswith("  \u23bf  ")
    assert lines[2].startswith("     ")
    assert "\u23bf" not in lines[2]


def _emit_label(printer: ConsolePrinter) -> None:
    printer.write_tool_label("Bash\n" + "x" * 200)


def _emit_output(printer: ConsolePrinter) -> None:
    printer.write_tool_output("x" * 200)


@pytest.mark.parametrize(
    "emit",
    [_emit_label, _emit_output],
    ids=["label", "output"],
)
def test_rows_keep_their_indent_when_wrapped(
    emit: Callable[[ConsolePrinter], None],
) -> None:
    """A row wider than the pane lost its gutter on every wrapped line.

    Printing the row verbatim let Rich wrap it, so a 200-char line broke
    back to column 0 -- outdented past the indent, reading as top-level.
    """
    printer, buf = _printer(width=80)
    emit(printer)
    for line in buf.getvalue().rstrip("\n").split("\n"):
        assert line.startswith("  "), line
        assert cell_len(line) <= 80, line


def test_write_tool_label_empty_string_emits_blank_indent() -> None:
    printer, buf = _printer()
    printer.write_tool_label("")
    # No crash; single line with indent.
    assert "  " in buf.getvalue()


def test_write_tool_error_first_line_carries_glyph() -> None:
    printer, buf = _printer()
    printer.write_tool_error("oops\nmore")
    out = buf.getvalue()
    assert "   ✗ oops" in out
    assert "     more" in out


def test_write_tool_error_blank_renders_placeholder() -> None:
    """REPL-028: empty error body still surfaces a placeholder line.

    Silently dropping the call would let upstream report a failure that
    leaves no trace on the operator's screen.
    """
    printer, buf = _printer()
    printer.write_tool_error("")
    assert "<no error message>" in buf.getvalue()
    printer, buf = _printer()
    printer.write_tool_error("\n  \n")
    assert "<no error message>" in buf.getvalue()


def test_write_tool_summary_carries_arrow_glyph() -> None:
    printer, buf = _printer()
    printer.write_tool_summary("3 lines")
    assert "⎿  3 lines" in buf.getvalue()


def test_write_hint_renders_prefix() -> None:
    printer, buf = _printer()
    printer.write_hint("use grep")
    assert "hint: use grep" in buf.getvalue()


def test_write_interrupted_writes_marker() -> None:
    printer, buf = _printer()
    printer.write_interrupted()
    assert "[interrupted]" in buf.getvalue()


def test_write_dim_line_emits_dim_payload() -> None:
    printer, buf = _printer()
    printer.write_dim_line("[compacting history…]")
    assert "[compacting history…]" in buf.getvalue()


def test_write_halt_emits_banner_lines() -> None:
    printer, buf = _printer()
    printer.write_halt("agent halted")
    out = buf.getvalue()
    assert "agent halted" in out
    # Banner lines composed of ── characters.
    assert "─" in out


def test_write_thinking_includes_header_and_body() -> None:
    printer, buf = _printer()
    printer.write_thinking("plan A\nplan B")
    out = buf.getvalue()
    assert "Thinking" in out
    assert "plan A" in out
    assert "plan B" in out


def test_thinking_deltas_are_visible_immediately_under_one_header() -> None:
    printer, buf = _printer()
    obs = make_render_observer(printer)
    with strict_observer():
        obs(ModelResponseThinking(text="Keep"))
        assert "Keep" in buf.getvalue()
        obs(ModelResponseThinking(text=" track"))
        obs(ModelResponseThinking(text=" of progress."))
        obs(ModelResponseComplete(message=AssistantMessage(text="")))
    assert buf.getvalue() == "∴ Thinking\n  Keep track of progress.\n\n"


@pytest.mark.parametrize("width", [16, 25, 80])
def test_thinking_wrap_is_independent_of_delta_boundaries(width: int) -> None:
    text = "Keep track of progress.\n\n界面 e\u0301 stays readable.\r\nNext\tline."
    whole, whole_buf = _printer(width)
    whole.write_thinking(text)
    split, split_buf = _printer(width)
    # Include splits inside words, CRLF and a combining sequence.
    for char in text:
        split.write_thinking_chunk(char)
    split.finish_thinking()
    assert split_buf.getvalue() == whole_buf.getvalue()
    lines = split_buf.getvalue().splitlines()
    assert all(cell_len(line) <= width for line in lines)
    assert all(line.startswith("  ") for line in lines[1:] if line)


def test_repl_thinking_commits_complete_lines_and_flushes_the_tail() -> None:
    buf = io.StringIO()
    printer = ConsolePrinter(
        Console(file=buf, width=24, color_system=None),
        thinking_line_buffered=True,
    )
    printer.write_thinking_chunk("Keep")
    assert buf.getvalue() == "∴ Thinking\n"
    printer.write_thinking_chunk(" track\nNext")
    assert buf.getvalue() == "∴ Thinking\n  Keep track\n"
    printer.finish_thinking()
    assert buf.getvalue() == "∴ Thinking\n  Keep track\n  Next\n\n"


@pytest.mark.parametrize("width", [16, 25, 80])
def test_repl_thinking_tail_survives_arbitrary_fragments_and_wrapping(
    width: int,
) -> None:
    text = "A longer reasoning line wraps safely.\n界面 e\u0301 stays readable.\r\nNext\tline."
    whole, whole_buf = _printer(width)
    whole.write_thinking(text)
    buf = io.StringIO()
    printer = ConsolePrinter(
        Console(file=buf, width=width, color_system=None),
        thinking_line_buffered=True,
    )
    for char in text:
        printer.write_thinking_chunk(char)
        assert buf.getvalue().endswith("\n")
    printer.finish_thinking()
    assert buf.getvalue() == whole_buf.getvalue()


def test_empty_thinking_delta_does_not_print_a_header() -> None:
    printer, buf = _printer()
    printer.write_thinking_chunk("")
    printer.finish_thinking()
    assert buf.getvalue() == ""


@pytest.mark.parametrize(
    "boundary",
    [
        ModelResponseComplete(message=AssistantMessage(text="")),
        ModelResponseCancelled(),
        ToolLabel(text="Bash inspect", call_id="c1"),
        ToolResult(call_id="c1", content="done", summary="tool completed"),
        ModelResponseError(exception=RuntimeError("failed")),
        UserMessage(text="new request"),
    ],
)
def test_thinking_boundary_closes_line_and_starts_a_new_block(
    boundary: RuntimeEvent,
) -> None:
    printer, buf = _printer()
    obs = make_render_observer(printer)
    with strict_observer():
        obs(ModelResponseThinking(text="first"))
        obs(boundary)
        obs(ModelResponseThinking(text="second"))
        obs(ModelResponseComplete(message=AssistantMessage(text="")))
    out = buf.getvalue()
    assert "  first\n\n" in out
    assert out.count("∴ Thinking") == 2
    assert out.index("first") < out.index("second")


def test_answer_then_reasoning_preserves_output_order() -> None:
    printer, buf = _printer()
    obs = make_render_observer(printer)
    with strict_observer():
        obs(ModelResponsePartial(text="Earlier answer"))
        obs(ModelResponseThinking(text="Later reasoning"))
        obs(ModelResponsePartial(text="Final answer"))
        obs(ModelResponseComplete(message=AssistantMessage(text="")))
    out = buf.getvalue()
    assert out.index("Earlier answer") < out.index("Later reasoning")
    assert out.index("Later reasoning") < out.index("Final answer")


def test_thinking_hide_show_mid_stream_closes_visible_text() -> None:
    printer, buf = _printer()
    obs = make_render_observer(printer)
    with strict_observer():
        obs(ModelResponseThinking(text="visible"))
        printer.show_thinking = False
        obs(ModelResponseThinking(text="hidden"))
        printer.show_thinking = True
        obs(ModelResponseThinking(text="shown again"))
        obs(ModelResponseComplete(message=AssistantMessage(text="")))
    out = buf.getvalue()
    assert "hidden" not in out
    assert out.count("∴ Thinking") == 2
    assert "  visible\n\n" in out


def test_status_updates_do_not_split_thinking() -> None:
    printer, buf = _printer()
    obs = make_render_observer(printer)
    with strict_observer():
        obs(ModelResponseThinking(text="one"))
        obs(StatusChanged(text="working"))
        obs(ModelResponseThinking(text=" thought"))
        obs(ModelResponseComplete(message=AssistantMessage(text="")))
    assert buf.getvalue() == "∴ Thinking\n  one thought\n\n"


def test_child_reasoning_stream_retains_label_and_one_header() -> None:
    printer, buf = _printer()
    obs = make_render_observer(printer)
    with strict_observer():
        for delta in ["child", " reasoning"]:
            obs(ChildEvent(label="Agent_0", inner=ModelResponseThinking(text=delta)))
        obs(ChildDoneEvent(label="Agent_0", elapsed=1, tokens=0, cost=0))
    out = buf.getvalue()
    assert out.count("Thinking") == out.count("Agent_0") == 1
    assert "child reasoning" in out
    assert out.endswith("\n\n")


def test_interleaved_reasoning_keeps_parent_and_children_separate() -> None:
    printer, buf = _printer()
    obs = make_render_observer(printer)
    with strict_observer():
        obs(ModelResponseThinking(text="parent"))
        obs(ChildEvent(label="Agent_0", inner=ModelResponseThinking(text="child zero")))
        obs(ChildEvent(label="Agent_1", inner=ModelResponseThinking(text="child one")))
        obs(ModelResponseThinking(text="parent again"))
        obs(ModelResponseComplete(message=AssistantMessage(text="")))
    out = buf.getvalue()
    assert out.count("Thinking") == 4
    assert out.index("parent") < out.index("child zero") < out.index("child one")
    assert out.index("child one") < out.index("parent again")


def test_hidden_child_thinking_does_not_leak() -> None:
    printer, buf = _printer()
    printer.show_thinking = False
    obs = make_render_observer(printer)
    obs(ChildEvent(label="Agent_0", inner=ModelResponseThinking(text="private")))
    assert buf.getvalue() == ""


def test_unrelated_child_completion_does_not_split_parent_thinking() -> None:
    printer, buf = _printer()
    obs = make_render_observer(printer)
    with strict_observer():
        obs(ModelResponseThinking(text="one"))
        obs(ChildDoneEvent(label="Agent_0", elapsed=1, tokens=0, cost=0))
        obs(ModelResponseThinking(text=" thought"))
        obs(ModelResponseComplete(message=AssistantMessage(text="")))
    assert buf.getvalue() == "∴ Thinking\n  one thought\n\n"


def test_set_terminal_title_does_not_raise() -> None:
    printer, _ = _printer()
    # No tty -> no-op; just verify it doesn't raise.
    printer.set_terminal_title("session foo")


def test_write_diff_renders_added_removed_count() -> None:
    printer, buf = _printer()
    diff = "@@ -1,2 +1,2 @@\n-old line\n+new line\n unchanged\n"
    printer.write_diff(diff, file_path="x.py")
    out = buf.getvalue()
    assert "Added 1 lines" in out
    assert "removed 1 lines" in out


def test_write_child_block_empty_items_is_noop() -> None:
    printer, buf = _printer()
    printer.write_child_block("Agent_0", [])
    assert buf.getvalue() == ""


def test_write_child_block_renders_assistant_text() -> None:
    printer, buf = _printer()
    items: list[ChildItem] = [AssistantMessage(text="child output")]
    printer.write_child_block("Agent_0", items)
    out = buf.getvalue()
    assert "child output" in out
    assert "Agent_0" in out


def test_write_child_block_renders_tool_label() -> None:
    printer, buf = _printer()
    items: list[ChildItem] = [ToolLabel(call_id="c1", text="Bash")]
    printer.write_child_block("Agent_1", items)
    out = buf.getvalue()
    assert "Bash" in out


def test_write_child_block_renders_thinking() -> None:
    printer, buf = _printer()
    items: list[ChildItem] = [ModelResponseThinking(text="hmm")]
    printer.write_child_block("Agent_2", items)
    out = buf.getvalue()
    assert "hmm" in out


def test_write_child_block_renders_tool_result_summary() -> None:
    printer, buf = _printer()
    items: list[ChildItem] = [
        ToolResult(call_id="c1", content="ok", summary="one line"),
    ]
    printer.write_child_block("Agent_3", items)
    out = buf.getvalue()
    assert "one line" in out


def test_write_child_block_renders_user_message() -> None:
    printer, buf = _printer()
    items: list[ChildItem] = [UserMessage(text="user note")]
    printer.write_child_block("Agent_4", items)
    out = buf.getvalue()
    assert "user note" in out


def test_rich_still_emits_ansi_full_reset_pinning() -> None:
    r"""Pin Rich's ``\x1b[0m`` reset emission used by ``_dim_baseline``.

    ``_dim_baseline`` (private helper in ``console_pane``) re-applies the
    dim attribute after every Rich-emitted full reset by string-replacing
    ``\x1b[0m`` -> ``\x1b[0m\x1b[2m``. If a future Rich release
    switches to attribute-specific resets (e.g. ``\x1b[39;49m``) or
    drops the trailing reset entirely, the dim baseline would silently
    leak past per-span styles. Fail here so the regression surfaces in
    test rather than at runtime in the child-block renderer.
    """
    buf = io.StringIO()
    con = Console(
        file=buf,
        width=40,
        force_terminal=True,
        color_system="truecolor",
        highlight=False,
        no_color=False,
    )
    con.print("[bold red]styled[/]")
    rendered = buf.getvalue()
    assert "\x1b[0m" in rendered, (
        "Rich no longer emits \\x1b[0m full resets after styled spans;"
        " _dim_baseline's rewrite is now a no-op. Update the helper to"
        " match the new reset shape."
    )


def test_wrap_label_total_lines_respect_the_cap() -> None:
    """The omission marker is part of the cap, not an extra line past it."""
    lines = _wrap_label("x\n" * 20, 80)
    assert len(lines) <= _LABEL_MAX_LINES, (
        f"cap is {_LABEL_MAX_LINES} but {len(lines)} lines were returned"
    )
    assert "more lines" in lines[-1], "elision happened with no marker"


@pytest.mark.parametrize("width", [16, 20, 24, 33])
def test_child_block_labels_fit_narrow_terminals(width: int) -> None:
    """A child-block label must not overflow the terminal it renders into.

    The inner console is floored at 20 columns while the gutter is 14,
    so below ~34 columns the composed line is wider than the terminal
    and every label wraps raggedly in the user's pane.
    """
    printer, buf = _printer(width=width)
    printer.write_child_block("Agent_0", [ToolLabel(call_id="c", text="B" * 300)])
    rendered = [
        _ANSI_RE.sub("", line) for line in buf.getvalue().split("\n") if line.strip()
    ]
    widest = max(len(line) for line in rendered)
    assert widest <= width, (
        f"child-block label rendered {widest} columns into a {width}-column terminal"
    )


def test_child_block_renders_a_tool_body_when_the_policy_says_so() -> None:
    """A subagent's Bash body must render exactly as the parent's does.

    The child path called ``render_tool_result`` with no policy, which
    reads as hidden -- so ``--tool Bash.output=on`` showed bodies live
    and nothing at all under a child label.
    """
    printer, buf = _printer()
    items: list[ChildItem] = [ToolResult(call_id="c1", content="SENTINEL")]
    printer.write_child_block(
        "Agent_0",
        items,
        output_policy=lambda _cid: ToolDisplay(
            output=OutputSpec(show=True, unbounded=True),
        ),
    )
    assert "SENTINEL" in buf.getvalue()


def test_child_block_hides_the_body_without_a_policy() -> None:
    printer, buf = _printer()
    items: list[ChildItem] = [ToolResult(call_id="c1", content="SENTINEL")]
    printer.write_child_block("Agent_0", items)
    assert "SENTINEL" not in buf.getvalue()


def test_child_user_facing_error_drops_the_class_name_prefix() -> None:
    """The parent strips it and pins that in a test; the child must too."""
    printer, buf = _printer()
    items: list[ChildItem] = [
        ModelResponseError(exception=AuthRefreshError("run /login")),
    ]
    printer.write_child_block("Agent_0", items)
    out = _ANSI_RE.sub("", buf.getvalue())
    assert "run /login" in out
    assert "AuthRefreshError" not in out


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
