"""Tests for ``repl.console_pane``: rich-backed ``Printer`` implementation."""

from __future__ import annotations

import io

from rich.console import Console

from sagent.repl.console_pane import ConsolePrinter
from sagent.types.history import (
    AssistantMessage,
    ToolResult,
    UserMessage,
)
from sagent.types.runtime import ModelResponseThinking, ToolLabel


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


def test_write_tool_label_indents_each_line() -> None:
    printer, buf = _printer()
    printer.write_tool_label("step 1\nstep 2")
    out = buf.getvalue()
    assert "  step 1" in out
    assert "  step 2" in out


def test_write_tool_label_empty_string_emits_blank_indent() -> None:
    printer, buf = _printer()
    printer.write_tool_label("")
    # No crash; single line with indent.
    assert "  " in buf.getvalue()


def test_write_tool_error_first_line_carries_glyph() -> None:
    printer, buf = _printer()
    printer.write_tool_error("oops\nmore")
    out = buf.getvalue()
    assert "✗ oops" in out
    assert "      more" in out


def test_write_tool_error_blank_is_noop() -> None:
    printer, buf = _printer()
    printer.write_tool_error("")
    assert buf.getvalue() == ""


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
    items: list[object] = [AssistantMessage(text="child output")]
    printer.write_child_block("Agent_0", items)
    out = buf.getvalue()
    assert "child output" in out
    assert "Agent_0" in out


def test_write_child_block_renders_tool_label() -> None:
    printer, buf = _printer()
    items: list[object] = [ToolLabel(call_id="c1", text="Bash")]
    printer.write_child_block("Agent_1", items)
    out = buf.getvalue()
    assert "Bash" in out


def test_write_child_block_renders_thinking() -> None:
    printer, buf = _printer()
    items: list[object] = [ModelResponseThinking(text="hmm")]
    printer.write_child_block("Agent_2", items)
    out = buf.getvalue()
    assert "hmm" in out


def test_write_child_block_renders_tool_result_summary() -> None:
    printer, buf = _printer()
    items: list[object] = [
        ToolResult(call_id="c1", content="ok", summary="one line"),
    ]
    printer.write_child_block("Agent_3", items)
    out = buf.getvalue()
    assert "one line" in out


def test_write_child_block_renders_user_message() -> None:
    printer, buf = _printer()
    items: list[object] = [UserMessage(text="user note")]
    printer.write_child_block("Agent_4", items)
    out = buf.getvalue()
    assert "user note" in out


def test_write_child_block_ignores_unknown_item_type() -> None:
    printer, buf = _printer()
    # ``int`` is not a child-block item type; ``_render_child_item`` skips it.
    printer.write_child_block("Agent_X", [42])
    # Nothing written.
    assert buf.getvalue() == ""


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
