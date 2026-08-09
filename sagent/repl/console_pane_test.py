"""Tests for ``repl.console_pane``: rich-backed ``Printer`` implementation."""

from __future__ import annotations

import io
import re

from rich.console import Console

import pytest

from sagent.repl.console_pane import (
    _LABEL_MAX_LINES,
    ConsolePrinter,
    _wrap_label,
)
from sagent.repl.render import ChildItem
from sagent.types.runtime import (
    AssistantMessage,
    ModelResponseThinking,
    ToolLabel,
    ToolResult,
    UserMessage,
)


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


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
