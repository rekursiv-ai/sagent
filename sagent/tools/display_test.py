"""Tests for ``tools.display``: tool output display policy."""

from __future__ import annotations

from sagent.tools.display import (
    OutputSpec,
    ToolDisplay,
    format_output,
    row_spec,
)


def test_hidden_body_renders_nothing() -> None:
    assert format_output("alpha", OutputSpec(show=False)) == []


def test_blank_text_renders_nothing() -> None:
    assert format_output("   \n\n", OutputSpec(show=True)) == []


def test_unbounded_body_keeps_every_line() -> None:
    assert format_output("a\nb\nc", OutputSpec(show=True)) == ["a", "b", "c"]


def test_head_and_tail_are_kept_with_a_marker_between() -> None:
    body = "\n".join(str(i) for i in range(10))
    got = format_output(body, OutputSpec(show=True, head_rows=2, tail_rows=2))
    assert got == ["0", "1", "\u22ef 6 lines \u22ef", "8", "9"]


def test_tail_only_keeps_the_last_rows() -> None:
    """A test run's answer is its final line; head-only truncation loses it."""
    body = "\n".join(str(i) for i in range(10))
    got = format_output(body, OutputSpec(show=True, tail_rows=2))
    assert got == ["\u22ef 8 lines \u22ef", "8", "9"]


def test_head_only_does_not_duplicate_the_body() -> None:
    """``rows[-0:]`` is the whole list -- a zero tail must slice empty."""
    body = "\n".join(str(i) for i in range(10))
    got = format_output(body, OutputSpec(show=True, head_rows=2))
    assert got == ["0", "1", "\u22ef 8 lines \u22ef"]


def test_body_within_budget_is_untouched() -> None:
    got = format_output("a\nb", OutputSpec(show=True, head_rows=2, tail_rows=2))
    assert got == ["a", "b"]


def test_budget_counts_logical_lines_and_keeps_them_whole() -> None:
    """A kept line must survive intact, not as a wrapped fragment.

    Slicing already-wrapped rows kept the TAIL of one long line -- the
    reader saw ``il -2`` where the final command line belonged, which
    reads as corruption rather than elision.
    """
    body = "\n".join(f"{i}" + "x" * 39 for i in range(5))
    got = format_output(
        body, OutputSpec(show=True, head_rows=1, tail_rows=1, max_width=10)
    )
    assert "".join(got[:4]) == "0" + "x" * 39, got
    assert got[4].startswith("\u22ef"), got
    assert "".join(got[5:]) == "4" + "x" * 39, got


def test_wrapping_still_bounds_each_kept_line() -> None:
    body = "\n".join("x" * 40 for _ in range(2))
    got = format_output(body, OutputSpec(show=True, max_width=10))
    assert all(len(row) <= 10 for row in got), got


def test_the_elision_marker_is_never_wrapped() -> None:
    """The marker is ours, not content; chopping it hides the count."""
    body = "\n".join(str(i) for i in range(10))
    got = format_output(
        body, OutputSpec(show=True, head_rows=1, tail_rows=1, max_width=4)
    )
    assert "\u22ef 8 lines \u22ef" in got, got


def test_chop_keeps_the_head_and_marks_the_cut() -> None:
    assert format_output("abcdef", OutputSpec(show=True, wrap="chop"), width=4) == [
        "abc\u2026"
    ]


def test_chop_leaves_a_fitting_line_unmarked() -> None:
    assert format_output("abc", OutputSpec(show=True, wrap="chop"), width=4) == ["abc"]


def test_max_width_overrides_the_caller_width() -> None:
    got = format_output("abcdef", OutputSpec(show=True, max_width=2), width=99)
    assert got == ["ab", "cd", "ef"]


def test_row_spec_reads_flat_attributes() -> None:
    class _Tool:
        output = "on"
        output_head_rows = 2
        output_tail_rows = 3
        output_max_width = 40
        output_wrap = "chop"

    assert row_spec(_Tool()).output == OutputSpec(
        show=True, head_rows=2, tail_rows=3, max_width=40, wrap="chop"
    )


def test_row_spec_defaults_to_hidden_for_an_unaware_tool() -> None:
    """Tools with no knob must not start dumping bodies into the pane."""

    class _Tool:
        pass

    assert row_spec(_Tool()) == ToolDisplay()


def test_command_knobs_bound_the_input_row() -> None:
    """A heredoc must not flood the pane before the tool even runs."""

    class _Tool:
        output = "on"
        output_head_rows = 2
        output_tail_rows = 2
        output_max_width = 0
        output_wrap = "wrap"
        command_head_rows = 3
        command_tail_rows = 1
        command_lang = "bash"

    display = row_spec(_Tool())
    assert display.command == OutputSpec(
        show=True, head_rows=3, tail_rows=1, max_width=0, wrap="wrap"
    )
    assert display.command_lang == "bash"


def test_command_is_unbounded_without_command_knobs() -> None:
    """A one-line path or receipt needs no budget."""

    class _Tool:
        output = "on"
        output_head_rows = 2
        output_tail_rows = 2
        output_max_width = 0
        output_wrap = "wrap"

    assert row_spec(_Tool()).command == OutputSpec(show=True)


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
