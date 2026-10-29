"""Tests for how a tool call's input row and output show in the terminal."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from sagent.tools.display import DisplaySpec, ToolDisplay, override


if TYPE_CHECKING:
    from collections.abc import Callable


def _numbered(n: int) -> str:
    return "\n".join(str(i) for i in range(n))


def test_a_hidden_or_blank_block_renders_nothing() -> None:
    assert DisplaySpec(lines=(0, 0)).render("a\nb", width=80) == []
    assert DisplaySpec(lines=None).render("   \n\n", width=80) == []


def test_keeping_all_lines_shows_every_one() -> None:
    assert DisplaySpec(lines=None).render("a\nb\nc\n", width=80) == ["a", "b", "c"]


def test_head_and_tail_are_kept_with_a_count_between() -> None:
    assert DisplaySpec(lines=(2, 2)).render(_numbered(10), width=80) == [
        "0",
        "1",
        "⋯ 6 lines ⋯",
        "8",
        "9",
    ]


def test_a_tail_alone_keeps_the_last_lines() -> None:
    assert DisplaySpec(lines=(0, 2)).render(_numbered(10), width=80) == [
        "⋯ 8 lines ⋯",
        "8",
        "9",
    ]


def test_a_head_alone_does_not_repeat_the_block() -> None:
    """``rows[-0:]`` is every row, so a zero tail must keep none."""
    assert DisplaySpec(lines=(2, 0)).render(_numbered(10), width=80) == [
        "0",
        "1",
        "⋯ 8 lines ⋯",
    ]


def test_a_block_within_the_budget_is_untouched() -> None:
    assert DisplaySpec(lines=(2, 2)).render(_numbered(4), width=80) == [
        "0",
        "1",
        "2",
        "3",
    ]


def test_kept_lines_stay_whole_when_folded() -> None:
    """Lines are counted before folding, so a kept line never shows as a fragment."""
    text = "\n".join(f"{i}" + "x" * 19 for i in range(5))
    rows = DisplaySpec(lines=(1, 1)).render(text, width=10)
    assert rows == [
        "0xxxxxxxxx",
        "xxxxxxxxxx",
        "⋯ 3 lines ⋯",
        "4xxxxxxxxx",
        "xxxxxxxxxx",
    ]


def test_the_count_marker_is_never_folded() -> None:
    rows = DisplaySpec(lines=(1, 1)).render(_numbered(10), width=4)
    assert "⋯ 8 lines ⋯" in rows


def test_ellipsis_cuts_a_wide_line_and_marks_the_cut() -> None:
    spec = DisplaySpec(lines=None, overflow="ellipsis")
    assert spec.render("abcdef", width=4) == ["abc…"]
    assert spec.render("abcd", width=4) == ["abcd"]
    assert spec.render("abc", width=1) == ["a…"]


def test_folding_counts_cells_not_characters() -> None:
    assert DisplaySpec(lines=None).render("日本語", width=4) == ["日本", "語"]


def test_a_spec_width_overrides_the_panes() -> None:
    assert DisplaySpec(lines=None, width=2).render("abcdef", width=99) == [
        "ab",
        "cd",
        "ef",
    ]


def test_a_zero_width_does_not_fold() -> None:
    assert DisplaySpec(lines=None).render("abcdef", width=0) == ["abcdef"]


def test_an_empty_line_survives_folding() -> None:
    assert DisplaySpec(lines=None).render("a\n\nb", width=4) == ["a", "", "b"]


def test_every_line_of_an_uncut_block_is_folded() -> None:
    assert DisplaySpec(lines=(2, 2)).render("a\nbbbbb\nc", width=2) == [
        "a",
        "bb",
        "bb",
        "b",
        "c",
    ]


def test_only_trailing_newlines_are_dropped() -> None:
    assert DisplaySpec(lines=None).render("a  \n\t\n\n", width=0) == ["a  ", "\t"]


@pytest.mark.parametrize(
    "build",
    [
        lambda: DisplaySpec(lines=(-1, 1)),
        lambda: DisplaySpec(lines=(1, -1)),
        lambda: DisplaySpec(width=-1),
    ],
    ids=["head", "tail", "width"],
)
def test_a_negative_count_is_rejected(build: Callable[[], DisplaySpec]) -> None:
    with pytest.raises(ValueError, match="negative count"):
        build()


def test_a_tool_shows_its_input_whole_and_hides_its_output_by_default() -> None:
    display = ToolDisplay()
    assert display.command == DisplaySpec(lines=None)
    assert display.output.render("a\nb", width=80) == []
    assert display.command_language == ""


@pytest.mark.parametrize(
    ("setting", "expected"),
    [
        ("output.lines=3,1", ToolDisplay(output=DisplaySpec(lines=(3, 1)))),
        ("output.lines=all", ToolDisplay(output=DisplaySpec(lines=None))),
        ("output.width=40", ToolDisplay(output=DisplaySpec(lines=(0, 0), width=40))),
        (
            "command.overflow=ellipsis",
            ToolDisplay(command=DisplaySpec(lines=None, overflow="ellipsis")),
        ),
        (
            "command.overflow=fold",
            ToolDisplay(command=DisplaySpec(lines=None, overflow="fold")),
        ),
        ("command.lines=2,0", ToolDisplay(command=DisplaySpec(lines=(2, 0)))),
        ("command_language=bash", ToolDisplay(command_language="bash")),
        ("command_language=a=b", ToolDisplay(command_language="a=b")),
    ],
)
def test_override_changes_one_setting(setting: str, expected: ToolDisplay) -> None:
    assert override(ToolDisplay(), setting) == expected


def test_override_keeps_the_other_settings() -> None:
    display = ToolDisplay(
        command=DisplaySpec(lines=(3, 1)),
        output=DisplaySpec(),
        command_language="bash",
    )
    changed = override(display, "output.width=9")
    assert changed.output == DisplaySpec(width=9)
    assert changed.command == display.command
    assert changed.command_language == "bash"


@pytest.mark.parametrize(
    ("setting", "message"),
    [
        ("output.lines", "lacks '='"),
        (
            "input.lines=1,1",
            (
                r"^Unknown display setting 'input.lines'\. Settable: command_language, and"
                r" command or output with \.lines, \.width, or \.overflow\.$"
            ),
        ),
        ("output.color=red", "Unknown display setting 'output.color'"),
        ("output.lines=3", "lines must be HEAD,TAIL or all, not '3'"),
        ("output.lines=a,1", "lines must be HEAD,TAIL or all"),
        ("output.lines=1,b", "lines must be HEAD,TAIL or all"),
        ("output.width=wide", "width must be a whole number, not 'wide'"),
        ("output.overflow=wrap", "overflow must be fold or ellipsis, not 'wrap'"),
    ],
)
def test_override_rejects_a_setting_that_does_not_fit(
    setting: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        override(ToolDisplay(), setting)


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
