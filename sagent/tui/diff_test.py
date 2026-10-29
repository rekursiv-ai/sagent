"""Tests for diffs in the scrollback: numbered lines, colors, and changed words."""

from __future__ import annotations

import io
import re

from rich.console import Console

import pytest

from sagent.session.types import FileEditResult, FileWriteResult, Splice
from sagent.tui.diff import (
    DiffStyle,
    highlight_source,
    render_diff,
    splice_diff,
    write_diff,
)


_ADDED = "48;2;34;92;43"
_REMOVED = "48;2;122;41;54"
_ADDED_WORD = "48;2;56;166;96"
_REMOVED_WORD = "48;2;179;89;107"
_GUTTER = "38;2;160;160;160"


def _plain(diff: str, *, path: str = "", width: int = 40) -> list[str]:
    out = io.StringIO()
    console = Console(file=out, width=width, color_system=None, highlight=False)
    render_diff(console, diff, path=path)
    return [row.rstrip() for row in out.getvalue().splitlines()]


def _ansi(
    diff: str,
    *,
    path: str = "",
    style: DiffStyle | None = None,
    width: int = 40,
) -> list[str]:
    out = io.StringIO()
    console = Console(
        file=out,
        width=width,
        color_system="truecolor",
        force_terminal=True,
    )
    render_diff(console, diff, path=path, style=style)
    return out.getvalue().splitlines()


def _segments(row: str) -> list[tuple[str, str]]:
    """Return a row's runs of text, each with the SGR codes it was printed in."""
    return [
        (str(m.group(1)), str(m.group(2)))
        for m in re.finditer(r"\x1b\[([0-9;]*)m([^\x1b]*)", row)
        if m.group(2)
    ]


def _styled(row: str, code: str) -> str:
    """Return the text of ``row`` printed with ``code`` among its codes."""
    return "".join(text for codes, text in _segments(row) if code in codes)


def _visible(row: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", row)


# ---- numbering and counts --------------------------------------------------


def test_a_dim_header_counts_what_the_diff_adds_and_removes() -> None:
    diff = "@@ -1,2 +1,3 @@\n-a\n+b\n+c\n d\n"
    assert _plain(diff)[0] == "  ⎿  Added 2 lines, removed 1 lines"
    assert _segments(_ansi(diff)[0]) == [("2", "  ⎿  Added 2 lines, removed 1 lines")]


def test_lines_are_numbered_from_the_hunk_header() -> None:
    rows = _plain(
        "@@ -10,3 +20,3 @@\n keep\n-totally different old\n+brand new line here\n tail\n",
    )
    assert rows[1:] == [
        "  20 keep",
        "  11 totally different old",
        "  21 brand new line here",
        "  22 tail",
    ]


def test_an_addition_does_not_advance_the_old_numbering() -> None:
    assert _plain("@@ -5 +5 @@\n+new\n-old\n")[1:] == ["  5 new", "  5 old"]


def test_a_second_hunk_restarts_the_numbering() -> None:
    rows = _plain("@@ -1 +1 @@\n-x\n@@ -50 +60 @@\n+y\n")
    assert rows[1:] == ["   1 x", "  60 y"]


def test_a_diff_without_a_hunk_header_numbers_from_zero() -> None:
    assert _plain("+added\n same\n")[1:] == ["  0 added", "  1 same"]


def test_file_headers_are_neither_shown_nor_counted() -> None:
    rows = _plain("--- a/foo\n+++ b/foo\n@@ -1 +1 @@\n-old\n+new\n")
    assert rows == [
        "  ⎿  Added 1 lines, removed 1 lines",
        "  1 old",
        "  1 new",
    ]


def test_a_line_starting_like_a_header_without_a_space_is_content() -> None:
    rows = _plain("@@ -1 +1 @@\n---x\n+++y\n")
    assert rows[1:] == ["  1 --x", "  1 ++y"]


def test_an_empty_changed_line_renders() -> None:
    assert _plain("@@ -1 +1 @@\n-\n+\n")[1:] == ["  1", "  1"]


# ---- colors ----------------------------------------------------------------


def test_a_short_number_s_padding_fills_like_the_rest_of_its_line() -> None:
    rows = _ansi("@@ -99 +99 @@\n-a\n-b\n+c\n+d\n")[1:]
    for row in rows:
        assert _visible(row).startswith("  ")
        assert len(_styled(row, _REMOVED) + _styled(row, _ADDED)) == 38


def test_a_changed_line_fills_from_its_number_to_the_edge_not_the_indent() -> None:
    removed, added = _ansi("@@ -1 +1 @@\n-abcdefgh\n+uvwxyzst\n")[1:]
    for row, background, text in (
        (removed, _REMOVED, "abcdefgh"),
        (added, _ADDED, "uvwxyzst"),
    ):
        assert len(_visible(row)) == 40
        assert _visible(row).startswith("  ")
        assert _styled(row, background) == _visible(row)[2:]
        assert text in _styled(row, background)
        assert _styled(row, _GUTTER).strip() == "1"
    assert not any(
        _REMOVED_WORD in row or _ADDED_WORD in row for row in (removed, added)
    )


def test_a_changed_line_fills_to_80_in_a_wide_window() -> None:
    rows = _ansi("@@ -1 +1 @@\n-a\n+b\n", width=200)[1:]
    assert [len(_visible(r)) for r in rows] == [80, 80]


def test_a_long_changed_line_widens_every_fill_to_fit_it() -> None:
    rows = _ansi(f"@@ -1 +1 @@\n-a\n+{'b' * 100}\n", width=200)[1:]
    assert [len(_visible(r)) for r in rows] == [104, 104]


def test_code_past_100_wraps_under_its_content_not_its_gutter() -> None:
    rows = _ansi(f"@@ -1 +1 @@\n+{'b' * 150}\n", width=200)[1:]
    visible = [_visible(r) for r in rows]
    assert [len(v) for v in visible] == [104, 104]
    assert visible[0].endswith("b" * 100)
    assert visible[1].startswith(" " * 4 + "b" * 50)


def test_a_context_line_has_a_dim_gutter_and_no_background() -> None:
    (row,) = _ansi("@@ -1 +1 @@\n context\n")[1:]
    segments = _segments(row)
    assert _visible(row).startswith("  1")
    assert segments[0] == ("2", "1 ")
    assert not any("48;" in codes for codes, _ in segments)


def test_a_small_change_marks_only_its_changed_words() -> None:
    removed, added = _ansi("@@ -1 +1 @@\n-value = 1 + 2 + 3\n+value = 1 + 2 + 4\n")[1:]
    assert _styled(removed, _REMOVED_WORD) == "3"
    assert _styled(added, _ADDED_WORD) == "4"
    assert "value = 1 + 2 + " in _styled(removed, _REMOVED)
    assert "value = 1 + 2 + " in _styled(added, _ADDED)
    for row in (removed, added):
        assert len(_visible(row)) == 40
        assert _GUTTER in _segments(row)[0][0]
    assert _visible(added).endswith(" " * 10)
    assert _segments(added)[-1][0].endswith(_ADDED)


def test_a_word_diff_keeps_each_side_s_own_text() -> None:
    diff = "@@ -1 +1 @@\n-result = compute(alpha.beta, 1, 2, 3, 4, 5, 6)\n+result = compute(gamma-delta, 1, 2, 3, 4, 5, 6)\n"
    rows = _plain(diff, width=60)
    assert rows[1:] == [
        "  1 result = compute(alpha.beta, 1, 2, 3, 4, 5, 6)",
        "  1 result = compute(gamma-delta, 1, 2, 3, 4, 5, 6)",
    ]
    added = _ansi(diff, width=60)[2]
    assert _styled(added, _ADDED_WORD) == "gamma-delta"


def test_a_deletion_or_insertion_inside_a_line_is_marked() -> None:
    removed, added = _ansi("@@ -1 +1 @@\n-x = 1 + 2\n+x = 1\n")[1:]
    assert _styled(removed, _REMOVED_WORD) == " + 2"
    assert _styled(added, _ADDED_WORD) == ""
    removed, added = _ansi("@@ -1 +1 @@\n-x = 1\n+x = 1 + 2\n")[1:]
    assert _styled(added, _ADDED_WORD) == " + 2"


@pytest.mark.parametrize(
    ("old", "new"),
    [("abcd", "wxyz"), ("a b c", "x y z")],
    ids=["one-word", "many-small-words"],
)
def test_lines_that_differ_mostly_are_not_word_marked(old: str, new: str) -> None:
    """The share is of every changed word together, not of the largest one."""
    rows = _ansi(f"@@ -1 +1 @@\n-{old}\n+{new}\n")[1:]
    assert not any(_REMOVED_WORD in row or _ADDED_WORD in row for row in rows)


def test_the_style_picks_the_colors() -> None:
    style = DiffStyle(
        added="on rgb(1,2,3)",
        removed="on rgb(4,5,6)",
        added_word="on rgb(7,8,9)",
        gutter="rgb(10,11,12)",
    )
    rows = _ansi(
        "@@ -1 +1 @@\n-all of this goes\n+every bit replaced\n context\n",
        style=style,
    )
    assert "48;2;4;5;6" in rows[1]
    assert "48;2;1;2;3" in rows[2]
    assert "38;2;10;11;12" in rows[2]
    rows = _ansi("@@ -1 +1 @@\n-x = 1 + 2\n+x = 1 + 3\n", style=style)
    assert _styled(rows[2], "48;2;7;8;9") == "3"


def test_a_threshold_of_one_marks_words_in_any_change() -> None:
    rows = _ansi("@@ -1 +1 @@\n-alpha\n+omega\n", style=DiffStyle(word_threshold=1.0))
    assert _styled(rows[1], _REMOVED_WORD) == "alpha"


@pytest.mark.parametrize("word", [False, True])
def test_a_line_that_fills_the_console_takes_one_row(word: bool) -> None:
    content = "abcdefghi j" if word else "abcdefghijk"
    other = "abcdefghi k" if word else "zzzzzzzzzzz"
    rows = _ansi(f"@@ -1 +1 @@\n-{other}\n+{content}\n", width=15)
    body = [row for row in rows if _GUTTER in row]
    assert len(body) == 2
    assert _visible(rows[-1]) == f"  1 {content}"


def test_a_narrow_console_wraps_without_padding() -> None:
    rows = _plain("@@ -1 +1 @@\n context that is wider than the console\n", width=10)
    assert "".join(rows).replace(" ", "").endswith("1contextthatiswiderthantheconsole")


# ---- pairing ---------------------------------------------------------------


def test_a_run_of_removals_pairs_in_order_with_the_additions_after_it() -> None:
    diff = "@@ -1,3 +1,1 @@\n-alpha one\n-beta two\n-gamma three\n+alpha one!\n"
    assert _plain(diff)[1:] == [
        "  1 alpha one",
        "  2 beta two",
        "  3 gamma three",
        "  1 alpha one!",
    ]
    rows = _ansi(diff)
    assert _styled(rows[4], _ADDED_WORD) == "!"
    assert not any(_REMOVED_WORD in row for row in rows)


def test_pairs_are_found_after_context_and_between_runs() -> None:
    rows = _ansi("@@ -1,4 +1,4 @@\n keep\n-a b c\n+a b d\n mid\n-x y 1\n+x y 2\n")
    assert _styled(rows[3], _ADDED_WORD) == "d"
    assert _styled(rows[6], _ADDED_WORD) == "2"


def test_a_removal_at_the_end_pairs_with_nothing() -> None:
    rows = _ansi("@@ -1,2 +1,1 @@\n keep\n-gone\n")
    assert _styled(rows[2], _REMOVED) != ""
    assert _REMOVED_WORD not in rows[2]


def test_an_unchanged_line_in_a_run_pairs_with_its_twin() -> None:
    rows = _ansi("@@ -1,2 +1,2 @@\n-same\n-tail b\n+same\n+tail c\n")
    assert _styled(rows[2], _REMOVED_WORD) == "b"
    assert _styled(rows[4], _ADDED_WORD) == "c"


def test_an_inserted_line_does_not_stop_the_pairing_after_it() -> None:
    rows = _ansi("@@ -1,2 +1,3 @@\n-same\n-tail b\n+new\n+same\n+tail c\n")
    assert _styled(rows[5], _ADDED_WORD) == "c"
    assert _styled(rows[2], _REMOVED_WORD) == "b"


# ---- highlighting ----------------------------------------------------------


def test_the_file_name_picks_the_highlighting() -> None:
    diff = "@@ -1 +1 @@\n-def foo(): pass\n+def bar(): pass\n"
    assert _ansi(diff, path="x.py") != _ansi(diff, path="x.unknownext")
    assert _ansi(diff, path="x.unknownext") == _ansi(diff)


def test_highlight_source_colors_a_known_language() -> None:
    text = highlight_source("ls -la | grep x", "bash")
    assert text.plain == "ls -la | grep x"
    assert text.spans


@pytest.mark.parametrize(
    ("code", "plain"),
    [("echo hi\n", "echo hi"), ("\necho hi", "\necho hi"), ("echo hi  ", "echo hi  ")],
)
def test_highlight_source_keeps_the_text_but_its_final_newline(
    code: str,
    plain: str,
) -> None:
    assert highlight_source(code, "bash").plain == plain


def test_highlight_source_is_plain_for_an_unknown_language_or_no_code() -> None:
    unknown = highlight_source("ls", "no-such-language")
    assert (unknown.plain, unknown.spans) == ("ls", [])
    assert highlight_source("", "bash").plain == ""


# ---- splice_diff -----------------------------------------------------------


def test_splice_diff_numbers_each_hunk_for_the_file_after_the_edit() -> None:
    result = FileEditResult(
        call_id="1",
        path="a.py",
        edits=(
            Splice(before="a\nb", after="c", start=3),
            Splice(before="", after="x\ny", start=10),
            Splice(after="z"),
            Splice(before="w", start=20),
        ),
    )
    assert splice_diff(result).split("\n") == [
        "@@ -3,2 +3,1 @@",
        "-a",
        "-b",
        "+c",
        "@@ -10,0 +9,2 @@",
        "+x",
        "+y",
        "@@ -1,0 +2,1 @@",
        "+z",
        "@@ -20,1 +22,0 @@",
        "-w",
    ]


def test_an_edit_with_no_splices_is_an_empty_diff() -> None:
    assert splice_diff(FileEditResult(call_id="1", path="a.py")) == ""


def test_a_write_is_a_diff_adding_every_line_from_one() -> None:
    write = FileWriteResult(call_id="1", path="a.py", content="a\nb\n")
    assert write_diff(write) == "@@ -0,0 +1,2 @@\n+a\n+b"
    assert write_diff(FileWriteResult(call_id="2")) == "@@ -0,0 +1,0 @@"


def test_a_cut_diff_keeps_its_head_and_tail_and_counts_every_line() -> None:
    diff = "@@ -0,0 +1,5 @@\n+a\n+b\n+c\n+d\n+e"
    out = io.StringIO()
    console = Console(file=out, width=40, color_system=None, highlight=False)
    render_diff(console, diff, lines=(1, 2))
    assert [row.rstrip() for row in out.getvalue().splitlines()] == [
        "  ⎿  Added 5 lines, removed 0 lines",
        "  1 a",
        "    ⋯ 2 lines ⋯",
        "  4 d",
        "  5 e",
    ]
    assert _plain(diff)[1:] == [f"  {i} {c}" for i, c in enumerate("abcde", 1)]


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
