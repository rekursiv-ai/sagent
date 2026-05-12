"""Tests for ``repl.render_diff``: diff rendering + stable-boundary detection."""

from __future__ import annotations

import io

from rich.console import Console

from sagent.repl.render_diff import (
    find_stable_boundary,
    render_diff_detail,
)


def _capture(width: int = 80) -> tuple[Console, io.StringIO]:
    buf = io.StringIO()
    return (
        Console(
            file=buf,
            width=width,
            force_terminal=False,
            color_system=None,
            highlight=False,
        ),
        buf,
    )


def test_find_stable_boundary_single_block_returns_zero() -> None:
    # One paragraph -> nothing committed yet.
    assert find_stable_boundary("hello world") == 0


def test_find_stable_boundary_two_paragraphs() -> None:
    text = "first para\n\nsecond"
    offset = find_stable_boundary(text)
    assert offset > 0
    assert text[:offset].startswith("first para")


def test_find_stable_boundary_three_blocks_commits_first_two() -> None:
    text = "p1\n\np2\n\np3"
    offset = find_stable_boundary(text)
    # Last block is open; everything before should commit.
    assert text[:offset].startswith("p1")
    assert "p2" in text[:offset]


def test_render_diff_detail_emits_header_count() -> None:
    con, buf = _capture()
    diff = "@@ -1,2 +1,2 @@\n-foo\n+bar\n unchanged\n"
    render_diff_detail(con, diff)
    out = buf.getvalue()
    assert "Added 1 lines" in out
    assert "removed 1 lines" in out


def test_render_diff_detail_with_filename_uses_lexer() -> None:
    con, buf = _capture(width=120)
    diff = "@@ -1,2 +1,2 @@\n-def foo():\n+def bar():\n    pass\n"
    render_diff_detail(con, diff, file_path="x.py")
    out = buf.getvalue()
    assert "Added 1 lines" in out
    # Some marker of the function name lands in output.
    assert "bar" in out
    assert "foo" in out


def test_render_diff_detail_unknown_extension_falls_back() -> None:
    con, buf = _capture()
    # Filename with no known lexer should still render.
    diff = "@@ -1 +1 @@\n-x\n+y\n"
    render_diff_detail(con, diff, file_path="x.unknownext")
    out = buf.getvalue()
    assert "Added 1 lines" in out


def test_render_diff_detail_large_change_falls_back_to_line_diff() -> None:
    con, buf = _capture()
    # Word-diff threshold is 0.4: build a heavy change.
    diff = (
        "@@ -1,2 +1,2 @@\n"
        "-totally different removed line\n"
        "+completely new added text here\n"
    )
    render_diff_detail(con, diff)
    out = buf.getvalue()
    assert "totally different removed" in out
    assert "completely new added" in out


def test_render_diff_detail_handles_no_hunk_header() -> None:
    con, buf = _capture()
    # No @@ header: line numbers start at 0; should not crash.
    diff = "+added\n unchanged\n"
    render_diff_detail(con, diff)
    out = buf.getvalue()
    assert "added" in out


def test_render_diff_detail_empty_content_line_does_not_crash() -> None:
    # Empty +/- lines exercise the ``if not code: return Text('')``
    # branch in ``_highlight``.
    con, buf = _capture()
    diff = "@@ -1,2 +1,2 @@\n-\n+\n"
    render_diff_detail(con, diff)
    out = buf.getvalue()
    # Header still emits totals.
    assert "Added 1 lines" in out


def test_render_diff_detail_pure_deletion_block() -> None:
    # ``-`` block with no following ``+`` block: ``_pair_word_diffs``
    # hits the ``if a_start == a_end: continue`` branch (line 327).
    # Also exercises ``_word_diff_pair``'s ``delete`` tag (lines 252-254).
    con, buf = _capture()
    diff = "@@ -1,3 +1,2 @@\n keep\n-going to be deleted\n also keep\n"
    render_diff_detail(con, diff)
    out = buf.getvalue()
    assert "going to be deleted" in out


def test_render_diff_detail_pure_addition_block() -> None:
    # Pure addition: ``_word_diff_pair`` only fires when there's a paired
    # ``-`` block, so this skips the pairing entirely.
    con, buf = _capture()
    diff = "@@ -1,1 +1,2 @@\n keep\n+inserted line\n"
    render_diff_detail(con, diff)
    out = buf.getvalue()
    assert "inserted line" in out


def test_render_diff_detail_imbalanced_replace_aligns_pairs() -> None:
    # 3 removes paired with 1 add: ``_align_blocks`` enters the ``replace``
    # branch with min(3, 1) == 1 pair (line 286).
    con, buf = _capture()
    diff = "@@ -1,3 +1,1 @@\n-alpha\n-beta\n-gamma\n+alpha replaced\n"
    render_diff_detail(con, diff)
    out = buf.getvalue()
    assert "alpha" in out
    assert "beta" in out
    assert "gamma" in out


def test_render_diff_detail_word_diff_with_insert_only() -> None:
    # Word-diff pair where the differing tokens are pure ``insert`` opcodes
    # (no replace). Covers lines 256-258 in ``_word_diff_pair``.
    con, buf = _capture()
    diff = "@@ -1,1 +1,1 @@\n-x = 1\n+x = 1 + 2\n"
    render_diff_detail(con, diff)
    out = buf.getvalue()
    assert "x = 1" in out


def test_render_diff_detail_word_diff_with_delete_only() -> None:
    # Word-diff pair where the differing tokens are pure ``delete``
    # opcodes (the added line is a strict prefix of the removed line).
    # Covers lines 252-254 in ``_word_diff_pair``.
    con, buf = _capture()
    diff = "@@ -1,1 +1,1 @@\n-x = 1 + 2\n+x = 1\n"
    render_diff_detail(con, diff)
    out = buf.getvalue()
    assert "x = 1" in out


def test_render_diff_detail_narrow_width_no_padding() -> None:
    # When the line equals or exceeds ``width``, the no-padding branch
    # of ``_render_diff_line`` fires (line 363) and the same branch in
    # ``_render_word_diff_line`` (line 395). Use a context (unchanged)
    # line so the non-word-diff path also fires.
    con, buf = _capture(width=10)
    diff = (
        "@@ -1,2 +1,2 @@\n"
        " unchanged context line that overflows width\n"
        "-abcdef ghijkl mnopqr stuv\n"
        "+abcdef ghijkl mnopqr stuw\n"
    )
    render_diff_detail(con, diff)
    out = buf.getvalue()
    assert "abcdef" in out
    assert "unchanged" in out


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
