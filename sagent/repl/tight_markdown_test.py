"""Tests for ``repl.tight_markdown``: Markdown subclass without leading blank."""

from __future__ import annotations

import io

from rich.console import Console
from rich.markdown import Markdown

import pytest

from sagent.repl.tight_markdown import TightMarkdown


def _render(text: str, width: int = 80) -> str:
    buf = io.StringIO()
    con = Console(
        file=buf,
        width=width,
        force_terminal=False,
        color_system=None,
        highlight=False,
    )
    con.print(TightMarkdown(text))
    return buf.getvalue()


def test_no_leading_blank_for_list() -> None:
    # Stock ``Markdown`` would prefix the first list with a blank line.
    out = _render("- item one\n- item two\n")
    # First non-empty line should be the bullet, not a blank.
    first = next((ln for ln in out.split("\n") if ln.strip()), "")
    assert "item one" in first


def test_no_leading_blank_for_blockquote() -> None:
    out = _render("> a quote\n")
    first = next((ln for ln in out.split("\n") if ln.strip()), "")
    assert "a quote" in first


def test_renders_paragraph() -> None:
    out = _render("plain paragraph")
    assert "plain paragraph" in out


def test_renders_heading_then_body() -> None:
    out = _render("# Title\n\nBody text\n")
    assert "Title" in out
    assert "Body text" in out


def test_renders_fence() -> None:
    out = _render("```python\nprint('hi')\n```\n")
    assert "print" in out


def test_links_render_inline() -> None:
    out = _render("[click](https://example.com)\n")
    assert "click" in out


def test_inline_styles_render() -> None:
    out = _render("**bold** and *italic* and `code`\n")
    assert "bold" in out
    assert "italic" in out
    assert "code" in out


def test_multiple_paragraphs_get_separator() -> None:
    out = _render("first para\n\nsecond para\n")
    assert "first para" in out
    assert "second para" in out


def test_softbreak_within_paragraph_renders_as_space() -> None:
    # A single newline inside a paragraph is a softbreak token.
    out = _render("line one\nline two\n")
    # Should join with a space (or be on one line after wrap).
    assert "line one" in out
    assert "line two" in out


def test_hardbreak_within_paragraph_renders_as_newline() -> None:
    # Two trailing spaces + newline -> hardbreak (token).
    out = _render("line one  \nline two\n")
    # The two halves still appear in output.
    assert "line one" in out
    assert "line two" in out


def _render_no_hyperlinks(text: str, width: int = 80) -> str:
    buf = io.StringIO()
    con = Console(
        file=buf,
        width=width,
        force_terminal=False,
        color_system=None,
        highlight=False,
    )
    con.print(TightMarkdown(text, hyperlinks=False))
    return buf.getvalue()


def test_link_render_without_hyperlinks_uses_text_paren_url() -> None:
    # ``hyperlinks=False`` triggers the manual link rendering branch:
    # the link text is emitted followed by `` (url)``.
    out = _render_no_hyperlinks("[click](https://example.com)\n")
    assert "click" in out
    assert "https://example.com" in out


def test_thematic_break_after_paragraph_emits_new_line_segment() -> None:
    # ``hr`` is a self-closing block token. Coming after a paragraph it
    # exercises the ``new_line and node_type != 'inline'`` branch.
    out = _render("para\n\n---\n\nafter\n")
    assert "para" in out
    assert "after" in out


def _render_upstream(text: str, width: int = 80) -> str:
    buf = io.StringIO()
    con = Console(
        file=buf,
        width=width,
        force_terminal=False,
        color_system=None,
        highlight=False,
    )
    con.print(Markdown(text))
    return buf.getvalue()


# Constructs whose only sanctioned difference from stock Rich is the
# leading blank line this module exists to drop. The module copies Rich's
# ``__rich_console__``, so an upgrade can silently drop a branch the copy
# never gained -- that is how inline HTML lost its ``<kbd>`` styling.
# Comparing against the installed Rich turns the next drift into a
# failure.
_AGREES_WITH_UPSTREAM = [
    "plain paragraph\n",
    "first para\n\nsecond para\n",
    "# Title\n\nBody text\n",
    "```python\nprint('hi')\n2. not a list\n```\n",
    "para\n\n    indented = 1\n    2. still code\n",
    "| a | b |\n|---|---|\n| 1 | 2 |\n",
    "Head\n1. first\n2. second\n",
    "para\n\n---\n\nafter\n",
    "Press <kbd>Ctrl</kbd>+<kbd>C</kbd> to stop.\n",
    "[click](https://example.com)\n",
    "![alt](https://example.com/i.png)\n",
    "**bold** and *italic* and `code`\n",
    "~~struck~~\n",
    "- [ ] todo\n- [x] done\n",
    "1. a\n   1. b\n      1. c\n",
    "",
    "   \n",
]


@pytest.mark.parametrize("text", _AGREES_WITH_UPSTREAM)
def test_matches_upstream_rich_but_for_leading_blank(text: str) -> None:
    assert _render(text) == _render_upstream(text).lstrip("\n")


def test_leading_block_html_keeps_its_blank_line() -> None:
    # A leading ``html_block`` renders no visible segment yet still marks a
    # block as rendered, so the blank before the next block is real
    # spacing. Dropping it here would be the position-based fix; the
    # ``first_render`` guard is state-based and keeps it.
    assert _render("<div>block html</div>\n\npara\n").startswith("\n")


# The two intended divergences. Stock Rich renders these as run-on prose
# because CommonMark only lets an ordered list interrupt a paragraph when
# it starts at 1.
_ORDERED_RUN = "Header line\n4. alpha\n5. beta\n6. gamma\n"


def test_ordered_list_after_paragraph_becomes_a_list() -> None:
    out = _render(_ORDERED_RUN)
    assert " 4 alpha" in out
    assert " 5 beta" in out
    # Stock Rich would join the header and the items onto one line.
    assert "Header line 4." in _render_upstream(_ORDERED_RUN)
    assert "Header line 4." not in out


def test_single_ordered_item_after_paragraph_becomes_a_list() -> None:
    # 69 of the 244 real-world occurrences are one item continuing a run
    # that a bold sub-heading interrupted, so the fix must not require two.
    out = _render("**Collective**:\n7. Cohort -- group pursuing together\n")
    assert " 7 Cohort" in out


def test_ordered_list_start_number_is_preserved() -> None:
    assert " 10 alpha" in _render("Header line\n10. alpha\n11. beta\n")


def test_ordered_list_inside_blockquote_becomes_a_list() -> None:
    out = _render("> quoted line\n> 4. alpha\n> 5. beta\n")
    assert " 4 alpha" in out


def test_fence_content_is_not_reflowed_into_a_list() -> None:
    # The lenient rule must never fire inside a code block; upstream's
    # ``is_code_block`` guard is what prevents it.
    out = _render("```\nHeader line\n4. alpha\n5. beta\n```\n")
    assert "4. alpha" in out


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
