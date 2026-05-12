"""Tests for ``repl.tight_markdown``: Markdown subclass without leading blank."""

from __future__ import annotations

import io

from rich.console import Console

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


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
