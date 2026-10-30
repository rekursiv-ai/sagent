"""Tests for the scrollback's markdown: Rich's, but tight and list-friendly."""

from __future__ import annotations

import io
import re

from rich.console import Console
from rich.markdown import Markdown

import pytest

from sagent.tui.markdown import TightMarkdown, stable_boundary
from sagent.tui.types import Sticky


def _render(markdown: Markdown, *, width: int = 80) -> str:
    out = io.StringIO()
    Console(file=out, width=width, color_system=None, highlight=False).print(markdown)
    return out.getvalue()


def _tight(text: str, *, hyperlinks: bool = True) -> str:
    return _render(TightMarkdown(text, hyperlinks=hyperlinks))


def _rows(text: str, *, hyperlinks: bool = True) -> str:
    """Return the tight rendering without the padding Rich puts at each row's end."""
    return "\n".join(
        row.rstrip() for row in _tight(text, hyperlinks=hyperlinks).split("\n")
    )


def _rich(text: str, *, hyperlinks: bool = True) -> str:
    return _render(Markdown(text, hyperlinks=hyperlinks))


def _bare(rendered: str) -> str:
    """Return ``rendered`` without each row's end padding, which a quote widens."""
    return "\n".join(row.rstrip() for row in rendered.split("\n"))


def _ansi(markdown: Markdown) -> str:
    out = io.StringIO()
    Console(file=out, width=80, color_system="truecolor", force_terminal=True).print(
        markdown,
    )
    # Each hyperlink gets a random id; the link itself is what must match.
    return re.sub(r"id=\d+;", "id=;", out.getvalue())


# The copy of Rich's ``__rich_console__`` may differ from Rich only by the
# leading blank line, so every construct it handles renders as Rich's would.
_AGREES_WITH_RICH = [
    "plain paragraph\n",
    "first para\n\nsecond para\n",
    "# Title\n\nBody text\n",
    "## Section\n\nBody text\n",
    "```python\nprint('hi')\n2. not a list\n```\n",
    "para\n\n    indented = 1\n    2. still code\n",
    "| a | b |\n|---|---|\n| 1 | 2 |\n",
    "Head\n1. first\n2. second\n",
    "para\n\n---\n\nafter\n",
    "[click](https://example.com)\n",
    "![alt](https://example.com/i.png)\n",
    "**bold** and *italic* and `code`\n",
    "~~struck~~\n",
    "- [ ] todo\n- [x] done\n",
    "1. a\n   1. b\n      1. c\n",
    "line one  \nline two\n",
    "line one\nline two\n",
    "para\n*\nmore\n",
    "para\n2.\nmore\n",
    "para\n1.\nmore\n",
    "- item\n\n  ```\n  code\n  ```\n",
    "- item\n\n  ---\n",
    "",
    "   \n",
]


@pytest.mark.parametrize("text", _AGREES_WITH_RICH)
@pytest.mark.parametrize("hyperlinks", [True, False])
def test_it_matches_rich_but_for_a_leading_blank_line(
    text: str,
    hyperlinks: bool,
) -> None:
    assert _tight(text, hyperlinks=hyperlinks) == _rich(
        text,
        hyperlinks=hyperlinks,
    ).lstrip("\n")


@pytest.mark.parametrize("text", ["- item one\n- item two\n", "> a quote\n"])
def test_a_list_or_quote_that_opens_the_text_gets_no_blank_line(text: str) -> None:
    assert _rich(text).startswith("\n")
    assert not _tight(text).startswith("\n")
    assert _bare(_tight(text)) == _bare(_rich(text)[1:])


@pytest.mark.parametrize("text", ["- a\n\n> b\n", "- item\n\n> quote\n"])
def test_blocks_after_the_first_keep_their_blank_line(text: str) -> None:
    assert _bare(_tight(text)) == _bare(_rich(text).lstrip("\n"))
    assert "\n\n" in _tight(text)


def test_a_leading_html_block_keeps_the_blank_line_after_it() -> None:
    """The html renders nothing, yet the blank before the next block is spacing."""
    assert _tight("<div>block html</div>\n\npara\n").startswith("\n")


def test_styles_match_rich_too() -> None:
    for text in ("**bold** `code` [l](https://e.com)\n", "# T\n\n~~s~~ *i*\n"):
        assert _ansi(TightMarkdown(text)) == _ansi(Markdown(text)).removeprefix("\n")


def test_kbd_is_styled_and_other_inline_html_is_dropped() -> None:
    text = "Press <kbd>Ctrl</kbd>+<kbd>C</kbd> or <b>this</b>.\n"
    assert _rows(text) == "Press Ctrl+C or this.\n"
    styled = _ansi(TightMarkdown(text))
    assert re.search(r"\x1b\[[0-9;]+mCtrl\x1b\[0m", styled)
    assert styled == _ansi(Markdown(text))
    assert _ansi(TightMarkdown("Press Ctrl+C or this.\n")) != styled


def test_a_link_without_hyperlinks_shows_its_url() -> None:
    assert _rows("[click](https://example.com)\n", hyperlinks=False) == (
        "click (https://example.com)\n"
    )


def test_a_link_with_hyperlinks_hides_its_url() -> None:
    assert _rows("[click](https://example.com)\n") == "click\n"
    # Carried by the terminal's hyperlink escape, never shown.
    shown = _ansi(TightMarkdown("[click](https://example.com)\n"))
    assert shown.count("https://example.com") == 1


# CommonMark lets only a list starting at 1 interrupt a paragraph, so stock
# Rich runs a continued numbering into the paragraph above it as prose.
@pytest.mark.parametrize(
    ("text", "item"),
    [
        ("Header line\n4. alpha\n5. beta\n", " 4 alpha"),
        ("**Collective**:\n7. Cohort -- one item\n", " 7 Cohort"),
        ("Header line\n10. alpha\n11. beta\n", " 10 alpha"),
        ("> quoted line\n> 4. alpha\n> 5. beta\n", " 4 alpha"),
    ],
)
def test_an_ordered_list_may_interrupt_a_paragraph_at_any_number(
    text: str,
    item: str,
) -> None:
    assert item in _tight(text)
    assert item not in _rich(text)


@pytest.mark.parametrize(
    ("window", "code", "fill"),
    [
        (200, "x = 1", 80),
        (200, "y" * 100, 102),
        (200, "y" * 150, 102),
        (90, "y" * 150, 90),
        (60, "x = 1", 60),
    ],
)
def test_a_code_block_wraps_its_code_at_100_and_fills_at_least_80(
    window: int,
    code: str,
    fill: int,
) -> None:
    rows = _render(TightMarkdown(f"```\n{code}\n```"), width=window).splitlines()
    assert {len(row) for row in rows} == {fill}


def test_a_horizontal_rule_fills_like_a_code_block() -> None:
    rows = _render(TightMarkdown("a\n\n---\n\nb\n"), width=200).splitlines()
    assert [row.rstrip() for row in rows if "-" in row] == ["-" * 80]


def test_a_title_centers_within_the_fill_not_the_window() -> None:
    (row,) = _render(TightMarkdown("# Title\n"), width=200).splitlines()
    assert row.rstrip() == " " * 37 + "Title"


def test_a_rule_and_a_title_widen_with_the_turn() -> None:
    sticky = Sticky(row=100)
    rows = _render(TightMarkdown("# Ti\n\n---\n", sticky=sticky), width=200)
    assert [row.rstrip() for row in rows.splitlines() if row.strip()] == [
        " " * 49 + "Ti",
        "-" * 100,
    ]


def _edges(text: str, *, width: int = 200) -> set[int]:
    """Return the columns where the rows painted with a background end."""
    console = Console(file=io.StringIO(), width=width, color_system="truecolor")
    ends: set[int] = set()
    for line in console.render_lines(TightMarkdown(text), console.options):
        column = end = 0
        for segment in line:
            column += segment.cell_length
            if segment.style is not None and segment.style.bgcolor is not None:
                end = column
        ends |= {end} - {0}
    return ends


@pytest.mark.parametrize(
    "text",
    [
        "```\nx = 1\n```\n",
        "1. **The rule:**\n   ```\n   x = 1\n   ```\n",
        "- item\n\n  > ```\n  > x = 1\n  > ```\n",
        "> > ```\n> > x = 1\n> > ```\n",
    ],
)
def test_a_nested_code_block_ends_where_a_top_level_one_does(text: str) -> None:
    assert _edges(text) == {80}


def test_a_nested_rule_and_title_end_where_top_level_ones_do() -> None:
    rows = _render(TightMarkdown("- # Ti\n\n  ---\n"), width=200).splitlines()
    assert [row.rstrip() for row in rows if row.strip()] == [
        " • " + " " * 37 + "Ti",
        "   " + "-" * 77,
    ]


def test_a_quote_gives_up_columns_only_on_its_left() -> None:
    (row,) = _render(TightMarkdown("> " + "q " * 50), width=40).splitlines()[:1]
    assert len(row) == 40
    assert row.rstrip() == "▌ " + "q " * 18 + "q"


def test_a_fence_is_never_reflowed_into_a_list() -> None:
    assert "4. alpha" in _tight("```\nHeader line\n4. alpha\n5. beta\n```\n")


def test_the_reparse_keeps_strikethrough_and_tables() -> None:
    """The lenient list rule needs a parser of its own, with Rich's extensions."""
    assert "~~" not in _tight("~~struck~~\n")
    table = _rows("| a | b |\n|---|---|\n| 1 | 2 |\n")
    assert "|" not in table
    assert [row.split() for row in table.split("\n") if row.strip()] == [
        ["a", "b"],
        ["────"],
        ["1", "2"],
    ]


@pytest.mark.parametrize(
    ("text", "cut"),
    [
        ("", 0),
        ("hello world", 0),
        ("first para\n\nsecond", len("first para\n\n")),
        ("p1\n\np2\n\np3", len("p1\n\np2\n\n")),
        ("- a\n- b\n\npara", len("- a\n- b\n\n")),
        ("para\n\n```\nunclosed", len("para\n\n")),
        ("# head\nbody", len("# head\n")),
    ],
)
def test_the_stable_boundary_is_where_the_last_block_starts(
    text: str,
    cut: int,
) -> None:
    assert stable_boundary(text) == cut


def test_a_nested_block_is_not_a_boundary() -> None:
    assert stable_boundary("- a\n\n  nested\n") == 0


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
