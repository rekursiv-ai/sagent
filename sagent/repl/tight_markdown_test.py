"""Tests for TightMarkdown — no spurious leading blank line."""

from __future__ import annotations

from io import StringIO

from rich.console import Console

from sagent.repl.tight_markdown import TightMarkdown


def _render(markup: str) -> str:
    f = StringIO()
    Console(file=f, width=80).print(TightMarkdown(markup))
    return f.getvalue()


class TestNoLeadingBlank:
    def test_list_only(self) -> None:
        out = _render("1. One\n2. Two")
        assert not out.startswith("\n")
        assert " 1 " in out
        assert " 2 " in out

    def test_bullet_list_only(self) -> None:
        out = _render("- A\n- B")
        assert not out.startswith("\n")

    def test_blockquote_only(self) -> None:
        out = _render("> quoted text")
        assert not out.startswith("\n")

    def test_paragraph_only(self) -> None:
        out = _render("Just a paragraph.")
        assert not out.startswith("\n")


class TestInterBlockSpacing:
    def test_paragraph_then_list(self) -> None:
        out = _render("Para.\n\n1. One\n2. Two")
        assert out.startswith("Para.")
        lines = out.split("\n")
        para_end = next(i for i, ln in enumerate(lines) if "Para." in ln)
        list_start = next(i for i, ln in enumerate(lines) if " 1 " in ln)
        assert list_start - para_end == 2  # one blank line between

    def test_paragraph_then_paragraph(self) -> None:
        out = _render("First.\n\nSecond.")
        lines = out.split("\n")
        first = next(i for i, ln in enumerate(lines) if "First." in ln)
        second = next(i for i, ln in enumerate(lines) if "Second." in ln)
        assert second - first == 2

    def test_list_then_paragraph(self) -> None:
        out = _render("1. One\n2. Two\n\nAfter.")
        lines = out.split("\n")
        last_item = max(i for i, ln in enumerate(lines) if " 2 " in ln)
        after = next(i for i, ln in enumerate(lines) if "After." in ln)
        assert after - last_item == 2

    def test_full_sequence(self) -> None:
        out = _render("Para.\n\n1. A\n2. B\n\nEnd.")
        assert not out.startswith("\n")
        assert "Para." in out
        assert " 1 " in out
        assert "End." in out


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
