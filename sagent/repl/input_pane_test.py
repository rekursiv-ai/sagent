"""Tests for repl.input_pane."""

from __future__ import annotations

from sagent.repl.input_pane import _collapse_prompt_preview


class TestCollapsePromptPreview:
    def test_single_line_passthrough(self) -> None:
        assert _collapse_prompt_preview("hello") == "hello"

    def test_long_single_line_truncated(self) -> None:
        got = _collapse_prompt_preview("x" * 100)
        assert got.endswith("…")
        assert len(got) == 60  # _PROMPT_PREVIEW_WIDTH

    def test_multi_line_adds_suffix(self) -> None:
        got = _collapse_prompt_preview("first\nsecond\nthird")
        assert got == "first (+2 more lines)"

    def test_one_extra_line_singular(self) -> None:
        assert _collapse_prompt_preview("a\nb") == "a (+1 more line)"


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
