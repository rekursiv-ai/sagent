"""Tests for ``tools.input_errors``: tool directive error formatting."""

from __future__ import annotations

from sagent.tools.input_errors import (
    INPUT_VALIDATION_PREFIX,
    MULTIPLE_TOOL_INPUT_ERRORS_HINT,
    TOOL_INPUT_PREFIX,
    TOOL_INPUT_RECOVERY_HINT,
    is_tool_input_error_text,
    tool_input_error_text,
)


class TestToolInputErrorText:
    def test_basic_message_includes_prefix_and_issue(self) -> None:
        out = tool_input_error_text("MyTool", "missing args")
        assert out.startswith(TOOL_INPUT_PREFIX)
        assert "MyTool failed: missing args" in out
        assert TOOL_INPUT_RECOVERY_HINT in out

    def test_lists_required_fields(self) -> None:
        out = tool_input_error_text("MyTool", "bad input", required=("path", "text"))
        assert "MyTool requires: `path`, `text`." in out

    def test_no_required_section_when_empty(self) -> None:
        out = tool_input_error_text("MyTool", "bad input")
        assert "requires:" not in out

    def test_paragraphs_blank_line_separated(self) -> None:
        out = tool_input_error_text("MyTool", "bad", required=("path",))
        assert "\n\n" in out


class TestIsToolInputErrorText:
    def test_recognises_tool_input_prefix(self) -> None:
        assert is_tool_input_error_text(f"{TOOL_INPUT_PREFIX} oops")

    def test_recognises_validation_prefix(self) -> None:
        assert is_tool_input_error_text(f"{INPUT_VALIDATION_PREFIX} oops")

    def test_other_text_rejected(self) -> None:
        assert not is_tool_input_error_text("just some output")
        assert not is_tool_input_error_text("")


def test_constants_distinct() -> None:
    assert TOOL_INPUT_PREFIX != INPUT_VALIDATION_PREFIX
    assert MULTIPLE_TOOL_INPUT_ERRORS_HINT
    assert TOOL_INPUT_RECOVERY_HINT


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
