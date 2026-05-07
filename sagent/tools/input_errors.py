"""Shared model-facing errors for malformed tool directives."""

from __future__ import annotations


TOOL_INPUT_RECOVERY_HINT = (
    "This tool call was not executed because its JSON directive was missing or "
    "misstated required fields. Do not repeat the same empty or incomplete call. "
    "Either retry this tool with the required fields, choose a different tool "
    "that fits the task, or explain why the required value is unavailable."
)
MULTIPLE_TOOL_INPUT_ERRORS_HINT = (
    "Multiple tool calls in the previous response were malformed. Stop issuing "
    "tool calls with missing required parameters. Re-read each tool's required "
    "fields and continue with only valid calls."
)
INPUT_VALIDATION_PREFIX = "InputValidationError:"
TOOL_INPUT_PREFIX = "ToolInputError:"


def tool_input_error_text(
    tool_name: str,
    issue: str,
    *,
    required: tuple[str, ...] = (),
) -> str:
    """Return a model-facing tool-input error for operation-specific fields."""
    parts = [f"{TOOL_INPUT_PREFIX} {tool_name} failed: {issue}"]
    if required:
        keys = ", ".join(f"`{k}`" for k in required)
        parts.append(f"{tool_name} requires: {keys}.")
    parts.append(TOOL_INPUT_RECOVERY_HINT)
    return "\n\n".join(parts)


def is_tool_input_error_text(text: str) -> bool:
    return text.startswith((INPUT_VALIDATION_PREFIX, TOOL_INPUT_PREFIX))
