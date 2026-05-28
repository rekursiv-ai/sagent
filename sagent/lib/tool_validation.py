"""Tool directive schema validation shared by agent and provider bridges."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

from sagent.lib.json import JSON, validate_json_schema


_INPUT_VALIDATION_PREFIX = "InputValidationError:"
_TOOL_INPUT_RECOVERY_HINT = (
    "This tool call was not executed because its JSON directive was missing or "
    "misstated required fields. Do not repeat the same empty or incomplete call. "
    "Either retry this tool with the required fields, choose a different tool "
    "that fits the task, or explain why the required value is unavailable."
)


def validate_tool_input(
    tool_name: str,
    schema: JSON,
    args: Mapping[str, object],
) -> str | None:
    """Pre-check tool args against a directive schema.

    Args:
      tool_name: Tool name used in the error header.
      schema: Tool directive schema.
      args: Directive args parsed from the model output.

    Returns:
      error: Multi-line input-validation error, or ``None`` for valid input.

    """
    issues = validate_json_schema(schema, args)
    if not issues:
        return None
    plural = "issues" if len(issues) > 1 else "issue"
    parts = [
        f"{_INPUT_VALIDATION_PREFIX} {tool_name} failed due to the following {plural}:",
        *issues,
    ]
    required = _schema_strings(schema.get("required"))
    props_raw = schema.get("properties")
    accepted = [str(k) for k in props_raw] if isinstance(props_raw, Mapping) else []
    if required:
        keys = ", ".join(f"`{k}`" for k in required)
        parts.append(f"\n{tool_name} requires: {keys}.")
    if any(issue.startswith("Unexpected parameter") for issue in issues) and accepted:
        keys = ", ".join(f"`{k}`" for k in accepted)
        parts.append(f"{tool_name} accepts: {keys}.")
    parts.append(f"\n{_TOOL_INPUT_RECOVERY_HINT}")
    return "\n".join(parts)


def _schema_strings(value: object) -> list[str]:
    """Return string items from a schema list field."""
    if not isinstance(value, (list, tuple)):
        return []
    items = cast(Sequence[object], value)
    return [item for item in items if isinstance(item, str)]
