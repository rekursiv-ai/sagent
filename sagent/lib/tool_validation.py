"""Tool directive schema validation shared by agent and provider bridges."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

from sagent.lib.custom_json import JSON, validate_json_schema


_MAX_ADVERTISED_KEYS = 50  # config-globals: ignore -- max advertised keys in error


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
        f"InputValidationError: {tool_name} failed due to the following {plural}:",
        *issues,
    ]
    required = _schema_strings(schema.get("required"))
    props_raw = schema.get("properties")
    accepted = [str(k) for k in props_raw] if isinstance(props_raw, Mapping) else []
    if required:
        keys = ", ".join(f"`{k}`" for k in required)
        parts.append(f"\n{tool_name} requires: {keys}.")
    # ``validate_json_schema`` returns this literal prefix when
    # ``additionalProperties: false`` rejects a key; branch on it because the
    # upstream function returns plain strings. Changes to that prefix must be
    # reflected here.
    if any(issue.startswith("Unexpected parameter") for issue in issues) and accepted:
        parts.append(f"{tool_name} accepts: {_format_keys(accepted)}.")
    parts.append(
        "\nThis tool call was not executed because its JSON directive was missing "
        "or misstated required fields. Do not repeat the same empty or incomplete "
        "call. Either retry this tool with the required fields, choose a different "
        "tool that fits the task, or explain why the required value is unavailable."
    )
    return "\n".join(parts)


def _schema_strings(value: object) -> list[str]:
    """Return string items from a schema list field."""
    if not isinstance(value, (list, tuple)):
        return []
    items = cast(Sequence[object], value)
    return [item for item in items if isinstance(item, str)]


def _format_keys(keys: list[str]) -> str:
    """Render a comma-separated, backtick-quoted key list capped at 50 names."""
    if len(keys) <= _MAX_ADVERTISED_KEYS:
        return ", ".join(f"`{k}`" for k in keys)
    shown = ", ".join(f"`{k}`" for k in keys[:_MAX_ADVERTISED_KEYS])
    return f"{shown} and {len(keys) - _MAX_ADVERTISED_KEYS} more"
