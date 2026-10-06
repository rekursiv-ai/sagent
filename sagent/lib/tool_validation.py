"""Tool directive schema validation shared by agent and provider bridges."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from functools import cache
from typing import TYPE_CHECKING, cast

import json

import fastjsonschema

from sagent.lib.custom_json import json_unfreeze


if TYPE_CHECKING:
    from sagent.lib.custom_json import JSON


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
    try:
        schema_text = json.dumps(json_unfreeze(schema), sort_keys=True)
        _compile_schema(schema_text)(args)
    except fastjsonschema.JsonSchemaValueException as error:
        return _render_error(tool_name, schema, error)
    return None


# Only a missing-key or unknown-key failure is about the call's SHAPE. A type or bound
# failure names a value the caller did supply, so telling it the fields were "missing"
# sends it to re-add a key it already sent.
def _render_error(
    tool_name: str,
    schema: JSON,
    error: fastjsonschema.JsonSchemaValueException,
) -> str:
    """Word a validation failure by what kind of rule it broke."""
    parts = [
        f"InputValidationError: {tool_name} failed due to the following issue:",
        error.message,
    ]
    if error.rule == "required":
        keys = ", ".join(f"`{k}`" for k in _schema_strings(schema.get("required")))
        parts.append(f"\n{tool_name} requires: {keys}.")
        reason = "was missing required fields"
    elif error.rule == "additionalProperties":
        props_raw = schema.get("properties")
        accepted = list(props_raw) if isinstance(props_raw, Mapping) else []
        keys = ", ".join(f"`{k}`" for k in accepted)
        parts.append(f"\n{tool_name} accepts: {keys}.")
        reason = "named fields this tool does not accept"
    else:
        reason = "gave a field a value outside its schema"
    parts.append(
        f"\nThis tool call was not executed because its JSON directive {reason}."
        " Do not repeat the same call. Either retry this tool with a corrected"
        " directive, choose a different tool that fits the task, or explain why"
        " the required value is unavailable.",
    )
    return "\n".join(parts)


def _schema_strings(value: object) -> list[str]:
    """Return string items from a schema list field."""
    if not isinstance(value, (list, tuple)):
        return []
    items = cast(Sequence[object], value)
    return [item for item in items if isinstance(item, str)]


@cache
def _compile_schema(schema_text: str) -> Callable[[object], object]:
    return fastjsonschema.compile(cast(dict[str, object], json.loads(schema_text)))
