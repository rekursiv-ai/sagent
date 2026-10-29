"""Tests for tools made from functions: the signature is the schema."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

from treekle import PlainTree, from_plain

import pytest

from sagent.session.types import UncategorizedToolResult
from sagent.tools import types
from sagent.tools.function import tool


if TYPE_CHECKING:
    # Named for the checker only, as a module offering a tool would: the
    # ``context`` parameter's annotation must not need it at run time.
    from sagent.tools.types import Tool, ToolContext


def _context(max_result_chars: int = 200_000) -> ToolContext:
    return types.ToolContext(cwd=Path("/"), max_result_chars=max_result_chars)


def greet(name: str, times: int = 1, *, shout: bool = False) -> str:
    """Say hello.

    Args:
      name: Who to greet.
      times: How many times, from 1. Said once
        when omitted.
      shout: Whether to shout.

    Returns:
      greeting: The greeting.

    """
    text = " ".join(["hello", name] * times)
    return text.upper() if shout else text


def test_the_schema_is_the_signature_with_the_docstrings_argument_notes() -> None:
    made = tool(greet)

    assert made.name == "greet"
    assert made.description.startswith("Say hello.")
    assert made.schema == {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Who to greet."},
            "times": {
                "type": "integer",
                "description": "How many times, from 1. Said once when omitted.",
            },
            "shout": {"type": "boolean", "description": "Whether to shout."},
        },
        "required": ["name"],
        "additionalProperties": False,
    }


def test_a_name_description_and_schema_may_be_given() -> None:
    schema: dict[str, PlainTree] = {"type": "object", "properties": {}}
    made = tool(greet, name="Greet", description="Greets.", schema=schema)

    assert (made.name, made.description, made.schema) == ("Greet", "Greets.", schema)


@pytest.mark.asyncio
async def test_a_call_passes_the_arguments_and_returns_the_text() -> None:
    made: Tool = tool(greet, name="Greet")

    result = await made.call(_context(), {"name": "ann", "times": 2})

    assert result == UncategorizedToolResult(call_id="", content="hello ann hello ann")
    assert made.summary({"name": "ann"}) == "Greet"
    assert made.serialize_key(_context(), {"name": "ann"}) is None


@pytest.mark.asyncio
async def test_an_async_function_is_awaited() -> None:
    async def shout(word: Literal["hi", "yo"]) -> str:
        """Shout.

        Args:
          word: What to shout.

        """
        return word.upper()

    result = await tool(shout).call(_context(), {"word": "yo"})

    assert result == UncategorizedToolResult(call_id="", content="YO")


@pytest.mark.asyncio
async def test_a_function_taking_the_context_gets_the_callers() -> None:
    seen: list[ToolContext] = []

    def where(context: ToolContext, path: str) -> str:
        """Say where a path is.

        Args:
          context: The caller's state.
          path: A path.

        """
        seen.append(context)
        return str(context.path(path))

    made = tool(where)
    context = _context()

    assert "context" not in from_plain(made.schema["properties"], dict[str, object])
    assert await made.call(context, {"path": "a"}) == UncategorizedToolResult(
        call_id="",
        content="/a",
    )
    assert seen == [context]


@pytest.mark.asyncio
async def test_a_long_text_is_cut_to_what_one_result_holds() -> None:
    def long() -> str:
        """Say a lot."""
        return "x" * 50

    result = await tool(long).call(_context(max_result_chars=10), {})

    assert isinstance(result, UncategorizedToolResult)
    assert result.content == "x" * 10 + "\n... (40 chars omitted)"


@pytest.mark.asyncio
async def test_a_record_returned_is_returned_as_it_is() -> None:
    record = UncategorizedToolResult(call_id="", content="done", is_error=True)

    def fail() -> UncategorizedToolResult:
        """Fail."""
        return record

    assert await tool(fail).call(_context(), {}) is record


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("arguments", "error"),
    [
        ({}, "Missing argument 'name'."),
        ({"name": "ann", "loud": True}, "Unknown arguments: loud."),
        ({"name": 3}, "'name' must be string, got 3."),
    ],
    ids=["missing", "unknown", "mistyped"],
)
async def test_arguments_not_fitting_the_signature_are_refused(
    arguments: dict[str, PlainTree],
    error: str,
) -> None:
    with pytest.raises(ValueError, match=error):
        await tool(greet).call(_context(), arguments)


def test_a_type_with_no_json_schema_is_refused_when_the_tool_is_made() -> None:
    def odd(when: Path) -> str:
        """Odd.

        Args:
          when: A path, which JSON has no type for.

        """
        return str(when)

    with pytest.raises(TypeError, match="No JSON schema"):
        tool(odd)


def test_tool_as_a_decorator_takes_the_same_settings() -> None:
    @tool(name="Ping")
    def ping() -> str:
        """Ping."""
        return "pong"

    assert ping.name == "Ping"


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
