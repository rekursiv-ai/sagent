"""Tests for tools made from functions: the signature is the schema."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal

import weakref

from treekle import PlainTree, from_plain

import pytest

from sagent.session.types import FunctionResult, UncategorizedToolResult
from sagent.tools import types
from sagent.tools.display import DisplaySpec, ToolDisplay
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

    assert result == FunctionResult(
        call_id="",
        name="Greet",
        content="hello ann hello ann",
    )
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

    assert result == FunctionResult(call_id="", name="shout", content="YO")


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
    assert await made.call(context, {"path": "a"}) == FunctionResult(
        call_id="",
        name="where",
        content="/a",
    )
    assert seen == [context]


@pytest.mark.asyncio
async def test_a_long_text_is_cut_to_what_one_result_holds() -> None:
    def long() -> str:
        """Say a lot."""
        return "x" * 50

    result = await tool(long).call(_context(max_result_chars=10), {})

    assert isinstance(result, FunctionResult)
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


@pytest.mark.asyncio
async def test_a_given_schema_spares_a_type_with_no_json_schema() -> None:
    def odd(when: Path = Path("/")) -> str:
        """Odd.

        Args:
          when: A path, which JSON has no type for.

        """
        return str(when)

    made = tool(odd, schema={"type": "object", "properties": {}})

    assert await made.call(_context(), {}) == FunctionResult(
        call_id="",
        name="odd",
        content="/",
    )


def test_tool_as_a_decorator_takes_the_same_settings() -> None:
    @tool(name="Ping")
    def ping() -> str:
        """Ping."""
        return "pong"

    assert ping.name == "Ping"


@pytest.mark.asyncio
async def test_an_omitted_argument_takes_its_default() -> None:
    result = await tool(greet).call(_context(), {"name": "ann"})

    assert result == FunctionResult(call_id="", name="greet", content="hello ann")


def test_an_annotated_description_and_bound_outrank_the_docstring() -> None:
    def pick(count: Annotated[int, "How many.", types.Arg(minimum=1)]) -> str:
        """Pick.

        Args:
          count: Ignored, as the annotation says it.

        """
        return str(count)

    assert tool(pick).schema["properties"] == {
        "count": {"type": "integer", "description": "How many.", "minimum": 1},
    }


def test_a_parameter_the_docstring_leaves_out_is_described_by_nothing() -> None:
    def bare(thing: int, other: str) -> str:
        return f"{thing}{other}"

    # As if written with no annotation: such a parameter takes a string.
    del bare.__annotations__["other"]
    made = tool(bare)

    assert made.description == ""
    assert made.schema["properties"] == {
        "thing": {"type": "integer", "description": ""},
        "other": {"type": "string", "description": ""},
    }


def test_the_args_section_ends_at_the_margin_and_keeps_colons_in_a_note() -> None:
    def ratio(a: str, b: str) -> str:
        return a + b

    ratio.__doc__ = """Ratio.

        Args:
           A continuation before any argument is ignored.
         a: As x:y, one space in.
         b: The other: also x:y.

        Returns:
          the ratio.

        """

    assert tool(ratio).schema["properties"] == {
        "a": {"type": "string", "description": "As x:y, one space in."},
        "b": {"type": "string", "description": "The other: also x:y."},
    }


def test_a_required_parameter_may_follow_one_with_a_default() -> None:
    def late(first: int = 1, *, second: str) -> str:
        """Late.

        Args:
          first: A count.
          second: A word.

        """
        return f"{second} {first}"

    assert tool(late).schema["required"] == ["second"]


@pytest.mark.asyncio
async def test_a_partial_keeps_the_types_and_defaults_of_what_it_wraps() -> None:
    """A partial has no name, and none of the annotations it wraps, of its own."""
    made = tool(partial(greet, times=2))

    assert made.name == "tool"
    assert (
        from_plain(made.schema["properties"], dict[str, dict[str, str]])["times"][
            "type"
        ]
        == "integer"
    )
    result = await made.call(_context(), {"name": "ann"})
    assert result == FunctionResult(
        call_id="",
        name="tool",
        content="hello ann hello ann",
    )


def test_a_display_shows_as_given_and_hides_the_output_by_default() -> None:
    shown = ToolDisplay(output=DisplaySpec(lines=None))

    assert tool(greet).display == ToolDisplay()
    assert tool(greet, display=shown).display is shown


class _Greeter:
    """An object a tool is made over, as one made for a conversation's line is."""

    def __init__(self, word: str) -> None:
        self.word = word

    def greet(self, name: str) -> str:
        """Say the greeter's word.

        Args:
          name: Who to greet.

        """
        return f"{self.word} {name}"


@pytest.mark.asyncio
async def test_tools_over_one_method_share_a_schema_but_call_their_own_object() -> None:
    hello = tool(_Greeter("hello").greet)
    bye = tool(_Greeter("bye").greet)

    assert (
        hello.schema
        == bye.schema
        == {
            "type": "object",
            "properties": {"name": {"type": "string", "description": "Who to greet."}},
            "required": ["name"],
            "additionalProperties": False,
        }
    )
    # Each tool has a schema of its own, so changing one changes no other.
    assert hello.schema is not bye.schema
    assert await hello.call(_context(), {"name": "ann"}) == FunctionResult(
        call_id="",
        name="greet",
        content="hello ann",
    )
    assert await bye.call(_context(), {"name": "ann"}) == FunctionResult(
        call_id="",
        name="greet",
        content="bye ann",
    )
    # The function itself still takes the object it would be bound to.
    assert from_plain(tool(_Greeter.greet).schema["required"], list[str]) == [
        "self",
        "name",
    ]


def test_the_object_a_dropped_tool_was_bound_to_is_let_go() -> None:
    greeter = _Greeter("hello")
    made = tool(greeter.greet)
    gone = weakref.ref(greeter)

    del greeter, made

    assert gone() is None


@pytest.mark.asyncio
async def test_a_function_returning_its_result_unnamed_is_named_as_its_tool() -> None:
    def found(pattern: str) -> FunctionResult:
        return FunctionResult(call_id="", content=pattern)

    result = await tool(found, name="Find").call(_context(), {"pattern": "x"})

    assert result == FunctionResult(call_id="", name="Find", content="x")


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
