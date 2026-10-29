"""A tool made from a function: its signature is the schema, its docstring what the model reads.

Each parameter's type gives its JSON schema, and its line in the docstring's
``Args:`` section its description; one without a default is required. A
parameter named ``context`` gets the calling agent's ``ToolContext`` instead
of an argument. A ``str`` return is cut to what one result holds; any session
record returned is the result as it is.

Typical usage example:

  def greet(name: str, times: int = 1) -> str:
      \"\"\"Say hello.

Args:
        name: Who to greet.
        times: How many times.

      \"\"\"
      return " ".join(["hello", name] * times)

  tools = [tool(greet, name="Greet")]

"""

from __future__ import annotations

from dataclasses import (
    field as dataclass_field,
    fields,
    make_dataclass,
)
from typing import TYPE_CHECKING, cast, get_type_hints, overload

import asyncio
import inspect

from sagent.session.types import UncategorizedToolResult
from sagent.tools.display import ToolDisplay
from sagent.tools.types import ToolContext, parse, schema_of


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping

    from _typeshed import DataclassInstance
    from treekle import PlainTree

    from sagent.session.types import AnyToolResult


__all__ = ["FunctionTool", "tool"]


class FunctionTool:
    """A ``Tool`` that calls a function with the model's arguments.

    Args:
      fn: The function, sync or async.
      name: What the model calls it by.
      description: What the model reads about it.
      schema: The JSON schema of its arguments.
      display: How its calls show in the terminal.

    Raises:
      TypeError: A parameter's type has no JSON schema.

    """

    __slots__ = (
        "_arguments",
        "_fn",
        "_takes_context",
        "description",
        "display",
        "name",
        "schema",
    )

    def __init__(
        self,
        fn: Callable[..., object],
        *,
        name: str,
        description: str,
        schema: Mapping[str, PlainTree] | None,
        display: ToolDisplay,
    ) -> None:
        self._fn = fn
        self._arguments = _signature(fn)
        self._takes_context = "context" in inspect.signature(fn).parameters
        self.name = name
        self.description = description
        self.schema = schema_of(self._arguments) if schema is None else schema
        self.display = display

    def summary(self, arguments: Mapping[str, PlainTree]) -> str:
        """Return the tool's name; a function says nothing more about a call."""
        del arguments
        return self.name

    def serialize_key(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> str | None:
        """Return ``None``: a function's calls run beside anything."""
        del context, arguments
        return None

    async def call(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> AnyToolResult:
        """Call the function with ``arguments``, checked, and return what it did.

        Args:
          context: The calling agent's state.
          arguments: What the model passed.

        Returns:
          result: The record the function returned, or its text as one.

        Raises:
          ValueError: The arguments do not fit the signature.

        """
        parsed = parse(arguments, self._arguments)
        kwargs: dict[str, object] = {
            f.name: getattr(parsed, f.name) for f in fields(parsed)
        }
        if self._takes_context:
            kwargs["context"] = context
        if inspect.iscoroutinefunction(self._fn):
            done = await cast("Callable[..., Awaitable[object]]", self._fn)(**kwargs)
        else:
            done = await asyncio.to_thread(self._fn, **kwargs)
        if isinstance(done, str):
            text, _ = context.fit([done])
            return UncategorizedToolResult(call_id="", content=text)
        return cast("AnyToolResult", done)


@overload
def tool(
    fn: Callable[..., object],
    /,
    *,
    name: str | None = ...,
    description: str | None = ...,
    schema: Mapping[str, PlainTree] | None = ...,
    display: ToolDisplay | None = ...,
) -> FunctionTool: ...


@overload
def tool(
    *,
    name: str | None = ...,
    description: str | None = ...,
    schema: Mapping[str, PlainTree] | None = ...,
    display: ToolDisplay | None = ...,
) -> Callable[[Callable[..., object]], FunctionTool]: ...


def tool(
    fn: Callable[..., object] | None = None,
    /,
    *,
    name: str | None = None,
    description: str | None = None,
    schema: Mapping[str, PlainTree] | None = None,
    display: ToolDisplay | None = None,
) -> FunctionTool | Callable[[Callable[..., object]], FunctionTool]:
    """Return a tool calling ``fn``; with no ``fn``, a decorator making one.

    Args:
      fn: The function, sync or async.
      name: What the model calls it; by default the function's name.
      description: What the model reads; by default the docstring.
      schema: The arguments' JSON schema; by default from the signature.
      display: How calls show; by default the output is hidden.

    Returns:
      tool: The tool, or a decorator returning one.

    Raises:
      TypeError: A parameter's type has no JSON schema.

    """

    def make(fn: Callable[..., object]) -> FunctionTool:
        return FunctionTool(
            fn,
            name=name or getattr(fn, "__name__", "tool"),
            description=(inspect.getdoc(fn) or "")
            if description is None
            else description,
            schema=schema,
            display=ToolDisplay() if display is None else display,
        )

    return make if fn is None else make(fn)


def _signature(fn: Callable[..., object]) -> type[DataclassInstance]:
    """Return ``fn``'s parameters, but ``context``, as an ``Arguments`` dataclass."""
    # A module offering a tool may import `ToolContext` for its checker only.
    hints = cast(
        "dict[str, object]",
        get_type_hints(fn, localns={"ToolContext": ToolContext}, include_extras=True),
    )
    notes = _notes(inspect.getdoc(fn) or "")
    specs: list[tuple[str, object, object]] = []
    for param in inspect.signature(fn).parameters.values():
        if param.name == "context":
            continue
        note = {"description": notes.get(param.name, "")}
        hint = hints.get(param.name, str)
        default = cast("object", param.default)
        if default is inspect.Parameter.empty:
            specs.append((param.name, hint, dataclass_field(metadata=note)))
        else:
            specs.append(
                (param.name, hint, dataclass_field(default=default, metadata=note)),
            )
    made = make_dataclass(
        f"{getattr(fn, '__name__', 'tool')}_arguments",
        specs,
        frozen=True,
        kw_only=True,
    )
    return cast("type[DataclassInstance]", made)


def _notes(doc: str) -> dict[str, str]:
    """Return what a Google-style docstring's ``Args:`` section says of each argument."""
    lines = doc.splitlines()
    starts = [i for i, line in enumerate(lines) if line == "Args:"]
    section: list[str] = []
    for line in lines[starts[0] + 1 :] if starts else []:
        if line.strip() and not line[0].isspace():
            break
        if line.strip():
            section.append(line)
    if not section:
        return {}
    # An entry starts at the section's own indent; a deeper line continues it.
    indent = min(len(line) - len(line.lstrip()) for line in section)
    out: dict[str, str] = {}
    current = ""
    for line in section:
        if len(line) - len(line.lstrip()) == indent:
            head, _, text = line.strip().partition(":")
            current = head.split()[0]
            out[current] = text.strip()
        elif current:
            out[current] = f"{out[current]} {line.strip()}"
    return out
