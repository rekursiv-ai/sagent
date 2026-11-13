"""A tool made from a function: its signature is the schema, its docstring what the model reads.

Each parameter's type gives its JSON schema, and its line in the docstring's
``Args:`` section its description; one without a default is required. A
parameter named ``context`` gets the calling agent's ``ToolContext`` instead
of an argument. The function returns ``str``, which is truncated to what one
result holds and becomes the tool's standard output, or a ``ToolOutput``,
which is the tool's output unchanged.

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
from functools import lru_cache
from typing import TYPE_CHECKING, cast, overload

import asyncio
import inspect

from treekle import immutable, mutable

from sagent.session.types import UncategorizedToolResult
from sagent.tools.display import ToolDisplay
from sagent.tools.types import (
    ToolContext,
    ToolOutput,
    ToolStreamingResult,
    json_schema,
    parse,
    whole_output,
)


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping

    from _typeshed import DataclassInstance
    from treekle import PlainTree

    type _Shape = tuple[type[DataclassInstance], Mapping[str, PlainTree], bool]
    """A tool's arguments type, their schema, and whether it takes the context."""


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
        self._arguments, self.schema, self._takes_context = _shape(
            fn,
            name=name,
            schema=schema,
        )
        self._fn = fn
        self.name = name
        self.description = description
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

    def call(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> ToolStreamingResult[object, object, object]:
        """Call the function with ``arguments`` and return its output.

        Args:
          context: The calling agent's state.
          arguments: What the model passed.

        Returns:
          streaming_result: The text the function returned, truncated to what one result
            holds, or the ``ToolOutput`` it returned.

        Raises:
          ValueError: The arguments do not fit the signature.
          TypeError: The function returned neither ``str`` nor ``ToolOutput``.

        """
        return whole_output(self._run(context, arguments))

    async def _run(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> ToolOutput[object, object, object]:
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
            return ToolOutput(stdout=(text,), result=UncategorizedToolResult())
        if not isinstance(done, ToolOutput):
            raise TypeError(
                f"{self.name} returned {type(done).__name__}; a function tool returns"
                " str or ToolOutput.",
            )
        return done


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


# A function's own shape is read once per function and name; a given ``schema`` is the
# schema, and none is derived for it.
def _shape(
    fn: Callable[..., object],
    *,
    name: str,
    schema: Mapping[str, PlainTree] | None,
) -> _Shape:
    """Return the tool ``name``'s arguments over ``fn``, their schema, and its context."""
    # A tool is often made anew for each line of a conversation, over a method of an
    # object made for that line: every bound method of one function has one shape.
    # The cache holds functions only, never the object a method is bound to, nor a
    # closure, which may hold one.
    if inspect.ismethod(fn) and _plain(fn.__func__):
        key: tuple[Callable[..., object], bool] | None = (fn.__func__, True)
    elif _plain(fn):
        key = (fn, False)
    else:
        key = None
    if key is None:
        # A partial's, a closure's or a callable object's shape is its own.
        arguments, takes_context = _arguments(fn, _read(fn), name=name)
        derived = json_schema(arguments) if schema is None else schema
        return arguments, derived, takes_context
    arguments, takes_context = _function_arguments(key[0], name, bound=key[1])
    if schema is not None:
        return arguments, schema, takes_context
    # Each tool holds its own copy, so a caller changing one changes no other.
    derived = mutable(_function_schema(key[0], name, bound=key[1]))
    return arguments, derived, takes_context


def _plain(fn: object) -> bool:
    """Return whether ``fn`` is a function that closes over nothing."""
    return inspect.isfunction(fn) and fn.__closure__ is None


def _read(fn: Callable[..., object]) -> inspect.Signature:
    """Return ``fn``'s signature, its annotations evaluated."""
    # A module offering a tool may import `ToolContext` for its checker only.
    return inspect.signature(fn, locals={"ToolContext": ToolContext}, eval_str=True)


# A bound function's first parameter is the object it is bound to, and no argument.
@lru_cache(maxsize=1_024)
def _function_arguments(
    fn: Callable[..., object],
    name: str,
    /,
    *,
    bound: bool,
) -> tuple[type[DataclassInstance], bool]:
    """Return ``fn``'s arguments and whether it takes the context."""
    signature = _read(fn)
    if bound:
        signature = signature.replace(parameters=[*signature.parameters.values()][1:])
    return _arguments(fn, signature, name=name)


@lru_cache(maxsize=1_024)
def _function_schema(
    fn: Callable[..., object],
    name: str,
    /,
    *,
    bound: bool,
) -> Mapping[str, PlainTree]:
    """Return the schema of ``fn``'s arguments, read only for a tool given none."""
    arguments, _ = _function_arguments(fn, name, bound=bound)
    return immutable(json_schema(arguments))


# The arguments are ``signature``'s parameters, described by ``fn``'s docstring.
def _arguments(
    fn: Callable[..., object],
    signature: inspect.Signature,
    *,
    name: str,
) -> tuple[type[DataclassInstance], bool]:
    """Return the tool ``name``'s arguments, and whether it takes the context."""
    return _signature(fn, signature, name=name), "context" in signature.parameters


def _signature(
    fn: Callable[..., object],
    signature: inspect.Signature,
    *,
    name: str,
) -> type[DataclassInstance]:
    """Return ``fn``'s parameters, but ``context``, as the tool ``name``'s arguments."""
    # The signature, not the function's own annotations, so a ``partial`` keeps the
    # types of what it wraps.
    notes = _notes(inspect.getdoc(fn) or "")
    specs: list[tuple[str, object, object]] = []
    for param in signature.parameters.values():
        if param.name == "context":
            continue
        note = {"description": notes.get(param.name, "")}
        annotation = cast("object", param.annotation)
        hint = str if annotation is inspect.Parameter.empty else annotation
        default = cast("object", param.default)
        if default is inspect.Parameter.empty:
            specs.append((param.name, hint, dataclass_field(metadata=note)))
        else:
            specs.append(
                (param.name, hint, dataclass_field(default=default, metadata=note)),
            )
    # Keyword-only, as a call names every argument: a required parameter after
    # one with a default needs no positional slot.
    made = make_dataclass(name, specs, frozen=True, kw_only=True)
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
