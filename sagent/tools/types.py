"""The ``Tool`` protocol: something the model calls by name.

``Tool.call`` returns a ``ToolStreamingResult`` at once. It has two streams, the
tool's standard output and standard error, and a future of the tool's result: a
dataclass with an ``exit_code``. The caller passes it to ``settled``, which
reads the streams and returns the call's ``ToolResult`` record.

A tool holds no agent's state; each call gets that agent's ``ToolContext``, so
one tool serves every agent. The context also carries what the session's agents
share, such as the fetch session, which a spawned agent inherits.

A tool declares its arguments once, as a dataclass whose fields are each
``Annotated[T, "description"]``, with an ``Arg`` for bounds or a key that is
no Python name: ``json_schema`` makes the JSON schema the model sees from it,
and ``parse`` turns the model's arguments back into it.

Typical usage example:

  @dataclass(frozen=True, slots=True, kw_only=True)
  class Echo:
      @dataclass(frozen=True, slots=True, kw_only=True)
      class Arguments:
          text: Annotated[str, "What to say back."]
          times: Annotated[int, "How many times to say it.", Arg(minimum=1)] = 1

      name = "Echo"
      description = "Say the text back."
      schema = json_schema(Arguments)
      display = ToolDisplay(output=DisplaySpec())

      def summary(self, arguments: Mapping[str, PlainTree]) -> str:
          return "Echo"

      def serialize_key(self, context, arguments) -> str | None:
          return None

      def call(
          self,
          context,
          arguments,
      ) -> ToolStreamingResult[UncategorizedToolResult, str, str]:
          return whole_output(self._say(arguments))

      async def _say(self, arguments) -> ToolOutput[UncategorizedToolResult, str, str]:
          a = parse(arguments, Echo.Arguments)
          return ToolOutput(
              stdout=(a.text,) * a.times,
              result=UncategorizedToolResult(),
          )

  echo: Tool = Echo()
  result = await settled(echo.call(context, {"text": "hi"}))  # A ``ToolResult``.
"""

from __future__ import annotations

from asyncio import Future
from collections.abc import (
    AsyncIterator,
    Callable,
    Coroutine,
    Mapping,
    Sequence,
    Set as AbstractSet,
)
from dataclasses import MISSING, dataclass, field, fields, is_dataclass, replace
from pathlib import Path
from types import MappingProxyType, NoneType, UnionType
from typing import (
    TYPE_CHECKING,
    Annotated,
    Generic,
    Literal,
    Protocol,
    TypeAliasType,
    TypeGuard,
    cast,
    final,
    get_args,
    get_origin,
    get_type_hints,
    overload,
    runtime_checkable,
)
from typing_extensions import NoDefault, TypeVar

import asyncio
import contextlib
import difflib
import functools
import hashlib
import reprlib
import uuid

from treekle import PlainTree
from wesearch.fetch import FetchSession
from wesearch.fetch.session import default_session

from sagent.runtime.exceptions import describe
from sagent.session.types import ToolResult


if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from _typeshed import DataclassInstance

    from sagent.tools.display import ToolDisplay


__all__ = [
    "Arg",
    "Briefing",
    "FileTracker",
    "Reminding",
    "Skipping",
    "Stream",
    "Tool",
    "ToolContext",
    "ToolOutput",
    "ToolStreamingResult",
    "checked",
    "json_schema",
    "parse",
    "settled",
    "whole_output",
]


# Defined here as well as in ``session.types``: ruff reads a type variable's
# default only in the module that defines it, and seeing none, UP046 rewrites
# each class to ``class ToolOutput[...]``, which drops the defaults.
Result_co = TypeVar("Result_co", covariant=True, default=object)
"""Type of a tool's result, such as ``ReadResult``; it has an ``exit_code``."""

StreamOut_co = TypeVar("StreamOut_co", covariant=True, default=object)
"""Type of each item a tool writes to standard output."""

StreamErr_co = TypeVar("StreamErr_co", covariant=True, default=object)
"""Type of each item a tool writes to standard error."""


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolStreamingResult(Generic[Result_co, StreamOut_co, StreamErr_co]):
    """A running tool call: its two output streams and the future of its result.

    The streams need not be read: an unread stream never blocks the tool. A
    ``Stream`` the tool left open ends when ``result`` is done, however the call
    ends, as a process's pipes end when it exits.
    """

    result: Future[Result_co]
    """The tool's result. It raises what the tool raised; cancelling it stops the tool."""

    stdout: AsyncIterator[StreamOut_co]
    """The tool's standard output, item by item as the tool writes it."""

    stderr: AsyncIterator[StreamErr_co]
    """The tool's standard error, item by item as the tool writes it."""

    def __post_init__(self) -> None:
        self.result.add_done_callback(self._end_streams)

    # Any other iterator ends itself: only a ``Stream`` waits for a write.
    def _end_streams(self, result: object) -> None:
        """Close each ``Stream``, so a reader stops after what the tool wrote."""
        del result
        for stream in (self.stdout, self.stderr):
            if isinstance(stream, Stream):
                stream.close()


class Tool(Protocol[Result_co, StreamOut_co, StreamErr_co]):
    """Something the model can call by name.

    The type parameters are the result class, the standard output item type,
    and the standard error item type. A bare ``Tool`` is a tool of any three.
    A ``Tool[EditResult, Splice | str, str]`` returns an ``EditResult``, writes
    text and ``Splice`` items to standard output, and text to standard error.
    """

    @property
    def name(self) -> str:
        """Return the name the model calls this by."""
        ...

    @property
    def description(self) -> str:
        """Return what the model reads about when and how to call this."""
        ...

    @property
    def schema(self) -> Mapping[str, PlainTree]:
        """Return the JSON schema of the arguments ``call`` takes."""
        ...

    @property
    def display(self) -> ToolDisplay:
        """Return how this tool's calls show in the terminal."""
        ...

    def summary(self, arguments: Mapping[str, PlainTree]) -> str:
        """Return a short label for a call, such as ``Edit foo.py``."""
        ...

    def serialize_key(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> str | None:
        """Return which calls this one must not overlap.

        Calls with the same key run one at a time, in the order the model made
        them. ``"*"`` waits for every keyed call before it, and every later
        call waits for it. ``None`` runs beside anything.

        Args:
          context: The calling agent's state.
          arguments: What the model passed, perhaps not fitting the schema.

        Returns:
          key: The key, ``"*"``, or ``None``.

        """
        ...

    def call(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> ToolStreamingResult[Result_co, StreamOut_co, StreamErr_co]:
        """Start the tool with the model's arguments and return at once.

        A tool writes each item when it has it, such as a line its process
        printed, a match, or a path. A tool that has nothing until it finishes
        writes everything then.

        Args:
          context: The calling agent's state.
          arguments: What the model passed.

        Returns:
          streaming_result: The tool's output streams and the future of its
            result. The result's ``exit_code`` is ``0`` for success.

        Raises:
          ValueError: The arguments do not fit the schema. It is raised here or
            from ``streaming_result.result``; the caller returns the message to
            the model as an error.

        """
        ...


# Final, so a checker knows that no ``ToolOutput`` is also a coroutine, which
# ``whole_output`` takes in its place.
@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ToolOutput(Generic[Result_co, StreamOut_co, StreamErr_co]):
    """The complete output of a tool that does not stream; see ``whole_output``."""

    stdout: tuple[StreamOut_co, ...] = ()
    """The items for standard output."""

    stderr: tuple[StreamErr_co, ...] = ()
    """The items for standard error, such as why the tool did nothing."""

    result: Result_co
    """The tool's result: its own class or an ``UncategorizedToolResult``."""


def whole_output[R, O, E](
    output: Coroutine[object, object, ToolOutput[R, O, E]] | ToolOutput[R, O, E],
) -> ToolStreamingResult[R, O, E]:
    """Return a ``ToolOutput``, or the one a coroutine returns, as a ``ToolStreamingResult``.

    Args:
      output: The tool's output, or the tool's work that returns it. A
        coroutine starts now, on the running loop.

    Returns:
      streaming_result: Its streams get the output's items when the work
        returns, and nothing when it raises or is cancelled. Its ``result``
        raises what the work raised.

    """
    stdout, stderr = Stream[O](), Stream[E]()
    return ToolStreamingResult(
        result=asyncio.ensure_future(_write_output(output, stdout, stderr)),
        stdout=stdout,
        stderr=stderr,
    )


class Stream[T]:
    """One output stream of a tool: the tool writes items, the caller iterates.

    A write never waits for a reader. Items are held until read, and dropped
    with the stream if nobody reads them.
    """

    def __init__(self) -> None:
        self._queue: asyncio.Queue[T | _Closed] = asyncio.Queue()

    def write(self, item: T) -> None:
        """Append ``item`` to the stream."""
        self._queue.put_nowait(item)

    def close(self) -> None:
        """End the stream; iteration stops after the items already written."""
        self._queue.put_nowait(_Closed())

    def __aiter__(self) -> Stream[T]:
        """Return the stream; it can be iterated once."""
        return self

    async def __anext__(self) -> T:
        """Return the next item, waiting for it; stop when the stream is closed."""
        item = await self._queue.get()
        if isinstance(item, _Closed):
            # Put back, so a second iteration past the end stops too and never waits.
            self._queue.put_nowait(item)
            raise StopAsyncIteration
        return item


@overload
async def settled(
    streaming_result: ToolStreamingResult[ToolResult, object, object],
    *,
    call_id: str = "",
) -> ToolResult: ...


@overload
async def settled[R, O, E](
    streaming_result: ToolStreamingResult[R, O, E],
    *,
    call_id: str = "",
) -> ToolResult[R, O, E | str]: ...


async def settled(
    streaming_result: ToolStreamingResult[object, object, object],
    *,
    call_id: str = "",
) -> ToolResult:
    """Read ``streaming_result`` to its end and return the call's record.

    This never raises for the tool's failure or for a cancel, so every call
    gets a record. When cancelled, it stops the tool and waits for it to stop.

    Args:
      streaming_result: What ``Tool.call`` returned.
      call_id: The id of the call the record answers.

    Returns:
      result: The record, with the items read from both streams. Its state is
        ``done`` with the tool's result; ``failed`` when the tool raised, with
        the error as the last item of ``stderr``; or ``cancelled``. A tool
        that relays a call run elsewhere returns that call's record as its
        result, and that record is returned with ``call_id`` set.

    """
    stdout: list[object] = []
    stderr: list[object] = []
    readers = (
        asyncio.ensure_future(_collect(streaming_result.stdout, stdout)),
        asyncio.ensure_future(_collect(streaming_result.stderr, stderr)),
    )
    result_future = streaming_result.result
    try:
        # Not ``gather``: a cancel must reach the tool now, not when the readers end.
        _ = await asyncio.wait(readers, return_when=asyncio.FIRST_EXCEPTION)
        for reader in readers:
            if reader.done():
                reader.result()
        result = await result_future
    except asyncio.CancelledError:
        state, errors = "cancelled", ()
    except Exception as error:  # noqa: BLE001 -- A tool's failure is the call's answer, not its runner's.
        state, errors = "failed", (describe(error),)
    else:
        if isinstance(result, ToolResult):
            return replace(result, call_id=call_id)
        return ToolResult(
            call_id=call_id,
            state="done",
            stdout=tuple(stdout),
            stderr=tuple(stderr),
            result=result,
        )
    for reader in readers:
        _ = reader.cancel()
    # The tool may still run: this call was cancelled, or a stream raised.
    _ = result_future.cancel()
    # A second cancel must not leave the tool's process half stopped.
    while not result_future.done():
        with contextlib.suppress(asyncio.CancelledError):
            _ = await asyncio.wait((result_future,))
    return ToolResult(
        call_id=call_id,
        state=state,
        stdout=tuple(stdout),
        stderr=(*stderr, *errors),
    )


@runtime_checkable
class Reminding(Protocol):
    """A tool with something to tell the model each round, such as who it can reach."""

    def reminder(self, context: ToolContext) -> str:
        """Return what the model should know now; ``""`` for nothing."""
        ...


@runtime_checkable
class Skipping(Protocol):
    """A tool that reads the user's files and can leave one unread, such as a skill."""

    def skipped(self, context: ToolContext) -> tuple[str, ...]:
        """Return which files were left unread, and why; one line each."""
        ...


@runtime_checkable
class Briefing(Protocol):
    """A tool with standing context for the model, such as an index of memories.

    The agent reads it once a window, into the system prompt, so the cached
    prefix holds until the next clear or compaction; what changes mid-window
    reaches the model as the call that changed it.
    """

    def briefing(self, context: ToolContext) -> str:
        """Return what the window's system prompt says for this tool; ``""`` for nothing."""
        ...


@dataclass(frozen=True, slots=True, kw_only=True)
class Arg:
    """What an argument's description cannot say, for ``Annotated[T, "...", Arg(...)]``."""

    minimum: float | None = None
    maximum: float | None = None
    key: str | None = None
    """The model's name for the argument when it is no Python name, such as ``-B``."""


def json_schema(arguments: type[DataclassInstance]) -> Mapping[str, PlainTree]:
    """Return the JSON schema of a tool's ``Arguments`` dataclass.

    Each field is ``Annotated[T, "description"]``, with an optional ``Arg``;
    one without a default is required. A dataclass built at runtime, which
    cannot spell ``Annotated``, may give a field's description as
    ``field(metadata={"description": ...})`` instead.

    Args:
      arguments: The dataclass.

    Returns:
      schema: An ``object`` schema that rejects unknown keys.

    """
    properties: dict[str, PlainTree] = {}
    required: list[PlainTree] = []
    for f in _fields(arguments):
        schema = _schema(f.hint)
        bounds: dict[str, PlainTree] = {
            **({"minimum": f.arg.minimum} if f.arg.minimum is not None else {}),
            **({"maximum": f.arg.maximum} if f.arg.maximum is not None else {}),
        }
        # A bound on a sequence bounds each item; on the array it means nothing.
        items = schema.get("items")
        if isinstance(items, Mapping):
            schema["items"] = {**items, **bounds}
            bounds = {}
        properties[f.key] = {
            **schema,
            "description": f.description,
            **bounds,
        }
        if f.required:
            required.append(f.key)
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def parse[A: DataclassInstance](arguments: Mapping[str, PlainTree], kind: type[A]) -> A:
    """Return the model's ``arguments`` as ``kind``, checked against its schema.

    Args:
      arguments: What the model sent.
      kind: The tool's ``Arguments`` dataclass.

    Returns:
      parsed: ``arguments`` as ``kind``, with defaults for what was omitted.

    Raises:
      ValueError: An argument is unknown, missing, of the wrong type, or out
        of bounds.

    """
    known = _fields(kind)
    unknown = set(arguments) - {f.key for f in known}
    if unknown:
        raise ValueError(f"Unknown arguments: {', '.join(sorted(unknown))}.")
    values: dict[str, object] = {}
    for f in known:
        if f.key not in arguments:
            if f.required:
                raise ValueError(f"Missing argument {f.key!r}.")
            continue
        values[f.name] = checked(arguments[f.key], f.hint, key=f.key)
        _bound(arguments[f.key], f)
    return kind(**values)


@dataclass(slots=True, kw_only=True)
class FileTracker:
    """The files a model has seen, as it saw them, by the file each path names.

    ``sub/../a.py`` and a link to ``a.py`` are ``a.py``: a file read by one
    spelling is known by every other.
    """

    _seen: dict[Path, _Snapshot] = field(default_factory=dict[Path, "_Snapshot"])
    window: str = field(default_factory=lambda: uuid.uuid4().hex)
    """Which context window knows these files; ``forget`` starts another. A tool
    that keeps the model's knowledge elsewhere, as on another machine, keys it by
    this, so a new window, or a new agent, starts with none."""

    def __contains__(self, path: object) -> bool:
        """Return whether the model has seen ``path``."""
        return isinstance(path, Path) and path.resolve() in self._seen

    def saw(self, path: Path) -> None:
        """Record that the model now knows ``path`` as it is on disk.

        Args:
          path: An absolute path to a file that exists.

        """
        path = path.resolve()
        # Stat before reading: a write landing between them then shows as a
        # change, never as the file the model saw.
        stamp = _stamp(path)
        data = path.read_bytes()
        # Re-inserted, so the order of ``_seen`` is the order of last sight.
        self._seen.pop(path, None)
        self._seen[path] = _Snapshot(
            stamp=stamp,
            digest=hashlib.sha256(data).hexdigest(),
            text=_text(data),
        )

    def showed(self, path: Path, *, lines: object) -> None:
        """Record that the model was shown ``lines`` of ``path``, as last seen.

        Args:
          path: A file ``saw`` recorded.
          lines: Which lines, as the reader asked for them, compared by
            equality; Read passes its ``(offset, limit, last_lines)``.

        """
        path = path.resolve()
        self._seen[path] = replace(self._seen[path], lines=lines)

    def shown(self, path: Path, *, lines: object) -> bool:
        """Return whether the model holds ``lines`` of ``path`` as they still are.

        Args:
          path: An absolute path.
          lines: Which lines, as ``showed`` recorded them.

        Returns:
          shown: ``True`` when those lines were shown and the file has not
            changed since.

        """
        snapshot = self._seen.get(path.resolve())
        return (
            snapshot is not None and snapshot.lines == lines and not self.changed(path)
        )

    def recent(self) -> list[Path]:
        """Return the files seen, the most recently seen last."""
        return list(self._seen)

    def forget(self) -> None:
        """Forget every file; call after compacting, pruning, or a server's clearing."""
        self._seen.clear()
        self.window = uuid.uuid4().hex

    def changed(self, path: Path) -> bool:
        """Return whether ``path`` on disk differs from what the model last saw.

        A file the model never saw has not changed under it.

        Args:
          path: An absolute path.

        Returns:
          changed: ``True`` when the file was edited or removed since.

        """
        path = path.resolve()
        snapshot = self._seen.get(path)
        if snapshot is None:
            return False
        try:
            if _stamp(path) == snapshot.stamp:
                return False
            data = path.read_bytes()
        except FileNotFoundError:
            return True
        # Sync tools and formatters touch the mtime without changing a byte.
        return hashlib.sha256(data).hexdigest() != snapshot.digest

    def changes(self) -> Iterator[tuple[Path, str]]:
        """Yield each seen file changed since, with a diff, and see it anew.

        Yields:
          path: A file that changed on disk.
          diff: What changed, as a unified diff, or a note when the old text
            was too large to keep.

        """
        for path in list(self._seen):
            if not self.changed(path) or not path.is_file():
                continue
            before = self._seen[path].text
            self.saw(path)
            after = self._seen[path].text
            if before is None or after is None:
                yield path, f"{path} changed on disk; read it again."
                continue
            yield (
                path,
                "".join(
                    difflib.unified_diff(
                        before.splitlines(keepends=True),
                        after.splitlines(keepends=True),
                        fromfile=str(path),
                        tofile=str(path),
                    ),
                ),
            )


@dataclass(slots=True, kw_only=True)
class ToolContext:
    """One agent's state that its tools share, and what its session shares."""

    cwd: Path = field(default_factory=Path.cwd)
    """The working directory; a shell ``cd`` moves it."""

    root: Path | None = None
    """Where the agent started, which its AGENTS.md and skills are read from, and a
    new window returns to; ``None`` for ``cwd``."""

    files: FileTracker = field(default_factory=FileTracker)

    max_result_chars: int = 50_000
    """How much text one result may hold; a tool may hold its own to less."""

    label: str = ""
    """The agent's name in its team; ``""`` outside one."""

    spill_dir: Path | None = None
    """Where a tool writes the whole of an output it cut; ``None`` for nowhere."""

    columns: int | Callable[[], int] = 80
    """The width a command's output is shown in, as ``COLUMNS``; a callable is read
    as each command starts, so a frontend's window can change size between them."""

    fetch_session: FetchSession = field(default_factory=default_session, repr=False)
    """The connections, cookies, browser and egress every web fetch reuses; whoever
    made it closes it. Unset, the process's shared one, whose files go where the
    user's data directory was when it was first made."""

    @property
    def origin(self) -> Path:
        """Return where the agent started: ``root``, else ``cwd``."""
        return self.root if self.root is not None else self.cwd

    def path(self, name: str) -> Path:
        """Return the absolute path a tool argument names, from ``cwd``.

        Args:
          name: A path, absolute, relative, or starting with ``~``.

        Returns:
          path: Absolute, but with links and ``..`` as given.

        """
        return self.cwd / Path(name).expanduser()

    def fit(
        self,
        units: Iterable[str],
        *,
        reserve: int = 0,
        cap: int = 0,
    ) -> tuple[str, int]:
        """Join ``units`` while they fit in one result.

        A unit is kept whole -- a line, a match, a path -- so the cut falls on
        a boundary the caller can resume from. The first unit is always kept,
        cut mid-way if it alone is too large, so a result is never empty.

        Args:
          units: Rendered pieces, in order.
          reserve: Room to leave for a note the caller appends.
          cap: The tool's own limit, if below ``max_result_chars``; ``0`` for none.

        Returns:
          body: The kept units, joined.
          kept: How many units ``body`` holds.

        """
        limit = min(cap, self.max_result_chars) if cap else self.max_result_chars
        budget = max(1, limit - reserve)
        out: list[str] = []
        used = 0
        for unit in units:
            if not out and len(unit) > budget:
                omitted = len(unit) - budget
                return f"{unit[:budget]}\n... ({omitted:,} chars omitted)", 1
            if used + len(unit) > budget:
                break
            used += len(unit)
            out.append(unit)
        return "".join(out), len(out)


@dataclass(frozen=True, slots=True, kw_only=True)
class _Field:
    """One argument, as the model sees it and as the dataclass holds it."""

    key: str
    """The model's name for it."""

    name: str
    """The dataclass field's name."""

    hint: object
    description: str
    arg: Arg
    required: bool


def checked(value: object, hint: object, *, key: str) -> object:
    """Return ``value`` as ``hint``, a sequence as a tuple; raise if it is not one.

    A tool's arguments arrive as JSON, so a sequence may be a list and a number an
    ``int``. A wire message arrives decoded, so a field may hold a record: it is checked
    field by field, as treekle does not below the root.

    Args:
      value: What to check.
      hint: The type it must be.
      key: What names it in an error: an argument, or a record and its fields' path.

    Returns:
      value: ``value`` as ``hint``: a sequence as a tuple, a number as a ``float``
        where one is asked for, a record as it is.

    Raises:
      ValueError: ``value``, or a part of it, is not of its type; ``key`` says where.
      TypeError: ``hint`` is a type this does not check.

    """
    if hint is PlainTree or hint is object:
        return value
    if isinstance(hint, TypeAliasType):
        return checked(value, _value(hint), key=key)
    # A bare generic record leaves its type parameters unbound; each means its default.
    if isinstance(hint, TypeVar):
        # On 3.12 a ``typing.TypeVar`` or a PEP 695 parameter has no ``__default__``.
        default = cast(object, getattr(hint, "__default__", NoDefault))
        return checked(value, object if default is NoDefault else default, key=key)
    origin, args = get_origin(hint), _args(hint)
    if origin is Annotated:
        return checked(value, args[0], key=key)
    if _is_record(origin):
        parameters = cast("_Generic", origin).__parameters__
        return _checked_record(
            value,
            origin,
            key=key,
            # ``typing`` fills each omitted argument with its default, so every
            # parameter has one.
            type_arguments={
                parameter: args[i] for i, parameter in enumerate(parameters)
            },
        )
    if origin is Literal:
        # ``True == 1``, so the type must match as well as the value.
        if any(value == a and type(value) is type(a) for a in args):
            return value
        raise _misfit(key, f"one of {list(args)}", value)
    if _is_union(origin):
        return _checked_member(value, args, key=key)
    if origin in {tuple, list, Sequence}:
        return _checked_items(value, origin, args, key=key)
    if origin in {dict, Mapping}:
        if not isinstance(value, Mapping):
            raise _misfit(key, "object", value)
        entries = cast("Mapping[object, object]", value)
        return {
            checked(k, args[0], key=key): checked(v, args[1], key=key)
            for k, v in entries.items()
        }
    if origin is frozenset:
        if not isinstance(value, AbstractSet):
            raise _misfit(key, "set", value)
        return frozenset(checked(v, args[0], key=key) for v in value)
    if _is_record(hint):
        return _checked_record(value, hint, key=key)
    if (
        (hint is bool and isinstance(value, bool))
        or (hint is str and isinstance(value, str))
        or (hint is bytes and isinstance(value, bytes))
    ):
        return value
    if hint is int and isinstance(value, int) and not isinstance(value, bool):
        return value
    if hint is float and isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    raise _misfit(key, "bytes" if hint is bytes else _schema(hint).get("type"), value)


def _checked_member(value: object, args: tuple[object, ...], *, key: str) -> object:
    """Return ``value`` as the first member of a union it is; raise if it is none."""
    if value is None and NoneType in args:
        return None
    members: list[object] = [a for a in args if a is not NoneType]
    # A record is of its own class: a record member it is no instance of cannot hold it.
    if is_dataclass(value) and not isinstance(value, type):
        members = [m for m in members if not _is_record(m) or isinstance(value, m)]
        if not members:
            raise _misfit(key, "a record of its type", value)
    errors: list[str] = []
    for member in members:
        try:
            return checked(value, member, key=key)
        except ValueError as e:
            errors.append(str(e))
    raise ValueError(" ".join(errors))


def _checked_items(
    value: object,
    origin: object,
    args: tuple[object, ...],
    *,
    key: str,
) -> tuple[object, ...]:
    """Return ``value`` as a tuple of its items, each checked; raise if it is none."""
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise _misfit(key, "array", value)
    if origin is not tuple or args[-1:] == (Ellipsis,):
        return tuple(checked(v, args[0], key=key) for v in value)
    if len(value) != len(args):
        raise _misfit(key, f"{len(args)} items", value)
    return tuple(checked(v, args[i], key=key) for i, v in enumerate(value))


# ``type_arguments`` maps each type parameter of a generic ``kind`` to the type given
# for it.
def _checked_record[R: DataclassInstance](
    value: object,
    kind: type[R],
    *,
    key: str,
    type_arguments: Mapping[object, object] = MappingProxyType({}),
) -> R:
    """Return ``value`` if it is a ``kind`` and each field is of its type."""
    if not isinstance(value, kind):
        raise _misfit(key, kind.__name__, value)
    # ``value`` may be a subclass of ``kind``, whose fields are its own.
    hints = _hints(type(value))
    for f in fields(value):
        _ = checked(
            cast(object, getattr(value, f.name)),
            _substituted(hints[f.name], type_arguments),
            key=f"{key}.{f.name}",
        )
    return value


class _Parameterized(Protocol):
    """A type with unbound type parameters, such as ``tuple[T, ...]``."""

    def __getitem__(self, arguments: tuple[object, ...]) -> object: ...


class _Generic(Protocol):
    """A generic class."""

    __parameters__: tuple[object, ...]


def _substituted(hint: object, type_arguments: Mapping[object, object]) -> object:
    """Return ``hint`` with each type parameter replaced by its type argument."""
    if isinstance(hint, TypeVar):
        return type_arguments.get(hint, hint)
    parameters = cast("tuple[object, ...]", getattr(hint, "__parameters__", ()))
    if not type_arguments or not parameters:
        return hint
    return cast("_Parameterized", hint)[
        tuple(type_arguments.get(p, p) for p in parameters)
    ]


def _is_record(hint: object) -> TypeGuard[type[DataclassInstance]]:
    """Return whether ``hint`` is a record class."""
    return isinstance(hint, type) and is_dataclass(hint)


def _misfit(key: str, expected: object, value: object) -> ValueError:
    """Return the error saying ``key`` holds ``value``, which is no ``expected``."""
    # A value is shown cut short: a peer's may be megabytes.
    return ValueError(f"{key!r} must be {expected}, got {reprlib.repr(value)}.")


# Before 3.14, ``Literal[...] | None`` is a ``typing.Union``, not a ``UnionType``.
def _is_union(origin: object) -> bool:
    """Return whether ``origin`` is either spelling of a union."""
    return origin is UnionType or origin is get_origin(Literal[0] | None)


# Typeshed types ``typing``'s introspection as ``Any``; called directly, it
# would leave its callers untyped. Resolving a class's hints costs as much as
# checking the rest of a record, and ``checked`` reads them for every one.
@functools.cache
def _hints(kind: type[DataclassInstance]) -> Mapping[str, object]:
    return get_type_hints(kind, include_extras=True)


def _args(hint: object) -> tuple[object, ...]:
    return get_args(hint)


def _value(alias: TypeAliasType) -> object:
    """Return what a ``type X = ...`` alias stands for."""
    return cast(object, alias.__value__)


def _leaf(value: object) -> PlainTree:
    """Return a ``Literal`` member as a JSON value."""
    if value is None or isinstance(value, str | int | float):
        return value
    raise TypeError(f"Literal member {value!r} is not a JSON value.")


@dataclass(frozen=True, slots=True, kw_only=True)
class _Snapshot:
    """A file as the model last saw it."""

    stamp: tuple[int, int, int, int]
    digest: str
    text: str | None
    """The file's text, kept to diff against; ``None`` if too large or binary."""

    lines: object = None
    """Which lines the model was last shown, as ``showed`` recorded them."""


def _stamp(path: Path) -> tuple[int, int, int, int]:
    """Return file identity and nanosecond times without reading its bytes."""
    info = path.stat()
    return info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def _bound(value: PlainTree, f: _Field) -> None:
    """Raise if ``value``, or an item of it, is a number outside ``f``'s bounds."""
    if isinstance(value, Sequence) and not isinstance(value, str):
        for item in value:
            _bound(item, f)
        return
    if not isinstance(value, int | float) or isinstance(value, bool):
        return
    if f.arg.minimum is not None and value < f.arg.minimum:
        raise ValueError(f"{f.key!r} must be at least {f.arg.minimum}, got {value}.")
    if f.arg.maximum is not None and value > f.arg.maximum:
        raise ValueError(f"{f.key!r} must be at most {f.arg.maximum}, got {value}.")


def _text(data: bytes, *, max_bytes: int = 1 << 20) -> str | None:
    """Return ``data`` as text to diff against, or ``None`` if large or not UTF-8."""
    if len(data) > max_bytes:
        return None
    try:
        return data.decode()
    except UnicodeDecodeError:
        return None


def _fields(kind: type[DataclassInstance]) -> list[_Field]:
    """Return how the model sees each field of ``kind``."""
    hints = _hints(kind)
    out: list[_Field] = []
    for f in fields(kind):
        hint, *extras = (
            _args(hints[f.name])
            if get_origin(hints[f.name]) is Annotated
            else (hints[f.name],)
        )
        description = next(
            (x for x in extras if isinstance(x, str)),
            cast("str | None", f.metadata.get("description")),
        )
        if description is None:
            raise TypeError(
                f'{kind.__qualname__}.{f.name} needs Annotated[T, "description"].',
            )
        arg = next((x for x in extras if isinstance(x, Arg)), Arg())
        out.append(
            _Field(
                key=arg.key or f.name,
                name=f.name,
                hint=hint,
                description=description,
                arg=arg,
                required=f.default is MISSING and f.default_factory is MISSING,
            ),
        )
    return out


def _schema(hint: object) -> dict[str, PlainTree]:
    """Return the JSON schema of one Python type."""
    if hint is PlainTree:
        return {}
    if isinstance(hint, TypeAliasType):
        return _schema(_value(hint))
    origin, args = get_origin(hint), _args(hint)
    if origin is Literal:
        return {**_schema(type(args[0])), "enum": [_leaf(a) for a in args]}
    if _is_union(origin):
        members = [_schema(a) for a in args if a is not NoneType]
        return members[0] if len(members) == 1 else {"anyOf": members}
    if origin in {tuple, list, Sequence}:
        return {"type": "array", "items": _schema(args[0])}
    if origin in {dict, Mapping}:
        return {"type": "object", "additionalProperties": _schema(args[1])}
    for kind, name in (
        (str, "string"),
        (bool, "boolean"),
        (int, "integer"),
        (float, "number"),
    ):
        if hint is kind:
            return {"type": name}
    raise TypeError(f"No JSON schema for {hint!r}.")


class _Closed:
    """The marker that ends a ``Stream``; a tool cannot write one."""


async def _collect[T](stream: AsyncIterator[T], items: list[T]) -> None:
    """Append each item of ``stream`` to ``items`` until the stream ends."""
    # Appended one at a time, so a cancel midway keeps the items already read.
    while True:
        try:
            items.append(await anext(stream))
        except StopAsyncIteration:
            return


# A coroutine is awaited inside the task that is the call's ``result`` future,
# so cancelling that future cancels the tool, and the tool's
# ``asyncio.current_task()`` is that future, which ``Runtime.push_after`` matches.
async def _write_output[R, O, E](
    output: Coroutine[object, object, ToolOutput[R, O, E]] | ToolOutput[R, O, E],
    stdout: Stream[O],
    stderr: Stream[E],
) -> R:
    """Write ``output`` to the streams, awaiting it first if need be; return its result."""
    # Closed here, not left to the result's end: the call's record is then made
    # in the loop turn the tool ends in, with whatever the tool queued for its
    # agent, and not a turn after it.
    try:
        whole = await output if isinstance(output, Coroutine) else output
        for item in whole.stdout:
            stdout.write(item)
        for item in whole.stderr:
            stderr.write(item)
    finally:
        stdout.close()
        stderr.close()
    return whole.result
