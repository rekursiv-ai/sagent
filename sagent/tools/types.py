"""What a tool is: something the model calls by name, which says what it did.

A tool returns a session record typed by what it did -- a ``EditResult``
names its edits, a ``BashResult`` its exit code -- so the session keeps
the act itself, not a description of it. The runtime fills in ``call_id``.

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

      async def call(self, context, arguments) -> EchoResult:
          a = parse(arguments, Echo.Arguments)
          return EchoResult(call_id="", content=a.text)  # Echo's own result type
"""

from __future__ import annotations

from collections.abc import (
    Callable,
    Mapping,
    Sequence,
    Set as AbstractSet,
)
from dataclasses import MISSING, dataclass, field, fields, is_dataclass, replace
from pathlib import Path
from types import NoneType, UnionType
from typing import (
    TYPE_CHECKING,
    Annotated,
    Literal,
    Protocol,
    TypeAliasType,
    TypeGuard,
    cast,
    get_args,
    get_origin,
    get_type_hints,
    runtime_checkable,
)

import difflib
import functools
import hashlib
import reprlib
import uuid

from treekle import PlainTree
from wesearch.fetch import FetchSession
from wesearch.fetch.session import default_session


if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from _typeshed import DataclassInstance

    from sagent.session.types import AnyToolResult
    from sagent.tools.display import ToolDisplay


__all__ = [
    "Arg",
    "Briefing",
    "FileTracker",
    "Reminding",
    "Skipping",
    "Tool",
    "ToolContext",
    "checked",
    "json_schema",
    "parse",
]


class Tool(Protocol):
    """Something the model can call by name."""

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
        them; ``"*"`` runs alone, after every keyed call before it; ``None``
        runs beside anything.

        Args:
          context: The calling agent's state.
          arguments: What the model passed, perhaps not fitting the schema.

        Returns:
          key: The key, ``"*"``, or ``None``.

        """
        ...

    async def call(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> AnyToolResult:
        """Run with the model's arguments, and return what the tool did.

        Args:
          context: The calling agent's state.
          arguments: What the model passed.

        Returns:
          result: The session record of what the tool did.

        Raises:
          ValueError: The arguments do not fit the schema; the runtime returns
            the message to the model as an error.

        """
        ...


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

    def fit_head_tail(
        self,
        units: Sequence[str],
        *,
        reserve: int = 0,
        cap: int = 0,
        head_ratio: float = 0.4,
        omission_note: str = "",
    ) -> tuple[str, int]:
        """Join ``units`` keeping head and tail when total length exceeds budget.

        For command execution outputs (e.g. test suites, compiler logs), critical
        debugging information (assertion failures, stack traces, exit summaries)
        is concentrated at the tail, while invocation context is at the head.
        When outputs exceed the character budget, preserves both boundaries and
        elides the middle, aligning with the U-shaped attention distribution
        (Liu et al., 'Lost in the Middle') and SWE-bench observation designs.

        Args:
          units: Rendered lines or pieces, in order.
          reserve: Room to leave for diagnostics or trailing notes.
          cap: Tool-specific character cap, if below ``max_result_chars``; 0 for none.
          head_ratio: Share of character budget allocated to the start (default 0.4).
          omission_note: Optional note appended to the middle omission marker (e.g. spill path).

        Returns:
          body: The head units, middle omission marker, and tail units, joined.
          kept: Total count of units preserved across head and tail.

        """
        if not units:
            return "", 0

        limit = min(cap, self.max_result_chars) if cap else self.max_result_chars
        effective_limit = max(1, limit - reserve)
        total_len = sum(len(u) for u in units)

        # Fast path: all units fit without truncation.
        if total_len <= effective_limit:
            return "".join(units), len(units)

        # Reserve room for the omission marker: "\n... (N lines omitted<note>)\n"
        sample_marker = f"\n... ({len(units):,} lines omitted{omission_note})\n"
        content_budget = max(1, effective_limit - len(sample_marker))

        head_budget = max(0, int(content_budget * head_ratio))
        tail_budget = max(0, content_budget - head_budget)

        # Collect head units from the beginning.
        head: list[str] = []
        head_used = 0
        for u in units:
            if head_used + len(u) > head_budget:
                break
            head.append(u)
            head_used += len(u)

        # Collect tail units from the end without overlapping the head.
        tail: list[str] = []
        tail_used = 0
        head_count = len(head)
        for i in range(len(units) - 1, head_count - 1, -1):
            u = units[i]
            if tail_used + len(u) > tail_budget:
                break
            tail.append(u)
            tail_used += len(u)
        tail.reverse()

        # If strict ratio left both empty, guarantee at least the first unit partially.
        if not head and not tail:
            first = units[0]
            omitted = max(0, len(first) - content_budget)
            return (
                f"{first[:content_budget]}\n... ({omitted:,} chars omitted{omission_note})",
                1,
            )

        omitted_lines = len(units) - (len(head) + len(tail))
        marker = f"\n... ({omitted_lines:,} lines omitted{omission_note})\n"
        body = f"{''.join(head)}{marker}{''.join(tail)}"
        return body, len(head) + len(tail)


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
    if hint is PlainTree:
        return value
    if isinstance(hint, TypeAliasType):
        return checked(value, _value(hint), key=key)
    origin, args = get_origin(hint), _args(hint)
    if origin is Annotated:
        return checked(value, args[0], key=key)
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


def _checked_record[R: DataclassInstance](
    value: object,
    kind: type[R],
    *,
    key: str,
) -> R:
    """Return ``value``, a ``kind``, once each of its fields is of its type."""
    if not isinstance(value, kind):
        raise _misfit(key, kind.__name__, value)
    # ``value`` may be a subclass of ``kind``, whose fields are its own.
    hints = _hints(type(value))
    for f in fields(value):
        _ = checked(
            cast(object, getattr(value, f.name)),
            hints[f.name],
            key=f"{key}.{f.name}",
        )
    return value


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
