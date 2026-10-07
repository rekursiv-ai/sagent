"""Convert Python values to and from plain data, losslessly.

Plain data is ``str``, ``int``, ``float``, ``bool``, ``None``, and mappings and
sequences of them. :data:`PlainTree` is any such tree, read-only;
:data:`MutablePlainTree` is the dict-and-list form, which is also a
``PlainTree``.
This module never produces text or bytes; a caller hands plain data to
whichever writer it chooses.

What a dialect cannot carry natively is written as a ``py/*`` tag, using
jsonpickle's spelling where one exists:

- ``py/object``, ``py/type``, ``py/function``: a class instance, a class, or a
  function, by dotted import path.
- ``py/tuple``, ``py/set``, ``py/frozenset``, ``py/mappingproxy``: containers a
  dialect flattens to lists or dicts.
- ``py/float``, ``py/complex``, ``py/b64``, ``py/path``, ``py/uuid``,
  ``py/datetime``: values a dialect lacks.
- ``py/reduce``, ``py/inline``, ``py/hook``: a pickle reduce recipe, a deferred
  call, and a caller-supplied codec.
- ``py/id``: a back-reference, so a shared object or a cycle survives.
- ``json://``: a dict key that is not a string, or that reads like a tag.

``to_plain`` takes a ``dialect`` naming the target format, which decides what
must be tagged; ``"json"`` tags non-finite floats, which ``"python"`` keeps
native. Both tag everything else alike.
``mutable`` picks dicts and lists, the default, or ``MappingProxyType`` and
tuples; either way the result is freshly built and shared with no one.
``from_plain`` takes no dialect: it accepts the tagged and native form of every
value alike, mutable or not.

``from_plain(data, T)`` lets a tag decide where one is present and ``T``
everywhere else, so a Protocol-typed field reads back as the class its tag
names. A value that fits neither raises :class:`ReadError`; there are no silent
defaults. ``allow_imports`` gates every tag that imports or calls code.

Each tagged type has an encoder and a decoder, paired by name. A tag only
selects which decoder runs; the decoder is the same one untagged data reaches
through ``T``. Leaves pass through unencoded, and some targets (``Literal``,
unions, ``object``) only decode.

``immutable`` and ``mutable`` copy a plain tree to ``MappingProxyType`` and
tuples, or to dicts and lists, whichever form it starts in. Both always copy.

``parse(text, T)`` reads what a person typed, such as a command-line override.
It is lenient about syntax and strict about type: it accepts JSON, falls back
to the bare word, and lets ``T`` decide, so ``parse("123", str)`` is ``"123"``.
The result then passes through ``from_plain``, so text that does not fit ``T``
raises :class:`ReadError`, never a default.
"""

from __future__ import annotations

from collections.abc import (
    Callable,
    Hashable,
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    MutableSequence,
    MutableSet,
    Sequence,
    Set as AbstractSet,
)
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from functools import partial
from numbers import Real
from pathlib import Path, PurePath
from types import (
    BuiltinFunctionType,
    FunctionType,
    MappingProxyType,
    ModuleType,
    UnionType,
)
from typing import (
    Annotated,
    Final,
    Literal,
    NotRequired,
    Protocol,
    Required,
    TypeGuard,
    cast,
    get_args,
    get_origin,
    get_type_hints,
    is_typeddict,
    overload,
)
from uuid import UUID
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import base64
import binascii
import importlib
import inspect
import json
import math
import sys
import typing
import weakref

from sagent.lib.absent import ABSENT


__all__ = [
    "Dialect",
    "FieldPath",
    "Hooks",
    "Invalid",
    "MutablePlainTree",
    "Plain",
    "PlainTree",
    "ReadError",
    "from_plain",
    "immutable",
    "loads",
    "mutable",
    "parse",
    "to_plain",
]


# A leaf of a plain tree. ``float`` includes IEEE-754 NaN and signed infinities.
type Plain = str | int | float | bool | None


# The leaf union is inlined rather than naming ``Plain``: ty 0.0.52 panics on a
# self-recursive alias that names a union alias.
# https://github.com/astral-sh/ty/issues/3835
type PlainTree = (
    str | int | float | bool | Sequence[PlainTree] | Mapping[str, PlainTree] | None
)
"""Any plain tree, read-only: dicts and lists, or ``MappingProxyType`` and tuples."""


type MutablePlainTree = (
    str
    | int
    | float
    | bool
    | list[MutablePlainTree]
    | dict[str, MutablePlainTree]
    | None
)
"""A plain tree of dicts and lists, which may be edited."""


type Dialect = Literal["json", "python"]
"""A target format; it decides which values ``to_plain`` must tag.

``"json"`` tags non-finite floats; ``"python"`` keeps them native. Both tag
everything else alike, so either result fits the same tree types.
"""


type Hooks = Mapping[type, tuple[Callable[..., object], Callable[..., object]]]
"""Runtime types paired with their ``(encode, decode)`` callbacks."""


type FieldPath = tuple[str | int, ...]
"""Where a value sits: field names and indices leading to it; ``()`` is the value."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Invalid:
    """A value that did not match its target type, kept as it was stated."""

    raw: PlainTree
    reason: str = ""


class ReadError(TypeError, ValueError):
    """Plain data does not match the requested type.

    Attributes:
      partial: The value as far as it reads, an :class:`Invalid` in place of
        each bad part; the whole value is one when nothing reads.
      bad: Each bad part, by where it sits.

    """

    partial: object
    bad: Mapping[FieldPath, Invalid]

    def __init__(
        self,
        message: str,
        *,
        partial: object = None,
        bad: Mapping[FieldPath, Invalid] = MappingProxyType({}),
    ) -> None:
        super().__init__(message)
        self.partial = partial
        self.bad = bad


@overload
def to_plain(
    obj: object,
    *,
    dialect: Dialect = ...,
    mutable: Literal[True] = ...,
    hooks: Hooks = ...,
) -> MutablePlainTree: ...


@overload
def to_plain(
    obj: object,
    *,
    dialect: Dialect = ...,
    mutable: bool,
    hooks: Hooks = ...,
) -> PlainTree: ...


def to_plain(
    obj: object,
    *,
    dialect: Dialect = "json",
    mutable: bool = True,
    hooks: Hooks = MappingProxyType({}),
) -> PlainTree:
    """Convert a Python value to plain data, tagging what ``dialect`` lacks.

    Args:
      obj: The value to convert; any object graph, shared references and cycles
        included.
      dialect: The target format, which decides which values need a tag.
      mutable: Build dicts and lists when true; ``MappingProxyType`` and tuples
        when false.
      hooks: Codecs for leaf types with no built-in conversion, such as tensors.

    Returns:
      tree: Fresh plain data that ``from_plain`` turns back into ``obj``.

    Raises:
      TypeError: A leaf has no conversion and no hook.

    """
    state = _Encoding(
        dialect=dialect,
        hooks=hooks,
        make_list=_thawed_list if mutable else _frozen_list,
        make_dict=_thawed_dict if mutable else _frozen_dict,
    )
    return _encode(obj, state)


@overload
def from_plain[T](
    data: object,
    target: type[T],
    *,
    hooks: Hooks = ...,
    allow_imports: bool = ...,
    strict: bool = ...,
) -> T: ...


@overload
def from_plain[T](
    data: object,
    target: type[T],
    *,
    default: None,
    hooks: Hooks = ...,
    allow_imports: bool = ...,
    strict: bool = ...,
) -> T | None: ...


@overload
def from_plain[T](
    data: object,
    target: type[T],
    *,
    default: T,
    hooks: Hooks = ...,
    allow_imports: bool = ...,
    strict: bool = ...,
) -> T: ...


@overload
def from_plain(
    data: object,
    target: object,
    *,
    default: object = ...,
    hooks: Hooks = ...,
    allow_imports: bool = ...,
    strict: bool = ...,
) -> object: ...


def from_plain(
    data: object,
    target: object,
    *,
    default: object = ABSENT,
    hooks: Hooks = MappingProxyType({}),
    allow_imports: bool = False,
    strict: bool = True,
) -> object:
    """Convert plain data to ``target``; a ``py/*`` tag decides where present.

    Read one field as ``from_plain(row.get("name"), str, default="")``.

    Args:
      data: A plain tree, mutable or not, as any reader or ``to_plain`` made it.
        A non-plain leaf already of its target type, such as a ``datetime``
        from ``tomllib``, passes through.
      target: The type to produce; ``object`` reads by tags alone.
      default: Returned when ``data`` is ``None`` (a missing or null field);
        omit to read ``None`` like any other value.
      hooks: The codecs ``to_plain`` used.
      allow_imports: Whether tags may import modules and call code they name.
        A ``py/object`` tag naming the target dataclass itself needs none.
      strict: When false, also read numeric and boolean text (``"3"`` as
        ``3``) and integral numbers across ``int``, ``float``, and ``bool``.

    Returns:
      value: ``data`` as a ``target``, or ``default``.

    Raises:
      ReadError: ``data`` does not fit ``target``, or a tag needs imports while
        ``allow_imports`` is false.

    """
    if data is None and default is not ABSENT:
        return default
    state = _Decoding(
        hooks=hooks,
        allow_imports=allow_imports,
        strict=strict,
        untagged=not _holds_tags(data),
    )
    # A non-plain leaf is checked where it is read, by ``_passes_through``.
    tree = cast("PlainTree", data)
    try:
        value = _decode(tree, target, (), state)
    except ReadError as error:
        whole = Invalid(raw=tree, reason=str(error))
        raise ReadError(
            str(error),
            partial=whole,
            bad=MappingProxyType({**state.bad, (): whole}),
        ) from error
    if state.bad:
        at, first = next(iter(state.bad.items()))
        raise ReadError(
            f"{len(state.bad)} part(s) do not read; first at {list(at)}: "
            f"{first.reason}",
            partial=value,
            bad=MappingProxyType(state.bad),
        )
    return value


def loads(text: str | bytes) -> MutablePlainTree:
    """Parse JSON text into a mutable plain tree; ``NaN``/``Infinity`` read too.

    Args:
      text: JSON document.

    Returns:
      tree: Plain dicts, lists, and scalars.

    Raises:
      json.JSONDecodeError: ``text`` is not valid JSON.

    """
    return cast("MutablePlainTree", json.loads(text))


@overload
def parse[T](text: str, target: type[T]) -> T: ...


@overload
def parse(text: str, target: object) -> object: ...


def parse(text: str, target: object) -> object:
    """Read typed ``text`` as ``target``, accepting JSON or the bare word.

    Args:
      text: What a person typed, e.g. ``3e-4``, ``run_a``, or ``[1, 2]``.
      target: The type to produce.

    Returns:
      value: ``text`` as a ``target``.

    Raises:
      ReadError: ``text`` does not read as ``target``.

    """
    plain = _text_to_plain(text, target)
    try:
        return from_plain(plain, target)
    except ReadError as error:
        if plain == text:
            raise
        # The bare word is the fallback reading; when it fails too, the JSON
        # reading's error says more about what the person meant.
        try:
            return from_plain(text, target)
        except ReadError:
            raise error from None


@overload
def immutable(tree: Plain) -> Plain: ...


@overload
def immutable(tree: Mapping[str, object]) -> Mapping[str, PlainTree]: ...


@overload
def immutable(tree: Sequence[object]) -> Sequence[PlainTree]: ...


@overload
def immutable(tree: object) -> PlainTree: ...


def immutable(tree: object) -> PlainTree:
    """Copy a plain tree: mappings to ``MappingProxyType``, sequences to tuples.

    Args:
      tree: A plain tree, mutable or not.

    Returns:
      frozen: The read-only copy.

    Raises:
      TypeError: ``tree`` holds a non-str key or a value that is not plain.

    """
    return _freeze(tree)


@overload
def mutable(tree: Plain) -> Plain: ...


@overload
def mutable(tree: Mapping[str, object]) -> dict[str, MutablePlainTree]: ...


@overload
def mutable(tree: Sequence[object]) -> list[MutablePlainTree]: ...


@overload
def mutable(tree: object) -> MutablePlainTree: ...


def mutable(tree: object) -> MutablePlainTree:
    """Copy a plain tree: each mapping to a dict, each sequence to a list.

    Args:
      tree: A plain tree, mutable or not.

    Returns:
      thawed: The mutable copy.

    Raises:
      TypeError: ``tree`` holds a non-str key or a value that is not plain.

    """
    return _thaw(tree)


def _freeze(tree: object) -> PlainTree:
    """Copy ``tree`` to ``MappingProxyType`` and tuples at every level."""
    if tree is None or isinstance(tree, (str, int, float)):
        return tree
    if isinstance(tree, Mapping):
        node = cast("Mapping[object, object]", tree)
        return MappingProxyType(
            {_plain_key(key): _freeze(item) for key, item in node.items()},
        )
    return tuple(_freeze(item) for item in _plain_sequence(tree))


def _thaw(tree: object) -> MutablePlainTree:
    """Copy ``tree`` to dicts and lists at every level."""
    if tree is None or isinstance(tree, (str, int, float)):
        return tree
    if isinstance(tree, Mapping):
        node = cast("Mapping[object, object]", tree)
        return {_plain_key(key): _thaw(item) for key, item in node.items()}
    return [_thaw(item) for item in _plain_sequence(tree)]


def _plain_key(key: object) -> str:
    """Return ``key`` when it is a str, the only key plain data allows."""
    if isinstance(key, str):
        return key
    raise TypeError(f"plain mapping key must be str, got {key!r}")


def _plain_sequence(tree: object) -> Sequence[object]:
    """Return ``tree`` when it is a plain sequence."""
    if isinstance(tree, Sequence) and not isinstance(tree, (bytes, bytearray)):
        return tree
    raise TypeError(f"cannot represent {type(tree).__name__} as plain data")


@dataclass(slots=True, kw_only=True)
class _Encoding:
    """State of one ``to_plain`` walk.

    Attributes:
      dialect: The target format.
      hooks: Codecs for leaf types with no built-in conversion.
      make_list: Builds a sequence node: ``list`` or ``tuple``.
      make_dict: Builds a mapping node: ``dict`` or ``MappingProxyType``.
      seen: Encounter number by ``id``, for ``py/id``.
      alive: Every numbered object, so its ``id`` is not reused mid-walk.

    """

    dialect: Dialect
    hooks: Hooks
    make_list: Callable[[list[PlainTree]], PlainTree]
    make_dict: Callable[[dict[str, PlainTree]], PlainTree]
    seen: dict[int, int] = field(default_factory=dict[int, int])
    alive: list[object] = field(default_factory=list[object])


@dataclass(slots=True, kw_only=True)
class _Decoding:
    """State of one ``from_plain`` walk.

    Attributes:
      hooks: The codecs ``to_plain`` used.
      allow_imports: Whether tags may import modules and call code they name.
      strict: Whether scalars must arrive as their own plain kind.
      untagged: Whether the input holds no tag, so an ``object`` part is
        returned as the input itself rather than copied.
      built: Every numbered object, by encounter order, for ``py/id``.
      bad: Each part that did not read, by where it sits.

    """

    hooks: Hooks
    allow_imports: bool
    strict: bool = True
    untagged: bool = False
    built: list[object] = field(default_factory=list[object])
    bad: dict[FieldPath, Invalid] = field(default_factory=dict[FieldPath, Invalid])


class _Inline(Protocol):
    """A value owning both halves of the ``py/inline`` deferred-call protocol.

    Decoding allocates the class without ``__init__``, so a cycle can reference
    it before its children exist, then hands the recipe back to populate it.
    """

    def __custom_json_inline__(
        self,
    ) -> tuple[object, Sequence[object], Mapping[str, object]]: ...

    def __custom_json_inline_init__(
        self,
        func: object,
        args: Sequence[object],
        kwargs: Mapping[str, object],
    ) -> None: ...


type _Encoder = Callable[[object, _Encoding], PlainTree]
"""Encode one value into whichever kind of tree ``state`` builds."""


type _Decoder = Callable[[PlainTree, object, FieldPath, _Decoding], object]
"""Decode one node, untagged or a tag's payload, as ``target``."""


def _thawed_list(items: list[PlainTree]) -> PlainTree:
    """Return a sequence node as the list it was built in."""
    return items


def _frozen_list(items: list[PlainTree]) -> PlainTree:
    """Return a sequence node as a tuple."""
    return tuple(items)


def _thawed_dict(items: dict[str, PlainTree]) -> PlainTree:
    """Return a mapping node as the dict it was built in."""
    return items


def _frozen_dict(items: dict[str, PlainTree]) -> PlainTree:
    """Return a mapping node as a ``MappingProxyType``."""
    return MappingProxyType(items)


def _encode(value: object, state: _Encoding) -> PlainTree:
    """Encode one node: plain passthrough, ``py/id`` repeat, else its encoder."""
    kind = type(value)
    if value is None or (isinstance(value, (str, int)) and kind in _NATIVE_LEAVES):
        return value
    if kind is float:
        return _encode_float(value, state)
    seen = state.seen.get(id(value))
    if seen is not None:
        return state.make_dict({"py/id": seen})
    if kind in state.hooks:
        return _encode_hooked(value, state)
    return _encoder_for(kind)(value, state)


def _text_to_plain(text: str, target: object) -> MutablePlainTree:
    """Turn ``text`` into a plain tree the way ``target`` reads it."""
    if _is_verbatim(target):
        loaded = _loads_or_word(text)
        return loaded if isinstance(loaded, str) else text
    number = _non_finite(text)
    if number is not None:
        return number
    loaded = _loads_or_word(text)
    # Quoting is syntax: ``count="7"`` reads the quoted text as typed.
    if isinstance(loaded, str) and loaded != text:
        return _text_to_plain(loaded, target)
    return loaded


def _is_verbatim(target: object) -> bool:
    """Return whether ``target`` reads the raw text, e.g. str, Path, or UUID."""
    return isinstance(target, type) and issubclass(
        target,
        (str, PurePath, UUID, datetime),
    )


def _loads_or_word(text: str) -> MutablePlainTree:
    """Return ``text`` as JSON, or the bare word when it is not JSON."""
    try:
        return cast("MutablePlainTree", json.loads(text))
    except json.JSONDecodeError:
        return text


def _non_finite(text: str) -> float | None:
    """Return the float ``inf``, ``-inf``, or ``nan`` spells, else ``None``."""
    try:
        number = float(text.strip())
    except ValueError:
        return None
    return None if math.isfinite(number) else number


def _decode(
    data: PlainTree,
    target: object,
    at: FieldPath,
    state: _Decoding,
) -> object:
    """Decode one node: ``py/id``, else its tag's decoder, else ``target``'s."""
    target = _resolve_alias(target)
    if not _is_plain(data):
        if _fits(data, target):
            return data
        # ``json.loads(parse_float=Decimal)`` and similar readers yield numbers
        # that are not ``float``; ``custom_json.convert`` read them as floats.
        if target is float and isinstance(data, (Real, Decimal)):
            return float(data)
        raise ReadError(f"cannot read {data!r} as {target}")
    decoder = _decoder_for(target)
    if decoder is _decode_union or not isinstance(data, Mapping):
        return decoder(data, target, at, state)
    if "py/id" in data:
        value = _decode_reference(data, state)
    else:
        tag = _tag_of(data)
        # A mapping target reads a tag key as data, as ``custom_json.convert``
        # did; callers then pick the payload by key. Without imports no class
        # can be built, so an untyped ``py/object`` record stays a record too.
        if (
            tag is None
            or (decoder is _decode_dict and tag != "py/mappingproxy")
            or (
                tag == "py/object"
                and decoder is _decode_any
                and not state.allow_imports
            )
        ):
            return decoder(data, target, at, state)
        if tag != "py/object" and len(data) != 1:
            raise ReadError(f"invalid {tag} envelope: {dict(data)!r}")
        payload = data if tag == "py/object" else data[tag]
        value = _TAG_DECODERS[tag](payload, target, at, state)
    if not _fits(value, target):
        raise ReadError(f"cannot read {value!r} as {target}")
    return value


def _is_plain(data: object) -> bool:
    """Return whether ``data``'s top node is plain: a leaf, mapping, or sequence."""
    return data is None or (
        isinstance(data, (str, int, float, Mapping, Sequence))
        and not isinstance(data, (bytes, bytearray))
    )


def _encoder_for(kind: type) -> _Encoder:
    """Return the encoder for ``kind``, choosing once per class and caching it."""
    encoder = _ENCODERS.get(kind) or _ENCODER_CACHE.get(kind)
    if encoder is not None:
        return encoder
    bases = kind.__mro__
    if type in bases:
        encoder = _encode_type
    elif Enum in bases:
        encoder = _encode_enum
    elif kind is Path or kind is _CONCRETE_PATH:
        encoder = _encode_path
    elif kind is UUID:
        encoder = _encode_uuid
    elif kind is datetime:
        encoder = _encode_datetime
    elif _owns_inline(kind):
        encoder = _encode_inline
    elif is_dataclass(kind):
        encoder = _encode_object
    else:
        encoder = _encode_reduce
    _ENCODER_CACHE[kind] = encoder
    return encoder


def _decoder_for(target: object) -> _Decoder:
    """Return the decoder for ``target``, keyed on its origin."""
    origin: object = get_origin(target) or target
    decoder = _DECODERS.get(origin)
    if decoder is not None:
        return decoder
    if not _is_class(origin):
        return _decode_any
    if Enum in origin.__mro__:
        return _decode_enum
    if is_dataclass(origin):
        return _decode_object
    if is_typeddict(origin):
        return _decode_typeddict
    if (
        origin.__dict__.get("_is_protocol") is True
        or inspect.isabstract(origin)
        or (origin.__module__ == "typing" and origin.__qualname__ == "Any")
    ):
        return _decode_any
    return _decode_object


def _tag_of(node: Mapping[str, PlainTree]) -> str | None:
    """Return the highest-precedence ``py/*`` tag on ``node``, if any."""
    if not _has_tag_key(node):
        return None
    for tag in _TAGS:
        if tag in node:
            return tag
    raise ReadError(f"unknown tag among {sorted(map(str, node))}")


def _holds_tags(data: object) -> bool:
    """Return whether any mapping in ``data`` has a tag or escaped key."""
    if isinstance(data, Mapping):
        node = cast("Mapping[object, object]", data)
        return any(
            isinstance(key, str) and key.startswith(("py/", "json://")) for key in node
        ) or any(_holds_tags(item) for item in node.values())
    if isinstance(data, Sequence) and not isinstance(data, (str, bytes, bytearray)):
        return any(_holds_tags(item) for item in data)
    return False


def _has_tag_key(keys: Iterable[object]) -> bool:
    """Return whether any key reads as a tag; a YAML reader may yield non-str keys."""
    return any(isinstance(key, str) and key.startswith("py/") for key in keys)


def _fits(value: object, target: object) -> bool:
    """Return whether a tag-decoded ``value`` is acceptable as ``target``."""
    target = _resolve_alias(target)
    origin: object = get_origin(target) or target
    if origin is UnionType or origin is typing.Union:  # pyright: ignore[reportDeprecated] -- A runtime origin test for pre-3.14 ``Optional``/``Union``, not an annotation.
        return any(
            _fits(value, member)
            for member in cast("tuple[object, ...]", get_args(target))
        )
    if origin is Literal:
        return value in cast("tuple[object, ...]", get_args(target))
    if not isinstance(origin, type):
        return True
    try:
        return isinstance(value, origin)
    except TypeError:
        return True


# Registering before encoding children is what makes cycles terminate: a
# back-reference reached below finds the parent already numbered.
def _register(value: object, state: _Encoding) -> None:
    """Assign ``value`` its encounter number, before its children are encoded."""
    state.seen[id(value)] = len(state.alive)
    state.alive.append(value)


def _rollback(state: _Encoding, mark: int) -> None:
    """Forget every number assigned since ``len(state.alive)`` was ``mark``."""
    for value in state.alive[mark:]:
        del state.seen[id(value)]
    del state.alive[mark:]


def _decode_reference(
    node: Mapping[str, PlainTree],
    state: _Decoding,
) -> object:
    """Return the object a ``py/id`` envelope names."""
    index = node["py/id"]
    if (
        len(node) != 1
        or not isinstance(index, int)
        or isinstance(index, bool)
        or index < 0
        or index >= len(state.built)
    ):
        raise ReadError(f"invalid py/id reference: {dict(node)!r}")
    value = state.built[index]
    if value is _PENDING:
        raise ReadError(f"py/id {index} names a value still being built")
    return value


def _invalid(
    value: PlainTree,
    reason: str,
    at: FieldPath,
    state: _Decoding,
) -> Invalid:
    """Record ``value`` as bad at ``at`` and return its placeholder."""
    invalid = Invalid(raw=value, reason=reason)
    state.bad[at] = invalid
    return invalid


def _decode_part(
    data: PlainTree,
    target: object,
    at: FieldPath,
    state: _Decoding,
) -> object:
    """Decode a container's part, recording an :class:`Invalid` if it fails."""
    try:
        return _decode(data, target, at, state)
    except ReadError as error:
        return _invalid(data, str(error), at, state)


def _decode_none(
    data: PlainTree,
    target: object,
    at: FieldPath,
    state: _Decoding,
) -> None:
    """Decode ``None``."""
    del target, at, state
    if data is not None:
        raise ReadError(f"cannot read {data!r} as None")


def _decode_bool(
    data: PlainTree,
    target: object,
    at: FieldPath,
    state: _Decoding,
) -> bool:
    """Decode a bool; lax also reads ``0``/``1`` and true/false text."""
    del target, at
    if isinstance(data, bool):
        return data
    if not state.strict:
        if isinstance(data, str):
            data = _LAX_BOOLS.get(data.lower(), data)
        if isinstance(data, int) and data in (0, 1):
            return bool(data)
    raise ReadError(f"cannot read {data!r} as bool")


def _decode_int(
    data: PlainTree,
    target: object,
    at: FieldPath,
    state: _Decoding,
) -> int:
    """Decode an int; lax also reads integral floats and numeric text."""
    del target, at
    if isinstance(data, int) and not isinstance(data, bool):
        return data
    if not state.strict and not isinstance(data, bool):
        number = _lax_number(data)
        if number is not None and number.is_integer():
            return int(number)
    raise ReadError(f"cannot read {data!r} as int")


def _lax_number(data: PlainTree) -> float | None:
    """Return the finite number ``data`` is or spells, else ``None``."""
    if isinstance(data, str):
        try:
            data = float(data)
        except ValueError:
            return None
    if isinstance(data, (int, float)) and math.isfinite(data):
        return float(data)
    return None


def _decode_str(
    data: PlainTree,
    target: object,
    at: FieldPath,
    state: _Decoding,
) -> str:
    """Decode a str."""
    del target, at, state
    if isinstance(data, str):
        return data
    raise ReadError(f"cannot read {data!r} as str")


def _encode_float(value: object, state: _Encoding) -> PlainTree:
    """Encode a float, as ``py/float`` when non-finite and the dialect must."""
    number = cast("float", value)
    if math.isfinite(number) or state.dialect == "python":
        return number
    return state.make_dict({"py/float": repr(number)})


def _decode_float(
    data: PlainTree,
    target: object,
    at: FieldPath,
    state: _Decoding,
) -> float:
    """Decode a float from an int, a float, or a ``py/float`` payload."""
    del target, at
    if isinstance(data, float):
        return data
    if isinstance(data, int) and not isinstance(data, bool):
        return float(data)
    if isinstance(data, str):
        try:
            number = float(data.strip())
        except ValueError:
            number = None
        if number is not None and (not math.isfinite(number) or not state.strict):
            return number
    raise ReadError(f"cannot read {data!r} as float")


def _encode_complex(value: object, state: _Encoding) -> PlainTree:
    """Encode a complex as ``py/complex`` of its real and imaginary floats."""
    number = cast("complex", value)
    parts = [_encode_float(number.real, state), _encode_float(number.imag, state)]
    return state.make_dict({"py/complex": state.make_list(parts)})


def _decode_complex(
    data: PlainTree,
    target: object,
    at: FieldPath,
    state: _Decoding,
) -> complex:
    """Decode a complex: a real number, or a ``py/complex`` pair."""
    if isinstance(data, (int, float)) and not isinstance(data, bool):
        return complex(data)
    if isinstance(data, Sequence) and not isinstance(data, str) and len(data) == 2:
        return complex(
            _decode_float(data[0], float, (*at, 0), state),
            _decode_float(data[1], float, (*at, 1), state),
        )
    del target
    raise ReadError(f"cannot read {data!r} as complex")


def _encode_bytes(value: object, state: _Encoding) -> PlainTree:
    """Encode bytes as ``py/b64``."""
    text = base64.b64encode(cast("bytes", value)).decode()
    return state.make_dict({"py/b64": text})


def _decode_bytes(
    data: PlainTree,
    target: object,
    at: FieldPath,
    state: _Decoding,
) -> bytes:
    """Decode bytes from base64 text."""
    del target, at, state
    if isinstance(data, str):
        try:
            return base64.b64decode(data, validate=True)
        except (binascii.Error, ValueError) as error:
            raise ReadError(f"cannot read {data!r} as bytes: {error}") from error
    raise ReadError(f"cannot read {data!r} as bytes")


def _encode_path(value: object, state: _Encoding) -> PlainTree:
    """Encode a path as ``py/path``."""
    return state.make_dict({"py/path": str(value)})


def _decode_path(
    data: PlainTree,
    target: object,
    at: FieldPath,
    state: _Decoding,
) -> Path:
    """Decode a ``Path`` from text."""
    del target, at, state
    if isinstance(data, str):
        return Path(data)
    raise ReadError(f"cannot read {data!r} as Path")


def _encode_uuid(value: object, state: _Encoding) -> PlainTree:
    """Encode a UUID as ``py/uuid``."""
    return state.make_dict({"py/uuid": str(value)})


def _decode_uuid(
    data: PlainTree,
    target: object,
    at: FieldPath,
    state: _Decoding,
) -> UUID:
    """Decode a ``UUID`` from its canonical text."""
    del target, at, state
    if isinstance(data, str):
        try:
            return UUID(data)
        except ValueError as error:
            raise ReadError(f"cannot read {data!r} as UUID") from error
    raise ReadError(f"cannot read {data!r} as UUID")


def _encode_datetime(value: object, state: _Encoding) -> PlainTree:
    """Encode a datetime as ``py/datetime``, keeping a named zone."""
    moment = cast("datetime", value)
    text = moment.isoformat()
    if isinstance(moment.tzinfo, ZoneInfo):
        text = f"{text}[{moment.tzinfo.key}]"
    return state.make_dict({"py/datetime": text})


def _decode_datetime(
    data: PlainTree,
    target: object,
    at: FieldPath,
    state: _Decoding,
) -> datetime:
    """Decode an ISO 8601 ``datetime``, restoring a bracketed zone name."""
    del target, at, state
    if not isinstance(data, str):
        raise ReadError(f"cannot read {data!r} as datetime")
    text, bracket, zone = data.partition("[")
    try:
        moment = datetime.fromisoformat(text)
    except ValueError as error:
        raise ReadError(f"cannot read {data!r} as datetime") from error
    if not bracket:
        return moment
    if moment.utcoffset() is None:
        raise ReadError(f"cannot read {data!r} as datetime: timestamp is naive")
    if not zone.endswith("]") or len(zone) == 1:
        raise ReadError(f"cannot read {data!r} as datetime: invalid named zone")
    try:
        named = ZoneInfo(zone.removesuffix("]"))
    except (ValueError, ZoneInfoNotFoundError) as error:
        raise ReadError(
            f"cannot read {data!r} as datetime: invalid named zone",
        ) from error
    return moment.astimezone(named)


def _encode_enum(value: object, state: _Encoding) -> PlainTree:
    """Encode an enum member as ``py/reduce`` of its class and value."""
    return _encode_reduce(value, state)


def _decode_enum(
    data: PlainTree,
    target: object,
    at: FieldPath,
    state: _Decoding,
) -> Enum:
    """Decode an untagged value as a member of ``target``; tagged ones are reduces."""
    del at, state
    if (
        isinstance(target, type)
        and issubclass(target, Enum)
        and isinstance(data, (str, int, float))
        and not isinstance(data, bool)
    ):
        try:
            return target(data)
        except ValueError:
            pass
    raise ReadError(f"cannot read {data!r} as {target}")


def _decode_literal(
    data: PlainTree,
    target: object,
    at: FieldPath,
    state: _Decoding,
) -> Plain | Enum:
    """Decode one of a ``Literal``'s values, matching bool and int apart."""
    for option in cast("tuple[object, ...]", get_args(target)):
        if isinstance(option, Enum):
            try:
                member = _decode_enum(data, type(option), at, state)
            except ReadError:
                continue
            if member is option:
                return option
        elif (
            (option is None or isinstance(option, (str, int, float)))
            and type(option) is type(data)
            and option == data
        ):
            return option
    raise ReadError(f"cannot read {data!r} as {target}")


def _encode_list(value: object, state: _Encoding) -> PlainTree:
    """Encode a list, numbering it for ``py/id``."""
    items = cast("Sequence[object]", value)
    _register(value, state)
    return state.make_list([_encode(item, state) for item in items])


def _decode_list(
    data: PlainTree,
    target: object,
    at: FieldPath,
    state: _Decoding,
) -> list[object]:
    """Decode a list, numbering it for ``py/id``."""
    if not isinstance(data, Sequence) or isinstance(data, str):
        raise ReadError(f"cannot read {data!r} as {target}")
    item = _element_type(target)
    result: list[object] = []
    state.built.append(result)
    result.extend(
        _decode_part(raw, item, (*at, index), state) for index, raw in enumerate(data)
    )
    return result


def _encode_tuple(value: object, state: _Encoding) -> PlainTree:
    """Encode a tuple as ``py/tuple``; tuples take no ``py/id`` number."""
    items = [_encode(item, state) for item in cast("tuple[object, ...]", value)]
    return state.make_dict({"py/tuple": state.make_list(items)})


def _decode_tuple(
    data: PlainTree,
    target: object,
    at: FieldPath,
    state: _Decoding,
) -> tuple[object, ...]:
    """Decode a fixed or variadic tuple; tuples take no ``py/id`` number."""
    if not isinstance(data, Sequence) or isinstance(data, str):
        raise ReadError(f"cannot read {data!r} as {target}")
    args = cast("tuple[object, ...]", get_args(target))
    if get_origin(target) is not tuple:
        types: tuple[object, ...] = (_element_type(target),) * len(data)
    elif len(args) == 2 and args[1] is Ellipsis:
        types = (args[0],) * len(data)
    elif len(args) != len(data):
        raise ReadError(f"expected {len(args)} items for {target}, got {len(data)}")
    else:
        types = args
    return tuple(
        _decode_part(raw, kind, (*at, index), state)
        for index, (raw, kind) in enumerate(zip(data, types, strict=True))
    )


def _encode_set(value: object, state: _Encoding) -> PlainTree:
    """Encode a set as ``py/set`` with members in a deterministic order."""
    members = cast("AbstractSet[object]", value)
    _register(value, state)
    ordered = sorted(members, key=lambda member: _set_order_key(member, state))
    items = [_encode(member, state) for member in ordered]
    return state.make_dict({"py/set": state.make_list(items)})


def _decode_set(
    data: PlainTree,
    target: object,
    at: FieldPath,
    state: _Decoding,
) -> set[object]:
    """Decode a set, numbering it for ``py/id``."""
    if not isinstance(data, Sequence) or isinstance(data, str):
        raise ReadError(f"cannot read {data!r} as {target}")
    item = _element_type(target)
    result: set[object] = set()
    state.built.append(result)
    for index, raw in enumerate(data):
        member = _decode_part(raw, item, (*at, index), state)
        try:
            result.add(member)
        except TypeError as error:
            raise ReadError(f"set member {member!r} is unhashable") from error
    return result


def _encode_frozenset(value: object, state: _Encoding) -> PlainTree:
    """Encode a frozenset as ``py/frozenset``; it takes no ``py/id`` number."""
    members = cast("AbstractSet[object]", value)
    ordered = sorted(members, key=lambda member: _set_order_key(member, state))
    items = [_encode(member, state) for member in ordered]
    return state.make_dict({"py/frozenset": state.make_list(items)})


def _decode_frozenset(
    data: PlainTree,
    target: object,
    at: FieldPath,
    state: _Decoding,
) -> frozenset[object]:
    """Decode a frozenset; frozensets take no ``py/id`` number."""
    if not isinstance(data, Sequence) or isinstance(data, str):
        raise ReadError(f"cannot read {data!r} as {target}")
    item = _element_type(target)
    members = [
        _decode_part(raw, item, (*at, index), state) for index, raw in enumerate(data)
    ]
    try:
        return frozenset(members)
    except TypeError as error:
        raise ReadError(f"frozenset members {members!r} are unhashable") from error


def _set_order_key(value: object, state: _Encoding) -> tuple[str, str]:
    """Return a set member's sort key without consuming ``py/id`` numbers."""
    mark = len(state.alive)
    try:
        encoded = _encode(value, state)
    finally:
        _rollback(state, mark)
    text = json.dumps(mutable(encoded), sort_keys=True, separators=(",", ":"))
    return repr(value), text


def _encode_dict(value: object, state: _Encoding) -> PlainTree:
    """Encode a dict, escaping keys through ``_encode_key``."""
    mapping = cast("Mapping[object, object]", value)
    _register(value, state)
    return state.make_dict(
        {
            _encode_key(key, state): _encode(member, state)
            for key, member in mapping.items()
        },
    )


def _decode_dict(
    data: PlainTree,
    target: object,
    at: FieldPath,
    state: _Decoding,
) -> dict[Hashable, object]:
    """Decode a dict, typing keys and values, numbering it for ``py/id``."""
    if not isinstance(data, Mapping):
        raise ReadError(f"cannot read {data!r} as {target}")
    args = cast("tuple[object, ...]", get_args(target))
    key_type, value_type = args if len(args) == 2 else (object, object)
    result: dict[Hashable, object] = {}
    state.built.append(result)
    for key, raw in data.items():
        where = (*at, key)
        try:
            decoded = _decode_key(key, key_type, where, state)
        except ReadError as error:
            _invalid(key, str(error), where, state)
            continue
        result[decoded] = _decode_part(raw, value_type, where, state)
    return result


def _encode_mappingproxy(value: object, state: _Encoding) -> PlainTree:
    """Encode a ``MappingProxyType`` as ``py/mappingproxy`` around its dict."""
    copied = dict(cast("Mapping[object, object]", value))
    return state.make_dict({"py/mappingproxy": _encode_dict(copied, state)})


def _decode_mappingproxy(
    data: PlainTree,
    target: object,
    at: FieldPath,
    state: _Decoding,
) -> MappingProxyType[Hashable, object]:
    """Decode a ``MappingProxyType`` around a decoded dict."""
    return MappingProxyType(_decode_dict(data, target, at, state))


def _bad_key(key: Hashable, target: object) -> Hashable:
    """Raise for a non-str key that does not fit ``target``."""
    raise ReadError(f"cannot read key {key!r} as {target}")


def _encode_key(key: object, state: _Encoding) -> str:
    """Return a dict key, ``json://``-escaping non-strings and tag lookalikes."""
    if isinstance(key, str) and not key.startswith(("py/", "json://")):
        return key
    return "json://" + json.dumps(mutable(_encode(key, state)))


# A YAML reader yields non-str keys such as ``True``; they read as is.
def _decode_key(
    key: Hashable,
    target: object,
    at: FieldPath,
    state: _Decoding,
) -> Hashable:
    """Return a dict key as ``target``, unescaping a ``json://`` key."""
    if not isinstance(key, str):
        return key if _fits(key, target) else _bad_key(key, target)
    raw: PlainTree = key
    if key.startswith("json://"):
        try:
            raw = cast(
                "PlainTree",
                json.loads(key.removeprefix("json://")),
            )
        except json.JSONDecodeError as error:
            raise ReadError(f"invalid escaped key {key!r}") from error
    decoded = _decode(raw, target, at, state)
    if not isinstance(decoded, Hashable):
        raise ReadError(f"key {decoded!r} is unhashable")
    return decoded


def _encode_type(value: object, state: _Encoding) -> PlainTree:
    """Encode a class as ``py/type``."""
    return state.make_dict({"py/type": _import_path(cast("type", value))})


def _decode_type(
    data: PlainTree,
    target: object,
    at: FieldPath,
    state: _Decoding,
) -> object:
    """Decode a ``py/type`` path to the class or module global it names."""
    del at
    if not isinstance(data, str):
        raise ReadError(f"cannot read {data!r} as a class path")
    resolved = _resolve(data, state)
    if (get_origin(target) or target) is type and not isinstance(resolved, type):
        raise ReadError(f"{data!r} does not name a class")
    return resolved


def _encode_function(value: object, state: _Encoding) -> PlainTree:
    """Encode a module-level function as ``py/function``."""
    if not (isinstance(getattr(value, "__self__", None), (ModuleType, type(None)))):
        return _encode_reduce(value, state)
    path = _import_path(cast("Callable[..., object]", value))
    return state.make_dict({"py/function": path})


def _decode_function(
    data: PlainTree,
    target: object,
    at: FieldPath,
    state: _Decoding,
) -> Callable[..., object]:
    """Decode a ``py/function`` path to the function it names."""
    del target, at
    if not isinstance(data, str):
        raise ReadError(f"cannot read {data!r} as a function path")
    resolved = _resolve(data, state)
    if not callable(resolved):
        raise ReadError(f"{data!r} does not name a callable")
    return resolved


def _encode_object(value: object, state: _Encoding) -> PlainTree:
    """Encode a dataclass or slotted object as ``py/object`` with its state."""
    _register(value, state)
    payload: dict[str, PlainTree] = {"py/object": _import_path(type(value))}
    for name in _attribute_names(value):
        member: object = getattr(value, name, _PENDING)
        if member is not _PENDING:
            payload[name] = _encode(member, state)
    return state.make_dict(payload)


def _decode_typeddict(
    data: PlainTree,
    target: object,
    at: FieldPath,
    state: _Decoding,
) -> dict[str, object]:
    """Decode a ``TypedDict``: declared keys by their types, unknown ones dropped."""
    kind = cast("type", target)
    if not isinstance(data, Mapping):
        raise ReadError(f"expected object for {kind.__name__}, got {data!r}")
    hints = _field_types(kind)
    missing = sorted(_required_keys(kind) - set(data))
    if missing:
        raise ReadError(f"{kind.__name__}: missing required field(s) {missing}")
    return {
        key: _decode_part(raw, hints[key], (*at, key), state)
        for key, raw in data.items()
        if key in hints
    }


# Under ``from __future__ import annotations`` a class body's ``NotRequired``
# is a string, so ``__required_keys__`` lists every key; the resolved hints
# carry the qualifier.
def _required_keys(kind: type) -> frozenset[str]:
    """Return a TypedDict's required keys, read from its resolved hints."""
    total = cast("bool", getattr(kind, "__total__", True))
    hints = get_type_hints(kind, include_extras=True)
    return frozenset(
        key
        for key, hint in hints.items()
        if get_origin(hint) is not NotRequired
        and (total or get_origin(hint) is Required)
    )


def _decode_object(
    data: PlainTree,
    target: object,
    at: FieldPath,
    state: _Decoding,
) -> object:
    """Decode an object: the ``py/object`` class, else ``target``'s dataclass."""
    name = target.__name__ if isinstance(target, type) else str(target)
    if not isinstance(data, Mapping):
        raise ReadError(f"expected object for {name}, got {data!r}")
    if _names_dataclass(data, target, state):
        fields_only: dict[str, PlainTree] = {
            key: raw for key, raw in data.items() if key != "py/object"
        }
        data = fields_only
    if "py/object" in data:
        path = data["py/object"]
        if not isinstance(path, str):
            raise ReadError(f"invalid py/object path: {path!r}")
        kind = _resolve(path, state)
        if not isinstance(kind, type):
            raise ReadError(f"{path!r} does not name a class")
        value = _allocate(kind)
        state.built.append(value)
        for attribute, raw in data.items():
            if attribute != "py/object":
                member = _decode_part(raw, object, (*at, attribute), state)
                object.__setattr__(value, attribute, member)
        return value
    if not isinstance(target, type) or not is_dataclass(target):
        raise ReadError(f"cannot read {dict(data)!r} as {name}")
    types = _field_types(target)
    unknown = sorted(set(data) - set(types))
    if unknown:
        raise ReadError(
            f"{name}: unknown field(s) {unknown}; valid: {sorted(types)}",
        )
    index = len(state.built)
    state.built.append(_PENDING)
    before = len(state.bad)
    kwargs = {
        key: _decode_part(raw, types[key], (*at, key), state)
        for key, raw in data.items()
    }
    if len(state.bad) > before:
        value = _allocate(target)
        for key, member in kwargs.items():
            object.__setattr__(value, key, member)
    else:
        try:
            value = target(**kwargs)
        except (TypeError, ValueError) as error:
            raise ReadError(f"cannot build {name}: {error}") from error
    state.built[index] = value
    return value


# It does when the tag names ``target``, or when imports are off: the tag's class cannot
# be built then, and data written before a class moved still names its old path.
def _names_dataclass(
    data: Mapping[str, PlainTree],
    target: object,
    state: _Decoding,
) -> bool:
    """Return whether dataclass ``target`` reads ``data`` with its own fields."""
    return (
        isinstance(target, type)
        and is_dataclass(target)
        and (
            not state.allow_imports
            or data.get("py/object") == f"{target.__module__}.{target.__qualname__}"
        )
    )


def _encode_inline(value: object, state: _Encoding) -> PlainTree:
    """Encode a value owning the ``py/inline`` deferred-call protocol."""
    func, args, kwargs = cast("_Inline", value).__custom_json_inline__()
    _register(value, state)
    recipe = state.make_dict(
        {
            "func": _encode(func, state),
            "args": state.make_list([_encode(item, state) for item in args]),
            "kwargs": state.make_dict(
                {key: _encode(item, state) for key, item in kwargs.items()},
            ),
        },
    )
    path = _import_path(type(value))
    return state.make_dict({"py/inline": state.make_list([path, recipe])})


def _decode_inline(
    data: PlainTree,
    target: object,
    at: FieldPath,
    state: _Decoding,
) -> object:
    """Allocate a ``py/inline`` value, then hand its recipe back to it."""
    del target
    path, recipe = _pair(data, "py/inline")
    if (
        not isinstance(path, str)
        or not isinstance(recipe, Mapping)
        or set(recipe) != {"func", "args", "kwargs"}
    ):
        raise ReadError(f"invalid py/inline payload: {data!r}")
    args, kwargs = recipe["args"], recipe["kwargs"]
    if (
        not isinstance(args, Sequence)
        or isinstance(args, str)
        or not isinstance(kwargs, Mapping)
    ):
        raise ReadError(f"invalid py/inline payload: {data!r}")
    kind = _resolve(path, state)
    if not isinstance(kind, type) or not _owns_inline(kind):
        raise ReadError(f"{path!r} does not own the py/inline protocol")
    value = cast("_Inline", _allocate(kind))
    state.built.append(value)
    value.__custom_json_inline_init__(
        _decode_part(recipe["func"], object, (*at, "func"), state),
        [
            _decode_part(raw, object, (*at, "args", index), state)
            for index, raw in enumerate(args)
        ],
        {
            key: _decode_part(raw, object, (*at, "kwargs", key), state)
            for key, raw in kwargs.items()
        },
    )
    return value


def _encode_hooked(value: object, state: _Encoding) -> PlainTree:
    """Encode a value through its caller-supplied hook as ``py/hook``."""
    encode_hook = state.hooks[type(value)][0]
    _register(value, state)
    # The payload is arbitrary caller data, so it takes the same walk as any
    # other value: it may hold non-finite floats or tag-like keys.
    path = _import_path(type(value))
    payload = _encode(encode_hook(value), state)
    return state.make_dict({"py/hook": state.make_list([path, payload])})


def _decode_hooked(
    data: PlainTree,
    target: object,
    at: FieldPath,
    state: _Decoding,
) -> object:
    """Rebuild a ``py/hook`` value through its registered decode callback."""
    del target
    path, payload = _pair(data, "py/hook")
    hook = next(
        (
            pair
            for kind, pair in state.hooks.items()
            if f"{kind.__module__}.{kind.__qualname__}" == path
        ),
        None,
    )
    if hook is None:
        raise ReadError(f"hook {path!r} is not registered")
    # The encoder numbers the hooked value before its payload, so the slot is
    # reserved in that order and filled once the hook rebuilds it.
    index = len(state.built)
    state.built.append(_PENDING)
    value = hook[1](_decode(payload, object, (*at, 1), state))
    state.built[index] = value
    return value


def _encode_reduce(value: object, state: _Encoding) -> PlainTree:
    """Encode a value by its pickle reduce recipe as ``py/reduce``."""
    recipe = _canonical_reduce(value)
    if recipe is None:
        return _encode_unreduced(value, state)
    # A bare-name reduce names a module global, written as an import path.
    if isinstance(recipe, str):
        path = _verified_path(f"{type(value).__module__}.{recipe}", value)
        return state.make_dict({"py/type": path})
    func, args, *rest = recipe
    mark = len(state.alive)
    try:
        if any(part is not None for part in rest):
            _register(value, state)
        elements = [
            _encode(func, state),
            _encode(args, state),
            *(_encode(part, state) for part in rest),
        ]
    except TypeError:
        _rollback(state, mark)
        return _encode_unreduced(value, state)
    while len(elements) > 2 and elements[-1] is None:
        elements.pop()
    return state.make_dict({"py/reduce": state.make_list(elements)})


def _encode_unreduced(value: object, state: _Encoding) -> PlainTree:
    """Encode a value with no usable reduce recipe by the protocols it has."""
    kind = type(value)
    name = kind.__name__
    if hasattr(kind, "__dataclass_fields__") or "__slots__" in kind.__dict__:
        return _encode_object(value, state)
    if issubclass(kind, Mapping):
        return _encode_dict(value, state)
    if issubclass(kind, Sequence) and not issubclass(kind, (str, bytes)):
        return _encode_list(value, state)
    if issubclass(kind, AbstractSet):
        return _encode_set(value, state)
    raise TypeError(
        f"Cannot serialize leaf of type {name!r}. "
        f"Pass hooks={{{name}: (encode, decode)}} to to_plain().",
    )


def _decode_reduce(
    data: PlainTree,
    target: object,
    at: FieldPath,
    state: _Decoding,
) -> object:
    """Replay a ``py/reduce`` recipe."""
    del target
    if not isinstance(data, Sequence) or isinstance(data, str):
        raise ReadError(f"invalid py/reduce payload: {data!r}")
    if len(data) < 2 or len(data) > 5:
        raise ReadError("py/reduce requires two to five elements")
    stateful = any(part is not None for part in data[2:])
    index = len(state.built)
    if stateful:
        state.built.append(_PENDING)
    func = _decode(data[0], object, (*at, 0), state)
    args = _decode(data[1], object, (*at, 1), state)
    if not callable(func) or not isinstance(args, tuple):
        raise ReadError("py/reduce needs a callable and an argument tuple")
    try:
        value = func(*cast("tuple[object, ...]", args))
    except (TypeError, ValueError) as error:
        raise ReadError(f"py/reduce call failed: {error}") from error
    if stateful:
        state.built[index] = value
    if len(data) > 2 and data[2] is not None:
        _apply_state(value, _decode(data[2], object, (*at, 2), state))
    if len(data) > 3 and data[3] is not None:
        items = _decode(data[3], object, (*at, 3), state)
        extend: object = getattr(value, "extend", None)
        if not callable(extend) or not isinstance(items, list):
            raise ReadError(f"reduce target cannot take list items: {value!r}")
        extend(items)
    if len(data) > 4 and data[4] is not None:
        pairs = _decode(data[4], object, (*at, 4), state)
        setitem: object = getattr(value, "__setitem__", None)
        if not callable(setitem) or not isinstance(pairs, list):
            raise ReadError(f"reduce target cannot take dict items: {value!r}")
        for pair in cast("list[object]", pairs):
            entry = cast("tuple[object, ...]", pair) if isinstance(pair, tuple) else ()
            if len(entry) != 2:
                raise ReadError(f"invalid reduce dict item: {pair!r}")
            key, member = entry
            setitem(key, member)
    return value


def _apply_state(value: object, attributes: object) -> None:
    """Apply reduce state through ``__setstate__`` or direct attributes."""
    setstate: object = getattr(value, "__setstate__", None)
    if callable(setstate):
        setstate(attributes)
        return
    chunks: tuple[object, ...] = (attributes, None)
    if isinstance(attributes, tuple):
        pair = cast("tuple[object, ...]", attributes)
        if len(pair) == 2:
            chunks = pair
    for chunk in chunks:
        if isinstance(chunk, dict):
            for key, member in cast("dict[str, object]", chunk).items():
                object.__setattr__(value, key, member)


def _decode_any(
    data: PlainTree,
    target: object,
    at: FieldPath,
    state: _Decoding,
) -> object:
    """Decode untagged data for ``object``, ``Any``, or a Protocol, as is."""
    del target
    # Callers edit a read record's untyped members in place, so tag-free data
    # must come back as the very objects passed in.
    if state.untagged or data is None or isinstance(data, (str, int, float)):
        return data
    if isinstance(data, Mapping):
        return _decode_dict(data, object, at, state)
    return _decode_list(data, object, at, state)


def _decode_union(
    data: PlainTree,
    target: object,
    at: FieldPath,
    state: _Decoding,
) -> object:
    """Decode the member the data's kind selects, else the first that fits."""
    members = cast("tuple[object, ...]", get_args(target))
    tagged = isinstance(data, Mapping) and _has_tag_key(data)
    if tagged:
        # The member a ``py/object`` tag names goes first, so member order
        # cannot pick a sibling whose fields happen to fit too.
        path = cast("Mapping[str, PlainTree]", data).get("py/object")
        members = tuple(
            sorted(members, key=lambda member: _import_name(member) != path),
        )
    for member in members if tagged else _union_candidates(data, members):
        built, bad = len(state.built), set(state.bad)
        try:
            value = _decode(data, member, at, state)
        except ReadError:
            value = _PENDING
        if value is not _PENDING and len(state.bad) == len(bad):
            return value
        del state.built[built:]
        for key in set(state.bad) - bad:
            del state.bad[key]
    raise ReadError(f"cannot read {data!r} as {target}")


def _import_name(member: object) -> str | None:
    """Return a class member's ``module.qualname``, else ``None``."""
    if isinstance(member, type):
        return f"{member.__module__}.{member.__qualname__}"
    return None


def _union_candidates(
    data: PlainTree,
    members: tuple[object, ...],
) -> tuple[object, ...]:
    """Return the union members to try: the data's own kind first."""
    if data is None or isinstance(data, (str, int, float)):
        native: type = type(data)
    elif isinstance(data, Mapping):
        native = dict
    else:
        native = list
    origins = _NATIVE_ORIGINS[native]
    exact = tuple(
        member
        for member in members
        if (get_origin(_resolve_alias(member)) or _resolve_alias(member)) in origins
    )
    return exact + tuple(member for member in members if member not in exact)


def _element_type(target: object) -> object:
    """Return a one-parameter container's element type, else ``object``."""
    args = cast("tuple[object, ...]", get_args(target))
    return args[0] if len(args) == 1 else object


def _pair(data: PlainTree, tag: str) -> tuple[PlainTree, ...]:
    """Return a two-element tag payload's parts, naming the tag when malformed."""
    if not isinstance(data, Sequence) or isinstance(data, str) or len(data) != 2:
        raise ReadError(f"invalid {tag} payload: {data!r}")
    return tuple(cast("Sequence[PlainTree]", data))


def _is_class(value: object) -> TypeGuard[type[object]]:
    """Return whether ``value`` is a class."""
    return isinstance(value, type)


def _allocate(kind: type) -> object:
    """Allocate ``kind`` without running ``__init__``."""
    allocate = cast("Callable[[type], object]", kind.__new__)
    return allocate(kind)


# A path's arguments are restated as one joined string. CPython 3.12 reduces
# ``PurePath`` to one argument per segment and 3.14 to a single string, so the
# raw recipe would differ per interpreter; every version rebuilds from the join.
def _canonical_reduce(
    value: object,
) -> tuple[Callable[..., object], tuple[object, ...], *tuple[object, ...]] | str | None:
    """Return ``value``'s reduce recipe with interpreter-independent arguments."""
    try:
        reduced = cast("object", value.__reduce_ex__(2))
    except (AttributeError, NotImplementedError, TypeError, ValueError):
        return None
    if isinstance(reduced, str):
        return reduced
    if not isinstance(reduced, tuple):
        return None
    parts = list(cast("tuple[object, ...]", reduced))
    if len(parts) < 2 or len(parts) > 5:
        return None
    func, args = parts[0], parts[1]
    if not callable(func) or not isinstance(args, tuple):
        return None
    arguments = (
        (str(value),) if isinstance(value, PurePath) else cast("tuple[object]", args)
    )
    rest = [
        list(cast("Iterable[object]", part))
        if index in (1, 2) and part is not None
        else part
        for index, part in enumerate(parts[2:])
    ]
    # A partial's state ends with its instance ``__dict__``, which CPython creates,
    # empty, the first time anything reads the attribute; empty rebuilds as none.
    if isinstance(value, partial) and rest and isinstance(rest[0], tuple):
        *fields, instance_dict = cast("tuple[object, ...]", rest[0])
        if instance_dict == {}:
            rest[0] = (*fields, None)
    return (func, arguments, *rest)


def _owns_inline(kind: type) -> bool:
    """Return whether ``kind`` implements both halves of ``py/inline``, cached."""
    cached = _OWNS_INLINE.get(kind)
    if cached is None:
        cached = callable(getattr(kind, "__custom_json_inline__", None)) and callable(
            getattr(kind, "__custom_json_inline_init__", None),
        )
        _OWNS_INLINE[kind] = cached
    return cached


def _import_path(value: type | Callable[..., object]) -> str:
    """Return the verified dotted import path of a class or function, cached."""
    cached = _IMPORT_PATHS.get(id(value))
    if cached is not None and cached[0] is value:
        return cached[1]
    module: object = getattr(value, "__module__", None)
    qualname: object = getattr(value, "__qualname__", None)
    if (
        not isinstance(module, str)
        or not isinstance(qualname, str)
        or "<locals>" in qualname
    ):
        raise TypeError(
            f"Cannot serialize {value!r}: it has no importable path "
            "(module-level __qualname__). Local/lambda callables and local "
            "classes/subclasses cannot be deserialized.",
        )
    path = _verified_path(f"{module}.{qualname}", value)
    _IMPORT_PATHS[id(value)] = (value, path)
    return path


def _verified_path(path: str, value: object) -> str:
    """Return ``path`` after proving it imports as ``value`` itself."""
    message = (
        f"Cannot serialize {value!r}: import path {path!r} does not resolve to "
        "the same object."
    )
    try:
        resolved = _resolve_import(path)
    except (AttributeError, ImportError) as error:
        raise TypeError(message) from error
    if resolved is not value:
        raise TypeError(message)
    return path


def _resolve(path: str, state: _Decoding) -> object:
    """Import the object ``path`` names, refusing unless imports are allowed."""
    if not state.allow_imports:
        raise ReadError(f"{path!r} needs allow_imports=True")
    resolved = _RESOLVED.get(path, _PENDING)
    if resolved is _PENDING:
        try:
            resolved = _resolve_import(path)
        except (AttributeError, ImportError) as error:
            raise ReadError(f"cannot import {path!r}: {error}") from error
        _RESOLVED[path] = resolved
    return resolved


def _resolve_import(path: str) -> object:
    """Import the longest importable prefix of ``path``, then walk the rest."""
    parts = path.split(".")
    for split in range(len(parts) - 1, 0, -1):
        module_name = ".".join(parts[:split])
        # A loaded module skips the import machinery, whose lock and finder
        # lookups cost more than the attribute walk below.
        module = sys.modules.get(module_name)
        if module is None:
            try:
                module = importlib.import_module(module_name)
            except ImportError:
                continue
        resolved: object = module
        for part in parts[split:]:
            resolved = getattr(resolved, part, _PENDING)
            if resolved is _PENDING:
                raise AttributeError(f"{path!r} has no attribute {part!r}")
        return resolved
    raise ImportError(f"Cannot resolve path: {path!r}")


def _resolve_alias(target: object) -> object:
    """Unwrap PEP 695 alias chains and ``Annotated`` to the underlying type."""
    resolved = target
    seen: set[int] = set()
    while (identity := id(resolved)) not in seen:
        seen.add(identity)
        if get_origin(resolved) is Annotated:
            resolved = cast("tuple[object, ...]", get_args(resolved))[0]
            continue
        value: object = getattr(resolved, "__value__", None)
        if value is None:
            break
        resolved = value
    return resolved


def _slot_names(kind: type) -> tuple[str, ...]:
    """Return ``kind``'s state slots in MRO order, cached."""
    cached = _SLOT_NAMES.get(kind)
    if cached is not None:
        return cached
    names: dict[str, None] = {}
    for base in kind.__mro__:
        slots: object = getattr(base, "__slots__", ())
        for slot in (
            (slots,) if isinstance(slots, str) else cast("Iterable[str]", slots)
        ):
            if slot not in _SKIPPED_ATTRIBUTES:
                names.setdefault(slot)
    result = tuple(names)
    _SLOT_NAMES[kind] = result
    return result


def _attribute_names(value: object) -> Iterator[str]:
    """Yield state attributes: slots, then sorted ``__dict__`` keys."""
    slots = _slot_names(type(value))
    yield from slots
    if hasattr(value, "__dict__"):
        seen = set(slots)
        for key in sorted(vars(value)):
            if key not in seen and key not in _SKIPPED_ATTRIBUTES:
                seen.add(key)
                yield key


# A cached value naming its own class would keep the weak key alive forever, so
# only module-level classes whose hints do not name themselves are cached.
def _field_types(kind: type) -> Mapping[str, object]:
    """Return a dataclass's init-field or a TypedDict's key types, cached."""
    cached = _FIELD_TYPES.get(kind)
    if cached is not None:
        return cached
    module_name: str = kind.__module__
    qualname: str = kind.__qualname__
    try:
        hints: dict[str, object] = get_type_hints(kind)
    except (AttributeError, NameError, TypeError) as error:
        raise ReadError(f"cannot resolve field types of {kind}: {error}") from error
    result = MappingProxyType(
        {key: _resolve_alias(hint) for key, hint in hints.items()}
        if is_typeddict(kind)
        else {
            item.name: hints.get(item.name, object)
            for item in (fields(kind) if is_dataclass(kind) else ())
            if item.init
        },
    )
    module = sys.modules.get(module_name)
    if (
        module is not None
        and vars(module).get(qualname) is kind
        and not any(
            hint is kind or kind in cast("tuple[object, ...]", get_args(hint))
            for hint in result.values()
        )
    ):
        _FIELD_TYPES[kind] = result
    return result


_TAGS: Final = (
    "py/type",
    "py/function",
    "py/tuple",
    "py/set",
    "py/frozenset",
    "py/mappingproxy",
    "py/b64",
    "py/float",
    "py/complex",
    "py/path",
    "py/uuid",
    "py/datetime",
    "py/reduce",
    "py/hook",
    "py/inline",
    "py/object",
)
"""Every tag, in the precedence a node carrying several resolves by."""


_NATIVE_LEAVES: Final = frozenset((str, int, bool))


_LAX_BOOLS: Final[Mapping[str, int]] = MappingProxyType(
    {"true": 1, "false": 0, "1": 1, "0": 0},
)
"""Text a lax read takes as a bool, by its lowercased spelling."""


_CONCRETE_PATH: Final = type(Path())


_SKIPPED_ATTRIBUTES: Final = frozenset(("__weakref__", "__dict__", "_finalized"))


_PENDING: Final = object()
"""Fills a ``py/id`` slot reserved for a value built after its children."""


_NATIVE_ORIGINS: Final[Mapping[type, tuple[object, ...]]] = MappingProxyType(
    {
        type(None): (type(None),),
        bool: (bool,),
        int: (int,),
        float: (float,),
        str: (str,),
        dict: (dict, Mapping, MutableMapping),
        list: (list, Sequence, MutableSequence),
    },
)
"""The union-member origins each plain kind reads as without conversion."""


_ENCODERS: Final[Mapping[type, _Encoder]] = MappingProxyType(
    {
        float: _encode_float,
        complex: _encode_complex,
        bytes: _encode_bytes,
        list: _encode_list,
        tuple: _encode_tuple,
        set: _encode_set,
        frozenset: _encode_frozenset,
        dict: _encode_dict,
        MappingProxyType: _encode_mappingproxy,
        FunctionType: _encode_function,
        BuiltinFunctionType: _encode_function,
    },
)
"""Encoders by exact type; ``_encoder_for`` resolves and caches every other class."""


_DECODERS: Final[Mapping[object, _Decoder]] = MappingProxyType(
    {
        object: _decode_any,
        type(None): _decode_none,
        bool: _decode_bool,
        int: _decode_int,
        float: _decode_float,
        complex: _decode_complex,
        str: _decode_str,
        bytes: _decode_bytes,
        Path: _decode_path,
        UUID: _decode_uuid,
        datetime: _decode_datetime,
        Literal: _decode_literal,
        list: _decode_list,
        Sequence: _decode_list,
        MutableSequence: _decode_list,
        tuple: _decode_tuple,
        set: _decode_set,
        MutableSet: _decode_set,
        frozenset: _decode_frozenset,
        AbstractSet: _decode_frozenset,
        dict: _decode_dict,
        Mapping: _decode_dict,
        MutableMapping: _decode_dict,
        MappingProxyType: _decode_mappingproxy,
        type: _decode_type,
        UnionType: _decode_union,
        # Before 3.14, ``Optional``/``Union`` have this origin, not ``UnionType``.
        typing.Union: _decode_union,  # pyright: ignore[reportDeprecated] -- A runtime origin key, not an annotation.
    },
)
"""Decoders by origin; ``_decoder_for`` adds Enum, dataclass, and Protocol targets."""


_TAG_DECODERS: Final[Mapping[str, _Decoder]] = MappingProxyType(
    {
        "py/float": _decode_float,
        "py/complex": _decode_complex,
        "py/b64": _decode_bytes,
        "py/path": _decode_path,
        "py/uuid": _decode_uuid,
        "py/datetime": _decode_datetime,
        "py/tuple": _decode_tuple,
        "py/set": _decode_set,
        "py/frozenset": _decode_frozenset,
        "py/mappingproxy": _decode_mappingproxy,
        "py/type": _decode_type,
        "py/function": _decode_function,
        "py/object": _decode_object,
        "py/inline": _decode_inline,
        "py/hook": _decode_hooked,
        "py/reduce": _decode_reduce,
    },
)
"""The decoder each tag selects; the same decoder untagged data reaches by type."""


_ENCODER_CACHE: Final[weakref.WeakKeyDictionary[type, _Encoder]] = (
    weakref.WeakKeyDictionary()
)


# Keyed by ``id`` rather than weakly: a builtin function such as ``max`` cannot
# be weakly referenced. Each entry keeps its object, so the ``id`` is never
# reused while the entry stands.
_IMPORT_PATHS: Final[dict[int, tuple[object, str]]] = {}


_RESOLVED: Final[dict[str, object]] = {}


_SLOT_NAMES: Final[weakref.WeakKeyDictionary[type, tuple[str, ...]]] = (
    weakref.WeakKeyDictionary()
)


_FIELD_TYPES: Final[weakref.WeakKeyDictionary[type, Mapping[str, object]]] = (
    weakref.WeakKeyDictionary()
)


_OWNS_INLINE: Final[weakref.WeakKeyDictionary[type, bool]] = weakref.WeakKeyDictionary()
