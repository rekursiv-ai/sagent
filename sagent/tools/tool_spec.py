"""``--tool NAME.key=value`` override parsing.

The CLI spelling is a literal transcription of the constructor call:
``--tool Bash.output=off`` is ``Bash(output="off")``. There is no
registry of permitted keys -- the tool's own ``__init__`` signature is
the contract, so a key it does not accept aborts with the keys it does.

Only the syntax is parsed here. Meaning, defaults, and validity all live
on the tool, which is the only thing that knows them.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import (
    Annotated,
    Literal,
    TypeAliasType,
    get_args,
    get_origin,
    get_type_hints,
)

import inspect


__all__ = ["CLI_SETTABLE", "ToolSpecError", "coerce_kwargs", "parse_tool_overrides"]

CLI_SETTABLE = "cli-settable"
"""Marker opting a constructor parameter into ``--tool NAME.key=value``.

Applied as ``Annotated[T, CLI_SETTABLE]``. Opt-IN rather than inferred
from the type: ``Slack.token`` is a ``str`` like any display knob, and a
rule based on coercibility would offer it on the command line.
"""


class ToolSpecError(ValueError):
    """A ``--tool`` override was malformed, unknown, or out of range."""


def parse_tool_overrides(specs: Iterable[str]) -> dict[str, dict[str, str]]:
    """Group ``NAME.key=value`` strings by tool name.

    Splits on the FIRST ``=`` only, so a value may contain ``=`` or
    ``,`` without quoting -- the property that makes the repeated-flag
    form preferable to a comma-separated one.

    Args:
      specs: Raw ``--tool`` values, e.g. ``["Bash.output=off"]``.

    Returns:
      overrides: ``{tool_name: {key: raw_value}}``, later flags winning.

    Raises:
      ToolSpecError: A spec is missing its ``.`` or its ``=``, or either
          side is empty.

    """
    out: dict[str, dict[str, str]] = {}
    for spec in specs:
        path, sep, value = spec.partition("=")
        if not sep:
            raise ToolSpecError(
                f"--tool {spec!r}: expected NAME.key=value (missing '=')"
            )
        name, dot, key = path.rpartition(".")
        if not dot or not name or not key:
            raise ToolSpecError(
                f"--tool {spec!r}: expected NAME.key=value (missing '.')"
            )
        out.setdefault(name, {})[key] = value
    return out


def coerce_kwargs(cls: type, overrides: Mapping[str, str]) -> dict[str, object]:
    """Convert raw string overrides to constructor kwargs for ``cls``.

    The annotation drives the conversion: ``Literal[...]`` checks
    membership, ``int`` / ``float`` / ``bool`` parse, ``str`` passes
    through. An annotation outside that set is rejected rather than
    guessed at, so a tool cannot silently accept an unparseable knob.

    Args:
      cls: Tool class whose ``__init__`` defines the settable keys.
      overrides: Raw ``{key: value}`` strings from the command line.

    Returns:
      kwargs: Converted values ready to splat into ``cls(**kwargs)``.

    Raises:
      ToolSpecError: An unknown key, an unconvertible value, or a key
          whose annotation this function cannot coerce.

    """
    try:
        # ``include_extras`` keeps the ``Annotated`` wrapper carrying
        # ``CLI_SETTABLE``; without it the marker is stripped and nothing
        # reads as settable.
        hints = get_type_hints(cls.__init__, include_extras=True)
    except (NameError, TypeError) as exc:  # unresolvable forward ref
        raise ToolSpecError(f"{cls.__name__}: cannot inspect settings: {exc}") from exc
    params = inspect.signature(cls.__init__).parameters
    # Keyword-only AND coercible from a string. A parameter typed as an
    # object graph or a secret (``peers``, ``tools``, ``token``) is not a
    # command-line knob, so it must not appear in the settable list -- an
    # error message that offers ``token`` invites putting one on argv.
    settable = [
        name
        for name, p in params.items()
        if name != "self"
        and p.kind is p.KEYWORD_ONLY
        and _is_cli_settable(hints.get(name))
    ]
    kwargs: dict[str, object] = {}
    for key, raw in overrides.items():
        if key not in settable:
            available = ", ".join(settable) or "(none)"
            raise ToolSpecError(
                f"{cls.__name__}: unknown setting {key!r}. Settable: {available}"
            )
        kwargs[key] = _coerce(cls.__name__, key, _unwrap(hints[key]), raw)
    return kwargs


def _is_cli_settable(annotation: object) -> bool:
    """Whether a parameter opted into command-line configuration."""
    return (
        get_origin(annotation) is Annotated and CLI_SETTABLE in get_args(annotation)[1:]
    )


def _unwrap(annotation: object) -> object:
    """Resolve wrappers until the coercible type is reached.

    Two wrappers hide it. ``Annotated[T, ...]`` carries the
    ``CLI_SETTABLE`` marker, and a PEP 695 ``type X = Literal[...]``
    binds a ``TypeAliasType`` whose ``get_origin`` is ``None`` -- so an
    unresolved alias reads as an uncoercible annotation and every knob
    declared through one is rejected.
    """
    if get_origin(annotation) is Annotated:
        return _unwrap(get_args(annotation)[0])
    if isinstance(annotation, TypeAliasType):
        return _unwrap(annotation.__value__)
    return annotation


def _coerce(tool: str, key: str, annotation: object, raw: str) -> object:
    """Convert one raw value per its annotation."""
    if get_origin(annotation) is Literal:
        choices = get_args(annotation)
        if raw not in choices:
            valid = ", ".join(str(c) for c in choices)
            raise ToolSpecError(f"{tool}.{key}: invalid value {raw!r}. Valid: {valid}")
        return raw
    if annotation is bool:
        lowered = raw.strip().lower()
        if lowered in ("1", "true", "yes", "on"):
            return True
        if lowered in ("0", "false", "no", "off"):
            return False
        raise ToolSpecError(f"{tool}.{key}: expected a boolean, got {raw!r}")
    if annotation is int or annotation is float:
        caster: Callable[[str], object] = int if annotation is int else float
        try:
            return caster(raw)
        except ValueError as exc:
            kind = "int" if annotation is int else "float"
            raise ToolSpecError(f"{tool}.{key}: expected {kind}, got {raw!r}") from exc
    if annotation is str:
        return raw
    raise ToolSpecError(
        f"{tool}.{key}: setting is not settable from the command line"
        f" (annotation {annotation!r})"
    )
