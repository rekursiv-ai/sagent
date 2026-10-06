"""Which model a name means, and what it can do over one transport.

A vendor catalog is a :class:`ModelTable`: its rows, newest-first within
each family, plus a few roles (``default``, ``utility``, ``best``). Its keys
are the exact names, each ``family-major.minor`` (``opus-5.0``, never
``opus-5``). One name reaches a row four ways, in this order:

1. its exact name (``opus-5.5``);
2. its vendor wire id (``claude-opus-5-5``);
3. a role, which names a family or an exact name (``default`` -> ``opus``);
4. a family, alone or with a major, which means its newest row
   (``opus`` and ``opus-5`` -> ``opus-5.5``; ``opus-4`` -> ``opus-4.8``).
   The family is everything before the last ``-``, so ``gemini-flash``
   never reaches ``gemini-flash-lite-3.5``.

The order is the precedence: a vendor id that is also a family's spelling
(``gpt-4``, ``qwen-plus``) names its own row, since a server reporting it
back means that model and billing it at the family's newest would misprice.

Because exact names always carry a minor version, a major alone is never an
exact name: ``sonnet-5`` is the newest ``sonnet-5.x``. Because rows are
newest-first, adding a release moves every prefix and every role that points
at its family, with no alias to edit.

:class:`ModelCatalog` meets a table with one transport and resolves a
possibly context-tagged name (``opus+1m``) to a capability and the settings
that name selected.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Final, cast, get_args, override

from sagent.types.capability import (
    ContextTag,
    ModelCapability,
    ModelSettings,
)


__all__ = [
    "CONTEXT_TAGS",
    "ModelCatalog",
    "ModelTable",
    "UnknownModelError",
    "UnsupportedTagError",
    "base_model_id",
    "split_model_id",
]


# Derived from ``ContextTag``, not restated: a tag the type admits but a
# hand-written tuple omitted would be unparseable while type-checking clean.
# The default window's ``""`` is filtered out -- no id carries it.
CONTEXT_TAGS: Final[tuple[ContextTag, ...]] = tuple(
    t
    for t in cast(tuple[ContextTag, ...], get_args(cast(object, ContextTag.__value__)))
    if t
)
"""Window-size suffixes a model id may carry (e.g. ``...+1m``)."""


class UnknownModelError(ValueError):
    """No row answers to the name."""


class UnsupportedTagError(ValueError):
    """The name carries a context tag the model does not offer."""


def split_model_id(model_id: str) -> tuple[str, frozenset[ContextTag]]:
    """Split a model id into its base id and trailing context tags.

    Tags may appear in any order; matching is case-insensitive, and an
    unknown suffix stays part of the base id.

    Args:
      model_id: Model id, possibly with trailing context tags.

    Returns:
      base_id: ``model_id`` without its tags.
      tags: The stripped tags, lowercased (e.g. ``{"+1m"}``).

    """
    tags: set[ContextTag] = set()
    base = model_id
    while True:
        lower = base.lower()
        tag: ContextTag | None = next(
            (t for t in CONTEXT_TAGS if lower.endswith(t)),
            None,
        )
        if tag is None:
            return base, frozenset(tags)
        tags.add(tag)
        base = base[: -len(tag)]


def base_model_id(model_id: str) -> str:
    """Strip trailing context tags, yielding the canonical model id."""
    return split_model_id(model_id)[0]


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelTable(Mapping[str, ModelCapability]):
    """A vendor's rows, keyed by exact name and reachable by wire id, role, or prefix.

    Iterating yields exact names only: a role or prefix is a way to name a
    row, not another model.
    """

    rows: tuple[ModelCapability, ...]
    """Every row, newest-first within each family."""

    roles: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))
    """Role name to the family or id it means (``{"default": "opus"}``)."""

    _by_id: Mapping[str, ModelCapability] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Index rows and check every role names a row."""
        by_id: dict[str, ModelCapability] = {}
        for row in self.rows:
            if row.model_id in by_id:
                raise ValueError(f"duplicate model id {row.model_id!r}")
            by_id[row.model_id] = row
        object.__setattr__(self, "_by_id", MappingProxyType(by_id))
        object.__setattr__(self, "roles", MappingProxyType(dict(self.roles)))
        for role, target in self.roles.items():
            if self._named(target) is None:
                raise ValueError(f"role {role!r} names no model: {target!r}")

    @override
    def __getitem__(self, key: str) -> ModelCapability:
        row = self._resolved(key)
        if row is None:
            raise KeyError(key)
        return row

    @override
    def __iter__(self) -> Iterator[str]:
        return iter(self._by_id)

    @override
    def __len__(self) -> int:
        return len(self._by_id)

    @override
    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and self._resolved(key) is not None

    def exact(self, name: str) -> ModelCapability | None:
        """Return the row whose exact name or wire id is ``name``, never a guess.

        For ids a server reports back: billing ``gpt-5`` at the newest
        ``gpt-5.x`` because the prefix matched would misprice it silently.

        Args:
          name: Exact catalog name or vendor wire id.

        Returns:
          row: The row, or ``None`` when no row carries that id.

        """
        row = self._by_id.get(name)
        if row is not None:
            return row
        return next((row for row in self.rows if row.wire_model_id == name), None)

    def _resolved(self, key: str) -> ModelCapability | None:
        """Return the row ``key`` names, in the module's precedence order."""
        row = self.exact(key)
        return row if row is not None else self._named(self.roles.get(key, key))

    # A prefix names a whole family, optionally with a major: ``opus`` and
    # ``opus-5`` name ``opus-*``; ``opu`` names nothing, and ``gemini-flash``
    # never reaches ``gemini-flash-lite-*``, a different family.
    def _named(self, name: str) -> ModelCapability | None:
        """Return the row ``name`` means by exact name, wire id, or prefix."""
        row = self.exact(name)
        if row is not None:
            return row
        return next((row for row in self.rows if _in_family(row.model_id, name)), None)


def _in_family(model_id: str, name: str) -> bool:
    """Whether ``name`` is ``model_id``'s family, or its family plus a version prefix."""
    family, _, version = model_id.rpartition("-")
    if name == family:
        return True
    head, _, major = name.rpartition("-")
    return head == family and (version == major or version.startswith(f"{major}."))


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelCatalog:
    """A vendor table met with one transport."""

    models: ModelTable
    """The rows, roles, and families the transport serves."""

    transport: ModelCapability
    """What the transport allows; met with every row it resolves."""

    def resolve(self, model_id: str) -> tuple[ModelCapability, ModelSettings]:
        """Resolve a possibly context-tagged name against this transport.

        Args:
          model_id: Catalog id, wire id, role, or family, optionally tagged.

        Returns:
          capability: The row narrowed by the transport.
          settings: Settings selecting the requested context.

        Raises:
          UnknownModelError: No row answers to the name.
          UnsupportedTagError: Context tags conflict or are unavailable.

        """
        base, id_tags = split_model_id(model_id)
        tags: list[ContextTag] = sorted(id_tags)
        if len(tags) > 1:
            raise UnsupportedTagError(
                f"Model {model_id!r} selects conflicting contexts: {', '.join(tags)}",
            )
        context: ContextTag = tags[0] if tags else ""
        row = self.models.get(base)
        if row is None:
            known = ", ".join([*self.models, *self.models.roles])
            raise UnknownModelError(
                f"Unknown model {model_id!r}. Known models: {known}",
            )
        capability = row & self.transport
        # A window the default context already spans needs no row of its own.
        if (
            context == "+1m"
            and context not in capability.context
            and capability.context[""].max_request_tokens >= 1_000_000
        ):
            capability = replace(
                capability,
                context=MappingProxyType(
                    {**capability.context, "+1m": capability.context[""]},
                ),
            )
        if context not in capability.context:
            offered = ", ".join(sorted(t for t in capability.context if t)) or "(none)"
            raise UnsupportedTagError(
                f"Model {model_id!r} has no {context} context; offers: {offered}",
            )
        return capability, ModelSettings.narrowest(capability, context=context)
