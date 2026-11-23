"""The models one provider serves, looked up by name.

Typical usage example:

  table = models(anthropic, section="anthropic-cli")  # A section of anthropic.yaml.
  table["opus-5.5"]                         # Its exact name.
  table["claude-opus-5-5"]                  # The vendor's id.
  table["default"]                          # A role: best, default, everyday, utility.
  table["opus"], table["opus-5"]            # The newest of the family.
  capability, settings = table.resolve("opus+smallcontext")  # Below its first price band.
  tags = ModelTags.parse("work@anthropic/opus+fullcontext+medthink budget=auto")
  str(ModelTags.from_settings(capability, settings, provider="anthropic"))
"""

from __future__ import annotations

from collections.abc import Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import date, datetime
from functools import cache
from importlib.resources import files
from types import MappingProxyType, ModuleType
from typing import TYPE_CHECKING, ClassVar, Self, cast, override

import math
import re

from treekle import from_plain

from sagent.lib.userdirs import config_dir
from sagent.lib.yaml import parse
from sagent.model.capability import (
    Compaction,
    ModelCapability,
    ModelLimits,
    ModelSettingName,
    ModelSettings,
    ThinkingBudget,
    ThinkingEffort,
    ThinkingOutput,
    literal_members,
    typed_text,
)
from sagent.model.context import ContextSizeName
from sagent.model.cost import PriceCatalog, PriceCondition, ServiceTier, TokenPrice
from sagent.model.exceptions import UnknownModelError, UnsupportedTagError


if TYPE_CHECKING:
    from importlib.resources.abc import Traversable
    from pathlib import Path


__all__ = [
    "CatalogDocument",
    "HasModels",
    "ModelTable",
    "ModelTags",
    "models",
    "tag_words",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelTable(Mapping[str, ModelCapability]):
    """A vendor's models, by exact name, and reachable by vendor id, role, or family.

    Iterating yields exact names only: a role or a family is a way to name a
    model, not another model.
    """

    rows: tuple[ModelCapability, ...]
    """Every model, newest first within each family."""

    roles: Mapping[str, str] = field(
        default_factory=lambda: MappingProxyType[str, str]({}),
    )
    """Each role's family or exact name (``{"default": "opus"}``)."""

    _by_id: Mapping[str, ModelCapability] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Index the rows, and check their order and that each role names one."""
        by_id: dict[str, ModelCapability] = {}
        newest: dict[str, tuple[int, int]] = {}
        for row in self.rows:
            if row.model_id in by_id:
                raise ValueError(f"Model {row.model_id!r} is listed twice.")
            by_id[row.model_id] = row
            # A caller's own id (``local-model``) has no version to order by.
            parsed = _family_version(row.model_id)
            if parsed is None:
                continue
            family, version = parsed
            if family in newest and version >= newest[family]:
                raise ValueError(
                    f"Family {family!r} must list its models newest first;"
                    f" {row.model_id!r} follows an older one.",
                )
            newest[family] = version
        object.__setattr__(self, "_by_id", MappingProxyType(by_id))
        roles = dict(self.roles)
        if "default" in roles:
            for role in ("best", "everyday"):
                roles.setdefault(role, roles["default"])
        object.__setattr__(self, "roles", MappingProxyType(roles))
        for role, target in self.roles.items():
            if self._named(target) is None:
                raise ValueError(f"Role {role!r} names no model: {target!r}.")

    @classmethod
    def from_yaml(cls, path: Traversable, *, section: str) -> Self:
        """Read the models one provider serves from a vendor's catalog file.

        Args:
          path: The vendor's YAML document.
          section: The name of a section under ``providers``.

        Returns:
          table: As ``from_text`` reads the file's text, named in errors.

        """
        return cls.from_text(path.read_text(), section=section, source=str(path))

    @classmethod
    def from_text(cls, text: str, *, section: str, source: str = "") -> Self:
        """Read the models one provider serves from a vendor's catalog document.

        Layers, lowest first: the reference row, the model's profile, the
        model, the provider's section, and the section's entry for the model
        under its ``models``. Each replaces whole fields, except that the
        section's ``thinking`` replaces only the axes it names. A section
        serves every model in the document: the document's list is the only list
        of models, so a model added there reaches every provider, and one an
        account is not served fails with the vendor's own error. The section's
        ``roles`` replace the document's it names.
        The section's ``sends`` names the settings the provider's requests
        carry; each other one offers only the row's default, what the vendor
        does unasked, since a choice nothing sends would change only the label.
        The section's ``service_tier`` names the tiers its provider can send; each
        row offers those of them it prices.

        Args:
          text: The vendor's YAML document.
          section: The name of a section under ``providers``.
          source: Where ``text`` came from, such as its path, named in errors.

        Returns:
          table: The section's models, validated, and the document's roles.

        """
        document = from_plain(
            parse(text, source=source, aliases=True),
            CatalogDocument,
        )
        for layer in (document.reference, *document.profiles.values()):
            _require_whole_thinking(layer)
        # Read only to validate: a bad profile fails here, not at the first
        # model that names it.
        from_plain(document.reference, _CapabilityProfile)
        for profile in document.profiles.values():
            from_plain(document.reference | profile, _CapabilityProfile)
        if section not in document.providers:
            raise ValueError(f"The catalog has no section {section!r}.")
        layer = dict(document.providers[section])
        if "sends" not in layer:
            raise ValueError(f"Section {section!r} does not say what it sends.")
        sends = frozenset(from_plain(layer.pop("sends"), list[ModelSettingName]))
        entries = from_plain(layer.pop("models", {}), dict[str, object])
        # Catalogs once named the models a section served (bare names) or left
        # out (``omits``); a user's old document says so instead of loading wrong.
        served = (
            "A section now serves every model in the document; delete {}. A model an"
            " account is not served fails with the vendor's own error."
        )
        if bare := sorted(k for k, v in entries.items() if v is None):
            raise ValueError(
                f"Section {section!r} lists models by bare name: {bare}. "
                + served.format("the bare names"),
            )
        if "omits" in layer:
            omitted = sorted(from_plain(layer["omits"], dict[str, object]))
            raise ValueError(
                f"Section {section!r} omits models: {omitted}. "
                + served.format("its omits"),
            )
        overrides = from_plain(entries, dict[str, dict[str, object]])
        # A provider serving other models than the vendor's own names its own.
        roles = document.roles | from_plain(layer.pop("roles", {}), dict[str, str])
        named = [(from_plain(raw.get("model_id"), str), raw) for raw in document.models]
        if unknown := sorted(set(overrides) - {model_id for model_id, _ in named}):
            raise ValueError(
                f"Section {section!r} overrides unknown models: {unknown}.",
            )
        rows = tuple(
            _unsent_fixed(
                _model(
                    raw,
                    document=document,
                    section=layer,
                    section_name=section,
                    entry=overrides.get(model_id, {}),
                ),
                sends=sends,
            )
            for model_id, raw in named
        )
        return cls(rows=rows, roles=roles)

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
        """Return the row whose exact name or vendor id is ``name``, never a guess.

        For ids a server reports back: billing ``gpt-5`` at the newest
        ``gpt-5.x`` because the family matched would misprice it silently.

        Args:
          name: An exact name or a vendor id.

        Returns:
          row: The row, or ``None`` when no row carries that id.

        """
        row = self._by_id.get(name)
        if row is not None:
            return row
        return next((row for row in self.rows if row.wire_model_id == name), None)

    def resolve(self, model_id: str) -> tuple[ModelCapability, ModelSettings]:
        """Return the model a name means, and settings selecting its context's size.

        Args:
          model_id: An exact name, vendor id, role, or family, with any size tag
            trailing it: ``opus+smallcontext`` (below its first price band),
            ``+midcontext``, ``+fullcontext``, or a count such as ``+258k``.

        Returns:
          capability: The row.
          settings: The row's defaults, at the named size.

        Raises:
          UnknownModelError: No row answers to the name.
          UnsupportedTagError: Two size tags conflict.

        """
        tags = ModelTags.parse(model_id)
        # The effort, tier, and choices are the caller's to apply; read here,
        # they would vanish.
        sized_only = (
            tags.effort is None and tags.service_tier is None and not tags.choices
        )
        capability = self.get(tags.model) if sized_only else None
        families = self._families(self.roles.get(tags.model, tags.model))
        if capability is None and len(families) > 1:
            raise UnknownModelError(
                f"Model {model_id!r} names more than one family:"
                f" {', '.join(families)}; name one.",
            )
        if capability is None:
            known = ", ".join([*self, *self.roles])
            raise UnknownModelError(
                f"Unknown model {model_id!r}; known models: {known}.",
            )
        settings = capability.default
        if tags.context is not None:
            settings = replace(
                settings,
                context_size=capability.prices[tags.context],
            )
        return capability, settings

    def _resolved(self, key: str) -> ModelCapability | None:
        """Return the row ``key`` names: by exact name or vendor id, then role or family."""
        row = self.exact(key)
        return row if row is not None else self._named(self.roles.get(key, key))

    # A family is named alone or with a major: ``opus`` and ``opus-5`` name
    # ``opus-*``; ``opu`` names nothing, and ``gemini-flash`` never reaches
    # ``gemini-flash-lite-*``, a family of its own.
    def _named(self, name: str) -> ModelCapability | None:
        """Return the row ``name`` means by exact name, vendor id, or one family."""
        row = self.exact(name)
        if row is not None:
            return row
        families = self._families(name)
        if len(families) != 1:
            return None
        return next(
            (
                row
                for row in self.rows
                if row.model_id.rpartition("-")[0] == families[0]
                and _in_family(row.model_id, name=name)
            ),
            None,
        )

    # ``neon`` is ``gemini-neon``, but ``pro`` ends both ``gpt-pro`` and ``o-pro``:
    # which one it means is no row order's to decide.
    def _families(self, name: str) -> tuple[str, ...]:
        """Return the families ``name`` may mean: one named whole, else each it ends."""
        known = tuple(
            dict.fromkeys(
                parsed[0]
                for row in self.rows
                if (parsed := _family_version(row.model_id)) is not None
            ),
        )
        head, _, major = name.rpartition("-")
        words = (name, head) if major.isdecimal() else (name,)
        whole = tuple(family for family in known if family in words)
        if whole:
            return whole
        return tuple(
            family
            for family in known
            if any(_names_family(word, family=family) for word in words)
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelTags:
    """A model line: the model, and the size, effort, tier, and choices tagged on it.

    ``work@anthropic/opus-5.5+fullcontext+medthink+fast budget=auto``. Tags
    follow the model in any order and case; each word belongs to one kind, so
    position decides nothing, and a value tagged twice is tagged once. A word of
    no kind, such as SelfHosted's ``+cuda``, stays in the model. Choices follow
    after a space, from the first word holding ``=``; the model before them may
    hold spaces, as a local path does.
    """

    model: str
    """The model as named, with any account and provider; its tags lifted off."""

    context: ContextSizeName | int | None = None
    """The size it names, as ``fullcontext`` or ``258_000``; ``None`` for none."""

    effort: ThinkingEffort | None = None
    """The effort it names; ``None`` for none."""

    service_tier: ServiceTier | None = None
    """The tier it names, ``+fast`` being ``priority``; ``None`` for none."""

    choices: Mapping[ModelSettingName, str] = field(
        default_factory=lambda: MappingProxyType[ModelSettingName, str]({}),
    )
    """Each ``CHOICE=VALUE`` after the tags, by setting, its value as typed:
    ``budget=auto`` is ``{"thinking_budget": "auto"}``."""

    @classmethod
    def parse(cls, line: str) -> Self:
        """Return what ``line`` names.

        Sizes and efforts read in their old words too (``+large``, ``+high``):
        journals keep lines written before the tags were renamed.

        Args:
          line: ``MODEL[+TAG...] [CHOICE=VALUE...]``, with any account and provider.

        Returns:
          tags: The model, and what its tags and choices name.

        Raises:
          UnsupportedTagError: Two tags name different sizes, efforts, or
            tiers; a word after the first choice is no ``CHOICE=VALUE``, or
            names a choice twice.

        """
        # Choices start at the first word holding ``=``: a local path may hold spaces.
        spec, *after = re.split(r"\s+(?=\S*=)", line.strip(), maxsplit=1)
        choices: dict[ModelSettingName, str] = {}
        # A line spells a choice with or without ``thinking_``: ``budget=auto``.
        spellings: dict[str, ModelSettingName] = {
            spelling: name
            for name in _choices()
            for spelling in (name, name.removeprefix("thinking_"))
        }
        for word in after[0].split() if after else ():
            choice, equals, value = word.partition("=")
            if not equals:
                raise UnsupportedTagError(
                    f"{line!r} names {word!r}, not a CHOICE=VALUE",
                )
            if choice not in spellings:
                names = ", ".join(c.removeprefix("thinking_") for c in _choices())
                raise UnsupportedTagError(f"{choice} is no choice; give {names}")
            if spellings[choice] in choices:
                raise UnsupportedTagError(f"{line!r} names {choice!r} twice")
            choices[spellings[choice]] = value
        model, *words = spec.split("+")
        kept: list[str] = [model]
        # Each kind keeps one entry per value: a value spelled twice is one tag.
        sizes: dict[ContextSizeName | int, str] = {}
        efforts: list[ThinkingEffort] = []
        tiers: list[ServiceTier] = []
        for raw in words:
            word = raw.lower()
            if (size := _size(word)) is not None:
                sizes.setdefault(size, f"+{word}")
            elif (named := _effort(word)) is not None:
                efforts.append(named)
            elif (served := _tier(word)) is not None:
                tiers.append(served)
            else:
                kept.append(raw)
        effort, tier = tuple(dict.fromkeys(efforts)), tuple(dict.fromkeys(tiers))
        if len(sizes) > 1:
            raise UnsupportedTagError(
                f"Model {line!r} selects conflicting contexts:"
                f" {', '.join(sorted(sizes.values()))}.",
            )
        for kind, found in (("efforts", effort), ("service tiers", tier)):
            if len(found) > 1:
                raise UnsupportedTagError(
                    f"{line!r} names two {kind}: {', '.join(found)}",
                )
        return cls(
            model="+".join(kept),
            context=next(iter(sizes), None),
            effort=next(iter(effort), None),
            service_tier=next(iter(tier), None),
            choices=MappingProxyType(choices),
        )

    @classmethod
    def from_settings(
        cls,
        capability: ModelCapability,
        settings: ModelSettings,
        *,
        provider: str = "",
        account: str | None = None,
    ) -> Self:
        """Return the line naming ``capability`` as ``settings`` set it.

        The size and effort are always named; the tier and every other choice
        only where they differ from what the model starts with.

        Args:
          capability: The model.
          settings: Its settings.
          provider: The provider serving it; ``""`` names none.
          account: The login its provider signs in as; ``None`` for the default.

        Returns:
          tags: What a line naming it says, as ``parse`` reads it back.

        """
        fresh = capability.default
        size = settings.context_size
        named = capability.prices[size.context_name] == size
        signed = "" if account is None else f"{account}@"
        return cls(
            model=f"{signed}{provider}/{capability.model_id}"
            if provider
            else capability.model_id,
            context=size.context_name if named else size.tokens,
            effort=settings.thinking_effort,
            service_tier=None
            if settings.service_tier == fresh.service_tier
            else settings.service_tier,
            choices=MappingProxyType(
                {
                    name: typed_text(settings.value(name))
                    for name in _choices()
                    if settings.value(name) != fresh.value(name)
                },
            ),
        )

    @property
    def name(self) -> str:
        """Return what a provider looks the model up by: the model and its size."""
        return (
            self.model
            if self.context is None
            else f"{self.model}+{_size_word(self.context)}"
        )

    @override
    def __str__(self) -> str:
        tags = (
            "" if self.effort is None else f"+{_effort_word(self.effort)}",
            "" if self.service_tier is None else f"+{self.service_tier}",
        )
        return " ".join(
            [
                f"{self.name}{''.join(tags)}",
                *(
                    f"{name.removeprefix('thinking_')}={value}"
                    for name, value in self.choices.items()
                ),
            ],
        )


def tag_words() -> dict[str, tuple[str, ...]]:
    """Return each kind of tag and the words a model line spells it with.

    Returns:
      words: ``size``, ``effort``, and ``tier``, each word one ``ModelTags.parse``
        reads as that kind; ``fast`` is ``priority``. Old spellings it still reads
        (``+large``, ``+high``) are left out.

    """
    efforts = cast(tuple[ThinkingEffort, ...], literal_members(ThinkingEffort))
    return {
        "size": literal_members(ContextSizeName),
        "effort": tuple(_effort_word(effort) for effort in efforts),
        "tier": (*literal_members(ServiceTier), "fast"),
    }


def models(package: ModuleType, *, section: str) -> ModelTable:
    """Return a provider's user catalog, or its shipped catalog when none exists.

    ``config_dir()/rekursiv-ai/sagent/providers/<section>.yaml`` replaces the
    shipped document, using the same schema and named provider section. Invalid
    overrides raise rather than falling back. Catalogs are read once per user
    path and section; restart the process to reload an edited document.

    Args:
      package: The vendor's package, which holds its catalog as
        ``<leaf>.yaml``: ``anthropic`` holds ``anthropic.yaml``.
      section: The section, as ``anthropic-cli``.

    Returns:
      table: Its models, each setting the section does not send fixed at what
        the vendor does.

    """
    return _read(
        package.__name__,
        section=section,
        user_path=config_dir()
        / "rekursiv-ai"
        / "sagent"
        / "providers"
        / f"{section}.yaml",
    )


class HasModels:
    """A provider class whose models are one section of its vendor's catalog."""

    package: ClassVar[ModuleType]
    """The vendor's package, holding its catalog."""

    section: ClassVar[str]
    """The catalog section whose models and settings this provider serves."""

    aliases: ClassVar[tuple[str, ...]]
    """The other names it answers to; each listed class names its own."""

    @classmethod
    def models(cls) -> ModelTable:
        """Return the models this provider serves, offering only what it sends."""
        return models(cls.package, section=cls.section)


# Keyed by name: pyrefly rejects a module as a cache key, though it hashes.
@cache
def _read(package: str, *, section: str, user_path: Path) -> ModelTable:
    """Read one section, preferring the user's document over the shipped one."""
    if user_path.exists():
        return ModelTable.from_yaml(user_path, section=section)
    shipped = _shipped(package, section=section)
    return ModelTable(
        rows=tuple(
            replace(row, prices=replace(row.prices, prices=dict(row.prices.prices)))
            for row in shipped.rows
        ),
        roles=shipped.roles,
    )


@cache
def _shipped(package: str, *, section: str) -> ModelTable:
    """Read the bundled catalog once, independently of per-user snapshot paths."""
    leaf = package.rpartition(".")[2]
    return ModelTable.from_yaml(files(package) / f"{leaf}.yaml", section=section)


def _choices() -> tuple[ModelSettingName, ...]:
    """Return the fields a line spells after its tags, as ``CHOICE=VALUE``."""
    return (
        "thinking_budget",
        "thinking_output",
        "cache_ttl_sec",
        "manage_context_server_side",
        "compaction",
    )


def _family_version(model_id: str) -> tuple[str, tuple[int, int]] | None:
    """Split ``opus-5.5`` into ``("opus", (5, 5))``; ``None`` for other names."""
    family, _, version = model_id.rpartition("-")
    # pragma: no mutate start -- A second dot fails either split: ``2.3`` is no decimal.
    major, dot, minor = version.partition(".")
    # pragma: no mutate end
    if (
        not dot
        or not major.isdecimal()
        or not minor.isdecimal()
        or re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", family) is None
    ):
        return None
    return family, (int(major), int(minor))


# A family is named whole or by its last words: ``gemini-neon`` by ``neon``.
def _in_family(model_id: str, *, name: str) -> bool:
    """Whether ``name`` is ``model_id``'s family, alone or with a major version."""
    # A caller's own id (``local-model``) has no family: ``local`` names nothing.
    if _family_version(model_id) is None:
        return False
    family, _, version = model_id.rpartition("-")
    if _names_family(name, family=family):
        return True
    head, _, major = name.rpartition("-")
    return _names_family(head, family=family) and (
        version == major or version.startswith(f"{major}.")
    )


def _names_family(name: str, *, family: str) -> bool:
    """Whether ``name`` is ``family`` or its last dash-separated words."""
    return bool(name) and (name == family or family.endswith(f"-{name}"))


@dataclass(frozen=True, slots=True, kw_only=True)
class CatalogDocument:
    """A vendor's catalog document, before any model is resolved."""

    models: tuple[dict[str, object], ...]
    reference: dict[str, object] = field(default_factory=dict)
    profiles: dict[str, dict[str, object]] = field(default_factory=dict)
    providers: dict[str, dict[str, object]] = field(default_factory=dict)
    roles: dict[str, str] = field(default_factory=dict)
    sources: dict[str, str] = field(default_factory=dict)


# Nested objects replace whole, so an omitted axis would become ``none`` rather
# than inherit from the layer below.
def _require_whole_thinking(layer: dict[str, object]) -> None:
    """Reject a ``thinking`` override that omits an axis."""
    thinking = layer.get("thinking")
    if isinstance(thinking, dict) and thinking.keys() != {"effort", "budget", "output"}:
        raise ValueError("A catalog's thinking must state effort, budget, and output.")


@dataclass(frozen=True, slots=True, kw_only=True)
class _Thinking:
    """The reasoning a row offers, axis by axis, as the catalog states it."""

    effort: frozenset[ThinkingEffort] = frozenset({"none"})
    budget: frozenset[ThinkingBudget] = frozenset({"none"})
    output: frozenset[ThinkingOutput] = frozenset({"none"})


@dataclass(frozen=True, slots=True, kw_only=True)
class _CapabilityProfile:
    """Model facts several rows share; no identity, prices, or profile of its own."""

    knowledge_cutoff: str | None = None
    """ISO 8601 at the precision the vendor publishes: ``2026-06`` or ``2026-05-18``."""

    tokenizer: str | None = None
    approx_chars_per_token: float = 4.0
    limits: ModelLimits = field(default_factory=ModelLimits)
    thinking: _Thinking = field(default_factory=_Thinking)
    effort_as_level: bool = False
    thinking_off_max_effort: ThinkingEffort | None = None
    """The highest effort at which thinking may be off; ``None`` is any."""

    thinking_effort_default: ThinkingEffort | None = None
    """The vendor's effort when none is sent; ``None`` is the lowest offered."""

    thinking_budget_default: ThinkingBudget | None = None
    """Whether the vendor thinks when told nothing; ``None`` is the lowest offered."""

    thinking_output_default: ThinkingOutput | None = None
    """What of the reasoning the vendor returns unasked; ``None`` is the lowest offered."""

    context_size_default: str | int = "fullcontext"
    """The size a bare id starts at: ``fullcontext`` and the other names, or a count."""

    cache_ttl_sec: frozenset[float] = frozenset({0.0})
    cache_ttl_sec_default: float | None = None
    """The cache lifetime a bare id starts at; ``None`` is the shortest offered."""

    manage_context_server_side: frozenset[bool] = frozenset({False, True})
    compaction: frozenset[Compaction] = frozenset({"bespoke"})
    compaction_default: Compaction | None = None
    """How a bare id compacts; ``None`` is the first offered, ``bespoke`` before ``provider``."""

    retries_internally: bool = False
    account_auth: bool = False
    keeps_user_turns_apart: bool = False
    service_tier: frozenset[ServiceTier] | None = None
    """Tiers offered; ``None`` offers every tier the row prices."""


# ``limits`` replaces whole, as ``thinking`` does, so a limit a higher layer leaves
# out would silently fall to 0 (no cap) rather than keep the value below it.
def _require_restated_limits(
    below: Mapping[str, object],
    layer: Mapping[str, object],
    *,
    name: str,
) -> None:
    """Reject a ``limits`` override that drops a limit the layers below it state."""
    if "limits" not in layer or "limits" not in below:
        return
    stated = from_plain(below["limits"], dict[str, object]).keys()
    kept = from_plain(layer["limits"], dict[str, object]).keys()
    for dropped in sorted(stated - kept):
        raise ValueError(
            f"{name} restates no {dropped} of the limits below it;"
            " state it, 0 for none.",
        )


def _model(
    raw: dict[str, object],
    *,
    document: CatalogDocument,
    section: dict[str, object],
    section_name: str,
    entry: dict[str, object],
) -> ModelCapability:
    """Return one model's row, its layers merged, as one provider serves it."""
    values = dict(raw)
    _require_whole_thinking(values)
    _require_whole_thinking(entry)
    profile = from_plain(values.pop("profile", ""), str)
    if profile and profile not in document.profiles:
        raise ValueError(f"The catalog has no profile {profile!r}.")
    profiled = document.profiles.get(profile, {})
    model_id = from_plain(values.get("model_id"), str)
    merged = document.reference
    for layer, name in (
        (profiled, f"profile {profile!r}"),
        (values, f"model {model_id!r}"),
        (section, f"section {section_name!r}"),
        (entry, f"section {section_name!r}"),
    ):
        _require_restated_limits(merged, layer, name=name)
        merged = merged | {k: v for k, v in layer.items() if k == "limits"}
    values = document.reference | profiled | values
    axes = from_plain(section.get("thinking", {}), dict[str, object])
    if axes.keys() - {"effort", "budget", "output"}:
        raise ValueError(f"Unknown thinking axes: {sorted(axes)}.")
    # A section's tiers are those its provider sends, not a claim each model prices.
    sent = section.get("service_tier")
    values |= {
        key: value
        for key, value in section.items()
        if key not in {"thinking", "service_tier"}
    }
    values["thinking"] = (
        from_plain(values.get("thinking", {}), dict[str, object]) | axes
    )
    values |= entry
    row = from_plain(values, _ModelRow)
    if _family_version(row.model_id) is None:
        raise ValueError(f"Model id {row.model_id!r} is not family-major.minor.")
    _check_limits(row)
    _check_cutoff(row)
    if (
        not row.prices
        or row.approx_chars_per_token <= 0
        or not math.isfinite(row.approx_chars_per_token)
    ):
        raise ValueError(
            f"Catalog model {row.model_id!r} is missing a price or setting.",
        )
    prices = _prices(row)
    tiers = prices.service_tiers if row.service_tier is None else row.service_tier
    for unpriced in sorted(tiers - prices.service_tiers):
        raise ValueError(
            f"Model {row.model_id!r} offers service_tier {unpriced!r} with no price.",
        )
    if sent is not None:
        sendable = frozenset(from_plain(sent, list[ServiceTier]))
        tiers &= sendable
        if not tiers:
            raise ValueError(
                f"Model {row.model_id!r} is priced at none of the service tiers"
                f" section {section_name!r} sends: {sorted(sendable)}.",
            )
    tier = _lowest("service_tier", tiers, literal_members(ServiceTier))
    size = (
        _size(row.context_size_default.lower())
        if isinstance(row.context_size_default, str)
        else row.context_size_default
    )
    if size is None or (isinstance(size, int) and size < 1):
        raise ValueError(
            f"Model {row.model_id!r} has a bad context_size_default:"
            f" {row.context_size_default!r}.",
        )
    capability = ModelCapability(
        model_id=row.model_id,
        wire_model_id=row.wire_model_id,
        knowledge_cutoff=row.knowledge_cutoff,
        tokenizer=row.tokenizer,
        approx_chars_per_token=row.approx_chars_per_token,
        limits=row.limits,
        thinking_effort=row.thinking.effort,
        thinking_budget=row.thinking.budget,
        thinking_output=row.thinking.output,
        effort_as_level=row.effort_as_level,
        thinking_off_max_effort=row.thinking_off_max_effort,
        prices=prices,
        service_tier=tiers,
        cache_ttl_sec=row.cache_ttl_sec,
        manage_context_server_side=row.manage_context_server_side,
        compaction=row.compaction,
        retries_internally=row.retries_internally,
        account_auth=row.account_auth,
        keeps_user_turns_apart=row.keeps_user_turns_apart,
        default=ModelSettings(
            thinking_effort=_lowest(
                "thinking_effort",
                row.thinking.effort,
                _effort_ladder(
                    row.thinking_effort_default,
                    literal_members(ThinkingEffort),
                ),
            ),
            thinking_budget=_lowest(
                "thinking_budget",
                row.thinking.budget,
                _default_first(
                    row.thinking_budget_default,
                    literal_members(ThinkingBudget),
                ),
            ),
            # Readable thinking wherever a row offers it and names no default of its
            # own, so a session keeps every summary the model returns; the TUI hides
            # it unless asked.
            thinking_output=_lowest(
                "thinking_output",
                row.thinking.output,
                _default_first(
                    row.thinking_output_default or "text",
                    literal_members(ThinkingOutput),
                ),
            ),
            service_tier=tier,
            cache_ttl_sec=_lowest(
                "cache_ttl_sec",
                row.cache_ttl_sec,
                _default_first(
                    row.cache_ttl_sec_default,
                    tuple(sorted(row.cache_ttl_sec)),
                ),
            ),
            manage_context_server_side=_lowest(
                "manage_context_server_side",
                row.manage_context_server_side,
                (False, True),
            ),
            compaction=_lowest(
                "compaction",
                row.compaction,
                _default_first(row.compaction_default, literal_members(Compaction)),
            ),
            context_size=prices[size],
        ),
    )
    # Each default is picked alone; a pairing of them can still be refused.
    capability.check(capability.default)
    return capability


def _unsent_fixed(
    capability: ModelCapability,
    *,
    sends: frozenset[ModelSettingName],
) -> ModelCapability:
    """Return ``capability`` offering only its default on each setting not in ``sends``."""
    d = capability.default
    return replace(
        capability,
        thinking_effort=capability.thinking_effort
        if "thinking_effort" in sends
        else frozenset({d.thinking_effort}),
        thinking_budget=capability.thinking_budget
        if "thinking_budget" in sends
        else frozenset({d.thinking_budget}),
        thinking_output=capability.thinking_output
        if "thinking_output" in sends
        else frozenset({d.thinking_output}),
        service_tier=capability.service_tier
        if "service_tier" in sends
        else frozenset({d.service_tier}),
        cache_ttl_sec=capability.cache_ttl_sec
        if "cache_ttl_sec" in sends
        else frozenset({d.cache_ttl_sec}),
        manage_context_server_side=capability.manage_context_server_side
        if "manage_context_server_side" in sends
        else frozenset({d.manage_context_server_side}),
        compaction=capability.compaction
        if "compaction" in sends
        else frozenset({d.compaction}),
    )


# A default the row lacks falls to the nearest level below it, then above, and only then
# to ``none``, which stops the reasoning.
def _effort_ladder(
    default: ThinkingEffort | None,
    efforts: tuple[ThinkingEffort, ...],
) -> tuple[ThinkingEffort, ...]:
    """Return ``efforts`` in the order a row's default tries them."""
    at = efforts.index(default) if default is not None else 0
    return (*efforts[at:0:-1], *efforts[at + 1 :], efforts[0]) if at else efforts


def _default_first[T](default: T | None, ladder: tuple[T, ...]) -> tuple[T, ...]:
    """Return ``ladder`` with ``default``, when one is given, tried first."""
    # pragma: no mutate start -- A ``None`` tried first changes nothing: no row offers it.
    return ladder if default is None else (default, *ladder)
    # pragma: no mutate end


def _lowest[T](name: str, offered: Collection[T], ladder: Sequence[T]) -> T:
    """Return the first rung of ``ladder`` that ``offered`` contains."""
    for rung in ladder:
        if rung in offered:
            return rung
    raise ValueError(f"{name} offers nothing selectable: {offered!r}")


@dataclass(frozen=True, slots=True, kw_only=True)
class _PriceRow(TokenPrice):
    """One rate card, and the conditions that select it."""

    service_tier: ServiceTier = "auto"
    effective_prompt_tokens: int = 0
    """Prompts larger than this; ``0`` is every prompt."""

    effective_date: str | date = "0001-01-01"


@dataclass(frozen=True, slots=True, kw_only=True)
class _ModelRow(_CapabilityProfile):
    """One model's document, its profile applied."""

    model_id: str
    prices: tuple[_PriceRow, ...]
    wire_model_id: str = ""


def _check_limits(row: _ModelRow) -> None:
    """Reject a negative limit, or a shared window smaller than its input."""
    limits = row.limits
    if (
        min(
            limits.max_request_tokens,
            limits.max_response_tokens,
            limits.max_context_tokens,
            limits.max_request_bytes,
            limits.max_image_edge_px,
            limits.max_image_bytes,
        )
        < 0
    ):
        raise ValueError(f"Model {row.model_id!r} has a negative limit.")
    if 0 < limits.max_context_tokens < limits.max_request_tokens:
        raise ValueError(
            f"Model {row.model_id!r} has a context smaller than its request.",
        )


def _check_cutoff(row: _ModelRow) -> None:
    """Reject a knowledge cutoff that is not an ISO date."""
    cutoff = row.knowledge_cutoff
    if cutoff is None:
        return
    # ``fromisoformat`` also reads ``20250501`` and week dates: only one spelling.
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", cutoff):
        try:
            date.fromisoformat(cutoff)
        except ValueError:
            pass
        else:
            return
    raise ValueError(
        f"Model {row.model_id!r} has knowledge_cutoff {cutoff!r}; write it as YYYY-MM-DD.",
    )


def _prices(row: _ModelRow) -> PriceCatalog:
    """Return the row's rate cards, keyed by tier, prompt band, and start date."""
    # A card applies to prompts larger than its threshold, up to the next one's.
    thresholds: dict[ServiceTier, list[int]] = {}
    for card in row.prices:
        thresholds.setdefault(card.service_tier, []).append(
            card.effective_prompt_tokens,
        )
    prices: dict[PriceCondition, TokenPrice] = {}
    for card in row.prices:
        rates = (
            card.request,
            card.response,
            card.cache_write,
            card.cache_write_1h,
            card.cache_read,
        )
        if (
            card.tokens_per_unit <= 0
            or card.effective_prompt_tokens < 0
            or card.service_tier == "default"
            or isinstance(card.effective_date, datetime)
            or any(rate < 0 or not math.isfinite(rate) for rate in rates)
        ):
            raise ValueError(f"Model {row.model_id!r} has an invalid price.")
        above = card.effective_prompt_tokens
        ceil = min(
            (t for t in thresholds[card.service_tier] if t > above),
            default=None,
        )
        key = PriceCondition(
            service_tier=card.service_tier,
            token_size_range=(
                above + 1 if above else 0,
                math.inf if ceil is None else ceil + 1,
            ),
            effective_date=date.fromisoformat(card.effective_date)
            if isinstance(card.effective_date, str)
            else card.effective_date,
        )
        if key in prices:
            raise ValueError(f"Model {row.model_id!r} prices {key!r} twice.")
        prices[key] = TokenPrice(
            request=card.request,
            response=card.response,
            cache_write=card.cache_write,
            cache_write_1h=card.cache_write_1h,
            cache_read=card.cache_read,
            tokens_per_unit=card.tokens_per_unit,
        )
    # A lookup raises for a request no card covers; only an undated card with
    # no prompt band covers every small prompt on every day.
    for tier in thresholds:
        if not any(
            c.service_tier == tier
            and c.token_size_range[0] == 0
            and c.effective_date == date.min
            for c in prices
        ):
            raise ValueError(
                f"Model {row.model_id!r} has no base card for tier {tier!r}"
                " (no prompt band, no effective date).",
            )
    # A size is the model's, named where its price bands change: a tier or date
    # priced flat is fine, but two banding apart would name a size one never bills at.
    banded = {
        frozenset(
            c.effective_prompt_tokens
            for c in row.prices
            if (c.service_tier, c.effective_date) == when and c.effective_prompt_tokens
        )
        for when in {(c.service_tier, c.effective_date) for c in row.prices}
    } - {frozenset()}
    if len(banded) > 1:
        raise ValueError(f"Model {row.model_id!r} bands its tiers at different sizes.")
    return PriceCatalog(prices=prices, window=row.limits.max_request_tokens)


def _size(word: str) -> ContextSizeName | int | None:
    """Return the size ``word`` names: ``fullcontext``, its old ``large``, or ``258k``."""
    names: dict[str, ContextSizeName] = {
        "smallcontext": "smallcontext",
        "small": "smallcontext",
        "midcontext": "midcontext",
        "mid": "midcontext",
        "fullcontext": "fullcontext",
        "large": "fullcontext",
    }
    if word in names:
        return names[word]
    counted = re.fullmatch(r"([1-9][0-9]*)([km]?)", word)
    if counted is None:
        return None
    return int(counted[1]) * {"": 1, "k": 1_000, "m": 1_000_000}[counted[2]]


def _size_word(size: ContextSizeName | int) -> str:
    """Return the word ``_size`` reads as ``size``: ``fullcontext``, ``258k``, ``258123``."""
    if isinstance(size, str):
        return size
    for unit, scale in (("m", 1_000_000), ("k", 1_000)):
        if size % scale == 0:
            return f"{size // scale}{unit}"
    return str(size)


def _effort(word: str) -> ThinkingEffort | None:
    """Return the effort ``word`` names: ``medthink``, ``mediumthink``, or the old ``medium``."""
    efforts = cast(tuple[ThinkingEffort, ...], literal_members(ThinkingEffort))
    named: dict[str, ThinkingEffort] = {
        **{effort: effort for effort in efforts},
        **{_effort_word(effort): effort for effort in efforts},
        "mediumthink": "medium",
    }
    return named.get(word)


def _effort_word(effort: ThinkingEffort) -> str:
    """Return the word a line spells ``effort`` as: ``medium`` is ``medthink``."""
    return f"{ {'none': 'no', 'medium': 'med'}.get(effort, effort) }think"


def _tier(word: str) -> ServiceTier | None:
    """Return the tier ``word`` names, ``fast`` being ``priority``."""
    tiers = cast(tuple[ServiceTier, ...], literal_members(ServiceTier))
    word = "priority" if word == "fast" else word
    return tiers[tiers.index(word)] if word in tiers else None
