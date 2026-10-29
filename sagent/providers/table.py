"""The models one provider serves, looked up by name.

Typical usage example:

  table = ModelTable.from_yaml(path, provider="anthropic")
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
from types import MappingProxyType
from typing import TYPE_CHECKING, Self, cast, override

import math
import re

from treekle import from_plain

import yaml

from sagent.model.capability import (
    ModelCapability,
    ModelLimits,
    ModelSettings,
    ThinkingBudget,
    ThinkingEffort,
    ThinkingOutput,
    literal_members,
)
from sagent.model.context import ContextSizeName
from sagent.model.cost import PriceCatalog, PriceCondition, ServiceTier, TokenPrice
from sagent.model.exceptions import UnknownModelError, UnsupportedTagError


if TYPE_CHECKING:
    from importlib.resources.abc import Traversable


__all__ = [
    "ModelTable",
    "ModelTags",
    "chosen",
    "spelled",
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
    def from_yaml(cls, path: Traversable, *, provider: str) -> Self:
        """Read the models one provider serves from a vendor's catalog document.

        Layers, lowest first: the reference row, the model's profile, the
        model, the provider's section, and the section's entry for the model.
        Each replaces whole fields, except that the section's ``thinking``
        replaces only the axes it names. A model the section does not list is
        not served. The section's ``roles`` replace the document's it names.

        Args:
          path: The vendor's YAML document.
          provider: The name of a section under ``providers``.

        Returns:
          table: The section's models, validated, and the document's roles.

        """
        document = from_plain(_load(path.read_text()), _CatalogDocument)
        for layer in (document.reference, *document.profiles.values()):
            _require_whole_thinking(layer)
        # Read only to validate: a bad profile fails here, not at the first
        # model that names it.
        from_plain(document.reference, _CapabilityProfile)
        for profile in document.profiles.values():
            from_plain(document.reference | profile, _CapabilityProfile)
        if provider not in document.providers:
            raise ValueError(f"The catalog has no provider {provider!r}.")
        section = dict(document.providers[provider])
        listed = from_plain(
            section.pop("models", {}),
            dict[str, dict[str, object] | None],
        )
        # A provider serving other models than the vendor's own names its own.
        roles = document.roles | from_plain(section.pop("roles", {}), dict[str, str])
        ids = [from_plain(raw.get("model_id"), str) for raw in document.models]
        if unknown := sorted(set(listed) - set(ids)):
            raise ValueError(f"Provider {provider!r} lists unknown models: {unknown}.")
        rows = tuple(
            _model(
                raw,
                document=document,
                section=section,
                entry=listed[model_id] or {},
            )
            for raw, model_id in zip(document.models, ids, strict=True)
            if model_id in listed
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
        if capability is None:
            known = ", ".join([*self, *self.roles])
            raise UnknownModelError(
                f"Unknown model {model_id!r}; known models: {known}.",
            )
        settings = capability.default
        if tags.context is not None:
            settings = replace(
                settings,
                context_size=capability.prices[settings.service_tier, tags.context],
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
        """Return the row ``name`` means by exact name, vendor id, or family."""
        row = self.exact(name)
        if row is not None:
            return row
        return next(
            (row for row in self.rows if _in_family(row.model_id, name=name)),
            None,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelTags:
    """A model line: the model, and the size, effort, tier, and choices tagged on it.

    ``work@anthropic/opus-5.5+fullcontext+medthink+fast budget=auto``. Tags
    follow the model in any order and case; each word belongs to one kind, so
    position decides nothing. A word of no kind, such as SelfHosted's ``+cuda``,
    stays in the model. Choices follow after spaces.
    """

    model: str
    """The model as named, with any account and provider; its tags lifted off."""

    context: ContextSizeName | int | None = None
    """The size it names, as ``fullcontext`` or ``258_000``; ``None`` for none."""

    effort: ThinkingEffort | None = None
    """The effort it names; ``None`` for none."""

    service_tier: ServiceTier | None = None
    """The tier it names, ``+fast`` being ``priority``; ``None`` for none."""

    choices: Mapping[str, str] = field(
        default_factory=lambda: MappingProxyType[str, str]({}),
    )
    """Each ``CHOICE=VALUE`` after the tags, as typed: ``{"budget": "auto"}``."""

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
          UnsupportedTagError: Two tags name different sizes, two an effort, or
            two a tier; a word after the tags is no ``CHOICE=VALUE``, or names a
            choice twice.

        """
        spec, *after = line.split()
        choices: dict[str, str] = {}
        for word in after:
            choice, equals, value = word.partition("=")
            if not equals:
                raise UnsupportedTagError(
                    f"{line!r} names {word!r}, not a CHOICE=VALUE",
                )
            if choice in choices:
                raise UnsupportedTagError(f"{line!r} names {choice!r} twice")
            if f"thinking_{choice}" not in _choices() and choice not in _choices():
                names = ", ".join(c.removeprefix("thinking_") for c in _choices())
                raise UnsupportedTagError(f"{choice} is no choice; give {names}")
            choices[choice] = value
        model, *words = spec.split("+")
        kept: list[str] = [model]
        sizes: dict[ContextSizeName | int, str] = {}
        effort: list[ThinkingEffort] = []
        tier: list[ServiceTier] = []
        for raw in words:
            word = raw.lower()
            if (size := _size(word)) is not None:
                sizes.setdefault(size, f"+{word}")
            elif (named := _effort(word)) is not None:
                effort.append(named)
            elif (served := _tier(word)) is not None:
                tier.append(served)
            else:
                kept.append(raw)
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
            effort=effort[0] if effort else None,
            service_tier=tier[0] if tier else None,
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
        named = capability.prices[settings.service_tier, size.context_name] == size
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
                    field.removeprefix("thinking_"): spelled(chosen(settings, field))
                    for field in _choices()
                    if chosen(settings, field) != chosen(fresh, field)
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
                *(f"{choice}={value}" for choice, value in self.choices.items()),
            ],
        )


def chosen(settings: ModelSettings, field: str) -> object:
    """Return the choice ``settings`` made on ``field``.

    Args:
      settings: The model's settings.
      field: A ``ModelSettings`` field.

    Returns:
      value: What it holds.

    """
    match field:
        case "thinking_effort":
            return settings.thinking_effort
        case "thinking_budget":
            return settings.thinking_budget
        case "thinking_output":
            return settings.thinking_output
        case "service_tier":
            return settings.service_tier
        case "cache_ttl_sec":
            return settings.cache_ttl_sec
        case _:
            return settings.manage_context_server_side


def spelled(value: object) -> str:
    """Return ``value`` as a user types it: ``true``, ``300``, ``high``."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _choices() -> tuple[str, ...]:
    """Return the fields a line spells after its tags, as ``CHOICE=VALUE``."""
    return (
        "thinking_budget",
        "thinking_output",
        "cache_ttl_sec",
        "manage_context_server_side",
    )


def _family_version(model_id: str) -> tuple[str, tuple[int, int]] | None:
    """Split ``opus-5.5`` into ``("opus", (5, 5))``; ``None`` for other names."""
    family, _, version = model_id.rpartition("-")
    major, dot, minor = version.partition(".")
    if (
        not dot
        or not major.isdecimal()
        or not minor.isdecimal()
        or re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", family) is None
    ):
        return None
    return family, (int(major), int(minor))


def _in_family(model_id: str, *, name: str) -> bool:
    """Whether ``name`` is ``model_id``'s family, alone or with a major version."""
    # A caller's own id (``local-model``) has no family: ``local`` names nothing.
    if _family_version(model_id) is None:
        return False
    family, _, version = model_id.rpartition("-")
    if name == family:
        return True
    head, _, major = name.rpartition("-")
    return head == family and (version == major or version.startswith(f"{major}."))


def _load(text: str) -> object:
    """Return the YAML document ``text``, refused if ``_validate_yaml`` refuses it."""
    loader = (yaml.CSafeLoader if yaml.__with_libyaml__ else yaml.SafeLoader)(text)
    try:
        node = loader.get_single_node()
        _validate_yaml(node, seen=set())
        return None if node is None else loader.construct_document(node)
    finally:
        loader.dispose()


def _validate_yaml(node: yaml.Node | None, *, seen: set[int]) -> None:
    """Reject aliases, keys that are not strings, and keys given twice."""
    if node is None:
        return
    if id(node) in seen:
        raise ValueError("Catalogs may not use YAML aliases.")
    seen.add(id(node))
    if isinstance(node, yaml.MappingNode):
        keys: set[str] = set()
        for key, value in node.value:
            if (
                not isinstance(key, yaml.ScalarNode)
                or key.tag != "tag:yaml.org,2002:str"
            ):
                raise ValueError("Catalog mapping keys must be strings.")
            if key.value in keys:
                raise ValueError(f"The catalog gives key {key.value!r} twice.")
            keys.add(key.value)
            _validate_yaml(value, seen=seen)
    elif isinstance(node, yaml.SequenceNode):
        for value in node.value:
            _validate_yaml(value, seen=seen)


@dataclass(frozen=True, slots=True, kw_only=True)
class _CatalogDocument:
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
    approx_chars_per_token: float = 4.0
    limits: ModelLimits = field(default_factory=ModelLimits)
    thinking: _Thinking = field(default_factory=_Thinking)
    effort_as_level: bool = False
    thinking_effort_default: ThinkingEffort | None = None
    """The vendor's effort when none is sent; ``None`` is the lowest offered."""

    thinking_budget_default: ThinkingBudget | None = None
    """Whether the vendor thinks when told nothing; ``None`` is the lowest offered."""

    context_size_default: str | int = "fullcontext"
    """The size a bare id starts at: ``fullcontext`` and the other names, or a count."""

    cache_ttl_sec: frozenset[float] = frozenset({0.0})
    manage_context_server_side: frozenset[bool] = frozenset({False, True})
    retries_internally: bool = False
    account_auth: bool = False
    service_tier: frozenset[ServiceTier] | None = None
    """Tiers offered; ``None`` offers every tier the row prices."""


def _model(
    raw: dict[str, object],
    *,
    document: _CatalogDocument,
    section: dict[str, object],
    entry: dict[str, object],
) -> ModelCapability:
    """Return one model's row, its layers merged, as one provider serves it."""
    values = dict(raw)
    _require_whole_thinking(values)
    _require_whole_thinking(entry)
    profile = from_plain(values.pop("profile", ""), str)
    if profile and profile not in document.profiles:
        raise ValueError(f"The catalog has no profile {profile!r}.")
    values = document.reference | document.profiles.get(profile, {}) | values
    axes = from_plain(section.get("thinking", {}), dict[str, object])
    if axes.keys() - {"effort", "budget", "output"}:
        raise ValueError(f"Unknown thinking axes: {sorted(axes)}.")
    values |= {key: value for key, value in section.items() if key != "thinking"}
    values["thinking"] = (
        from_plain(values.get("thinking", {}), dict[str, object]) | axes
    )
    values |= entry
    row = from_plain(values, _ModelRow)
    if _family_version(row.model_id) is None:
        raise ValueError(f"Model id {row.model_id!r} is not family-major.minor.")
    _check_limits(row)
    if (
        not row.prices
        or not row.thinking.effort
        or not row.thinking.budget
        or not row.thinking.output
        or row.approx_chars_per_token <= 0
        or not math.isfinite(row.approx_chars_per_token)
    ):
        raise ValueError(
            f"Catalog model {row.model_id!r} is missing a price or setting.",
        )
    prices = _prices(row)
    tiers = prices.service_tiers if row.service_tier is None else row.service_tier
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
    return ModelCapability(
        model_id=row.model_id,
        wire_model_id=row.wire_model_id,
        knowledge_cutoff=row.knowledge_cutoff,
        approx_chars_per_token=row.approx_chars_per_token,
        limits=row.limits,
        thinking_effort=row.thinking.effort,
        thinking_budget=row.thinking.budget,
        thinking_output=row.thinking.output,
        effort_as_level=row.effort_as_level,
        prices=prices,
        service_tier=tiers,
        cache_ttl_sec=row.cache_ttl_sec,
        manage_context_server_side=row.manage_context_server_side,
        retries_internally=row.retries_internally,
        account_auth=row.account_auth,
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
            thinking_output=_lowest(
                "thinking_output",
                row.thinking.output,
                literal_members(ThinkingOutput),
            ),
            service_tier=tier,
            cache_ttl_sec=_lowest(
                "cache_ttl_sec",
                row.cache_ttl_sec,
                sorted(row.cache_ttl_sec),
            ),
            manage_context_server_side=_lowest(
                "manage_context_server_side",
                row.manage_context_server_side,
                (False, True),
            ),
            context_size=prices[tier, size],
        ),
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
    return ladder if default is None else (default, *ladder)


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
