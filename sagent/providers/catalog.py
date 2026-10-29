"""Every provider, by name, sharing one HTTP client, process pools, and tool server.

Typical usage example:

  async with ModelCatalog() as catalog:
      opus = catalog["anthropic-cli"]["opus"]        # No I/O until it streams.
      catalog.use_account("work")                   # Sign-ins as work from now.

  work = ModelCatalog(
      factories={"anthropic-work": partial(AnthropicCLI, credentials=work_login)},
  )
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from functools import partial
from types import MappingProxyType
from typing import TYPE_CHECKING, ClassVar, Self, cast, override

import re

from sagent.model.capability import (
    ModelCapability,
    ModelSettings,
    ThinkingBudget,
    ThinkingEffort,
    ThinkingOutput,
    literal_members,
)
from sagent.model.cost import ServiceTier
from sagent.model.exceptions import UnknownModelError, UnknownProviderError
from sagent.providers.anthropic.api import Anthropic
from sagent.providers.anthropic.cli import AnthropicCLI
from sagent.providers.anthropic.login import ClaudeLogin
from sagent.providers.dashscope.api import DashScope
from sagent.providers.google.api import Google
from sagent.providers.google.cli import GoogleCLI
from sagent.providers.listing import Listing
from sagent.providers.llamacpp.server import LlamaCpp
from sagent.providers.minimax.api import MiniMax
from sagent.providers.moonshot.api import Moonshot
from sagent.providers.openai.api import OpenAI
from sagent.providers.openai.login import CodexLogin
from sagent.providers.openai.sub import OpenAISub
from sagent.providers.resources import SharedResources, close_all
from sagent.providers.selfhosted.server import SelfHosted
from sagent.providers.table import ModelTags, spelled


if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Iterable, Sequence
    from pathlib import Path

    from sagent.providers.types import (
        Model,
        ModelCatalog as CatalogProtocol,
        Provider,
        ProviderFactory,
    )


__all__ = [
    "CachedModel",
    "ModelCatalog",
    "ModelMemory",
    "PerAccount",
    "account_credentials",
    "accounts",
    "cache_key",
    "choose",
    "default_provider",
    "listings",
    "offered",
    "pick",
    "recall",
    "successor",
]


def listings() -> tuple[Listing, ...]:
    """Return every provider the catalog offers.

    Returns:
      listings: In the order a model name is asked of them.

    """
    return (
        Listing.of(Anthropic),
        Listing.signing_in(AnthropicCLI, default=ClaudeLogin.default_path),
        Listing.of(OpenAI),
        Listing.signing_in(
            OpenAISub,
            default=CodexLogin.default_path,
            former_names=("OpenAISubscription",),
        ),
        Listing.of(Google),
        Listing.of(GoogleCLI),
        Listing.of(DashScope),
        Listing.of(MiniMax),
        Listing.of(Moonshot),
        Listing.of(LlamaCpp),
        Listing.of(SelfHosted),
    )


def default_provider() -> str:
    """Return the provider a session starts on unless told otherwise.

    Returns:
      name: The one listed as preferred, else the first.

    """
    offered = listings()
    return next((o.section for o in offered if o.preferred), offered[0].section)


def account_credentials(provider: str, account: str) -> Path:
    """Return the login file ``provider`` keeps for the account ``account``.

    It is the default file with ``-<account>`` added to its stem, beside it.

    Args:
      provider: A signing-in provider's own name, as ``openai-sub``.
      account: The account's name: letters, digits, ``_``, ``-``.

    Returns:
      path: The account's credentials file.

    Raises:
      ValueError: ``provider`` keeps no named accounts, or ``account`` is no name.

    """
    logins = _logins()
    default = logins.get(provider)
    if default is None:
        raise ValueError(
            f"provider {provider!r} has no named accounts;"
            f" ones that do: {', '.join(logins)}",
        )
    _check_name(account)
    path = default()
    return path.with_name(f"{path.stem}-{account}{path.suffix}")


def accounts(provider: str) -> tuple[str, ...]:
    """Return the named accounts ``provider`` has a login saved for, sorted.

    Args:
      provider: A provider's own name, as ``openai-sub``.

    Returns:
      names: Each account whose file ``account_credentials`` names exists;
        none for a provider that keeps no named accounts.

    """
    default = _logins().get(provider)
    if default is None:
        return ()
    path = default()
    names = (
        found.name.removeprefix(f"{path.stem}-").removesuffix(path.suffix)
        for found in path.parent.glob(f"{path.stem}-*{path.suffix}")
    )
    return tuple(sorted(n for n in names if _ACCOUNT.fullmatch(n)))


class ModelCatalog(SharedResources):
    """Providers by name, each built on first use and closed with the catalog.

    A provider answers to its name in any case and with or without separators
    (``OpenAISub``, ``openai_sub``), to an alias (``anthropic-api``), to a
    former name (``OpenAISubscription``), and to an abbreviation: its letters
    in order, from the first (``antcli``, ``oaisub``, ``aapi``). A name or an
    alias abbreviated wins over a former name abbreviated, so ``oapi`` is
    ``openai``, not ``OpenAISubscription``.

    Args:
      factories: How to build each provider, by the name it is looked up by,
        or a function returning them for an account; ``None`` offers every
        provider under its catalog section's name, each signing in as the
        account.
      account: A named account each signing-in provider signs in as;
        ``None``, ``""`` or ``"default"`` uses each one's default login.

    Raises:
      ValueError: ``account`` is no account name, or ``factories`` is fixed
        and ``account`` names one.

    """

    aliases: ClassVar[Mapping[str, str]] = MappingProxyType(
        {
            "anthropic-api": "anthropic",
            "google-api": "google",
            "openai-api": "openai",
        },
    )
    """Other names a provider goes by: each vendor's API is its bare name."""

    def __init__(
        self,
        *,
        factories: Mapping[str, ProviderFactory] | PerAccount | None = None,
        account: str | None = None,
    ) -> None:
        super().__init__()
        self._per_account: PerAccount = (
            _registry
            if factories is None
            else partial(_fixed, factories)
            if isinstance(factories, Mapping)
            else factories
        )
        self.factories: Mapping[str, ProviderFactory] = self._per_account(
            _named(account),
        )
        self._account = _named(account)
        # Each account's providers, open until the catalog closes.
        self._opened: dict[str | None, dict[str, Provider]] = {self._account: {}}
        self._providers = self._opened[self._account]
        # Every name offered at first, kept through ``allow``: a name that is a
        # provider's own never abbreviates another, as ``anthropic`` would the
        # ``anthropic-cli`` left after ``anthropic`` is no longer allowed.
        self._full_names = frozenset(_key(name) for name in self.factories)

    @property
    def account(self) -> str | None:
        """Return the account each signing-in provider signs in as; ``None`` for the default."""
        return self._account

    def use_account(self, account: str | None) -> None:
        """Sign each provider in as ``account`` from now on, offering the same names.

        Each account's providers stay open, for whatever holds their models,
        and close with the catalog: switching back finds them as left.

        Args:
          account: The account; ``None``, ``""`` or ``"default"`` for each
            provider's default login.

        Raises:
          ValueError: ``account`` is no account name, or the catalog's
            factories are fixed.

        """
        named = _named(account)
        fresh = self._per_account(named)
        self.factories = {name: fresh[name] for name in self.factories}
        self._account = named
        self._providers = self._opened.setdefault(named, {})

    def __getitem__(self, provider: str) -> Provider:
        provider = self.name(provider)
        opened = self._providers.get(provider)
        if opened is None:
            build = self.factories[provider]
            opened = self._providers[provider] = build(name=provider, resources=self)
        return opened

    def signed_in(self, provider: str, *, account: str | None) -> Provider:
        """Return ``provider`` signed in as ``account``, leaving the catalog's own.

        Args:
          provider: Its name, in any spelling ``name`` takes.
          account: The account; ``None``, ``""`` or ``"default"`` for the
            provider's default login.

        Returns:
          provider: Built on first use for that account, open until the
            catalog closes.

        Raises:
          ValueError: ``account`` is no account name, or the catalog's
            factories are fixed and ``account`` names one.

        """
        named = _named(account)
        if named == self._account:
            return self[provider]
        name = self.name(provider)
        opened = self._opened.setdefault(named, {})
        if name not in opened:
            build = self._per_account(named)[name]
            opened[name] = build(name=name, resources=self)
        return opened[name]

    def model(self, name: str, *, current: str) -> Model:
        """Return the model ``name`` names, from whichever provider serves it.

        ``P/M`` asks the provider ``P`` for ``M``, and only it: a typo in ``M``
        must not reach a provider that takes any ``ORG/NAME``, as SelfHosted
        takes a Hugging Face repo. Otherwise the whole name is asked of
        ``current``, then of each provider in the order offered, as ``allow``
        left it.

        Args:
          name: ``[PROVIDER/]MODEL``, with any window tag; a repo whose org
            names a provider is ``selfhosted/ORG/NAME``.
          current: The provider in use, asked first.

        Returns:
          model: The first model a provider answers with.

        Raises:
          UnknownModelError: No provider serves it; the first refusal says why.

        """
        prefix, slash, rest = name.partition("/")
        if slash and self._names_provider(prefix):
            try:
                return self[prefix][rest]
            except UnknownModelError as e:
                if SelfHosted.section not in self.factories:
                    raise
                raise UnknownModelError(
                    f"{e} A Hugging Face repo whose org names a provider is"
                    f" {SelfHosted.section}/{name}.",
                ) from e
        refusals: list[UnknownModelError] = []
        asks = [(current, name), *((p, name) for p in self.factories if p != current)]
        for provider, wanted in asks:
            try:
                return self[provider][wanted]
            except UnknownModelError as e:
                refusals.append(e)
        raise refusals[0]

    def name(self, provider: str) -> str:
        """Return the name ``factories`` keys the provider ``provider`` names by.

        Args:
          provider: Its name, in any spelling the class docstring allows.

        Returns:
          name: The provider's own name.

        Raises:
          UnknownProviderError: No provider answers to it, or several do.

        """
        if provider in self.factories:
            return provider
        wanted = _key(provider)
        current = {
            name: {
                _key(name),
                *(_key(other) for other, to in self.aliases.items() if to == name),
            }
            for name in self.factories
        }
        renamed = {
            listing.section: {_key(old) for old in listing.former_names}
            for listing in listings()
        }
        former = {name: renamed.get(name, set[str]()) for name in self.factories}
        for spellings, exact in (
            (current, True),
            (former, True),
            (current, False),
            (former, False),
        ):
            matches = [
                name
                for name, keys in spellings.items()
                if (
                    wanted in keys
                    if exact
                    else wanted not in self._full_names
                    and any(_abbreviates(wanted, key) for key in keys)
                )
            ]
            if len(matches) == 1:
                return matches[0]
            if matches:
                raise UnknownProviderError(
                    f"provider {provider!r} could be any of: {', '.join(matches)}",
                )
        known = ", ".join(self.factories)
        raise UnknownProviderError(f"unknown provider {provider!r}; known: {known}")

    def allow(self, providers: Iterable[str]) -> list[str]:
        """Offer only ``providers``, and return their names.

        Args:
          providers: Each in any spelling ``name`` takes.

        Returns:
          names: The providers' own names, in order, each once.

        Raises:
          UnknownProviderError: Naming every one no single provider answers to.

        """
        names: list[str] = []
        unknown: list[str] = []
        for provider in providers:
            try:
                names.append(self.name(provider))
            except UnknownProviderError:
                unknown.append(provider)
        if unknown:
            known = ", ".join(self.factories)
            raise UnknownProviderError(
                f"unknown providers {', '.join(sorted(unknown))}; known: {known}",
            )
        self.factories = {name: self.factories[name] for name in dict.fromkeys(names)}
        return list(self.factories)

    @override
    async def aclose(self) -> None:
        providers = [p for opened in self._opened.values() for p in opened.values()]
        self._opened = {self._account: {}}
        self._providers = self._opened[self._account]
        try:
            await close_all(providers)
        finally:
            await super().aclose()

    # ``Qwen/Qwen3.6-27B`` is a repo id, not the provider ``Qwen``'s model.
    def _names_provider(self, prefix: str) -> bool:
        """Return whether exactly one provider answers to ``prefix``."""
        try:
            self.name(prefix)
        except UnknownProviderError:
            return False
        return True


type PerAccount = Callable[[str | None], Mapping[str, ProviderFactory]]
"""The providers to build for an account, by name; ``None`` is each one's default."""


@dataclass(frozen=True, slots=True, kw_only=True)
class CachedModel:
    """A model switched away from, and the settings it was left with."""

    capability: ModelCapability
    settings: ModelSettings


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelMemory:
    """An agent's model and the settings each model it left was left with, as lines.

    Each line is a ``ModelTags``: what ``recall`` builds back.
    """

    model: str
    """The model in use."""

    cache: tuple[str, ...] = ()
    """Each other model the agent was switched away from, by its key's order."""

    @property
    def account(self) -> str | None:
        """Return the named login the model in use signs in as; ``None`` for the default."""
        account, at, _ = (
            ModelTags.parse(self.model).model.partition("/")[0].rpartition("@")
        )
        return account if at else None

    @classmethod
    def from_model(
        cls,
        model: Model,
        cache: Mapping[str, CachedModel],
        *,
        account: str | None = None,
    ) -> Self:
        """Return the memory of ``model`` in use, and of ``cache``'s other models.

        Args:
          model: The model in use.
          cache: Each model switched away from, by ``cache_key``.
          account: The named login ``model``'s provider signs in as.

        Returns:
          memory: The lines.

        """
        current = cache_key(model)
        return cls(
            model=str(
                ModelTags.from_settings(
                    model.capability,
                    model.settings,
                    provider=model.provider.name,
                    account=account,
                ),
            ),
            cache=tuple(
                str(
                    ModelTags.from_settings(
                        cached.capability,
                        cached.settings,
                        provider=key.partition("/")[0],
                    ),
                )
                for key, cached in sorted(cache.items())
                if key != current
            ),
        )


def cache_key(model: Model) -> str:
    """Return what a model's settings are cached under: ``provider/model_id``."""
    return f"{model.provider.name}/{model.capability.model_id}"


async def successor(
    build: Callable[[str], Model],
    name: str,
    *,
    current: Model,
    model_cache: Mapping[str, CachedModel] | None = None,
) -> Model:
    """Return the model ``name`` names to replace ``current``, keeping its choices.

    The new model starts from the settings it was last left with in
    ``model_cache``, unless it is ``current``'s model, else from ``current``'s:
    its context size, unless ``name``
    names one, and each setting it offers. The tags and choices ``name``
    spells (``opus+highthink+fast budget=auto``) are applied last.

    Args:
      build: Looks a model up by name, such as a provider's ``__getitem__``.
      name: ``MODEL[+TAG...] [CHOICE=VALUE...]``, with what ``build`` takes
        before it.
      current: The model replaced.
      model_cache: Each model switched away from, by ``cache_key``.

    Returns:
      model: The new model, not yet assigned to anything.

    Raises:
      ValueError: ``build`` refused the name, or the new model does not offer
        what it names; a model built is closed first.

    """
    tags = ModelTags.parse(name)
    model = build(tags.name)
    try:
        key = cache_key(model)
        cached = None if key == cache_key(current) else (model_cache or {}).get(key)
        source = (
            cached
            if cached is not None
            else CachedModel(capability=current.capability, settings=current.settings)
        )
        fitted = model.capability.fit(source.settings, was=source.capability)
        if tags.context is not None:
            fitted = replace(fitted, context_size=model.settings.context_size)
        model.settings = fitted
        _apply(model, tags)
    except ValueError:
        await model.aclose()
        raise
    return model


async def recall(
    memory: ModelMemory,
    *,
    catalog: CatalogProtocol,
) -> tuple[Model, dict[str, CachedModel]]:
    """Return the model ``memory`` names, and the settings it says each other was left with.

    Each line's provider serves its model, or none does: no other is asked. The
    model in use signs in as the account its line names, whatever the catalog's.

    Args:
      memory: The lines.
      catalog: Where the providers are.

    Returns:
      model: The model in use, not yet assigned to anything.
      cache: Each other model and its settings, by ``cache_key``.

    Raises:
      ValueError: A line names what is no longer offered; a model built is
        closed first.

    """
    cache: dict[str, CachedModel] = {}
    for line in memory.cache:
        model = await _built(catalog, line)
        cache[cache_key(model)] = CachedModel(
            capability=model.capability,
            settings=model.settings,
        )
        await model.aclose()
    return await _built(catalog, memory.model), cache


def choose(settings: ModelSettings, field: str, value: object) -> ModelSettings:
    """Return ``settings`` with ``field`` set to ``value``, as ``chosen`` reads it back.

    Args:
      settings: The model's settings.
      field: A ``ModelSettings`` field ``chosen`` reads.
      value: What ``offered`` lists for it.

    Returns:
      settings: The new choice made; unchecked against any capability.

    Raises:
      ValueError: ``field`` is no choice.

    """
    match field:
        case "thinking_effort":
            return replace(settings, thinking_effort=cast(ThinkingEffort, value))
        case "thinking_budget":
            return replace(settings, thinking_budget=cast(ThinkingBudget, value))
        case "thinking_output":
            return replace(settings, thinking_output=cast(ThinkingOutput, value))
        case "service_tier":
            return replace(settings, service_tier=cast(ServiceTier, value))
        case "cache_ttl_sec" if isinstance(value, int | float):
            return replace(settings, cache_ttl_sec=float(value))
        case "manage_context_server_side" if isinstance(value, bool):
            return replace(settings, manage_context_server_side=value)
        case _:
            raise ValueError(f"{field}={value!r} is no choice")


def offered(capability: ModelCapability, field: str) -> tuple[object, ...]:
    """Return what ``capability`` offers on ``field``, least committing first.

    Args:
      capability: The model.
      field: A ``ModelSettings`` field.

    Returns:
      values: Each value it may take.

    """
    match field:
        case "thinking_effort":
            return _ladder(capability.thinking_effort, literal_members(ThinkingEffort))
        case "thinking_budget":
            return _ladder(capability.thinking_budget, literal_members(ThinkingBudget))
        case "thinking_output":
            return _ladder(capability.thinking_output, literal_members(ThinkingOutput))
        case "service_tier":
            return _ladder(capability.service_tier, literal_members(ServiceTier))
        case "cache_ttl_sec":
            return tuple(sorted(capability.cache_ttl_sec))
        case _:
            return tuple(sorted(capability.manage_context_server_side))


def pick(capability: ModelCapability, field: str, typed: str) -> object:
    """Return the value ``capability`` offers on ``field`` whose text is ``typed``.

    Args:
      capability: The model.
      field: A ``ModelSettings`` field.
      typed: The value as a user types it, in any case.

    Returns:
      value: The offered value.

    Raises:
      ValueError: The model offers no such value; the message lists those it does.

    """
    values = offered(capability, field)
    hits = [v for v in values if spelled(v) == typed.strip().lower()]
    if not hits:
        texts = ", ".join(spelled(v) for v in values)
        raise ValueError(f"{typed!r} is not one of: {texts}")
    return hits[0]


def _ladder(offered: Collection[object], order: Sequence[object]) -> tuple[object, ...]:
    return tuple(v for v in order if v in offered)


def _apply(model: Model, tags: ModelTags) -> None:
    """Apply the effort, tier and choices ``tags`` name to ``model``; raise if unoffered."""
    capability = model.capability
    settings = model.settings
    if tags.effort is not None:
        settings = replace(settings, thinking_effort=tags.effort)
    # ``ModelTags.parse`` admits only the choices a line spells.
    for choice, typed in tags.choices.items():
        field = f"thinking_{choice}" if choice in ("budget", "output") else choice
        try:
            settings = choose(settings, field, pick(capability, field, typed))
        except ValueError as e:
            raise ValueError(f"{choice}: {e}") from e
    if tags.service_tier is not None:
        capability.check(replace(settings, service_tier=tags.service_tier))
        settings = capability.fit(
            settings,
            was=capability,
            service_tier=tags.service_tier,
        )
    model.settings = settings


async def _built(catalog: CatalogProtocol, line: str) -> Model:
    """Return the model ``line`` names, set as it spells, from its defaults."""
    tags = ModelTags.parse(line)
    signed, _, name = tags.name.partition("/")
    account, _, provider = signed.rpartition("@")
    model = catalog.signed_in(provider, account=account or None)[name]
    try:
        _apply(model, tags)
    except ValueError:
        await model.aclose()
        raise
    return model


def _registry(account: str | None) -> Mapping[str, ProviderFactory]:
    """Return every provider by its section's name, each signing in as ``account``."""
    known: dict[str, ProviderFactory] = {}
    for listing in listings():
        known[listing.section] = (
            listing.factory
            if account is None or listing.signed_in is None
            else listing.signed_in(account_credentials(listing.section, account))
        )
    return known


def _logins() -> dict[str, Callable[[], Path]]:
    """Return where each provider that signs in keeps its default login, by name."""
    return {
        listing.section: listing.default_login
        for listing in listings()
        if listing.default_login is not None
    }


def _fixed(
    factories: Mapping[str, ProviderFactory],
    account: str | None,
) -> Mapping[str, ProviderFactory]:
    """Return ``factories``, the same for every account; a named one has none."""
    if account is not None:
        raise ValueError(
            f"a catalog of fixed factories builds its providers for no account;"
            f" asked for {account!r}",
        )
    return factories


def _named(account: str | None) -> str | None:
    """Return ``account``, checked, or ``None`` for each spelling of the default."""
    if account in (None, "", "default"):
        return None
    _check_name(account)
    return account


_ACCOUNT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")


def _check_name(account: str) -> None:
    """Refuse ``account`` unless it is a name: letters, digits, ``_``, ``-``."""
    if not _ACCOUNT.fullmatch(account):
        raise ValueError(
            f"account {account!r} is not a name: use letters, digits, '_', '-'",
        )


def _key(name: str) -> str:
    """Return ``name`` lower-cased, its separators dropped: ``OpenAI_Sub`` is ``openaisub``."""
    return re.sub(r"[^0-9a-z]", "", name.casefold())


# Letters in order, the first kept: ``oaisub`` abbreviates ``openaisub``; ``sub``
# abbreviates nothing, since every provider would answer to it.
def _abbreviates(short: str, name: str) -> bool:
    """Return whether ``short`` is ``name``'s letters in order, starting at its first."""
    letters = iter(name)
    return bool(short) and short[0] == name[:1] and all(c in letters for c in short)
