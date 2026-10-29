"""The agent's model as settings: ``model``, and every choice the catalog names under it.

::

    model                          [ACCOUNT@][PROVIDER/]MODEL[+TAG...]
    model.provider                 the provider; refused unless it serves the model
    model.provider.account         the login providers sign in as
    model.context                  the window, by size
    model.thinking                 effort, budget, and output at once
    model.thinking.effort          ModelSettings.thinking_effort
    model.thinking.budget          ModelSettings.thinking_budget
    model.thinking.output          ModelSettings.thinking_output
    model.service_tier             ModelSettings.service_tier
    model.cache_ttl_sec            ModelSettings.cache_ttl_sec
    model.manage_context_server_side

A switch keeps what the old model chose wherever the new one offers it; each
choice offers what the model's catalog row does, in the row's own words. A
choice inherited from an ancestor takes the closest value this model offers:
the nearest effort, a lower one on a tie; the nearest number; else, on an axis
with no order, the value a fresh model of it starts with.

Typical usage example:

  owners = [..., (("model",), AgentModel(agent=agent, catalog=catalog))]
  /model work@oaisub/gpt+highthink+fast
  /effort max
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial
from typing import TYPE_CHECKING, cast

from sagent.agent.types import Agent
from sagent.model.capability import (
    ModelCapability,
    ModelSettings,
    ThinkingEffort,
    literal_members,
)
from sagent.model.cost import ServiceTier
from sagent.providers.catalog import accounts, choose, offered, pick, successor
from sagent.providers.table import ModelTags, chosen, spelled
from sagent.providers.types import ModelCatalog
from sagent.tui.settings.types import Setting, SettingError


if TYPE_CHECKING:
    from collections.abc import Callable

    from sagent.providers.types import Model


__all__ = [
    "AgentModel",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class AgentModel:
    """Exports the agent's model and its choices, each switch keeping the rest."""

    agent: Agent
    catalog: ModelCatalog
    """Where a switch finds models; its account is the team's too."""

    def settings(self) -> list[Setting]:
        """Return ``model`` first, then each choice under it.

        Returns:
          settings: Each, named relative to where the model is mounted.

        """
        return [
            Setting(
                name=(),
                help_short="the model, by name, family, or role, with any tags",
                help_long=(
                    "ACCOUNT@ signs providers in as that login. PROVIDER/ names"
                    " the provider; without it, the current one serves the model"
                    " if it can, else the first allowed one that does. Tags follow"
                    " in any order: a context size (smallcontext, midcontext, fullcontext,"
                    " each also spelled out as smallcontext and so on, or a count"
                    " such as 258k), an effort (nothink, lowthink, medthink,"  # codespell:ignore nothink
                    " highthink, xhighthink, and so on), a tier (auto, default, flex,"
                    " priority or fast, ultrafast). What the name leaves out is"
                    " kept from the model before."
                ),
                usage="[ACCOUNT@][PROVIDER/]MODEL[+TAG...]",
                value=self._described,
                apply=self._model,
                cli=True,
            ),
            Setting(
                name=("provider",),
                help_short="the provider serving this model; one that cannot is refused",
                usage="NAME",
                value=lambda: self.agent.model.provider.name,
                options=lambda: tuple(self.catalog.factories),
                apply=self._provider,
                cli=True,
            ),
            Setting(
                name=("provider", "account"),
                help_short="the login providers sign in as; default: each one's own",
                usage="NAME",
                value=lambda: self.catalog.account or "default",
                options=self._accounts,
                apply=self._account,
                cli=True,
            ),
            Setting(
                name=("context",),
                help_short="how much the context may fill before it is compacted",
                usage="SIZE",
                value=self._context_value,
                options=self._sizes,
                apply=self._context,
                cli=True,
            ),
            Setting(
                name=("thinking",),
                help_short="the reasoning's effort, budget, and output at once",
                usage="AXIS=VALUE ...",
                value=self._thinking_value,
                apply=self._thinking,
                cli=True,
                inherit=partial(self._thinking, closest=True),
            ),
            self._axis(("thinking", "effort"), "how hard the model reasons"),
            self._axis(("thinking", "budget"), "how its reasoning budget is set"),
            self._axis(("thinking", "output"), "what of its reasoning comes back"),
            self._axis(("service_tier",), "the serving tier: speed against price"),
            self._axis(("cache_ttl_sec",), "how long the prompt cache lives"),
            self._axis(
                ("manage_context_server_side",),
                "whether the server rolls the history",
            ),
        ]

    def _axis(self, name: tuple[str, ...], help_short: str) -> Setting:
        """Return the setting for the ``ModelSettings`` field ``name`` spells."""
        field = "_".join(name)
        return Setting(
            name=name,
            help_short=help_short,
            usage="VALUE",
            value=partial(self._axis_value, field),
            options=lambda: (
                "default",
                *(spelled(v) for v in offered(self.agent.model.capability, field)),
            ),
            apply=partial(self._set, field),
            cli=True,
            inherit=partial(self._set, field, closest=True),
        )

    def _described(self) -> str:
        model = self.agent.model
        return str(
            ModelTags.from_settings(
                model.capability,
                model.settings,
                provider=model.provider.name,
                account=self.agent.account,
            ),
        )

    # ``build`` signs in as the catalog's account, which the agent's is then.
    async def _swap(self, build: Callable[[str], Model], name: str) -> None:
        """Make the agent's model ``build(name)``, keeping what ``name`` leaves out."""
        self.agent.model = await successor(
            build,
            name,
            current=self.agent.model,
            model_cache=self.agent.model_cache,
        )
        self.agent.account = self.catalog.account

    async def _model(self, spec: str) -> str:
        account, at, name = spec.rpartition("@")
        was = self.catalog.account
        if at:
            self.catalog.use_account(account)
        try:
            await self._swap(
                partial(self.catalog.model, current=self.agent.model.provider.name),
                name,
            )
        except ValueError:
            if at:
                self.catalog.use_account(was)
            raise
        return self._described()

    async def _provider(self, name: str) -> str:
        await self._swap(
            self.catalog[name].__getitem__,
            self.agent.model.capability.model_id,
        )
        return self._described()

    # The catalog is the team's too: what ``/model`` and a spawn look up next
    # signs in as the account, until a switch onto it fails.
    async def _account(self, name: str) -> str:
        was = self.catalog.account
        self.catalog.use_account(name)
        model = self.agent.model
        try:
            await self._swap(
                self.catalog[model.provider.name].__getitem__,
                model.capability.model_id,
            )
        except ValueError:
            self.catalog.use_account(was)
            raise
        return self.catalog.account or "default"

    def _fresh(self) -> ModelSettings:
        """Return what this model starts with, in its default window."""
        return self.agent.model.capability.default

    def _axis_value(self, field: str) -> str:
        now = chosen(self.agent.model.settings, field)
        return _marked(spelled(now), default=now == chosen(self._fresh(), field))

    def _context_value(self) -> str:
        size = self.agent.model.settings.context_size
        return _marked(
            size.display_name,
            default=size == self._fresh().context_size,
        )

    def _accounts(self) -> tuple[str, ...]:
        """Return ``default``, then each login this provider keeps and the one in use."""
        saved = set(accounts(self.agent.model.provider.name))
        if self.catalog.account is not None:
            saved.add(self.catalog.account)
        return ("default", *sorted(saved))

    def _sizes(self) -> tuple[str, ...]:
        """Return ``default``, then each size this model's price bands name."""
        model = self.agent.model
        sizes = model.capability.prices.sizes(model.settings.service_tier)
        return ("default", *(size.context_name for size in sizes))

    async def _context(self, size: str) -> str:
        model = self.agent.model
        if size.strip().lower() == "default":
            # Named bare, a successor would carry the current window over.
            fresh = model.provider[model.capability.model_id]
            fresh.settings = replace(
                fresh.capability.fit(model.settings, was=model.capability),
                context_size=fresh.settings.context_size,
            )
            self.agent.model = fresh
        else:
            await self._swap(
                model.provider.__getitem__,
                f"{model.capability.model_id}+{size.removeprefix('+')}",
            )
        return self._context_value()

    def _thinking_value(self) -> str:
        s = self.agent.model.settings
        return (
            f"effort={s.thinking_effort} budget={s.thinking_budget}"
            f" output={s.thinking_output}"
        )

    async def _thinking(self, args: str, *, closest: bool = False) -> str:
        picked: dict[str, object] = {}
        for word in args.split():
            axis, equals, value = word.partition("=")
            if not equals or axis not in ("effort", "budget", "output"):
                raise SettingError(
                    f"give AXIS=VALUE for effort, budget, output; not {word!r}",
                )
            field = f"thinking_{axis}"
            try:
                picked[field] = self._picked(field, value, closest=closest)
            except SettingError as e:
                raise SettingError(f"{axis}: {e}") from e
        if not picked:
            raise SettingError("give AXIS=VALUE for effort, budget, output")
        self._choose(picked)
        return self._thinking_value()

    async def _set(self, field: str, args: str, *, closest: bool = False) -> str:
        self._choose({field: self._picked(field, args, closest=closest)})
        return self._axis_value(field)

    def _choose(self, picked: dict[str, object]) -> None:
        """Set the model's ``picked`` choices; a new tier sizes the context again."""
        model = self.agent.model
        capability = model.capability
        tier = picked.pop("service_tier", None)
        settings = model.settings
        for field, value in picked.items():
            settings = choose(settings, field, value)
        if tier is not None:
            settings = capability.fit(
                settings,
                was=capability,
                service_tier=cast(ServiceTier, tier),
            )
        model.settings = settings

    # ``default`` is a fresh model's.
    def _picked(self, field: str, typed: str, *, closest: bool = False) -> object:
        """Return the value ``typed`` names on ``field``."""
        if typed.strip().lower() == "default":
            return chosen(self._fresh(), field)
        capability = self.agent.model.capability
        try:
            return pick(capability, field, typed)
        except ValueError as e:
            if not closest:
                raise SettingError(str(e)) from e
        near = _closest(capability, field, typed.strip().lower())
        return chosen(self._fresh(), field) if near is None else near


def _marked(shown: str, *, default: bool) -> str:
    """Return ``shown``, said to be the default when it is."""
    return f"{shown} (default)" if default else shown


_EFFORTS = tuple(
    str(e) for e in cast(tuple[ThinkingEffort, ...], literal_members(ThinkingEffort))
)


def _closest(capability: ModelCapability, field: str, typed: str) -> object | None:
    """Return the offered value nearest ``typed``, a lower one on a tie."""
    values = offered(capability, field)
    if field == "thinking_effort" and typed in _EFFORTS:
        at = _EFFORTS.index(typed)
        rungs = [_EFFORTS.index(str(v)) for v in values]
        return values[
            min(range(len(values)), key=lambda i: (abs(rungs[i] - at), rungs[i]))
        ]
    numbers = [v for v in values if isinstance(v, float)]
    try:
        wanted = float(typed)
    except ValueError:
        return None
    return min((abs(n - wanted), n) for n in numbers)[1] if numbers else None
