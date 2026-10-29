"""Tests for the agent's model as settings: ``model`` and every choice under it."""

from __future__ import annotations

from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING

import math
import re

import pytest

from sagent.agent.agent import Agent
from sagent.model.capability import ModelCapability, ModelLimits, ModelSettings
from sagent.model.cost import PriceCatalog, PriceCondition, TokenPrice
from sagent.model.exceptions import UnknownModelError
from sagent.model.scripted import ScriptedProvider
from sagent.providers.catalog import ModelCatalog
from sagent.providers.table import ModelTable, ModelTags
from sagent.tui.settings.harvest import harvest
from sagent.tui.settings.model import AgentModel


if TYPE_CHECKING:
    from pathlib import Path

    from sagent.providers.types import ProviderFactory
    from sagent.tui.settings.types import Setting


def _windowed(model_id: str, window: int) -> ModelCapability:
    """Return a model with only a window, at its defaults."""
    prices = PriceCatalog(window=window)
    return ModelCapability(
        model_id=model_id,
        limits=ModelLimits(max_request_tokens=window),
        prices=prices,
        default=ModelSettings(context_size=prices["auto", "fullcontext"]),
    )


_WIDE_PRICES = PriceCatalog(
    prices={
        PriceCondition(token_size_range=(0, 200_001)): TokenPrice(),
        PriceCondition(token_size_range=(200_001, math.inf)): TokenPrice(),
    },
    window=1_000_000,
)

_WIDE = ModelCapability(
    model_id="wide-1.0",
    limits=ModelLimits(max_request_tokens=1_000_000),
    prices=_WIDE_PRICES,
    thinking_effort=frozenset({"none", "low", "high", "max"}),
    thinking_budget=frozenset({"none", "auto"}),
    thinking_output=frozenset({"none", "text"}),
    service_tier=frozenset({"auto", "priority"}),
    cache_ttl_sec=frozenset({300.0, 3600.0}),
    manage_context_server_side=frozenset({False, True}),
    default=ModelSettings(
        thinking_effort="high",
        cache_ttl_sec=300.0,
        context_size=_WIDE_PRICES["auto", "fullcontext"],
    ),
)

_NARROW = _windowed("narrow-1.0", 200_000)

_OTHER = _windowed("other-1.0", 128_000)


def _per_account(account: str | None) -> dict[str, ProviderFactory]:
    """Return the providers; every account is served the same models."""
    del account
    return {
        "vendor": partial(
            ScriptedProvider,
            models=ModelTable(rows=(_WIDE, _NARROW), roles={"default": "narrow"}),
        ),
        "other": partial(
            ScriptedProvider,
            models=ModelTable(rows=(_OTHER,), roles={"default": "other"}),
        ),
    }


def _model(name: str = "wide") -> tuple[AgentModel, Agent, ModelCatalog]:
    catalog = ModelCatalog(factories=_per_account)
    agent = Agent(model=catalog["vendor"][name])
    return AgentModel(agent=agent, catalog=catalog), agent, catalog


def _setting(owner: AgentModel, name: str) -> Setting:
    settings = {".".join(s.name): s for s in harvest([(("model",), owner)])}
    return settings[name]


async def _apply(owner: AgentModel, name: str, args: str) -> str:
    return await _setting(owner, name).apply(args)


def _value(owner: AgentModel, name: str) -> str:
    value = _setting(owner, name).value
    assert value is not None
    return value()


def test_the_model_exports_each_choice_under_it_for_the_command_line() -> None:
    owner, _, _ = _model()
    settings = harvest([(("model",), owner)])
    assert [".".join(s.name) for s in settings] == [
        "model",
        "model.provider",
        "model.provider.account",
        "model.context",
        "model.thinking",
        "model.thinking.effort",
        "model.thinking.budget",
        "model.thinking.output",
        "model.service_tier",
        "model.cache_ttl_sec",
        "model.manage_context_server_side",
    ]
    assert all(s.cli and s.value is not None for s in settings)


def test_each_setting_shows_its_value_and_what_it_may_take() -> None:
    owner, _, _ = _model()
    shown = {
        ".".join(s.name): (s.value() if s.value else "", tuple(s.options()))
        for s in harvest([(("model",), owner)])
    }
    assert shown == {
        "model": ("vendor/wide-1.0+fullcontext+highthink", ()),
        "model.provider": ("vendor", ("vendor", "other")),
        "model.provider.account": ("default", ("default",)),
        "model.context": ("1m (default)", ("default", "smallcontext", "fullcontext")),
        "model.thinking": ("effort=high budget=none output=none", ()),
        "model.thinking.effort": (
            "high (default)",
            ("default", "none", "low", "high", "max"),
        ),
        "model.thinking.budget": ("none (default)", ("default", "none", "auto")),
        "model.thinking.output": ("none (default)", ("default", "none", "text")),
        "model.service_tier": ("auto (default)", ("default", "auto", "priority")),
        "model.cache_ttl_sec": ("300 (default)", ("default", "300", "3600")),
        "model.manage_context_server_side": (
            "false (default)",
            ("default", "false", "true"),
        ),
    }


@pytest.mark.asyncio
async def test_default_sets_what_a_fresh_model_starts_with_and_says_so() -> None:
    owner, agent, catalog = _model()
    await _apply(owner, "model", "work@vendor/wide+smallcontext+low+fast")
    await _apply(owner, "model.cache_ttl_sec", "3600")
    assert _value(owner, "model.thinking.effort") == "low"
    for name in (
        "model.context",
        "model.thinking.effort",
        "model.service_tier",
        "model.cache_ttl_sec",
        "model.provider.account",
    ):
        await _apply(owner, name, "default")
    settings = agent.model.settings
    assert (
        settings.context_size.tokens,
        settings.thinking_effort,
        settings.service_tier,
        settings.cache_ttl_sec,
        catalog.account,
    ) == (1_000_000, "high", "auto", 300.0, None)
    assert _value(owner, "model.thinking.effort") == "high (default)"
    said = await _apply(owner, "model.thinking", "effort=low budget=default")
    assert said == "effort=low budget=none output=none"


@pytest.mark.asyncio
async def test_model_takes_an_account_and_tags_in_any_order() -> None:
    owner, agent, catalog = _model()
    said = await _apply(owner, "model", "work@vendor/wide+fast+258k+lowthink")
    assert said == "work@vendor/wide-1.0+258k+lowthink+priority"
    assert (catalog.account, agent.account) == ("work", "work")
    assert agent.model.settings.context_size.tokens == 258_000


@pytest.mark.asyncio
async def test_a_bare_model_keeps_every_choice_it_does_not_name() -> None:
    owner, agent, _ = _model()
    await _apply(owner, "model.context", "smallcontext")
    await _apply(owner, "model.thinking.effort", "low")
    await _apply(owner, "model", "wide")
    assert (
        agent.model.settings.context_size.tokens,
        agent.model.settings.thinking_effort,
    ) == (
        200_000,
        "low",
    )


@pytest.mark.asyncio
async def test_a_model_refused_leaves_the_model_and_account_as_they_were() -> None:
    owner, agent, catalog = _model()
    before = agent.model
    with pytest.raises(UnknownModelError):
        await _apply(owner, "model", "work@vendor/nope")
    assert (agent.model, catalog.account) == (before, None)


@pytest.mark.asyncio
async def test_provider_keeps_the_model_or_is_refused() -> None:
    owner, agent, _ = _model("narrow")
    await _apply(owner, "model.provider", "vendor")
    assert agent.model.capability.model_id == "narrow-1.0"
    owner, agent, _ = _model()
    before = agent.model
    with pytest.raises(UnknownModelError, match=r"Unknown model 'wide-1\.0'"):
        await _apply(owner, "model.provider", "other")
    assert agent.model is before


@pytest.mark.asyncio
async def test_account_offers_the_saved_logins_and_the_one_in_use(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    (tmp_path / "auth-home.json").write_text("{}")
    catalog = ModelCatalog(factories=_signing_in)
    owner = AgentModel(
        agent=Agent(model=catalog["openai-sub"]["wide"]),
        catalog=catalog,
    )
    setting = _setting(owner, "model.provider.account")
    assert tuple(setting.options()) == ("default", "home")
    await setting.apply("work")
    assert tuple(setting.options()) == ("default", "home", "work")
    # Another provider's logins are its own: ``vendor`` keeps none.
    await _apply(owner, "model", "vendor/wide")
    assert tuple(setting.options()) == ("default", "work")


async def _inherit(owner: AgentModel, name: str, args: str) -> str:
    inherit = _setting(owner, name).inherit
    assert inherit is not None
    return await inherit(args)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("name", "given", "taken"),
    [
        ("model.thinking.effort", "max", "max"),
        ("model.thinking.effort", "xhigh", "high"),
        ("model.thinking.effort", "medium", "low"),
        ("model.cache_ttl_sec", "1000", "300"),
        ("model.cache_ttl_sec", "3000", "3600"),
        ("model.thinking.budget", "fixed", "none"),
        ("model.service_tier", "flex", "auto"),
    ],
)
async def test_an_inherited_choice_takes_the_closest_this_model_offers(
    name: str,
    given: str,
    taken: str,
) -> None:
    owner, _, _ = _model()

    said = await _inherit(owner, name, given)

    assert said.removesuffix(" (default)") == taken


def _owner(capability: ModelCapability) -> AgentModel:
    """Return the settings of an agent on ``capability``, its only model."""
    catalog = ModelCatalog(
        factories={
            "one": partial(
                ScriptedProvider,
                models=ModelTable(
                    rows=(capability,),
                    roles={"default": capability.model_id},
                ),
            ),
        },
    )
    return AgentModel(
        agent=Agent(model=catalog["one"][capability.model_id]),
        catalog=catalog,
    )


@pytest.mark.asyncio
async def test_an_unordered_choice_this_model_lacks_takes_its_default() -> None:
    """Not its lowest value: the default a fresh model of it starts with."""
    windowed = _windowed("auto-first", 200_000)
    owner = _owner(
        replace(
            windowed,
            thinking_effort=frozenset({"none", "high"}),
            thinking_budget=frozenset({"none", "auto"}),
            default=replace(windowed.default, thinking_budget="auto"),
        ),
    )

    said = await _inherit(owner, "model.thinking.budget", "fixed")

    assert said == "auto (default)"


@pytest.mark.asyncio
async def test_an_inherited_thinking_takes_the_closest_on_each_axis() -> None:
    owner, agent, _ = _model()

    said = await _inherit(owner, "model.thinking", "effort=xhigh budget=fixed")

    assert said == "effort=high budget=none output=none"
    assert agent.model.settings.thinking_effort == "high"


@pytest.mark.asyncio
async def test_a_model_without_reasoning_inherits_none() -> None:
    owner, _, _ = _model("narrow")
    assert (await _inherit(owner, "model.thinking.effort", "max")).startswith("none")


def _signing_in(account: str | None) -> dict[str, ProviderFactory]:
    """Return ``openai-sub``, which keeps named logins, and ``vendor``, which does not."""
    del account
    table = ModelTable(rows=(_WIDE,), roles={"default": "wide"})
    return {
        "openai-sub": partial(ScriptedProvider, models=table),
        "vendor": partial(ScriptedProvider, models=table),
    }


@pytest.mark.asyncio
async def test_account_moves_the_model_onto_it_or_changes_nothing() -> None:
    owner, agent, catalog = _model()
    before = agent.model
    assert await _apply(owner, "model.provider.account", "work") == "work"
    assert (catalog.account, agent.account, agent.model is before) == (
        "work",
        "work",
        False,
    )
    with pytest.raises(ValueError, match="is not a name"):
        await _apply(owner, "model.provider.account", "../x")
    assert catalog.account == "work"


@pytest.mark.asyncio
async def test_an_account_whose_provider_lacks_the_model_changes_nothing() -> None:
    """The account ``spare`` serves no ``wide``; the switch fails, the account stays."""

    def per_account(account: str | None) -> dict[str, ProviderFactory]:
        rows = (_NARROW,) if account == "spare" else (_WIDE, _NARROW)
        table = ModelTable(rows=rows, roles={"default": "narrow"})
        return {"vendor": partial(ScriptedProvider, models=table)}

    catalog = ModelCatalog(factories=per_account)
    agent = Agent(model=catalog["vendor"]["wide"])
    owner = AgentModel(agent=agent, catalog=catalog)
    before = agent.model

    with pytest.raises(ValueError, match="wide"):
        await _apply(owner, "model.provider.account", "spare")

    assert (catalog.account, agent.model) == (None, before)


@pytest.mark.asyncio
async def test_context_takes_a_price_band_or_a_count() -> None:
    owner, agent, _ = _model()
    assert await _apply(owner, "model.context", "+258k") == "258k"
    assert agent.model.settings.context_size.tokens == 258_000
    assert await _apply(owner, "model.context", "smallcontext") == "200k"
    assert agent.model.settings.context_size.tokens == 200_000
    assert await _apply(owner, "model.context", "fullcontext") == "1m (default)"
    assert await _apply(owner, "model.context", "small") == "200k"
    assert await _apply(owner, "model.context", "default") == "1m (default)"
    assert agent.model.settings.context_size.tokens == 1_000_000


@pytest.mark.asyncio
async def test_thinking_sets_its_axes_together_or_not_at_all() -> None:
    owner, agent, _ = _model()
    said = await _apply(owner, "model.thinking", "effort=low budget=auto")
    assert said == "effort=low budget=auto output=none"
    message = "'fixed' is not one of: none, auto"
    with pytest.raises(ValueError, match=f"^budget: {re.escape(message)}$"):
        await _apply(owner, "model.thinking", "effort=max budget=fixed")
    assert agent.model.settings.thinking_effort == "low"
    with pytest.raises(
        ValueError,
        match=r"^give AXIS=VALUE for effort, budget, output",
    ):
        await _apply(owner, "model.thinking", "speed=fast")
    with pytest.raises(
        ValueError,
        match=r"^give AXIS=VALUE for effort, budget, output$",
    ):
        await _apply(owner, "model.thinking", "")


@pytest.mark.asyncio
async def test_an_axis_takes_only_what_the_model_offers() -> None:
    owner, agent, _ = _model()
    assert await _apply(owner, "model.thinking.effort", "MAX") == "max"
    assert await _apply(owner, "model.cache_ttl_sec", "3600") == "3600"
    assert await _apply(owner, "model.manage_context_server_side", "true") == "true"
    assert await _apply(owner, "model.service_tier", "priority") == "priority"
    settings = agent.model.settings
    assert (settings.cache_ttl_sec, settings.manage_context_server_side) == (
        3600.0,
        True,
    )
    message = "'xhigh' is not one of: none, low, high, max"
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        await _apply(owner, "model.thinking.effort", "xhigh")


def test_every_effort_is_a_tag() -> None:
    _, agent, _ = _model()
    model = agent.model
    model.settings = replace(model.settings, thinking_effort="max")
    assert (
        str(
            ModelTags.from_settings(
                model.capability,
                model.settings,
                provider="vendor",
                account="work",
            ),
        )
        == "work@vendor/wide-1.0+fullcontext+maxthink"
    )


def test_model_shows_only_the_choices_that_differ_from_the_default() -> None:
    _, agent, _ = _model()
    model = agent.model
    model.settings = replace(
        model.settings,
        thinking_budget="auto",
        thinking_output="text",
        cache_ttl_sec=3600.0,
        manage_context_server_side=True,
    )
    assert str(
        ModelTags.from_settings(model.capability, model.settings, provider="vendor"),
    ) == (
        "vendor/wide-1.0+fullcontext+highthink budget=auto output=text cache_ttl_sec=3600"
        " manage_context_server_side=true"
    )


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
