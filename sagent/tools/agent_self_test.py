"""Tests for AgentSelf: a patch to the calling agent, planned whole, then applied."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, override

import importlib

import pytest

from sagent.agent.agent import Agent
from sagent.agent.team import Team
from sagent.model.capability import ModelCapability, ModelLimits, ModelSettings
from sagent.model.cost import PriceCatalog
from sagent.model.exceptions import UnknownProviderError
from sagent.model.scripted import ScriptedModel, ScriptedProvider
from sagent.providers.catalog import ModelCatalog
from sagent.providers.table import ModelTable
from sagent.runtime import types
from sagent.tools import agent_self
from sagent.tools.agent_self import AgentSelf
from sagent.tools.types import ToolContext


if TYPE_CHECKING:
    from treekle import PlainTree

    from sagent.providers.types import ProviderFactory, ProviderSharedResources
    from sagent.runtime.types import RuntimeEvent


def _capability(model_id: str) -> ModelCapability:
    prices = PriceCatalog(window=1_000)
    return ModelCapability(
        model_id=model_id,
        limits=ModelLimits(max_request_tokens=1_000, max_response_tokens=100),
        prices=prices,
        thinking_effort=frozenset({"none", "low", "high"}),
        thinking_budget=frozenset({"none", "auto"}),
        thinking_output=frozenset({"none", "text"}),
        default=ModelSettings(context_size=prices["auto", "fullcontext"]),
    )


def _setup(tmp_path: Path) -> tuple[Team, Agent, AgentSelf]:
    catalog = ModelCatalog(
        factories={
            "scripted": partial(ScriptedProvider, capability=_capability("scripted")),
            "other": partial(ScriptedProvider, capability=_capability("other")),
        },
    )
    team = Team(catalog=catalog)
    agent = Agent(model=catalog["scripted"]["scripted"], cwd=tmp_path)
    _ = team.join(agent)
    return team, agent, AgentSelf(team=team)


async def _patch(
    tool: AgentSelf,
    tmp_path: Path,
    arguments: dict[str, PlainTree],
    *,
    label: str = "Agent",
) -> str:
    result = await tool.call(ToolContext(cwd=tmp_path, label=label), arguments)
    return result.text()


@pytest.mark.parametrize(
    ("arguments", "summary"),
    [
        ({}, "AgentSelf"),
        ({"status": "Fixing", "effort": "high"}, "AgentSelf status effort"),
        ({"status": "", "diagnostics": False, "model": None}, "AgentSelf"),
    ],
)
def test_the_summary_names_the_keys_set(
    tmp_path: Path,
    arguments: dict[str, PlainTree],
    summary: str,
) -> None:
    _, _, tool = _setup(tmp_path)

    assert tool.summary(arguments) == summary
    assert tool.serialize_key(ToolContext(cwd=tmp_path), arguments) == "*"


@pytest.mark.asyncio
async def test_a_caller_outside_the_team_is_refused(tmp_path: Path) -> None:
    _, _, tool = _setup(tmp_path)

    with pytest.raises(ValueError, match="needs a calling agent"):
        _ = await _patch(tool, tmp_path, {"status": "x"}, label="nobody")


@pytest.mark.asyncio
async def test_an_empty_patch_changes_nothing(tmp_path: Path) -> None:
    _, agent, tool = _setup(tmp_path)

    assert await _patch(tool, tmp_path, {}) == "No change."
    assert agent.runtime.inbox.drain_nowait() == []


@pytest.mark.asyncio
async def test_a_status_is_published(tmp_path: Path) -> None:
    _, agent, tool = _setup(tmp_path)
    seen: list[RuntimeEvent] = []
    agent.runtime.observers.append(seen.append)

    assert await _patch(tool, tmp_path, {"status": "Fixing the parser"}) == (
        "status: Fixing the parser"
    )
    assert seen == [types.StatusChanged(text="Fixing the parser")]


@pytest.mark.asyncio
async def test_reasoning_and_tier_changes_are_applied_and_reported(
    tmp_path: Path,
) -> None:
    _, agent, tool = _setup(tmp_path)

    text = await _patch(
        tool,
        tmp_path,
        {
            "budget": "auto",
            "output": "text",
            "effort": "high",
            "service_tier": "default",
        },
    )

    settings = agent.model.settings
    assert (
        settings.thinking_budget,
        settings.thinking_output,
        settings.thinking_effort,
        settings.service_tier,
    ) == ("auto", "text", "high", "default")
    assert text.splitlines() == [
        "thinking_budget: none -> auto",
        "thinking_output: none -> text",
        "thinking_effort: none -> high",
        "service_tier: auto -> default",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "arguments",
    [
        {"effort": "high", "service_tier": "priority"},
        {"effort": "high", "budget": "fixed"},
        {"effort": "high", "output": "redacted"},
        {"provider": "other", "effort": "max"},
    ],
    ids=["tier", "budget", "output", "effort-with-switch"],
)
async def test_a_patch_with_any_setting_not_offered_changes_nothing(
    tmp_path: Path,
    arguments: dict[str, PlainTree],
) -> None:
    _, agent, tool = _setup(tmp_path)
    model = agent.model

    with pytest.raises(ValueError, match="not offered"):
        _ = await _patch(tool, tmp_path, arguments)

    assert agent.model is model
    assert model.settings.thinking_effort == "none"


@pytest.mark.asyncio
async def test_a_model_switch_keeps_the_settings_and_says_what_changed(
    tmp_path: Path,
) -> None:
    _, agent, tool = _setup(tmp_path)
    agent.model.settings = replace(agent.model.settings, thinking_effort="low")

    text = await _patch(tool, tmp_path, {"provider": "other", "model": "other"})

    assert text == "model: scripted/scripted+fullcontext -> other/other+fullcontext"
    assert agent.model.capability.model_id == "other"
    assert agent.model.settings.thinking_effort == "low"


@pytest.mark.asyncio
async def test_a_switch_with_settings_applies_them_to_the_new_model(
    tmp_path: Path,
) -> None:
    _, agent, tool = _setup(tmp_path)
    before = agent.model

    text = await _patch(tool, tmp_path, {"model": "scripted", "effort": "high"})

    assert text.splitlines() == [
        "model: scripted/scripted+fullcontext -> scripted/scripted+fullcontext",
        "thinking_effort: none -> high",
    ]
    assert agent.model is not before
    assert agent.model.settings.thinking_effort == "high"
    assert before.settings.thinking_effort == "none"


def _windowed(model_id: str) -> ModelTable:
    """Return a table serving ``model_id`` with a 1M window."""
    prices = PriceCatalog(window=1_000_000)
    return ModelTable(
        rows=(
            ModelCapability(
                model_id=model_id,
                limits=ModelLimits(max_request_tokens=1_000_000),
                prices=prices,
                default=ModelSettings(context_size=prices["auto", "fullcontext"]),
            ),
        ),
    )


@pytest.mark.asyncio
async def test_a_switch_finds_who_serves_the_model_and_keeps_the_window(
    tmp_path: Path,
) -> None:
    catalog = ModelCatalog(
        factories={
            "first": partial(ScriptedProvider, models=_windowed("a-1.0")),
            "second": partial(ScriptedProvider, models=_windowed("b-1.0")),
        },
    )
    team = Team(catalog=catalog)
    agent = Agent(model=catalog["first"]["a+200k"], cwd=tmp_path)
    _ = team.join(agent)

    text = await _patch(AgentSelf(team=team), tmp_path, {"model": "b"})

    assert text == "model: first/a-1.0+200k -> second/b-1.0+200k"
    assert agent.model.settings.context_size.tokens == 200_000


@pytest.mark.asyncio
async def test_a_switch_signs_in_as_the_catalogs_account(tmp_path: Path) -> None:
    def per_account(account: str | None) -> dict[str, ProviderFactory]:
        del account
        return {"first": partial(ScriptedProvider, models=_windowed("a-1.0"))}

    team = Team(catalog=ModelCatalog(factories=per_account))
    agent = Agent(model=team.catalog["first"]["a"], cwd=tmp_path)
    _ = team.join(agent)
    team.catalog.use_account("work")

    _ = await _patch(AgentSelf(team=team), tmp_path, {"model": "a+200k"})

    assert agent.account == "work"
    assert agent.model.provider is team.catalog.signed_in("first", account="work")


@pytest.mark.asyncio
async def test_an_unknown_provider_is_refused(tmp_path: Path) -> None:
    _, agent, tool = _setup(tmp_path)
    model = agent.model

    with pytest.raises(
        UnknownProviderError,
        match=r"^unknown provider 'nope'; known: scripted, other$",
    ):
        _ = await _patch(tool, tmp_path, {"provider": "nope"})
    assert agent.model is model


@pytest.mark.asyncio
async def test_diagnostics_report_what_the_agent_runs_on(tmp_path: Path) -> None:
    _, _, tool = _setup(tmp_path)

    assert (await _patch(tool, tmp_path, {"diagnostics": True})).splitlines() == [
        "label: Agent",
        "model: scripted/scripted+fullcontext",
        "context: 1k",
        "thinking: budget=none output=none effort=none",
        "service_tier: auto; cache_ttl_sec: 0",
        "limits: request=1000 response=100",
        f"cwd: {tmp_path}",
        "tools: ",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("arguments", "line"),
    [
        ({"catalog": "providers"}, "providers: scripted, other"),
        ({"catalog": "models"}, "scripted models: scripted"),
        ({"catalog": "models", "catalog_provider": "other"}, "other models: other"),
    ],
)
async def test_the_catalog_lists_providers_or_one_providers_models(
    tmp_path: Path,
    arguments: dict[str, PlainTree],
    line: str,
) -> None:
    _, _, tool = _setup(tmp_path)

    assert await _patch(tool, tmp_path, arguments) == line


@pytest.mark.asyncio
async def test_a_refused_switch_closes_the_model_it_built(tmp_path: Path) -> None:
    team, _, tool = _setup(tmp_path)
    built: list[ScriptedModel] = []
    closed: list[ScriptedModel] = []
    team.catalog = ModelCatalog(
        factories={"other": partial(_Recording, built=built, closed=closed)},
    )

    with pytest.raises(ValueError, match="not offered"):
        _ = await _patch(tool, tmp_path, {"provider": "other", "effort": "max"})

    assert closed == built
    assert len(built) == 1


class _Recording(ScriptedProvider):
    """A provider whose models note when they are built and closed."""

    def __init__(
        self,
        *,
        built: list[ScriptedModel],
        closed: list[ScriptedModel],
        name: str = "",
        resources: ProviderSharedResources | None = None,
    ) -> None:
        super().__init__(
            name=name,
            resources=resources,
            capability=_capability("other"),
        )
        self.built = built
        self.closed = closed

    @override
    def __getitem__(self, model: str) -> ScriptedModel:
        del model
        made = _Closing(
            provider=self,
            capability=self.capability,
            settings=self.capability.default,
            closed=self.closed,
        )
        self.built.append(made)
        return made


class _Closing(ScriptedModel):
    """A scripted model that notes its close."""

    def __init__(
        self,
        *,
        provider: ScriptedProvider,
        capability: ModelCapability,
        settings: ModelSettings,
        closed: list[ScriptedModel],
    ) -> None:
        super().__init__(provider=provider, capability=capability, settings=settings)
        self._closed = closed

    @override
    async def aclose(self) -> None:
        self._closed.append(self)


def test_the_description_reads_as_utf8_whatever_the_locale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asked = _Asked()
    with monkeypatch.context() as patched:
        patched.setattr(Path, "read_text", asked)
        _ = importlib.reload(agent_self)
    _ = importlib.reload(agent_self)

    assert asked.encodings["agentself.md"] == "utf-8"


@dataclass(slots=True, kw_only=True)
class _Asked:
    """Read text as UTF-8, noting the encoding each read asked for."""

    encodings: dict[str, str | None] = field(default_factory=dict)

    def __get__(self, path: Path, owner: type[Path] | None = None) -> partial[str]:
        """Bind to the path read, as the method it stands in for would."""
        del owner
        return partial(self, path)

    def __call__(
        self,
        path: Path,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
    ) -> str:
        del errors, newline
        self.encodings[path.name] = encoding
        return path.read_bytes().decode()


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
