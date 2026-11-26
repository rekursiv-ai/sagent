"""Tests for AgentSelf: a patch to the calling agent, planned whole, then applied."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, override
from unittest.mock import Mock

import math
import re
import runpy

import pytest

from sagent.agent.agent import Agent
from sagent.agent.checkpoints import pending_text
from sagent.agent.team import Team
from sagent.journal.jsonl import JsonlJournal
from sagent.journal.types import AgentSelfResult
from sagent.model.capability import ModelCapability, ModelLimits, ModelSettings
from sagent.model.cost import PriceCatalog, PriceCondition, TokenPrice
from sagent.model.exceptions import UnknownProviderError
from sagent.model.scripted import ScriptedModel, ScriptedProvider, call, reply
from sagent.providers.catalog import ModelCatalog
from sagent.providers.table import ModelTable
from sagent.runtime import types
from sagent.runtime.types import ToolWakeSchedules, WakeSchedule
from sagent.tools import agent_self
from sagent.tools.agent_self import AgentSelf
from sagent.tools.agent_send import AgentSend
from sagent.tools.types import ToolContext, settled


if TYPE_CHECKING:
    from treekle import PlainTree

    from sagent.providers.types import ProviderFactory, ProviderResources
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
        default=ModelSettings(context_size=prices["fullcontext"]),
    )


def _setup(tmp_path: Path) -> tuple[Team, Agent, AgentSelf]:
    catalog = ModelCatalog(
        factories={
            "scripted": partial(ScriptedProvider, capability=_capability("scripted")),
            "other": partial(ScriptedProvider, capability=_capability("other")),
        },
    )
    team = Team(catalog=catalog)
    agent = Agent(
        model=catalog["scripted"]["scripted"],
        tool_context=ToolContext(cwd=tmp_path),
    )
    _ = team.join(agent)
    return team, agent, AgentSelf(team=team)


async def _patch(
    tool: AgentSelf,
    tmp_path: Path,
    arguments: dict[str, PlainTree],
    *,
    label: str = "Agent",
) -> str:
    streaming_result = tool.call(ToolContext(cwd=tmp_path, label=label), arguments)
    result = await settled(streaming_result)
    # ``settled`` records a raise instead of raising; a refusal test expects the raise.
    _ = await streaming_result.result
    assert result.result == AgentSelfResult()
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
    tool = AgentSelf(team=Mock(spec=Team))

    assert tool.summary(arguments) == summary
    assert tool.serialize_key(ToolContext(cwd=tmp_path), arguments) == "*"


@pytest.mark.asyncio
async def test_a_caller_outside_the_team_is_refused(tmp_path: Path) -> None:
    _, _, tool = _setup(tmp_path)

    with pytest.raises(ValueError, match=r"^AgentSelf needs a calling agent"):
        _ = await _patch(tool, tmp_path, {"status": "x"}, label="nobody")


@pytest.mark.asyncio
async def test_an_empty_patch_changes_nothing(tmp_path: Path) -> None:
    _, agent, tool = _setup(tmp_path)

    assert await _patch(tool, tmp_path, {}) == "No change."
    assert agent.runtime.inbox.drain_nowait() == []


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [False, True])
async def test_peer_participation_is_applied_and_reported(
    tmp_path: Path,
    enabled: bool,
) -> None:
    team, _, tool = _setup(tmp_path)
    team.set_peer_messaging("Agent", not enabled)
    assert await _patch(tool, tmp_path, {"peer_messaging": enabled}) == (
        f"peer_messaging: {'on' if enabled else 'off'}"
    )
    assert team.peer_messaging("Agent") is enabled


@pytest.mark.asyncio
async def test_a_status_is_published(tmp_path: Path) -> None:
    _, agent, tool = _setup(tmp_path)
    seen: list[RuntimeEvent] = []
    agent.runtime.observers.append(seen.append)

    assert await _patch(tool, tmp_path, {"status": "Fixing the parser"}) == (
        "status: Fixing the parser"
    )
    assert seen == [types.SessionTitleChanged(title="Fixing the parser")]


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
async def test_a_compaction_mode_the_model_offers_is_applied_and_reported(
    tmp_path: Path,
) -> None:
    capability = replace(
        _capability("scripted"),
        compaction=frozenset({"bespoke", "provider"}),
    )
    catalog = ModelCatalog(
        factories={"scripted": partial(ScriptedProvider, capability=capability)},
    )
    team = Team(catalog=catalog)
    agent = Agent(
        model=catalog["scripted"]["scripted"],
        tool_context=ToolContext(cwd=tmp_path),
    )
    _ = team.join(agent)

    text = await _patch(AgentSelf(team=team), tmp_path, {"compaction": "provider"})

    assert agent.model.settings.compaction == "provider"
    assert text == "compaction: bespoke -> provider"


@pytest.mark.asyncio
async def test_an_effort_too_high_for_thinking_off_turns_thinking_on(
    tmp_path: Path,
) -> None:
    capped = replace(
        _capability("capped"),
        thinking_off_max_effort="low",
        default=replace(_capability("capped").default, thinking_budget="auto"),
    )
    catalog = ModelCatalog(
        factories={"scripted": partial(ScriptedProvider, capability=capped)},
    )
    team = Team(catalog=catalog)
    agent = Agent(
        model=catalog["scripted"]["capped"],
        tool_context=ToolContext(cwd=tmp_path),
    )
    _ = team.join(agent)
    tool = AgentSelf(team=team)
    await _patch(tool, tmp_path, {"budget": "none", "effort": "low"})

    text = await _patch(tool, tmp_path, {"effort": "high"})

    assert text.splitlines() == [
        "thinking_budget: none -> auto",
        "thinking_effort: low -> high",
    ]
    with pytest.raises(ValueError, match="allowed: thinking_effort up to 'low'"):
        _ = await _patch(tool, tmp_path, {"effort": "high", "budget": "none"})


@pytest.mark.asyncio
async def test_when_long_calls_wake_is_patched_and_reported(tmp_path: Path) -> None:
    _, agent, tool = _setup(tmp_path)

    text = await _patch(
        tool,
        tmp_path,
        {
            "detach": {
                "default": {"first_gap_sec": 30, "max_gap_sec": 600},
                "per_tool": {"Bash": {"off": True}, "Read": {"gap_growth": 0}},
            },
        },
    )

    schedule = agent.runtime.wake_schedules
    assert schedule == ToolWakeSchedules(
        default=WakeSchedule(first_gap_sec=30.0, max_gap_sec=600.0),
        per_tool={
            "Bash": None,
            "Read": WakeSchedule(first_gap_sec=30.0, gap_growth=0.0, max_gap_sec=600.0),
        },
    )
    assert text == "detach: 30s 1 600s; Bash off; Read 30s 0 600s"
    assert await _patch(tool, tmp_path, {"detach": {"default": {"off": True}}}) == (
        "detach: off; Bash off; Read 30s 0 600s"
    )
    # A patch to a schedule that is off starts from the default one.
    assert (
        await _patch(
            tool,
            tmp_path,
            {"detach": {"per_tool": {"Bash": {"first_gap_sec": 5}}}},
        )
        == "detach: off; Bash 5s 1 300s; Read 30s 0 600s"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("detach", "message"),
    [
        (
            {"default": {"first_gap_sec": 0}},
            "first_gap_sec must be a number of seconds",
        ),
        (
            {"per_tool": {"Bash": {"soon": 1}}},
            "detach.per_tool.Bash takes first_gap_sec, gap_growth, max_gap_sec, off; not soon.",
        ),
        ({"default": {"later": 1}}, "detach.default takes .*; not later."),
        ({"later": {}}, "detach takes default, per_tool; not later."),
        ({"default": {"gap_growth": "fast"}}, "gap_growth"),
    ],
)
async def test_a_detach_patch_that_does_not_fit_changes_nothing(
    tmp_path: Path,
    detach: dict[str, PlainTree],
    message: str,
) -> None:
    _, agent, tool = _setup(tmp_path)

    with pytest.raises(ValueError, match=message):
        _ = await _patch(tool, tmp_path, {"detach": detach, "status": "x"})

    assert agent.runtime.wake_schedules == ToolWakeSchedules()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "arguments",
    [
        {"effort": "high", "service_tier": "priority"},
        {"effort": "high", "budget": "fixed"},
        {"effort": "high", "output": "redacted"},
        {"provider": "other", "effort": "max"},
        {"effort": "high", "compaction": "provider"},
    ],
    ids=["tier", "budget", "output", "effort-with-switch", "compaction"],
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


@pytest.mark.asyncio
async def test_a_tier_priced_flat_keeps_the_size_another_tier_bands(
    tmp_path: Path,
) -> None:
    prices = PriceCatalog(
        prices={
            PriceCondition(token_size_range=(0, 200_001)): TokenPrice(request=1.0),
            PriceCondition(token_size_range=(200_001, math.inf)): TokenPrice(),
            PriceCondition(service_tier="priority"): TokenPrice(request=2.0),
        },
        window=1_000_000,
    )
    capability = ModelCapability(
        model_id="banded",
        limits=ModelLimits(max_request_tokens=1_000_000),
        prices=prices,
        service_tier=frozenset({"auto", "priority"}),
        default=ModelSettings(context_size=prices["smallcontext"]),
    )
    catalog = ModelCatalog(
        factories={"scripted": partial(ScriptedProvider, capability=capability)},
    )
    team = Team(catalog=catalog)
    agent = Agent(
        model=catalog["scripted"]["banded"],
        tool_context=ToolContext(cwd=tmp_path),
    )
    _ = team.join(agent)

    _ = await _patch(AgentSelf(team=team), tmp_path, {"service_tier": "priority"})

    assert agent.model.settings.service_tier == "priority"
    assert agent.model.settings.context_size == prices["smallcontext"]


def _windowed(model_id: str) -> ModelTable:
    """Return a table serving ``model_id`` with a 1M window."""
    prices = PriceCatalog(window=1_000_000)
    return ModelTable(
        rows=(
            ModelCapability(
                model_id=model_id,
                limits=ModelLimits(max_request_tokens=1_000_000),
                prices=prices,
                default=ModelSettings(context_size=prices["fullcontext"]),
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
    agent = Agent(
        model=catalog["first"]["a+200k"],
        tool_context=ToolContext(cwd=tmp_path),
    )
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
    agent = Agent(
        model=team.catalog["first"]["a"],
        tool_context=ToolContext(cwd=tmp_path),
    )
    _ = team.join(agent)
    team.catalog.use_account("work")

    _ = await _patch(AgentSelf(team=team), tmp_path, {"model": "a+200k"})

    assert agent.account == "work"
    assert agent.model.provider is team.catalog.signed_in("first", account="work")


@pytest.mark.asyncio
async def test_a_provider_without_the_model_serves_the_one_in_its_role(
    tmp_path: Path,
) -> None:
    catalog = ModelCatalog(
        factories={
            "anthropic": partial(
                ScriptedProvider,
                models=ModelTable(
                    rows=(_capability("opus-5.5"),),
                    roles={"default": "opus"},
                ),
            ),
            "openai": partial(
                ScriptedProvider,
                models=ModelTable(
                    rows=(_capability("sol-6.2"),),
                    roles={"default": "sol"},
                ),
            ),
        },
    )
    team = Team(catalog=catalog)
    agent = Agent(
        model=catalog["anthropic"]["opus"],
        tool_context=ToolContext(cwd=tmp_path),
    )
    _ = team.join(agent)

    _ = await _patch(AgentSelf(team=team), tmp_path, {"provider": "openai"})

    assert agent.model.provider is catalog["openai"]
    assert agent.model.capability.model_id == "sol-6.2"


@pytest.mark.asyncio
async def test_switching_back_restores_what_the_model_was_left_with(
    tmp_path: Path,
) -> None:
    _, agent, tool = _setup(tmp_path)
    _ = await _patch(tool, tmp_path, {"effort": "low"})
    _ = await _patch(tool, tmp_path, {"provider": "other", "effort": "high"})

    _ = await _patch(tool, tmp_path, {"provider": "scripted"})

    assert agent.model.settings.thinking_effort == "low"


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
        "compaction: bespoke",
        "limits: request=1000 response=100",
        f"cwd: {tmp_path}",
        "tools: ",
        "offered_tools: ",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("allowed", "callable_tools"),
    [
        (None, "AgentSelf, AgentSend"),
        (frozenset[str](), ""),
        (frozenset({"AgentSend"}), "AgentSend"),
    ],
)
@pytest.mark.parametrize("capped", [False, True])
async def test_diagnostics_distinguishes_callable_tools_from_offered_schemas(
    tmp_path: Path,
    allowed: frozenset[str] | None,
    callable_tools: str,
    capped: bool,
) -> None:
    team, root, tool = _setup(tmp_path)
    child = Agent(
        model=ScriptedProvider()["scripted"],
        tools=[tool, AgentSend(team=team)],
        allowed=allowed,
        environment=False,
    )
    team.join(child, label="restricted", parent="Agent")
    if capped:
        child.max_rounds = 1
        child._charge(reply(calls=[call("AgentSelf")]))
        callable_tools = (
            "AgentSend" if allowed is None or "AgentSend" in allowed else ""
        )
    result = await settled(
        tool.call(
            root.runtime.tool_context,
            {"agent": "restricted", "diagnostics": True},
        ),
    )
    lines = result.text().splitlines()
    assert "offered_tools: AgentSelf, AgentSend" in lines
    assert f"tools: {callable_tools}" in lines


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
async def test_a_provider_s_models_are_listed_comma_separated(tmp_path: Path) -> None:
    rows = (_capability("first-1.0"), _capability("second-1.0"))
    catalog = ModelCatalog(
        factories={
            "scripted": partial(ScriptedProvider, models=ModelTable(rows=rows)),
        },
    )
    team = Team(catalog=catalog)
    agent = Agent(
        model=catalog["scripted"]["first"],
        tool_context=ToolContext(cwd=tmp_path),
    )
    _ = team.join(agent)

    text = await _patch(AgentSelf(team=team), tmp_path, {"catalog": "models"})

    assert text == "scripted models: first-1.0, second-1.0"


@pytest.mark.asyncio
async def test_a_patch_keeps_each_knob_it_leaves_out(tmp_path: Path) -> None:
    """The default's own knobs, and a tool's own, not the schedule's defaults."""
    _, _, tool = _setup(tmp_path)
    _ = await _patch(
        tool,
        tmp_path,
        {
            "detach": {
                "default": {"first_gap_sec": 30},
                "per_tool": {"Bash": {"first_gap_sec": 5}},
            },
        },
    )

    text = await _patch(
        tool,
        tmp_path,
        {
            "detach": {
                "default": {"gap_growth": 2},
                "per_tool": {"Bash": {"gap_growth": 3}},
            },
        },
    )

    assert text == "detach: 30s 2 300s; Bash 5s 3 300s"


@pytest.mark.asyncio
async def test_every_unknown_knob_is_named(tmp_path: Path) -> None:
    _, _, tool = _setup(tmp_path)
    message = "detach.default takes first_gap_sec, gap_growth, max_gap_sec, off; not later, soon."
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        _ = await _patch(
            tool,
            tmp_path,
            {"detach": {"default": {"soon": 1, "later": 2}}},
        )


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

    # The switch built one model; finding its role built and closed one a role.
    assert closed == built
    assert built


class _Recording(ScriptedProvider):
    """A provider whose models note when they are built and closed."""

    def __init__(
        self,
        *,
        built: list[ScriptedModel],
        closed: list[ScriptedModel],
        name: str = "",
        resources: ProviderResources | None = None,
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
        _ = runpy.run_path(
            agent_self.__file__ or "",
            run_name=f"{agent_self.__name__}_fresh",
        )

    assert asked.encodings["agentself.md"] == "utf-8"
    assert agent_self.AgentSelf is AgentSelf


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


@pytest.mark.asyncio
async def test_checkpoint_survives_resume_and_clears(tmp_path: Path) -> None:
    path = tmp_path / "session.jsonl"
    catalog = ModelCatalog(factories={"scripted": ScriptedProvider})
    data: dict[str, PlainTree] = {
        "progress": "Fetched files",
        "pending": "Review",
        "user_action": "Choose version",
    }
    async with (
        JsonlJournal(path) as journal,
        Team(catalog=catalog) as team,
        Agent(model=catalog["scripted"]["scripted"], journal=journal) as root,
    ):
        team.join(root)
        worker = Agent(model=catalog["scripted"]["scripted"])
        team.join(worker, label="worker", parent="Agent")
        tool = AgentSelf(team=team)
        await _patch(tool, tmp_path, {"checkpoint": data}, label="worker")
        count = len(list(root.session.history()))
        await _patch(tool, tmp_path, {"checkpoint": data}, label="worker")
        assert len(list(root.session.history())) == count
    async with (
        JsonlJournal(path) as journal,
        Team(catalog=catalog) as team,
        Agent(model=catalog["scripted"]["scripted"], journal=journal) as root,
    ):
        team.join(root)
        assert (
            pending_text(root.session, brief=True)
            == "worker: Action needed: Choose version"
        )
        assert "Waiting: Review" in pending_text(root.session)
        worker = Agent(model=catalog["scripted"]["scripted"])
        team.join(worker, label="worker", parent="Agent")
        await _patch(
            AgentSelf(team=team),
            tmp_path,
            {"checkpoint": None},
            label="worker",
        )
        assert pending_text(root.session) == ""


@pytest.mark.asyncio
async def test_invalid_checkpoint_rejects_the_whole_patch(tmp_path: Path) -> None:
    _, agent, tool = _setup(tmp_path)
    seen: list[RuntimeEvent] = []
    agent.runtime.observers.append(seen.append)
    with pytest.raises(ValueError, match="progress must contain text"):
        await _patch(
            tool,
            tmp_path,
            {
                "status": "New title",
                "checkpoint": {"progress": " ", "pending": "", "user_action": ""},
            },
        )
    assert seen == []


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
