"""Tests for AgentSpawn: children that run a task, or stay, and come back on resume."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, override

import asyncio
import importlib
import logging
import uuid

from treekle import PlainTree

import pytest

from sagent.agent.agent import Agent, NoShrinker, RuntimeSession, spent
from sagent.agent.agents_md import AgentsMdConfig
from sagent.agent.compactor import SummaryCompactor
from sagent.agent.team import ChildSpec, Team
from sagent.model.capability import ModelCapability, ModelLimits, ModelSettings
from sagent.model.cost import PriceCatalog, TokenCost, TokenCount
from sagent.model.scripted import (
    Reply,
    ScriptedModel,
    ScriptedProvider,
    call,
    reply,
)
from sagent.providers.catalog import CachedModel, ModelCatalog
from sagent.providers.table import ModelTable
from sagent.runtime import types
from sagent.runtime.exceptions import BudgetExhaustedError
from sagent.runtime.types import RuntimeEvent
from sagent.session.jsonl import JsonlJournal
from sagent.session.projects import child_journal
from sagent.session.types import (
    AgentStatusResult,
    AgentToAgentMessage,
    SystemMessage,
    UncategorizedRecord,
    UncategorizedToolResult,
    UserMessage,
)
from sagent.tools import agent_spawn
from sagent.tools.agent_send import AgentSend
from sagent.tools.agent_spawn import AgentSpawn
from sagent.tools.display import ToolDisplay
from sagent.tools.types import ToolContext


if TYPE_CHECKING:
    from sagent.agent.types import Factory, Journal, Shrinker
    from sagent.providers.types import ProviderSharedResources


@pytest.fixture(autouse=True)
def isolate_git(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the children's git probe off the user's config and above ``tmp_path``."""
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / "gitconfig"))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))


def _capability(model_id: str) -> ModelCapability:
    prices = PriceCatalog(window=200_000)
    return ModelCapability(
        model_id=model_id,
        limits=ModelLimits(max_request_tokens=200_000, max_response_tokens=8_000),
        prices=prices,
        thinking_effort=frozenset({"none", "low", "high"}),
        thinking_budget=frozenset({"none", "auto"}),
        thinking_output=frozenset({"none", "text"}),
        default=ModelSettings(context_size=prices["auto", "fullcontext"]),
    )


@dataclass(slots=True, kw_only=True)
class _Tool:
    """A tool for the parent to hand down."""

    name: str
    description: str = "A tool."
    schema: Mapping[str, PlainTree] = field(
        default_factory=lambda: {
            "type": "object",
            "properties": dict[str, PlainTree](),
        },
    )
    display: ToolDisplay = field(default_factory=ToolDisplay)

    def summary(self, arguments: Mapping[str, PlainTree]) -> str:
        del arguments
        return self.name

    def serialize_key(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> str | None:
        del context, arguments
        return None

    async def call(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> UncategorizedToolResult:
        del context, arguments
        return UncategorizedToolResult(call_id="", content="ok")


@dataclass(slots=True, kw_only=True)
class _Wait(_Tool):
    """A tool that runs until cancelled."""

    started: asyncio.Event = field(default_factory=asyncio.Event)
    stopped: asyncio.Event = field(default_factory=asyncio.Event)

    @override
    async def call(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> UncategorizedToolResult:
        del context, arguments
        self.started.set()
        try:
            await asyncio.Event().wait()
        finally:
            self.stopped.set()
        return UncategorizedToolResult(call_id="", content="never")


@dataclass(slots=True, kw_only=True)
class _Seen:
    """What an agent's runtime publishes, and a way to wait for more of it."""

    events: list[types.RuntimeEvent] = field(default_factory=list)
    arrived: asyncio.Event = field(default_factory=asyncio.Event)

    def __call__(self, event: types.RuntimeEvent) -> None:
        self.events.append(event)
        self.arrived.set()

    def count(self, kind: type[object]) -> int:
        return sum(isinstance(e, kind) for e in self.events)

    def answered(self) -> bool:
        """Return whether the agent went idle after its last model reply."""
        replies = [
            i
            for i, e in enumerate(self.events)
            if isinstance(e, types.ModelResponseComplete)
        ]
        return bool(replies) and any(
            isinstance(e, types.AgentIdle) for e in self.events[replies[-1] :]
        )

    async def until(self, done: Callable[[], bool]) -> None:
        """Wait, an event at a time, until ``done``."""
        async with asyncio.timeout(5):
            while not done():
                self.arrived.clear()
                await self.arrived.wait()

    async def idle(self) -> None:
        """Wait until the agent goes idle once more."""
        idles = self.count(types.AgentIdle)
        await self.until(lambda: self.count(types.AgentIdle) > idles)


@dataclass(slots=True, kw_only=True)
class _Setup:
    """A team whose root may spawn, and the script its children answer from."""

    team: Team
    parent: Agent
    children: ScriptedProvider
    spawn: AgentSpawn
    seen: list[RuntimeEvent]


def _setup(
    tmp_path: Path,
    *replies: Reply,
    journal: Journal | None = None,
    compactor: Factory[Shrinker] = NoShrinker,
    tools: Sequence[_Tool] = (),
    other: Sequence[Reply] = (),
    send: bool = False,
) -> _Setup:
    catalog = ModelCatalog(
        factories={
            "scripted": partial(
                ScriptedProvider,
                replies=replies,
                capability=_capability("scripted"),
            ),
            "other": partial(
                ScriptedProvider,
                replies=other,
                capability=_capability("other"),
            ),
        },
    )
    team = Team(catalog=catalog)
    parent = Agent(
        model=ScriptedProvider(capability=_capability("scripted"))["scripted"],
        tools=[
            _Tool(name="Read"),
            _Tool(name="Grep"),
            *tools,
            *([AgentSend(team=team)] if send else []),
        ],
        system_prompt="the parent's prompt",
        journal=journal,
        compactor=compactor,
        cwd=tmp_path,
        agents_md=AgentsMdConfig(system_dir=tmp_path / "s", user_dir=tmp_path / "u"),
    )
    _ = team.join(parent)
    seen: list[RuntimeEvent] = []
    parent.runtime.observers.append(seen.append)
    children = catalog["scripted"]
    assert isinstance(children, ScriptedProvider)
    return _Setup(
        team=team,
        parent=parent,
        children=children,
        spawn=AgentSpawn(team=team),
        seen=seen,
    )


async def _spawn(
    setup: _Setup,
    tmp_path: Path,
    arguments: dict[str, PlainTree],
    *,
    label: str = "Agent",
) -> AgentStatusResult:
    return await setup.spawn.call(ToolContext(cwd=tmp_path, label=label), arguments)


def _session(agent: Agent) -> RuntimeSession:
    session = agent.session
    assert isinstance(session, RuntimeSession)
    return session


async def _messages(agent: Agent, count: int) -> list[AgentToAgentMessage]:
    """Wait, a turn of the loop at a time, until ``count`` messages reach ``agent``."""
    got: list[AgentToAgentMessage] = []
    async with asyncio.timeout(5):
        while len(got) < count:
            got.extend(
                e
                for e in agent.runtime.inbox.drain_nowait()
                if isinstance(e, AgentToAgentMessage)
            )
            await asyncio.sleep(0)
    return got


@pytest.mark.parametrize(
    ("arguments", "summary"),
    [
        ({"prompt": "list   the\nTODOs"}, "AgentSpawn: list the TODOs"),
        (
            {"prompt": "go", "label": "r"},
            "AgentSpawn r: go",
        ),
        ({}, "AgentSpawn"),
    ],
)
def test_the_summary_says_what_kind_of_child_and_its_task(
    tmp_path: Path,
    arguments: dict[str, PlainTree],
    summary: str,
) -> None:
    setup = _setup(tmp_path)

    assert setup.spawn.summary(arguments) == summary
    assert setup.spawn.serialize_key(ToolContext(cwd=tmp_path), arguments) is None


def test_the_reminder_lists_the_children_who_owes_whom_and_a_leaf(
    tmp_path: Path,
) -> None:
    setup = _setup(tmp_path)
    for label in ("leaf", "asked", "asker"):
        _ = setup.team.join(
            Agent(model=ScriptedProvider()["scripted"], cwd=tmp_path),
            label=label,
            parent="Agent",
            max_depth=0,
        )
    setup.team.send("Agent", "asked", "how goes it?")
    setup.team.send("asker", "Agent", "may I?")

    def said(label: str) -> str:
        return setup.spawn.reminder(ToolContext(cwd=tmp_path, label=label))

    assert said("Agent") == (
        "Your children: leaf, asked (owes you a reply), asker (you owe it a reply)."
    )
    assert said("stranger") == ""
    assert said("leaf") == "Spawn budget: you are a leaf agent; AgentSpawn will refuse."


@pytest.mark.asyncio
async def test_a_child_spends_from_its_parents_budget_which_halts_past_it(
    tmp_path: Path,
) -> None:
    async with JsonlJournal(tmp_path / "parent" / "journal.jsonl") as journal:
        setup = _setup(
            tmp_path,
            reply("looking", calls=[call("Read", "c1")], spend=TokenCost(response=1.2)),
            reply("never asked"),
            journal=journal,
        )
        _session(setup.parent).resume()
        setup.parent.max_spend_usd = 1.0

        result = await _spawn(setup, tmp_path, {"prompt": "spend"})

        assert (result.state, result.text()) == (
            "failed",
            "BudgetExhaustedError: Budget exhausted: $1.20 >= $1.00",
        )
        assert setup.parent.spend_usd == pytest.approx(1.2)
        assert spent(journal) == pytest.approx(1.2)
        with pytest.raises(BudgetExhaustedError):
            _ = await setup.parent.send("more")


@pytest.mark.asyncio
async def test_a_childs_compaction_spends_from_its_parent(tmp_path: Path) -> None:
    setup = _setup(
        tmp_path,
        reply("ready"),
        reply("<summary>gist</summary>", spend=TokenCost(response=2.0)),
        compactor=SummaryCompactor,
    )
    async with setup.team:
        _ = await _spawn(
            setup,
            tmp_path,
            {"prompt": "wait"},
        )
        child = setup.team["Agent_0"]
        await child.session.compactor.run()

    assert child.spend_usd == pytest.approx(2.0)
    assert setup.parent.spend_usd == pytest.approx(2.0)


@pytest.mark.asyncio
async def test_a_child_tells_its_parent_when_it_stops_but_not_after_its_spawn(
    tmp_path: Path,
) -> None:
    """The spawn's result is the parent's news of the first run."""
    setup = _setup(tmp_path, reply("ready"), reply("did more"))
    async with setup.team:
        _ = await _spawn(setup, tmp_path, {"prompt": "wait", "label": "helper"})
        after_spawn = setup.parent.runtime.inbox.drain_nowait()
        child = setup.team["helper"]
        child.runtime.push(UserMessage(content="more"))

        (told,) = await _messages(setup.parent, 1)

    assert after_spawn == []
    assert told == AgentToAgentMessage(
        sender="helper",
        recipient="Agent",
        content="[helper is idle] did more",
    )
    assert setup.team.awaited == set()
    # The spawn's count of its run stops with the run.
    assert not any(isinstance(o, agent_spawn._Tally) for o in child.runtime.observers)


@pytest.mark.asyncio
async def test_a_child_runs_its_task_answers_and_stays(tmp_path: Path) -> None:
    setup = _setup(
        tmp_path,
        reply(
            "looking",
            calls=[call("Read", "c1")],
            tokens=TokenCount(request=10),
            spend=TokenCost(response=0.5),
        ),
        reply("three TODOs", tokens=TokenCount(request=5)),
    )

    result = await _spawn(setup, tmp_path, {"prompt": "count the TODOs"})

    assert result.is_error is False
    assert (result.agent_id, result.agent_kind, result.state) == (
        "Agent_0",
        "persistent",
        "completed",
    )
    assert (result.content, result.prompt, result.model) == (
        "three TODOs",
        "count the TODOs",
        "scripted",
    )
    assert (result.tokens, result.tool_calls, result.output_file) == (15, 1, None)
    # The runtime stamps the call's id; a tool leaves it empty.
    assert result.call_id == ""
    assert result.duration_sec is not None
    assert 0.0 <= result.duration_sec < 60.0
    assert list(setup.team) == ["Agent", "Agent_0"]
    (done,) = [e for e in setup.seen if isinstance(e, types.ChildDoneEvent)]
    assert (done.label, done.tokens, done.cost) == ("Agent_0", 15, 0.5)
    assert 0.0 <= done.elapsed <= result.duration_sec
    forwarded = [e for e in setup.seen if isinstance(e, types.ChildEvent)]
    assert {e.label for e in forwarded} == {"Agent_0"}
    assert [type(e.inner) for e in forwarded] == [
        types.ModelResponseComplete,
        UncategorizedToolResult,
        types.ModelResponseComplete,
    ]
    first = setup.children.requests[0]
    # One conversation to vendors, the child a named agent within it.
    mine = setup.parent.runtime.request()
    assert (mine.session, mine.agent, mine.parent) == (mine.session, mine.session, "")
    assert first.session == mine.session
    assert first.agent not in ("", mine.session)
    assert (first.parent, first.parent_turn) == (mine.session, mine.turn)
    assert first.root_prompt == mine.prompt_id
    assert (mine.name, first.name) == ("/root", "/root/Agent_0")
    # Offered no ``BackgroundTask``, the child is not told how deliveries arrive.
    assert first.system[-1] == "the parent's prompt"
    assert first.records[0] == UserMessage(content="count the TODOs")
    # Offered every tool its parent is, the child's prefix is its parent's.
    assert [t.name for t in first.tools] == ["Grep", "Read"]
    assert first.allowed is None


@pytest.mark.asyncio
async def test_a_child_that_fails_says_why_and_stays(tmp_path: Path) -> None:
    setup = _setup(tmp_path, ValueError("model down"))

    result = await _spawn(setup, tmp_path, {"prompt": "go", "label": "worker"})

    assert result.is_error is True
    assert (result.state, result.content) == ("failed", "ValueError: model down")
    assert list(setup.team) == ["Agent", "worker"]
    assert any(isinstance(e, types.ChildDoneEvent) for e in setup.seen)


@pytest.mark.asyncio
async def test_a_child_held_to_its_round_cap_fails_with_the_cap(tmp_path: Path) -> None:
    setup = _setup(
        tmp_path,
        reply(calls=[call("Read", "a")]),
        reply(calls=[call("Read", "b")]),
        reply("never"),
    )

    result = await _spawn(setup, tmp_path, {"prompt": "go", "max_tool_call_rounds": 1})

    assert result.state == "failed"
    assert (result.content or "").startswith("RoundLimitError")
    assert len(setup.children.requests) == 2


@pytest.mark.asyncio
async def test_a_child_at_its_round_cap_answers_with_what_it_has(
    tmp_path: Path,
) -> None:
    setup = _setup(tmp_path, reply(calls=[call("Read")]), reply("found it"))

    result = await _spawn(setup, tmp_path, {"prompt": "go", "max_tool_call_rounds": 1})

    assert (result.state, result.content) == ("completed", "found it")


@pytest.mark.asyncio
async def test_a_cancelled_spawn_halts_its_child_which_stays(tmp_path: Path) -> None:
    wait = _Wait(name="Wait")
    setup = _setup(tmp_path, reply(calls=[call("Wait")]), tools=[wait])
    async with setup.team:
        task = asyncio.ensure_future(_spawn(setup, tmp_path, {"prompt": "go"}))
        await wait.started.wait()
        _ = task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.wait_for(wait.stopped.wait(), timeout=5)
        assert list(setup.team) == ["Agent", "Agent_0"]
    assert any(isinstance(e, types.ChildDoneEvent) for e in setup.seen)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("arguments", "match"),
    [
        ({"prompt": "go", "tools": ["Edit"]}, "Unknown tools Edit; yours: Grep, Read"),
        ({"prompt": "go", "provider": "nope"}, "unknown provider .nope.; known: "),
        ({"prompt": "go", "effort": "max"}, "not offered"),
        ({"prompt": "go", "label": "Agent"}, "already labelled 'Agent'"),
        ({"prompt": "go", "label": "parent"}, "'parent' is reserved: all, ancestors"),
        ({"prompt": "go", "label": "model"}, "'model' is reserved"),
    ],
)
async def test_a_spawn_that_cannot_be_built_is_refused(
    tmp_path: Path,
    arguments: dict[str, PlainTree],
    match: str,
) -> None:
    setup = _setup(tmp_path)

    with pytest.raises(ValueError, match=match):
        _ = await _spawn(setup, tmp_path, arguments)
    assert list(setup.team) == ["Agent"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("label", "match"),
    [("stranger", "needs a calling agent"), ("leaf", "you are a leaf")],
)
async def test_a_caller_that_may_not_spawn_is_refused(
    tmp_path: Path,
    label: str,
    match: str,
) -> None:
    setup = _setup(tmp_path)
    _ = setup.team.join(
        Agent(model=ScriptedProvider()["scripted"], cwd=tmp_path),
        label="leaf",
        parent="Agent",
        max_depth=0,
    )

    with pytest.raises(ValueError, match=match):
        _ = await _spawn(setup, tmp_path, {"prompt": "go"}, label=label)


@pytest.mark.asyncio
async def test_a_child_compacts_as_its_parent_does(tmp_path: Path) -> None:
    setup = _setup(tmp_path, reply("ready"), compactor=SummaryCompactor)
    setup.parent.model.settings = replace(
        setup.parent.model.settings,
        context_size=setup.parent.model.capability.prices["auto", 1_000],
    )
    async with setup.team:
        _ = await _spawn(setup, tmp_path, {"prompt": "wait"})
        mine, theirs = (
            setup.parent.session.compactor,
            setup.team["Agent_0"].session.compactor,
        )

        assert isinstance(mine, SummaryCompactor)
        assert isinstance(theirs, SummaryCompactor)
        assert theirs is not mine
        assert theirs.trigger_tokens() == mine.trigger_tokens() < 1_000


@pytest.mark.asyncio
async def test_a_persistent_child_inherits_from_its_parent_unless_told(
    tmp_path: Path,
) -> None:
    setup = _setup(
        tmp_path,
        reply("ready"),
        compactor=SummaryCompactor,
        other=[reply("ready")],
    )
    setup.parent.model.settings = replace(
        setup.parent.model.settings,
        thinking_effort="low",
    )
    setup.parent.runtime.tool_context.max_result_chars = 4321
    setup.parent.environment = False
    async with setup.team:
        inherited = await _spawn(
            setup,
            tmp_path,
            {"prompt": "wait"},
        )
        told = await _spawn(
            setup,
            tmp_path,
            {
                "prompt": "wait",
                "label": "told",
                "tools": ["Grep"],
                "system": "the child's prompt",
                "provider": "other",
                "model": "other",
                "budget": "auto",
                "output": "text",
                "effort": "high",
                "max_depth": 0,
            },
        )
        heir, own = setup.team["Agent_0"], setup.team["told"]

        assert inherited.state == told.state == "completed"
        assert list(heir.tools) == list(own.tools) == ["Read", "Grep"]
        assert (heir.allowed, own.allowed) == (None, frozenset({"Grep"}))
        assert (heir.system_prompt, own.system_prompt) == (
            "the parent's prompt",
            "the child's prompt",
        )
        assert heir.runtime.tool_context.cwd == tmp_path
        assert heir.runtime.tool_context.max_result_chars == 4321
        assert heir.environment is False
        assert heir.agents_md is setup.parent.agents_md
        assert isinstance(heir.session.compactor, SummaryCompactor)
        assert heir.model is not setup.parent.model
        assert heir.model.settings.thinking_effort == "low"
        assert own.model.capability.model_id == "other"
        assert (
            own.model.settings.thinking_budget,
            own.model.settings.thinking_output,
            own.model.settings.thinking_effort,
        ) == ("auto", "text", "high")
        assert not setup.team.can_spawn("told")
        assert setup.team.persistent() == ["Agent_0", "told"]
    assert list(setup.team) == ["Agent"]


@pytest.mark.asyncio
async def test_a_child_waits_out_rate_limits_when_its_parent_does(
    tmp_path: Path,
) -> None:
    setup = _setup(tmp_path, reply("ready"), reply("ready"))
    async with setup.team:
        _ = await _spawn(setup, tmp_path, {"prompt": "wait"})
        setup.parent.runtime.persistent_retry = True
        _ = await _spawn(setup, tmp_path, {"prompt": "wait", "label": "patient"})

        assert setup.team["Agent_0"].runtime.persistent_retry is False
        assert setup.team["patient"].runtime.persistent_retry is True


@pytest.mark.asyncio
async def test_a_child_given_no_tools_may_call_none(tmp_path: Path) -> None:
    setup = _setup(tmp_path, reply("ready"))
    async with setup.team:
        _ = await _spawn(
            setup,
            tmp_path,
            {"prompt": "go", "tools": []},
        )
        child = setup.team["Agent_0"]

        assert list(child.tools) == ["Read", "Grep"]
        assert child.allowed == frozenset()


@pytest.mark.asyncio
async def test_a_child_may_call_only_what_its_gated_parent_may(tmp_path: Path) -> None:
    setup = _setup(tmp_path, reply("ready"), reply("ready"))
    setup.parent.allowed = frozenset({"Grep"})
    async with setup.team:
        _ = await _spawn(setup, tmp_path, {"prompt": "go"})
        with pytest.raises(ValueError, match="Unknown tools Read; yours: Grep"):
            _ = await _spawn(
                setup,
                tmp_path,
                {"prompt": "go", "tools": ["Read"]},
            )

        assert setup.team["Agent_0"].allowed == frozenset({"Grep"})


@pytest.mark.asyncio
async def test_a_child_asked_again_is_nudged_until_it_replies_by_agentsend(
    tmp_path: Path,
) -> None:
    setup = _setup(
        tmp_path,
        reply("first answer"),
        reply("second answer"),
        reply(""),
        reply(calls=[call("AgentSend", "s1", to="Agent", content="done")]),
        reply("sent it"),
        send=True,
    )
    async with setup.team:
        result = await _spawn(setup, tmp_path, {"prompt": "work", "label": "helper"})
        helper = setup.team["helper"]
        seen = _Seen()
        helper.runtime.observers.append(seen)
        setup.team.send("Agent", "helper", "and then?")
        assert setup.team.owes("helper") == {"Agent": "and then?"}

        (told,) = await _messages(setup.parent, 1)
        await seen.idle()

    assert result.content == "first answer"
    assert told == AgentToAgentMessage(
        sender="helper",
        recipient="Agent",
        content="done",
    )
    records = setup.children.requests[-1].records
    nudged = [
        r for r in records if isinstance(r, SystemMessage) and r.subtype == "nudge"
    ]
    assert len(nudged) == 2
    assert (nudged[0].content or "").startswith(
        "You owe a reply to: Agent ('and then?')",
    )
    assert setup.team.owes("helper") == {}
    del helper


@pytest.mark.asyncio
async def test_a_child_that_replies_by_agentsend_is_not_nudged(
    tmp_path: Path,
) -> None:
    setup = _setup(
        tmp_path,
        reply("ready"),
        reply(calls=[call("AgentSend", "s1", to="Agent", content="done")]),
        reply("sent it"),
        send=True,
    )
    async with setup.team:
        _ = await _spawn(setup, tmp_path, {"prompt": "work", "label": "helper"})
        seen = _Seen()
        setup.team["helper"].runtime.observers.append(seen)
        setup.team.send("Agent", "helper", "and then?")

        (told,) = await _messages(setup.parent, 1)
        await seen.idle()

    assert told.content == "done"
    assert setup.team.owes("helper") == {}
    assert len(setup.children.requests) == 3


@pytest.mark.asyncio
async def test_a_child_of_a_journaled_parent_keeps_its_own_journal(
    tmp_path: Path,
) -> None:
    async with JsonlJournal(tmp_path / "parent" / "journal.jsonl") as journal:
        setup = _setup(tmp_path, reply("done"), journal=journal)

        result = await _spawn(setup, tmp_path, {"prompt": "go"})

    path = child_journal(tmp_path / "parent" / "journal.jsonl", "Agent_0")
    assert result.output_file == str(path)
    assert "done" in path.read_text()
    assert [s.label for s in setup.team.standing("Agent")] == ["Agent_0"]


@pytest.mark.asyncio
async def test_a_persistent_child_stands_in_its_parents_journal_until_stopped(
    tmp_path: Path,
) -> None:
    async with JsonlJournal(tmp_path / "parent" / "journal.jsonl") as journal:
        setup = _setup(tmp_path, reply("ready"), journal=journal)
        async with setup.team:
            _ = await _spawn(
                setup,
                tmp_path,
                {
                    "prompt": "wait",
                    "label": "helper",
                    "tools": ["Grep"],
                    "effort": "high",
                    "budget": "auto",
                    "max_tool_call_rounds": 3,
                },
            )
            (spec,) = setup.team.standing("Agent")

            await setup.team.stop("helper")

            assert setup.team.standing("Agent") == []
    assert spec == ChildSpec(
        label="helper",
        journal=str(child_journal(tmp_path / "parent" / "journal.jsonl", "helper")),
        tools=("Grep",),
        max_rounds=3,
    )


@pytest.mark.asyncio
async def test_a_child_keeps_its_parent_s_window_and_stands_with_it(
    tmp_path: Path,
) -> None:
    wide = ModelCapability(
        model_id="wide-1.0",
        limits=ModelLimits(max_request_tokens=1_000_000),
    )
    catalog = ModelCatalog(
        factories={
            "wide": partial(
                ScriptedProvider,
                models=ModelTable(rows=(wide,)),
                replies=[reply("ready")],
            ),
        },
    )
    async with JsonlJournal(tmp_path / "parent" / "journal.jsonl") as journal:
        setup = _setup(tmp_path, journal=journal)
        setup.team.catalog = catalog
        setup.parent.model = catalog["wide"]["wide+200k"]
        async with setup.team:
            _ = await _spawn(
                setup,
                tmp_path,
                {"prompt": "wait", "label": "helper"},
            )
            (spec,) = setup.team.standing("Agent")

            assert setup.team["helper"].model.settings.context_size.tokens == 200_000
    async with JsonlJournal(Path(spec.journal)) as kept:
        stood = [
            r.payload["model"]
            for r in kept
            if isinstance(r, UncategorizedRecord) and r.kind == "model"
        ]
    assert stood == ["wide/wide-1.0+200k+nothink"]  # codespell:ignore nothink


@pytest.mark.asyncio
async def test_a_persistent_child_of_a_parent_in_memory_does_not_stand(
    tmp_path: Path,
) -> None:
    setup = _setup(tmp_path, reply("ready"))
    async with setup.team:
        result = await _spawn(
            setup,
            tmp_path,
            {"prompt": "wait"},
        )

        assert setup.team.standing("Agent") == []
    assert result.state == "completed"


async def _first_session(tmp_path: Path, *specs: dict[str, PlainTree]) -> None:
    """Spawn persistent children of a journaled root, then end the session."""
    async with JsonlJournal(tmp_path / "parent" / "journal.jsonl") as journal:
        setup = _setup(
            tmp_path,
            *[reply("noted")] * len(specs),
            journal=journal,
            other=[reply("noted")],
        )
        async with setup.team:
            for spec in specs:
                _ = await _spawn(setup, tmp_path, spec)


@pytest.mark.asyncio
async def test_a_resumed_parent_starts_its_running_children_again(
    tmp_path: Path,
) -> None:
    await _first_session(
        tmp_path,
        {
            "prompt": "remember 7",
            "label": "helper",
            "system": "the helper's prompt",
        },
    )

    async with JsonlJournal(tmp_path / "parent" / "journal.jsonl") as journal:
        setup = _setup(tmp_path, reply("7"), journal=journal)
        async with setup.parent, setup.team:
            assert await setup.spawn.resume("Agent") == ["helper"]
            helper = setup.team["helper"]

            assert await helper.send("which number?") == "7"
            assert setup.team.persistent() == ["helper"]
            assert helper.system_prompt == "the helper's prompt"
            assert [s.label for s in setup.team.standing("Agent")] == ["helper"]
    sent = setup.children.requests[0].records
    assert UserMessage(content="remember 7") in sent
    assert any(isinstance(e, types.ChildEvent) for e in setup.seen)
    # The ids come from the journals, so the resumed child is the same agent.
    assert setup.children.requests[0].agent == str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            (tmp_path / "parent" / "helper" / "session.jsonl").resolve().as_uri(),
        ),
    )


@pytest.mark.asyncio
async def test_a_resumed_parent_starts_its_childrens_children_again(
    tmp_path: Path,
) -> None:
    async with JsonlJournal(tmp_path / "parent" / "journal.jsonl") as journal:
        setup = _setup(tmp_path, reply("noted"), reply("noted"), journal=journal)
        async with setup.team:
            for spawner, label in (("Agent", "helper"), ("helper", "grand")):
                _ = await _spawn(
                    setup,
                    tmp_path,
                    {
                        "prompt": "wait",
                        "label": label,
                    },
                    label=spawner,
                )

    async with JsonlJournal(tmp_path / "parent" / "journal.jsonl") as journal:
        setup = _setup(tmp_path, journal=journal)
        async with setup.parent, setup.team:
            assert await setup.spawn.resume("Agent") == ["helper", "grand"]

            assert setup.team.parent("grand") == "helper"
            assert setup.team.persistent() == ["helper", "grand"]


@pytest.mark.asyncio
async def test_a_child_stood_before_its_journal_kept_its_model_resumes_on_it(
    tmp_path: Path,
) -> None:
    """Its spec named the model then; its journal names none."""
    child = tmp_path / "parent" / "helper" / "session.jsonl"
    child.parent.mkdir(parents=True)
    async with JsonlJournal(child) as journal:
        journal.append(UserMessage(content="remember 7"))
    async with JsonlJournal(tmp_path / "parent" / "journal.jsonl") as journal:
        setup = _setup(tmp_path, journal=journal, other=[reply("7")])
        _session(setup.parent).resume()
        setup.parent.session.stand(
            "agent:helper",
            UncategorizedRecord(
                kind="persistent_agent",
                payload={
                    "label": "helper",
                    "journal": str(child),
                    "tools": [],
                    "provider": "other",
                    "model": "other",
                },
            ),
        )
        async with setup.parent, setup.team:
            assert await setup.spawn.resume("Agent") == ["helper"]
            helper = setup.team["helper"]
            assert helper.model.provider.name == "other"
            assert await helper.send("which number?") == "7"


@pytest.mark.asyncio
async def test_a_child_journaled_under_the_old_name_resumes_renamed(
    tmp_path: Path,
) -> None:
    await _first_session(
        tmp_path,
        {"prompt": "remember 7", "label": "helper"},
    )
    written = tmp_path / "parent" / "helper" / "session.jsonl"
    old = written.rename(written.with_name("journal.jsonl"))
    parent = tmp_path / "parent" / "journal.jsonl"
    parent.write_text(parent.read_text().replace(str(written), str(old)))

    async with JsonlJournal(parent) as journal:
        setup = _setup(tmp_path, reply("7"), journal=journal)
        async with setup.parent, setup.team:
            assert await setup.spawn.resume("Agent") == ["helper"]
            assert await setup.team["helper"].send("which number?") == "7"
    assert (written.exists(), old.exists()) == (True, False)


@pytest.mark.asyncio
async def test_a_child_starts_from_its_parents_cache_and_keeps_a_copy(
    tmp_path: Path,
) -> None:
    setup = _setup(tmp_path, other=[reply("ready")])
    capability = _capability("other")
    setup.parent.model_cache["other/other"] = CachedModel(
        capability=capability,
        settings=replace(capability.default, thinking_effort="high"),
    )
    async with setup.team:
        _ = await _spawn(
            setup,
            tmp_path,
            {"prompt": "wait", "provider": "other"},
        )
        child = setup.team["Agent_0"]

        assert child.model.settings.thinking_effort == "high"
        assert child.model_cache == setup.parent.model_cache
        assert child.model_cache is not setup.parent.model_cache


@pytest.mark.asyncio
async def test_a_resumed_child_comes_back_on_its_own_model_and_prompt(
    tmp_path: Path,
) -> None:
    await _first_session(
        tmp_path,
        {
            "prompt": "remember 7",
            "label": "helper",
            "provider": "other",
            "effort": "high",
            "system": "the helper's prompt",
        },
    )

    async with JsonlJournal(tmp_path / "parent" / "journal.jsonl") as journal:
        setup = _setup(tmp_path, journal=journal)
        _session(setup.parent).resume()
        async with setup.team:
            assert await setup.spawn.resume("Agent") == ["helper"]
            helper = setup.team["helper"]
            assert (
                helper.model.provider.name,
                helper.model.settings.thinking_effort,
            ) == (
                "other",
                "high",
            )
            assert helper.system_prompt == "the helper's prompt"


@pytest.mark.asyncio
async def test_a_reply_owed_at_the_end_is_still_owed_on_resume(tmp_path: Path) -> None:
    async with JsonlJournal(tmp_path / "parent" / "journal.jsonl") as journal:
        setup = _setup(tmp_path, reply("noted"), journal=journal)
        async with setup.team:
            _ = await _spawn(setup, tmp_path, {"prompt": "wait", "label": "helper"})
            setup.team.ask("helper", "still there?")

    async with JsonlJournal(tmp_path / "parent" / "journal.jsonl") as journal:
        setup = _setup(tmp_path, journal=journal)
        _session(setup.parent).resume()
        async with setup.team:
            _ = await setup.spawn.resume("Agent")

            assert setup.team.owes("helper") == {"user": "still there?"}


@pytest.mark.asyncio
async def test_a_child_whose_label_is_taken_resumes_under_a_new_one(
    tmp_path: Path,
) -> None:
    await _first_session(
        tmp_path,
        {
            "prompt": "wait",
            "label": "helper",
        },
    )

    async with JsonlJournal(tmp_path / "parent" / "journal.jsonl") as journal:
        setup = _setup(tmp_path, journal=journal)
        _session(setup.parent).resume()
        _ = setup.team.join(
            Agent(model=ScriptedProvider()["scripted"], cwd=tmp_path),
            label="helper",
            parent="Agent",
        )
        async with setup.team:
            assert await setup.spawn.resume("Agent") == ["Agent_0"]

            assert [s.label for s in setup.team.standing("Agent")] == ["Agent_0"]


@pytest.mark.asyncio
async def test_a_child_that_cannot_be_rebuilt_is_skipped(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    await _first_session(
        tmp_path,
        {
            "prompt": "a",
            "label": "gone",
            "provider": "other",
            "model": "other",
        },
        {
            "prompt": "b",
            "label": "locked",
        },
        {"prompt": "c", "label": "kept"},
    )

    async with JsonlJournal(tmp_path / "parent" / "journal.jsonl") as journal:
        factories = {
            "scripted": partial(ScriptedProvider, capability=_capability("scripted")),
        }
        setup = _setup(tmp_path, journal=journal)
        setup.team.catalog = ModelCatalog(factories=factories)
        _session(setup.parent).resume()
        held = child_journal(tmp_path / "parent" / "journal.jsonl", "locked")
        async with JsonlJournal(held), setup.team:
            with caplog.at_level(logging.WARNING):
                assert await setup.spawn.resume("Agent") == ["kept"]
    assert any("not resuming agent 'gone'" in m for m in caplog.messages)
    assert any("not resuming agent 'locked'" in m for m in caplog.messages)


@pytest.mark.asyncio
async def test_a_child_owing_a_reply_whose_model_fails_says_so(
    tmp_path: Path,
) -> None:
    setup = _setup(tmp_path, reply("ready"), RuntimeError("model gone"))
    async with setup.team:
        _ = await _spawn(setup, tmp_path, {"prompt": "work", "label": "helper"})
        setup.team.send("Agent", "helper", "more")

        (told,) = await _messages(setup.parent, 1)

    assert (told.sender, told.recipient, told.content) == (
        "helper",
        "Agent",
        "[helper failed] RuntimeError: model gone",
    )


@pytest.mark.asyncio
async def test_parallel_spawns_each_get_a_label_of_their_own(tmp_path: Path) -> None:
    async with JsonlJournal(tmp_path / "parent" / "journal.jsonl") as journal:
        setup = _setup(tmp_path, reply("one"), reply("two"), journal=journal)

        results = await asyncio.gather(
            _spawn(setup, tmp_path, {"prompt": "a"}),
            _spawn(setup, tmp_path, {"prompt": "b"}),
        )

    assert sorted(r.agent_id or "" for r in results) == ["Agent_0", "Agent_1"]
    assert all(r.state == "completed" for r in results)


@pytest.mark.asyncio
async def test_stopping_a_parent_mid_spawn_ends_the_spawn(
    tmp_path: Path,
) -> None:
    wait = _Wait(name="Wait")
    setup = _setup(tmp_path, reply("ready"), reply(calls=[call("Wait")]), tools=[wait])
    async with setup.team:
        _ = await _spawn(
            setup,
            tmp_path,
            {
                "prompt": "wait",
                "label": "p",
            },
        )
        task = asyncio.ensure_future(
            _spawn(setup, tmp_path, {"prompt": "go"}, label="p"),
        )
        await wait.started.wait()

        await setup.team.stop("p")

        result = await asyncio.wait_for(task, timeout=5)
        assert result.state == "failed"
        assert "p" not in setup.team


@pytest.mark.asyncio
async def test_a_child_closes_the_model_it_ran_on(tmp_path: Path) -> None:
    setup = _setup(tmp_path)
    models: list[_CountingModel] = []
    setup.team.catalog = ModelCatalog(
        factories={
            "scripted": partial(_Counting, replies=[reply("done")], models=models),
        },
    )

    async with setup.team:
        _ = await _spawn(setup, tmp_path, {"prompt": "go"})
        assert [m.closed for m in models] == [0]

    assert [m.closed for m in models] == [1]


@pytest.mark.asyncio
async def test_a_child_whose_journal_is_held_elsewhere_is_refused(
    tmp_path: Path,
) -> None:
    models: list[_CountingModel] = []
    async with JsonlJournal(tmp_path / "parent" / "journal.jsonl") as journal:
        setup = _setup(tmp_path, journal=journal)
        setup.team.catalog = ModelCatalog(
            factories={"scripted": partial(_Counting, models=models)},
        )
        held = child_journal(tmp_path / "parent" / "journal.jsonl", "Agent_0")
        async with JsonlJournal(held):
            with pytest.raises(RuntimeError, match="open in another process"):
                _ = await _spawn(setup, tmp_path, {"prompt": "go"})

        assert list(setup.team) == ["Agent"]
        assert setup.team.standing("Agent") == []
    assert [m.closed for m in models] == [1]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("arguments", "said"),
    [
        ({"account": "work"}, "as work"),
        ({"account": "work", "provider": "scripted"}, "as work"),
        ({}, "as None"),
    ],
    ids=["account", "account-and-provider", "inherited"],
)
async def test_a_child_told_an_account_signs_in_as_it(
    tmp_path: Path,
    arguments: dict[str, PlainTree],
    said: str,
) -> None:
    setup = _setup(tmp_path)
    setup.team.catalog = ModelCatalog(factories=_as_account)

    result = await _spawn(setup, tmp_path, {"prompt": "who?", **arguments})

    assert (result.state, result.text()) == ("completed", said)
    assert setup.team.catalog.account is None


@pytest.mark.asyncio
async def test_a_refused_spawn_closes_the_model_it_built(tmp_path: Path) -> None:
    setup = _setup(tmp_path)
    models: list[_CountingModel] = []
    setup.team.catalog = ModelCatalog(
        factories={"scripted": partial(_Counting, models=models)},
    )

    with pytest.raises(ValueError, match="not offered"):
        _ = await _spawn(setup, tmp_path, {"prompt": "go", "budget": "fixed"})

    assert [m.closed for m in models] == [1]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("told", "catalogs", "account"),
    [("work", None, "work"), (None, "home", "home"), (None, None, None)],
    ids=["told", "the-catalogs", "default"],
)
async def test_a_child_knows_the_account_it_signed_in_as(
    tmp_path: Path,
    told: str | None,
    catalogs: str | None,
    account: str | None,
) -> None:
    """So its journal stands its model with the account, and a resume signs in so."""
    setup = _setup(tmp_path)
    setup.team.catalog = ModelCatalog(factories=_as_account)
    setup.team.catalog.use_account(catalogs)
    arguments: dict[str, PlainTree] = {"prompt": "who?"}
    if told is not None:
        arguments["account"] = told
    async with setup.team:
        _ = await _spawn(setup, tmp_path, arguments)
        assert setup.team["Agent_0"].account == account


@pytest.mark.asyncio
async def test_watch_starts_the_callers_watcher_its_prompt_the_first_request(
    tmp_path: Path,
) -> None:
    team = Team(catalog=ModelCatalog(factories={"scripted": ScriptedProvider}))
    spawn = AgentSpawn(team=team)
    root = Agent(model=ScriptedProvider()["scripted"], tools=[spawn], cwd=tmp_path)
    _ = team.join(root)
    context = root.runtime.tool_context
    async with team:
        with pytest.raises(ValueError, match="needs a policy"):
            _ = await spawn.call(context, {"prompt": " ", "watch": True})
        started = await spawn.call(context, {"prompt": "keep me honest", "watch": True})

        assert (started.agent_id, started.state) == ("watcher", "running")
        assert team.resolve("parent", by="Agent") == ["watcher"]
        assert team.owes("watcher") == {"Agent": "keep me honest"}
        told = [
            m.content or ""
            for m in team["watcher"].runtime.inbox.drain_nowait()
            if isinstance(m, AgentToAgentMessage)
        ]
        assert told[0].startswith("<skill name='watch' source='builtin'>")
        assert told[1] == "keep me honest"
        assert spawn.serialize_key(context, {"watch": True}) == "parent"


@pytest.mark.asyncio
async def test_a_watcher_of_a_journaled_root_keeps_its_journal_beside_it(
    tmp_path: Path,
) -> None:
    async with JsonlJournal(tmp_path / "parent" / "journal.jsonl") as journal:
        setup = _setup(tmp_path, journal=journal)
        async with setup.parent, setup.team:
            other = Agent(model=ScriptedProvider()["scripted"], cwd=tmp_path)
            _ = setup.team.join(other, label="other")
            setup.parent.runtime.persistent_retry = True

            assert await setup.spawn.watch("Agent", before="how") == "watcher"
            assert await setup.spawn.watch("other") == "watcher_0"
            with pytest.raises(ValueError, match="'Agent' is no unwatched root"):
                _ = await setup.spawn.watch("Agent")
            watcher = setup.team["watcher"]
            assert watcher.runtime.persistent_retry is True
            assert setup.team["watcher_0"].runtime.persistent_retry is False

            assert watcher.session.journal.path == child_journal(
                tmp_path / "parent" / "journal.jsonl",
                "watcher",
            )
            assert setup.team.resolve("parent", by="Agent") == ["watcher"]
            assert watcher.system_prompt == "the parent's prompt"
            (first,) = watcher.runtime.inbox.drain_nowait()
            assert isinstance(first, AgentToAgentMessage)
            assert first.content == "how"


@pytest.mark.asyncio
async def test_a_watcher_comes_back_on_resume_and_is_sent_only_what_is_new(
    tmp_path: Path,
) -> None:
    async with JsonlJournal(tmp_path / "parent" / "journal.jsonl") as journal:
        setup = _setup(tmp_path, journal=journal)
        async with setup.parent, setup.team:
            setup.parent.session.append(UserMessage(content="before"))
            _ = await setup.spawn.watch("Agent")

    async with JsonlJournal(tmp_path / "parent" / "journal.jsonl") as journal:
        setup = _setup(tmp_path, journal=journal)
        async with setup.parent, setup.team:
            assert await setup.spawn.resume("Agent") == ["watcher"]
            watcher = setup.team["watcher"]
            _ = watcher.runtime.inbox.drain_nowait()
            setup.parent.session.append(UserMessage(content="after"))
            setup.parent.runtime.publish(types.AgentIdle())

            (told,) = watcher.runtime.inbox.drain_nowait()
            assert isinstance(told, AgentToAgentMessage)
            assert told.content == "[watch]\n> user: after"

            await setup.team.stop("watcher")
            assert await setup.spawn.resume("Agent") == []


@pytest.mark.asyncio
async def test_a_watcher_that_cannot_open_is_refused_or_skipped_on_resume(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    root = tmp_path / "parent" / "journal.jsonl"
    async with JsonlJournal(root) as journal:
        setup = _setup(tmp_path, journal=journal)
        async with setup.parent, setup.team:
            _ = await setup.spawn.watch("Agent")
            setup.parent.session.stand(
                "watcher:odd",
                UncategorizedRecord(kind="watcher", payload={"label": 7}),
            )

    async with (
        JsonlJournal(root) as journal,
        JsonlJournal(child_journal(root, "watcher")),
    ):
        setup = _setup(tmp_path, journal=journal)
        async with setup.parent, setup.team:
            with caplog.at_level(logging.WARNING):
                assert await setup.spawn.resume("Agent") == []
            with pytest.raises(RuntimeError, match="open in another process"):
                _ = await setup.spawn.watch("Agent")
            assert list(setup.team) == ["Agent"]
    assert any("not resuming watcher 'watcher'" in m for m in caplog.messages)


def _as_account(account: str | None) -> Mapping[str, Callable[..., ScriptedProvider]]:
    """Return a provider whose model says which account it signed in as."""
    return {
        "scripted": partial(
            ScriptedProvider,
            replies=[reply(f"as {account}")],
            capability=_capability("scripted"),
        ),
    }


class _CountingModel(ScriptedModel):
    """A scripted model that counts its closes."""

    closed: int = 0

    @override
    async def aclose(self) -> None:
        self.closed += 1


class _Counting(ScriptedProvider):
    """A scripted provider whose models count their closes, in ``models``."""

    def __init__(
        self,
        *,
        models: list[_CountingModel],
        name: str = "",
        resources: ProviderSharedResources | None = None,
        replies: Sequence[Reply] = (),
    ) -> None:
        super().__init__(
            name=name,
            resources=resources,
            replies=replies,
            capability=_capability("scripted"),
        )
        self.built = models

    @override
    def __getitem__(self, model: str) -> _CountingModel:
        del model
        built = _CountingModel(
            provider=self,
            capability=self.capability,
            settings=self.capability.default,
        )
        self.built.append(built)
        return built


def test_the_description_reads_as_utf8_whatever_the_locale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asked = _Asked()
    with monkeypatch.context() as patched:
        patched.setattr(Path, "read_text", asked)
        _ = importlib.reload(agent_spawn)
    _ = importlib.reload(agent_spawn)

    assert asked.encodings["agentspawn.md"] == "utf-8"


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
