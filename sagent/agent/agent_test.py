"""Tests for the agent: its turns, its caps, the session it keeps, what it restates."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, override

import asyncio
import contextlib
import uuid

from treekle import PlainTree

import pytest

from sagent.agent import agent as agent_module
from sagent.agent.agent import (
    Agent,
    NoJournal,
    NoShrinker,
    RuntimeSession,
    default_system_prompt,
)
from sagent.agent.agents_md import AgentsMdConfig, directives
from sagent.agent.compactor import SummaryCompactor
from sagent.agent.pruner import drop
from sagent.model.capability import ModelCapability, ModelLimits, ModelSettings
from sagent.model.cost import PriceCatalog, TokenCost, TokenCount
from sagent.model.exceptions import ContextOverflowError
from sagent.model.scripted import (
    Reply,
    ScriptedModel,
    ScriptedProvider,
    call,
    reply,
)
from sagent.model.types import Request, Response
from sagent.providers.catalog import ModelCatalog
from sagent.providers.table import ModelTable
from sagent.runtime import types
from sagent.runtime.exceptions import BudgetExhaustedError, RoundLimitError
from sagent.runtime.tape import MaskRange
from sagent.session.jsonl import JsonlJournal
from sagent.session.recording import RecordingJournal
from sagent.session.types import (
    AgentToAgentMessage,
    AssistantMessage,
    Attachment,
    ContextClear,
    SessionRecord,
    SystemMessage,
    TokenUsage,
    ToolCall,
    ToolResult,
    TurnContext,
    UncategorizedRecord,
    UncategorizedToolResult,
    UserMessage,
    record_classes,
)
from sagent.tools.display import ToolDisplay


if TYPE_CHECKING:
    from sagent.agent.types import Journal
    from sagent.runtime.types import ModelContextEvent
    from sagent.tools.types import ToolContext


@pytest.fixture(autouse=True)
def isolate_git(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the environment's git probe off the user's config and above ``tmp_path``."""
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / "gitconfig"))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))


def _agent(
    tmp_path: Path,
    *replies: Reply,
    provider: ScriptedProvider | None = None,
    tools: Sequence[_Tool] = (),
    journal: Journal | None = None,
    max_rounds: int | None = None,
    max_spend_usd: float | None = None,
    system_prompt: str | None = "S",
) -> tuple[ScriptedProvider, Agent]:
    provider = provider if provider is not None else ScriptedProvider(replies=replies)
    agent = Agent(
        model=provider["scripted"],
        tools=tools,
        journal=journal,
        cwd=tmp_path,
        system_prompt=system_prompt,
        max_rounds=max_rounds,
        max_spend_usd=max_spend_usd,
        agents_md=_config(tmp_path),
    )
    return provider, agent


def _config(tmp_path: Path) -> AgentsMdConfig:
    return AgentsMdConfig(system_dir=tmp_path / "system", user_dir=tmp_path / "user")


def _own(journal: Journal) -> list[SessionRecord]:
    """Return ``journal``'s window less the model and prompt the agent stands itself."""
    return [
        r
        for r in journal
        if not (
            isinstance(r, UncategorizedRecord) and r.kind in ("model", "system_prompt")
        )
    ]


def _session(agent: Agent) -> RuntimeSession:
    session = agent.session
    assert isinstance(session, RuntimeSession)
    return session


def _texts(records: Sequence[SessionRecord]) -> list[str]:
    return [
        r.content or ""
        for r in records
        if isinstance(r, (UserMessage, AssistantMessage))
    ]


def _capability(
    model_id: str = "scripted",
    cutoff: str | None = None,
) -> ModelCapability:
    prices = PriceCatalog(window=200_000)
    return ModelCapability(
        model_id=model_id,
        knowledge_cutoff=cutoff,
        limits=ModelLimits(max_request_tokens=200_000, max_response_tokens=8_000),
        prices=prices,
        default=ModelSettings(context_size=prices["auto", "fullcontext"]),
    )


@dataclass(slots=True, kw_only=True)
class _Tool:
    """A tool to give an agent: it answers ``ok``."""

    name: str = "_Tool"
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
class _Reminding(_Tool):
    """A tool with something to tell the model each round."""

    told: str = ""

    def reminder(self, context: ToolContext) -> str:
        del context
        return self.told


@dataclass(slots=True, kw_only=True)
class _Briefing(_Tool):
    """A tool with standing context, read each time the agent asks."""

    said: str = ""

    def briefing(self, context: ToolContext) -> str:
        del context
        return self.said


class _CountingModel(ScriptedModel):
    """A scripted model that counts its closes."""

    closed: int = 0

    @override
    async def aclose(self) -> None:
        self.closed += 1


class _Counting(ScriptedProvider):
    """A scripted provider whose models count their closes."""

    @override
    def __getitem__(self, model: str) -> _CountingModel:
        del model
        return _CountingModel(
            provider=self,
            capability=self.capability,
            settings=self.capability.default,
        )


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


class _LoopCrash(BaseException):
    """What kills the runtime's loop: nothing in it catches a ``BaseException``."""


async def _crash(agent: Agent, text: str, raised: list[BaseException]) -> None:
    """Send ``text`` to ``agent`` with its loop set to die on it; note what ``send`` raised."""
    async with agent:
        agent.runtime.observers.append(_crash_on_user)
        try:
            _ = await agent.send(text)
        except (_LoopCrash, RuntimeError) as e:
            raised.append(e)
            raise


def _crash_on_user(event: types.RuntimeEvent) -> None:
    if isinstance(event, UserMessage):
        raise _LoopCrash


@pytest.mark.asyncio
async def test_send_returns_what_the_model_said_last(tmp_path: Path) -> None:
    provider, agent = _agent(
        tmp_path,
        reply("looking", calls=[call("Nope", "c1")]),
        reply("done"),
    )
    async with agent:
        assert await agent.send("go") == "done"

    second = provider.requests[1].records
    results = [r for r in second if isinstance(r, UncategorizedToolResult)]
    assert [r.text() for r in results] == ["Unknown tool: Nope"]


@pytest.mark.asyncio
async def test_a_wordless_last_turn_keeps_what_was_said_before(tmp_path: Path) -> None:
    _, agent = _agent(tmp_path, reply("looking", calls=[call("Nope")]), reply(""))
    async with agent:
        assert await agent.send("go") == "looking"


@pytest.mark.asyncio
async def test_stream_yields_the_words_as_they_are_written(tmp_path: Path) -> None:
    _, agent = _agent(tmp_path, reply("one ", calls=[call("Nope")]), reply("two"))
    async with agent:
        assert [words async for words in agent.stream("go")] == ["one ", "two"]


@pytest.mark.asyncio
async def test_attachments_go_with_the_message(tmp_path: Path) -> None:
    image = Attachment(mime_descriptor="image/png", data=b"png")
    provider, agent = _agent(tmp_path, reply("seen"))
    async with agent:
        _ = await agent.send("look", attachments=[image])

    first = provider.requests[0].records[0]
    assert first == UserMessage(content="look", attachments=(image,))


@pytest.mark.asyncio
async def test_a_model_error_is_raised_from_the_turn(tmp_path: Path) -> None:
    _, agent = _agent(tmp_path, ValueError("boom"))
    async with agent:
        with pytest.raises(ValueError, match="boom"):
            _ = [words async for words in agent.stream("go")]


@pytest.mark.asyncio
async def test_a_halt_from_outside_cancels_the_turn(tmp_path: Path) -> None:
    provider = ScriptedProvider()
    _, agent = _agent(tmp_path, provider=provider)
    provider.replies.append(lambda request: _halted(agent, request))
    async with agent:
        with pytest.raises(asyncio.CancelledError):
            _ = await agent.send("go")


def _halted(agent: Agent, request: Request) -> Response:
    del request
    agent.runtime.halt()
    return reply(calls=[call("Nope")])


@pytest.mark.asyncio
async def test_a_turn_needs_an_open_agent(tmp_path: Path) -> None:
    _, agent = _agent(tmp_path)

    with pytest.raises(
        RuntimeError,
        match=r"^The agent is not open: use it with 'async with'\.$",
    ):
        _ = await agent.send("go")


@pytest.mark.asyncio
async def test_an_agent_runs_from_its_open_until_its_loop_stops(tmp_path: Path) -> None:
    _, agent = _agent(tmp_path, reply("done"))
    seen = [agent.running]
    async with agent:
        seen.append(agent.running)
        _ = await agent.send("go")
        seen.append(agent.running)
        agent.runtime.push(types.Quit())
        with pytest.raises(RuntimeError, match="runtime stopped"):
            _ = await agent.send("again")
        seen.append(agent.running)
    seen.append(agent.running)

    assert seen == [False, True, True, False, False]


@pytest.mark.asyncio
async def test_an_agent_whose_loop_died_is_not_running(tmp_path: Path) -> None:
    _, agent = _agent(tmp_path)
    running: list[bool] = []

    async def crash() -> None:
        async with agent:
            agent.runtime.observers.append(_crash_on_user)
            try:
                _ = await agent.send("go")
            finally:
                running.append(agent.running)

    with pytest.raises(_LoopCrash):
        await crash()

    assert running == [False]


@pytest.mark.asyncio
async def test_a_turn_ends_when_the_loop_stops_first(tmp_path: Path) -> None:
    _, agent = _agent(tmp_path)
    async with agent:
        agent.runtime.push(types.Quit())

        async with asyncio.timeout(5):
            with pytest.raises(RuntimeError, match=r"^The agent's runtime stopped\.$"):
                _ = await agent.send("go")


@pytest.mark.asyncio
async def test_a_loop_that_dies_raises_from_the_turn_and_the_journal_is_kept(
    tmp_path: Path,
) -> None:
    journal = RecordingJournal()
    _, agent = _agent(tmp_path, journal=journal)

    raised: list[BaseException] = []
    async with asyncio.timeout(5):
        with pytest.raises(_LoopCrash):
            await _crash(agent, "remember me", raised)

    assert [type(e) for e in raised] == [_LoopCrash]
    assert UserMessage(content="remember me") in list(journal)
    assert agent.runtime.observers == [_crash_on_user]


@pytest.mark.asyncio
async def test_closing_twice_or_unopened_is_harmless(tmp_path: Path) -> None:
    journal = RecordingJournal()
    _, agent = _agent(tmp_path, journal=journal)
    async with agent:
        pass

    await agent.aclose()
    await _agent(tmp_path)[1].aclose()

    assert len(_own(journal)) == 1


@pytest.mark.asyncio
async def test_at_the_round_cap_the_model_answers_with_no_tools(
    tmp_path: Path,
) -> None:
    provider, agent = _agent(
        tmp_path,
        reply(calls=[call("Nope")]),
        reply("what I found"),
        tools=[_Tool(name="Nope")],
        max_rounds=1,
    )
    async with agent:
        assert await agent.send("go") == "what I found"

    last = provider.requests[1]
    assert last.allowed == frozenset()
    told = [r for r in last.records if isinstance(r, TurnContext)][-1]
    assert "Tools you may call: none" in told.text()


@pytest.mark.asyncio
async def test_the_round_that_reaches_the_cap_runs_its_calls(tmp_path: Path) -> None:
    provider, agent = _agent(
        tmp_path,
        reply(calls=[call("Nope")]),
        reply("done"),
        tools=[_Tool(name="Nope")],
        max_rounds=1,
    )
    async with agent:
        _ = await agent.send("go")

    answered = [r for r in provider.requests[1].records if isinstance(r, ToolResult)]
    assert [r.text() for r in answered] == ["ok"]


@pytest.mark.asyncio
async def test_a_message_from_another_agent_mid_turn_keeps_the_round_cap(
    tmp_path: Path,
) -> None:
    """A persistent child's idle ping arriving mid-turn starts no turn of its own."""
    agents: list[Agent] = []

    def pinged(request: Request) -> Response:
        del request
        agents[0].runtime.push(
            AgentToAgentMessage(sender="kid", content="[kid is idle]"),
        )
        return reply("what I found")

    provider = ScriptedProvider(
        replies=[
            reply(calls=[call("Nope", "a")]),
            pinged,
            reply(calls=[call("Nope", "b")]),
            reply("never"),
        ],
    )
    _, agent = _agent(
        tmp_path,
        provider=provider,
        tools=[_Tool(name="Nope")],
        max_rounds=1,
    )
    agents.append(agent)
    async with agent:
        with contextlib.suppress(RoundLimitError):
            _ = await agent.send("go")

    assert [r.allowed for r in provider.requests[1:]] == [frozenset()] * (
        len(provider.requests) - 1
    )


@pytest.mark.asyncio
async def test_a_call_past_the_round_cap_halts_the_turn(tmp_path: Path) -> None:
    provider, agent = _agent(
        tmp_path,
        reply(calls=[call("Nope", "a")]),
        reply(calls=[call("Nope", "b")]),
        reply("never"),
        tools=[_Tool(name="Nope")],
        max_rounds=1,
    )
    async with agent:
        with pytest.raises(RoundLimitError) as raised:
            _ = await agent.send("go")

    assert (raised.value.rounds, raised.value.max_rounds) == (1, 1)
    assert len(provider.requests) == 2


@pytest.mark.asyncio
async def test_agentsend_is_outside_the_round_cap(tmp_path: Path) -> None:
    provider, agent = _agent(
        tmp_path,
        reply(calls=[call("Nope", "a")]),
        reply(calls=[call("AgentSend", "s1")]),
        reply(calls=[call("AgentSend", "s2")]),
        reply("replied"),
        tools=[_Tool(name="Nope"), _Tool(name="AgentSend")],
        max_rounds=1,
    )
    async with agent:
        assert await agent.send("go") == "replied"

    assert [r.allowed for r in provider.requests[1:]] == [
        frozenset({"AgentSend"}),
    ] * 3


@pytest.mark.asyncio
async def test_the_round_count_starts_over_with_each_message(tmp_path: Path) -> None:
    _, agent = _agent(
        tmp_path,
        reply(calls=[call("Nope", "a")]),
        reply("one"),
        reply(calls=[call("Nope", "b")]),
        reply("two"),
        max_rounds=2,
    )
    async with agent:
        assert await agent.send("first") == "one"
        assert await agent.send("second") == "two"


@pytest.mark.asyncio
async def test_a_turn_stopped_at_its_cap_lets_the_next_one_run(tmp_path: Path) -> None:
    provider, agent = _agent(
        tmp_path,
        reply(calls=[call("Nope", "a")]),
        reply(calls=[call("Nope", "b")]),
        reply("again"),
        tools=[_Tool(name="Nope")],
        max_rounds=1,
    )
    async with agent:
        with pytest.raises(RoundLimitError):
            _ = await agent.send("go")

        assert await agent.send("go on") == "again"

    # The next message gives back every tool.
    assert provider.requests[2].allowed is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("spends", "raises"),
    [((0.6, 0.6), True), ((0.5, 0.5), True), ((0.5, 0.4), False)],
    ids=["over", "exactly", "under"],
)
async def test_the_spend_cap_halts_once_reached(
    tmp_path: Path,
    spends: tuple[float, float],
    raises: bool,
) -> None:
    first, second = (TokenCost(response=s) for s in spends)
    _, agent = _agent(
        tmp_path,
        reply("a", calls=[call("Nope")], spend=first),
        reply("b", spend=second),
        max_spend_usd=1.0,
    )
    async with agent:
        if not raises:
            assert await agent.send("go") == "b"
            return
        with pytest.raises(BudgetExhaustedError) as raised:
            _ = await agent.send("go")

    assert raised.value.total_cost_usd == pytest.approx(sum(spends))
    assert raised.value.max_budget_usd == 1.0


@pytest.mark.asyncio
async def test_spend_adds_up_across_turns(tmp_path: Path) -> None:
    _, agent = _agent(
        tmp_path,
        reply("a", spend=TokenCost(response=0.6)),
        reply("b", spend=TokenCost(response=0.6)),
        max_spend_usd=1.0,
    )
    async with agent:
        assert await agent.send("first") == "a"
        with pytest.raises(BudgetExhaustedError):
            _ = await agent.send("second")


@pytest.mark.asyncio
async def test_the_spend_cap_is_named_when_both_caps_are_reached(
    tmp_path: Path,
) -> None:
    _, agent = _agent(
        tmp_path,
        reply(calls=[call("Nope")], spend=TokenCost(response=2.0)),
        max_rounds=1,
        max_spend_usd=1.0,
    )
    async with agent:
        with pytest.raises(BudgetExhaustedError):
            _ = await agent.send("go")


@pytest.mark.asyncio
async def test_a_charge_past_the_cap_halts_and_refuses_the_next_turn(
    tmp_path: Path,
) -> None:
    """A child's spend, charged between turns, trips the cap its parent holds."""
    provider, agent = _agent(tmp_path, reply("unsent"), max_spend_usd=1.0)
    async with agent:
        agent.charge(0.5, by="kid")
        assert agent.spend_usd == 0.5
        agent.charge(0.75, by="kid")
        with pytest.raises(BudgetExhaustedError) as raised:
            _ = await agent.send("go")
    assert raised.value.total_cost_usd == pytest.approx(1.25)
    assert provider.requests == []


@pytest.mark.asyncio
async def test_a_session_with_no_journal_s_history_is_what_the_model_sees(
    tmp_path: Path,
) -> None:
    _, agent = _agent(tmp_path, reply("hi"))
    async with agent:
        assert await agent.send("go") == "hi"
        assert list(agent.session.history()) == list(agent.session)
    assert [
        r.content for r in agent.session.history() if isinstance(r, UserMessage)
    ] == [
        "go",
    ]


@pytest.mark.asyncio
async def test_the_system_prompt_is_sent_and_can_be_changed(tmp_path: Path) -> None:
    provider, agent = _agent(tmp_path, reply("1"), reply("2"), reply("3"))
    async with agent:
        _ = await agent.send("one")
        agent.system_prompt = "T"
        assert agent.system_prompt == "T"
        _ = await agent.send("two")
        agent.system_prompt = None
        _ = await agent.send("three")

    assert [r.system for r in provider.requests] == [
        ("S",),
        ("T",),
        (default_system_prompt(),),
    ]
    assert default_system_prompt().strip()


def test_no_system_prompt_is_the_default_one(tmp_path: Path) -> None:
    _, agent = _agent(tmp_path, system_prompt=None)

    assert agent.system_prompt is None
    assert agent.runtime.system == (default_system_prompt(),)


@pytest.mark.asyncio
async def test_a_new_model_answers_the_next_call_and_is_named(tmp_path: Path) -> None:
    first, agent = _agent(tmp_path, reply("from the first"))
    other = ScriptedProvider(
        replies=[reply("from the other")],
        capability=_capability(model_id="other"),
    )
    async with agent:
        assert await agent.send("one") == "from the first"
        agent.model = other["other"]
        assert await agent.send("two") == "from the other"

    assert len(first.requests) == 1
    stated = [r for r in other.requests[0].records if isinstance(r, TurnContext)]
    assert [s.model for s in stated] == ["scripted+fullcontext", "other+fullcontext"]


def test_the_agent_hands_its_settings_to_the_runtime(tmp_path: Path) -> None:
    provider = ScriptedProvider()
    agent = Agent(
        model=provider["scripted"],
        tools=[_Tool(name="A"), _Tool(name="B")],
        cwd=tmp_path,
        persistent_retry=True,
        max_result_chars=1234,
    )

    assert list(agent.tools) == ["A", "B"]
    assert agent.runtime.persistent_retry is True
    assert agent.runtime.tool_context.cwd == tmp_path
    assert agent.runtime.tool_context.max_result_chars == 1234
    default = Agent(model=provider["scripted"])
    assert default.runtime.persistent_retry is False
    assert default.runtime.tool_context.max_result_chars == 50_000
    assert default.runtime.tool_context.spill_dir is None


def test_a_journaled_agent_spills_cut_output_beside_its_journal(tmp_path: Path) -> None:
    provider = ScriptedProvider()
    journal = JsonlJournal(tmp_path / "s1" / "session.jsonl")
    agent = Agent(model=provider["scripted"], journal=journal, cwd=tmp_path)
    assert agent.runtime.tool_context.spill_dir == tmp_path / "s1"


def test_tools_must_have_distinct_names(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Duplicate tool name"):
        _ = _agent(tmp_path, tools=[_Tool(), _Tool()])


@pytest.mark.asyncio
async def test_without_a_compactor_nothing_shrinks(tmp_path: Path) -> None:
    _, agent = _agent(tmp_path)

    for shrinker in (agent.session.compactor, agent.session.pruner):
        assert isinstance(shrinker, NoShrinker)
        assert not shrinker.should_run()
        with pytest.raises(NotImplementedError):
            await shrinker.run()
    assert agent.runtime.compactor is None
    assert agent.runtime.compact_due is None
    assert agent.runtime.reply_room is None
    assert agent.runtime.prune is None


class _Dropper(NoShrinker):
    """A pruner the runtime can ask before each round."""

    def prune(self) -> bool:
        return False


def test_a_pruner_that_prunes_is_asked_by_the_runtime(tmp_path: Path) -> None:
    agent = Agent(model=ScriptedProvider()["scripted"], cwd=tmp_path, pruner=_Dropper)

    pruner = agent.session.pruner
    assert isinstance(pruner, _Dropper)
    assert agent.runtime.prune == pruner.prune


def test_a_summary_compactor_is_the_runtimes(tmp_path: Path) -> None:
    agent = Agent(
        model=ScriptedProvider()["scripted"],
        cwd=tmp_path,
        compactor=SummaryCompactor,
    )

    compactor = agent.session.compactor
    assert isinstance(compactor, SummaryCompactor)
    assert agent.runtime.compactor is compactor
    assert agent.runtime.compact_due == compactor.should_run
    assert agent.runtime.reply_room == compactor.reply_room


@pytest.mark.asyncio
async def test_an_overflow_is_compacted_and_called_again(tmp_path: Path) -> None:
    provider = ScriptedProvider(
        replies=[
            ContextOverflowError("too long"),
            reply("<summary>the gist</summary>"),
            reply("ok"),
        ],
    )
    agent = Agent(
        model=provider["scripted"],
        cwd=tmp_path,
        system_prompt="S",
        compactor=SummaryCompactor,
        agents_md=_config(tmp_path),
    )
    async with agent:
        agent.session.append(UserMessage(content="old"), AssistantMessage(content="ok"))
        assert await agent.send("go") == "ok"

    lead = provider.requests[2].records[0]
    assert isinstance(lead, UserMessage)
    assert "Summary:\nthe gist" in (lead.content or "")


@pytest.mark.asyncio
async def test_an_overflow_without_a_compactor_is_raised(tmp_path: Path) -> None:
    _, agent = _agent(tmp_path, ContextOverflowError("too long"))
    async with agent:
        with pytest.raises(ContextOverflowError):
            _ = await agent.send("go")


@pytest.mark.asyncio
async def test_an_agent_given_no_journal_keeps_nothing(tmp_path: Path) -> None:
    _, agent = _agent(tmp_path, reply("hi"))
    async with agent:
        assert await agent.send("go") == "hi"

    journal = agent.session.journal
    assert isinstance(journal, NoJournal)
    assert journal.path is None
    assert list(journal) == []
    assert list(journal.history()) == []
    journal.flush()


@pytest.mark.asyncio
async def test_a_session_is_named_by_its_directory(tmp_path: Path) -> None:
    _, first = _agent(tmp_path)
    _, second = _agent(tmp_path)
    async with JsonlJournal(tmp_path / "the-session" / "journal.jsonl") as journal:
        _, named = _agent(tmp_path, journal=journal)

    assert named.session.id == "the-session"
    assert len(first.session.id) == 32
    assert int(first.session.id, 16) >= 0
    assert first.session.id != second.session.id


@pytest.mark.asyncio
async def test_a_new_session_starts_its_journal_with_the_prompt(tmp_path: Path) -> None:
    journal = RecordingJournal()
    _, agent = _agent(tmp_path, journal=journal)
    async with agent:
        pass

    (start,) = _own(journal)
    assert isinstance(start, ContextClear)
    assert start.system_prompt == "S"
    assert start.history == ()


@pytest.mark.asyncio
async def test_a_resumed_session_is_what_the_model_sees_and_is_not_rewritten(
    tmp_path: Path,
) -> None:
    journal = RecordingJournal()
    window: list[SessionRecord] = [
        ContextClear(system_prompt="S"),
        UserMessage(content="remember 42"),
        AssistantMessage(content="noted"),
    ]
    journal.append(*window)
    provider, agent = _agent(tmp_path, reply("42"), journal=journal)
    async with agent:
        assert _own(journal) == window

        assert await agent.send("what was it?") == "42"

    assert _texts(provider.requests[0].records)[:3] == [
        "remember 42",
        "noted",
        "what was it?",
    ]


@pytest.mark.asyncio
async def test_a_call_still_running_when_its_session_ended_resumes_killed(
    tmp_path: Path,
) -> None:
    markers = types.Markers()
    journal = RecordingJournal()
    journal.append(
        ContextClear(system_prompt="S"),
        UserMessage(content="go"),
        ToolCall(call_id="c1", name="Echo"),
        UncategorizedToolResult(call_id="c1", content=markers.detached, stub="pending"),
        AssistantMessage(content="waiting"),
    )
    provider, agent = _agent(tmp_path, reply("ok"), journal=journal)
    async with agent:
        _ = await agent.send("and?")

    killed = [
        (r.content, r.stub)
        for r in journal
        if isinstance(r, UncategorizedToolResult)
        and r.call_id == f"c1{markers.arrival}"
    ]
    assert killed == [(markers.cancelled, "cancelled")]
    assert any(
        isinstance(r, UncategorizedToolResult) and r.content == markers.cancelled
        for r in provider.requests[0].records
    )


@pytest.mark.asyncio
async def test_each_turn_is_journaled_with_what_it_cost(tmp_path: Path) -> None:
    journal = RecordingJournal()
    tokens = TokenCount(
        request=10,
        response=5,
        cache_write=3,
        cache_write_1h=2,
        cache_read=1,
    )
    _, agent = _agent(
        tmp_path,
        reply("hello", tokens=tokens, spend=TokenCost(response=0.25)),
        journal=journal,
    )
    async with agent:
        _ = await agent.send("hi")

    records = _own(journal)
    assert [type(r) for r in records] == [
        ContextClear,
        UserMessage,
        TurnContext,
        AssistantMessage,
        TokenUsage,
    ]
    usage = records[-1]
    assert isinstance(usage, TokenUsage)
    assert dict(usage.info) == {
        "request": 10,
        "response": 5,
        "cache_write": 3,
        "cache_write_1h": 2,
        "cache_read": 1,
        "spend_usd": 0.25,
    }


@pytest.mark.asyncio
async def test_a_compaction_starts_a_new_window_in_the_journal(tmp_path: Path) -> None:
    journal = RecordingJournal()
    provider = ScriptedProvider(
        replies=[reply("<summary>the gist</summary>"), reply("resumed")],
    )
    agent = Agent(
        model=provider["scripted"],
        cwd=tmp_path,
        journal=journal,
        compactor=SummaryCompactor,
        agents_md=_config(tmp_path),
    )
    seen = tmp_path / "seen.txt"
    _ = seen.write_text("x")
    async with agent:
        agent.session.append(UserMessage(content="old"), AssistantMessage(content="ok"))
        agent.runtime.tool_context.files.saw(seen)
        await agent.session.compactor.run()

        start = next(iter(journal))
        assert seen not in agent.runtime.tool_context.files
    assert isinstance(start, ContextClear)
    lead = start.history[0]
    assert isinstance(lead, UserMessage)
    assert "Summary:\nthe gist" in (lead.content or "")
    assert all(type(r).__name__ in record_classes() for r in journal)


@pytest.mark.asyncio
async def test_a_session_resumes_from_its_journal_file(tmp_path: Path) -> None:
    path = tmp_path / "s.jsonl"
    async with JsonlJournal(path) as journal:
        _, agent = _agent(tmp_path, reply("noted"), journal=journal)
        async with agent:
            _ = await agent.send("remember 42")

    async with JsonlJournal(path) as journal:
        provider, agent = _agent(tmp_path, reply("42"), journal=journal)
        async with agent:
            assert await agent.send("what was it?") == "42"

    assert _texts(provider.requests[0].records)[:2] == ["remember 42", "noted"]


@pytest.mark.asyncio
async def test_a_resumed_session_tells_vendors_it_is_the_same_conversation(
    tmp_path: Path,
) -> None:
    path, other = tmp_path / "a" / "session.jsonl", tmp_path / "b" / "session.jsonl"
    sessions: list[str] = []
    for journaled in (path, path, other):
        async with JsonlJournal(journaled) as journal:
            provider, agent = _agent(tmp_path, reply("ok"), journal=journal)
            async with agent:
                _ = await agent.send("hi")
        sessions.append(provider.requests[0].session)
    first, resumed, elsewhere = sessions
    assert first == resumed != elsewhere
    assert str(uuid.UUID(first)) == first


def test_appended_records_are_turns_and_what_no_turn_holds_is_dropped(
    tmp_path: Path,
) -> None:
    _, agent = _agent(tmp_path)
    turn = (AssistantMessage(content="reading"), ToolCall(call_id="c1", name="Read"))
    result = UncategorizedToolResult(call_id="c1", content="text")

    agent.session.append(
        ContextClear(summary="the gist", history=(UserMessage(content="kept"),)),
        *turn,
        result,
        TokenUsage(info={"request": 1}),
        AssistantMessage(content="done"),
    )

    assert list(agent.session) == [
        UserMessage(content="the gist"),
        UserMessage(content="kept"),
        *turn,
        result,
        AssistantMessage(content="done"),
    ]
    entries = agent.runtime.context().messages
    assert len(entries) == 5
    # A stored turn says nothing of how it ended or what it cost.
    assert entries[2] == Response(
        records=turn,
        tokens=TokenCount(),
        spend=TokenCost(),
        stop_reason="",
    )


@pytest.mark.asyncio
async def test_clear_starts_over_and_reads_the_window_prompt_afresh(
    tmp_path: Path,
) -> None:
    journal = RecordingJournal()
    briefing = _Briefing(name="Briefing", said="index v1")
    provider, agent = _agent(
        tmp_path,
        reply("first"),
        reply("second"),
        tools=[_Tool(name="Plain"), briefing],
        journal=journal,
    )
    seen = tmp_path / "seen.txt"
    _ = seen.write_text("x")
    async with agent:
        _ = await agent.send("old news")
        agent.runtime.tool_context.files.saw(seen)
        briefing.said = "index v2"
        await agent.session.clear()

        assert list(agent.session) == []
        (start,) = _own(journal)
        assert isinstance(start, ContextClear)
        assert seen not in agent.runtime.tool_context.files
        assert agent.runtime.system == ("S", "index v2")
        _ = await agent.send("fresh")

    assert provider.requests[0].system[-1] == "index v1"
    assert provider.requests[1].system[-1] == "index v2"
    assert "old news" not in _texts(provider.requests[1].records)


@pytest.mark.asyncio
async def test_a_window_prompt_holds_the_prompt_then_agents_md_and_briefings(
    tmp_path: Path,
) -> None:
    """Read once at the start: a byte changed mid-window rewrites the whole cache."""
    _ = (tmp_path / "AGENTS.md").write_text("be terse\n")
    briefing = _Briefing(name="Briefing", said="index")
    _, agent = _agent(tmp_path, reply("1"), tools=[briefing, _Briefing(name="Quiet")])
    async with agent:
        framed = agent.runtime.system
        briefing.said = "index, saved since"
        _ = await agent.send("one")
        (turn,) = (r for r in agent.session if isinstance(r, TurnContext))

    text = directives(tmp_path, config=_config(tmp_path))
    assert "be terse" in text
    assert framed == ("S", f"{text}\n\nindex")
    assert agent.runtime.system == framed
    assert turn.agents_md is None


@pytest.mark.asyncio
async def test_with_nothing_standing_the_window_prompt_is_the_prompt(
    tmp_path: Path,
) -> None:
    _, agent = _agent(tmp_path, system_prompt="")
    async with agent:
        assert agent.runtime.system == ()
        agent.system_prompt = None
        assert agent.runtime.system == (default_system_prompt(),)


@pytest.mark.asyncio
async def test_a_compaction_reads_the_window_prompt_afresh(tmp_path: Path) -> None:
    briefing = _Briefing(name="Briefing", said="before")
    _, agent = _agent(tmp_path, tools=[briefing])
    async with agent:
        briefing.said = "after"
        assert agent.runtime.system == ("S", "before")
        agent.runtime.publish(types.CompactComplete())
        assert agent.runtime.system == ("S", "after")


def test_a_new_session_restates_where_and_as_what_it_runs(tmp_path: Path) -> None:
    _ = (tmp_path / "AGENTS.md").write_text("be terse\n")
    provider = ScriptedProvider(capability=_capability(cutoff="June 2026"))
    agent = Agent(model=provider["scripted"], cwd=tmp_path, agents_md=_config(tmp_path))

    (turn,) = _session(agent).updates()

    assert isinstance(turn, TurnContext)
    assert turn.cwd == str(tmp_path)
    assert turn.model == "scripted+fullcontext"
    # The root's AGENTS.md is the window's prompt, never a restated setting.
    assert turn.agents_md is None
    assert "be terse" in agent.runtime.system[-1]
    assert turn.environment is not None
    assert "Git repository: false" in turn.environment
    assert turn.environment.endswith("Knowledge cutoff: June 2026.")


def test_an_agent_without_the_environment_is_told_only_its_model(
    tmp_path: Path,
) -> None:
    """An agent serving outsiders tells them nothing of the host it runs on."""
    _ = (tmp_path / "AGENTS.md").write_text("be terse\n")
    provider = ScriptedProvider(capability=_capability(cutoff="June 2026"))
    agent = Agent(
        model=provider["scripted"],
        cwd=tmp_path,
        agents_md=AgentsMdConfig(tiers=frozenset()),
        environment=False,
    )
    session = _session(agent)

    (turn,) = session.updates()
    session.append(turn)

    assert isinstance(turn, TurnContext)
    assert (turn.cwd, turn.environment, turn.agents_md) == (None, None, None)
    assert turn.model == "scripted+fullcontext"
    assert str(tmp_path) not in turn.text()
    assert session.updates() == []


def test_settings_already_stated_are_not_stated_again(tmp_path: Path) -> None:
    _, agent = _agent(tmp_path)
    session = _session(agent)
    session.append(*session.updates())

    assert session.updates() == []


def test_each_setting_is_stated_again_only_when_it_changes(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    _, agent = _agent(tmp_path)
    session = _session(agent)
    session.append(*session.updates())

    agent.runtime.tool_context.cwd = tmp_path / "sub"
    (moved,) = session.updates()
    session.append(moved)
    agent.model = ScriptedProvider(capability=_capability(model_id="other"))["other"]
    (switched,) = session.updates()

    assert isinstance(moved, TurnContext)
    assert (moved.cwd, moved.environment, moved.model) == (
        str(tmp_path / "sub"),
        None,
        None,
    )
    assert isinstance(switched, TurnContext)
    assert (switched.cwd, switched.model) == (None, "other+fullcontext")


def test_a_directorys_agents_md_is_given_once_where_the_model_works(
    tmp_path: Path,
) -> None:
    _ = (tmp_path / "AGENTS.md").write_text("be terse\n")
    own = tmp_path / "own"
    own.mkdir()
    _ = (own / "AGENTS.md").write_text("be thorough\n")
    _ = (tmp_path / "x.py").write_text("x")
    _, agent = _agent(tmp_path)
    session = _session(agent)
    session.append(*session.updates())

    agent.runtime.tool_context.cwd = own
    moved, given = session.updates()
    session.append(moved, given)
    agent.runtime.tool_context.cwd = tmp_path
    back = session.updates()
    agent.runtime.tool_context.files.saw(tmp_path / "x.py")
    agent.runtime.tool_context.cwd = own
    again = [u for u in session.updates() if isinstance(u, SystemMessage)]

    assert isinstance(given, SystemMessage)
    assert given.subtype == "rules"
    assert "be thorough" in (given.content or "")
    assert "be terse" not in (given.content or "")
    assert [u for u in back if isinstance(u, SystemMessage)] == []
    assert again == []


def test_a_file_read_below_brings_its_directorys_agents_md(tmp_path: Path) -> None:
    deep = tmp_path / "pkg" / "deep"
    deep.mkdir(parents=True)
    _ = (tmp_path / "pkg" / "AGENTS.md").write_text("pkg-rule\n")
    _ = (deep / "a.py").write_text("x")
    _, agent = _agent(tmp_path)
    session = _session(agent)
    session.append(*session.updates())

    agent.runtime.tool_context.files.saw(deep / "a.py")
    (given,) = session.updates()

    assert isinstance(given, SystemMessage)
    assert "pkg-rule" in (given.content or "")


def test_the_hour_rolling_over_states_the_time_again(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, agent = _agent(tmp_path)
    session = _session(agent)
    monkeypatch.setattr(agent_module, "_now", lambda: "2026-10-24T13:00:00+00:00")
    session.append(*session.updates())
    monkeypatch.setattr(agent_module, "_now", lambda: "2026-10-24T13:59:59+00:00")
    same_hour = session.updates()
    monkeypatch.setattr(agent_module, "_now", lambda: "2026-10-24T14:00:00+00:00")
    (later,) = session.updates()

    assert same_hour == []
    assert isinstance(later, TurnContext)
    assert later.timestamp == "2026-10-24T14:00:00+00:00"
    # The time alone: nothing else changed.
    assert later.text().startswith("Current time: ")
    assert "\n" not in later.text()


def test_the_tools_allowed_are_stated_and_again_when_they_change(
    tmp_path: Path,
) -> None:
    _, agent = _agent(tmp_path, tools=[_Tool(name="a"), _Tool(name="b")])
    session = _session(agent)
    (everything,) = session.updates()
    session.append(everything)
    agent.allowed = frozenset({"b"})
    (gated,) = session.updates()
    session.append(gated)
    unchanged = session.updates()
    agent.allowed = frozenset()

    assert isinstance(everything, TurnContext)
    assert everything.tools is None
    assert isinstance(gated, TurnContext)
    assert gated.tools == ("b",)
    assert "Tools you may call: b" in gated.text()
    assert unchanged == []
    assert agent.runtime.allowed == frozenset()
    (none,) = session.updates()
    assert isinstance(none, TurnContext)
    assert "Tools you may call: none" in none.text()
    session.append(none)
    agent.allowed = None
    (lifted,) = session.updates()
    assert isinstance(lifted, TurnContext)
    assert "Tools you may call: a, b" in lifted.text()


def test_allowing_a_tool_the_agent_lacks_is_refused() -> None:
    provider = ScriptedProvider()
    with pytest.raises(ValueError, match=r"Unknown tools: Absent\."):
        _ = Agent(model=provider["scripted"], tools=[_Tool()], allowed=["Absent"])
    agent = Agent(model=provider["scripted"], tools=[_Tool()], allowed=["_Tool"])
    assert agent.allowed == frozenset({"_Tool"})
    agent.allowed = None
    assert agent.allowed is None


def test_a_rule_is_given_once_when_a_file_it_governs_is_seen(tmp_path: Path) -> None:
    rules = tmp_path / ".sagent" / "rules"
    rules.mkdir(parents=True)
    py = rules / "py.md"
    md = rules / "md.md"
    _ = py.write_text("---\npaths: ['**/*.py']\n---\npy-rule\n")
    _ = md.write_text("---\npaths: ['**/*.md']\n---\nmd-rule\n")
    for name in ("a.py", "b.py", "c.md"):
        _ = (tmp_path / name).write_text("x")
    _, agent = _agent(tmp_path)
    session = _session(agent)
    files = agent.runtime.tool_context.files
    session.append(*session.updates())

    files.saw(tmp_path / "a.py")
    (given,) = session.updates()
    session.append(given)
    files.saw(tmp_path / "b.py")
    again = session.updates()
    files.saw(tmp_path / "c.md")
    (second,) = session.updates()

    assert isinstance(given, SystemMessage)
    assert given.subtype == "rules"
    assert "py-rule" in (given.content or "")
    assert given.extra == {"rules": (str(py.resolve()),)}
    assert again == []
    assert isinstance(second, SystemMessage)
    assert second.extra == {"rules": (str(md.resolve()),)}
    assert "py-rule" not in (second.content or "")


def test_a_tools_reminder_is_told_once_and_again_when_it_changes(
    tmp_path: Path,
) -> None:
    reminding = _Reminding(name="Peers")
    _, agent = _agent(tmp_path, tools=[_Tool(), reminding])
    session = _session(agent)
    session.append(*session.updates())
    assert session.updates() == []

    reminding.told = "peers: a"
    (told,) = session.updates()
    session.append(told)
    unchanged = session.updates()
    reminding.told = "peers: a, b"
    (changed,) = session.updates()

    assert told == SystemMessage(
        timestamp=told.timestamp,
        content="peers: a",
        subtype="reminder",
        extra={"tool": "Peers"},
    )
    assert unchanged == []
    assert isinstance(changed, SystemMessage)
    assert changed.content == "peers: a, b"


def test_a_gated_off_tool_reminds_of_nothing(tmp_path: Path) -> None:
    reminding = _Reminding(name="Peers", told="reply by Peers")
    _, agent = _agent(tmp_path, tools=[_Tool(), reminding])
    agent.allowed = frozenset({"_Tool"})
    session = _session(agent)
    gated = session.updates()
    agent.allowed = None
    opened = session.updates()

    assert [u for u in gated if isinstance(u, SystemMessage)] == []
    assert [u.content for u in opened if isinstance(u, SystemMessage)] == [
        "reply by Peers",
    ]


def test_a_reminder_that_empties_is_withdrawn_once(tmp_path: Path) -> None:
    reminding = _Reminding(name="Peers", told="peers: a")
    _, agent = _agent(tmp_path, tools=[reminding])
    session = _session(agent)
    session.append(*session.updates())

    reminding.told = ""
    (withdrawn,) = session.updates()
    session.append(withdrawn)
    quiet = session.updates()
    reminding.told = "peers: b"
    (back,) = session.updates()

    assert withdrawn == SystemMessage(
        timestamp=withdrawn.timestamp,
        content="No longer in force: peers: a",
        subtype="reminder",
        extra={"tool": "Peers", "withdrawn": True},
    )
    assert quiet == []
    assert isinstance(back, SystemMessage)
    assert back.content == "peers: b"


@pytest.mark.asyncio
@pytest.mark.parametrize("ends", ["compaction", "clear"])
async def test_a_new_window_returns_to_where_the_session_started(
    tmp_path: Path,
    ends: str,
) -> None:
    (tmp_path / "elsewhere").mkdir()
    _, agent = _agent(tmp_path)
    async with agent:
        agent.runtime.tool_context.cwd = tmp_path / "elsewhere"
        if ends == "clear":
            await agent.session.clear()
        else:
            agent.runtime.publish(types.CompactComplete())
        notes = [
            r
            for r in agent.session
            if isinstance(r, SystemMessage) and r.subtype == "moved"
        ]

    assert agent.runtime.tool_context.cwd == tmp_path
    assert [n.content for n in notes] == [
        f"Working directory reset to {tmp_path}; it was {tmp_path / 'elsewhere'}.",
    ]


@pytest.mark.asyncio
async def test_a_window_at_its_root_says_nothing_of_moving(tmp_path: Path) -> None:
    _, agent = _agent(tmp_path)
    async with agent:
        agent.runtime.publish(types.CompactComplete())

        assert [r for r in agent.session if isinstance(r, SystemMessage)] == []


def test_the_window_prompt_holds_the_roots_agents_md_wherever_the_model_went(
    tmp_path: Path,
) -> None:
    _ = (tmp_path / "AGENTS.md").write_text("root-rule\n")
    (tmp_path / "away").mkdir()
    _ = (tmp_path / "away" / "AGENTS.md").write_text("away-rule\n")
    _, agent = _agent(tmp_path)
    agent.runtime.tool_context.cwd = tmp_path / "away"

    _session(agent).frame()

    assert "root-rule" in agent.runtime.system[-1]
    assert "away-rule" not in agent.runtime.system[-1]


def test_the_model_is_named_with_its_window(tmp_path: Path) -> None:
    prices = PriceCatalog(window=1_000_000)
    capability = ModelCapability(
        model_id="opus",
        limits=ModelLimits(max_request_tokens=1_000_000, max_response_tokens=8_000),
        prices=prices,
        default=ModelSettings(context_size=prices["auto", "fullcontext"]),
    )
    provider = ScriptedProvider(capability=capability)
    _, agent = _agent(tmp_path, provider=provider)
    session = _session(agent)
    (first,) = session.updates()
    session.append(first)
    agent.model.settings = replace(
        agent.model.settings,
        context_size=prices["auto", 258_000],
    )
    (smaller,) = session.updates()

    assert isinstance(first, TurnContext)
    assert first.model == "opus+fullcontext"
    assert isinstance(smaller, TurnContext)
    assert smaller.model == "opus+258k"


def _two_models() -> ModelCatalog:
    """Return a catalog whose one provider serves ``a`` and ``b``."""
    models = ModelTable(rows=(_capability("a"), _capability("b")))
    return ModelCatalog(
        factories={"scripted": partial(ScriptedProvider, models=models)},
    )


def test_a_model_switched_away_from_is_cached_as_it_was_left(tmp_path: Path) -> None:
    catalog = _two_models()
    agent = Agent(model=catalog["scripted"]["a+2k"], cwd=tmp_path)
    left = agent.model.settings

    agent.model = catalog["scripted"]["b"]
    agent.model = catalog["scripted"]["b"]

    assert list(agent.model_cache) == ["scripted/a"]
    assert agent.model_cache["scripted/a"].settings is left
    assert left.context_size.tokens == 2_000


@pytest.mark.asyncio
async def test_a_resumed_agent_recalls_its_model_its_cache_and_its_prompt(
    tmp_path: Path,
) -> None:
    catalog = _two_models()
    path = tmp_path / "s" / "session.jsonl"
    async with JsonlJournal(path) as journal:
        agent = Agent(
            model=catalog["scripted"]["a+1k"],
            journal=journal,
            system_prompt="mine",
            cwd=tmp_path,
            agents_md=_config(tmp_path),
        )
        async with agent:
            agent.model = catalog["scripted"]["b+2k"]

    async with JsonlJournal(path) as journal:
        again = Agent(
            model=catalog["scripted"]["a"],
            journal=journal,
            cwd=tmp_path,
            agents_md=_config(tmp_path),
        )
        await again.recall(catalog)

    assert (
        again.model.capability.model_id,
        again.model.settings.context_size.tokens,
    ) == ("b", 2_000)
    assert list(again.model_cache) == ["scripted/a"]
    assert again.model_cache["scripted/a"].settings.context_size.tokens == 1_000
    assert again.system_prompt == "mine"


@pytest.mark.asyncio
async def test_an_agent_with_nothing_stood_recalls_nothing(tmp_path: Path) -> None:
    catalog = _two_models()
    model = catalog["scripted"]["a"]
    async with JsonlJournal(tmp_path / "s" / "session.jsonl") as journal:
        agent = Agent(model=model, journal=journal, cwd=tmp_path)
        await agent.recall(catalog)
    assert (agent.model, agent.model_cache, agent.system_prompt) == (model, {}, None)


@pytest.mark.asyncio
async def test_the_prompt_stands_once_and_the_default_withdraws_it(
    tmp_path: Path,
) -> None:
    journal = RecordingJournal()
    agent = Agent(
        model=ScriptedProvider()["scripted"],
        journal=journal,
        system_prompt="mine",
        cwd=tmp_path,
    )
    async with agent:
        _session(agent).persist()
        agent.system_prompt = None
        _session(agent).persist()
        _session(agent).persist()

    stood = [
        (r.kind, r.payload.get("text"))
        for r in journal.history()
        if isinstance(r, UncategorizedRecord)
        and r.payload.get("standing") == "system_prompt"
    ]
    assert stood == [("system_prompt", "mine"), ("withdrawn", None)]


@pytest.mark.asyncio
async def test_the_model_stands_once_until_it_changes(tmp_path: Path) -> None:
    catalog = _two_models()
    journal = RecordingJournal()
    agent = Agent(model=catalog["scripted"]["a"], journal=journal, cwd=tmp_path)
    async with agent:
        _session(agent).persist()
        agent.model.settings = replace(
            agent.model.settings,
            context_size=agent.model.capability.prices["auto", 3_000],
        )
        _session(agent).persist()

    stood = [
        r.payload["model"]
        for r in journal.history()
        if isinstance(r, UncategorizedRecord) and r.kind == "model"
    ]
    assert stood == [
        "scripted/a+fullcontext+nothink",  # codespell:ignore nothink
        "scripted/a+3k+nothink",  # codespell:ignore nothink
    ]


@pytest.mark.asyncio
async def test_the_journal_keeps_local_time_with_its_offset(tmp_path: Path) -> None:
    journal = RecordingJournal()
    _, agent = _agent(tmp_path, journal=journal)
    async with agent:
        pass

    (start,) = _own(journal)
    assert isinstance(start, ContextClear)
    assert start.timestamp is not None
    assert datetime.fromisoformat(start.timestamp).utcoffset() is not None
    assert "." not in start.timestamp


@pytest.mark.asyncio
async def test_the_settings_ride_the_first_round_only(tmp_path: Path) -> None:
    provider, agent = _agent(tmp_path, reply("1"), reply("2"))
    async with agent:
        _ = await agent.send("one")
        _ = await agent.send("two")

    stated = [r for r in provider.requests[1].records if isinstance(r, TurnContext)]
    assert len(stated) == 1


@pytest.mark.asyncio
async def test_a_standing_record_outlives_a_new_window_until_withdrawn(
    tmp_path: Path,
) -> None:
    journal = RecordingJournal()
    _, agent = _agent(tmp_path, journal=journal)
    record = UncategorizedRecord(kind="note", payload={"what": "child"})
    async with agent:
        agent.session.stand("child", record)
        agent.session.stand("other", UncategorizedRecord(kind="note"))
        agent.session.stand("other", None)
        agent.session.append(UserMessage(content="before the clear"))
        await agent.runtime.schedule(_nothing)
        await agent.session.clear()

    window = _own(journal)
    assert isinstance(window[0], ContextClear)
    assert window[1:] == [
        UncategorizedRecord(
            kind="note",
            payload={"what": "child", "standing": "child"},
        ),
    ]
    assert [r for r in agent.session.standing() if r.kind == "note"] == [record]


@pytest.mark.asyncio
async def test_a_resumed_session_reads_what_stands_and_what_was_withdrawn(
    tmp_path: Path,
) -> None:
    journal = RecordingJournal()
    _, first = _agent(tmp_path, journal=journal)
    async with first:
        first.session.stand("kept", UncategorizedRecord(kind="note", payload={"n": 1}))
        first.session.stand("gone", UncategorizedRecord(kind="note", payload={"n": 2}))
        first.session.stand("gone", None)
    journal.append(UncategorizedRecord(kind="note", payload={"n": 3}))

    _, resumed = _agent(tmp_path, journal=journal)
    async with resumed:
        assert [r for r in resumed.session.standing() if r.kind == "note"] == [
            UncategorizedRecord(kind="note", payload={"n": 1}),
        ]


@pytest.mark.asyncio
async def test_messages_from_other_agents_start_the_round_count_over(
    tmp_path: Path,
) -> None:
    _, agent = _agent(
        tmp_path,
        reply(calls=[call("Nope", "a")]),
        reply("one"),
        reply(calls=[call("Nope", "b")]),
        reply("two"),
        max_rounds=2,
    )
    seen = _Seen()
    async with agent:
        agent.runtime.observers.append(seen)
        for text in ("first", "second"):
            agent.runtime.push(AgentToAgentMessage(sender="peer", content=text))
            await seen.idle()

    assert seen.count(types.Halt) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("carried", [False, True], ids=["usage", "new-window"])
async def test_a_resumed_session_keeps_what_it_spent(
    tmp_path: Path,
    carried: bool,
) -> None:
    journal = RecordingJournal()
    _, first = _agent(
        tmp_path,
        reply("a", spend=TokenCost(response=5.0)),
        reply("<summary>gist</summary>"),
        journal=journal,
    )
    async with first:
        _ = await first.send("spend")
        if carried:
            await first.session.clear()
    assert first.spend_usd == 5.0

    provider, resumed = _agent(tmp_path, reply("b"), journal=journal, max_spend_usd=5.0)
    async with resumed:
        assert resumed.spend_usd == 5.0
        with pytest.raises(BudgetExhaustedError):
            _ = await resumed.send("more")

    assert provider.requests == []


@pytest.mark.asyncio
async def test_coalesced_turns_grow_the_journal_linearly(tmp_path: Path) -> None:
    sizes: list[int] = []
    async with JsonlJournal(tmp_path / "s.jsonl") as journal:
        _, agent = _agent(tmp_path, *[reply("ok")] * 30, journal=journal)
        async with agent:
            seen = _Seen()
            agent.runtime.observers.append(seen)
            for i in range(30):
                agent.runtime.push(UserMessage(content=f"part one of {i}"))
                agent.runtime.push(UserMessage(content=f"part two of {i}"))
                await seen.idle()
                sizes.append((tmp_path / "s.jsonl").stat().st_size)

    first, second = sizes[14] - sizes[0], sizes[29] - sizes[15]
    assert second < 1.5 * first


@pytest.mark.asyncio
async def test_a_coalesced_turn_resumes_as_one_message(tmp_path: Path) -> None:
    journal = RecordingJournal()
    _, first = _agent(tmp_path, reply("ok"), journal=journal)
    async with first:
        seen = _Seen()
        first.runtime.observers.append(seen)
        first.runtime.push(UserMessage(content="part one"))
        first.runtime.push(UserMessage(content="part two"))
        await seen.idle()

    provider, resumed = _agent(tmp_path, reply("again"), journal=journal)
    async with resumed:
        _ = await resumed.send("next")

    assert _texts(provider.requests[0].records)[:3] == [
        "part one\n\npart two",
        "ok",
        "next",
    ]


@pytest.mark.asyncio
async def test_a_message_merged_into_an_unanswered_one_journals_only_itself(
    tmp_path: Path,
) -> None:
    journal = RecordingJournal()
    _, first = _agent(tmp_path, ValueError("model down"), reply("ok"), journal=journal)
    async with first:
        with pytest.raises(ValueError, match="model down"):
            _ = await first.send("part one")
        assert await first.send("part two") == "ok"

    assert sum(isinstance(r, ContextClear) for r in journal) == 1
    assert UserMessage(content="part two") in [
        replace(r, timestamp=None) for r in journal if isinstance(r, UserMessage)
    ]
    provider, resumed = _agent(tmp_path, reply("again"), journal=journal)
    async with resumed:
        _ = await resumed.send("next")

    assert _texts(provider.requests[0].records)[:3] == [
        "part one\n\npart two",
        "ok",
        "next",
    ]


@pytest.mark.asyncio
async def test_a_resumed_session_signs_in_as_the_account_it_stood(
    tmp_path: Path,
) -> None:
    def per_account(account: str | None) -> dict[str, partial[ScriptedProvider]]:
        del account
        return {"vendor": partial(ScriptedProvider, capability=_capability("wide"))}

    journal = RecordingJournal()
    catalog = ModelCatalog(factories=per_account)
    first = Agent(
        model=catalog.signed_in("vendor", account="work")["wide"],
        journal=journal,
        cwd=tmp_path,
        agents_md=_config(tmp_path),
        account="work",
    )
    async with first:
        pass

    fresh = ModelCatalog(factories=per_account)
    again = Agent(
        model=fresh["vendor"]["wide"],
        journal=journal,
        cwd=tmp_path,
        agents_md=_config(tmp_path),
    )
    await again.recall(fresh)

    assert again.account == "work"
    assert again.model.provider is fresh.signed_in("vendor", account="work")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tail",
    [
        [
            reply(calls=[call("Nope", "c1")]),
            UncategorizedToolResult(call_id="c1", content="[detached]", stub="pending"),
        ],
        [UserMessage(content="are you there?")],
        [
            reply(calls=[call("Nope", "c1")]),
            UncategorizedToolResult(call_id="c1", content="ok"),
        ],
    ],
    ids=["job-running-at-quit", "unanswered-message", "unanswered-result"],
)
async def test_a_resumed_session_waits_for_the_user_before_answering(
    tmp_path: Path,
    tail: list[ModelContextEvent],
) -> None:
    """What the last process left unanswered is answered with the user's next message."""
    journal = RecordingJournal()
    _, first = _agent(tmp_path, reply("hello"), journal=journal)
    async with first:
        _ = await first.send("hi")
        for record in tail:
            _ = first.runtime.append_history(record)

    provider = ScriptedProvider(replies=[reply("answered")])
    _, resumed = _agent(tmp_path, provider=provider, journal=journal)
    # As ``recall`` does: the model the session stood replaces the one built.
    resumed.model = provider["scripted"]
    async with resumed:
        assert resumed.runtime.inbox.gate_armed
        assert provider.requests == []
        assert await resumed.send("next") == "answered"

    (asked,) = provider.requests
    assert _texts(asked.records)[-1].endswith("next")


@pytest.mark.asyncio
async def test_a_cap_tripped_outside_a_turn_does_not_haunt_the_next(
    tmp_path: Path,
) -> None:
    provider = ScriptedProvider(
        replies=[reply(calls=[call("Nope", "a")]), reply(calls=[call("Nope", "b")])],
    )
    _, agent = _agent(tmp_path, provider=provider, max_rounds=1)
    provider.replies.append(lambda request: _halting(agent, request))
    seen = _Seen()
    async with agent:
        agent.runtime.observers.append(seen)
        agent.runtime.push(AgentToAgentMessage(sender="peer", content="work"))
        await seen.until(lambda: seen.count(types.Halt) > 0)

        with pytest.raises(asyncio.CancelledError):
            _ = await agent.send("then this")


def _halting(agent: Agent, request: Request) -> Response:
    del request
    agent.runtime.halt()
    return reply("halted by the user")


@pytest.mark.asyncio
async def test_an_agent_closes_the_models_it_was_given(tmp_path: Path) -> None:
    provider = _Counting()
    first, second, third = (provider["scripted"] for _ in range(3))
    agent = Agent(model=first, cwd=tmp_path, agents_md=_config(tmp_path))
    agent.model = second
    async with agent:
        agent.model = third
        agent.model = third
        await agent.runtime.schedule(_nothing)

        assert (first.closed, second.closed, third.closed) == (1, 1, 0)
    assert third.closed == 1


async def _nothing() -> None:
    return


def test_an_edit_of_earlier_history_writes_a_new_window(tmp_path: Path) -> None:
    journal = RecordingJournal()
    _, agent = _agent(tmp_path, journal=journal)
    session = _session(agent)
    session.resume()
    first = agent.runtime.append_history(UserMessage(content="one"))
    session.persist()
    _ = agent.runtime.append_history(reply("noted", spend=TokenCost(response=0.5)))
    session.persist()
    edited = UserMessage(content="one, edited")

    _ = agent.runtime.append_splice(
        mask=(MaskRange(session_id="", lo=first.ordinal, hi=first.ordinal),),
        payload=(edited,),
        strategy="edit",
        discards_content=True,
    )
    session.persist()

    (window,) = _own(journal)
    assert isinstance(window, ContextClear)
    assert window.history == (edited, AssistantMessage(content="noted"))


class _FailingJournal(RecordingJournal):
    """Raises on the next ``fails`` appends, as a full disk does."""

    def __init__(self) -> None:
        super().__init__()
        self.fails = 0

    @override
    def append(self, *records: SessionRecord) -> None:
        if self.fails:
            self.fails -= 1
            raise OSError("No space left on device")
        super().append(*records)


@pytest.mark.parametrize("edit", [False, True], ids=["append", "new-window"])
def test_what_a_failed_write_dropped_the_next_write_writes(
    tmp_path: Path,
    edit: bool,
) -> None:
    journal = _FailingJournal()
    _, agent = _agent(tmp_path, journal=journal)
    session = _session(agent)
    session.resume()
    first = agent.runtime.append_history(UserMessage(content="one"))
    if edit:
        _ = agent.runtime.append_splice(
            mask=(MaskRange(session_id="", lo=first.ordinal, hi=first.ordinal),),
            payload=(UserMessage(content="one, edited"),),
            strategy="edit",
            discards_content=True,
        )
    journal.fails = 1

    with pytest.raises(OSError, match="No space left"):
        session.persist()
    session.persist()

    _, resumed = _agent(tmp_path, journal=journal)
    _session(resumed).resume()
    said = [r.content for r in _session(resumed) if isinstance(r, UserMessage)]
    assert said == (["one, edited"] if edit else ["one"])


def test_a_removal_from_earlier_history_writes_a_new_window(tmp_path: Path) -> None:
    journal = RecordingJournal()
    _, agent = _agent(tmp_path, journal=journal)
    session = _session(agent)
    session.resume()
    _ = agent.runtime.append_history(UserMessage(content="one"))
    second = agent.runtime.append_history(UserMessage(content="two"))
    session.persist()

    _ = agent.runtime.append_splice(
        mask=(MaskRange(session_id="", lo=second.ordinal, hi=second.ordinal),),
        strategy="edit",
        discards_content=True,
    )
    session.persist()

    (window,) = _own(journal)
    assert isinstance(window, ContextClear)
    assert window.history == (UserMessage(content="one"),)


def test_a_pruned_result_is_journaled_as_itself_and_resumes_pruned(
    tmp_path: Path,
) -> None:
    journal = RecordingJournal()
    _, agent = _agent(tmp_path, journal=journal)
    session = _session(agent)
    session.resume()
    _ = agent.runtime.append_history(UserMessage(content="go"))
    _ = agent.runtime.append_history(reply(calls=[call("Echo", "c1")]))
    _ = agent.runtime.append_history(
        UncategorizedToolResult(call_id="c1", content="long output", is_error=True),
    )
    session.persist()

    drop(agent.runtime, [2], stub="[pruned]")
    _ = agent.runtime.append_history(reply("done"))
    session.persist()

    assert sum(isinstance(r, ContextClear) for r in journal.history()) == 1
    _, resumed = _agent(tmp_path, journal=journal)
    _session(resumed).resume()
    assert list(_session(resumed)) == list(session)
    assert UncategorizedToolResult(
        call_id="c1",
        content="[pruned]",
        is_error=True,
    ) in list(session)


def _result(n: int) -> UncategorizedToolResult:
    return UncategorizedToolResult(
        call_id=f"c{n}",
        content=f"output {n}",
        is_error=True,
    )


def test_a_result_pruned_in_the_pass_it_came_is_journaled_whole(
    tmp_path: Path,
) -> None:
    journal = RecordingJournal()
    _, agent = _agent(tmp_path, journal=journal)
    session = _session(agent)
    session.resume()
    _ = agent.runtime.append_history(UserMessage(content="go"))
    _ = agent.runtime.append_history(reply(calls=[call("Echo", "c1")]))
    _ = agent.runtime.append_history(_result(1))
    drop(agent.runtime, [2], stub="[pruned]")
    _ = agent.runtime.append_history(reply(calls=[call("Echo", "c2")]))
    _ = agent.runtime.append_history(_result(2))
    drop(agent.runtime, [4], stub="[pruned]")
    _ = agent.runtime.append_history(reply("done"))
    session.persist()

    kept = list(journal.history())
    assert _result(1) in kept
    assert _result(2) in kept
    assert sum(isinstance(r, ContextClear) for r in kept) == 1
    _, resumed = _agent(tmp_path, journal=journal)
    _session(resumed).resume()
    assert list(_session(resumed)) == list(session)
    stubs = [r for r in session if isinstance(r, ToolResult) and r.text() == "[pruned]"]
    assert [r.call_id for r in stubs] == ["c1", "c2"]


def test_a_result_edited_away_in_the_pass_it_came_is_journaled_whole(
    tmp_path: Path,
) -> None:
    journal = RecordingJournal()
    _, agent = _agent(tmp_path, journal=journal)
    session = _session(agent)
    session.resume()
    first = agent.runtime.append_history(UserMessage(content="go"))
    _ = agent.runtime.append_history(reply(calls=[call("Echo", "c1")]))
    last = agent.runtime.append_history(_result(1))
    _ = agent.runtime.append_splice(
        mask=(MaskRange(session_id="", lo=first.ordinal, hi=last.ordinal),),
        payload=(UserMessage(content="summary"),),
        strategy="edit",
        discards_content=True,
    )
    session.persist()

    assert _result(1) in list(journal.history())
    (window,) = _own(journal)
    assert isinstance(window, ContextClear)
    assert window.history == (UserMessage(content="summary"),)


def test_a_reply_in_an_appending_pass_still_journals_what_it_cost(
    tmp_path: Path,
) -> None:
    journal = RecordingJournal()
    _, agent = _agent(tmp_path, journal=journal)
    session = _session(agent)
    session.resume()
    _ = agent.runtime.append_history(UserMessage(content="one"))
    session.persist()

    _ = agent.runtime.append_history(reply("noted", spend=TokenCost(response=0.5)))
    last = agent.runtime.append_history(UserMessage(content="two"))
    _ = agent.runtime.append_splice(insert_after=last, strategy="note")
    session.persist()

    assert UserMessage(content="two") in list(journal)
    usage = [r for r in journal if isinstance(r, TokenUsage)]
    assert [u.info["spend_usd"] for u in usage] == [0.5]


def test_the_default_prompt_reads_as_utf8_whatever_the_locale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    latin1 = _Latin1()
    monkeypatch.setattr(Path, "read_text", latin1)

    _ = default_system_prompt()

    assert latin1.asked == {"system_prompt.md": "utf-8"}


@dataclass(slots=True, kw_only=True)
class _Latin1:
    """Read text as a latin-1 locale would, and note the encodings asked for."""

    asked: dict[str, str | None] = field(default_factory=dict)

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
        self.asked[path.name] = encoding
        return path.read_bytes().decode(encoding or "latin-1")


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
