"""Tests that the agent's parts meet the protocols they are used through."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from sagent.agent.agent import Agent, NoJournal, NoShrinker, RuntimeSession
from sagent.agent.agents_md import AgentsMdConfig
from sagent.agent.compactor import SummaryCompactor
from sagent.agent.pruner import Pruner
from sagent.agent.types import (
    Agent as AgentProtocol,
    Factory,
    Journal,
    Session,
    Shrinker,
)
from sagent.model.scripted import ScriptedProvider, reply
from sagent.session.jsonl import JsonlJournal
from sagent.session.recording import RecordingJournal
from sagent.session.types import (
    AssistantMessage,
    ContextClear,
    UncategorizedRecord,
    UserMessage,
)


if TYPE_CHECKING:
    from pathlib import Path


def _agent(tmp_path: Path) -> Agent:
    return Agent(
        model=ScriptedProvider(replies=[reply("hello")])["scripted"],
        cwd=tmp_path,
        system_prompt="S",
        agents_md=AgentsMdConfig(system_dir=tmp_path / "s", user_dir=tmp_path / "u"),
    )


@pytest.mark.asyncio
async def test_an_agent_is_one_and_runs_with_async_with(tmp_path: Path) -> None:
    agent: AgentProtocol = _agent(tmp_path)

    async with agent as entered:
        assert entered is agent
        assert await agent.send("hi") == "hello"

    assert agent.spend_usd == 0.0
    assert dict(agent.tools) == {}
    assert agent.system_prompt == "S"


@pytest.mark.asyncio
async def test_a_runtime_session_is_one_and_iterates_from_the_last_clear(
    tmp_path: Path,
) -> None:
    session: Session = RuntimeSession(
        _agent(tmp_path),
        journal=RecordingJournal(),
        compactor=NoShrinker,
        pruner=NoShrinker,
    )

    session.append(
        UserMessage(content="old"),
        ContextClear(history=(UserMessage(content="kept"),)),
        AssistantMessage(content="new"),
    )
    session.stand("k", UncategorizedRecord(kind="note"))

    assert list(session) == [
        UserMessage(content="old"),
        UserMessage(content="kept"),
        AssistantMessage(content="new"),
    ]
    assert session.standing() == [UncategorizedRecord(kind="note")]
    assert len(session.id) == 32


@pytest.mark.asyncio
async def test_each_journal_is_one_and_iterates_from_the_last_clear(
    tmp_path: Path,
) -> None:
    async with JsonlJournal(tmp_path / "s.jsonl") as jsonl:
        journals: list[Journal] = [RecordingJournal(), jsonl]
        for journal in journals:
            journal.append(
                UserMessage(content="before"),
                ContextClear(system_prompt="S"),
                UserMessage(content="after"),
            )
            journal.flush()

            assert list(journal) == [
                ContextClear(system_prompt="S"),
                UserMessage(content="after"),
            ]
    assert [j.path for j in journals] == [None, tmp_path / "s.jsonl"]


@pytest.mark.asyncio
async def test_each_shrinker_is_one_and_a_class_builds_it(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    factories: list[Factory[Shrinker]] = [SummaryCompactor, Pruner, NoShrinker]

    shrinkers: list[Shrinker] = [factory(agent) for factory in factories]

    assert [type(s) for s in shrinkers] == [SummaryCompactor, Pruner, NoShrinker]
    assert [s.should_run() for s in shrinkers] == [False, False, False]
    with pytest.raises(ValueError, match="no instructions"):
        await shrinkers[1].run("keep this")
    with pytest.raises(NotImplementedError):
        await shrinkers[2].run()


@pytest.mark.parametrize(
    ("protocol", "implementations"),
    [
        (AgentProtocol, (Agent,)),
        (Session, (RuntimeSession,)),
        (Journal, (RecordingJournal, JsonlJournal, NoJournal)),
        (Shrinker, (SummaryCompactor, Pruner, NoShrinker)),
    ],
)
def test_each_implementation_has_every_member_its_protocol_names(
    tmp_path: Path,
    protocol: type,
    implementations: tuple[type, ...],
) -> None:
    agent = _agent(tmp_path)
    built: dict[type, object] = {
        Agent: agent,
        RuntimeSession: agent.session,
        RecordingJournal: RecordingJournal(),
        NoJournal: NoJournal(),
        JsonlJournal: JsonlJournal(tmp_path / "j.jsonl"),
        SummaryCompactor: SummaryCompactor(agent),
        Pruner: Pruner(agent),
        NoShrinker: NoShrinker(agent),
    }
    names = {
        n
        for n in (*vars(protocol), *protocol.__annotations__)
        if not n.startswith("_") or n == "__iter__"
    }

    assert names
    for implementation in implementations:
        assert names <= set(dir(built[implementation])), implementation


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
