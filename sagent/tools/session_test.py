"""Tests for the Session tool: an agent reading and editing its own session."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import asyncio

import pytest

from sagent.agent.agent import Agent, NoShrinker
from sagent.agent.compactor import SummaryCompactor
from sagent.agent.team import Team
from sagent.model.capability import ModelCapability, ModelLimits, ModelSettings
from sagent.model.cost import PriceCatalog, PriceCondition, TokenPrice
from sagent.model.scripted import ScriptedProvider
from sagent.providers.catalog import ModelCatalog
from sagent.runtime import types
from sagent.session.recording import RecordingJournal
from sagent.session.types import (
    AgentToAgentMessage,
    AssistantMessage,
    ContextClear,
    SystemMessage,
    Thinking,
    ToolCall,
    ToolResult,
    TurnContext,
    UncategorizedRecord,
    UncategorizedToolResult,
    UserMessage,
)
from sagent.tools.session import Session
from sagent.tools.types import ToolContext


if TYPE_CHECKING:
    from pathlib import Path

    from treekle import PlainTree

    from sagent.agent.types import Factory, Shrinker


# Anthropic's shape: a read is a tenth of input, a five-minute write a quarter more.
_CACHED = TokenPrice(
    request=3.0,
    response=15.0,
    cache_write=3.75,
    cache_write_1h=6.0,
    cache_read=0.3,
)


def _setup(
    tmp_path: Path,
    *,
    price: TokenPrice | None = _CACHED,
    compactor: Factory[Shrinker] = SummaryCompactor,
    journal: RecordingJournal | None = None,
) -> tuple[Agent, Session]:
    prices = PriceCatalog(
        prices={} if price is None else {PriceCondition(): price},
        window=200_000,
    )
    capability = ModelCapability(
        model_id="priced",
        limits=ModelLimits(max_request_tokens=200_000, max_response_tokens=8_000),
        prices=prices,
        default=ModelSettings(context_size=prices["auto", "fullcontext"]),
    )
    catalog = ModelCatalog(
        factories={"scripted": partial(ScriptedProvider, capability=capability)},
    )
    team = Team(catalog=catalog)
    agent = Agent(
        model=catalog["scripted"]["priced"],
        cwd=tmp_path,
        compactor=compactor,
        journal=journal,
        system_prompt="",
    )
    _ = team.join(agent)
    agent.session.append(
        UserMessage(content="go"),
        ToolCall(call_id="c0", name="Read", arguments={"file_path": "a.py"}),
        UncategorizedToolResult(call_id="c0", content="a" * 4_000),
        ToolCall(call_id="c1", name="Grep"),
        UncategorizedToolResult(call_id="c1", content="b" * 400, stub="cancelled"),
        AssistantMessage(content="read it"),
        AgentToAgentMessage(content="peer says", sender="Agent_0", recipient="Agent"),
        SystemMessage(content="a note"),
        TurnContext(cwd="/w"),
    )
    return agent, Session(team=team)


async def _ask(
    tool: Session,
    tmp_path: Path,
    arguments: dict[str, PlainTree],
    *,
    label: str = "Agent",
) -> str:
    result = await tool.call(ToolContext(cwd=tmp_path, label=label), arguments)
    # The runtime stamps the call's id; a tool leaves it empty.
    assert (result.call_id, result.is_error) == ("", None)
    return result.text()


@pytest.mark.parametrize(
    ("arguments", "summary", "key"),
    [
        ({"action": "list"}, "Session list", None),
        ({"action": "prune", "ids": [3, 7]}, "Session prune 3 7", "*"),
        ({"action": "compact"}, "Session compact", "*"),
        ({"action": "clear"}, "Session clear", "*"),
        ({"action": "read", "ids": "x"}, "Session read", None),
    ],
)
def test_the_summary_names_the_action_and_edits_run_alone(
    tmp_path: Path,
    arguments: dict[str, PlainTree],
    summary: str,
    key: str | None,
) -> None:
    _, tool = _setup(tmp_path)
    assert tool.summary(arguments) == summary
    assert tool.serialize_key(ToolContext(cwd=tmp_path), arguments) == key


@pytest.mark.asyncio
async def test_a_caller_outside_the_team_is_refused(tmp_path: Path) -> None:
    _, tool = _setup(tmp_path)
    with pytest.raises(ValueError, match="needs a calling agent"):
        _ = await _ask(tool, tmp_path, {"action": "list"}, label="nobody")


@pytest.mark.asyncio
async def test_list_says_what_each_entry_is_and_how_it_starts(tmp_path: Path) -> None:
    _, tool = _setup(tmp_path)

    lines = (await _ask(tool, tmp_path, {"action": "list"})).splitlines()

    assert lines == [
        "#0 user message (2 chars): go",
        '#1 your turn, calling Read (21 chars): {"file_path": "a.py"}',
        f"#2 result of Read (4,000 chars): {'a' * 80}...",
        "#3 your turn, calling Grep (2 chars): {}",
        f"#4 cancelled result of Grep (400 chars): {'b' * 80}...",
        "#5 your turn (7 chars): read it",
        "#6 message from Agent_0 (9 chars): peer says",
        "#7 note (6 chars): a note",
        "#8 settings (21 chars): Working directory: /w",
    ]


@pytest.mark.asyncio
async def test_an_empty_context_says_so(tmp_path: Path) -> None:
    _, tool = _setup(tmp_path)
    agent = tool.team["Agent"]
    agent.runtime.append_clear()

    assert await _ask(tool, tmp_path, {"action": "list"}) == "The context is empty."


@pytest.mark.asyncio
async def test_read_gives_each_entry_whole_once(tmp_path: Path) -> None:
    _, tool = _setup(tmp_path)

    text = await _ask(tool, tmp_path, {"action": "read", "ids": [6, 0, 6]})

    assert text == (
        "#0 user message (2 chars): go\ngo\n\n"
        "#6 message from Agent_0 (9 chars): peer says\npeer says"
    )


@pytest.mark.parametrize(
    ("ids", "match"),
    [
        ([], "Name the entries"),
        ([9], r"No entry #9; your context has 9\."),
        ([-1], "No entry #-1"),
    ],
)
@pytest.mark.asyncio
async def test_ids_must_name_entries(
    tmp_path: Path,
    ids: list[PlainTree],
    match: str,
) -> None:
    _, tool = _setup(tmp_path)
    with pytest.raises(ValueError, match=match):
        _ = await _ask(tool, tmp_path, {"action": "read", "ids": ids})


@pytest.mark.asyncio
async def test_search_reads_the_whole_session_compacted_windows_too(
    tmp_path: Path,
) -> None:
    journal = RecordingJournal()
    journal.append(
        ContextClear(summary="an older SUMMARY of the plan"),
        Thinking(content="thinking of the plan"),
        UncategorizedRecord(kind="note"),
    )
    _, tool = _setup(tmp_path, journal=journal)

    found = await _ask(tool, tmp_path, {"action": "search", "text": "Plan"})
    tool_args = await _ask(tool, tmp_path, {"action": "search", "text": "a.py"})
    none = await _ask(tool, tmp_path, {"action": "search", "text": "absent"})

    assert found.splitlines() == [
        "record 0, ContextClear: ...an older SUMMARY of the plan...",
        "record 1, Thinking: ...thinking of the plan...",
    ]
    assert tool_args == 'record 4, ToolCall: ...Read {"file_path": "a.py"}...'
    assert none == "No record holds 'absent'."
    with pytest.raises(ValueError, match="Say what to find"):
        _ = await _ask(tool, tmp_path, {"action": "search", "text": " "})


@pytest.mark.asyncio
async def test_search_without_a_journal_reads_what_the_model_sees(
    tmp_path: Path,
) -> None:
    _, tool = _setup(tmp_path)

    found = await _ask(tool, tmp_path, {"action": "search", "text": "PEER"})
    settings = await _ask(tool, tmp_path, {"action": "search", "text": "/w"})
    result = await _ask(tool, tmp_path, {"action": "search", "text": "bbbb"})

    assert found == "record 6, AgentToAgentMessage: ...peer says..."
    assert settings == "record 8, TurnContext: ...Working directory: /w..."
    assert result.startswith("record 4, UncategorizedToolResult: ...bbbb")


@pytest.mark.asyncio
async def test_an_estimate_says_what_a_prune_saves_and_costs(tmp_path: Path) -> None:
    """D, T, S, and k as ``pruner.saving`` states them."""
    _, tool = _setup(tmp_path)

    text = await _ask(tool, tmp_path, {"action": "estimate", "ids": [2]})

    # σ = 59/4; D = 1000 − σ; T = 1111 − D; k = (3.75 − 0.3)·T / (0.3·D).
    lines = text.splitlines()
    assert lines[0] == (
        "Dropping #2 saves 985 tokens on each later request and writes 126 tokens"
        " to the cache once."
    )
    assert lines[1].startswith(
        "It pays for itself after 1.5 requests, and saves about $",
    )


@pytest.mark.asyncio
async def test_an_estimate_of_several_names_each(tmp_path: Path) -> None:
    agent, tool = _setup(tmp_path)
    agent.session.append(
        ToolCall(call_id="c2", name="Read"),
        UncategorizedToolResult(call_id="c2", content="c" * 400),
    )

    text = await _ask(tool, tmp_path, {"action": "estimate", "ids": [10, 2]})

    assert text.startswith("Dropping #2, #10 saves 1,070 tokens on each later request")


@pytest.mark.asyncio
async def test_an_unpriced_estimate_says_only_the_tokens(tmp_path: Path) -> None:
    _, tool = _setup(tmp_path, price=None)

    text = await _ask(tool, tmp_path, {"action": "estimate", "ids": [2]})

    assert text == (
        "Dropping #2 saves 985 tokens on each later request and writes 126 tokens"
        " to the cache once."
    )


@pytest.mark.parametrize("ids", [[0], [4]], ids=["not-a-result", "a-stub"])
@pytest.mark.asyncio
async def test_only_a_real_result_can_be_pruned(
    tmp_path: Path,
    ids: list[PlainTree],
) -> None:
    _, tool = _setup(tmp_path)
    with pytest.raises(ValueError, match="is no tool result that can be pruned"):
        _ = await _ask(tool, tmp_path, {"action": "prune", "ids": ids, "reason": "x"})


@pytest.mark.asyncio
async def test_a_prune_says_why_and_drops_after_the_call(tmp_path: Path) -> None:
    agent, tool = _setup(tmp_path)
    with pytest.raises(ValueError, match=r"^Say why in 'reason'\.$"):
        _ = await _ask(tool, tmp_path, {"action": "prune", "ids": [2]})

    text = await _ask(
        tool,
        tmp_path,
        {"action": "prune", "ids": [2], "reason": "the file is read"},
    )

    assert text.startswith("Prune queued: the file is read\nDropping #2 saves")
    assert agent.runtime.inbox.drain_nowait() == []
    await asyncio.sleep(0)
    (queued,) = agent.runtime.inbox.drain_nowait()
    assert isinstance(queued, types.Scheduled)
    assert isinstance(queued.done, asyncio.Future)
    assert not queued.done.done()
    # The drop finds the result by its call, wherever it is by then.
    agent.session.append(AssistantMessage(content="later"))
    await queued.work()
    results = {r.call_id: r for r in agent.session if isinstance(r, ToolResult)}
    assert (
        results["c0"].text()
        == "[pruned: old tool output removed to keep the context small]"
    )
    assert results["c1"].stub == "cancelled"


@pytest.mark.asyncio
async def test_compact_queues_a_compaction_in_the_style_asked(tmp_path: Path) -> None:
    agent, tool = _setup(tmp_path)

    plain = await _ask(tool, tmp_path, {"action": "compact", "prompt": "keep it"})
    await asyncio.sleep(0)
    first = agent.runtime.inbox.drain_nowait()
    audit = await _ask(
        tool,
        tmp_path,
        {"action": "compact", "style": "session-audit"},
    )
    await asyncio.sleep(0)
    second = agent.runtime.inbox.drain_nowait()

    assert plain == "Compaction queued (summary)."
    assert first == [types.Compact(args="keep it")]
    assert audit == "Compaction queued (session-audit)."
    assert second == [types.Compact(args="")]
    compactor = agent.session.compactor
    assert isinstance(compactor, SummaryCompactor)
    assert compactor.style == "session-audit"


@pytest.mark.asyncio
async def test_compact_needs_a_compactor(tmp_path: Path) -> None:

    _, tool = _setup(tmp_path, compactor=NoShrinker)
    with pytest.raises(ValueError, match="no compactor"):
        _ = await _ask(tool, tmp_path, {"action": "compact"})


@pytest.mark.asyncio
async def test_clear_queues_a_clear_after_the_call(tmp_path: Path) -> None:
    agent, tool = _setup(tmp_path)

    assert await _ask(tool, tmp_path, {"action": "clear"}) == "Clear queued."
    assert agent.runtime.inbox.drain_nowait() == []
    await asyncio.sleep(0)
    assert agent.runtime.inbox.drain_nowait() == [types.Clear()]


@pytest.mark.asyncio
async def test_read_by_record_finds_a_call_with_its_result_in_another_s_session(
    tmp_path: Path,
) -> None:
    _, tool = _setup(tmp_path, journal=RecordingJournal())
    watcher = Agent(model=ScriptedProvider()["scripted"], cwd=tmp_path)
    _ = tool.team.join(watcher, label="watcher")
    tool.team.watch("watcher", "Agent")

    said = await _ask(tool, tmp_path, {"action": "list", "agent": "parent"})
    read = await _ask(
        tool,
        tmp_path,
        {"action": "read", "agent": "Agent", "records": [1, 1]},
        label="watcher",
    )

    assert said == "The context is empty."
    head, result = read.split("\n\n")
    assert head == 'record 1, ToolCall:\nRead {"file_path": "a.py"}'
    assert result.startswith("its result, record 3:\n" + "a" * 100)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("arguments", "match"),
    [
        ({"action": "read", "records": [99]}, "No record 99; the session has"),
        ({"action": "list", "agent": "children"}, "'children' names 0 agents"),
        ({"action": "prune", "agent": "kid", "ids": [2]}, "Only list, read and search"),
    ],
)
async def test_another_s_session_is_only_read(
    tmp_path: Path,
    arguments: dict[str, PlainTree],
    match: str,
) -> None:
    _, tool = _setup(tmp_path)
    _ = tool.team.join(
        Agent(model=ScriptedProvider()["scripted"], cwd=tmp_path),
        label="kid",
        parent="Agent",
    )
    _ = tool.team.leave("kid") if arguments.get("agent") == "children" else None

    with pytest.raises(ValueError, match=match):
        _ = await tool.call(ToolContext(cwd=tmp_path, label="Agent"), arguments)


@pytest.mark.asyncio
async def test_list_and_search_reach_a_child_s_session(tmp_path: Path) -> None:
    _, tool = _setup(tmp_path)
    kid = Agent(model=ScriptedProvider()["scripted"], cwd=tmp_path)
    _ = tool.team.join(kid, label="kid", parent="Agent")
    kid.session.append(UserMessage(content="kid's own words"))

    listed = await _ask(tool, tmp_path, {"action": "list", "agent": "kid"})
    found = await _ask(
        tool,
        tmp_path,
        {"action": "search", "agent": "kid", "text": "own words"},
    )

    assert listed.startswith("#0 user message")
    assert found == "record 0, UserMessage: ...kid's own words..."


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
