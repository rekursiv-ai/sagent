"""Tests for the AgentSession tool: an agent reading and editing its own session."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, override
from unittest.mock import Mock

import asyncio
import re
import threading
import tracemalloc

import pytest

from sagent.agent.agent import Agent, NoShrinker
from sagent.agent.compaction.compactor import SummaryCompactor
from sagent.agent.compaction.pruner import Pruner
from sagent.agent.team import Team
from sagent.journal.jsonl import read_record, record_line
from sagent.journal.recording import RecordingJournal
from sagent.journal.types import (
    AgentMessage,
    AssistantMessage,
    Cancelled,
    ContextClear,
    ContextCompaction,
    Failed,
    Outcome,
    SessionContext,
    SessionResult,
    Succeeded,
    SystemMessage,
    Thinking,
    ToolCall,
    ToolMessage,
    UncategorizedRecord,
    UncategorizedToolResult,
    UserMessage,
)
from sagent.model.capability import ModelCapability, ModelLimits, ModelSettings
from sagent.model.cost import PriceCatalog, PriceCondition, TokenPrice
from sagent.model.scripted import ScriptedProvider
from sagent.providers.catalog import ModelCatalog
from sagent.runtime import types
from sagent.tools import agent_session
from sagent.tools.agent_session import AgentSession
from sagent.tools.types import ToolContext, settled


if TYPE_CHECKING:
    from collections.abc import Iterator

    from treekle import PlainTree

    from sagent.agent.types import Factory, Shrinker
    from sagent.journal.types import SessionRecord


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
    pruner: Factory[Shrinker] = Pruner,
    journal: RecordingJournal | None = None,
    routed: bool = False,
) -> tuple[Agent, AgentSession]:
    prices = PriceCatalog(
        prices={} if price is None else {PriceCondition(): price},
        window=200_000,
    )
    capability = ModelCapability(
        model_id="priced",
        limits=ModelLimits(max_request_tokens=200_000, max_response_tokens=8_000),
        prices=prices,
        default=ModelSettings(context_size=prices["fullcontext"]),
    )
    catalog = ModelCatalog(
        factories={"scripted": partial(ScriptedProvider, capability=capability)},
    )
    team = Team(catalog=catalog)
    tool = AgentSession(team=team)
    agent = Agent(
        model=catalog["scripted"]["priced"],
        tools=[tool] if routed else [],
        tool_context=ToolContext(cwd=tmp_path),
        compactor=compactor,
        pruner=pruner,
        journal=journal,
        system_prompt="",
    )
    _ = team.join(agent)
    agent.session.append(
        UserMessage(content="go"),
        ToolCall(call_id="c0", name="Read", arguments={"file_path": "a.py"}),
        ToolMessage(
            call_id="c0",
            state="done",
            stdout=("a" * 4_000,),
            result=UncategorizedToolResult(),
        ),
        ToolCall(call_id="c1", name="Grep"),
        ToolMessage(call_id="c1", state="cancelled", stdout=("b" * 400,)),
        AssistantMessage(content="read it"),
        AgentMessage(content="peer says", sender="Agent_0", recipient="Agent"),
        SystemMessage(content="a note"),
        SessionContext(cwd=Path("/w")),
    )
    return agent, tool


async def _ask(
    tool: AgentSession,
    tmp_path: Path,
    arguments: dict[str, PlainTree],
    *,
    label: str = "Agent",
) -> str:
    streaming_result = tool.call(ToolContext(cwd=tmp_path, label=label), arguments)
    result = await settled(streaming_result)
    # ``settled`` records a raise instead of raising; a refusal test expects the raise.
    _ = await streaming_result.result
    assert (result.state, result.is_error) == ("done", False)
    assert result.result == SessionResult(action=str(arguments["action"]))
    return result.text()


@pytest.mark.parametrize(
    ("arguments", "summary", "key"),
    [
        ({"action": "list"}, "AgentSession list", None),
        ({"action": "prune_queue", "ids": [3, 7]}, "AgentSession prune_queue 3 7", "*"),
        ({"action": "prune_unqueue", "ids": [3]}, "AgentSession prune_unqueue 3", "*"),
        ({"action": "prune_list"}, "AgentSession prune_list", None),
        ({"action": "compact"}, "AgentSession compact", "*"),
        ({"action": "clear"}, "AgentSession clear", "*"),
        ({"action": "read", "ids": "x"}, "AgentSession read", None),
        ({}, "AgentSession", None),
    ],
)
def test_the_summary_names_the_action_and_edits_run_alone(
    tmp_path: Path,
    arguments: dict[str, PlainTree],
    summary: str,
    key: str | None,
) -> None:
    tool = AgentSession(team=Mock(spec=Team))
    assert tool.summary(arguments) == summary
    assert tool.serialize_key(ToolContext(cwd=tmp_path), arguments) == key


@pytest.mark.asyncio
async def test_a_caller_outside_the_team_is_refused(tmp_path: Path) -> None:
    _, tool = _setup(tmp_path)
    with pytest.raises(
        ValueError,
        match=r"^AgentSession needs a calling agent in the team\.$",
    ):
        _ = await _ask(tool, tmp_path, {"action": "list"}, label="nobody")


@pytest.mark.asyncio
async def test_list_cuts_a_head_only_past_80_and_names_unknown_and_several_calls(
    tmp_path: Path,
) -> None:
    agent, tool = _setup(tmp_path)
    agent.runtime.tape.append_clear()
    agent.session.append(
        UserMessage(content="x" * 80),
        UserMessage(content="y" * 81),
        ToolCall(call_id="c2", name="Read"),
        ToolCall(call_id="c3", name="Grep"),
        ToolMessage(call_id="c2", state="pending", stdout=("so far",)),
        ToolMessage(
            call_id="gone",
            state="done",
            stdout=("orphan",),
            result=UncategorizedToolResult(),
        ),
        UserMessage(content="hi", sender="bob"),
    )

    lines = (await _ask(tool, tmp_path, {"action": "list"})).splitlines()

    assert lines == [
        f"#0 user message (80 chars): {'x' * 80}",
        f"#1 user message (81 chars): {'y' * 80}...",
        "#2 your turn, calling Read, Grep (4 chars): {}{}",
        "#3 pending result of Read (6 chars): so far",
        "#4 result of ? (6 chars): orphan",
        "#5 user message from bob (2 chars): hi",
    ]


@pytest.mark.asyncio
async def test_list_names_a_vendor_s_summary_and_shows_its_text(tmp_path: Path) -> None:
    agent, tool = _setup(tmp_path)
    agent.runtime.tape.append_clear()
    agent.session.append(
        ContextCompaction(
            summary="The user asked.",
            provider="anthropic",
            block={
                "type": "compaction",
                "content": "The user asked.",
                "signature": "s",
            },
        ),
    )

    lines = (await _ask(tool, tmp_path, {"action": "list"})).splitlines()

    assert lines == ["#0 summary by anthropic (15 chars): The user asked."]


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
    agent.runtime.tape.append_clear()

    assert await _ask(tool, tmp_path, {"action": "list"}) == "The context is empty."


@pytest.mark.asyncio
async def test_read_gives_each_entry_whole_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent, tool = _setup(tmp_path)

    text = await _ask(tool, tmp_path, {"action": "read", "ids": [6, 0, 6]})
    result = await _ask(tool, tmp_path, {"action": "read", "ids": [2]})

    assert text == (
        "#0 user message (2 chars): go\ngo\n\n"
        "#6 message from Agent_0 (9 chars): peer says\npeer says"
    )
    assert result == f"#2 result of Read (4,000 chars): {'a' * 80}...\n{'a' * 4_000}"

    failure = ValueError("invalid entry")
    picked = Mock(side_effect=failure)
    monkeypatch.setattr(agent_session, "_picked", picked)
    with pytest.raises(ValueError, match="invalid entry") as caught:
        await _ask(tool, tmp_path, {"action": "read", "ids": [2]})
    assert caught.value is failure
    picked.assert_called_once_with(agent.runtime.tape.context().messages, (2,))


@pytest.mark.parametrize(
    ("ids", "match"),
    [
        ([], r"^Name the entries in 'ids'; list shows them\.$"),
        ([1], r"^No entry #1; your context has 1\.$"),
        ([-1], r"^No entry #-1; your context has 1\.$"),
    ],
)
def test_ids_must_name_entries(ids: list[int], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        agent_session._picked([UserMessage(content="x")], ids)


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
    with pytest.raises(ValueError, match=r"^Say what to find in 'text'\.$"):
        _ = await _ask(tool, tmp_path, {"action": "search", "text": " "})


@pytest.mark.asyncio
async def test_search_shows_60_chars_around_the_first_match(tmp_path: Path) -> None:
    journal = RecordingJournal()
    journal.append(UserMessage(content=f"{'a' * 70}needle{'b' * 70}needle"))
    _, tool = _setup(tmp_path, journal=journal)

    found = await _ask(tool, tmp_path, {"action": "search", "text": "NEEDLE"})

    assert found == f"record 0, UserMessage: ...{'a' * 60}needle{'b' * 60}..."


@pytest.mark.asyncio
async def test_a_record_with_no_words_matches_nothing(tmp_path: Path) -> None:
    journal = RecordingJournal()
    journal.append(
        ContextClear(),
        UserMessage(content=None),
        UncategorizedRecord(kind="note"),
    )
    agent, tool = _setup(tmp_path, journal=journal)
    agent.runtime.tape.append_clear()

    found = await _ask(tool, tmp_path, {"action": "search", "text": "x"})

    assert found == "No record holds 'x'."


@pytest.mark.asyncio
async def test_search_without_a_journal_reads_the_session_kept_in_memory(
    tmp_path: Path,
) -> None:
    """Numbered as a journal's: each model turn is followed by its usage."""
    _, tool = _setup(tmp_path)

    found = await _ask(tool, tmp_path, {"action": "search", "text": "PEER"})
    settings = await _ask(tool, tmp_path, {"action": "search", "text": "/w"})
    result = await _ask(tool, tmp_path, {"action": "search", "text": "bbbb"})

    assert found == "record 9, AgentMessage: ...peer says..."
    assert settings == "record 11, SessionContext: ...Working directory: /w..."
    assert result.startswith("record 6, ToolMessage: ...bbbb")


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
        ToolMessage(
            call_id="c2",
            state="done",
            stdout=("c" * 400,),
            result=UncategorizedToolResult(),
        ),
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


@pytest.mark.parametrize("which", ["not-a-result", "cancelled", "pending"])
@pytest.mark.asyncio
async def test_only_output_a_drop_would_shrink_can_be_pruned(
    tmp_path: Path,
    which: str,
) -> None:
    """A bare stub would only grow: it keeps its first line, then the drop's note."""
    agent, tool = _setup(tmp_path)
    agent.session.append(
        ToolCall(call_id="c2", name="Grep"),
        ToolMessage(call_id="c2", state="pending", shown="[running]"),
    )
    entries = agent.runtime.tape.context().messages
    at = {
        "not-a-result": 0,
        **{e.state: i for i, e in enumerate(entries) if isinstance(e, ToolMessage)},
    }[which]
    with pytest.raises(
        ValueError,
        match=rf"^#{at} is no tool output that can be pruned\.$",
    ):
        _ = await _ask(
            tool,
            tmp_path,
            {"action": "prune_queue", "ids": [at], "reason": "x"},
        )


@pytest.mark.asyncio
async def test_a_long_call_s_stub_and_news_can_be_queued(tmp_path: Path) -> None:
    agent, tool = _setup(tmp_path)
    news = ToolMessage(
        call_id="c2",
        state="pending",
        stdout=("[talk c2 running after 15s]\n" + "y" * 400,),
    )
    agent.session.append(
        ToolCall(call_id="c2", name="Bash"),
        ToolMessage(call_id="c2", state="pending", shown="[detached]\n" + "x" * 400),
        AssistantMessage(content="waiting"),
        news,
        AssistantMessage(content="still"),
    )
    entries = agent.runtime.tape.context().messages
    picked = [i for i, e in enumerate(entries) if e in (news, entries[-4])]
    text = await _ask(
        tool,
        tmp_path,
        {"action": "prune_queue", "ids": picked, "reason": "x"},
    )
    assert text.startswith(f"Queued #{picked[0]}, #{picked[1]}: x")
    assert agent.runtime.queued == {"c2": "x"}
    assert agent.runtime.inbox.drain_nowait() == []


@pytest.mark.asyncio
async def test_an_estimate_of_one_output_of_a_long_call_names_every_one_dropped(
    tmp_path: Path,
) -> None:
    """A prune names the call, so it drops each of its outputs: the estimate says so."""
    agent, tool = _setup(tmp_path)
    news = ToolMessage(
        call_id="c2",
        state="pending",
        stdout=("[talk c2 running after 15s]\n" + "y" * 400,),
    )
    agent.session.append(
        ToolCall(call_id="c2", name="Bash"),
        ToolMessage(call_id="c2", state="pending", shown="[detached]"),
        AssistantMessage(content="waiting"),
        news,
        AssistantMessage(content="still"),
        ToolMessage(call_id="c2", state="done", stdout=("z" * 400,)),
        AssistantMessage(content="done"),
    )
    entries = agent.runtime.tape.context().messages
    at = [i for i, e in enumerate(entries) if e is news]
    late = len(entries) - 2

    text = await _ask(tool, tmp_path, {"action": "estimate", "ids": at})

    assert text.startswith(f"Dropping #{at[0]}, #{late} saves ")


@pytest.mark.asyncio
async def test_a_queue_says_why_and_how_far_it_is_from_paying(tmp_path: Path) -> None:
    agent, tool = _setup(tmp_path)
    with pytest.raises(ValueError, match=r"^Say why in 'reason'\.$"):
        _ = await _ask(tool, tmp_path, {"action": "prune_queue", "ids": [2]})
    # Ten thousand tokens after it to write again: more than a drop saves.
    agent.session.append(AssistantMessage(content="t" * 40_000))

    text = await _ask(
        tool,
        tmp_path,
        {"action": "prune_queue", "ids": [2], "reason": "the file is read"},
    )

    first, saved, short = text.splitlines()
    assert first == "Queued #2: the file is read"
    assert saved.startswith("Dropping #2 saves 985 tokens on each later request")
    assert re.fullmatch(
        r"It is \$[\d.]+ short of paying; it drops once it pays\.",
        short,
    )
    assert agent.runtime.queued == {"c0": "the file is read"}
    assert agent.runtime.inbox.drain_nowait() == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("price", "pruner", "said"),
    [
        (None, Pruner, r"It drops before the next request\."),
        (_CACHED, Pruner, r"It is \$[\d.]+ short of paying; it drops once it pays\."),
        (_CACHED, NoShrinker, r"It drops before the next request\."),
    ],
    ids=["unpriced", "short", "no-pruner"],
)
async def test_an_agent_queues_whatever_its_pruner(
    tmp_path: Path,
    price: TokenPrice | None,
    pruner: Factory[Shrinker],
    said: str,
) -> None:
    """With no pruner to price it, a queue drops at the next request."""
    agent, tool = _setup(tmp_path, price=price, pruner=pruner)
    # Ten thousand tokens after it to write again: a pruner waits.
    agent.session.append(AssistantMessage(content="t" * 40_000))

    text = await _ask(
        tool,
        tmp_path,
        {"action": "prune_queue", "ids": [2], "reason": "spent"},
    )

    assert re.fullmatch(said, text.splitlines()[-1])
    assert agent.runtime.queued == {"c0": "spent"}


@pytest.mark.asyncio
async def test_a_queue_past_the_trigger_waits_until_it_brings_the_context_under(
    tmp_path: Path,
) -> None:
    agent, tool = _setup(tmp_path, price=None)
    # 200,000 tokens: past the trigger by more than the result saves.
    agent.session.append(UserMessage(content="p" * 800_000))

    text = await _ask(
        tool,
        tmp_path,
        {"action": "prune_queue", "ids": [2], "reason": "spent"},
    )

    assert text.splitlines()[-1] == (
        "It drops once it brings the context under compaction."
    )


@pytest.mark.asyncio
async def test_the_queue_lists_and_unqueues_by_entry(tmp_path: Path) -> None:
    agent, tool = _setup(tmp_path)
    assert await _ask(tool, tmp_path, {"action": "prune_list"}) == "Nothing is queued."
    _ = await _ask(
        tool,
        tmp_path,
        {"action": "prune_queue", "ids": [2], "reason": "the file is read"},
    )

    listed = await _ask(tool, tmp_path, {"action": "prune_list"})
    unqueued = await _ask(tool, tmp_path, {"action": "prune_unqueue", "ids": [2]})

    assert listed.splitlines()[:2] == [
        "#2 result of Read (1,000 tokens): the file is read",
        (
            "Dropping #2 saves 985 tokens on each later request and writes 126 tokens"
            " to the cache once."
        ),
    ]
    assert unqueued == "Unqueued #2.\nNothing is queued."
    assert agent.runtime.queued == {}


def _kid(
    tool: AgentSession,
    tmp_path: Path,
    *,
    compactor: Factory[Shrinker] = NoShrinker,
    pruner: Factory[Shrinker] = NoShrinker,
) -> Agent:
    """Return a child of ``Agent`` in ``tool``'s team, with one output to prune."""
    kid = Agent(
        model=ScriptedProvider()["scripted"],
        tool_context=ToolContext(cwd=tmp_path),
        compactor=compactor,
        pruner=pruner,
    )
    _ = tool.team.join(kid, label="kid", parent="Agent")
    kid.session.append(
        UserMessage(content="go"),
        ToolCall(call_id="k0", name="Read"),
        ToolMessage(
            call_id="k0",
            state="done",
            stdout=("k" * 4_000,),
            result=UncategorizedToolResult(),
        ),
        AssistantMessage(content="read it"),
    )
    return kid


# The child's model did not ask: it is told nothing.
@pytest.mark.asyncio
async def test_a_queue_for_a_busy_child_waits_in_its_runtime(tmp_path: Path) -> None:
    agent, tool = _setup(tmp_path)
    kid = _kid(tool, tmp_path, pruner=Pruner)
    kid.runtime.turn_under_way = True

    text = await _ask(
        tool,
        tmp_path,
        {"agent": "kid", "action": "prune_queue", "ids": [2], "reason": "stale"},
    )

    assert text.startswith("Queued #2: stale\nDropping #2 saves")
    assert kid.runtime.queued == {"k0": "stale"}
    assert agent.runtime.queued == {}
    assert agent.runtime.inbox.drain_nowait() == []
    assert kid.runtime.inbox.drain_nowait() == []


@pytest.mark.asyncio
async def test_a_queue_for_an_idle_child_drops_at_once(tmp_path: Path) -> None:
    """No request of its own is coming to share the rewrite with."""
    agent, tool = _setup(tmp_path)
    kid = _kid(tool, tmp_path)

    async with kid:
        text = await _ask(
            tool,
            tmp_path,
            {"agent": "kid", "action": "prune_queue", "ids": [2], "reason": "stale"},
        )

    assert text.startswith("Dropped #2: stale\nDropping #2 saves")
    (dropped,) = [r for r in kid.session if isinstance(r, ToolMessage)]
    assert dropped.text() == kid.runtime.stubs.pruned
    assert kid.runtime.queued == {}
    assert agent.runtime.inbox.drain_nowait() == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("outcome", "said"),
    [
        (
            Failed(exception=IndexError("list index out of range")),
            "ValueError: The prune was not done: list index out of range",
        ),
        (Cancelled(), "ValueError: The prune was not done: the agent quit first."),
    ],
    ids=["failed", "the-child-quit"],
)
async def test_a_drop_at_once_that_is_not_done_says_why(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    outcome: Outcome[None],
    said: str,
) -> None:
    _, tool = _setup(tmp_path)
    kid = _kid(tool, tmp_path)
    asked: list[types.RuntimeEvent] = []

    async def ask(request: types.SessionPrune) -> Outcome[None]:
        asked.append(request)
        return outcome

    monkeypatch.setattr(kid.runtime, "ask", ask)
    result = await settled(
        tool.call(
            ToolContext(cwd=tmp_path, label="Agent"),
            {"agent": "kid", "action": "prune_queue", "ids": [2], "reason": "stale"},
        ),
    )

    assert result.is_error
    assert result.text().startswith(said)
    assert asked == [
        types.SessionPrune(sender="Agent", call_ids=("k0",), reason="stale"),
    ]


# As a prune of it does: one way to learn how an edit of another's session went.
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("arguments", "outcome", "reply", "asked_for"),
    [
        (
            {"action": "clear"},
            Succeeded(value=None),
            "Cleared.",
            types.SessionClear(sender="Agent"),
        ),
        (
            {"action": "clear"},
            Cancelled(),
            "ValueError: The clear was not done: the agent quit first.",
            types.SessionClear(sender="Agent"),
        ),
        (
            {"action": "compact", "prompt": "keep it"},
            Succeeded(
                value=types.SessionCompacted(
                    masked_tokens=9_000,
                    replacement_tokens=90,
                ),
            ),
            "Compacted (summary): ~9,000 to ~90 tokens.",
            types.SessionCompact(sender="Agent", summary_must_keep="keep it"),
        ),
        (
            {"action": "compact"},
            Failed(exception=RuntimeError("model down")),
            "ValueError: The compaction was not done: model down",
            types.SessionCompact(sender="Agent"),
        ),
        (
            {"action": "compact"},
            Cancelled(),
            "ValueError: The compaction was not done: it was stopped first.",
            types.SessionCompact(sender="Agent"),
        ),
    ],
    ids=["cleared", "clear-quit", "compacted", "compact-failed", "compact-stopped"],
)
async def test_a_clear_or_compaction_of_a_child_s_session_waits_and_tells_who_asked_how_it_went(
    *,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    arguments: dict[str, PlainTree],
    outcome: Outcome[types.SessionCompacted] | Outcome[None],
    reply: str,
    asked_for: types.SessionClear | types.SessionCompact,
) -> None:
    agent, tool = _setup(tmp_path)
    kid = _kid(tool, tmp_path, compactor=SummaryCompactor)
    asked: list[types.RuntimeEvent] = []

    async def ask(
        request: types.SessionClear | types.SessionCompact,
    ) -> Outcome[types.SessionCompacted] | Outcome[None]:
        asked.append(request)
        return outcome

    monkeypatch.setattr(kid.runtime, "ask", ask)
    result = await settled(
        tool.call(
            ToolContext(cwd=tmp_path, label="Agent"),
            {"agent": "kid", **arguments},
        ),
    )

    assert result.is_error is not isinstance(outcome, Succeeded)
    assert result.text() == reply
    assert asked == [asked_for]
    assert agent.runtime.inbox.drain_nowait() == []
    assert kid.runtime.inbox.drain_nowait() == []


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
    assert first == [types.SessionCompact(sender="Agent", summary_must_keep="keep it")]
    assert audit == "Compaction queued (session-audit)."
    assert second == [types.SessionCompact(sender="Agent", summary_must_keep="")]
    compactor = agent.session.compactor
    assert isinstance(compactor, SummaryCompactor)
    assert compactor.style == "session-audit"


@pytest.mark.asyncio
async def test_compact_needs_a_compactor(tmp_path: Path) -> None:

    _, tool = _setup(tmp_path, compactor=NoShrinker)
    with pytest.raises(ValueError, match=r"^This agent has no compactor\.$"):
        _ = await _ask(tool, tmp_path, {"action": "compact"})


@pytest.mark.asyncio
async def test_clear_queues_a_clear_after_the_call(tmp_path: Path) -> None:
    agent, tool = _setup(tmp_path)

    assert await _ask(tool, tmp_path, {"action": "clear"}) == "Clear queued."
    # The tool ran outside a runtime call, so the event waits behind no result.
    assert agent.runtime.inbox.drain_nowait() == [types.SessionClear(sender="Agent")]


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["clear", "compact"])
async def test_a_clear_or_compaction_is_queued_behind_the_calling_task(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    action: str,
) -> None:
    """Taken ahead of the call's result, the event would detach the call."""
    agent, tool = _setup(tmp_path)
    after: list[object] = []

    def noted(task: asyncio.Future[object] | None, event: types.RuntimeEvent) -> None:
        del event
        after.append(task)

    monkeypatch.setattr(agent.runtime, "push_after", noted)
    streaming_result = tool.call(
        ToolContext(cwd=tmp_path, label="Agent"),
        {"action": action},
    )
    _ = await settled(streaming_result)

    assert after == [streaming_result.result]


@pytest.mark.asyncio
async def test_read_by_record_finds_a_call_with_its_result_in_another_s_session(
    tmp_path: Path,
) -> None:
    _, tool = _setup(tmp_path, journal=RecordingJournal(), routed=True)
    watcher = Agent(
        model=ScriptedProvider()["scripted"],
        tools=[tool],
        tool_context=ToolContext(cwd=tmp_path),
    )
    _ = tool.team.join(watcher, label="watcher")
    tool.team.watch("watcher", "Agent")
    parent_list = await settled(
        tool.call(
            tool.team["Agent"].runtime.tool_context,
            {"agent": "parent", "action": "list"},
        ),
    )
    read = await settled(
        tool.call(
            watcher.runtime.tool_context,
            {"agent": "Agent", "action": "read", "records": [1, 1]},
        ),
    )

    assert parent_list.text() == "The context is empty."
    head, result = read.text().split("\n\n")
    assert head == 'record 1, ToolCall:\nRead {"file_path": "a.py"}'
    assert result.startswith("its result, record 3:\n" + "a" * 100)
    first = await _ask(tool, tmp_path, {"action": "read", "records": [0]})
    assert first == "record 0, UserMessage:\ngo"


class _Threaded(RecordingJournal):
    """A journal that notes where each read of its history starts, and if on the loop."""

    def __init__(self) -> None:
        super().__init__()
        self.reads: list[tuple[int, bool]] = []

    @override
    def history(self, start: int = 0) -> Iterator[SessionRecord]:
        records = super().history(start)

        def read() -> Iterator[SessionRecord]:
            on_loop = threading.current_thread() is threading.main_thread()
            self.reads.append((start, on_loop))
            yield from records

        return read()


@pytest.mark.asyncio
async def test_a_read_by_record_or_a_search_decodes_off_the_loop_from_the_first_asked(
    tmp_path: Path,
) -> None:
    """A long session's journal takes seconds to read; the screen must not wait."""
    journal = _Threaded()
    _, tool = _setup(tmp_path, journal=journal)

    read = await _ask(tool, tmp_path, {"action": "read", "records": [5, 3]})
    found = await _ask(tool, tmp_path, {"action": "search", "text": "peer"})

    assert read.startswith("record 3, ToolMessage:\naaaa")
    assert found.startswith("record 9, AgentMessage")
    assert journal.reads == [(3, False), (0, False)]


class _Decoding(RecordingJournal):
    """A journal that decodes each record afresh as its history is read, as a file's does."""

    @override
    def history(self, start: int = 0) -> Iterator[SessionRecord]:
        return (read_record(record_line(r)) for r in super().history(start))


@pytest.mark.asyncio
async def test_a_read_by_record_holds_no_record_after_what_it_shows(
    tmp_path: Path,
) -> None:
    """A read of an early record held the rest of the session, decoded, on a thread."""
    agent, tool = _setup(tmp_path, journal=_Decoding())
    system_text = "z" * (1 << 16)
    agent.session.append(*(SystemMessage(content=system_text) for _ in range(16)))
    # Read once first: what a first read sets up, a second does not, so the peak
    # is what the read holds.
    _ = await _ask(tool, tmp_path, {"action": "read", "records": [1]})
    tracemalloc.start()
    try:
        read = await _ask(tool, tmp_path, {"action": "read", "records": [1]})
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert read == (
        'record 1, ToolCall:\nRead {"file_path": "a.py"}\n\n'
        "its result, record 3:\n" + "a" * 4_000
    )
    assert peak < 8 * len(system_text)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("arguments", "match"),
    [
        ({"action": "read", "records": [99]}, "No record 99; the session has"),
        ({"action": "read", "records": [14]}, r"^No record 14; the session has 14\.$"),
        ({"action": "read", "records": [-1]}, r"^No record -1; the session has 14\.$"),
        ({"action": "list", "agent": "kid"}, "names 0 agents"),
    ],
)
async def test_direct_session_calls_reject_bad_record_ids_and_target_arguments(
    tmp_path: Path,
    arguments: dict[str, PlainTree],
    match: str,
) -> None:
    _, tool = _setup(tmp_path)
    with pytest.raises(ValueError, match=match):
        _ = await tool.call(ToolContext(cwd=tmp_path, label="Agent"), arguments).result


@pytest.mark.asyncio
async def test_list_and_search_reach_a_child_s_session(tmp_path: Path) -> None:
    _, tool = _setup(tmp_path, routed=True)
    kid = Agent(
        model=ScriptedProvider()["scripted"],
        tool_context=ToolContext(cwd=tmp_path),
    )
    _ = tool.team.join(kid, label="kid", parent="Agent")
    kid.session.append(UserMessage(content="kid's own words"))
    caller = tool.team["Agent"]
    listed = await settled(
        tool.call(
            caller.runtime.tool_context,
            {"agent": "kid", "action": "list"},
        ),
    )
    found = await settled(
        tool.call(
            caller.runtime.tool_context,
            {"agent": "kid", "action": "search", "text": "own words"},
        ),
    )

    assert listed.text().startswith("#0 user message")
    assert found.text() == "record 0, UserMessage: ...kid's own words..."


@pytest.mark.asyncio
async def test_search_reaches_an_unrelated_agents_journal(tmp_path: Path) -> None:
    _, tool = _setup(tmp_path)
    peer = Agent(model=ScriptedProvider()["scripted"], environment=False)
    tool.team.join(peer, label="peer")
    peer.session.append(UserMessage(content="the peer's words"))
    result = await settled(
        tool.call(
            tool.team["Agent"].runtime.tool_context,
            {"agent": "peer", "action": "search", "text": "words"},
        ),
    )
    assert "the peer's words" in result.text()


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
