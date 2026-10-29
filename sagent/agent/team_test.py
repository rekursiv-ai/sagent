"""Tests for the team: labels, who spawned whom, depth caps, and persistent children."""

from __future__ import annotations

from contextlib import AsyncExitStack
from dataclasses import replace
from typing import TYPE_CHECKING

import logging

import pytest

from sagent.agent.agent import Agent
from sagent.agent.team import ChildSpec, Team
from sagent.model.scripted import ScriptedProvider, call, reply
from sagent.providers.catalog import ModelCatalog
from sagent.runtime import types
from sagent.session.jsonl import JsonlJournal
from sagent.session.types import (
    AgentToAgentMessage,
    AssistantMessage,
    ToolCall,
    UncategorizedRecord,
    UserMessage,
)
from sagent.tools.agent_send import AgentSend


if TYPE_CHECKING:
    from pathlib import Path

    from treekle import PlainTree

    from sagent.model.types import Response


def _team() -> Team:
    return Team(catalog=ModelCatalog(factories={"scripted": ScriptedProvider}))


def _agent(tmp_path: Path, team: Team | None = None) -> Agent:
    """Return an agent; one in ``team`` can reply by AgentSend."""
    return Agent(
        model=ScriptedProvider()["scripted"],
        cwd=tmp_path,
        tools=[AgentSend(team=team)] if team is not None else [],
    )


def _closing(closed: list[str], label: str, *, fail: bool = False) -> AsyncExitStack:
    """Return a stack that records ``label`` as closed, then raises if ``fail``."""
    stack = AsyncExitStack()
    stack.push_async_callback(_close, closed, label, fail=fail)
    return stack


async def _close(closed: list[str], label: str, *, fail: bool) -> None:
    closed.append(label)
    if fail:
        raise RuntimeError(f"cannot close {label}")


def _spec(label: str) -> ChildSpec:
    return ChildSpec(label=label, journal=f"/j/{label}.jsonl", tools=("Read",))


def _readable(label: str) -> dict[str, PlainTree]:
    """Return a payload that reads as ``_spec(label)``."""
    return {"label": label, "journal": f"/j/{label}.jsonl", "tools": ["Read"]}


def test_a_root_is_agent_and_its_children_are_numbered(tmp_path: Path) -> None:
    team = _team()
    root, first, second = _agent(tmp_path), _agent(tmp_path), _agent(tmp_path)

    labels = [
        team.join(root),
        team.join(first, parent="Agent"),
        team.join(second, parent="Agent"),
    ]

    assert labels == ["Agent", "Agent_0", "Agent_1"]
    assert team.label_for("Agent") == "Agent_2"
    assert first.runtime.tool_context.label == "Agent_0"
    assert dict(team) == {"Agent": root, "Agent_0": first, "Agent_1": second}
    assert len(team) == 3
    assert (team.parent("Agent"), team.parent("Agent_1")) == (None, "Agent")


def test_a_label_left_free_is_given_again(tmp_path: Path) -> None:
    team = _team()
    _ = team.join(_agent(tmp_path))
    first = _agent(tmp_path)
    _ = team.join(first, parent="Agent")
    _ = team.join(_agent(tmp_path), parent="Agent")

    assert team.leave("Agent_0") is first
    assert team.label_for("Agent") == "Agent_0"
    assert team.label_for(None) == "Agent"
    assert "Agent_0" not in team


def test_a_child_may_be_named(tmp_path: Path) -> None:
    team = _team()
    _ = team.join(_agent(tmp_path))

    assert team.join(_agent(tmp_path), label="reviewer", parent="Agent") == "reviewer"
    assert team.label_for("reviewer") == "reviewer_0"


@pytest.mark.parametrize(
    ("label", "parent", "match"),
    [
        ("Agent", None, "already labelled 'Agent'"),
        ("x", "nobody", "No agent 'nobody'"),
    ],
)
def test_a_taken_label_or_an_unknown_parent_is_refused(
    tmp_path: Path,
    label: str,
    parent: str | None,
    match: str,
) -> None:
    team = _team()
    _ = team.join(_agent(tmp_path))

    with pytest.raises(ValueError, match=match):
        _ = team.join(_agent(tmp_path), label=label, parent=parent)
    assert list(team) == ["Agent"]


def test_leaving_or_asking_after_a_stranger_is_a_key_error() -> None:
    team = _team()

    for ask in (team.leave, team.parent, team.can_spawn, team.standing):
        with pytest.raises(KeyError):
            _ = ask("nobody")
    with pytest.raises(KeyError):
        _ = team["nobody"]


@pytest.mark.parametrize(
    ("root_cap", "child_cap", "expected"),
    [
        (None, None, (True, True, True)),
        (1, None, (True, False, False)),
        (0, None, (False, False, False)),
        (3, 1, (True, True, False)),
        (1, 5, (True, False, False)),
        (None, 0, (True, False, False)),
    ],
    ids=[
        "no-cap",
        "one-generation",
        "leaf-root",
        "child-tightens",
        "child-cannot-loosen",
        "leaf-child",
    ],
)
def test_a_depth_cap_holds_for_every_descendant(
    tmp_path: Path,
    root_cap: int | None,
    child_cap: int | None,
    expected: tuple[bool, bool, bool],
) -> None:
    team = _team()
    _ = team.join(_agent(tmp_path), max_depth=root_cap)
    _ = team.join(_agent(tmp_path), label="child", parent="Agent", max_depth=child_cap)
    _ = team.join(_agent(tmp_path), label="grandchild", parent="child")

    assert (
        team.can_spawn("Agent"),
        team.can_spawn("child"),
        team.can_spawn("grandchild"),
    ) == expected


@pytest.mark.asyncio
async def test_stopping_a_child_closes_its_children_first(tmp_path: Path) -> None:
    team = _team()
    closed: list[str] = []
    _ = team.join(_agent(tmp_path))
    for label, parent in (("a", "Agent"), ("a1", "a"), ("b", "Agent")):
        _ = team.join(
            _agent(tmp_path),
            label=label,
            parent=parent,
            closing=_closing(closed, label),
        )
    _ = team.join(_agent(tmp_path), label="oneshot", parent="a")
    assert team.persistent() == ["a", "a1", "b"]

    await team.stop("a")

    assert closed == ["a1", "a"]
    assert list(team) == ["Agent", "b"]
    with pytest.raises(KeyError):
        await team.stop("a")


@pytest.mark.asyncio
async def test_closing_the_team_closes_its_persistent_children_newest_first(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    closed: list[str] = []
    async with _team() as team:
        _ = team.join(_agent(tmp_path))
        _ = team.join(
            _agent(tmp_path),
            label="a",
            parent="Agent",
            closing=_closing(closed, "a"),
        )
        _ = team.join(
            _agent(tmp_path),
            label="b",
            parent="Agent",
            closing=_closing(closed, "b", fail=True),
        )
        _ = team.join(
            _agent(tmp_path),
            label="b1",
            parent="b",
            closing=_closing(closed, "b1"),
        )

    assert closed == ["b1", "b", "a"]
    assert list(team) == ["Agent"]
    assert any("closing agent 'b' failed" in m for m in caplog.messages)


@pytest.mark.asyncio
async def test_a_child_stopped_while_the_team_closes_is_closed_once(
    tmp_path: Path,
) -> None:
    closed: list[str] = []
    async with _team() as team:
        _ = team.join(_agent(tmp_path))
        _ = team.join(
            _agent(tmp_path),
            label="a",
            parent="Agent",
            closing=_closing(closed, "a"),
        )
        stopping = AsyncExitStack()
        stopping.push_async_callback(team.stop, "a")
        _ = team.join(_agent(tmp_path), label="b", parent="Agent", closing=stopping)

    assert closed == ["a"]
    assert list(team) == ["Agent"]


def test_a_spec_is_for_a_persistent_child_only(tmp_path: Path) -> None:
    team = _team()
    _ = team.join(_agent(tmp_path))

    with pytest.raises(ValueError, match="Only a persistent child"):
        _ = team.join(_agent(tmp_path), parent="Agent", spec=_spec("x"))
    with pytest.raises(ValueError, match="Only a persistent child"):
        _ = team.join(
            _agent(tmp_path),
            label="root2",
            closing=AsyncExitStack(),
            spec=_spec("x"),
        )


@pytest.mark.asyncio
async def test_a_persistent_childs_spec_stands_until_it_is_stopped(
    tmp_path: Path,
) -> None:
    team = _team()
    root = _agent(tmp_path)
    _ = team.join(root)
    for label in ("a", "b"):
        _ = team.join(
            _agent(tmp_path),
            label=label,
            parent="Agent",
            closing=AsyncExitStack(),
            spec=_spec(label),
        )
    root.session.stand(
        "other",
        UncategorizedRecord(kind="note", payload=_readable("x")),
    )
    assert team.standing("Agent") == [_spec("a"), _spec("b")]

    await team.stop("a")

    assert team.standing("Agent") == [_spec("b")]


@pytest.mark.asyncio
async def test_closing_the_team_leaves_the_specs_standing(tmp_path: Path) -> None:
    root = _agent(tmp_path)
    async with _team() as team:
        _ = team.join(root)
        _ = team.join(
            _agent(tmp_path),
            label="a",
            parent="Agent",
            closing=AsyncExitStack(),
            spec=_spec("a"),
        )

    assert [r.payload["label"] for r in root.session.standing()] == ["a"]


def test_a_spec_joined_under_a_new_label_stands_only_under_that_one(
    tmp_path: Path,
) -> None:
    team = _team()
    root = _agent(tmp_path)
    _ = team.join(root)
    root.session.stand(
        "agent:a",
        UncategorizedRecord(kind="persistent_agent", payload=_readable("a")),
    )

    _ = team.join(
        _agent(tmp_path),
        label="a_0",
        parent="Agent",
        closing=AsyncExitStack(),
        spec=_spec("a"),
    )

    assert team.standing("Agent") == [replace(_spec("a"), label="a_0")]


def test_a_spec_that_no_longer_reads_is_skipped(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    team = _team()
    root = _agent(tmp_path)
    _ = team.join(root)
    root.session.stand(
        "agent:old",
        UncategorizedRecord(kind="persistent_agent", payload={"label": "old"}),
    )

    with caplog.at_level(logging.WARNING):
        assert team.standing("Agent") == []
    assert any("unreadable child record" in m for m in caplog.messages)


@pytest.mark.parametrize(
    ("old", "model"),
    [
        (
            {"provider": "scripted", "model": "scripted+1m", "thinking": "adaptive"},
            "scripted/scripted+1m",
        ),
        ({"system": "S", "effort": "high", "budget": "auto", "output": "text"}, ""),
    ],
    ids=["with-thinking", "with-settings"],
)
def test_a_spec_stood_with_the_model_it_once_held_reads_as_today_s(
    tmp_path: Path,
    old: dict[str, PlainTree],
    model: str,
) -> None:
    """The child's settings now stand in its own journal; the model to start on stays."""
    team = _team()
    root = _agent(tmp_path)
    _ = team.join(root)
    root.session.stand(
        "agent:a",
        UncategorizedRecord(kind="persistent_agent", payload={**_readable("a"), **old}),
    )

    assert team.standing("Agent") == [replace(_spec("a"), model=model)]


def _family(tmp_path: Path) -> Team:
    """Return ``Agent``; its children ``a`` and ``b``; ``a``'s child ``a1``."""
    team = _team()
    _ = team.join(_agent(tmp_path, team))
    for label, parent in (("a", "Agent"), ("b", "Agent"), ("a1", "a")):
        _ = team.join(_agent(tmp_path, team), label=label, parent=parent)
    return team


@pytest.mark.parametrize(
    "label",
    ["parent", "children", "ancestors", "descendants", "all", "user"],
)
def test_a_relation_is_never_a_label(tmp_path: Path, label: str) -> None:
    team = _team()
    with pytest.raises(ValueError, match="names a relation"):
        _ = team.join(_agent(tmp_path), label=label)


@pytest.mark.parametrize(
    ("name", "by", "labels"),
    [
        ("parent", "a1", ["a"]),
        ("parent", "Agent", []),
        ("parent", "nobody", []),
        ("children", "Agent", ["a", "b"]),
        ("ancestors", "a1", ["a", "Agent"]),
        ("ancestors", "nobody", []),
        ("descendants", "Agent", ["a", "a1", "b"]),
        ("b", "a1", ["b"]),
        ("nobody", "a1", []),
    ],
)
def test_relations_resolve_from_who_asks(
    tmp_path: Path,
    name: str,
    by: str,
    labels: list[str],
) -> None:
    assert _family(tmp_path).resolve(name, by=by) == labels


def test_a_watcher_is_its_roots_parent_and_ancestor(tmp_path: Path) -> None:
    team = _family(tmp_path)
    _ = team.join(_agent(tmp_path), label="watcher")
    team.watch("watcher", "Agent")

    assert team.resolve("parent", by="Agent") == ["watcher"]
    assert team.resolve("ancestors", by="a1") == ["a", "Agent", "watcher"]
    assert team.watched("watcher") == "Agent"
    assert team.watched("a") is None
    assert team.parent("Agent") is None
    with pytest.raises(ValueError, match="only a root is watched"):
        team.watch("watcher", "a")
    with pytest.raises(ValueError, match="is watched by 'watcher'"):
        team.watch("b", "Agent")

    _ = team.leave("watcher")
    assert team.resolve("parent", by="Agent") == []


def test_a_request_is_owed_and_a_reply_settles_it(tmp_path: Path) -> None:
    team = _family(tmp_path)
    long = "one two three four five six seven eight nine"

    team.send("Agent", "a", long)
    team.send("a", "a", "note to self")

    assert team.owes("a") == {"Agent": "one two three four five six seven eight…"}
    team.send("a", "Agent", "done")
    assert (team.owes("a"), team.owes("Agent")) == ({}, {})
    with pytest.raises(ValueError, match="Unknown agent 'nobody'"):
        team.send("a", "nobody", "hi")
    with pytest.raises(ValueError, match="No user is watching"):
        team.send("a", "user", "hi")


def test_a_message_booked_apart_is_delivered_unbooked_again(tmp_path: Path) -> None:
    """A delayed send books at once and delivers later; the delivery books nothing."""
    team = _family(tmp_path)
    shown: list[tuple[str, str]] = []
    team.to_user = lambda sender, text: shown.append((sender, text))

    team.book("Agent", "a", "do x")
    team.deliver("Agent", "a", "do x")
    team.deliver("a", "user", "fyi")

    assert team.owes("a") == {"Agent": "do x"}
    assert shown == [("a", "fyi")]
    with pytest.raises(ValueError, match="Unknown agent 'nobody'"):
        team.deliver("a", "nobody", "hi")


def test_whom_a_leaver_owed_and_was_owed_is_forgotten(tmp_path: Path) -> None:
    team = _family(tmp_path)
    team.send("Agent", "a", "x")
    team.send("Agent", "b", "y")
    agent = team["a"]

    _ = team.leave("b")
    _ = team.leave("a")

    assert team.owes("Agent") == {}
    assert not any(type(o).__name__ == "_Dun" for o in agent.runtime.observers)


@pytest.mark.asyncio
async def test_a_stopped_agent_tells_those_it_owes(tmp_path: Path) -> None:
    team = _team()
    _ = team.join(_agent(tmp_path))
    _ = team.join(
        _agent(tmp_path),
        label="a",
        parent="Agent",
        closing=AsyncExitStack(),
    )
    shown: list[tuple[str, str]] = []
    team.to_user = lambda sender, text: shown.append((sender, text))
    team.send("Agent", "a", "x")
    team.ask("a", "and you?")

    await team.stop("a")

    assert shown == [("a", "[a stopped]")]
    (told,) = team["Agent"].runtime.inbox.drain_nowait()
    assert told == AgentToAgentMessage(
        sender="a",
        recipient="Agent",
        content="[a stopped]",
    )


def test_a_reply_owed_an_unwatched_user_is_dropped(tmp_path: Path) -> None:
    team = _family(tmp_path)
    team.ask("a", "q")

    team.answer("a", "r")

    assert team.owes("a") == {}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("event", "said"),
    [
        (
            types.ModelResponseError(exception=ValueError("down")),
            "[a failed] ValueError: down",
        ),
        (types.ModelResponseCancelled(output_chars_estimate=0), "[a halted]"),
    ],
)
async def test_a_debtor_that_fails_or_halts_answers_for_itself(
    tmp_path: Path,
    event: types.RuntimeEvent,
    said: str,
) -> None:
    team = _family(tmp_path)
    team.send("Agent", "a", "x")

    team["a"].runtime.publish(event)

    (told,) = team["Agent"].runtime.inbox.drain_nowait()
    assert isinstance(told, AgentToAgentMessage)
    assert told.content == said
    assert team.owes("a") == {}


def _idle(agent: Agent, *said: str | None) -> None:
    """Have ``agent`` reply ``said`` in one response, then go idle."""
    records = tuple(AssistantMessage(content=s) for s in said)
    agent.runtime.publish(
        types.ModelResponseComplete(response=replace(reply(), records=records)),
    )
    agent.runtime.publish(types.AgentIdle())


def test_a_nudge_names_each_creditor_and_what_it_asked(tmp_path: Path) -> None:
    team = _family(tmp_path)
    team.send("Agent", "a", "x")
    team.send("b", "a", "y")

    _idle(team["a"])

    (*_, nudge) = team["a"].runtime.inbox.drain_nowait()
    assert nudge == types.Nudge(
        content="You owe a reply to: Agent ('x'); b ('y'). Answer each message"
        " with AgentSend. Do not acknowledge this note.",
    )


def test_a_debtor_is_nudged_at_every_idle_until_it_replies(tmp_path: Path) -> None:
    team = _family(tmp_path)
    a = team["a"]
    team.send("Agent", "a", "first")
    for _ in range(5):
        _idle(a, "said, but not sent")
    assert _nudged(a) == 5
    # Its words are no reply: only an AgentSend settles the debt.
    assert _told(team["Agent"]) == []
    assert team.owes("a") == {"Agent": "first"}

    team.send("a", "Agent", "answered")
    _idle(a)
    assert _nudged(a) == 0
    assert team.owes("a") == {}


def test_an_idle_debtor_is_nudged_and_its_parent_is_not_told(tmp_path: Path) -> None:
    team = _family(tmp_path)
    a = team["a"]
    _did(a, reply("half done"))
    team.send("Agent", "a", "x")

    for _ in range(3):
        _did(a, reply(""))
    inbox = a.runtime.inbox.drain_nowait()

    assert [
        m.sender if isinstance(m, AgentToAgentMessage) else "nudge"
        for m in inbox
        if isinstance(m, (AgentToAgentMessage, types.Nudge))
    ] == ["Agent", "nudge", "nudge", "nudge"]
    assert _told(team["Agent"]) == [("a", "[a is idle] half done")]


def test_a_debtor_without_agentsend_answers_with_an_error_not_a_nudge(
    tmp_path: Path,
) -> None:
    team = _family(tmp_path)
    _ = team.join(_agent(tmp_path), label="mute", parent="a")
    team.send("b", "mute", "x")
    _ = team["mute"].runtime.inbox.drain_nowait()

    _idle(team["mute"], "would answer")

    failed = ("mute", "[mute failed] it has no AgentSend to reply with")
    assert _nudged(team["mute"]) == 0
    assert _told(team["b"]) == [failed]
    assert _told(team["a"]) == [failed]
    assert team.owes("mute") == {}


def _did(agent: Agent, *responses: Response) -> None:
    """Have ``agent`` make ``responses``, then go idle."""
    for response in responses:
        agent.runtime.publish(types.ModelResponseComplete(response=response))
    agent.runtime.publish(types.AgentIdle())


def _told(agent: Agent) -> list[tuple[str | None, str | None]]:
    """Return who sent ``agent`` what; empty its inbox."""
    return [
        (m.sender, m.content)
        for m in agent.runtime.inbox.drain_nowait()
        if isinstance(m, AgentToAgentMessage)
    ]


def _nudged(agent: Agent) -> int:
    """Return how many nudges wait in ``agent``'s inbox; empty it."""
    return sum(isinstance(m, types.Nudge) for m in agent.runtime.inbox.drain_nowait())


def test_an_idle_with_no_work_since_the_last_is_not_acted_on(tmp_path: Path) -> None:
    """As each agent's first idle after a resume: it wakes no one."""
    team = _family(tmp_path)
    team.send("Agent", "a", "x")
    _ = team["a"].runtime.inbox.drain_nowait()

    for label in ("a", "b"):
        team[label].runtime.publish(types.AgentIdle())
    _did(team["b"], reply("done"))
    team["b"].runtime.publish(types.AgentIdle())

    assert _nudged(team["a"]) == 0
    assert _told(team["Agent"]) == [("b", "[b is idle] done")]


@pytest.mark.parametrize(
    "first",
    [
        UserMessage(content="hi"),
        AgentToAgentMessage(sender="user", recipient="Agent", content="hi"),
    ],
    ids=["typed", "sent"],
)
def test_the_users_first_message_nudges_each_debtor_once(
    tmp_path: Path,
    first: UserMessage | AgentToAgentMessage,
) -> None:
    team = _family(tmp_path)
    team.send("Agent", "a", "x")
    team.send("b", "a1", "y")
    for label in team:
        _ = team[label].runtime.inbox.drain_nowait()
    root = team["Agent"].runtime

    root.publish(AgentToAgentMessage(sender="b", recipient="Agent", content="hi"))
    assert [_nudged(team[label]) for label in team] == [0, 0, 0, 0]
    root.publish(first)
    root.publish(first)

    assert [_nudged(team[label]) for label in team] == [0, 1, 0, 1]


def _send(to: str, call_id: str = "s1") -> ToolCall:
    return call("AgentSend", call_id, to=to, content="done")


@pytest.mark.parametrize(
    ("responses", "told"),
    [
        ([reply("working", calls=[_send("Agent")])], []),
        ([reply("working", calls=[_send("parent")])], []),
        ([reply("working", calls=[_send("Agent")]), reply("sent it")], []),
        (
            [reply("working", calls=[_send("Agent")]), reply(calls=[call("Read")])],
            [("a", "[a is idle] working")],
        ),
        ([reply("working", calls=[_send("b")])], [("a", "[a is idle] working")]),
        ([reply("working"), reply("")], [("a", "[a is idle] working")]),
    ],
    ids=[
        "sent-it",
        "sent-parent",
        "sent-then-said",
        "sent-then-worked",
        "sent-another",
        "said",
    ],
)
def test_a_child_that_stops_tells_its_parent_unless_its_last_call_was_to_it(
    tmp_path: Path,
    responses: list[Response],
    told: list[tuple[str, str]],
) -> None:
    team = _family(tmp_path)

    _did(team["a"], *responses)

    assert _told(team["Agent"]) == told


def test_no_debtor_is_answered_for_whether_it_owes_its_parent_or_another(
    tmp_path: Path,
) -> None:
    team = _family(tmp_path)
    team.send("Agent", "a", "x")
    team.send("b", "a1", "y")
    assert _told(team["a"]) == [("Agent", "x")]
    assert _nudged(team["a1"]) == 0

    for _ in range(3):
        _did(team["a"], reply("a so far"))
        _did(team["a1"], reply("a1 so far"))

    assert _nudged(team["a"]) == 3
    assert _nudged(team["a1"]) == 3
    assert _told(team["Agent"]) == []
    assert _told(team["b"]) == []


@pytest.mark.parametrize(
    ("event", "said"),
    [
        (
            types.ModelResponseError(exception=ValueError("down")),
            "[a failed] ValueError: down",
        ),
        (types.ModelResponseCancelled(output_chars_estimate=0), "[a halted]"),
    ],
)
def test_a_child_that_fails_or_halts_owing_no_one_tells_its_parent(
    tmp_path: Path,
    event: types.RuntimeEvent,
    said: str,
) -> None:
    """It waits for the user then, so it never idles; the parent hears it stopped."""
    team = _family(tmp_path)
    a = team["a"].runtime
    a.publish(types.ModelResponseComplete(response=reply(calls=[_send("Agent")])))
    a.publish(event)
    _did(team["a"], reply("back"))

    assert _told(team["Agent"]) == [("a", said), ("a", "[a is idle] back")]


def test_a_run_that_says_nothing_tells_nothing_said_before_it(tmp_path: Path) -> None:
    team = _family(tmp_path)
    _did(team["a"], reply("old news"))
    _did(team["a"], reply(calls=[call("Read")]))
    team.send("Agent", "a", "x")
    for _ in range(3):
        _did(team["a"], reply(calls=[call("Read")]))

    assert _told(team["Agent"]) == [
        ("a", "[a is idle] old news"),
        ("a", "[a is idle]"),
    ]
    assert _nudged(team["a"]) == 3


def test_the_users_first_message_spares_the_agent_it_went_to(tmp_path: Path) -> None:
    """That one is at work on it already."""
    team = _family(tmp_path)
    team.send("b", "Agent", "q")
    team.send("Agent", "a", "x")
    for label in team:
        _ = team[label].runtime.inbox.drain_nowait()

    team["Agent"].runtime.publish(UserMessage(content="hi"))

    assert [_nudged(team[label]) for label in team] == [0, 1, 0, 0]


def test_a_root_and_a_child_whose_spawn_awaits_it_tell_no_one(tmp_path: Path) -> None:
    team = _family(tmp_path)
    team.awaited.add("b")

    _did(team["Agent"], reply("root done"))
    _did(team["b"], reply("b done"))

    assert [_told(team[label]) for label in team] == [[], [], [], []]


@pytest.mark.asyncio
async def test_debts_stand_in_the_journal_and_come_back(tmp_path: Path) -> None:
    path = tmp_path / "a.jsonl"
    async with JsonlJournal(path) as journal:
        team = _team()
        _ = team.join(_agent(tmp_path))
        agent = Agent(
            model=ScriptedProvider()["scripted"],
            cwd=tmp_path,
            journal=journal,
        )
        async with agent:
            _ = team.join(agent, label="a", parent="Agent")
            team.send("Agent", "a", "x")
    async with JsonlJournal(path) as journal:
        again = _team()
        _ = again.join(_agent(tmp_path))
        agent = Agent(
            model=ScriptedProvider()["scripted"],
            cwd=tmp_path,
            journal=journal,
        )
        async with agent:
            _ = again.join(agent, label="a", parent="Agent")
            assert again.owes("a") == {"Agent": "x"}
            again.answer("a", "late")
            assert again.owes("a") == {}


@pytest.mark.asyncio
async def test_a_watcher_stands_in_the_journal_it_watches_until_stopped(
    tmp_path: Path,
) -> None:
    team = _team()
    _ = team.join(_agent(tmp_path))
    _ = team.join(_agent(tmp_path), label="watcher", closing=AsyncExitStack())
    team.watch("watcher", "Agent", journal="/j/watcher.jsonl")

    (stood,) = team["Agent"].session.standing()
    await team.stop("watcher")

    assert (stood.kind, stood.payload) == (
        "watcher",
        {"label": "watcher", "journal": "/j/watcher.jsonl"},
    )
    assert team["Agent"].session.standing() == []


def test_choices_roll_from_the_nearest_ancestor_and_go_with_their_agent(
    tmp_path: Path,
) -> None:
    team = _family(tmp_path)
    team.choose("Agent", "model.thinking.effort", "high")
    team.choose("a", "model.thinking.effort", "low")
    team.choose("Agent", "screen.thinking", "hide")

    assert team.inherited("a1") == {
        "model.thinking.effort": "low",
        "screen.thinking": "hide",
    }
    assert team.inherited("Agent") == {}
    team.choose("a", "model.thinking.effort", None)
    assert team.inherited("a1")["model.thinking.effort"] == "high"
    _ = team.leave("a1")
    _ = team.leave("a")
    assert "a" not in team.choices


@pytest.mark.asyncio
async def test_choices_stand_in_the_journal_and_come_back(tmp_path: Path) -> None:
    path = tmp_path / "a.jsonl"
    for chosen in ("high", None):
        async with JsonlJournal(path) as journal:
            team = _team()
            agent = Agent(
                model=ScriptedProvider()["scripted"],
                cwd=tmp_path,
                journal=journal,
            )
            async with agent:
                _ = team.join(agent)
                if chosen is not None:
                    team.choose("Agent", "model.thinking.effort", chosen)
                else:
                    assert team.choices == {"Agent": {"model.thinking.effort": "high"}}


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
