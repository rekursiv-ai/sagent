"""Tests for the team: labels, who spawned whom, depth caps, persistent children,
and the login each member's model signs in as.
"""

from __future__ import annotations

from collections import deque
from contextlib import AsyncExitStack
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, override

import asyncio
import contextlib
import logging
import threading
import time

import pytest

from sagent.agent.agent import Agent
from sagent.agent.team import ChildSpec, MessageKind, Team
from sagent.lib.userdirs import config_dir
from sagent.model.exceptions import (
    AuthError,
    OverloadedError,
    RateLimitedError,
    UnknownModelError,
)
from sagent.model.scripted import (
    Reply,
    ScriptedModel,
    ScriptedProvider,
    call,
    reply,
)
from sagent.providers import login as login_module
from sagent.providers.catalog import ModelCatalog
from sagent.providers.login import Logins, Spent
from sagent.providers.testing import first_live, window
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
from sagent.tools.types import ToolContext


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from treekle import PlainTree

    from sagent.model.types import Request, Response
    from sagent.providers.types import (
        Model,
        ProviderFactory,
        ProviderResources,
    )
    from sagent.runtime.tape import ContextSplice, TapeRecord, TapeRef


def _team() -> Team:
    return Team(catalog=ModelCatalog(factories={"scripted": ScriptedProvider}))


def _agent(tmp_path: Path, team: Team | None = None) -> Agent:
    """Return an agent; one in ``team`` can reply by AgentSend."""
    return Agent(
        model=ScriptedProvider()["scripted"],
        tool_context=ToolContext(cwd=tmp_path),
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
    assert team.child_label("Agent") == "Agent_2"
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
    assert team.child_label("Agent") == "Agent_0"
    assert team.child_label(None) == "Agent"
    assert "Agent_0" not in team


def test_a_child_may_be_named(tmp_path: Path) -> None:
    team = _team()
    _ = team.join(_agent(tmp_path))

    assert team.join(_agent(tmp_path), label="reviewer", parent="Agent") == "reviewer"
    assert team.child_label("reviewer") == "reviewer_0"


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
        (2, None, (True, True, False)),
    ],
    ids=[
        "no-cap",
        "one-generation",
        "leaf-root",
        "child-tightens",
        "child-cannot-loosen",
        "leaf-child",
        "two-generations",
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
    published: list[tuple[str, ...]] = []
    team.changes.add(lambda: published.append(tuple(team)))

    await team.stop("a")

    assert closed == ["a1", "a"]
    assert list(team) == ["Agent", "b"]
    assert published == [
        ("Agent", "a", "b", "oneshot"),
        ("Agent", "a", "b"),
        ("Agent", "b"),
    ]
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

    refused = r"^Only a persistent child has a spec to stand\.$"
    with pytest.raises(ValueError, match=refused):
        _ = team.join(_agent(tmp_path), parent="Agent", spec=_spec("x"))
    with pytest.raises(ValueError, match=refused):
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


def test_a_spec_stood_after_other_records_is_found(tmp_path: Path) -> None:
    team = _team()
    root = _agent(tmp_path)
    _ = team.join(root)
    root.session.stand("other", UncategorizedRecord(kind="note", payload={}))

    _ = team.join(
        _agent(tmp_path),
        label="a",
        parent="Agent",
        closing=AsyncExitStack(),
        spec=_spec("a"),
    )

    assert team.standing("Agent") == [_spec("a")]


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
    (stood,) = root.session.standing()

    with caplog.at_level(logging.WARNING):
        assert team.standing("Agent") == []
    assert caplog.messages == [f"skipping an unreadable child record: {stood}"]


@pytest.mark.parametrize(
    ("old", "model"),
    [
        (
            {"provider": "scripted", "model": "scripted+1m", "thinking": "adaptive"},
            "scripted/scripted+1m",
        ),
        ({"system": "S", "effort": "high", "budget": "auto", "output": "text"}, ""),
        ({"model": "scripted+1m"}, ""),
    ],
    ids=["with-thinking", "with-settings", "model-without-provider"],
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

    assert team.watching == {"watcher": True}
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
    assert team.watching == {}


def test_a_request_is_owed_and_a_reply_settles_it(tmp_path: Path) -> None:
    team = _family(tmp_path)
    long = "one two three four five six seven eight nine"

    team.send("Agent", "a", long)
    team.send("a", "a", "note to self")
    team.send("b", "a1", long.removesuffix(" nine"))

    assert team.owes("a") == {"Agent": "one two three four five six seven eight…"}
    assert team.owes("a1") == {"b": "one two three four five six seven eight"}
    team.send("a", "Agent", "done")
    assert (team.owes("a"), team.owes("Agent")) == ({}, {})
    with pytest.raises(ValueError, match="Unknown agent 'nobody'"):
        team.send("a", "nobody", "hi")
    with pytest.raises(
        ValueError,
        match=r"^No user is watching; there is no one to tell\.$",
    ):
        team.send("a", "user", "hi")


def test_a_message_to_the_user_is_shown_and_owed_no_reply(tmp_path: Path) -> None:
    team = _family(tmp_path)
    shown: list[tuple[str, str]] = []
    team.to_user = lambda sender, text: shown.append((sender, text))

    team.send("a", "user", "fyi")

    assert shown == [("a", "fyi")]
    assert team.owes("user") == {}


def test_settled_debts_and_dropped_choices_leave_nothing_standing(
    tmp_path: Path,
) -> None:
    team = _family(tmp_path)
    team.send("Agent", "a", "x")
    team.send("a", "Agent", "done")
    team.ask("b", "q")
    team.answer("b", "r")
    team.choose("a1", "screen.thinking", "hide")
    team.choose("a1", "screen.thinking", None)
    # Giving up a setting never chosen is no error.
    team.choose("a1", "model.thinking.effort", None)

    assert [team[label].session.standing() for label in team] == [[], [], [], []]


def test_a_message_booked_apart_is_delivered_unbooked_again(tmp_path: Path) -> None:
    """A delayed send books at once and delivers later; the delivery books nothing."""
    team = _family(tmp_path)
    shown: list[tuple[str, str]] = []
    team.to_user = lambda sender, text: shown.append((sender, text))

    assert team.book("Agent", "a", "do x") == "ask"
    team.deliver("Agent", "a", "do x", kind="ask")
    assert team.book("a", "user", "fyi") == "tell"
    team.deliver("a", "user", "fyi", kind="tell")

    assert team.owes("a") == {"Agent": "do x"}
    assert team.book("a", "Agent", "done") == "reply"
    assert team.owes("a") == {}
    assert shown == [("a", "fyi")]
    with pytest.raises(ValueError, match="Unknown agent 'nobody'"):
        team.deliver("a", "nobody", "hi", kind="ask")


def test_whom_a_leaver_owed_and_was_owed_is_forgotten(tmp_path: Path) -> None:
    team = _family(tmp_path)
    team.send("Agent", "a", "x")
    team.send("Agent", "b", "y")
    agent = team["a"]

    _ = team.leave("b")
    _ = team.leave("a")

    assert team.owes("Agent") == {}
    assert not any(type(o).__name__ == "_Dun" for o in agent.runtime.observers)


def test_one_who_rejoins_under_a_leaver_s_label_takes_up_none_of_its_ties(
    tmp_path: Path,
) -> None:
    team = _family(tmp_path)
    team.send("Agent", "a", "x")
    team.send("a", "b", "y")
    team.awaited.add("a")

    _ = team.leave("a1")
    _ = team.leave("a")
    _ = team.join(_agent(tmp_path, team), label="a", parent="Agent")
    _complete_responses_and_idle(team["a"], reply("back"))

    assert (team.owes("a"), team.owes("b")) == ({}, {})
    assert _told(team["Agent"]) == [("a", "[a is idle] back")]


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


@pytest.mark.asyncio
async def test_a_child_stopped_with_its_parent_tells_those_it_owes(
    tmp_path: Path,
) -> None:
    team = _team()
    _ = team.join(_agent(tmp_path))
    for label, parent in (("a", "Agent"), ("a1", "a")):
        _ = team.join(
            _agent(tmp_path),
            label=label,
            parent=parent,
            closing=AsyncExitStack(),
        )
    team.send("Agent", "a1", "x")

    await team.stop("a")

    assert _told(team["Agent"]) == [("a1", "[a1 stopped]")]


def test_a_reply_owed_an_unwatched_user_is_dropped(tmp_path: Path) -> None:
    team = _family(tmp_path)
    team.ask("a", "q")
    assert team.owes("a") == {"user": "q"}

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


@pytest.mark.parametrize("allowed", [None, frozenset({"AgentSend"})])
def test_a_nudge_names_each_creditor_and_what_it_asked(
    tmp_path: Path,
    allowed: frozenset[str] | None,
) -> None:
    team = _family(tmp_path)
    team["a"].allowed = allowed
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
    _complete_responses_and_idle(a, reply("half done"))
    team.send("Agent", "a", "x")

    for _ in range(3):
        _complete_responses_and_idle(a, reply(""))
    inbox = a.runtime.inbox.drain_nowait()

    assert [
        m.sender if isinstance(m, AgentToAgentMessage) else "nudge"
        for m in inbox
        if isinstance(m, (AgentToAgentMessage, types.Nudge))
    ] == ["Agent", "nudge", "nudge", "nudge"]
    assert _told(team["Agent"]) == [("a", "[a is idle] half done")]


@pytest.mark.parametrize("disabled", [False, True])
@pytest.mark.parametrize("wake", [False, True])
def test_a_debtor_without_agentsend_answers_with_an_error_not_a_nudge(
    tmp_path: Path,
    disabled: bool,
    wake: bool,
) -> None:
    team = _family(tmp_path)
    mute = _agent(tmp_path, team if disabled else None)
    mute.allowed = frozenset()
    _ = team.join(mute, label="mute", parent="a")
    team.send("b", "mute", "x")
    _ = team["mute"].runtime.inbox.drain_nowait()

    if wake:
        team.wake("Agent")
    else:
        _idle(team["mute"], "would answer")

    failed = ("mute", "[mute failed] it has no AgentSend to reply with")
    assert _nudged(team["mute"]) == 0
    assert _told(team["b"]) == [failed]
    assert _told(team["a"]) == [failed]
    assert team.owes("mute") == {}


@pytest.mark.parametrize("wake", [False, True], ids=["idle", "resume"])
@pytest.mark.parametrize("missing", [False, True], ids=["excluded", "absent"])
def test_an_unavailable_reply_tool_fails_once_without_waking_the_debtor(
    tmp_path: Path,
    wake: bool,
    missing: bool,
) -> None:
    team = _family(tmp_path)
    debtor = _agent(tmp_path) if missing else _agent(tmp_path, team)
    debtor.allowed = None if missing else frozenset()
    team.join(debtor, label="mute", parent="Agent")
    team.send("Agent", "mute", "first")
    team.send("b", "mute", "second")
    if wake:
        team.wake("Agent")
    else:
        _idle(debtor, "cannot reply")
    assert _nudged(debtor) == 0
    assert team.owes("mute") == {}
    assert debtor.allowed == (None if missing else frozenset())
    assert ("AgentSend" in debtor.tools) is not missing
    failed = ("mute", "[mute failed] it has no AgentSend to reply with")
    assert _told(team["Agent"]) == [failed]
    assert _told(team["b"]) == [failed]
    assert (team.owes("Agent"), team.owes("b")) == ({}, {})
    team.wake("Agent")
    debtor.runtime.publish(types.AgentIdle())
    assert _nudged(debtor) == 0
    assert _told(team["Agent"]) == []
    assert _told(team["b"]) == []


@pytest.mark.parametrize("wake", [False, True], ids=["idle", "resume"])
def test_an_explicitly_allowed_reply_tool_keeps_the_reply_obligation(
    tmp_path: Path,
    wake: bool,
) -> None:
    team = _family(tmp_path)
    debtor = team["a"]
    debtor.allowed = frozenset({"AgentSend"})
    team.send("b", "a", "look")
    if wake:
        team.wake("Agent")
    else:
        _idle(debtor)
    assert _nudged(debtor) == 1
    assert team.owes("a") == {"b": "look"}
    team.send("a", "b", "done")
    assert team.owes("a") == {}
    assert _told(team["b"]) == [("a", "done")]


@pytest.mark.asyncio
async def test_the_round_cap_preserves_reply_enforcement(tmp_path: Path) -> None:
    team = _family(tmp_path)
    debtor = team["a"]
    debtor.max_rounds = 1
    async with debtor:
        team.send("b", "a", "look")
        debtor.runtime.publish(
            types.ModelResponseComplete(response=reply(calls=[call("Read")])),
        )
        assert debtor.allowed is None
        assert debtor.runtime.allowed == frozenset({"AgentSend"})
        team.wake("Agent")
        assert _nudged(debtor) == 1
        assert team.owes("a") == {"b": "look"}
        _idle(debtor)
        assert debtor.allowed is None
        assert _nudged(debtor) == 1
        team.send("a", "b", "done")
        assert team.owes("a") == {}
        assert _told(team["b"]) == [("a", "done")]


@pytest.mark.asyncio
async def test_an_excluded_reply_tool_settles_restored_debt_durably(
    tmp_path: Path,
) -> None:
    path = tmp_path / "debtor.jsonl"
    for stage in ("write", "settle", "verify"):
        async with JsonlJournal(path) as journal:
            team = _team()
            team.join(_agent(tmp_path))
            debtor = Agent(
                model=ScriptedProvider()["scripted"],
                tools=[AgentSend(team=team)],
                allowed=[],
                tool_context=ToolContext(cwd=tmp_path),
                journal=journal,
            )
            async with debtor:
                team.join(debtor, label="worker", parent="Agent")
                if stage == "write":
                    team.send("Agent", "worker", "look")
                elif stage == "settle":
                    assert team.owes("worker") == {"Agent": "look"}
                    team.wake("Agent")
                    assert _nudged(debtor) == 0
                    assert team.owes("worker") == {}
                    assert _told(team["Agent"]) == [
                        ("worker", "[worker failed] it has no AgentSend to reply with"),
                    ]
                else:
                    assert team.owes("worker") == {}
                    team.wake("Agent")
                    assert _nudged(debtor) == 0
                    assert _told(team["Agent"]) == []


def _complete_responses_and_idle(agent: Agent, *responses: Response) -> None:
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
    _complete_responses_and_idle(team["b"], reply("done"))
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

    _complete_responses_and_idle(team["a"], *responses)

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
        _complete_responses_and_idle(team["a"], reply("a so far"))
        _complete_responses_and_idle(team["a1"], reply("a1 so far"))

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
    _complete_responses_and_idle(team["a"], reply("back"))

    assert _told(team["Agent"]) == [("a", said), ("a", "[a is idle] back")]


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
def test_what_a_child_said_before_it_failed_is_not_told_again(
    tmp_path: Path,
    event: types.RuntimeEvent,
    said: str,
) -> None:
    team = _family(tmp_path)
    a = team["a"].runtime
    a.publish(types.ModelResponseComplete(response=reply("half done")))
    a.publish(event)
    _complete_responses_and_idle(team["a"], reply(calls=[call("Read")]))

    assert _told(team["Agent"]) == [("a", said), ("a", "[a is idle]")]


def test_an_idle_child_tells_its_parent_all_its_reply_said(tmp_path: Path) -> None:
    team = _family(tmp_path)

    _idle(team["a"], "one ", None, "two")

    assert _told(team["Agent"]) == [("a", "[a is idle] one two")]


# Noon local, so the clock reads the same in any zone.
_NOON = time.mktime((2026, 10, 28, 12, 0, 0, 0, 0, -1))


def test_a_child_waiting_out_a_spent_quota_tells_its_parent_once(
    tmp_path: Path,
) -> None:
    """It may wait a week, saying nothing; a backoff's short waits stay unsaid."""
    team = _family(tmp_path)
    a = team["a"].runtime
    for exception, attempt in (
        (RateLimitedError("spent"), 1),
        (RateLimitedError("spent"), 2),
        (OverloadedError("busy"), 1),
    ):
        a.publish(
            types.ModelCallSuspended(
                exception=exception,
                until=_NOON + 5.0,
                attempt=attempt,
            ),
        )

    assert _told(team["Agent"]) == [
        ("a", "[a waiting until Wed Oct 28 12:00:05] RateLimitedError: spent"),
    ]


def test_a_run_that_says_nothing_tells_nothing_said_before_it(tmp_path: Path) -> None:
    team = _family(tmp_path)
    _complete_responses_and_idle(team["a"], reply("old news"))
    _complete_responses_and_idle(team["a"], reply(calls=[call("Read")]))
    team.send("Agent", "a", "x")
    for _ in range(3):
        _complete_responses_and_idle(team["a"], reply(calls=[call("Read")]))

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

    _complete_responses_and_idle(team["Agent"], reply("root done"))
    _complete_responses_and_idle(team["b"], reply("b done"))

    assert [_told(team[label]) for label in team] == [[], [], [], []]


@pytest.mark.asyncio
async def test_debts_stand_in_the_journal_and_come_back(tmp_path: Path) -> None:
    path = tmp_path / "a.jsonl"
    async with JsonlJournal(path) as journal:
        team = _team()
        _ = team.join(_agent(tmp_path))
        agent = Agent(
            model=ScriptedProvider()["scripted"],
            tool_context=ToolContext(cwd=tmp_path),
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
            tool_context=ToolContext(cwd=tmp_path),
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
    assert team.resolve("parent", by="Agent") == []


def test_choices_roll_from_the_nearest_ancestor_and_go_with_their_agent(
    tmp_path: Path,
) -> None:
    team = _family(tmp_path)
    team.choose("Agent", "model.thinking.effort", "high")
    assert team.inherited("a1") == {"model.thinking.effort": "high"}
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
                tool_context=ToolContext(cwd=tmp_path),
                journal=journal,
            )
            async with agent:
                _ = team.join(agent)
                if chosen is not None:
                    team.choose("Agent", "model.thinking.effort", chosen)
                else:
                    assert team.choices == {"Agent": {"model.thinking.effort": "high"}}


@dataclass(slots=True, kw_only=True)
class _Accounts:
    """Each account's script, shared by every provider built for it, and who served."""

    replies: dict[str, deque[Reply]] = field(default_factory=dict[str, deque[Reply]])
    served: list[str] = field(default_factory=list[str])
    modelless: set[str] = field(default_factory=set[str])
    """Accounts whose provider serves no model."""

    def script(self, account: str, *replies: Reply) -> None:
        self.replies.setdefault(account, deque()).extend(replies)

    def factories(self, account: str | None) -> dict[str, ProviderFactory]:
        def build(*, name: str, resources: ProviderResources) -> _Signing:
            return _Signing(name=name, resources=resources, account=account, of=self)

        return {"sub": build, "plain": ScriptedProvider}


class _Signing(ScriptedProvider):
    """A provider signed in as ``account``, answering from that account's script."""

    def __init__(
        self,
        *,
        name: str,
        resources: ProviderResources,
        account: str | None,
        of: _Accounts,
    ) -> None:
        super().__init__(name=name, resources=resources)
        self.account = account or "default"
        self._of = of
        self.replies = of.replies.setdefault(self.account, deque())

    @property
    def logins(self) -> Logins:
        return Logins(config_dir() / "rekursiv-ai" / "secrets" / "sub.json")

    async def login(self, *, headless: bool = False) -> None:
        del headless

    @override
    def __getitem__(self, model: str) -> ScriptedModel:
        if self.account in self._of.modelless:
            raise UnknownModelError(f"No model {model!r}.")
        return super().__getitem__(model)

    @override
    def next_reply(self, request: Request) -> Response:
        self._of.served.append(self.account)
        return super().next_reply(request)


def _saved(*accounts: str) -> Logins:
    """Save a login file for ``default`` and each of ``accounts``; return their logins."""
    logins = Logins(config_dir() / "rekursiv-ai" / "secrets" / "sub.json")
    logins.default.parent.mkdir(parents=True, exist_ok=True)
    for account in ("default", *accounts):
        logins.path(account).write_text("{}")
    return logins


def _spent(window: str = "5-hour", *, wait_sec: float = 3_600.0) -> RateLimitedError:
    return RateLimitedError("spent", retry_after_sec=wait_sec, limit=window)


async def _session(
    accounts: _Accounts,
    tmp_path: Path,
    *,
    account: str | None = None,
    persistent_retry: bool = False,
) -> tuple[Team, Agent, list[types.RuntimeEvent]]:
    catalog = ModelCatalog(factories=accounts.factories, account=account)
    team = Team(catalog=catalog)
    agent = Agent(
        model=catalog["sub"]["scripted"],
        tool_context=ToolContext(cwd=tmp_path),
        account=account,
        persistent_retry=persistent_retry,
    )
    seen: list[types.RuntimeEvent] = []
    agent.runtime.observers.append(seen.append)
    team.join(agent)
    await agent.aopen()
    return team, agent, seen


def _switches(seen: list[types.RuntimeEvent]) -> list[tuple[str, str]]:
    return [(e.was, e.now) for e in seen if isinstance(e, types.LoginSwitched)]


@pytest.mark.asyncio
async def test_a_spent_window_moves_the_call_to_the_next_login_at_once(
    tmp_path: Path,
) -> None:
    logins = _saved("personal")
    accounts = _Accounts()
    accounts.script("default", _spent("5-hour"))
    accounts.script("personal", reply("from personal"))
    _, agent, seen = await _session(accounts, tmp_path)
    try:
        assert await agent.send("hi") == "from personal"
    finally:
        await agent.aclose()
    assert accounts.served == ["default", "personal"]
    assert _switches(seen) == [("default", "personal")]
    mark = window(logins, "default", "scripted", now=time.time())
    assert mark is not None
    assert mark.window == "5-hour"
    # The journal keeps the account the user chose, not the one spilled to.
    assert agent.account is None


@pytest.mark.asyncio
async def test_a_window_that_names_no_reset_is_spent_an_hour(tmp_path: Path) -> None:
    _saved("personal")
    accounts = _Accounts()
    accounts.script("default", RateLimitedError("spent", limit="5-hour"))
    accounts.script("personal", reply("from personal"))
    _, agent, seen = await _session(accounts, tmp_path)
    began = time.time()
    try:
        assert await agent.send("hi") == "from personal"
    finally:
        await agent.aclose()
    (moved,) = [e for e in seen if isinstance(e, types.LoginSwitched)]
    assert (moved.provider, moved.was, moved.now, moved.window, moved.first) == (
        "sub",
        "default",
        "personal",
        "5-hour",
        "default",
    )
    assert moved.until == pytest.approx(began + 3_600.0, abs=0.5)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("wait_sec", "served"),
    [(300.0, ["default"]), (300.5, ["default", "personal"])],
    ids=["five-minutes-is-a-throttle", "past-five-minutes-is-spent"],
)
async def test_an_unnamed_wait_past_five_minutes_is_a_spent_quota(
    tmp_path: Path,
    wait_sec: float,
    served: list[str],
) -> None:
    _saved("personal")
    accounts = _Accounts()
    accounts.script("default", _spent("", wait_sec=wait_sec))
    accounts.script("personal", reply("from personal"))
    _, agent, _ = await _session(accounts, tmp_path)
    try:
        with contextlib.suppress(RateLimitedError):
            await agent.send("hi")
    finally:
        await agent.aclose()
    assert accounts.served == served


@pytest.mark.asyncio
async def test_a_turn_on_a_login_with_quota_keeps_its_model_and_reads_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _saved("personal")
    accounts = _Accounts()
    accounts.script("default", reply("served"))
    _, agent, _ = await _session(accounts, tmp_path)
    model = agent.model
    reads: list[Path] = []
    raw = login_module._raw

    def counted(ledger: Path) -> dict[str, dict[str, object]]:
        reads.append(ledger)
        return raw(ledger)

    monkeypatch.setattr(login_module, "_raw", counted)
    try:
        assert await agent.send("hi") == "served"
    finally:
        await agent.aclose()
    assert agent.model is model
    assert len(reads) == 1


@pytest.mark.asyncio
async def test_a_model_the_user_picks_while_moved_signs_in_as_their_choice(
    tmp_path: Path,
) -> None:
    """Picked from the catalog, it is the chosen login's, not the one moved to."""
    logins = _saved("personal")
    accounts = _Accounts()
    accounts.script("default", _spent("5-hour"))
    accounts.script("personal", reply("one"), reply("two"))
    team, agent, seen = await _session(accounts, tmp_path)
    try:
        assert await agent.send("first") == "one"
        agent.model = team.catalog["sub"]["scripted"]
        assert await agent.send("second") == "two"
    finally:
        await agent.aclose()
    assert accounts.served == ["default", "personal", "personal"]
    assert _switches(seen) == [("default", "personal"), ("default", "personal")]
    assert first_live(logins, "scripted", now=time.time()) == "personal"


@pytest.mark.asyncio
async def test_a_turn_after_the_reset_goes_back_to_the_first_login(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [time.time()]
    monkeypatch.setattr(time, "time", lambda: clock[0])
    logins = _saved("personal")
    await logins.spend(
        None,
        Spent(window="5-hour", model="scripted", until=time.time() + 0.3),
    )
    accounts = _Accounts()
    accounts.script("personal", reply("one"))
    accounts.script("default", reply("two"))
    _, agent, seen = await _session(accounts, tmp_path)
    try:
        assert await agent.send("first") == "one"
        clock[0] += 0.4
        assert await agent.send("second") == "two"
    finally:
        await agent.aclose()
    assert accounts.served == ["personal", "default"]
    # The session starts off the spent login quietly; the move back is news.
    assert _switches(seen) == [("personal", "default")]
    (back,) = (e for e in seen if isinstance(e, types.LoginSwitched))
    assert (back.window, back.until, back.first) == ("", 0.0, "default")


@pytest.mark.asyncio
async def test_a_short_throttle_is_waited_as_ever(tmp_path: Path) -> None:
    logins = _saved("personal")
    accounts = _Accounts()
    accounts.script("default", _spent("", wait_sec=10.0))
    _, agent, seen = await _session(accounts, tmp_path)
    try:
        with pytest.raises(RateLimitedError):
            await agent.send("hi")
    finally:
        await agent.aclose()
    assert accounts.served == ["default"]
    assert _switches(seen) == []
    assert first_live(logins, "scripted", now=time.time()) == "default"


@pytest.mark.asyncio
async def test_every_login_spent_waits_for_the_earliest_reset_and_takes_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [time.time()]
    monkeypatch.setattr(time, "time", lambda: clock[0])
    logins = _saved("personal")
    accounts = _Accounts()
    accounts.script("default", _spent("weekly", wait_sec=86_400.0))
    accounts.script("personal", reply("from personal"))
    _, agent, seen = await _session(accounts, tmp_path, persistent_retry=True)
    waiting = asyncio.Event()

    def suspended(event: types.RuntimeEvent) -> None:
        if isinstance(event, types.ModelCallSuspended):
            waiting.set()

    agent.runtime.observers.append(suspended)
    soon = clock[0] + 60.0
    await logins.spend(
        "personal",
        Spent(window="5-hour", model="scripted", until=soon),
    )
    sending = asyncio.create_task(agent.send("hi"))
    try:
        await asyncio.wait_for(waiting.wait(), timeout=10)
        assert not sending.done()
        assert accounts.served == ["default"]
        clock[0] = soon
        agent.runtime.push(types.SaveSession())
        assert await asyncio.wait_for(sending, timeout=10) == "from personal"
    finally:
        sending.cancel()
        await asyncio.gather(sending, return_exceptions=True)
        await agent.aclose()
    assert accounts.served == ["default", "personal"]
    (wait,) = [e for e in seen if isinstance(e, types.ModelCallSuspended)]
    assert wait.until == soon
    assert _switches(seen) == [("default", "personal")]


@pytest.mark.asyncio
async def test_no_login_is_asked_twice_and_all_spent_waits_as_ever(
    tmp_path: Path,
) -> None:
    _saved("personal", "work")
    accounts = _Accounts()
    accounts.script("default", _spent("5-hour"))
    accounts.script("personal", _spent("weekly"))
    accounts.script("work", _spent("5-hour", wait_sec=600.0))
    _, agent, seen = await _session(accounts, tmp_path)
    try:
        with pytest.raises(RateLimitedError):
            await agent.send("hi")
    finally:
        await agent.aclose()
    assert accounts.served == ["default", "personal", "work"]
    (spent,) = [e for e in seen if isinstance(e, types.LoginsSpent)]
    assert spent.provider == "sub"
    assert [account for account, _ in spent.resets] == ["work", "default", "personal"]


@pytest.mark.asyncio
async def test_a_ledger_that_cannot_be_written_leaves_the_call_to_go_on(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Advisory state that fails must not stop the agent: before, it hung."""
    _saved("personal")

    async def unwritable(self: Logins, account: str | None, spent: Spent) -> None:
        del self, account, spent
        raise OSError("read-only file system")

    monkeypatch.setattr(Logins, "spend", unwritable)
    accounts = _Accounts()
    accounts.script("default", _spent("5-hour"), reply("answered"))
    _, agent, _ = await _session(accounts, tmp_path)
    try:
        with pytest.raises(RateLimitedError):
            await asyncio.wait_for(agent.send("hi"), timeout=5)
        # The next turn is the default's again: nothing was learned of it.
        assert await asyncio.wait_for(agent.send("again"), timeout=5) == "answered"
    finally:
        await agent.aclose()
    assert accounts.served == ["default", "default"]


@pytest.mark.asyncio
async def test_no_login_left_with_none_spent_says_nothing_of_resets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every other login is unserving or signed out; a note of no resets says nothing."""
    clock = [time.time()]
    monkeypatch.setattr(time, "time", lambda: clock[0])
    _saved("personal")
    accounts = _Accounts()
    accounts.script("default", _spent("5-hour", wait_sec=0.2))
    accounts.script("personal", reply("one"), AuthError("expired"))
    _, agent, seen = await _session(accounts, tmp_path)
    try:
        assert await agent.send("first") == "one"
        clock[0] += 0.3
        accounts.modelless.add("default")
        with pytest.raises(AuthError):
            await agent.send("second")
    finally:
        await agent.aclose()
    assert accounts.served == ["default", "personal", "personal"]
    assert [e for e in seen if isinstance(e, types.LoginsSpent)] == []


@pytest.mark.asyncio
async def test_the_ledger_is_read_off_the_event_loop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Read before every call, it would stall every other task while the disk answers."""
    _saved("personal")
    accounts = _Accounts()
    accounts.script("default", _spent("5-hour"))
    accounts.script("personal", reply("from personal"))
    _, agent, _ = await _session(accounts, tmp_path)
    looping = threading.current_thread()
    on_loop: list[bool] = []
    raw = login_module._raw

    def watched(ledger: Path) -> dict[str, dict[str, object]]:
        on_loop.append(threading.current_thread() is looping)
        return raw(ledger)

    monkeypatch.setattr(login_module, "_raw", watched)
    try:
        assert await agent.send("hi") == "from personal"
    finally:
        await agent.aclose()
    assert on_loop
    assert not any(on_loop)


@pytest.mark.asyncio
async def test_a_call_reads_the_ledger_once_per_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each account's window read alone cost a read each, on the path of every call."""
    logins = _saved("aaa", "personal", "work")
    await logins.spend(
        None,
        Spent(window="5-hour", model="scripted", until=time.time() + 60),
    )
    accounts = _Accounts()
    accounts.script("aaa", reply("from aaa"))
    _, agent, _ = await _session(accounts, tmp_path)
    reads: list[Path] = []
    raw = login_module._raw

    def counted(ledger: Path) -> dict[str, dict[str, object]]:
        reads.append(ledger)
        return raw(ledger)

    monkeypatch.setattr(login_module, "_raw", counted)
    try:
        assert await agent.send("hi") == "from aaa"
    finally:
        await agent.aclose()
    # One check before the call, one to pick the login it moves to.
    assert len(reads) == 2


@pytest.mark.asyncio
async def test_a_login_signed_out_a_day_ago_is_tried_again(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logins = _saved("other")
    now = time.time()
    monkeypatch.setattr(time, "time", lambda: now - 2 * 86_400)
    await logins.sign_out("other")
    monkeypatch.setattr(time, "time", lambda: now)
    await logins.spend(
        None,
        Spent(window="5-hour", model="scripted", until=now + 60),
    )
    accounts = _Accounts()
    accounts.script("other", reply("from other"))
    _, agent, _ = await _session(accounts, tmp_path)
    try:
        assert await agent.send("hi") == "from other"
    finally:
        await agent.aclose()
    assert accounts.served == ["other"]


@pytest.mark.asyncio
async def test_a_ledger_that_cannot_be_read_before_a_call_leaves_it_to_go_on(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failing check before the call is logged; the call is still made."""
    _saved("personal")

    def unreadable(
        self: Logins,
        model: str,
        *,
        now: float,
        first: str | None = None,
    ) -> None:
        del self, model, now, first
        raise OSError("read error")

    monkeypatch.setattr(Logins, "ranked", unreadable)
    accounts = _Accounts()
    accounts.script("default", reply("answered"))
    _, agent, _ = await _session(accounts, tmp_path)
    try:
        assert await asyncio.wait_for(agent.send("hi"), timeout=5) == "answered"
    finally:
        await agent.aclose()


class _Recording:
    """A compactor that notes which login it was given, then fails."""

    def __init__(self) -> None:
        self.logins: list[str] = []
        self.asked = asyncio.Event()

    async def compact(
        self,
        tape: Sequence[TapeRecord],
        context: Sequence[types.ModelContextEvent],
        model: Model,
        mint_ref: Callable[[], TapeRef],
        custom_instructions: str | None = None,
    ) -> ContextSplice:
        del tape, context, mint_ref, custom_instructions
        provider = model.provider
        assert isinstance(provider, _Signing)
        self.logins.append(provider.account)
        self.asked.set()
        raise RuntimeError("recorded")


@pytest.mark.asyncio
async def test_a_compaction_on_a_spent_login_moves_to_one_with_quota_first(
    tmp_path: Path,
) -> None:
    logins = _saved("personal")
    accounts = _Accounts()
    accounts.script("default", reply("one"))
    _, agent, seen = await _session(accounts, tmp_path)
    recording = _Recording()
    agent.runtime.compactor = recording
    until = time.time() + 60
    try:
        assert await agent.send("first") == "one"
        await logins.spend(
            None,
            Spent(window="5-hour", model="scripted", until=until),
        )
        agent.runtime.push(types.Compact())
        async with asyncio.timeout(5):
            await recording.asked.wait()
    finally:
        await agent.aclose()
    assert recording.logins == ["personal"]
    (moved,) = [e for e in seen if isinstance(e, types.LoginSwitched)]
    assert (moved.window, moved.until) == ("5-hour", until)


@pytest.mark.asyncio
async def test_a_compaction_first_leaves_the_session_s_start_quiet(
    tmp_path: Path,
) -> None:
    """A compaction that moves nothing does not spend the start's quiet move."""
    logins = _saved("personal")
    await logins.prefer(("personal", "default"))
    accounts = _Accounts()
    accounts.script("personal", reply("one"))
    _, agent, seen = await _session(accounts, tmp_path)
    recording = _Recording()
    agent.runtime.compactor = recording
    try:
        agent.runtime.push(types.Compact())
        async with asyncio.timeout(5):
            await recording.asked.wait()
        assert await agent.send("hi") == "one"
    finally:
        await agent.aclose()
    assert _switches(seen) == []


@pytest.mark.asyncio
async def test_a_window_naming_no_family_of_its_model_moves_on(tmp_path: Path) -> None:
    """``weekly Oauth_apps`` names no model; read as one, the login was never left."""
    _saved("personal")
    accounts = _Accounts()
    accounts.script("default", _spent("weekly Oauth_apps"))
    accounts.script("personal", reply("from personal"))
    _, agent, seen = await _session(accounts, tmp_path)
    try:
        assert await agent.send("hi") == "from personal"
    finally:
        await agent.aclose()
    assert accounts.served == ["default", "personal"]
    assert _switches(seen) == [("default", "personal")]


@pytest.mark.asyncio
async def test_a_login_whose_provider_lacks_the_model_is_passed_over(
    tmp_path: Path,
) -> None:
    _saved("aaa", "personal")
    accounts = _Accounts(modelless={"aaa"})
    accounts.script("default", _spent("5-hour"))
    accounts.script("personal", reply("from personal"))
    _, agent, seen = await _session(accounts, tmp_path)
    try:
        assert await agent.send("hi") == "from personal"
    finally:
        await agent.aclose()
    assert accounts.served == ["default", "personal"]
    assert _switches(seen) == [("default", "personal")]


@pytest.mark.asyncio
async def test_a_login_passed_over_stays_passed_over_the_next_turn(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _saved("aaa", "personal")
    accounts = _Accounts(modelless={"aaa"})
    accounts.script("default", _spent("5-hour"))
    accounts.script("personal", reply("one"), reply("two"))
    _, agent, seen = await _session(accounts, tmp_path)
    try:
        with caplog.at_level(logging.WARNING, logger="sagent.agent.team"):
            assert await agent.send("first") == "one"
            assert await agent.send("second") == "two"
    finally:
        await agent.aclose()
    assert accounts.served == ["default", "personal", "personal"]
    assert _switches(seen) == [("default", "personal")]
    assert [
        (
            r.getMessage(),
            isinstance(r.exc_info, tuple) and r.exc_info[0] is UnknownModelError,
        )
        for r in caplog.records
        if r.name == "sagent.agent.team"
    ] == [("cannot sign sub in as aaa", True)]


@pytest.mark.asyncio
async def test_a_login_passed_over_with_nowhere_to_move_is_not_tried_again(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _saved("aaa")
    accounts = _Accounts(modelless={"aaa"})
    accounts.script("default", _spent("5-hour"), _spent("5-hour"))
    _, agent, _ = await _session(accounts, tmp_path)
    try:
        with caplog.at_level(logging.WARNING, logger="sagent.agent.team"):
            for text in ("first", "second"):
                with pytest.raises(RateLimitedError):
                    await agent.send(text)
    finally:
        await agent.aclose()
    assert [
        r.getMessage() for r in caplog.records if r.name == "sagent.agent.team"
    ] == ["cannot sign sub in as aaa"]


@pytest.mark.asyncio
async def test_a_login_the_user_switches_to_is_where_the_order_starts(
    tmp_path: Path,
) -> None:
    _saved("personal", "work")
    accounts = _Accounts()
    accounts.script("default", _spent("5-hour"))
    accounts.script("personal", reply("one"))
    accounts.script("work", reply("two"))
    team, agent, seen = await _session(accounts, tmp_path)
    try:
        assert await agent.send("first") == "one"
        agent.model = team.catalog.signed_in("sub", account="work")["scripted"]
        agent.account = "work"
        assert await agent.send("second") == "two"
    finally:
        await agent.aclose()
    assert accounts.served == ["default", "personal", "work"]
    assert _switches(seen) == [("default", "personal")]


@pytest.mark.asyncio
async def test_a_login_passed_over_is_tried_again_once_the_user_switches(
    tmp_path: Path,
) -> None:
    """What one login could not serve says nothing of the order the user picks next."""
    _saved("aaa", "personal")
    accounts = _Accounts(modelless={"aaa"})
    accounts.script("default", _spent("5-hour"))
    accounts.script("personal", reply("one"), _spent("weekly"))
    accounts.script("aaa", reply("from aaa"))
    team, agent, _ = await _session(accounts, tmp_path)
    try:
        assert await agent.send("first") == "one"
        accounts.modelless.clear()
        agent.model = team.catalog.signed_in("sub", account="personal")["scripted"]
        agent.account = "personal"
        assert await agent.send("second") == "from aaa"
    finally:
        await agent.aclose()
    assert accounts.served == ["default", "personal", "personal", "aaa"]


@pytest.mark.asyncio
async def test_a_login_moved_onto_and_found_signed_out_is_passed_over(
    tmp_path: Path,
) -> None:
    _saved("personal", "work")
    accounts = _Accounts()
    accounts.script("default", _spent("5-hour"))
    accounts.script("personal", AuthError("expired"))
    accounts.script("work", reply("from work"))
    _, agent, seen = await _session(accounts, tmp_path)
    try:
        assert await agent.send("hi") == "from work"
    finally:
        await agent.aclose()
    assert accounts.served == ["default", "personal", "work"]
    assert _switches(seen) == [("default", "personal"), ("personal", "work")]
    (_, out) = [e for e in seen if isinstance(e, types.LoginSwitched)]
    assert (out.window, out.until) == ("signed out", 0.0)


@pytest.mark.asyncio
async def test_the_chosen_login_signed_out_is_the_user_s_to_fix(
    tmp_path: Path,
) -> None:
    _saved("personal")
    accounts = _Accounts()
    accounts.script("default", AuthError("expired"))
    _, agent, seen = await _session(accounts, tmp_path)
    try:
        with pytest.raises(AuthError):
            await agent.send("hi")
    finally:
        await agent.aclose()
    assert _switches(seen) == []


@pytest.mark.asyncio
async def test_a_window_of_one_family_leaves_the_login_for_the_others(
    tmp_path: Path,
) -> None:
    logins = _saved("personal")
    await logins.spend(
        None,
        Spent(window="weekly Opus", model="opus-5.5", until=time.time() + 60),
    )
    accounts = _Accounts()
    accounts.script("default", reply("served"))
    _, agent, _ = await _session(accounts, tmp_path)
    try:
        assert await agent.send("hi") == "served"
    finally:
        await agent.aclose()
    assert accounts.served == ["default"]


@pytest.mark.asyncio
async def test_the_chosen_login_comes_first_then_the_precedence(
    tmp_path: Path,
) -> None:
    logins = _saved("personal", "work")
    await logins.prefer(("work", "default", "personal"))
    accounts = _Accounts()
    accounts.script("personal", _spent("weekly"))
    accounts.script("work", reply("from work"))
    _, agent, seen = await _session(accounts, tmp_path, account="personal")
    try:
        assert await agent.send("hi") == "from work"
    finally:
        await agent.aclose()
    assert accounts.served == ["personal", "work"]
    assert _switches(seen) == [("personal", "work")]
    assert agent.account == "personal"


@pytest.mark.asyncio
async def test_a_new_session_starts_on_the_first_login_with_quota_without_a_note(
    tmp_path: Path,
) -> None:
    """What an earlier session learned, it does not announce again."""
    logins = _saved("other", "personal")
    await logins.spend(
        None,
        Spent(window="weekly", model="scripted", until=time.time() + 60),
    )
    await logins.sign_out("other")
    accounts = _Accounts()
    accounts.script("personal", reply("from personal"))
    _, agent, seen = await _session(accounts, tmp_path)
    try:
        assert await agent.send("hi") == "from personal"
    finally:
        await agent.aclose()
    assert accounts.served == ["personal"]
    assert _switches(seen) == []


@pytest.mark.asyncio
async def test_a_login_found_signed_out_stays_passed_over_until_signed_in_again(
    tmp_path: Path,
) -> None:
    logins = _saved("other", "personal")
    accounts = _Accounts()
    accounts.script("default", _spent("5-hour"))
    accounts.script("other", AuthError("expired"))
    accounts.script("personal", reply("from personal"))
    _, agent, _ = await _session(accounts, tmp_path)
    try:
        assert await agent.send("hi") == "from personal"
    finally:
        await agent.aclose()
    assert "other" not in logins.order()
    await logins.added("other")
    assert "other" in logins.order()


@pytest.mark.asyncio
async def test_the_precedence_picks_the_first_login_of_a_fresh_session(
    tmp_path: Path,
) -> None:
    logins = _saved("personal")
    await logins.prefer(("personal", "default"))
    accounts = _Accounts()
    accounts.script("personal", reply("from personal"))
    _, agent, seen = await _session(accounts, tmp_path)
    try:
        assert await agent.send("hi") == "from personal"
    finally:
        await agent.aclose()
    assert accounts.served == ["personal"]
    # A session's start is no switch: nothing is announced.
    assert _switches(seen) == []


@pytest.mark.asyncio
async def test_a_provider_without_logins_is_left_alone(tmp_path: Path) -> None:
    catalog = ModelCatalog(factories={"plain": ScriptedProvider})
    team = Team(catalog=catalog)
    provider = catalog["plain"]
    assert isinstance(provider, ScriptedProvider)
    provider.replies.extend([_spent("5-hour")])
    agent = Agent(model=provider["scripted"], tool_context=ToolContext(cwd=tmp_path))
    team.join(agent)
    await agent.aopen()
    try:
        with pytest.raises(RateLimitedError):
            await agent.send("hi")
    finally:
        await agent.aclose()


@dataclass(slots=True, kw_only=True)
class _Relayed:
    """A relay to another machine that keeps what it was given."""

    sent: list[tuple[str, str, str, MessageKind]] = field(
        default_factory=list[tuple[str, str, str, "MessageKind"]],
    )

    def send(
        self,
        *,
        sender: str,
        recipient: str,
        content: str,
        kind: MessageKind,
    ) -> None:
        self.sent.append((sender, recipient, content, kind))


def _reaching(tmp_path: Path, *far: str) -> tuple[Team, _Relayed]:
    """Return a team of ``Agent``, reaching ``far`` on another machine."""
    team, relay = _team(), _Relayed()
    _ = team.join(_agent(tmp_path, team))
    for label in far:
        team.reach(label, relay=relay)
    return team, relay


def test_a_message_to_an_outside_member_is_relayed_and_booked(tmp_path: Path) -> None:
    team, relay = _reaching(tmp_path, "far")
    team.send("Agent", recipient="far", content="look at a.py")
    assert relay.sent == [("Agent", "far", "look at a.py", "ask")]
    assert team.owes("far") == {"Agent": "look at a.py"}
    assert team.owes("Agent") == {}
    team.receive(sender="far", recipient="Agent", content="a.py is fine", kind="reply")
    assert team.owes("far") == {}
    assert team.owes("Agent") == {}


def test_a_note_from_outside_is_a_request_its_reply_settles(tmp_path: Path) -> None:
    team, relay = _reaching(tmp_path, "far")
    team.receive(sender="far", recipient="Agent", content="what changed?", kind="ask")
    assert _told(team["Agent"]) == [("far", "what changed?")]
    assert team.owes("Agent") == {"far": "what changed?"}
    team.send("Agent", recipient="far", content="nothing")
    assert team.owes("Agent") == {}
    assert team.owes("far") == {}
    assert relay.sent == [("Agent", "far", "nothing", "reply")]


# A request made through the far team's own tool was booked there only.
@pytest.mark.parametrize("kind", ["reply", "tell"])
def test_a_reply_or_news_from_outside_is_owed_nothing(
    tmp_path: Path,
    kind: MessageKind,
) -> None:
    team, _ = _reaching(tmp_path, "far")
    team.receive(sender="far", recipient="Agent", content="[far is idle]", kind=kind)
    assert _told(team["Agent"]) == [("far", "[far is idle]")]
    assert team.owes("Agent") == {}


def test_a_note_for_no_member_is_refused(tmp_path: Path) -> None:
    team, _ = _reaching(tmp_path, "far")
    with pytest.raises(ValueError, match="Unknown agent 'gone'"):
        team.receive(sender="far", recipient="gone", content="hi", kind="ask")


@pytest.mark.parametrize("kind", ["ask", "reply", "tell"])
def test_disabling_peers_refuses_in_transit_notes_but_keeps_private_messages(
    tmp_path: Path,
    kind: MessageKind,
) -> None:
    team, relay = _reaching(tmp_path, "far#session", "parent@laptop")
    changed: list[bool] = []
    team.changes.add(lambda: changed.append(team.peer_messaging("Agent")))
    team.set_peer_messaging("Agent", False)
    assert changed == [False]
    with pytest.raises(ValueError, match="disabled"):
        team.send("Agent", "far#session", "blocked")
    assert relay.sent == []
    assert team.owes("far#session") == {}
    with pytest.raises(ValueError, match="disabled"):
        team.receive(sender="far#session", recipient="Agent", content="late", kind=kind)
    assert _told(team["Agent"]) == []
    assert team.owes("Agent") == {}
    team.receive(
        sender="parent@laptop",
        recipient="Agent",
        content="private",
        kind=kind,
    )
    assert _told(team["Agent"]) == [("parent@laptop", "private")]
    team.set_peer_messaging("Agent", True)
    team.receive(sender="far#session", recipient="Agent", content="enabled", kind=kind)
    assert _told(team["Agent"]) == [("far#session", "enabled")]


def test_a_child_of_an_outside_member_tells_it_over_the_relay(tmp_path: Path) -> None:
    team, relay = _reaching(tmp_path, "far@laptop")
    child = _agent(tmp_path, team)
    label = team.join(child, parent="far@laptop")
    assert label == "far_0"
    assert team.child_label("far@laptop") == "far_1"
    team.reach("far@gpu1@laptop", relay=relay)
    assert team.child_label("far@gpu1@laptop") == "far@gpu1_0"
    _ = team.join(_agent(tmp_path, team), label="near@x")
    assert team.child_label("near@x") == "near@x_0"
    assert team.resolve("parent", by=label) == ["far@laptop"]
    assert team.can_spawn(label)
    assert team.can_spawn("far@laptop")
    _idle(child, "done")
    assert relay.sent == [(label, "far@laptop", f"[{label} is idle] done", "tell")]


def test_outside_members_are_reached_by_name_and_never_taken(tmp_path: Path) -> None:
    team, _ = _reaching(tmp_path, "far", "Agent_0")
    assert team.outside == ["far", "Agent_0"]
    assert team.resolve("far", by="Agent") == ["far"]
    assert team.resolve("near", by="Agent") == []
    assert team.child_label("Agent") == "Agent_1"
    with pytest.raises(ValueError, match="already labelled 'far'"):
        _ = team.join(_agent(tmp_path, team), label="far", parent="Agent")
    with pytest.raises(ValueError, match="'Agent' is a member's label"):
        team.reach("Agent", relay=_Relayed())
    with pytest.raises(ValueError, match="'parent' is a member's label"):
        team.reach("parent", relay=_Relayed())
    team.unreach("far")
    assert team.outside == ["Agent_0"]
    assert team.resolve("far", by="Agent") == []


def test_the_user_s_first_word_nudges_no_outside_debtor(tmp_path: Path) -> None:
    team, _ = _reaching(tmp_path, "far")
    team.send("Agent", recipient="far", content="look")
    team.wake("Agent")
    assert team.owes("far") == {"Agent": "look"}


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
