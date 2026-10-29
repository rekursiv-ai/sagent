"""Tests for the views: which agent the screen shows, and what follows the focus."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import AsyncExitStack
from typing import TYPE_CHECKING

import asyncio
import io

from rich.console import Console

import pytest
import pytest_asyncio

from sagent.agent.agent import Agent
from sagent.agent.team import Team
from sagent.model.scripted import ScriptedProvider, call, reply
from sagent.providers.catalog import ModelCatalog
from sagent.runtime import types
from sagent.session.types import AssistantMessage, UserMessage
from sagent.tui.console import ConsolePrinter
from sagent.tui.tree_test import _Wait
from sagent.tui.views import Views


if TYPE_CHECKING:
    from collections.abc import AsyncIterator


class _Screen:
    """A printer on a string, and what it has printed."""

    def __init__(self) -> None:
        self.file = io.StringIO()
        self.printer = ConsolePrinter(Console(file=self.file, width=80))

    def text(self) -> str:
        return self.file.getvalue()


def _catalog() -> ModelCatalog:
    return ModelCatalog(factories={})


async def _open(stack: AsyncExitStack, *replies: str) -> Agent:
    provider = ScriptedProvider(replies=[reply(r) for r in replies])
    return await stack.enter_async_context(Agent(model=provider["scripted"]))


async def _team(stack: AsyncExitStack) -> Team:
    """Return ``Agent``, and ``kid`` under it, a persistent child the team closes."""
    team = Team(catalog=_catalog())
    _ = team.join(await _open(stack, "root says"))
    kid = Agent(model=ScriptedProvider(replies=[reply("kid says")])["scripted"])
    closing = AsyncExitStack()
    await closing.enter_async_context(kid)
    _ = team.join(kid, label="kid", parent="Agent", closing=closing)
    stack.push_async_callback(team.aclose)
    return team


async def _idle(agent: Agent) -> None:
    """Wait until ``agent`` next goes idle."""
    idle = asyncio.Event()

    def watch(event: types.RuntimeEvent) -> None:
        if isinstance(event, types.AgentIdle):
            idle.set()

    agent.runtime.observers.append(watch)
    try:
        await asyncio.wait_for(idle.wait(), timeout=5)
    finally:
        agent.runtime.observers.remove(watch)


@pytest_asyncio.fixture
async def stack() -> AsyncIterator[AsyncExitStack]:
    async with AsyncExitStack() as stack:
        yield stack


@pytest.mark.asyncio
async def test_the_root_is_shown_first_its_session_replayed(
    stack: AsyncExitStack,
) -> None:
    root = await _open(stack)
    root.session.append(UserMessage(content="earlier"))
    screen = _Screen()

    views = Views(root, team=None, printer=screen.printer, catalog=_catalog())

    assert views.current.agent is root
    assert "earlier" in screen.text()
    assert views.toolbar() == views.current.status.frame()
    for observer in (views.current.scrollback, views.current.status, views.title):
        assert observer in root.runtime.observers
    await views.aclose()


@pytest.mark.asyncio
async def test_a_message_to_the_user_names_the_user_as_its_recipient(
    stack: AsyncExitStack,
) -> None:
    team = await _team(stack)
    screen = _Screen()
    views = Views(team["Agent"], team=team, printer=screen.printer, catalog=_catalog())

    team.send("kid", "user", "pong")

    assert "kid→user: pong" in screen.text()
    await views.aclose()
    assert team.to_user is None


@pytest.mark.asyncio
async def test_focus_moves_the_scrollback_and_what_is_typed(
    stack: AsyncExitStack,
) -> None:
    team = await _team(stack)
    root, kid = team["Agent"], team["kid"]
    kid.session.append(AssistantMessage(content="kid before"))
    screen, switched = _Screen(), list[str]()
    views = Views(
        root,
        team=team,
        printer=screen.printer,
        catalog=_catalog(),
        switched=lambda: switched.append(views.current.label),
    )
    before = views.current

    views.focus("kid")
    views.focus("kid")

    shown = views.current
    assert (shown.label, shown.agent, shown.parent, switched) == (
        "kid",
        kid,
        "Agent",
        ["kid"],
    )
    assert screen.text().index("── kid ──") < screen.text().index("kid before")
    assert before.scrollback not in root.runtime.observers
    assert before.status in root.runtime.observers
    assert shown.scrollback in kid.runtime.observers
    assert views.title in kid.runtime.observers
    assert views.title not in root.runtime.observers
    await shown.commands("hello kid")
    await _idle(kid)
    assert "kid says" in screen.text()
    assert "root says" not in screen.text()

    views.focus("Agent")

    assert views.current is before
    await views.aclose()
    for agent in (root, kid):
        assert not {before.status, before.scrollback, views.title} & set(
            agent.runtime.observers,
        )


@pytest.mark.asyncio
async def test_slash_focus_reaches_the_views(stack: AsyncExitStack) -> None:
    team = await _team(stack)
    views = Views(
        team["Agent"],
        team=team,
        printer=_Screen().printer,
        catalog=_catalog(),
    )

    await views.current.commands("/focus kid")

    assert views.current.label == "kid"
    await views.aclose()


@pytest.mark.asyncio
async def test_an_unknown_label_is_refused(stack: AsyncExitStack) -> None:
    views = Views(
        await _open(stack),
        team=None,
        printer=_Screen().printer,
        catalog=_catalog(),
    )

    with pytest.raises(KeyError):
        views.focus("nobody")
    await views.aclose()


@pytest.mark.asyncio
async def test_a_lone_agent_is_its_own_tree(stack: AsyncExitStack) -> None:
    root = await _open(stack)
    views = Views(root, team=None, printer=_Screen().printer, catalog=_catalog())
    lineage = views.lineage

    assert (list(lineage), lineage[""], lineage.parent("")) == ([""], root, None)
    assert isinstance(lineage, Mapping)
    assert len(lineage) == 1
    with pytest.raises(KeyError):
        _ = lineage["kid"]
    await views.aclose()


@pytest.mark.asyncio
async def test_when_the_shown_agent_leaves_its_parent_is_shown(
    stack: AsyncExitStack,
) -> None:
    team = await _team(stack)
    screen = _Screen()
    views = Views(team["Agent"], team=team, printer=screen.printer, catalog=_catalog())
    views.focus("kid")
    gone = views.current

    await team.stop("kid")
    views.follow()

    assert views.current.label == "Agent"
    assert "[kid stopped]" in screen.text()
    assert gone.status not in gone.agent.runtime.observers
    await views.aclose()


@pytest.mark.asyncio
async def test_with_its_parent_gone_too_the_root_is_shown(
    stack: AsyncExitStack,
) -> None:
    team = await _team(stack)
    grand = await _open(stack)
    _ = team.join(grand, label="grand", parent="kid")
    views = Views(
        team["Agent"],
        team=team,
        printer=_Screen().printer,
        catalog=_catalog(),
    )
    views.focus("grand")

    _ = team.leave("grand")
    _ = team.leave("kid")
    views.follow()

    assert views.current.label == "Agent"
    await views.aclose()


@pytest.mark.asyncio
async def test_a_gone_grandchild_hands_the_focus_to_its_parent_not_the_root(
    stack: AsyncExitStack,
) -> None:
    team = await _team(stack)
    _ = team.join(await _open(stack), label="grand", parent="kid")
    views = Views(
        team["Agent"],
        team=team,
        printer=_Screen().printer,
        catalog=_catalog(),
    )
    views.focus("grand")

    _ = team.leave("grand")
    views.follow()

    assert (views.current.label, views.current.agent) == ("kid", team["kid"])
    await views.aclose()


@pytest.mark.asyncio
async def test_a_label_taken_by_a_new_agent_gets_a_new_view(
    stack: AsyncExitStack,
) -> None:
    team = await _team(stack)
    views = Views(
        team["Agent"],
        team=team,
        printer=_Screen().printer,
        catalog=_catalog(),
    )
    views.focus("kid")
    old = views.current
    views.focus("Agent")

    _ = team.leave("kid")
    _ = team.join(await _open(stack), label="kid", parent="Agent")
    views.follow()
    assert old.status not in old.agent.runtime.observers
    views.focus("kid")

    assert views.current is not old
    assert views.current.agent is team["kid"]
    await views.aclose()


@pytest.mark.asyncio
async def test_watch_follows_until_cancelled(stack: AsyncExitStack) -> None:
    team = await _team(stack)
    views = Views(
        team["Agent"],
        team=team,
        printer=_Screen().printer,
        catalog=_catalog(),
    )
    views.focus("kid")
    watching = asyncio.create_task(views.watch(interval_sec=0))

    await team.stop("kid")
    for _ in range(5):
        await asyncio.sleep(0)

    assert views.current.label == "Agent"
    watching.cancel()
    await views.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(("show", "printed"), [(True, True), (False, False)])
async def test_what_happens_while_held_prints_on_release_if_shown(
    stack: AsyncExitStack,
    show: bool,
    printed: bool,
) -> None:
    root = await _open(stack, "said while held")
    screen = _Screen()
    views = Views(root, team=None, printer=screen.printer, catalog=_catalog())

    views.hold()
    root.runtime.push(UserMessage(content="go"))
    await _idle(root)
    assert "said while held" not in screen.text()
    views.release(show=show)

    assert ("said while held" in screen.text()) is printed
    assert views.current.scrollback in root.runtime.observers
    await views.aclose()


@pytest.mark.asyncio
async def test_stop_refuses_the_root_and_a_child_its_spawner_closes(
    stack: AsyncExitStack,
) -> None:
    team = await _team(stack)
    _ = team.join(await _open(stack), label="oneshot", parent="Agent")
    views = Views(
        team["Agent"],
        team=team,
        printer=_Screen().printer,
        catalog=_catalog(),
    )

    assert views.stop("Agent") == "Agent ends with /quit"
    assert views.stop("oneshot") == "oneshot is not a persistent agent"
    assert views.stop("kid") == "stopping kid"
    await views.aclose()

    assert "kid" not in team


@pytest.mark.asyncio
async def test_a_lone_agent_cannot_be_stopped(stack: AsyncExitStack) -> None:
    root = await _open(stack)
    views = Views(root, team=None, printer=_Screen().printer, catalog=_catalog())

    assert views.stop("") == " ends with /quit"
    await views.aclose()


@pytest.mark.asyncio
async def test_a_stop_that_fails_prints_why(
    stack: AsyncExitStack,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    team = await _team(stack)
    screen = _Screen()
    views = Views(team["Agent"], team=team, printer=screen.printer, catalog=_catalog())

    async def fail(label: str) -> None:
        raise RuntimeError(f"cannot stop {label}")

    monkeypatch.setattr(team, "stop", fail)
    _ = views.stop("kid")
    await views.aclose()

    assert "RuntimeError: cannot stop kid" in screen.text()


@pytest.mark.asyncio
async def test_cancel_ends_a_job_or_says_it_already_ended(
    stack: AsyncExitStack,
) -> None:
    wait = _Wait()
    provider = ScriptedProvider(
        replies=[reply(calls=[call("Wait", "w1")]), reply("ok")],
    )
    root = await stack.enter_async_context(
        Agent(model=provider["scripted"], tools=[wait]),
    )
    views = Views(root, team=None, printer=_Screen().printer, catalog=_catalog())
    root.runtime.push(UserMessage(content="go"))
    await asyncio.wait_for(wait.started.wait(), timeout=5)

    assert views.cancel("", "w1") == "cancelled w1"
    await _idle(root)
    assert views.cancel("", "w1") == "w1 already ended"
    await views.aclose()


@pytest.mark.asyncio
async def test_each_agent_has_its_own_screen_and_the_shown_one_rules(
    stack: AsyncExitStack,
) -> None:
    team = await _team(stack)
    screen = _Screen()
    views = Views(team["Agent"], team=team, printer=screen.printer, catalog=_catalog())

    seen: list[bool] = []
    await views.current.commands("/send kid /screen.thinking hide")
    seen.append(screen.printer.show_thinking)
    views.focus("kid")
    seen.append(screen.printer.show_thinking)
    await views.current.commands("/screen.thinking show")
    seen.append(screen.printer.show_thinking)
    views.focus("Agent")
    await views.current.commands("/send kid /screen.thinking hide")
    seen.append(screen.printer.show_thinking)

    assert seen == [True, False, True, True]
    assert views.screen("kid").show_thinking is False
    await views.aclose()


@pytest.mark.asyncio
async def test_an_agent_that_joins_takes_up_what_its_ancestors_chose(
    stack: AsyncExitStack,
) -> None:
    team = await _team(stack)
    views = Views(
        team["Agent"],
        team=team,
        printer=_Screen().printer,
        catalog=_catalog(),
    )
    await views.current.commands("/screen.thinking hide")
    _ = team.join(await _open(stack), label="late", parent="kid")

    views.follow()
    await views.aclose()

    assert views.screen("late").show_thinking is False
    assert views.screen("kid").show_thinking is False


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
