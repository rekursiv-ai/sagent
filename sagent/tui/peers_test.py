"""Peer messaging has one live choice across flags, slash commands and AgentSelf."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from sagent.agent.agent import Agent
from sagent.agent.team import Team
from sagent.model.scripted import ScriptedProvider
from sagent.providers.catalog import ModelCatalog
from sagent.settings.flags import flags
from sagent.settings.harvest import harvest
from sagent.tools.agent_self import AgentSelf
from sagent.tools.types import ToolContext, settled
from sagent.tui.main import _configure, _Screen
from sagent.tui.prompt import Panes
from sagent.tui.slash import Commands


if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.asyncio
async def test_flag_slash_and_conversation_share_the_inherited_choice(
    tmp_path: Path,
) -> None:
    catalog = ModelCatalog(factories={})
    team = Team(catalog=catalog)
    agent = Agent(
        model=ScriptedProvider()["scripted"],
        tool_context=ToolContext(cwd=tmp_path),
    )
    child = Agent(
        model=ScriptedProvider()["scripted"],
        tool_context=ToolContext(cwd=tmp_path),
    )
    _ = team.join(agent)
    _ = team.join(child, label="child", parent="Agent")
    commands = Commands(
        agent,
        catalog=catalog,
        team=team,
        say=lambda _: None,
        screen=_Screen(),
        panes=Panes(),
    )
    rest, given = flags(["--peer-messaging", "off"], static=frozenset())
    assert rest == []
    await _configure(harvest(commands.owners()), catalog=catalog, given=given)
    assert not team.peer_messaging("Agent")
    assert not team.peer_messaging("child")
    _ = await commands("/peer-messaging on")
    assert team.peer_messaging("Agent")
    assert team.peer_messaging("child")
    _ = await settled(
        AgentSelf(team=team).call(
            child.runtime.tool_context,
            {"peer_messaging": False},
        ),
    )
    _ = await commands("/peer-messaging off")
    _ = await commands("/peer-messaging on")
    assert team.peer_messaging("Agent")
    assert not team.peer_messaging("child")
    _ = await commands("/child.peer-messaging default")
    assert team.peer_messaging("child")
    _ = await commands("/peer-messaging invalid")
    assert team.peer_messaging("Agent")
    await commands.aclose()


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
