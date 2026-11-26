"""Tests for where Recall searches as settings, changed where the agent reads it."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from sagent.agent.team import Team
from sagent.providers.catalog import ModelCatalog
from sagent.settings.harvest import harvest
from sagent.tools.recall import Recall
from sagent.tui.settings.recall import RecallSettings


if TYPE_CHECKING:
    from sagent.settings.types import Setting
    from sagent.tools.types import Tool


def _recall(*, url: str = "", session: str = "") -> Recall:
    team = Team(catalog=ModelCatalog(factories={}))
    return Recall(team=team, url=url, session=session, key="k")


def _settings(tools: dict[str, Tool]) -> dict[str, Setting]:
    owner = RecallSettings(tools=tools)
    return {".".join(s.name): s for s in harvest([(("tool", "Recall"), owner)])}


def test_recall_exports_its_server_and_its_session() -> None:
    settings = _settings({"Recall": _recall(url="https://trax.example", session="s9")})
    assert [
        (name, s.help_short, s.usage, s.value() if s.value else None, s.options())
        for name, s in settings.items()
    ] == [
        (
            "tool.Recall.url",
            "the trackinizer server Recall asks; empty reads $TRACKINIZER_URL",
            "URL",
            "https://trax.example",
            ("default",),
        ),
        (
            "tool.Recall.session",
            "the stored session Recall searches near; empty reads $TRAX_SESSION",
            "ID",
            "s9",
            ("default",),
        ),
    ]
    assert all(s.cli for s in settings.values())


@pytest.mark.asyncio
async def test_a_change_replaces_recall_where_the_agent_reads_it() -> None:
    before = _recall()
    tools: dict[str, Tool] = {"Recall": before}
    settings = _settings(tools)

    assert await settings["tool.Recall.session"].apply("s1") == "s1"
    after = tools["Recall"]
    assert isinstance(after, Recall)
    assert (after.url, after.session, after.key) == ("", "s1", "k")

    assert await settings["tool.Recall.url"].apply("https://a.example") == (
        "https://a.example"
    )
    after = tools["Recall"]
    assert isinstance(after, Recall)
    # A key is for the server it was given with: another server is sent none.
    assert (after.url, after.session, after.key) == ("https://a.example", "s1", "")
    assert after.team is before.team
    assert (before.url, before.session, before.key) == ("", "", "k")


@pytest.mark.asyncio
async def test_default_reads_the_environment_again() -> None:
    tools: dict[str, Tool] = {"Recall": _recall(url="https://a.example", session="s1")}
    settings = _settings(tools)

    assert await settings["tool.Recall.url"].apply("default") == ""
    after = tools["Recall"]
    assert isinstance(after, Recall)
    # The environment's server takes the environment's key, not the one held.
    assert (after.url, after.session, after.key) == ("", "s1", "")

    assert await settings["tool.Recall.session"].apply("default") == ""
    after = tools["Recall"]
    assert isinstance(after, Recall)
    assert (after.url, after.session) == ("", "")


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
