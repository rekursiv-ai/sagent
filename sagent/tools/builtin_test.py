"""Tests for the built-in tools by name."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sagent.agent.team import Team
from sagent.host.wire import ToolSpec
from sagent.model.scripted import ScriptedProvider
from sagent.providers.catalog import ModelCatalog
from sagent.tools.bash import Bash
from sagent.tools.builtin import TOOLS, builders
from sagent.tools.memory import Memory
from sagent.tools.remote import RemoteSession, RemoteTool


if TYPE_CHECKING:
    from pathlib import Path


def _built(tmp_path: Path) -> dict[str, object]:
    team = Team(catalog=ModelCatalog(factories={"scripted": ScriptedProvider}))
    return {
        name: build()
        for name, build in builders(
            cwd=tmp_path,
            team=team,
            remote=RemoteSession([]),
        ).items()
    }


def test_each_builder_makes_the_tool_of_its_name(tmp_path: Path) -> None:
    built = _built(tmp_path)
    assert all(getattr(tool, "name", None) == name for name, tool in built.items())
    assert tuple(built) == TOOLS


def test_bash_shares_its_peers_and_memory_is_the_project_s(tmp_path: Path) -> None:
    built = _built(tmp_path)
    bash, memory = built["Bash"], built["Memory"]
    assert isinstance(bash, Bash)
    assert isinstance(memory, Memory)
    assert all(peer is built[peer.name] for peer in bash.peers)
    assert memory.directory.name == "memory"


def test_remote_tool_knows_each_local_tool_but_itself(tmp_path: Path) -> None:
    remote = _built(tmp_path)["RemoteTool"]
    assert isinstance(remote, RemoteTool)
    assert "RemoteTool" not in remote.local
    assert remote.local["Bash"] == ToolSpec.from_tool(Bash())


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
