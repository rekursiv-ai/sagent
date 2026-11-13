"""The built-in tools, by name: what a session offers, and what a tool server serves.

Typical usage example:

  built = builders(cwd=Path.cwd(), team=team, remote=remote)
  tools = [built[name]() for name in ("Bash", "Read")]
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from sagent.agent.agents_md import TIERS
from sagent.host.wire import ToolSpec
from sagent.session.projects import (
    merge_legacy_memories,
    project_dir,
    repo_root,
)
from sagent.tools.agent_self import AgentSelf
from sagent.tools.agent_send import AgentSend
from sagent.tools.agent_session import AgentSession
from sagent.tools.agent_spawn import AgentSpawn
from sagent.tools.background_task import BackgroundTask
from sagent.tools.bash import Bash
from sagent.tools.edit import Edit
from sagent.tools.glob_tool import Glob
from sagent.tools.grep import Grep
from sagent.tools.list import List
from sagent.tools.memory import Memory
from sagent.tools.paper_author import PaperAuthor
from sagent.tools.paper_details import PaperDetails
from sagent.tools.paper_fetch import PaperFetch
from sagent.tools.paper_search import PaperSearch
from sagent.tools.play_audio import PlayAudio
from sagent.tools.read import Read
from sagent.tools.remote import RemoteTool
from sagent.tools.skill import Skill
from sagent.tools.web_browser import WebBrowser
from sagent.tools.web_fetch import WebFetch
from sagent.tools.web_search import WebSearch
from sagent.tools.write import Write


if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from sagent.agent.agents_md import Tier
    from sagent.agent.team import Team
    from sagent.tools.remote import RemoteSession
    from sagent.tools.types import Tool


__all__ = [
    "TOOLS",
    "builders",
]


TOOLS: Final = (
    "Bash",
    "Read",
    "Write",
    "Edit",
    "Grep",
    "Glob",
    "List",
    "WebSearch",
    "WebFetch",
    "WebBrowser",
    "PaperSearch",
    "PaperDetails",
    "PaperAuthor",
    "PaperFetch",
    "Memory",
    "PlayAudio",
    "AgentSpawn",
    "AgentSend",
    "AgentSelf",
    "AgentSession",
    "Skill",
    "BackgroundTask",
    "RemoteTool",
)
"""Each built-in tool's name, in their default order: the keys of ``builders``."""


def builders(
    *,
    cwd: Path,
    team: Team,
    remote: RemoteSession,
    tiers: frozenset[Tier] = TIERS,
) -> dict[str, Callable[[], Tool]]:
    """Return how to build each built-in tool, by its name, in their default order.

    Args:
      cwd: The working directory; its repository's project keeps the memories.
      team: The agents the agent tools reach.
      remote: The servers ``RemoteTool`` reaches.
      tiers: The tiers ``Skill`` finds skills in, as the agent reads AGENTS.md.

    Returns:
      builders: One per tool; Read, Edit, Grep, Glob, List and WebFetch are
        shared, so Bash hands a command to the same tool the model would call.

    """
    read, edit, grep, glob, ls = Read(), Edit(), Grep(), Glob(), List()
    fetch = WebFetch()
    local: dict[str, Callable[[], Tool]] = {
        "Bash": lambda: Bash(peers=(read, edit, grep, glob, ls, fetch)),
        "Read": lambda: read,
        "Write": Write,
        "Edit": lambda: edit,
        "Grep": lambda: grep,
        "Glob": lambda: glob,
        "List": lambda: ls,
        "WebSearch": WebSearch,
        "WebFetch": lambda: fetch,
        "WebBrowser": WebBrowser,
        "PaperSearch": PaperSearch,
        "PaperDetails": PaperDetails,
        "PaperAuthor": PaperAuthor,
        "PaperFetch": PaperFetch,
        "Memory": lambda: Memory(directory=_memories(cwd)),
        "PlayAudio": PlayAudio,
        "AgentSpawn": lambda: AgentSpawn(team=team),
        "AgentSend": lambda: AgentSend(team=team),
        "AgentSelf": lambda: AgentSelf(team=team),
        "AgentSession": lambda: AgentSession(team=team),
        "Skill": lambda: Skill(tiers=tiers),
        "BackgroundTask": BackgroundTask,
    }
    return {
        **local,
        "RemoteTool": lambda: RemoteTool(
            remote=remote,
            local={name: ToolSpec.from_tool(build()) for name, build in local.items()},
        ),
    }


def _memories(cwd: Path) -> Path:
    """Return the memory directory of ``cwd``'s repository, else of ``cwd``."""
    repo = repo_root(cwd)
    if repo is None:
        return project_dir(cwd) / "memory"
    merge_legacy_memories(repo)
    return project_dir(repo) / "memory"
