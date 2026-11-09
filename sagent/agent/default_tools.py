"""Build Sagent's built-in tools for a frontend-owned agent.

The terminal and editor surfaces need the same tool instances and the same team
wiring. Keeping that construction here prevents a frontend from quietly growing
a second definition of what a coding agent can do.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sagent.session.projects import project_dir


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from wesearch.fetch import FetchSession

    from sagent.agent.team import Team
    from sagent.tools.types import Tool


__all__ = [
    "build_tools",
    "built_in_tool_builders",
    "preload_builtin_tools",
    "register_builtin_tool_classes",
]


_BUILT_IN_TOOL_CLASSES: list[Any] = []


def built_in_tool_builders(
    *,
    cwd: Path,
    team: Team,
    session: FetchSession,
    skill_directories: Sequence[Path] = (),
) -> dict[str, Callable[[], Tool]]:
    """Return constructors for every built-in tool in Sagent's normal order.

    Shared objects remain shared exactly as they are in the terminal surface:
    Bash sees the same file tools, and web tools use the same fetch session.
    """
    (
        AgentSelf,
        AgentSend,
        AgentSpawn,
        BackgroundTask,
        Bash,
        Edit,
        Glob,
        Grep,
        List,
        Memory,
        PaperAuthor,
        PaperDetails,
        PaperFetch,
        PaperSearch,
        PlayAudio,
        Read,
        Session,
        Skill,
        WebFetch,
        WebSearch,
        Write,
    ) = _tool_classes()
    read, edit, grep, glob, ls = Read(), Edit(), Grep(), Glob(), List()
    fetch = WebFetch(session=session)
    return {
        "Bash": lambda: Bash(peers=(read, edit, grep, glob, ls, fetch)),
        "Read": lambda: read,
        "Write": Write,
        "Edit": lambda: edit,
        "Grep": lambda: grep,
        "Glob": lambda: glob,
        "List": lambda: ls,
        "WebSearch": lambda: WebSearch(session=session),
        "WebFetch": lambda: fetch,
        "PaperSearch": lambda: PaperSearch(session=session),
        "PaperDetails": lambda: PaperDetails(session=session),
        "PaperAuthor": lambda: PaperAuthor(session=session),
        "PaperFetch": lambda: PaperFetch(session=session),
        "Memory": lambda: Memory(directory=_memory_directory(cwd)),
        "PlayAudio": PlayAudio,
        "AgentSpawn": lambda: AgentSpawn(team=team),
        "AgentSend": lambda: AgentSend(team=team),
        "AgentSelf": lambda: AgentSelf(team=team),
        "Session": lambda: Session(team=team),
        "Skill": lambda: Skill(additional_dirs=tuple(skill_directories)),
        "BackgroundTask": BackgroundTask,
    }


def build_tools(
    *,
    cwd: Path,
    team: Team,
    session: FetchSession,
    names: Sequence[str] | None = None,
    skill_directories: Sequence[Path] = (),
) -> list[Tool]:
    """Build the selected built-in tools, or all of them when omitted."""
    builders = built_in_tool_builders(
        cwd=cwd,
        team=team,
        session=session,
        skill_directories=skill_directories,
    )
    selected = list(builders) if names is None else list(names)
    if selected == ["none"]:
        return []
    unknown = sorted(set(selected) - set(builders))
    if unknown:
        raise ValueError(
            f"unknown tools {', '.join(unknown)}; known: {', '.join(builders)}",
        )
    return [builders[name]() for name in selected]


def preload_builtin_tools() -> None:
    """Import built-in tools before an async resource stack is opened."""
    _tool_classes()


def register_builtin_tool_classes(classes: tuple[Any, ...]) -> None:
    """Register classes already imported in Sagent's established order.

    The terminal imports these modules eagerly because of their existing
    Agent/Team cycle. Registering those exact class objects lets the shared
    constructor run without performing imports during async resource cleanup.
    """
    if _BUILT_IN_TOOL_CLASSES and tuple(_BUILT_IN_TOOL_CLASSES) != classes:
        raise RuntimeError("built-in tool classes were registered twice")
    if not _BUILT_IN_TOOL_CLASSES:
        _BUILT_IN_TOOL_CLASSES.extend(classes)


def _tool_classes() -> tuple[Any, ...]:
    """Return the cached tool classes in the CLI's established import order."""
    if _BUILT_IN_TOOL_CLASSES:
        return tuple(_BUILT_IN_TOOL_CLASSES)

    # These imports are intentionally lazy: importing this registry must not
    # trigger the Agent/Team cycle before the caller has established Sagent's
    # normal top-level import order.
    from sagent.tools.agent_self import AgentSelf  # noqa: PLC0415
    from sagent.tools.agent_send import AgentSend  # noqa: PLC0415
    from sagent.tools.agent_spawn import AgentSpawn  # noqa: PLC0415
    from sagent.tools.background_task import BackgroundTask  # noqa: PLC0415
    from sagent.tools.bash import Bash  # noqa: PLC0415
    from sagent.tools.edit import Edit  # noqa: PLC0415
    from sagent.tools.glob_tool import Glob  # noqa: PLC0415
    from sagent.tools.grep import Grep  # noqa: PLC0415
    from sagent.tools.list import List  # noqa: PLC0415
    from sagent.tools.memory import Memory  # noqa: PLC0415
    from sagent.tools.paper_author import PaperAuthor  # noqa: PLC0415
    from sagent.tools.paper_details import PaperDetails  # noqa: PLC0415
    from sagent.tools.paper_fetch import PaperFetch  # noqa: PLC0415
    from sagent.tools.paper_search import PaperSearch  # noqa: PLC0415
    from sagent.tools.play_audio import PlayAudio  # noqa: PLC0415
    from sagent.tools.read import Read  # noqa: PLC0415
    from sagent.tools.session import Session  # noqa: PLC0415
    from sagent.tools.skill import Skill  # noqa: PLC0415
    from sagent.tools.web_fetch import WebFetch  # noqa: PLC0415
    from sagent.tools.web_search import WebSearch  # noqa: PLC0415
    from sagent.tools.write import Write  # noqa: PLC0415

    _BUILT_IN_TOOL_CLASSES.extend(
        (
            AgentSelf,
            AgentSend,
            AgentSpawn,
            BackgroundTask,
            Bash,
            Edit,
            Glob,
            Grep,
            List,
            Memory,
            PaperAuthor,
            PaperDetails,
            PaperFetch,
            PaperSearch,
            PlayAudio,
            Read,
            Session,
            Skill,
            WebFetch,
            WebSearch,
            Write,
        ),
    )
    return tuple(_BUILT_IN_TOOL_CLASSES)


def _memory_directory(cwd: Path) -> Path:
    """Return project memory without importing a terminal-only module."""
    return project_dir(cwd) / "memory"
