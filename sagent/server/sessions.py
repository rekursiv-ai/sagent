"""Lifecycle registry for the native sessions hosted by the daemon."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
import asyncio

from sagent.server.commands import SagentCommandService
from sagent.session.host import AgentSession


@dataclass
class ManagedSession:
    """Daemon-local state associated with one native ``AgentSession``."""

    session_id: str
    session: AgentSession
    command_service: SagentCommandService | None = None
    active_agent_label: str | None = None
    notification_agent_label: str | None = None
    event_sequence: int = 0
    subscribed_agent_labels: set[str] = field(default_factory=set)
    unsubscribers: list[Callable[[], None]] = field(default_factory=list)
    team_watch_task: asyncio.Task[None] | None = None


class SessionManager:
    """Own independently journaled :class:`AgentSession` objects.

    The daemon is one process, but each chat gets its own native Sagent
    session.  This class deliberately owns only the registry and lifecycle;
    protocol routing and event serialization remain in ``server.py``.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, ManagedSession] = {}

    def add(self, entry: ManagedSession) -> None:
        session_id = entry.session_id
        if session_id in self._sessions:
            raise ValueError(f"session {session_id!r} is already open")
        self._sessions[session_id] = entry

    def get(self, session_id: str) -> ManagedSession | None:
        return self._sessions.get(session_id)

    def __len__(self) -> int:
        return len(self._sessions)

    def ids(self) -> tuple[str, ...]:
        return tuple(self._sessions)

    def values(self) -> Iterator[ManagedSession]:
        return iter(self._sessions.values())

    def remove(self, session_id: str) -> ManagedSession | None:
        return self._sessions.pop(session_id, None)

    async def close_all(self) -> None:
        sessions = tuple(self._sessions.values())
        self._sessions.clear()
        for entry in sessions:
            await entry.session.close()
