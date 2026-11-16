"""Lifecycle registry for the native sessions hosted by the daemon."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
import asyncio

from sagent.server.commands import SagentCommandService
from sagent.session.host import AgentSession


if TYPE_CHECKING:
    from sagent.runtime import types


@dataclass
class ManagedSession:
    """Daemon-local state associated with one native ``AgentSession``."""

    session_id: str
    session: AgentSession
    command_service: SagentCommandService | None = None
    active_agent_label: str | None = None
    notification_agent_label: str | None = None
    event_sequence: int = 0
    event_observer: Callable[[str, types.RuntimeEvent], None] | None = None
    team_watch_task: asyncio.Task[None] | None = None
    background_tasks: set[asyncio.Task[None]] = field(default_factory=set)
    _close_lock: asyncio.Lock = field(
        default_factory=asyncio.Lock,
        init=False,
        repr=False,
    )
    _closed: bool = field(default=False, init=False, repr=False)

    def track(self, task: asyncio.Task[None]) -> None:
        """Keep daemon work scoped to this session until it finishes."""
        self.background_tasks.add(task)
        task.add_done_callback(self.background_tasks.discard)

    async def close(self) -> None:
        """Stop daemon work, then delegate native cleanup to ``AgentSession``."""
        async with self._close_lock:
            if self._closed:
                return

            watcher, self.team_watch_task = self.team_watch_task, None
            if watcher is not None:
                watcher.cancel()
                await asyncio.gather(watcher, return_exceptions=True)

            tasks = tuple(self.background_tasks)
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            self.background_tasks.clear()

            # The controller already tracks every observer registered through
            # AgentSession.subscribe(). AgentSession.close() removes those
            # observers and closes the native resource stack in dependency order.
            await self.session.close()
            self.event_observer = None
            self._closed = True


class SessionManager:
    """Index independently journaled :class:`AgentSession` objects.

    The daemon is one process, but each chat gets its own native Sagent
    session. This class deliberately owns only the registry; each
    ``ManagedSession`` owns its lifecycle, while routing and serialization
    remain in ``server.py``.
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

    def require(self, session_id: str) -> ManagedSession:
        try:
            return self._sessions[session_id]
        except KeyError:
            raise ValueError(f"unknown session {session_id!r}") from None

    def __len__(self) -> int:
        return len(self._sessions)

    def ids(self) -> tuple[str, ...]:
        return tuple(self._sessions)

    def values(self) -> Iterator[ManagedSession]:
        return iter(self._sessions.values())

    def pop(self, session_id: str) -> ManagedSession | None:
        return self._sessions.pop(session_id, None)

    def drain(self) -> tuple[ManagedSession, ...]:
        """Remove and return every entry; each entry remains its own owner."""
        sessions = tuple(self._sessions.values())
        self._sessions.clear()
        return sessions
