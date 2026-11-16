"""Tests for daemon-local session ownership."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import asyncio

import pytest

from sagent.server.sessions import ManagedSession, SessionManager


@pytest.mark.asyncio
async def test_managed_session_closes_daemon_work_before_native_session() -> None:
    session = MagicMock()
    session.close = AsyncMock()
    watcher = asyncio.create_task(asyncio.Event().wait())
    background = asyncio.create_task(asyncio.Event().wait())
    entry = ManagedSession(
        session_id="one",
        session=session,
        team_watch_task=watcher,
    )
    entry.track(background)

    await asyncio.gather(entry.close(), entry.close())

    assert watcher.cancelled()
    assert background.cancelled()
    session.close.assert_awaited_once_with()


def test_session_manager_drain_only_manages_the_registry() -> None:
    manager = SessionManager()
    first = ManagedSession(session_id="one", session=MagicMock())
    second = ManagedSession(session_id="two", session=MagicMock())
    manager.add(first)
    manager.add(second)

    assert manager.require("one") is first
    assert manager.drain() == (first, second)
    assert manager.ids() == ()

    with pytest.raises(ValueError, match="unknown session 'one'"):
        manager.require("one")
