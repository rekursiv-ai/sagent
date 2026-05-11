"""Unit tests for ``Inbox``."""

from __future__ import annotations

import asyncio

import pytest

from sagent.agent.inbox import USER_SOURCE, Inbox, InboxItem
from sagent.custom_types import TextMessage


def _msg(text: str) -> TextMessage:
    return TextMessage(text, "text/x-user-message")


pytestmark = [pytest.mark.anyio, pytest.mark.real_sleep]


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class TestSendDrain:
    async def test_send_then_drain_returns_in_order(self) -> None:
        inbox = Inbox()
        inbox.send(_msg("first"), source=USER_SOURCE)
        inbox.send(_msg("second"), source="Agent_0")
        items = await inbox.drain()
        assert [i.msg.content for i in items] == ["first", "second"]
        assert [i.source for i in items] == [USER_SOURCE, "Agent_0"]

    async def test_drain_clears_buffer(self) -> None:
        inbox = Inbox()
        inbox.send(_msg("once"), source=USER_SOURCE)
        _ = await inbox.drain()
        assert len(inbox) == 0

    async def test_drain_blocks_until_send(self) -> None:
        inbox = Inbox()
        drained: list[InboxItem] = []

        async def reader() -> None:
            drained.extend(await inbox.drain())

        task = asyncio.create_task(reader())
        await asyncio.sleep(0)
        assert not task.done()
        inbox.send(_msg("woke up"), source=USER_SOURCE)
        await asyncio.wait_for(task, timeout=1.0)
        assert [i.msg.content for i in drained] == ["woke up"]


class TestDrainNowait:
    async def test_drain_nowait_empty_returns_empty(self) -> None:
        inbox = Inbox()
        assert inbox.drain_nowait() == []

    async def test_drain_nowait_after_send_returns_items(self) -> None:
        inbox = Inbox()
        inbox.send(_msg("hi"), source=USER_SOURCE)
        inbox.send(_msg("there"), source="bg_xyz")
        items = inbox.drain_nowait()
        assert [i.msg.content for i in items] == ["hi", "there"]
        assert len(inbox) == 0


class TestRequeueFront:
    async def test_requeue_front_preserves_order(self) -> None:
        inbox = Inbox()
        inbox.send(_msg("a"), source=USER_SOURCE)
        inbox.send(_msg("b"), source="Agent_0")
        first_batch = await inbox.drain()
        inbox.send(_msg("c"), source="bg_xx")
        inbox.requeue_front(first_batch)
        items = await inbox.drain()
        assert [i.msg.content for i in items] == ["a", "b", "c"]

    async def test_requeue_front_wakes_drain(self) -> None:
        inbox = Inbox()
        items_holder: list[InboxItem] = []

        async def reader() -> None:
            items_holder.extend(await inbox.drain())

        task = asyncio.create_task(reader())
        await asyncio.sleep(0)
        inbox.requeue_front([InboxItem(source=USER_SOURCE, msg=_msg("requeued"))])
        await asyncio.wait_for(task, timeout=1.0)
        assert [i.msg.content for i in items_holder] == ["requeued"]

    async def test_requeue_front_empty_is_noop(self) -> None:
        inbox = Inbox()
        inbox.requeue_front([])
        assert len(inbox) == 0


class TestBlockUntilUser:
    async def test_non_user_arrival_does_not_unblock(self) -> None:
        inbox = Inbox()
        inbox.block_until_user()
        items_holder: list[InboxItem] = []

        async def reader() -> None:
            items_holder.extend(await inbox.drain())

        task = asyncio.create_task(reader())
        await asyncio.sleep(0)
        inbox.send(_msg("peer ping"), source="Agent_0")
        await asyncio.sleep(0)
        assert not task.done()
        inbox.send(_msg("bg done"), source="bg_xx")
        await asyncio.sleep(0)
        assert not task.done()
        _ = task.cancel()  # cleanup
        with pytest.raises(asyncio.CancelledError):
            await task

    async def test_user_arrival_unblocks_and_returns_accumulated(self) -> None:
        inbox = Inbox()
        inbox.block_until_user()
        inbox.send(_msg("peer"), source="Agent_0")
        inbox.send(_msg("bg"), source="bg_xx")
        items_holder: list[InboxItem] = []

        async def reader() -> None:
            items_holder.extend(await inbox.drain())

        task = asyncio.create_task(reader())
        await asyncio.sleep(0)
        assert not task.done()
        inbox.send(_msg("user"), source=USER_SOURCE)
        await asyncio.wait_for(task, timeout=1.0)
        assert [i.source for i in items_holder] == ["Agent_0", "bg_xx", USER_SOURCE]

    async def test_block_cleared_after_unblocking_drain(self) -> None:
        inbox = Inbox()
        inbox.block_until_user()
        inbox.send(_msg("u"), source=USER_SOURCE)
        _ = await inbox.drain()
        # Now drain again with a non-user-only item; should NOT block.
        inbox.send(_msg("peer"), source="Agent_0")
        items = await asyncio.wait_for(inbox.drain(), timeout=1.0)
        assert [i.source for i in items] == ["Agent_0"]


class TestPeekPop:
    async def test_peek_by_source_finds_user(self) -> None:
        inbox = Inbox()
        inbox.send(_msg("peer"), source="Agent_0")
        inbox.send(_msg("user_msg"), source=USER_SOURCE)
        item = inbox.peek_by_source(USER_SOURCE)
        assert item is not None
        assert item.msg.content == "user_msg"
        # Peek does not remove.
        assert len(inbox) == 2

    async def test_peek_by_source_missing_returns_none(self) -> None:
        inbox = Inbox()
        inbox.send(_msg("peer"), source="Agent_0")
        assert inbox.peek_by_source(USER_SOURCE) is None

    async def test_pop_by_source_removes_last_match(self) -> None:
        inbox = Inbox()
        inbox.send(_msg("peer"), source="Agent_0")
        inbox.send(_msg("u1"), source=USER_SOURCE)
        inbox.send(_msg("u2"), source=USER_SOURCE)
        popped = inbox.pop_by_source(USER_SOURCE)
        assert popped is not None
        assert popped.msg.content == "u2"
        # Only the most recent match is removed.
        remaining = await inbox.drain()
        assert [i.msg.content for i in remaining] == ["peer", "u1"]

    async def test_peek_by_source_returns_most_recent(self) -> None:
        inbox = Inbox()
        inbox.send(_msg("u1"), source=USER_SOURCE)
        inbox.send(_msg("peer"), source="Agent_0")
        inbox.send(_msg("u2"), source=USER_SOURCE)
        item = inbox.peek_by_source(USER_SOURCE)
        assert item is not None
        assert item.msg.content == "u2"


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
