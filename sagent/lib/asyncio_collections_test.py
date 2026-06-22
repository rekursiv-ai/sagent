"""Tests for :class:`sagent.lib.asyncio_collections.Deque`."""

from __future__ import annotations

import asyncio

import pytest

from sagent.lib.asyncio_collections import Deque


class TestBounded:
    def test_rejects_non_positive_capacity(self) -> None:
        with pytest.raises(ValueError, match="capacity"):
            Deque[int](0)

    def test_rejects_negative_capacity(self) -> None:
        with pytest.raises(ValueError, match="capacity"):
            Deque[int](-5)

    def test_put_returns_true_until_full(self) -> None:
        q: Deque[int] = Deque(2)
        assert q.put(1) is True
        assert q.put(2) is True
        assert q.put(3) is False
        assert len(q) == 2
        assert q.full() is True

    def test_put_left_returns_true_until_full(self) -> None:
        q: Deque[int] = Deque(2)
        assert q.put_left(1) is True
        assert q.put_left(2) is True
        assert q.put_left(3) is False

    def test_capacity_and_empty(self) -> None:
        q: Deque[str] = Deque(3)
        assert q.capacity == 3
        assert q.empty() is True
        q.put("a")
        assert q.empty() is False
        assert len(q) == 1

    @pytest.mark.asyncio
    async def test_get_fifo(self) -> None:
        q: Deque[str] = Deque(5)
        q.put("a")
        q.put("b")
        q.put("c")
        assert await q.get() == "a"
        assert await q.get() == "b"
        assert await q.get() == "c"
        assert q.empty()

    @pytest.mark.asyncio
    async def test_put_left_jumps_ahead(self) -> None:
        q: Deque[str] = Deque(5)
        q.put("a")
        q.put("b")
        q.put_left("urgent")
        assert await q.get() == "urgent"
        assert await q.get() == "a"
        assert await q.get() == "b"

    @pytest.mark.asyncio
    async def test_interleaved_put_and_put_left(self) -> None:
        q: Deque[int] = Deque(5)
        q.put(1)
        q.put_left(0)
        q.put(2)
        q.put_left(-1)
        assert [await q.get() for _ in range(4)] == [-1, 0, 1, 2]

    @pytest.mark.asyncio
    async def test_get_waits_until_put(self) -> None:
        q: Deque[str] = Deque(1)

        async def producer() -> None:
            await asyncio.sleep(0.01)
            q.put("hi")

        consumer = asyncio.create_task(q.get())
        prod = asyncio.create_task(producer())
        result = await asyncio.wait_for(consumer, timeout=0.5)
        await prod
        assert result == "hi"

    @pytest.mark.asyncio
    async def test_multiple_consumers(self) -> None:
        q: Deque[int] = Deque(5)
        results: list[int] = []

        async def consume() -> None:
            results.append(await q.get())

        c1 = asyncio.create_task(consume())
        c2 = asyncio.create_task(consume())
        await asyncio.sleep(0)
        q.put(1)
        q.put(2)
        await asyncio.wait_for(asyncio.gather(c1, c2), timeout=0.5)
        assert sorted(results) == [1, 2]

    def test_drain(self) -> None:
        q: Deque[int] = Deque(5)
        for i in (1, 2, 3):
            q.put(i)
        assert q.drain() == [1, 2, 3]
        assert q.empty()

    def test_drain_empty(self) -> None:
        q: Deque[int] = Deque(5)
        assert q.drain() == []


class TestUnbounded:
    def test_capacity_is_none(self) -> None:
        q: Deque[int] = Deque()
        assert q.capacity is None
        assert not q.full()

    def test_put_always_succeeds(self) -> None:
        q: Deque[int] = Deque()
        for i in range(1000):
            assert q.put(i) is True
        assert len(q) == 1000

    def test_bool(self) -> None:
        q: Deque[str] = Deque()
        assert not q
        q.put("x")
        assert q

    @pytest.mark.asyncio
    async def test_get_fifo(self) -> None:
        q: Deque[str] = Deque()
        q.put("a")
        q.put("b")
        assert await q.get() == "a"
        assert await q.get() == "b"


class TestTailAccess:
    def test_peek_tail(self) -> None:
        q: Deque[str] = Deque()
        assert q.peek_tail() is None
        q.put("a")
        q.put("b")
        assert q.peek_tail() == "b"
        assert len(q) == 2

    def test_pop_tail(self) -> None:
        q: Deque[str] = Deque()
        assert q.pop_tail() is None
        q.put("a")
        q.put("b")
        assert q.pop_tail() == "b"
        assert len(q) == 1


class TestWaitForEmpty:
    @pytest.mark.asyncio
    async def test_returns_immediately_when_empty(self) -> None:
        q: Deque[int] = Deque()
        await asyncio.wait_for(q.wait_for_empty(), timeout=0.1)

    @pytest.mark.asyncio
    async def test_blocks_until_drained(self) -> None:
        q: Deque[int] = Deque()
        q.put(1)
        q.put(2)
        waiter = asyncio.create_task(q.wait_for_empty())
        await asyncio.sleep(0)
        assert not waiter.done()
        _ = await q.get()
        assert not waiter.done()
        _ = await q.get()
        await asyncio.wait_for(waiter, timeout=0.1)

    @pytest.mark.asyncio
    async def test_blocks_until_drain_call(self) -> None:
        q: Deque[int] = Deque()
        q.put(1)
        q.put(2)
        waiter = asyncio.create_task(q.wait_for_empty())
        await asyncio.sleep(0)
        assert not waiter.done()
        q.drain()
        await asyncio.wait_for(waiter, timeout=0.1)

    @pytest.mark.asyncio
    async def test_pop_tail_to_empty(self) -> None:
        q: Deque[int] = Deque()
        q.put(1)
        waiter = asyncio.create_task(q.wait_for_empty())
        await asyncio.sleep(0)
        assert not waiter.done()
        _ = q.pop_tail()
        await asyncio.wait_for(waiter, timeout=0.1)


class TestGetAll:
    @pytest.mark.asyncio
    async def test_drains_everything(self) -> None:
        q: Deque[str] = Deque()
        q.put("a")
        q.put("b")
        q.put("c")
        assert await q.get_all() == ["a", "b", "c"]
        assert q.empty()

    @pytest.mark.asyncio
    async def test_waits_then_drains(self) -> None:
        q: Deque[str] = Deque()

        async def producer() -> None:
            await asyncio.sleep(0.01)
            q.put("x")
            q.put("y")

        task = asyncio.create_task(q.get_all())
        prod = asyncio.create_task(producer())
        result = await asyncio.wait_for(task, timeout=0.5)
        await prod
        assert result == ["x", "y"]


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
