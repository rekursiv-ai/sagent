"""Async collection primitives for one-producer / one-consumer-per-queue patterns.

Exposes :class:`Deque`: an optionally-bounded async deque with both-ends
append, async pop-from-front, tail access, and bulk drain.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator

import asyncio


__all__ = ["Deque"]


class Deque[T]:
    """Optionally-bounded async deque with both-ends access.

    Backed by :class:`collections.deque` plus a single
    :class:`asyncio.Event` for not-empty signaling.

    ``capacity=None`` (default) means unbounded; ``put`` always succeeds.
    With a finite capacity, ``put`` / ``put_left`` return ``False``
    when full so the caller can drop instead of block.
    """

    __slots__ = ("_capacity", "_dq", "_empty", "_not_empty")

    def __init__(self, capacity: int | None = None) -> None:
        if capacity is not None and capacity <= 0:
            raise ValueError(f"capacity must be > 0 or None, got {capacity!r}")
        self._dq: deque[T] = deque()
        self._capacity = capacity
        self._not_empty = asyncio.Event()
        self._empty = asyncio.Event()
        self._empty.set()

    @property
    def capacity(self) -> int | None:
        return self._capacity

    def __len__(self) -> int:
        return len(self._dq)

    def __bool__(self) -> bool:
        return bool(self._dq)

    def __iter__(self) -> Iterator[T]:
        """Iterate items front-to-back without removing them.

        Snapshots the current contents into a tuple so concurrent
        ``put`` / ``get`` during iteration don't corrupt traversal.
        """
        return iter(tuple(self._dq))

    def empty(self) -> bool:
        return not self._dq

    def full(self) -> bool:
        return self._capacity is not None and len(self._dq) >= self._capacity

    def put(self, item: T) -> bool:
        """Append at the right (back). Returns ``False`` if at capacity."""
        if self._capacity is not None and len(self._dq) >= self._capacity:
            return False
        self._dq.append(item)
        self._not_empty.set()
        self._empty.clear()
        return True

    def put_left(self, item: T) -> bool:
        """Append at the left (front) for urgent / halt-style delivery."""
        if self._capacity is not None and len(self._dq) >= self._capacity:
            return False
        self._dq.appendleft(item)
        self._not_empty.set()
        self._empty.clear()
        return True

    def peek_tail(self) -> T | None:
        """Return the most recently appended item without removing it."""
        return self._dq[-1] if self._dq else None

    def pop_tail(self) -> T | None:
        """Remove and return the most recently appended item."""
        item = self._dq.pop() if self._dq else None
        if not self._dq:
            self._empty.set()
        return item

    async def get(self) -> T:
        """Pop from the left (front), awaiting until an item is available."""
        while not self._dq:
            self._not_empty.clear()
            await self._not_empty.wait()
        item = self._dq.popleft()
        if not self._dq:
            self._empty.set()
        return item

    async def get_all(self) -> list[T]:
        """Wait for at least one item, then drain everything available."""
        while not self._dq:
            self._not_empty.clear()
            await self._not_empty.wait()
        return self.drain()

    def drain(self) -> list[T]:
        """Pop all items in FIFO order. Non-blocking."""
        out = list(self._dq)
        self._dq.clear()
        self._empty.set()
        return out

    async def wait_for_empty(self) -> None:
        """Block until the deque is empty.

        Returns immediately when ``len(self) == 0``. Used by the
        agent dispatch loop's idle barrier alongside an in-flight
        task gather.

        Re-checks ``_dq`` after each ``Event.wait()`` because
        ``asyncio.Event`` resolves outstanding waiters on ``set()``
        and does not retract them on a subsequent ``clear()`` -- a
        post-set ``clear()`` therefore lets a waiter return
        spuriously while items are once again present.
        """
        while self._dq:
            self._empty.clear()
            await self._empty.wait()
