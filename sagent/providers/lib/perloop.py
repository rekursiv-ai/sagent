"""Resources that cannot be shared across event loops.

An ``asyncio.Lock`` binds to the loop that first *contends* on it, and an
``httpx.AsyncClient`` holds a connection pool belonging to the loop that
opened it. A provider built once and used from two loops therefore fails:
a second waiter raises ``RuntimeError: ... is bound to a different event
loop``, and a third blocks forever on a future that will never resolve on
its own loop.

Binding is lazy, which is why this class of defect survives a test suite.
``Lock.acquire`` returns before touching ``_get_loop`` when the lock is
free (CPython ``asyncio/locks.py``), so a single uncontended waiter passes
against broken code. Any regression test for this must force **two**
waiters on the same loop.

Keyed weakly so a value dies with its loop: a long-lived process opens a
fresh loop per ``asyncio.run``, and a strong key would grow forever. Weak
keys are necessary but not sufficient -- a contended lock stores its loop,
so the value pins its own key -- hence the closed-loop eviction on access.

Isolation is per loop, so this is the wrong tool for a resource that must
be exclusive process-wide. Loop-scoping such a lock does not fix it, it
silently deletes the exclusion.
"""

from __future__ import annotations

from collections.abc import Callable

import asyncio
import threading
import weakref


__all__ = ["PerLoop"]


class PerLoop[T]:
    """One value per event loop, built on first use.

    Example::

        self._lock = PerLoop(asyncio.Lock)
        ...
        async with self._lock.get():
            ...
    """

    def __init__(self, build: Callable[[], T]) -> None:
        """Store the factory; nothing is built until a loop asks.

        Args:
          build: Called once per loop to construct that loop's value.

        """
        self._build = build
        self._values: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, T] = (
            weakref.WeakKeyDictionary()
        )
        # Guards the map itself, not the values: two threads first-touching
        # different loops would otherwise race on insert. A threading lock
        # because the contention is between threads, not coroutines.
        self._guard = threading.Lock()

    def _prune_locked(self) -> None:
        """Drop entries for closed loops. Caller holds ``_guard``.

        Weak keys alone do not bound growth: a contended ``asyncio.Lock``
        stores ``lock._loop``, so the strongly-held value keeps its own
        weak key alive and the entry outlives the loop forever. Closure
        is the observable end of a loop's life, so evict on it.
        """
        for loop in [x for x in self._values if x.is_closed()]:
            _ = self._values.pop(loop, None)

    def get(self) -> T:
        """Return the running loop's value, building it on first use.

        Returns:
          value: This loop's own value.

        Raises:
          RuntimeError: If called outside a running event loop, where
            there is no correct answer to give.

        """
        loop = asyncio.get_running_loop()
        with self._guard:
            self._prune_locked()
            # ``in`` rather than a ``None`` probe: ``None`` is a legal ``T``
            # (a cached "no client yet"), so it cannot double as the miss
            # sentinel without rerunning the factory on every access.
            if loop not in self._values:
                self._values[loop] = self._build()
            return self._values[loop]

    def set(self, value: T) -> None:
        """Replace the running loop's value.

        Args:
          value: The value to store for this loop.

        Raises:
          RuntimeError: If called outside a running event loop, where
            there is no correct answer to give.

        """
        loop = asyncio.get_running_loop()
        with self._guard:
            self._prune_locked()
            self._values[loop] = value

    def peek(self) -> T | None:
        """Return the running loop's value without building one.

        Returns:
          value: The stored value, or ``None`` if this loop has none.

        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return None
        with self._guard:
            return self._values.get(loop)

    def clear(self) -> None:
        """Drop the running loop's value, if any."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        with self._guard:
            self._values.pop(loop, None)

    def size(self) -> int:
        """Return how many loops currently hold a value.

        Returns:
          count: Live entries.

        """
        with self._guard:
            self._prune_locked()
            return len(self._values)
