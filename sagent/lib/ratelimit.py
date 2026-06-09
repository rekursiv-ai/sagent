"""Rate limiters: sliding-window and token-bucket.

Each limiter blocks the caller until a request slot is available, via either
:meth:`acquire` (sync, sleeps the thread) or :meth:`acquire_async` (sleeps
the coroutine). Both paths share one pure reserve step that updates limiter
state under the store's lock and returns how long to wait; the wait then
happens *outside* the lock, so a slow caller never blocks the state update
for others. The two paths differ only in which sleep they call.

Scope of coordination depends on the limiter and its store. The default
backing coordinates callers within one process (threads and coroutines via
a ``threading.Lock``). :class:`TokenBucketRateLimiter` can instead take a
:class:`FileStore`, which holds its state in an ``fcntl``-locked file, so
several processes -- e.g. ones sharing an API key -- pace against one budget.

Choosing between them:
  - :class:`SlidingWindowRateLimiter` -- exact "no more than ``max_calls``
    in any ``per_seconds`` window". Tracks individual call timestamps;
    memory grows with ``max_calls``. In-process only (no cross-process
    store). Use when the provider enforces a true rolling window.
  - :class:`TokenBucketRateLimiter` -- smooth average rate with burst
    capacity ``max_calls``; O(1) state. The only limiter with a pluggable
    :class:`Store`, so the only one that shares a budget across processes.
    Use when you want steady pacing and an occasional burst is acceptable.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Protocol, cast, runtime_checkable

import asyncio
import fcntl
import os
import struct
import threading
import time


@runtime_checkable
class RateLimiter(Protocol):
    """Blocks the calling thread until a request slot is available."""

    def acquire(self) -> None:
        """Block the thread until the caller may proceed with one request."""
        ...


@runtime_checkable
class AsyncRateLimiter(Protocol):
    """Blocks the calling coroutine until a request slot is available."""

    async def acquire_async(self) -> None:
        """Block the coroutine until the caller may proceed with one request."""
        ...


@runtime_checkable
class Clock(Protocol):
    """Time source: a monotonic reader plus sync and async sleeps."""

    def time(self) -> float:
        """Return the current time in seconds."""
        ...

    def sleep(self, seconds: float) -> None:
        """Block the thread for ``seconds`` seconds."""
        ...

    async def sleep_async(self, seconds: float) -> None:
        """Block the coroutine for ``seconds`` seconds."""
        ...


class SystemClock:
    """Real time source: a configurable reader with sync/async sleeps.

    Defaults to :func:`time.monotonic`, correct for in-process limiters.
    Cross-process limiters backed by :class:`FileStore` must compare
    timestamps across processes, which share no monotonic epoch -- pass
    ``source=time.time`` (wall clock) there.

    Args:
      source: Zero-arg callable returning the current time in seconds.

    """

    def __init__(self, *, source: Callable[[], float] = time.monotonic) -> None:
        self._source = source

    def time(self) -> float:
        """Return the current time in seconds from the configured source."""
        return self._source()

    def sleep(self, seconds: float) -> None:
        """Sleep via ``time.sleep``."""
        time.sleep(seconds)

    async def sleep_async(self, seconds: float) -> None:
        """Sleep via ``asyncio.sleep``."""
        await asyncio.sleep(seconds)


_STATE_BYTES = 16  # two packed little-endian doubles: (tokens, updated)


@runtime_checkable
class Store(Protocol):
    """Mutually-excluded backing store for a token bucket's ``(tokens, updated)``.

    ``transact`` runs ``update`` while holding the store's lock, passing the
    current state and committing the returned state. The scope of the lock
    -- a process-local mutex or a cross-process file lock -- is what decides
    whether a limiter coordinates threads or whole processes.
    """

    def transact(
        self, update: Callable[[tuple[float, float] | None], tuple[float, float]]
    ) -> None:
        """Apply ``update`` to the stored state under the store's lock.

        Args:
          update: Receives the current ``(tokens, updated)`` state, or
            ``None`` when uninitialized, and returns the state to commit.

        """
        ...


class InProcessStore:
    """In-memory token-bucket state guarded by a ``threading.Lock``.

    Coordinates threads (and coroutines) within one process. The default
    backing for a limiter when no cross-process sharing is needed.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state: tuple[float, float] | None = None

    def transact(
        self, update: Callable[[tuple[float, float] | None], tuple[float, float]]
    ) -> None:
        """Run ``update`` on the in-memory state under the thread lock."""
        with self._lock:
            self._state = update(self._state)


class FileStore:
    """Token-bucket state in a file, guarded by an ``fcntl`` lock.

    Coordinates separate processes that share one key: the ``(tokens,
    updated)`` pair is packed into the file and read/written under an
    exclusive ``flock``, so every process paces against one shared budget.
    Pair with a wall-clock :class:`Clock` -- ``updated`` is wall-clock time,
    which (unlike ``monotonic``) is comparable across processes.

    Args:
      path: Lockfile holding the packed state. Created on first use.

    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._fd: int | None = None
        # ``flock`` is per-open-file-description, so it does NOT serialize
        # threads in this process that share ``self._fd``. The inner thread
        # lock provides that; the flock provides cross-process exclusion.
        # Mirrors providers/lib/oauth.py's two-layer lock.
        self._lock = threading.Lock()

    def transact(
        self, update: Callable[[tuple[float, float] | None], tuple[float, float]]
    ) -> None:
        """Run ``update`` on the on-disk state under thread + process locks."""
        with self._lock:
            if self._fd is None:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                self._fd = os.open(
                    self._path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o600
                )
            fcntl.flock(self._fd, fcntl.LOCK_EX)
            try:
                os.lseek(self._fd, 0, os.SEEK_SET)
                raw = os.read(self._fd, _STATE_BYTES)
                current = (
                    cast(tuple[float, float], struct.unpack("<dd", raw))
                    if len(raw) == _STATE_BYTES
                    else None
                )
                tokens, updated = update(current)
                os.lseek(self._fd, 0, os.SEEK_SET)
                _ = os.write(self._fd, struct.pack("<dd", tokens, updated))
            finally:
                fcntl.flock(self._fd, fcntl.LOCK_UN)


class SlidingWindowRateLimiter:
    """Allow at most ``max_calls`` requests in any ``per_seconds`` window.

    Records the timestamp of each granted call in a deque; on acquire,
    evicts timestamps older than the window. When the window is full, the
    new call reserves the first free instant (oldest + ``per_seconds``) and
    returns the wait until then. Unlike a fixed-window counter, this never
    permits a ``2x`` burst straddling a window boundary.

    Args:
      max_calls: Maximum number of calls permitted per window.
      per_seconds: Window length in seconds.
      clock: Time source; defaults to monotonic system time. Injectable
        for tests.

    """

    def __init__(
        self,
        max_calls: int,
        per_seconds: float = 1.0,
        *,
        clock: Clock | None = None,
    ) -> None:
        if max_calls < 1:
            raise ValueError(f"max_calls must be >= 1, got {max_calls}")
        if per_seconds <= 0:
            raise ValueError(f"per_seconds must be > 0, got {per_seconds}")
        self._max_calls = max_calls
        self._per_seconds = per_seconds
        self._clock: Clock = clock if clock is not None else SystemClock()
        self._lock = threading.Lock()
        self._calls: deque[float] = deque()

    def acquire(self) -> None:
        """Block the thread until within the window, then record the call."""
        wait = self._reserve()
        if wait > 0:
            self._clock.sleep(wait)

    async def acquire_async(self) -> None:
        """Block the coroutine until within the window, then record the call."""
        wait = self._reserve()
        if wait > 0:
            await self._clock.sleep_async(wait)

    def _reserve(self) -> float:
        """Reserve the next call slot; return seconds to wait for it.

        Returns:
          wait: Seconds the caller must sleep before its reserved slot.

        """
        with self._lock:
            now = self._clock.time()
            # Evict by the same expiry expression used for the reserved
            # slot below, so an entry is dropped exactly when it expires.
            # A separately-rounded ``now - per_seconds`` cutoff can leave a
            # boundary entry un-evicted and the wait at ~0 -> hot spin.
            while self._calls and self._calls[0] + self._per_seconds <= now:
                self._calls.popleft()
            if len(self._calls) < self._max_calls:
                self._calls.append(now)
                return 0.0
            # This caller fits once the ``max_calls``-th-most-recent entry
            # expires -- i.e. ``_calls[-max_calls]``, NOT ``_calls[0]``. Using
            # the front would let every queued caller reserve the same slot,
            # firing >max_calls in one window under burst.
            slot = self._calls[-self._max_calls] + self._per_seconds
            self._calls.append(slot)
            return slot - now


class TokenBucketRateLimiter:
    """Smooth average rate of ``max_calls`` per ``per_seconds`` with bursts.

    The bucket holds up to ``max_calls`` tokens and refills continuously at
    ``max_calls / per_seconds`` tokens per second. Each acquire spends one
    token, waiting for the shortfall when the bucket is empty. Idle time
    accrues tokens only up to the capacity, so a long pause never grants a
    burst larger than ``max_calls``.

    The ``(tokens, updated)`` state lives in a pluggable :class:`Store`. The
    default :class:`InProcessStore` coordinates threads in one process; pass
    a :class:`FileStore` (with a wall-clock ``clock``) to share one budget
    across processes -- e.g. several processes holding the same API key.

    Args:
      max_calls: Bucket capacity -- the largest instantaneous burst.
      per_seconds: Period over which ``max_calls`` tokens are refilled.
      clock: Time source; defaults to monotonic system time. Use a
        wall-clock source with :class:`FileStore`. Injectable for tests.
      store: Backing state + lock; defaults to :class:`InProcessStore`.

    """

    def __init__(
        self,
        max_calls: int,
        per_seconds: float = 1.0,
        *,
        clock: Clock | None = None,
        store: Store | None = None,
    ) -> None:
        if max_calls < 1:
            raise ValueError(f"max_calls must be >= 1, got {max_calls}")
        if per_seconds <= 0:
            raise ValueError(f"per_seconds must be > 0, got {per_seconds}")
        self._capacity = float(max_calls)
        self._refill_per_sec = max_calls / per_seconds
        self._clock: Clock = clock if clock is not None else SystemClock()
        self._store: Store = store if store is not None else InProcessStore()

    def acquire(self) -> None:
        """Block the thread until one token is available, then spend it."""
        wait = self._reserve()
        if wait > 0:
            self._clock.sleep(wait)

    async def acquire_async(self) -> None:
        """Block the coroutine until one token is available, then spend it."""
        wait = self._reserve()
        if wait > 0:
            await self._clock.sleep_async(wait)

    def _reserve(self) -> float:
        """Spend one token; return seconds to wait for it to be earned.

        Commits the spend immediately (advancing ``updated`` past the
        returned wait) inside the store's locked transaction, so sync and
        async callers need only sleep the returned duration -- no re-check
        loop -- and concurrent callers (threads or processes) serialize on
        the store's lock.

        Returns:
          wait: Seconds the caller must sleep before its token is earned.

        Note on reading ``now`` before the lock (a recurring review question):
        capturing ``now`` outside ``transact`` looks racy -- two callers can
        read near-identical ``now`` values, then commit in lock-serialized
        order. It is nonetheless correct, because ``updated`` is set to
        ``now + wait`` and ``wait`` absorbs any staleness:

        - If caller A's ``now`` is *earlier* than the ``updated`` already on
          disk (because B committed a future reservation first), then
          ``now - updated`` is negative, so A's ``tokens`` only shrink, A's
          ``wait`` only grows, and A commits ``updated = now + wait`` which is
          >= the value it read. ``updated`` never moves backward; the bucket
          is never over-credited.
        - Reading ``now`` *inside* the lock would tighten spacing by at most
          the lock-hold time (microseconds here), not fix a correctness bug.

        So the only way to over-grant is a wall clock that steps backward
        (e.g. NTP), which is out of scope -- ``FileStore`` already requires a
        well-behaved wall clock to compare timestamps across processes.

        """
        now = self._clock.time()
        wait = 0.0

        def update(state: tuple[float, float] | None) -> tuple[float, float]:
            nonlocal wait
            tokens, updated = state if state is not None else (self._capacity, now)
            tokens = min(
                self._capacity, tokens + (now - updated) * self._refill_per_sec
            )
            wait = max(0.0, (1.0 - tokens) / self._refill_per_sec)
            # ``now + wait`` is monotonic non-decreasing even when ``now`` is a
            # stale early read (see the method docstring), so ``updated`` never
            # regresses and the budget cannot be exceeded.
            tokens += wait * self._refill_per_sec - 1.0
            return tokens, now + wait

        self._store.transact(update)
        return wait
