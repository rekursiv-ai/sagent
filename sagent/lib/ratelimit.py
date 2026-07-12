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
from functools import cache
from pathlib import Path
from typing import Protocol, cast, runtime_checkable

import asyncio
import fcntl
import os
import random
import struct
import threading
import time

from sagent.lib.userdirs import data_dir


class CooldownActiveError(Exception):
    """A cooldown longer than the caller's tolerance is active -- do not wait.

    Raised by :meth:`CooldownRateLimiter.acquire` when the shared cooldown's
    remaining time exceeds the configured ``max_cooldown_wait_sec``. A long
    cooldown (e.g. a 48h scraping ban) must NOT be slept through -- the caller
    should abort and rotate/retry later instead of hanging the process.

    Attributes:
      remaining_sec: Seconds still left in the active cooldown window.

    """

    def __init__(self, remaining_sec: float) -> None:
        super().__init__(
            f"cooldown active for {remaining_sec:.0f}s more; "
            "not waiting -- retry later or change identity (IP / key)"
        )
        self.remaining_sec = remaining_sec


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


_STATE_BYTES = 16  # config-globals: ignore -- binary state width.


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


class RandomUniformPacer:
    """Sleep a fresh ``uniform(low, high)`` seconds before every grant.

    Unlike the fixed-interval limiters, this paces on a *randomized* gap: each
    :meth:`acquire` waits an independent ``uniform(low, high)`` draw. The
    irregular cadence is the point -- a scraper on a perfectly fixed interval is
    itself a bot signal, and a target (e.g. Google Scholar) with adaptive
    back-off trips on the regularity, not just the mean rate. This is the pacer
    the Scholar harvest ran on for months without a CAPTCHA: a 1 req/s floor
    plus ``uniform(6, 12)`` jitter on top. A fixed interval at the same *mean*
    (9s) burned in testing at ~21 requests; the variance is load-bearing.

    Satisfies the :class:`RateLimiter` / :class:`AsyncRateLimiter` protocols, so
    it drops into :class:`CooldownRateLimiter` as its ``limiter``. It keeps no
    call history -- each grant's wait depends only on the draw, not on when the
    previous grant fell -- so it is inherently process-local-agnostic (every
    process paces independently). Pair it with a shared :class:`CooldownGate`
    for the cross-process block signal; the jitter itself needs no shared state.

    Args:
      low: Minimum seconds to sleep before a grant (inclusive).
      high: Maximum seconds to sleep before a grant (inclusive). Must be ``>=
        low``.
      clock: Time source, for its ``sleep`` / ``sleep_async``; injectable for
        tests. Defaults to :class:`SystemClock`.
      rng: Draw source; defaults to :class:`random.SystemRandom` (the OG's
        ``_RNG``) so the sequence is unpredictable and un-seedable by an
        observer. Injectable for deterministic tests.

    """

    def __init__(
        self,
        low: float,
        high: float,
        *,
        clock: Clock | None = None,
        rng: random.Random | None = None,
    ) -> None:
        if low < 0:
            raise ValueError(f"low must be >= 0, got {low}")
        if high < low:
            raise ValueError(f"high must be >= low, got high={high}, low={low}")
        self._low = low
        self._high = high
        self._clock: Clock = clock if clock is not None else SystemClock()
        self._rng: random.Random = rng if rng is not None else random.SystemRandom()

    def _draw(self) -> float:
        """Return a fresh ``uniform(low, high)`` wait in seconds."""
        return self._rng.uniform(self._low, self._high)

    def acquire(self) -> None:
        """Sleep a fresh ``uniform(low, high)`` seconds, then grant."""
        self._clock.sleep(self._draw())

    async def acquire_async(self) -> None:
        """Async twin of :meth:`acquire`."""
        await self._clock.sleep_async(self._draw())


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


class CooldownGate:
    """Shared back-off window: one party's "slow down" makes all parties wait.

    A rate limiter paces *grants* at a steady rate. A cooldown is the orthogonal
    signal: an external authority (e.g. an HTTP 429 with a back-off) tells one
    requester to wait, and every concurrent requester sharing the same resource
    should honor it rather than independently rediscovering it (and amplifying
    the throttle by all retrying at once). Pair with the same scope-pluggable
    :class:`Store` a limiter uses: an :class:`InProcessStore` coordinates
    coroutines/threads in one process; a :class:`FileStore` (with a wall-clock
    :class:`Clock`) coordinates separate processes sharing the resource.

    The window is a single absolute deadline (``cooldown_until``). It is stored
    in the bucket :class:`Store`'s ``(float, float)`` slot as
    ``(cooldown_until, 0.0)`` -- the second slot is unused, reserved padding to
    reuse the existing store shape without widening its type.

    Args:
      store: Backing state + lock; defaults to :class:`InProcessStore`.
      clock: Time source; defaults to monotonic system time. Use a wall-clock
        source with :class:`FileStore` so the deadline is comparable across
        processes.

    """

    def __init__(
        self, *, store: Store | None = None, clock: Clock | None = None
    ) -> None:
        self._store: Store = store if store is not None else InProcessStore()
        self._clock: Clock = clock if clock is not None else SystemClock()

    def remaining(self) -> float:
        """Seconds left in the active cooldown window, or ``0.0`` if none."""
        until = 0.0

        def read(state: tuple[float, float] | None) -> tuple[float, float]:
            nonlocal until
            until = state[0] if state is not None else 0.0
            return state if state is not None else (0.0, 0.0)

        self._store.transact(read)
        return max(0.0, until - self._clock.time())

    def trigger(self, backoff_sec: float) -> None:
        """Open (or extend) the shared cooldown to ``now + backoff_sec``.

        Uses ``max`` against the stored deadline so a shorter back-off can
        never shrink a longer one already in flight; the window only grows.
        """
        target = self._clock.time() + backoff_sec

        def write(state: tuple[float, float] | None) -> tuple[float, float]:
            prior = state[0] if state is not None else 0.0
            return (max(prior, target), 0.0)

        self._store.transact(write)

    def wait(self, max_wait_sec: float | None = None) -> None:
        """Block the thread until the active cooldown elapses.

        Reads the window ONCE and acts on that single value, so the decision and
        the sleep can never disagree: with ``max_wait_sec`` set, a window longer
        than it raises :class:`CooldownActiveError` rather than being slept
        through -- there is no second read a concurrent :meth:`trigger` could
        extend between (the TOCTOU a check-then-wait pair would have).

        Args:
          max_wait_sec: Longest window to sleep. ``None`` sleeps any length.

        Raises:
          CooldownActiveError: When ``max_wait_sec`` is set and the window
            exceeds it.

        """
        wait = self.remaining()
        if max_wait_sec is not None and wait > max_wait_sec:
            raise CooldownActiveError(wait)
        if wait > 0:
            self._clock.sleep(wait)

    async def wait_async(self) -> None:
        """Block the coroutine until the active cooldown elapses.

        Reads the window once and sleeps it. An extension by another party
        *during* this sleep is not re-honored within this call -- the caller's
        own retry loop re-enters :meth:`wait_async` and picks it up, which
        bounds amplification without risking starvation under a continuous
        throttle.
        """
        wait = self.remaining()
        if wait > 0:
            await self._clock.sleep_async(wait)


class CooldownRateLimiter:
    """Steady rate limiting plus a shared block-triggered cooldown, as one gate.

    Composes the two orthogonal signals a polite client needs: a
    :class:`RateLimiter` paces grants at a steady rate, and a
    :class:`CooldownGate` absorbs an external "slow down" (an HTTP 429, a
    CAPTCHA) so every requester waits rather than each rediscovering the block.
    :meth:`acquire`, called before each request, honors an active cooldown first
    and then spends one rate-limit token; :meth:`trigger_cooldown`, called when
    a block is observed, opens the shared window.

    Share one :class:`CooldownRateLimiter` across every request to a resource so
    a burst cannot outrun the limit; back both the limiter and the gate with a
    :class:`FileStore` to extend that budget across processes on the same host.

    Args:
      limiter: Paces grants; one :meth:`RateLimiter.acquire` per
        :meth:`acquire`.
      cooldown: Shared back-off window honored before each grant.
      cooldown_sec: Seconds :meth:`trigger_cooldown` holds the window open when
        a block is observed.
      max_cooldown_wait_sec: The longest active cooldown :meth:`acquire` will
        sleep through. When the remaining window EXCEEDS this, :meth:`acquire`
        raises :class:`CooldownActiveError` instead of sleeping -- so a long ban
        (e.g. a multi-hour scraping block) aborts the caller for rotation rather
        than hanging it. ``None`` (default) preserves the sleep-always behavior,
        correct for short variable back-offs (a few-second 429 retry).

    """

    def __init__(
        self,
        *,
        limiter: RateLimiter,
        cooldown: CooldownGate,
        cooldown_sec: float = 120.0,
        max_cooldown_wait_sec: float | None = None,
    ) -> None:
        self._limiter = limiter
        self._cooldown = cooldown
        self._cooldown_sec = cooldown_sec
        self._max_cooldown_wait_sec = max_cooldown_wait_sec

    def acquire(self) -> None:
        """Wait out any active cooldown, then spend one rate-limit token.

        Raises:
          CooldownActiveError: When ``max_cooldown_wait_sec`` is set and the
            active cooldown's remaining time exceeds it -- the caller must not
            wait (rotate/retry later instead).

        """
        self._cooldown.wait(self._max_cooldown_wait_sec)
        self._limiter.acquire()

    def trigger_cooldown(self, backoff_sec: float | None = None) -> None:
        """Open (or extend) the shared cooldown.

        Args:
          backoff_sec: Seconds to hold the window open. Defaults to the
            configured ``cooldown_sec`` -- pass an explicit value for a
            variable back-off (e.g. exponential 429 retry) against the same
            shared window.

        """
        self._cooldown.trigger(
            self._cooldown_sec if backoff_sec is None else backoff_sec
        )


@cache
def cross_process_limiter(
    key: str,
    *,
    per_seconds: float,
    state_dir: Path | None = None,
    cooldown_sec: float = 120.0,
    max_cooldown_wait_sec: float | None = None,
) -> CooldownRateLimiter:
    """Return a host-wide :class:`CooldownRateLimiter` for ``key``, built once.

    Composes the standard cross-process gate: a one-token bucket refilling every
    ``per_seconds`` plus a shared cooldown, both backed by ``fcntl``-locked
    :class:`FileStore` files under ``state_dir`` and a wall clock. Every process
    on the host that asks for the same ``key`` shares one budget -- the unit a
    per-IP / per-API-key limit is enforced against -- not merely coroutines in
    one process.

    Cached on the full argument tuple so repeated calls with the same ``key``
    return one instance; a different ``per_seconds`` or ``state_dir`` yields a
    distinct gate. Built lazily (never at import) so it has no filesystem side
    effect until first use. The wall clock is mandatory with :class:`FileStore`:
    the persisted timestamp is compared across processes, which share no
    monotonic epoch.

    Args:
      key: Identity the limit is enforced against (e.g. an API-key / per-IP
        source name). Names the two lockfiles and the cache entry.
      per_seconds: Minimum seconds between grants (one token per window).
      state_dir: Directory for the ``{key}_ratelimit.lock`` /
        ``{key}_cooldown.lock`` files. Defaults to ``data_dir("ratelimit")``.
        Created on first write by :class:`FileStore`.
      cooldown_sec: Default seconds a triggered cooldown holds the shared window
        open (a caller may override per :meth:`CooldownRateLimiter.trigger_cooldown`).
      max_cooldown_wait_sec: When set, :meth:`CooldownRateLimiter.acquire` raises
        :class:`CooldownActiveError` instead of sleeping once the active cooldown
        exceeds it -- for a long ban that must abort the caller (rotate) rather
        than hang it. ``None`` (default) keeps sleep-always semantics.

    Returns:
      limiter: The shared cross-process cooldown-rate-limiter for ``key``.

    """
    base = state_dir if state_dir is not None else data_dir("ratelimit")
    clock = SystemClock(source=time.time)
    return CooldownRateLimiter(
        limiter=TokenBucketRateLimiter(
            max_calls=1,
            per_seconds=per_seconds,
            clock=clock,
            store=FileStore(base / f"{key}_ratelimit.lock"),
        ),
        cooldown=CooldownGate(
            store=FileStore(base / f"{key}_cooldown.lock"),
            clock=clock,
        ),
        cooldown_sec=cooldown_sec,
        max_cooldown_wait_sec=max_cooldown_wait_sec,
    )
