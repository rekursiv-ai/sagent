"""Tests for ``sagent.lib.ratelimit`` rate limiters."""

from __future__ import annotations

from pathlib import Path

import asyncio
import struct
import threading

from sagent.lib.ratelimit import (
    AsyncRateLimiter,
    CooldownGate,
    FileStore,
    Pacer,
    RateLimiter,
    SlidingWindowRateLimiter,
    TokenBucketRateLimiter,
)


class FakeClock:
    """Deterministic clock with controllable sync and async sleep.

    ``time()`` advances only when a sleep is called, so tests assert
    pacing without wall-clock waits. ``sleeps`` records every requested
    duration. ``sleep_async`` advances identically, letting one fake
    clock drive both the sync and async ``acquire`` paths.
    """

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def time(self) -> float:
        """Return the current fake time in seconds."""
        return self.now

    def sleep(self, seconds: float) -> None:
        """Advance the fake clock by ``seconds`` and record the request."""
        self.sleeps.append(seconds)
        self.now += seconds

    async def sleep_async(self, seconds: float) -> None:
        """Async twin of :meth:`sleep`; advances the same fake time."""
        self.sleeps.append(seconds)
        self.now += seconds


# -- Protocol conformance ----------------------------------------------------


def test_both_limiters_satisfy_protocol() -> None:
    sliding: RateLimiter = SlidingWindowRateLimiter(max_calls=1)
    bucket: RateLimiter = TokenBucketRateLimiter(max_calls=1)
    assert callable(sliding.acquire)
    assert callable(bucket.acquire)


def test_both_limiters_satisfy_async_protocol() -> None:
    sliding: AsyncRateLimiter = SlidingWindowRateLimiter(max_calls=1)
    bucket: AsyncRateLimiter = TokenBucketRateLimiter(max_calls=1)
    assert callable(sliding.acquire_async)
    assert callable(bucket.acquire_async)


# -- SlidingWindowRateLimiter ------------------------------------------------


def test_sliding_allows_burst_up_to_max_without_sleeping() -> None:
    clock = FakeClock()
    limiter = SlidingWindowRateLimiter(max_calls=3, per_seconds=1.0, clock=clock)
    for _ in range(3):
        limiter.acquire()
    assert clock.sleeps == []  # first max_calls are free


def test_sliding_blocks_the_call_that_exceeds_the_window() -> None:
    clock = FakeClock()
    limiter = SlidingWindowRateLimiter(max_calls=2, per_seconds=1.0, clock=clock)
    limiter.acquire()  # t=0
    limiter.acquire()  # t=0
    limiter.acquire()  # 3rd in a 2-call window: must wait for oldest to age out
    assert clock.sleeps == [1.0]
    assert clock.now == 1.0


def test_sliding_queues_beyond_one_future_window() -> None:
    """Callers arriving while the window is full must stagger, not stack.

    With a clock that does not advance on ``sleep``, all five callers
    arrive at t=0. A max-2 window means the 3rd/4th wait one window and the
    5th waits two -- if every queued caller reserved off the same oldest
    timestamp, three would fire in one window (limit violation).
    """

    class NonAdvancingClock:
        def __init__(self) -> None:
            self.sleeps: list[float] = []

        def time(self) -> float:
            return 0.0

        def sleep(self, seconds: float) -> None:
            self.sleeps.append(seconds)

        async def sleep_async(self, seconds: float) -> None:
            self.sleeps.append(seconds)

    clock = NonAdvancingClock()
    limiter = SlidingWindowRateLimiter(max_calls=2, per_seconds=1.0, clock=clock)
    for _ in range(5):
        limiter.acquire()
    assert clock.sleeps == [1.0, 1.0, 2.0]  # 3rd&4th wait 1 window, 5th waits 2


def test_sliding_no_double_rate_across_window_boundary() -> None:
    """The defining property fixed-window violates: never >max per window."""
    clock = FakeClock()
    limiter = SlidingWindowRateLimiter(max_calls=2, per_seconds=1.0, clock=clock)
    # Saturate the window late.
    clock.now = 0.9
    limiter.acquire()
    limiter.acquire()
    # A fixed-window limiter would let 2 more through immediately at 0.91s.
    # The sliding window must instead pace the next call to oldest + 1.0.
    limiter.acquire()
    assert clock.now >= 1.9


def test_sliding_aged_out_calls_are_evicted() -> None:
    clock = FakeClock()
    limiter = SlidingWindowRateLimiter(max_calls=1, per_seconds=1.0, clock=clock)
    limiter.acquire()  # t=0
    clock.now = 5.0  # long gap; prior call is far outside the window
    limiter.acquire()  # should be free, not throttled
    assert clock.sleeps == []


# -- TokenBucketRateLimiter --------------------------------------------------


def test_bucket_allows_initial_burst_up_to_capacity() -> None:
    clock = FakeClock()
    limiter = TokenBucketRateLimiter(max_calls=3, per_seconds=1.0, clock=clock)
    for _ in range(3):
        limiter.acquire()
    assert clock.sleeps == []


def test_bucket_paces_after_capacity_drained() -> None:
    clock = FakeClock()
    limiter = TokenBucketRateLimiter(max_calls=2, per_seconds=1.0, clock=clock)
    limiter.acquire()
    limiter.acquire()
    # Bucket empty; refill rate is 2 tokens/sec => 0.5s per token.
    limiter.acquire()
    assert clock.sleeps == [0.5]


def test_bucket_refills_proportionally_over_time() -> None:
    clock = FakeClock()
    limiter = TokenBucketRateLimiter(max_calls=4, per_seconds=2.0, clock=clock)
    for _ in range(4):
        limiter.acquire()  # drain
    clock.now = 1.0  # 1s at 2 tokens/sec => 2 tokens refilled
    limiter.acquire()
    limiter.acquire()
    assert clock.sleeps == []  # two refilled tokens cover these


def test_bucket_never_exceeds_capacity_on_long_idle() -> None:
    clock = FakeClock()
    limiter = TokenBucketRateLimiter(max_calls=2, per_seconds=1.0, clock=clock)
    clock.now = 100.0  # idle forever; tokens must cap at capacity, not 100
    limiter.acquire()
    limiter.acquire()
    limiter.acquire()  # 3rd must pace; capacity was 2, not 100
    assert clock.sleeps == [0.5]


# -- thread safety -----------------------------------------------------------


# -- async parity ------------------------------------------------------------


def test_async_sliding_paces_like_sync() -> None:
    clock = FakeClock()
    limiter = SlidingWindowRateLimiter(max_calls=2, per_seconds=1.0, clock=clock)

    async def go() -> None:
        await limiter.acquire_async()
        await limiter.acquire_async()
        await limiter.acquire_async()  # 3rd waits for oldest to age out

    asyncio.run(go())
    assert clock.sleeps == [1.0]
    assert clock.now == 1.0


def test_async_bucket_paces_like_sync() -> None:
    clock = FakeClock()
    limiter = TokenBucketRateLimiter(max_calls=2, per_seconds=1.0, clock=clock)

    async def go() -> None:
        await limiter.acquire_async()
        await limiter.acquire_async()
        await limiter.acquire_async()  # bucket empty: 0.5s per token

    asyncio.run(go())
    assert clock.sleeps == [0.5]


def test_async_bucket_never_exceeds_capacity_on_long_idle() -> None:
    clock = FakeClock()
    limiter = TokenBucketRateLimiter(max_calls=2, per_seconds=1.0, clock=clock)
    clock.now = 100.0

    async def go() -> None:
        await limiter.acquire_async()
        await limiter.acquire_async()
        await limiter.acquire_async()

    asyncio.run(go())
    assert clock.sleeps == [0.5]


# -- FileStore: cross-process backing ----------------------------------------


def test_file_store_shares_budget_across_limiter_instances(tmp_path: Path) -> None:
    """Two limiters on one FileStore path share one token budget.

    Models separate processes: each constructs its own limiter object, but
    the bucket state lives in the shared file, so their calls are paced as
    a single 1-token-per-second stream rather than two independent ones.
    """
    path = tmp_path / "rl.bin"
    clock = FakeClock()
    a = TokenBucketRateLimiter(
        max_calls=1, per_seconds=1.0, clock=clock, store=FileStore(path)
    )
    b = TokenBucketRateLimiter(
        max_calls=1, per_seconds=1.0, clock=clock, store=FileStore(path)
    )
    a.acquire()  # spends the one shared token at t=0
    b.acquire()  # must wait ~1s for the *shared* bucket to refill
    assert clock.sleeps == [1.0]


def test_file_store_persists_across_new_limiter(tmp_path: Path) -> None:
    """A fresh limiter on an existing FileStore resumes the saved state."""
    path = tmp_path / "rl.bin"
    clock = FakeClock()
    first = TokenBucketRateLimiter(
        max_calls=1, per_seconds=1.0, clock=clock, store=FileStore(path)
    )
    first.acquire()  # drains the token, persists empty bucket
    second = TokenBucketRateLimiter(
        max_calls=1, per_seconds=1.0, clock=clock, store=FileStore(path)
    )
    second.acquire()  # sees the drained bucket on disk, paces
    assert clock.sleeps == [1.0]


def test_file_store_serializes_concurrent_threads(tmp_path: Path) -> None:
    """One FileStore shared by threads must not lose read-modify-writes.

    ``flock`` alone is per-open-file-description and does not exclude threads
    sharing one fd; without the inner thread lock, interleaved
    read/seek/write loses updates. 8 threads x 100 increments must total 800.
    """
    store = FileStore(tmp_path / "rl.bin")

    def increment(state: tuple[float, float] | None) -> tuple[float, float]:
        tokens = (state[0] if state is not None else 0.0) + 1.0
        return tokens, 0.0

    def hit() -> None:
        for _ in range(100):
            store.transact(increment)

    threads = [threading.Thread(target=hit) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    tokens, _ = struct.unpack("<dd", (tmp_path / "rl.bin").read_bytes())
    assert tokens == 800.0  # no lost updates


def test_sliding_is_thread_safe_under_contention() -> None:
    limiter = SlidingWindowRateLimiter(max_calls=1000, per_seconds=1.0)
    count = 0
    lock = threading.Lock()

    def worker() -> None:
        nonlocal count
        for _ in range(50):
            limiter.acquire()
            with lock:
                count += 1

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert count == 400


def test_cooldown_inactive_by_default() -> None:
    gate = CooldownGate(clock=FakeClock())
    assert gate.remaining() == 0.0


def test_cooldown_trigger_opens_window() -> None:
    clock = FakeClock()
    gate = CooldownGate(clock=clock)
    gate.trigger(8.0)
    assert gate.remaining() == 8.0


def test_cooldown_trigger_only_grows_window() -> None:
    # A shorter back-off must not shrink a longer one already in flight.
    clock = FakeClock()
    gate = CooldownGate(clock=clock)
    gate.trigger(10.0)
    gate.trigger(2.0)
    assert gate.remaining() == 10.0


def test_cooldown_elapses_as_clock_advances() -> None:
    clock = FakeClock()
    gate = CooldownGate(clock=clock)
    gate.trigger(5.0)
    gate.wait()  # sleeps 5s, advancing the fake clock
    assert gate.remaining() == 0.0


def test_cooldown_wait_async_sleeps_remaining() -> None:
    clock = FakeClock()
    gate = CooldownGate(clock=clock)
    gate.trigger(3.0)
    asyncio.run(gate.wait_async())
    assert clock.sleeps == [3.0]
    assert gate.remaining() == 0.0


def test_cooldown_shared_across_instances_via_filestore(tmp_path: Path) -> None:
    # Two gates over the same FileStore share the window: one's trigger makes
    # the other wait -- the cross-process contract.
    clock = FakeClock()
    store_path = tmp_path / "cd.lock"
    a = CooldownGate(store=FileStore(store_path), clock=clock)
    b = CooldownGate(store=FileStore(store_path), clock=FakeClock())
    a.trigger(7.0)
    assert b.remaining() == 7.0


# -- Pacer -------------------------------------------------------------------


def test_pacer_spends_one_token_per_pace() -> None:
    clock = FakeClock()
    # Capacity-1 bucket: the second pace must wait ~1s for a refill.
    pacer = Pacer(
        limiter=TokenBucketRateLimiter(max_calls=1, per_seconds=1.0, clock=clock),
        cooldown=CooldownGate(clock=clock),
    )
    pacer.pace()  # first token is free (bucket starts full)
    pacer.pace()  # drained -> waits for one refill
    assert clock.sleeps == [1.0]


def test_pacer_honors_cooldown_before_granting() -> None:
    clock = FakeClock()
    pacer = Pacer(
        limiter=TokenBucketRateLimiter(max_calls=10, per_seconds=1.0, clock=clock),
        cooldown=CooldownGate(clock=clock),
        cooldown_sec=5.0,
    )
    pacer.trigger_cooldown()
    pacer.pace()  # bucket has tokens, but the cooldown must be waited out first
    assert clock.sleeps == [5.0]


def test_pacer_cooldown_shared_via_filestore(tmp_path: Path) -> None:
    # A Pacer built on a FileStore-backed cooldown honors a window another
    # party opened -- the cross-process back-off contract.
    store_path = tmp_path / "cd.lock"
    other = CooldownGate(store=FileStore(store_path), clock=FakeClock())
    clock = FakeClock()
    pacer = Pacer(
        limiter=TokenBucketRateLimiter(max_calls=10, per_seconds=1.0, clock=clock),
        cooldown=CooldownGate(store=FileStore(store_path), clock=clock),
    )
    other.trigger(4.0)
    pacer.pace()
    assert clock.sleeps == [4.0]


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
