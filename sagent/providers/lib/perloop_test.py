"""Per-loop resources: one lock and one client per event loop."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor

import asyncio
import gc
import threading

import pytest

from sagent.providers.lib.perloop import PerLoop


def test_one_loop_gets_one_value() -> None:
    """Repeated access on a loop returns the same object."""
    per = PerLoop(list[str])

    async def take() -> bool:
        return per.get() is per.get()

    assert asyncio.run(take())


def test_two_loops_get_different_values() -> None:
    """An asyncio.Lock binds to one loop; a second loop needs its own.

    Compared by holding both objects alive, not by ``id()``: CPython
    reuses the address of a freed object, so an id taken after the first
    loop closed can equal the second's.
    """
    per = PerLoop(asyncio.Lock)
    held: list[asyncio.Lock] = []

    async def take() -> None:
        held.append(per.get())

    asyncio.run(take())
    asyncio.run(take())
    assert held[0] is not held[1]


def _drive(contend: Callable[[], Awaitable[None]], runs: int = 3) -> list[str]:
    """Run ``contend`` on ``runs`` successive loops, collecting failures.

    Sequential loops, not a thread pool: a pool lets each worker release
    before the next contends, so the lock unbinds between runs and the
    defect never appears. Successive loops are also the real shape --
    every ``asyncio.run`` is a fresh loop.

    The sleep duration is load-bearing: without a real suspension the two
    holders never overlap, so the lock is never contended and the harness
    proves nothing. Do not replace it with a bare yield.

    Args:
      contend: Builds the coroutine to run on each loop.
      runs: How many loops to drive.

    Returns:
      failures: One message per loop that raised.

    """
    failures: list[str] = []
    for _ in range(runs):
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(contend())
        except RuntimeError as exc:
            failures.append(str(exc))
        finally:
            loop.close()
    return failures


def test_contended_lock_does_not_cross_loops() -> None:
    """The failure this exists to prevent, reproduced under contention.

    ``Lock.acquire`` returns without touching ``_get_loop`` when
    uncontended, so an unfixed provider passes any test with a single
    waiter. Only a SECOND waiter binds the lock to its loop, which is why
    this forces two waiters per loop.
    """
    per = PerLoop(asyncio.Lock)

    async def contend() -> None:
        lock = per.get()

        async def hold() -> None:
            async with lock:
                await asyncio.sleep(0.01)

        await asyncio.gather(hold(), hold())

    assert _drive(contend) == []


def test_a_shared_lock_fails_the_same_harness() -> None:
    """Proof the harness bites: one shared lock breaks under it.

    Without this the test above could pass for the wrong reason -- a
    harness that never contends is green against the very bug it claims
    to cover.
    """
    shared = asyncio.Lock()

    async def contend() -> None:
        async def hold() -> None:
            async with shared:
                await asyncio.sleep(0.01)

        await asyncio.gather(hold(), hold())

    failures = _drive(contend)
    assert failures, "the harness never contended, so it proves nothing"
    assert "different event loop" in failures[0]


def test_values_are_built_once_per_loop() -> None:
    """The factory runs once per loop, not once per access."""
    built = 0

    def build() -> object:
        nonlocal built
        built += 1
        return object()

    per = PerLoop(build)

    async def take() -> None:
        per.get()
        per.get()

    asyncio.run(take())
    asyncio.run(take())
    assert built == 2


def test_get_requires_a_running_loop() -> None:
    """Outside a loop there is no correct answer; say so rather than guess."""
    per = PerLoop(asyncio.Lock)
    with pytest.raises(RuntimeError, match="running event loop"):
        per.get()


def test_a_closed_loop_does_not_leak_its_value() -> None:
    """A per-loop cache keyed strongly would grow without bound.

    Long-lived processes open and close loops (every ``asyncio.run``); the
    values must not outlive them.
    """
    per = PerLoop(asyncio.Lock)

    async def take() -> None:
        per.get()

    for _ in range(3):
        asyncio.run(take())
    gc.collect()
    assert per.size() == 0


def test_a_contended_lock_does_not_pin_its_loop() -> None:
    """Weak keys alone do not bound growth: the value can pin the key.

    ``Lock.acquire`` stores ``lock._loop`` on the first *contended*
    acquisition, so a strongly-held value back-references the loop that
    is the weak key. Only contention builds that edge, which is why
    ``test_a_closed_loop_does_not_leak_its_value`` misses it.
    """
    per = PerLoop(asyncio.Lock)

    async def contend() -> None:
        lock = per.get()

        async def hold() -> None:
            async with lock:
                await asyncio.sleep(0.01)

        await asyncio.gather(hold(), hold())

    for _ in range(3):
        asyncio.run(contend())
    gc.collect()
    assert per.size() == 0


def test_a_none_value_is_cached_not_rebuilt() -> None:
    """``None`` is a legal ``T``, so it must not double as the miss sentinel.

    A provider storing ``tuple | None`` per loop is the live case: treating
    ``None`` as "absent" reruns the factory on every access.
    """
    calls = 0

    def build() -> None:
        nonlocal calls
        calls += 1

    per: PerLoop[None] = PerLoop(build)

    async def take() -> None:
        per.get()
        per.get()
        per.get()

    asyncio.run(take())
    assert calls == 1


def test_set_outside_a_loop_raises() -> None:
    """``set`` shares ``get``'s policy: no running loop, no correct answer."""
    per = PerLoop(asyncio.Lock)
    with pytest.raises(RuntimeError, match="running event loop"):
        per.set(asyncio.Lock())


def test_concurrent_first_access_is_safe() -> None:
    """Two threads first-touching the map must not race on insert."""
    per = PerLoop(asyncio.Lock)
    # Objects, not ids: every worker's lock stays alive in this list, so
    # no address can be recycled underneath the comparison.
    seen: list[asyncio.Lock] = []
    guard = threading.Lock()

    def worker() -> None:
        async def take() -> None:
            with guard:
                seen.append(per.get())

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(take())
        finally:
            loop.close()

    def run(_index: int) -> None:
        worker()

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(run, range(8)))
    assert len(seen) == 8
