"""Tests for ``providers.lib.hotspare``: active/spare lifecycle."""

from __future__ import annotations

from typing import override

import asyncio

import pytest

from sagent.providers.lib.hotspare import HotSpare
from sagent.providers.lib.subproc import Subproc


class _FakeSubproc(Subproc):
    """Subproc test double tracking close state without spawning a process."""

    def __init__(self) -> None:
        super().__init__(["python3", "-c", ""])
        self.close_count = 0
        self.closed = False

    @override
    async def close(self) -> None:
        self.close_count += 1
        self.closed = True

    @property
    @override
    def is_alive(self) -> bool:
        return not self.closed


def _make_subproc() -> Subproc:
    """Spawn an idle Python subprocess used as a stand-in ``Subproc``."""
    return Subproc(["python3", "-c", "import sys; sys.stdin.read()"])


@pytest.mark.real_sleep
@pytest.mark.asyncio
async def test_first_acquire_spawns_active() -> None:
    """The first ``acquire`` creates the active subprocess via the factory."""
    spawn_count = 0

    async def factory() -> Subproc:
        nonlocal spawn_count
        spawn_count += 1
        proc = _make_subproc()
        await proc.start()
        return proc

    pool = HotSpare(factory)
    active = await pool.acquire()
    assert active.is_alive
    # Give the background warm-up one tick.
    await asyncio.sleep(0.05)
    assert spawn_count >= 2  # active + warmed spare
    await pool.close()


@pytest.mark.real_sleep
@pytest.mark.asyncio
async def test_respawn_promotes_spare_in_place() -> None:
    """``respawn`` closes the active subprocess and swaps the spare in."""
    procs: list[Subproc] = []

    async def factory() -> Subproc:
        proc = _make_subproc()
        await proc.start()
        procs.append(proc)
        return proc

    pool = HotSpare(factory)
    first = await pool.acquire()
    # Wait for the spare to be ready so respawn doesn't fall through to
    # the synchronous-spawn fallback path.
    if pool._spare_task is not None:
        _ = await pool._spare_task
        pool._spare = pool._spare_task.result()
        pool._spare_task = None
    new_active = await pool.respawn()
    assert new_active is not first
    assert not first.is_alive
    assert new_active.is_alive
    await pool.close()


@pytest.mark.real_sleep
@pytest.mark.asyncio
async def test_concurrent_respawn_serialised_by_lock() -> None:
    """Two ``respawn`` calls in flight execute serially, never overlapping."""
    spawn_count = 0

    async def factory() -> Subproc:
        nonlocal spawn_count
        spawn_count += 1
        proc = _make_subproc()
        await proc.start()
        return proc

    pool = HotSpare(factory)
    _ = await pool.acquire()
    # Drain the warm-up so we control the next spawn count exactly.
    if pool._spare_task is not None:
        _ = await pool._spare_task
        pool._spare = pool._spare_task.result()
        pool._spare_task = None
    before = spawn_count
    await asyncio.gather(pool.respawn(), pool.respawn())
    # At most three new processes were spawned: two replacements + one
    # background warmer. The lock ensures we never re-entered the
    # promotion path.
    assert spawn_count - before <= 3
    await pool.close()


@pytest.mark.asyncio
async def test_respawn_factory_failure_closes_old_active() -> None:
    procs: list[_FakeSubproc] = []

    async def factory() -> Subproc:
        if procs:
            raise RuntimeError("spawn failed")
        proc = _FakeSubproc()
        procs.append(proc)
        return proc

    pool = HotSpare(factory)
    active = await pool.acquire()
    assert isinstance(active, _FakeSubproc)

    with pytest.raises(RuntimeError, match="spawn failed"):
        _ = await pool.respawn()

    assert active.closed
    assert pool.active is None
    await pool.close()


@pytest.mark.asyncio
async def test_transport_failure_budget_exhaustion_clears_poisoned_active() -> None:
    procs: list[_FakeSubproc] = []

    async def factory() -> Subproc:
        proc = _FakeSubproc()
        procs.append(proc)
        return proc

    pool = HotSpare(factory, max_consecutive_transport_failures=1)
    active = await pool.acquire()

    with pytest.raises(RuntimeError, match="transport failure budget exhausted"):
        _ = await pool.respawn_after_transport_failure()

    assert pool.active is None
    assert not active.is_alive
    assert [proc.close_count for proc in procs] == [1]
    await pool.close()


@pytest.mark.asyncio
async def test_transport_failure_budget_exhaustion_closes_warmed_spare() -> None:
    procs: list[_FakeSubproc] = []

    async def factory() -> Subproc:
        proc = _FakeSubproc()
        procs.append(proc)
        return proc

    pool = HotSpare(factory, max_consecutive_transport_failures=1)
    _ = await pool.acquire()
    if pool._spare_task is not None:
        _ = await pool._spare_task

    with pytest.raises(RuntimeError, match="transport failure budget exhausted"):
        _ = await pool.respawn_after_transport_failure()

    assert pool.active is None
    assert pool._spare_task is None
    assert pool._spare is None
    assert [proc.close_count for proc in procs] == [1, 1]
    assert all(not proc.is_alive for proc in procs)


@pytest.mark.asyncio
async def test_concurrent_transport_failure_budget_accounting_is_serialized() -> None:
    procs: list[_FakeSubproc] = []
    factory_entered = asyncio.Event()
    release_factory = asyncio.Event()

    async def factory() -> Subproc:
        proc = _FakeSubproc()
        procs.append(proc)
        if len(procs) == 2:
            factory_entered.set()
            await release_factory.wait()
        return proc

    pool = HotSpare(factory, max_consecutive_transport_failures=2)
    active = await pool.acquire()

    first_task = asyncio.create_task(pool.respawn_after_transport_failure())
    await factory_entered.wait()
    second_task = asyncio.create_task(pool.respawn_after_transport_failure())
    await asyncio.sleep(0)
    assert not second_task.done()
    release_factory.set()

    first_result, second_result = await asyncio.gather(
        first_task,
        second_task,
        return_exceptions=True,
    )

    assert first_result is procs[1]
    assert isinstance(second_result, RuntimeError)
    assert "transport failure budget exhausted" in str(second_result)
    assert pool.active is None
    assert not active.is_alive
    assert not procs[1].is_alive
    assert [proc.close_count for proc in procs[:2]] == [1, 1]
    await pool.close()
    assert all(not proc.is_alive for proc in procs)


@pytest.mark.real_sleep
@pytest.mark.asyncio
async def test_transport_failure_budget_resets_after_success() -> None:
    async def factory() -> Subproc:
        proc = _make_subproc()
        await proc.start()
        return proc

    pool = HotSpare(factory, max_consecutive_transport_failures=2)
    _ = await pool.acquire()

    _ = await pool.respawn_after_transport_failure()
    pool.record_success()
    _ = await pool.respawn_after_transport_failure()

    await pool.close()


@pytest.mark.real_sleep
@pytest.mark.asyncio
async def test_normal_respawn_excluded_from_transport_failure_budget() -> None:
    async def factory() -> Subproc:
        proc = _make_subproc()
        await proc.start()
        return proc

    pool = HotSpare(factory, max_consecutive_transport_failures=1)
    _ = await pool.acquire()

    _ = await pool.respawn()
    _ = await pool.respawn()

    await pool.close()


@pytest.mark.asyncio
async def test_close_waits_for_in_flight_respawn_to_close_new_active() -> None:
    procs: list[_FakeSubproc] = []
    spare_started = asyncio.Event()
    release_spare = asyncio.Event()

    async def factory() -> Subproc:
        proc = _FakeSubproc()
        procs.append(proc)
        if len(procs) == 2:
            spare_started.set()
            await release_spare.wait()
        return proc

    pool = HotSpare(factory, max_consecutive_transport_failures=2)
    first = await pool.acquire()
    await spare_started.wait()

    respawn_task = asyncio.create_task(pool.respawn_after_transport_failure())
    await asyncio.sleep(0)
    close_task = asyncio.create_task(pool.close())
    await asyncio.sleep(0)
    release_spare.set()
    _ = await respawn_task
    await close_task

    assert first is procs[0]
    assert pool.active is None
    assert [proc.close_count for proc in procs] == [1, 1]
    assert all(not proc.is_alive for proc in procs)


@pytest.mark.asyncio
async def test_close_cancels_warmup_without_leaking_produced_spare() -> None:
    procs: list[_FakeSubproc] = []
    warm_spare_produced = asyncio.Event()
    warm_spare_cancelled = asyncio.Event()

    async def factory() -> Subproc:
        proc = _FakeSubproc()
        procs.append(proc)
        if len(procs) == 2:
            warm_spare_produced.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                warm_spare_cancelled.set()
                return proc
        return proc

    pool = HotSpare(factory)
    active = await pool.acquire()
    await warm_spare_produced.wait()
    await pool.close()

    assert warm_spare_cancelled.is_set()
    assert active is procs[0]
    assert [proc.close_count for proc in procs] == [1, 1]
    assert all(not proc.is_alive for proc in procs)


@pytest.mark.real_sleep
@pytest.mark.asyncio
async def test_close_idempotent_and_tears_down_active() -> None:
    """``close`` terminates the active subprocess and is safe to call twice."""

    async def factory() -> Subproc:
        proc = _make_subproc()
        await proc.start()
        return proc

    pool = HotSpare(factory)
    active = await pool.acquire()
    assert active.is_alive
    await pool.close()
    assert not active.is_alive
    await pool.close()


@pytest.mark.real_sleep
@pytest.mark.asyncio
async def test_acquire_after_close_raises() -> None:
    """A closed pool refuses to hand out new subprocesses."""

    async def factory() -> Subproc:
        proc = _make_subproc()
        await proc.start()
        return proc

    pool = HotSpare(factory)
    await pool.close()
    with pytest.raises(RuntimeError, match="pool is closed"):
        _ = await pool.acquire()


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
