"""Tests for ``providers.lib.hotspare``: active/spare lifecycle."""

from __future__ import annotations

import asyncio

import pytest

from sagent.providers.lib.hotspare import _HotSpare
from sagent.providers.lib.subproc import _Subproc


def _make_subproc() -> _Subproc:
    """Spawn an idle Python subprocess used as a stand-in ``_Subproc``."""
    return _Subproc(["python3", "-c", "import sys; sys.stdin.read()"])


@pytest.mark.real_sleep
@pytest.mark.asyncio
async def test_first_acquire_spawns_active() -> None:
    """The first ``acquire`` creates the active subprocess via the factory."""
    spawn_count = 0

    async def factory() -> _Subproc:
        nonlocal spawn_count
        spawn_count += 1
        proc = _make_subproc()
        await proc.start()
        return proc

    pool = _HotSpare(factory)
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
    procs: list[_Subproc] = []

    async def factory() -> _Subproc:
        proc = _make_subproc()
        await proc.start()
        procs.append(proc)
        return proc

    pool = _HotSpare(factory)
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

    async def factory() -> _Subproc:
        nonlocal spawn_count
        spawn_count += 1
        proc = _make_subproc()
        await proc.start()
        return proc

    pool = _HotSpare(factory)
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


@pytest.mark.real_sleep
@pytest.mark.asyncio
async def test_close_idempotent_and_tears_down_active() -> None:
    """``close`` terminates the active subprocess and is safe to call twice."""

    async def factory() -> _Subproc:
        proc = _make_subproc()
        await proc.start()
        return proc

    pool = _HotSpare(factory)
    active = await pool.acquire()
    assert active.is_alive
    await pool.close()
    assert not active.is_alive
    await pool.close()


@pytest.mark.real_sleep
@pytest.mark.asyncio
async def test_acquire_after_close_raises() -> None:
    """A closed pool refuses to hand out new subprocesses."""

    async def factory() -> _Subproc:
        proc = _make_subproc()
        await proc.start()
        return proc

    pool = _HotSpare(factory)
    await pool.close()
    with pytest.raises(RuntimeError, match="pool is closed"):
        _ = await pool.acquire()


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
