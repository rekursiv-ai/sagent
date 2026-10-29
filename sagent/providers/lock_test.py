"""Tests for the lock a login holds around reading, renewing and writing its file."""

from __future__ import annotations

from typing import TYPE_CHECKING

import asyncio

import pytest

from sagent.providers.lock import file_lock


if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.asyncio
async def test_a_second_holder_waits_for_the_first(tmp_path: Path) -> None:
    lock = tmp_path / "auth.json.lock"
    order: list[str] = []
    held = asyncio.Event()

    async def first() -> None:
        async with file_lock(lock):
            order.append("first in")
            held.set()
            await asyncio.sleep(0.05)
            order.append("first out")

    async def second() -> None:
        await held.wait()
        async with file_lock(lock):
            order.append("second in")

    _ = await asyncio.wait_for(asyncio.gather(first(), second()), timeout=5)

    assert order == ["first in", "first out", "second in"]


@pytest.mark.asyncio
async def test_the_lock_file_and_its_directory_are_made_private(tmp_path: Path) -> None:
    lock = tmp_path / "new" / "auth.json.lock"

    async with file_lock(lock):
        pass

    assert lock.stat().st_mode & 0o777 == 0o600
    assert lock.parent.stat().st_mode & 0o777 == 0o700


@pytest.mark.asyncio
async def test_a_waiter_cancelled_leaves_the_lock_to_the_next(tmp_path: Path) -> None:
    lock = tmp_path / "auth.json.lock"
    release = asyncio.Event()
    held = asyncio.Event()

    async def holder() -> None:
        async with file_lock(lock):
            held.set()
            await release.wait()

    async def waiter() -> None:
        async with file_lock(lock):
            pass

    holding = asyncio.create_task(holder())
    await held.wait()
    waiting = asyncio.create_task(waiter())
    await asyncio.sleep(0.03)
    _ = waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting
    release.set()
    await holding

    async with asyncio.timeout(5), file_lock(lock):
        pass


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
