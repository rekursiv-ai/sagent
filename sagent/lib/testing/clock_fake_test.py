"""Tests for the event loop on a fake clock."""

from __future__ import annotations

import asyncio
import socket
import threading
import time

from sagent.lib.testing.clock_fake import FakeClockLoop


def test_the_clock_moves_only_to_the_next_timer() -> None:
    async def go() -> tuple[float, float]:
        loop = asyncio.get_running_loop()
        before = loop.time()
        await asyncio.sleep(3600)  # house-ignore[sleep] -- The jump under test.
        return before, loop.time()

    started = time.monotonic()
    assert asyncio.run(go(), loop_factory=FakeClockLoop) == (0.0, 3600.0)
    assert time.monotonic() - started < 60


def test_a_stalled_host_spends_no_loop_time() -> None:
    async def go() -> bool:
        async with asyncio.timeout(0.01):
            # A garbage collection on a loaded host: the CPU busy, the loop held.
            stalled = time.monotonic() + 0.05
            while time.monotonic() < stalled:
                pass
            await asyncio.sleep(0)
        return True

    assert asyncio.run(go(), loop_factory=FakeClockLoop)


def test_deadlines_fire_in_the_order_their_times_give() -> None:
    async def go() -> str:
        # Were the outer deadline first, its TimeoutError would escape ``go``.
        async with asyncio.timeout(0.04):
            try:
                await asyncio.wait_for(asyncio.Event().wait(), 0.02)
            except TimeoutError:
                return "inner"
        return "neither"

    assert asyncio.run(go(), loop_factory=FakeClockLoop) == "inner"


def test_a_ready_socket_is_read_with_no_jump() -> None:
    async def go() -> tuple[bytes, float]:
        loop = asyncio.get_running_loop()
        here, there = socket.socketpair()
        with here, there:
            here.setblocking(False)
            there.sendall(b"hi")
            before = loop.time()
            got = await asyncio.wait_for(loop.sock_recv(here, 2), 3600)
            return got, loop.time() - before

    got, waited = asyncio.run(go(), loop_factory=FakeClockLoop)
    assert got == b"hi"
    assert waited < 60


def test_with_no_timer_the_loop_blocks_for_another_thread() -> None:
    async def go() -> tuple[str, float]:
        loop = asyncio.get_running_loop()
        done = asyncio.Future[str]()

        def finish() -> None:
            time.sleep(0.01)  # house-ignore[sleep] -- Lands after the loop blocks.
            loop.call_soon_threadsafe(done.set_result, "from the thread")

        before = loop.time()
        thread = threading.Thread(target=finish)
        thread.start()
        try:
            return await done, loop.time() - before
        finally:
            thread.join()

    said, waited = asyncio.run(go(), loop_factory=FakeClockLoop)
    assert said == "from the thread"
    assert waited < 60


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
