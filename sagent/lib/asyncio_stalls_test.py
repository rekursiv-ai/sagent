"""Tests for naming what stalls an event loop, while it stalls."""

from __future__ import annotations

import asyncio
import logging
import re
import threading
import time

import pytest

from sagent.lib.asyncio_stalls import watch_stalls


def _hog_the_loop(seconds: float) -> None:
    """Hold the loop's thread, as synchronous work in a callback does."""
    time.sleep(seconds)


@pytest.mark.asyncio
async def test_a_stall_is_logged_once_with_its_length_and_the_code_holding_the_loop(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING), watch_stalls(threshold_sec=0.005):
        await asyncio.sleep(0.002)
        _hog_the_loop(0.04)
        await asyncio.sleep(0.01)
    (stalled,) = caplog.records
    said = stalled.getMessage()
    first, stack = said.split("\n", 1)
    match = re.fullmatch(r"event loop stalled (\d\.\d{3}) s; it was running:", first)
    assert match is not None, first
    # Seen at most a threshold late, and logged once the loop runs again.
    assert 0.025 <= float(match.group(1)) <= 0.08
    assert all(line.startswith(("  File ", "    ")) for line in stack.splitlines())
    assert "in _hog_the_loop\n    time.sleep(seconds)" in stack
    assert stalled.name == "sagent.lib.asyncio_stalls"


@pytest.mark.asyncio
async def test_a_loop_that_keeps_up_logs_nothing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING), watch_stalls(threshold_sec=0.05):
        for _ in range(5):
            await asyncio.sleep(0.005)
    assert caplog.records == []


def test_a_block_ending_in_a_stall_ends_and_logs_it(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Its end joins the watcher on the loop, so the watcher's ping never runs."""

    async def stall_to_the_end() -> None:
        with watch_stalls(threshold_sec=0.01):
            await asyncio.sleep(0.02)
            _hog_the_loop(0.05)

    ending = threading.Thread(target=asyncio.run, args=(stall_to_the_end(),))
    with caplog.at_level(logging.WARNING):
        ending.start()
        ending.join(timeout=5)
    assert not ending.is_alive()
    (stalled,) = caplog.records
    assert "in _hog_the_loop" in stalled.getMessage()


@pytest.mark.asyncio
async def test_the_watch_thread_ends_with_the_block() -> None:
    with watch_stalls(threshold_sec=0.02) as watcher:
        await asyncio.sleep(0)
    assert not watcher.is_alive()


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
