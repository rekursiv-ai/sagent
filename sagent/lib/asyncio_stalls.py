"""Name the code that stalls an event loop, while it is stalling.

Synchronous work in a callback holds the loop's thread, and everything the
loop serves -- keystrokes, repaints, sockets -- waits behind it. Measured from
inside the loop, a stall is visible only after it ends, when the code that
caused it has returned. :func:`watch_stalls` looks from another thread: it
pings the loop, and when a ping goes unanswered past the threshold it logs the
loop thread's stack as it stands, then how long the stall lasted.

Typical usage example:

  with watch_stalls(threshold_sec=0.05):
      await serve()
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

import asyncio
import logging
import sys
import threading
import time
import traceback


if TYPE_CHECKING:
    from collections.abc import Generator


__all__ = ["watch_stalls"]


_logger = logging.getLogger(__name__)


@contextmanager
def watch_stalls(*, threshold_sec: float = 0.05) -> Generator[threading.Thread]:
    """Log each stall of the running loop longer than ``threshold_sec``.

    Args:
      threshold_sec: How long the loop may go unanswering before it is a stall.

    Yields:
      watcher: The watching thread; it ends with the block.

    """
    loop = asyncio.get_running_loop()
    stop = threading.Event()
    watcher = threading.Thread(
        target=_watch,
        args=(loop, threading.get_ident(), stop, threshold_sec),
        name="stall-watch",
        daemon=True,
    )
    watcher.start()
    try:
        yield watcher
    finally:
        stop.set()
        watcher.join()


def _watch(
    loop: asyncio.AbstractEventLoop,
    thread: int,
    stop: threading.Event,
    threshold_sec: float,
) -> None:
    """Ping ``loop`` every ``threshold_sec``; log its stack whenever it stalls."""
    while not stop.wait(threshold_sec):
        answered = threading.Event()
        loop.call_soon_threadsafe(answered.set)
        began = time.monotonic()
        if answered.wait(threshold_sec) or stop.is_set():
            continue
        frame = sys._current_frames()[thread]  # noqa: SLF001 -- The one way to read another thread's stack.
        stack = "".join(traceback.format_stack(frame))
        # The block's end joins this thread on the loop, so the ping may never run.
        while not answered.wait(threshold_sec) and not stop.is_set():
            pass
        _logger.warning(
            "event loop stalled %.3f s; it was running:\n%s",
            time.monotonic() - began,
            stack,
        )
