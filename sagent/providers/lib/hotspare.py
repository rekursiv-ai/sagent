"""Hot-spare subprocess pool: one warm CLI process ready for respawn.

Spawning a vendor CLI costs ~5s on a cold start (Node load + auth +
skill scan; see ``docs/private/cli_provider.md`` §A.1). Keeping one
fully-initialized spare hidden in the background lets respawn paths
(sagent compaction, system-prompt change, mid-stream error, periodic
safety valve) hand the user a fresh subprocess in the time it takes
to swap a pointer.

The class is generic over the provider's per-subprocess init recipe:
the caller supplies a ``factory`` coroutine that returns a ready-to-
talk ``Subproc``. ``HotSpare`` owns active/spare state, serialises
respawns behind an ``asyncio.Lock``, and warms a replacement spare in
the background after each swap.
"""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from typing import Any

import asyncio
import logging

from sagent.providers.lib.subproc import Subproc
from sagent.types.exceptions import log_task_exception


__all__ = ["HotSpare"]

logger = logging.getLogger(__name__)


class HotSpare:
    """Manage one active subprocess plus one prewarmed spare.

    Args:
      factory: Coroutine factory producing a fully-initialised ``Subproc``.
          Called once per warm-up; must include any provider-specific
          handshake (claude has none, gemini's ACP ``initialize`` /
          ``authenticate`` / ``session/new`` lives here).

    """

    def __init__(
        self,
        factory: Callable[[], Coroutine[Any, Any, Subproc]],
    ) -> None:
        self._factory = factory
        self._active: Subproc | None = None
        self._spare: Subproc | None = None
        self._spare_task: asyncio.Task[Subproc] | None = None
        self._respawn_lock = asyncio.Lock()
        self._closed = False

    async def acquire(self) -> Subproc:
        """Return the active subprocess, promoting a spare if needed.

        Returns:
          subproc: The current active subprocess.

        Raises:
          RuntimeError: If the pool has been closed.

        """
        if self._closed:
            raise RuntimeError("HotSpare: pool is closed")
        if self._active is None:
            async with self._respawn_lock:
                if self._active is None:
                    self._active = await self._take_or_make_spare()
            self._kick_warm()
        return self._active

    async def respawn(self) -> Subproc:
        """Close the active subprocess and promote the spare.

        Returns:
          subproc: The new active subprocess (the prior spare, or a
              freshly-spawned one if no spare was ready).

        """
        async with self._respawn_lock:
            old = self._active
            self._active = None
            self._active = await self._take_or_make_spare()
            if old is not None:
                await old.close()
        self._kick_warm()
        return self._active

    async def close(self) -> None:
        """Tear down active + spare; idempotent."""
        if self._closed:
            return
        self._closed = True
        task = self._spare_task
        if task is not None and not task.done():
            _ = task.cancel()
            try:
                spare = await task
                await spare.close()
            except (asyncio.CancelledError, Exception) as exc:  # noqa: BLE001 -- shutdown must not raise
                logger.debug("hot spare close: cancelled warm-up: %s", exc)
        elif task is not None and task.done() and not task.cancelled():
            try:
                spare = task.result()
                await spare.close()
            except Exception as exc:  # noqa: BLE001 -- shutdown must not raise
                logger.debug("hot spare close: spare close raised: %s", exc)
        self._spare_task = None
        if self._spare is not None:
            await self._spare.close()
            self._spare = None
        if self._active is not None:
            await self._active.close()
            self._active = None

    @property
    def active(self) -> Subproc | None:
        """The currently-active subprocess, ``None`` before first acquire."""
        return self._active

    async def _take_or_make_spare(self) -> Subproc:
        """Consume the warmed spare; spawn synchronously if it isn't ready."""
        task = self._spare_task
        self._spare_task = None
        if task is not None:
            try:
                return await task
            except Exception as exc:  # noqa: BLE001 -- spare-warm failure: log and fall back to a fresh spawn
                logger.warning("hot spare warm-up failed; spawning fresh: %s", exc)
        if self._spare is not None:
            spare = self._spare
            self._spare = None
            return spare
        return await self._factory()

    def _kick_warm(self) -> None:
        """Start a background spare warm-up if none is in flight."""
        if self._closed:
            return
        if self._spare_task is not None or self._spare is not None:
            return
        self._spare_task = asyncio.create_task(self._factory())
        self._spare_task.add_done_callback(
            log_task_exception(logger, "hot spare warm-up crashed"),
        )
