"""Providers must survive being driven from more than one event loop.

A caller that builds one provider and uses it from several threads -- each
with its own loop -- previously got ``RuntimeError: ... is bound to a
different event loop`` from every thread but one, and a permanent hang
from the thread left waiting on a lock nobody would release.

Every sleep duration here is load-bearing: without a real suspension two
holders never overlap, so the lock is never contended and the harness proves
nothing. ``asyncio.Lock.acquire`` returns before touching its loop when
uncontended. Do not replace these with bare yields.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from contextlib import AbstractAsyncContextManager
from pathlib import Path

import asyncio
import threading
import time

from sagent.providers.anthropic.api import Anthropic
from sagent.providers.google.api import Google
from sagent.providers.lib.oauth import credential_file_lock
from sagent.providers.openai.sub import OpenAISubscription
from sagent.tools.core import locked_file_write


def _drive(work: Callable[[], Awaitable[None]], runs: int = 3) -> list[str]:
    """Run ``work`` on ``runs`` successive fresh loops.

    Args:
      work: Builds the coroutine to run on each loop.
      runs: How many loops to drive.

    Returns:
      failures: One message per loop that raised ``RuntimeError``.

    """
    failures: list[str] = []
    for _ in range(runs):
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(work())
        except RuntimeError as exc:
            failures.append(str(exc))
        finally:
            loop.close()
    return failures


def test_anthropic_sdk_lock_survives_a_second_loop() -> None:
    """The confirmed defect: one provider, several loops, contended."""
    provider = Anthropic(api_key="sk-test")

    async def work() -> None:
        async def take() -> None:
            await provider.get_sdk()
            await asyncio.sleep(0.01)

        await asyncio.gather(take(), take())
        await provider.close_sdk()

    assert _drive(work) == []


def test_anthropic_sdk_is_not_shared_across_loops() -> None:
    """An SDK client holds a pool owned by the loop that opened it.

    Handing loop B a client built on loop A leaves it talking to a dead
    selector, so each loop must get its own.
    """
    provider = Anthropic(api_key="sk-test")
    # Held, not compared by id(): CPython reuses a freed object's address,
    # so an id taken after the first loop closed can equal the second's.
    seen: list[object] = []

    async def work() -> None:
        seen.append(await provider.get_sdk())

    _drive(work, runs=2)
    assert seen[0] is not seen[1]


def test_openai_subscription_sdk_is_not_shared_across_loops() -> None:
    """The transport this suite never covered, which is why it stayed broken.

    ``AsyncOpenAI`` wraps an ``httpx`` pool with the same loop affinity
    as its Anthropic and Google peers.
    """
    provider = OpenAISubscription(
        access_token="t",  # noqa: S106 -- fixture; no network call is made
        refresh_token="r",  # noqa: S106 -- fixture; no network call is made
        account_id="a",
        expires_at=time.time() + 9_999,
    )
    seen: list[object] = []

    async def work() -> None:
        seen.append(await provider.get_sdk())

    assert _drive(work, runs=2) == []
    assert seen[0] is not seen[1]


def test_google_client_survives_a_second_loop() -> None:
    """Google's per-model httpx client has the same binding."""
    provider = Google(api_key="test-key")
    model = provider.model("gemini-2.0-flash")

    async def work() -> None:
        async def take() -> None:
            await model._get_client()
            await asyncio.sleep(0.01)

        await asyncio.gather(take(), take())

    assert _drive(work) == []


def test_file_write_lock_survives_a_second_loop(tmp_path: Path) -> None:
    """The process-global write-lock registry is reachable from any loop."""
    target = str(tmp_path / "edited.txt")

    async def work() -> None:
        async def hold() -> None:
            await locked_file_write(target, lambda: time.sleep(0.01))

        await asyncio.gather(hold(), hold())

    assert _drive(work) == []


def peak_concurrent_holders(
    enter: Callable[[], AbstractAsyncContextManager[object]],
    *,
    threads: int = 2,
) -> int:
    """Return the most holders ever inside ``enter`` at once, across threads.

    The only shape that tests exclusion. Asserting "no exception was
    raised" -- what every other test in this file does -- is passed
    trivially by a lock that excludes nothing, which is how two such
    locks reached the tree.

    Each worker drives its own loop, so this measures exclusion across
    loops as well as across coroutines.

    Args:
      enter: Builds the async context manager under test, per worker.
      threads: How many workers contend.

    Returns:
      peak: Maximum simultaneous holders observed.

    """
    active = 0
    peak = 0
    guard = threading.Lock()
    start = threading.Barrier(threads)

    def worker() -> None:
        async def hold() -> None:
            nonlocal active, peak
            start.wait()
            async with enter():
                with guard:
                    active += 1
                    peak = max(peak, active)
                await asyncio.sleep(0.01)
                with guard:
                    active -= 1

        asyncio.run(hold())

    workers = [threading.Thread(target=worker) for _ in range(threads)]
    for thread in workers:
        thread.start()
    for thread in workers:
        thread.join()
    return peak


def test_credential_file_lock_excludes_across_loops(tmp_path: Path) -> None:
    """Refresh-token rotation is destructive, so this lock must be total.

    Two POSTers consume one rotating ``refresh_token`` and the endpoint
    revokes it for the loser (``oauth.py`` module docstring). Per-loop
    exclusion does not deliver that: the guarded ``flock`` runs on one
    cached fd, and ``flock`` calls sharing an open file description do
    not exclude each other.
    """
    cred = tmp_path / "creds.json"

    def enter() -> AbstractAsyncContextManager[None]:
        return credential_file_lock(cred)

    assert peak_concurrent_holders(enter) == 1


def test_file_write_lock_excludes_across_loops(tmp_path: Path) -> None:
    """``tools/core.py`` promises "same path -> same lock" without qualification.

    Two agents mutating one file must not interleave between the
    staleness check and the write, whichever loop each runs on.
    """
    target = str(tmp_path / "edited.txt")
    active = 0
    peak = 0
    guard = threading.Lock()
    start = threading.Barrier(2)

    def mutate() -> None:
        nonlocal active, peak
        with guard:
            active += 1
            peak = max(peak, active)
        time.sleep(0.01)
        with guard:
            active -= 1

    def worker() -> None:
        async def hold() -> None:
            start.wait()
            await locked_file_write(target, mutate)

        asyncio.run(hold())

    workers = [threading.Thread(target=worker) for _ in range(2)]
    for thread in workers:
        thread.start()
    for thread in workers:
        thread.join()
    assert peak == 1
