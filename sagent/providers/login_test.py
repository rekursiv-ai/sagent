"""Tests for a login's file: its lock, and the files each named account keeps."""

from __future__ import annotations

from pathlib import Path

import asyncio

import pytest

from sagent.providers.login import (
    account_path,
    check_account,
    file_lock,
    saved_accounts,
)


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


def test_an_account_s_file_sits_beside_the_default_one() -> None:
    default = Path("/logins/auth.json")
    assert account_path(default, "work") == Path("/logins/auth-work.json")


@pytest.mark.parametrize("account", ["../x", "a/b", "-x", "a b", "work.x", "a..b", ""])
def test_an_account_that_is_no_name_is_refused(account: str) -> None:
    with pytest.raises(ValueError, match="is not a name"):
        check_account(account)
    with pytest.raises(ValueError, match="is not a name"):
        account_path(Path("/logins/auth.json"), account)


@pytest.mark.parametrize("account", ["Work", "w2", "a_b-C"])
def test_an_account_of_letters_digits_and_separators_is_a_name(account: str) -> None:
    check_account(account)


def test_the_saved_accounts_are_the_files_beside_the_default_one(
    tmp_path: Path,
) -> None:
    default = tmp_path / "auth.json"
    assert saved_accounts(default) == ()
    for name in ("auth.json", "auth-work.json", "auth-home.json", "auth-.json"):
        (tmp_path / name).write_text("{}")
    (tmp_path / "auth-x.txt").write_text("{}")
    assert saved_accounts(default) == ("home", "work")


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
