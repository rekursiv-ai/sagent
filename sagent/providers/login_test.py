"""Tests for a login's file: its lock, and the files each named account keeps."""

from __future__ import annotations

from pathlib import Path

import asyncio
import time

from treekle import json_loads

import pytest

from sagent.lib.userdirs import state_dir
from sagent.providers.login import Logins, Spent, check_account, file_lock


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
    logins = Logins(Path("/logins/auth.json"))
    assert logins.path("work") == Path("/logins/auth-work.json")
    assert logins.path(None) == logins.path("default") == Path("/logins/auth.json")


@pytest.mark.parametrize("account", ["../x", "a/b", "-x", "a b", "work.x", "a..b", ""])
def test_an_account_that_is_no_name_is_refused(account: str) -> None:
    with pytest.raises(ValueError, match="is not a name"):
        check_account(account)
    with pytest.raises(ValueError, match="is not a name"):
        Logins(Path("/logins/auth.json")).path(account)


@pytest.mark.parametrize("account", ["Work", "w2", "a_b-C"])
def test_an_account_of_letters_digits_and_separators_is_a_name(account: str) -> None:
    check_account(account)


def _save(logins: Logins, *accounts: str) -> None:
    """Write a login file for each of ``accounts``, ``default`` included."""
    logins.default.parent.mkdir(parents=True, exist_ok=True)
    for account in accounts:
        logins.path(account).write_text("{}")


def test_the_saved_accounts_are_the_files_beside_the_default_one(
    tmp_path: Path,
) -> None:
    logins = Logins(tmp_path / "auth.json")
    assert logins.accounts() == ()
    _save(logins, "default", "work", "home")
    (tmp_path / "auth-.json").write_text("{}")
    (tmp_path / "auth-x.txt").write_text("{}")
    assert logins.accounts() == ("home", "work")
    assert logins.order() == ("default", "home", "work")


@pytest.mark.asyncio
async def test_accounts_come_in_the_order_they_were_added(tmp_path: Path) -> None:
    logins = Logins(tmp_path / "auth.json")
    _save(logins, "default", "zed", "home", "work")
    await logins.added("zed")
    await logins.added("work")
    await logins.added("zed")
    await logins.added(None)
    assert logins.accounts() == ("zed", "work", "home")
    # Another holder of the same default reads the same ledger.
    assert Logins(tmp_path / "auth.json").order() == ("default", "zed", "work", "home")


@pytest.mark.asyncio
async def test_a_precedence_orders_the_saved_accounts_it_names(tmp_path: Path) -> None:
    logins = Logins(tmp_path / "auth.json")
    _save(logins, "default", "home", "work")
    await logins.prefer(("work", "gone", "default"))
    assert logins.precedence() == ("work", "gone", "default")
    assert logins.order() == ("work", "default")
    await logins.prefer(None)
    assert logins.precedence() is None
    assert logins.order() == ("default", "home", "work")
    with pytest.raises(ValueError, match="is not a name"):
        await logins.prefer(("../x",))


@pytest.mark.asyncio
async def test_two_families_keep_their_own_ledgers(tmp_path: Path) -> None:
    claude = Logins(tmp_path / "claude" / "auth.json")
    codex = Logins(tmp_path / "codex" / "auth.json")
    _save(claude, "default", "work")
    _save(codex, "default", "work")
    await claude.prefer(("work",))
    assert codex.order() == ("default", "work")


@pytest.mark.parametrize(
    ("window", "model", "covered"),
    [
        ("5-hour", "sonnet-5.5", True),
        ("weekly", "opus-5.5", True),
        ("weekly Opus", "opus-5.5", True),
        ("weekly Opus", "sonnet-5.5", False),
        ("", "opus-5.5", True),
        ("", "sonnet-5.5", False),
    ],
)
def test_a_window_covers_every_model_or_only_its_family(
    window: str,
    model: str,
    covered: bool,
) -> None:
    spent = Spent(window=window, model="opus-5.5", until=10.0)
    assert spent.covers(model) is covered


@pytest.mark.asyncio
async def test_the_first_live_account_skips_each_spent_until_it_resets(
    tmp_path: Path,
) -> None:
    logins = Logins(tmp_path / "auth.json")
    _save(logins, "default", "home", "work")
    now = time.time()
    five_hour = Spent(window="5-hour", model="opus-5.5", until=now + 100.0)
    await logins.spend(None, five_hour)
    await logins.spend(
        "home",
        Spent(window="weekly Opus", model="opus-5.5", until=now + 50.0),
    )
    assert logins.first_live("opus-5.5", now=now) == "work"
    assert logins.first_live("sonnet-5.5", now=now) == "home"
    assert logins.first_live("opus-5.5", now=now + 60.0) == "home"
    assert logins.first_live("opus-5.5", now=now + 100.0) == "default"
    assert logins.spent("default", "opus-5.5", now=now) == five_hour
    assert logins.spent("default", "opus-5.5", now=now + 100.0) is None


@pytest.mark.asyncio
async def test_no_account_is_live_when_every_one_is_spent(tmp_path: Path) -> None:
    logins = Logins(tmp_path / "auth.json")
    _save(logins, "default", "work")
    now = time.time()
    for account in ("default", "work"):
        await logins.spend(account, Spent(window="5-hour", model="m", until=now + 99))
    assert logins.first_live("m", now=now) is None


@pytest.mark.asyncio
async def test_a_mark_past_its_reset_is_dropped_on_the_next_write(
    tmp_path: Path,
) -> None:
    logins = Logins(tmp_path / "auth.json")
    _save(logins, "default", "work")
    await logins.spend("work", Spent(window="5-hour", model="m", until=1.0))
    await logins.spend("default", Spent(window="5-hour", model="m", until=9e12))
    ledger = state_dir() / "rekursiv-ai" / "sagent" / "logins.json"
    kept = json_loads(ledger.read_text(), dict[str, dict[str, dict[str, object]]])
    assert list(kept[str(logins.default)]["spent"]) == ["default"]


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
