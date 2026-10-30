"""A login's credentials files: the lock held while renewing one, and its accounts.

A refresh token works once, so two processes renewing one login at once leave
one of them signed out. Each holds ``file_lock`` across the whole read, renew
and write, and the one that waited reads the token the other just wrote.

A named account keeps its login beside the default one, ``-<account>`` added
to its stem. ``Logins`` owns every account kept beside one default file: their
files, the order they were added in, the order the user prefers, and which
have spent a quota window. What is not a file, it keeps in one ledger every
process shares.

Typical usage example:

  async with file_lock(credentials.with_suffix(".json.lock")):
      ...  # Read the file, renew if still stale, write it.

  logins = Logins(CodexLogin.default_path())
  logins.path("work")                            # ~/.codex/auth-work.json
  logins.order()                                 # ("default", "home", "work")
  await logins.spend("home", Spent(window="5-hour", model="sol-6.1", until=t))
  logins.first_live("sol-6.1", now=time.time())  # "default"
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import asyncio
import fcntl
import os
import re
import time

from treekle import json_dumps, json_loads

from sagent.lib.userdirs import state_dir


if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable
    from pathlib import Path


__all__ = [
    "Logins",
    "Spent",
    "check_account",
    "file_lock",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class Spent:
    """One quota window of an account, spent until it resets."""

    window: str
    """The window, as ``5-hour``, ``weekly`` or ``weekly Opus``; ``""`` unnamed."""

    model: str
    """The model the window was spent by."""

    until: float
    """When it resets, in seconds since the epoch."""

    def covers(self, model: str) -> bool:
        """Return whether ``model`` is refused until the reset.

        A window named for a family (``weekly Opus``) covers that family's
        models; one named for a span alone, every model; an unnamed one, only
        the model that spent it.

        Args:
          model: A model id, as ``opus-5.5``.

        Returns:
          covered: Whether the window refuses it.

        """
        if not self.window:
            return model == self.model
        _, _, family = self.window.partition(" ")
        return model.startswith(family.lower())


@dataclass(frozen=True, slots=True, kw_only=True)
class _Family:
    """What the ledger keeps for every account beside one default file."""

    added: tuple[str, ...] = ()
    precedence: tuple[str, ...] | None = None
    spent: dict[str, tuple[Spent, ...]] = field(
        default_factory=dict[str, tuple[Spent, ...]],
    )


class Logins:
    """Every account whose login is kept beside ``default``.

    Args:
      default: The default login's file; a named one sits beside it.

    """

    def __init__(self, default: Path) -> None:
        self.default = default

    def path(self, account: str | None) -> Path:
        """Return the file ``account`` keeps its login in.

        Args:
          account: A named account; ``None`` or ``default`` for the default.

        Returns:
          path: ``default`` for the default; else it, ``-<account>`` added to
            its stem.

        Raises:
          ValueError: ``account`` is no name.

        """
        if account is None or account == "default":
            return self.default
        check_account(account)
        stem, suffix = self.default.stem, self.default.suffix
        return self.default.with_name(f"{stem}-{account}{suffix}")

    def accounts(self) -> tuple[str, ...]:
        """Return each named account with a login saved.

        Returns:
          accounts: Those added, in the order they were; then the rest, sorted.

        """
        stem, suffix = self.default.stem, self.default.suffix
        names = (
            found.name.removeprefix(f"{stem}-").removesuffix(suffix)
            for found in self.default.parent.glob(f"{stem}-*{suffix}")
        )
        saved = {n for n in names if _ACCOUNT.fullmatch(n)}
        added = [a for a in self._family().added if a in saved]
        return (*added, *sorted(saved - set(added)))

    def precedence(self) -> tuple[str, ...] | None:
        """Return the order the user set; ``None`` when they set none."""
        return self._family().precedence

    def order(self) -> tuple[str, ...]:
        """Return the accounts to sign in as, first preferred first.

        Returns:
          order: The user's order, less the accounts with no login saved;
            else the default, then each named account as ``accounts`` orders them.

        """
        known = ("default", *self.accounts())
        chosen = self.precedence()
        return known if chosen is None else tuple(a for a in chosen if a in known)

    def spent(self, account: str | None, model: str, *, now: float) -> Spent | None:
        """Return the window keeping ``account`` from ``model`` longest; ``None`` if none."""
        marks = self._family().spent.get(account or "default", ())
        live = [s for s in marks if s.until > now and s.covers(model)]
        return max(live, key=lambda s: s.until) if live else None

    def first_live(self, model: str, *, now: float) -> str | None:
        """Return the first account in ``order`` that may call ``model``; ``None`` if none."""
        return next(
            (a for a in self.order() if self.spent(a, model, now=now) is None),
            None,
        )

    async def added(self, account: str | None) -> None:
        """Note ``account`` signed in, so it orders after those added before it.

        Args:
          account: The account; ``None`` or ``default`` notes nothing.

        """
        if account is None or account == "default":
            return
        check_account(account)
        await self._edit(
            lambda f: (
                f if account in f.added else replace(f, added=(*f.added, account))
            ),
        )

    async def spend(self, account: str | None, spent: Spent) -> None:
        """Note that ``account`` spent a window, until it resets.

        Args:
          account: The account; ``None`` for the default.
          spent: The window, and when it resets.

        """
        name = account or "default"
        await self._edit(
            lambda f: replace(
                f,
                spent=f.spent | {name: (*f.spent.get(name, ()), spent)},
            ),
        )

    async def prefer(self, order: tuple[str, ...] | None) -> None:
        """Set the order to sign in by.

        Args:
          order: The accounts, first preferred first; ``None`` goes back to
            the default order.

        Raises:
          ValueError: An account in ``order`` is no name.

        """
        for account in order or ():
            if account != "default":
                check_account(account)
        await self._edit(lambda f: replace(f, precedence=order))

    def _family(self) -> _Family:
        """Return what the ledger keeps for this default."""
        ledger = state_dir() / "rekursiv-ai" / "sagent" / "logins.json"
        return _read(ledger).get(str(self.default), _Family())

    async def _edit(self, change: Callable[[_Family], _Family]) -> None:
        """Change what the ledger keeps for this default, dropping marks past reset."""
        ledger = state_dir() / "rekursiv-ai" / "sagent" / "logins.json"
        async with file_lock(ledger.with_name(f"{ledger.name}.lock")):
            await asyncio.to_thread(_rewrite, ledger, str(self.default), change)


# A sidecar, not the credentials file itself: a write replaces the file's inode,
# which would drop a lock held on it.
@asynccontextmanager
async def file_lock(path: Path) -> AsyncGenerator[None]:
    """Hold an exclusive lock on ``path`` against every other holder, here or not.

    Args:
      path: The lock file; it and its directory are made private when created.

    Yields:
      nothing: The lock is held until the block ends.

    """
    fd = await asyncio.to_thread(_open, path)
    try:
        # Polled, not blocking in a thread: a thread in ``flock`` outlives a
        # cancelled caller.
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                await asyncio.sleep(0.01)
        yield
    finally:
        os.close(fd)


def check_account(account: str) -> None:
    """Refuse ``account`` unless it is a name: letters, digits, ``_``, ``-``."""
    if not _ACCOUNT.fullmatch(account):
        raise ValueError(
            f"account {account!r} is not a name: use letters, digits, '_', '-'",
        )


_ACCOUNT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")


def _open(path: Path) -> int:
    """Open ``path`` to lock it, made private with its directory if new."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    return os.open(path, os.O_RDWR | os.O_CREAT, 0o600)


def _read(ledger: Path) -> dict[str, _Family]:
    """Return every family the ledger keeps; none when it is not there yet."""
    try:
        text = ledger.read_text()
    except FileNotFoundError:
        return {}
    return json_loads(text, dict[str, _Family])


def _rewrite(
    ledger: Path,
    key: str,
    change: Callable[[_Family], _Family],
) -> None:
    """Apply ``change`` to the family ``key``, keeping only marks yet to reset."""
    families = _read(ledger)
    now = time.time()
    changed = change(families.get(key, _Family()))
    live = {
        account: kept
        for account, marks in changed.spent.items()
        if (kept := tuple(s for s in marks if s.until > now))
    }
    families[key] = replace(changed, spent=live)
    staged = ledger.with_name(f"{ledger.name}.tmp")
    staged.write_text(json_dumps(families, target=dict[str, _Family]))
    staged.replace(ledger)
