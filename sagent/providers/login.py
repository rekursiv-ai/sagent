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
  logins.ranked("sol-6.1", now=time.time())      # (("default", None), ("home", Spent(...)))
"""

from __future__ import annotations

from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, ClassVar
from urllib.parse import parse_qs, urlparse

import asyncio
import fcntl
import logging
import os
import re
import subprocess
import sys
import time
import webbrowser

from treekle import from_plain, json_dumps, json_loads, to_plain

from sagent.lib.files.owner_only import open_private, write_private
from sagent.lib.userdirs import state_dir
from sagent.model.exceptions import AuthError


if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable, Mapping
    from pathlib import Path


__all__ = [
    "SIGN_IN_ERRORS",
    "Logins",
    "Spent",
    "check_account",
    "file_lock",
    "headless_wanted",
    "open_browser",
    "pasted_code",
    "read_pasted_code",
    "redirected",
]


logger = logging.getLogger(__name__)


SIGN_IN_ERRORS = (AuthError, EOFError, OSError, RuntimeError, ValueError)
"""What stops a sign-in that the user should read, not a traceback: a refusal, a
paste of another sign-in's address, no answer at the prompt, a dead connection."""


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
        the model that spent it. A second word that is no family of the model
        that spent it (``weekly Oauth_apps``) names none: the window covers
        every model, since a guess that covers too little never moves on.

        Args:
          model: A model id, as ``opus-5.5``.

        Returns:
          covered: Whether the window refuses it.

        """
        if not self.window:
            return model == self.model
        family = self.window.partition(" ")[2].lower()
        return not self.model.startswith(family) or model.startswith(family)


@dataclass(frozen=True, slots=True, kw_only=True)
class _Family:
    """What the ledger keeps for every account beside one default file."""

    added: tuple[str, ...] = ()
    precedence: tuple[str, ...] | None = None
    signed_out_at: dict[str, float] = field(default_factory=dict[str, float])
    """When each account was found signed out, in seconds since the epoch."""

    spent: dict[str, tuple[Spent, ...]] = field(
        default_factory=dict[str, tuple[Spent, ...]],
    )


class Logins:
    """Every account whose login is kept beside ``default``.

    Args:
      default: The default login's file; a named one sits beside it.

    """

    signed_out_sec: ClassVar[float] = 86_400.0
    """How long a login found signed out is passed over, unless signed in again."""

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
        return self._accounts(self._family())

    def precedence(self) -> tuple[str, ...] | None:
        """Return the order the user set; ``None`` when they set none."""
        return self._family().precedence

    def order(self) -> tuple[str, ...]:
        """Return the accounts to sign in as, first preferred first.

        Returns:
          order: The user's order, less the accounts with no login saved;
            else the default, then each named account as ``accounts`` orders
            them. Either way, less those found signed out and not since signed
            in again: their file rewritten, or ``signed_out_sec`` gone by.

        """
        return self._order(self._family(), now=time.time())

    def ranked(
        self,
        model: str,
        *,
        now: float,
        first: str | None = None,
    ) -> tuple[tuple[str, Spent | None], ...]:
        """Return each account in ``order`` at ``now``, and what keeps it from ``model``.

        One read of the ledger answers every account at once.

        Args:
          model: The model to call, as ``opus-5.5``.
          now: The time, in seconds since the epoch, for windows and sign-outs alike.
          first: An account to put first, as one the user chose, even one
            ``order`` would pass over.

        Returns:
          ranked: Each account, and its longest window yet to reset; ``None``
            for an account that may call ``model``.

        """
        family = self._family()
        order = self._order(family, now=now)
        if first is not None:
            order = (first, *(a for a in order if a != first))
        return tuple((a, _spent(family, a, model, now=now)) for a in order)

    def _accounts(self, family: _Family) -> tuple[str, ...]:
        """Return ``accounts``, as ``family`` orders those added."""
        stem, suffix = self.default.stem, self.default.suffix
        names = (
            found.name.removeprefix(f"{stem}-").removesuffix(suffix)
            for found in self.default.parent.glob(f"{stem}-*{suffix}")
        )
        saved = {n for n in names if _ACCOUNT.fullmatch(n)}
        added = [a for a in family.added if a in saved]
        return (*added, *sorted(saved - set(added)))

    def _order(self, family: _Family, *, now: float) -> tuple[str, ...]:
        """Return ``order``, as ``family`` has it at ``now``."""
        known = ("default", *self._accounts(family))
        chosen = family.precedence
        order = known if chosen is None else tuple(a for a in chosen if a in known)
        return tuple(
            a
            for a in order
            if a not in family.signed_out_at
            or now - family.signed_out_at[a] > self.signed_out_sec
            or self._written(a) > family.signed_out_at[a]
        )

    async def added(self, account: str | None) -> None:
        """Note ``account`` signed in: ordered after those added before it, and in use.

        Args:
          account: The account; ``None`` or ``default`` for the default, which
            is never ordered among the added.

        """
        name = account or "default"
        if name != "default":
            check_account(name)

        def signed_in(f: _Family) -> _Family:
            added = (
                f.added if name in f.added or name == "default" else (*f.added, name)
            )
            out = {a: at for a, at in f.signed_out_at.items() if a != name}
            return replace(f, added=added, signed_out_at=out)

        await self._edit(signed_in)

    async def sign_out(self, account: str | None) -> None:
        """Note that ``account``'s login was refused: ``order`` passes it over until signed in.

        Args:
          account: The account; ``None`` for the default.

        """
        name, at = account or "default", time.time()
        await self._edit(
            lambda f: replace(f, signed_out_at=f.signed_out_at | {name: at}),
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

    def _written(self, account: str) -> float:
        """Return when ``account``'s file was last written; ``0`` with no file."""
        try:
            return self.path(account).stat().st_mtime
        except OSError:
            return 0.0

    def _family(self) -> _Family:
        """Return what the ledger keeps for this default."""
        ledger = state_dir() / "rekursiv-ai" / "sagent" / "logins.json"
        return _family(_raw(ledger).get(str(self.default), {}))

    async def _edit(self, change: Callable[[_Family], _Family]) -> None:
        """Change what the ledger keeps for this default, dropping marks past reset."""
        ledger = state_dir() / "rekursiv-ai" / "sagent" / "logins.json"
        async with file_lock(ledger.with_name(f"{ledger.name}.lock")):
            await asyncio.to_thread(
                _rewrite,
                ledger,
                str(self.default),
                change,
                signed_out_sec=self.signed_out_sec,
            )


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


def headless_wanted(
    environ: Mapping[str, str],
    *,
    platform: str = sys.platform,
    given: bool | None = None,
) -> bool:
    """Return whether a sign-in should print its URL rather than open a browser.

    Over ssh, a browser opens on the far machine's screen, not the user's; on
    Linux with no display, there is none to open. macOS always has a screen.

    Args:
      environ: The environment, such as ``os.environ``.
      platform: ``sys.platform``.
      given: What ``--headless`` or ``--no-headless`` said; ``None`` for neither.

    Returns:
      headless: ``given`` when set; else whether no browser the user can see
        would open.

    """
    if given is not None:
        return given
    if any(
        environ.get(name, "") != ""
        for name in ("SSH_CONNECTION", "SSH_CLIENT", "SSH_TTY")
    ):
        return True
    return platform.startswith("linux") and all(
        environ.get(name, "") == "" for name in ("DISPLAY", "WAYLAND_DISPLAY")
    )


# ``webbrowser`` runs a command browser ($BROWSER, xdg-open) on the terminal's own
# streams, so what it says lands in the scrollback past ``Screen``, under a call
# whose result then prints unlabelled. Its other browsers already discard theirs.
def open_browser(url: str) -> None:
    """Open ``url`` in the user's browser, which never writes to the terminal.

    Args:
      url: The page; the caller has shown it too, in case no browser opens.

    """
    try:
        browser = webbrowser.get()
    except webbrowser.Error:
        return
    if not isinstance(browser, webbrowser.GenericBrowser):
        browser.open(url)
        return
    # A shell starts the browser in the background and exits at once, so the browser
    # is never our child: none is left to reap, nor found by another's ``os.wait()``.
    # Its words go to ``"$@"``, never into the script.
    with suppress(OSError):
        _ = subprocess.run(  # noqa: S603 -- The user's own browser command; the script is fixed.
            [
                "/bin/sh",
                "-c",
                '"$@" &',
                "sh",
                browser.name,
                *(arg.replace("%s", url) for arg in browser.args),
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            check=False,
        )


async def redirected(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    *,
    path: str,
    state: str,
    code: asyncio.Future[str],
) -> None:
    """Answer a browser's sign-in redirect to ``path``, and take the code it carries.

    Serve it with ``asyncio.start_server`` on loopback. A redirect elsewhere, or
    with another sign-in's ``state``, or with no code, is refused and the wait
    goes on; the first good one sets ``code``.

    Args:
      reader: The browser's request.
      writer: Where the answer goes.
      path: The redirect's path, as the sign-in URL names it.
      state: The sign-in's own ``state``.
      code: Set to the code of the first good redirect.

    """
    match (await reader.readline()).decode(errors="replace").split():
        case [_, target, *_]:
            url = urlparse(target)
            params = parse_qs(url.query)
            received = params.get("code", [""])[0]
            ok = url.path == path and params.get("state") == [state] and bool(received)
        case _:
            received, ok = "", False
    page = (
        "Signed in; you can close this tab."
        if ok
        else "Sign-in failed; see the terminal."
    )
    writer.write(
        f"HTTP/1.1 {200 if ok else 400} {'OK' if ok else 'Bad Request'}\r\n"
        f"content-type: text/plain\r\ncontent-length: {len(page)}\r\n"
        f"connection: close\r\n\r\n{page}".encode(),
    )
    await writer.drain()
    writer.close()
    if ok and not code.done():
        code.set_result(received)


def pasted_code(pasted: str, *, state: str) -> str:
    """Return the code in what the user pasted: the address a sign-in landed on.

    Over ssh, or with no browser, the sign-in page sends the browser to a
    loopback address that fails to load, and the user pastes that address. A
    page that shows the code itself gives ``code#state``, which is taken too.

    Args:
      pasted: What the user pasted, with or without the address's scheme.
      state: The sign-in's own ``state``.

    Returns:
      code: The sign-in code.

    Raises:
      AuthError: What was pasted holds no code, or carries another sign-in's
        ``state``.

    """
    params = parse_qs(urlparse(pasted.strip()).query)
    if not params:
        code, _, pasted_state = pasted.strip().partition("#")
        params = {"code": [code] if code else [], "state": [pasted_state]}
    # ``parse_qs`` drops a blank ``code=``, so an empty list is no code.
    codes = params.get("code", [])
    if not codes:
        raise AuthError("The pasted address holds no sign-in code.")
    if params.get("state") != [state]:
        raise AuthError("The pasted address belongs to another sign-in.")
    return codes[0]


async def read_pasted_code(
    *,
    state: str,
    show: Callable[[str], object],
    browser_tried: bool,
) -> str:
    """Ask for the address a sign-in landed on; return the code it carries.

    Args:
      state: The sign-in's own ``state``.
      show: Shows the user what to paste.
      browser_tried: A browser was opened and never came back, as over ssh;
        else none was, and the user opens the page anywhere.

    Returns:
      code: The sign-in code.

    Raises:
      AuthError: What was pasted holds no code, or is another sign-in's.
      EOFError: Nothing was typed at all.

    """
    show(
        "No browser came back. Paste the address it was sent to, though it fails"
        " to load:\n"
        if browser_tried
        else "Open it in a browser anywhere, then paste the address it was sent"
        " to, though it fails to load:\n",
    )
    return pasted_code(await asyncio.to_thread(input, "Address: "), state=state)


def _open(path: Path) -> int:
    """Open ``path`` to lock it, made private with its directory if new."""
    return open_private(path, os.O_RDWR)


# Every session shares the ledger, of whatever version: what this one cannot read
# is passed over, never fatal, and what it does not know is written back as it was.
def _raw(ledger: Path) -> dict[str, dict[str, object]]:
    """Return each family the ledger keeps, as plain data; none when it does not read."""
    try:
        text = ledger.read_text()
    except FileNotFoundError:
        return {}
    try:
        tree = json_loads(text, dict[str, object])
    except ValueError:
        logger.warning("%s does not read; starting it afresh", ledger)
        return {}
    return {
        key: family
        for key, value in tree.items()
        if (family := _read(value, dict[str, object], default=None)) is not None
    }


def _family(raw: dict[str, object]) -> _Family:
    """Return what ``raw`` keeps that this version knows; a part that does not read is dropped."""
    marks = _read(raw.get("spent"), dict[str, object], default=None) or {}
    return _Family(
        added=_read(raw.get("added"), tuple[str, ...], default=None) or (),
        precedence=_read(raw.get("precedence"), tuple[str, ...], default=None),
        signed_out_at=_read(raw.get("signed_out_at"), dict[str, float], default=None)
        or {},
        spent={
            account: tuple(
                spent
                for mark in _read(every, list[object], default=None) or ()
                if (spent := _read(mark, Spent, default=None)) is not None
            )
            for account, every in marks.items()
        },
    )


def _read[T, D](value: object, target: type[T], *, default: D) -> T | D:
    """Return ``value`` as ``target``; ``default`` when it is absent or does not read."""
    if value is None:
        return default
    try:
        return from_plain(value, target)
    except ValueError:
        logger.warning("passing over a ledger entry that does not read: %r", value)
        return default


def _spent(
    family: _Family,
    account: str | None,
    model: str,
    *,
    now: float,
) -> Spent | None:
    """Return the window keeping ``account`` from ``model`` longest; ``None`` if none."""
    marks = family.spent.get(account or "default", ())
    live = [s for s in marks if s.until > now and s.covers(model)]
    return max(live, key=lambda s: s.until) if live else None


def _rewrite(
    ledger: Path,
    key: str,
    change: Callable[[_Family], _Family],
    *,
    signed_out_sec: float,
) -> None:
    """Apply ``change`` to the family ``key``, keeping only marks yet to expire."""
    families = _raw(ledger)
    kept = families.get(key, {})
    now = time.time()
    changed = change(_family(kept))
    live = {
        account: marks
        for account, every in changed.spent.items()
        if (marks := tuple(s for s in every if s.until > now))
    }
    out = {
        account: at
        for account, at in changed.signed_out_at.items()
        if now - at <= signed_out_sec
    }
    known = from_plain(
        to_plain(replace(changed, spent=live, signed_out_at=out), target=_Family),
        dict[str, object],
    )
    # A field left at its default is not written; the old value must not survive.
    theirs = {k: v for k, v in kept.items() if k not in _Family.__dataclass_fields__}
    families[key] = theirs | known
    # It names the user's accounts and when each ran out.
    write_private(ledger, json_dumps(families))
